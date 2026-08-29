from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from architecture_iq.families.synthetic_tabular_classification import (
    SUPPORTED_RULE_FAMILIES,
    balanced_rule_family_schedule,
)
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family
from architecture_iq.runtime.loader import load_synthesize_module


def test_v1_supports_xor_and_spiral_rules() -> None:
    profile = load_profile("v1")
    cfg = profile.family_config("synthetic_tabular_classification")
    assert 2 in cfg["input_dims"]
    assert set(cfg["rule_families"]) >= {"xor", "spiral"}
    assert "xor" in SUPPORTED_RULE_FAMILIES
    assert "spiral" in SUPPORTED_RULE_FAMILIES
    schedule = balanced_rule_family_schedule(10, seed=3, allowed_rules=("xor", "spiral"))
    assert Counter(schedule) == {"xor": 5, "spiral": 5}


def test_xor_and_spiral_materialize_on_v1(tmp_path: Path) -> None:
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family("synthetic_tabular_classification")

    xor_partial = family.create_instance(profile, 11, input_dim=2, rule_family="xor")
    xor_spec = family.build_spec_with_id(xor_partial)
    xor_dir = tmp_path / "xor"
    family.materialize({**xor_partial, **xor_spec}, xor_dir)
    assert xor_partial["params"]["rule_family"] == "xor"
    assert xor_partial["params"]["input_dim"] == 2
    module = load_synthesize_module(xor_dir / "synthesize.py")
    tx, ty, vx, vy = module.synthesize()
    assert tx.shape == (1024, 2)
    assert set(ty.unique().tolist()) <= {0, 1}

    spiral_partial = family.create_instance(profile, 13, rule_family="spiral")
    spiral_spec = family.build_spec_with_id(spiral_partial)
    spiral_dir = tmp_path / "spiral"
    family.materialize({**spiral_partial, **spiral_spec}, spiral_dir)
    params = spiral_partial["params"]
    assert params["rule_family"] == "spiral"
    assert params["input_dim"] == 2
    assert params["point_sampling"]["distribution"] == "two_spirals"
    train_x, train_y, test_x, test_y = family.load_tensors(spiral_dir)
    assert train_x.shape[1] == 2
    assert Counter(train_y.tolist())[0] == 512
    assert Counter(train_y.tolist())[1] == 512
    # Spiral arms should not collapse to a single orthant cluster.
    assert torch.linalg.vector_norm(train_x, dim=1).mean().item() > 1.0
    assert set(test_y.unique().tolist()) == {0, 1}


def test_spiral_rejects_non_2d_input() -> None:
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family("synthetic_tabular_classification")
    try:
        family.create_instance(profile, 0, input_dim=4, rule_family="spiral")
    except ValueError as exc:
        assert "input_dim=2" in str(exc)
    else:
        raise AssertionError("expected ValueError for spiral with input_dim=4")
