from __future__ import annotations

import random

from architecture_iq.candidates.generator import sample_model
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family


def test_v2_pool_includes_classification_kan_via_family_default() -> None:
    ensure_registries()
    profile = load_profile("v2")
    family = get_dataset_family("synthetic_tabular_classification")

    # Family declares mlp+kan; v2 pool includes both, so both are eligible.
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["mlp", "kan"]
    assert {
        sample_model(profile, random.Random(seed), family=family.name)["type"]
        for seed in range(128)
    } == {"mlp", "kan"}


def test_v21_explicitly_opens_classification_kan_gate() -> None:
    ensure_registries()
    profile = load_profile("v2.1")
    family = get_dataset_family("synthetic_tabular_classification")

    assert profile.profile_hash != "18d79b6ae61fc15b"
    assert profile.name == "v2.1"
    assert profile.kan["depth"] == [1, 2]
    assert profile.kan["width"] == [8]
    assert profile.kan["grid_size"] == [5]
    assert set(
        profile.model_types_for_family(family.name, family.compatible_model_types())
    ) == {"mlp", "kan"}
    sampled = {
        sample_model(profile, random.Random(seed), family=family.name)["type"]
        for seed in range(128)
    }
    assert sampled == {"mlp", "kan"}


def test_v24_xor_review_expands_only_the_xor_kan_pool() -> None:
    ensure_registries()
    v23 = load_profile("v2.3-xor-pilot")
    profile = load_profile("v2.4-xor-review")
    family = get_dataset_family("synthetic_tabular_classification")

    assert v23.kan == {
        "variant": "efficient_spline_v1",
        "depth": [1, 2],
        "width": [8],
        "grid_size": [5],
        "spline_order": [3],
        "grid_range": [[-1.0, 1.0]],
        "base_activation": ["silu"],
    }
    assert v23.significance == {
        "gap_min": 0.05,
        "win_rate_min": 0.7,
        "use_non_overlap": True,
    }
    assert profile.name == "v2.4-xor-review"
    assert profile.family_config("synthetic_tabular_classification")["rule_families"] == ["xor"]
    assert profile.kan == {
        "variant": "efficient_spline_v1",
        "depth": [1, 2],
        "width": [8, 16],
        "grid_size": [5, 7],
        "spline_order": [3],
        "grid_range": [[-1.0, 1.0], [-2.0, 2.0]],
        "base_activation": ["silu"],
    }
    assert profile.significance == {
        "gap_min": 0.0,
        "win_rate_min": 0.7,
        "use_non_overlap": True,
    }
    assert profile.model_types_for_family(
        family.name, family.compatible_model_types()
    ) == ["mlp", "kan"]
