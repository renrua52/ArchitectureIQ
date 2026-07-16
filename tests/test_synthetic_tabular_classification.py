from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.candidates.sets import sample_candidate_set_pool
from architecture_iq.families import synthetic_tabular_classification as classification_module
from architecture_iq.families.synthetic_tabular_classification import RULE_FAMILIES, balanced_rule_family_schedule
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train, load_synthesize_module


@pytest.fixture
def small_profile():
    profile = load_profile("v2")
    profile.ground_truth["n_seeds"] = 2
    return profile


def _materialize(profile, tmp_path: Path, *, seed: int, rule_family: str):
    ensure_registries()
    family = get_dataset_family("synthetic_tabular_classification")
    partial = family.create_instance(profile, seed, input_dim=8, rule_family=rule_family)
    spec = family.build_spec_with_id(partial)
    out = tmp_path / f"{rule_family}_{seed}"
    family.materialize({**partial, **spec}, out)
    return family, spec, out


def test_rule_family_schedule_is_balanced() -> None:
    schedule = balanced_rule_family_schedule(12, seed=7)
    assert set(schedule) == set(RULE_FAMILIES)
    assert Counter(schedule) == {name: 4 for name in RULE_FAMILIES}
    counts = Counter(balanced_rule_family_schedule(14, seed=7)).values()
    assert max(counts) - min(counts) <= 1


def test_classification_default_training_setting() -> None:
    profile = load_profile("v2")
    assert profile.family_config("synthetic_tabular_classification")["train_size"] == 1024
    assert profile.family_training_defaults("synthetic_tabular_classification") == {
        "batch_size": 32,
        "training_steps": 256,
        "total_samples_seen": 8192,
    }

    ensure_registries()
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="stabcls_default",
        family="synthetic_tabular_classification",
        budget=8192,
        count=2,
        varying_axes=frozenset({"model"}),
        rng=random.Random(0),
        fixed_shared={
            "optimizer": {
                "type": "AdamW",
                "lr": 3.0e-4,
                "weight_decay": 1.0e-5,
                "betas": [0.9, 0.999],
            },
            "loss": {"loss_id": "cross_entropy"},
        },
        dataset_params={"input_dim": 8, "num_classes": 2},
    )
    assert all(
        spec["budget"]
        == {
            "training_steps": 256,
            "batch_size": 32,
            "total_samples_seen": 8192,
        }
        for spec in specs
    )


@pytest.mark.parametrize("rule_family", RULE_FAMILIES)
def test_classification_rules_are_reproducible_and_well_typed(small_profile, tmp_path: Path, rule_family: str) -> None:
    family, spec, out = _materialize(small_profile, tmp_path, seed=42, rule_family=rule_family)
    _, same_spec, same_out = _materialize(small_profile, tmp_path, seed=42, rule_family=rule_family)
    assert spec["dataset_id"] == same_spec["dataset_id"]
    tensors = family.load_tensors(out)
    regenerated = load_synthesize_module(out / "synthesize.py").synthesize()
    repeated = family.load_tensors(same_out)
    assert all(torch.equal(actual, expected) for actual, expected in zip(tensors, regenerated, strict=True))
    assert all(torch.equal(actual, expected) for actual, expected in zip(tensors, repeated, strict=True))
    train_x, train_y, test_x, test_y = tensors
    assert train_x.shape == (1024, 8) and test_x.shape == (2048, 8)
    assert train_x.dtype == test_x.dtype == torch.float32
    assert train_y.shape == (1024,) and test_y.shape == (2048,)
    assert train_y.dtype == test_y.dtype == torch.int64
    assert set(train_y.unique().tolist()) <= {0, 1}
    assert set(test_y.unique().tolist()) <= {0, 1}
    assert 0.35 <= float(train_y.float().mean()) <= 0.65
    assert 0.35 <= float(test_y.float().mean()) <= 0.65
    assert spec["params"]["calibration"]["seed"] != spec["params"]["point_sampling"]["seed"]


def test_materialize_executes_rendered_synthesizer(small_profile, tmp_path: Path, monkeypatch) -> None:
    sentinel = """import torch

def target(x):
    return x[:, 0]

def synthesize():
    return (torch.tensor([[101.0]]), torch.tensor([0]), torch.tensor([[103.0]]), torch.tensor([1]))
"""
    monkeypatch.setattr(classification_module, "SYNTHESIZE_TEMPLATE", sentinel)
    ensure_registries()
    family = get_dataset_family("synthetic_tabular_classification")
    partial = family.create_instance(small_profile, 9, input_dim=4, rule_family="smooth_additive")
    spec = family.build_spec_with_id(partial)
    out = tmp_path / "sentinel"
    family.materialize({**partial, **spec}, out)
    assert [tensor.flatten()[0].item() for tensor in family.load_tensors(out)] == [101.0, 0, 103.0, 1]


def test_classification_candidate_executes_and_reports_auxiliary_accuracy(small_profile, tmp_path: Path) -> None:
    family, dataset_spec, dataset_path = _materialize(small_profile, tmp_path, seed=5, rule_family="sparse_interaction")
    model = {"type": "mlp", "input_dim": 8, "output_dim": 2, "depth": 2, "width": 16, "residual": False, "layer_norm": [False, False], "activations": ["relu", "relu"]}
    candidate_spec = build_candidate_spec(
        small_profile, dataset_id=dataset_spec["dataset_id"], family="synthetic_tabular_classification", budget=128, batch_size=16,
        model=model, optimizer={"type": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]}, loss={"loss_id": "cross_entropy"},
    )
    candidate_path = tmp_path / "candidate"
    write_candidate(candidate_spec, candidate_path, get_model_type("mlp"))
    train_module = load_candidate_train(candidate_path)
    train_x, _, _, _ = family.load_tensors(dataset_path)
    assert train_module.Model()(train_x).shape == (1024, 2)
    summary = run_ground_truth(candidate_path, small_profile, dataset_path)
    assert summary["execution"] == "candidate_py_files"
    assert summary["selection_metric"] == "test_ce"
    assert "mean_test_ce" in summary and "std_test_ce" in summary
    assert "mean_test_accuracy" in summary and "std_test_accuracy" in summary
    assert all("final_test_accuracy" in row for row in summary["seed_results"])
