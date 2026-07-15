from __future__ import annotations

import numpy as np
import pytest
import torch

from architecture_iq.datasets import create_dataset, format_dataset_summary_lines
from architecture_iq.families.bigram_lm import family as bigram_family_module
from architecture_iq.families.bigram_lm.bigram import make_bigram_dataset
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, list_dataset_families
from architecture_iq.runtime.loader import load_synthesize_module


def test_registry_lists_new_families() -> None:
    ensure_registries()
    names = list_dataset_families()
    assert "multivariate_regression" in names
    assert "bigram_lm" in names


def test_create_multivariate_dataset() -> None:
    ensure_registries()
    profile = load_profile("v1")
    spec, path = create_dataset(profile, 12, family_name="multivariate_regression")
    assert spec["family"] == "multivariate_regression"
    assert spec["params"]["input_dim"] in profile.family_config("multivariate_regression")["input_dims"]
    family = get_dataset_family("multivariate_regression")
    tx, ty, vx, vy = family.load_tensors(path)
    assert tx.ndim == 2 and tx.shape[1] == spec["params"]["input_dim"]
    assert ty.shape[1] == 1


def test_create_multivariate_dataset_explicit_input_dim() -> None:
    ensure_registries()
    profile = load_profile("v1")
    spec, path = create_dataset(
        profile,
        12,
        family_name="multivariate_regression",
        family_options={"input_dim": 5},
    )
    assert spec["params"]["input_dim"] == 5
    tx, _, _, _ = get_dataset_family("multivariate_regression").load_tensors(path)
    assert tx.shape[1] == 5


def test_resolve_input_dim_rejects_unknown() -> None:
    from architecture_iq.families.multivariate_regression.config import resolve_input_dim

    profile = load_profile("v1")
    with pytest.raises(ValueError, match="input_dim must be one of"):
        resolve_input_dim(profile, input_dim=99)


def test_format_dataset_summary_lines() -> None:
    ensure_registries()
    profile = load_profile("v1")
    uni, _ = create_dataset(profile, 1, family_name="univariate_regression")
    mv, _ = create_dataset(profile, 2, family_name="multivariate_regression")
    bg, _ = create_dataset(profile, 3, family_name="bigram_lm")

    uni_lines = format_dataset_summary_lines(uni)
    assert len(uni_lines) == 1
    assert uni_lines[0].startswith("Expression: ")

    mv_lines = format_dataset_summary_lines(mv)
    assert any(line.startswith("Input dimension: ") for line in mv_lines)
    assert any(line.startswith("Expression: ") for line in mv_lines)

    bg_lines = format_dataset_summary_lines(bg)
    assert any(line.startswith("Vocab size: ") for line in bg_lines)
    assert any(line.startswith("Context length: ") for line in bg_lines)
    assert "expression" not in " ".join(bg_lines).lower()


def test_create_bigram_dataset() -> None:
    ensure_registries()
    profile = load_profile("v1")
    spec, path = create_dataset(profile, 3, family_name="bigram_lm")
    assert spec["selection_metric"] == "test_ce"
    family = get_dataset_family("bigram_lm")
    tx, ty, vx, vy = family.load_tensors(path)
    assert tx.shape == ty.shape
    assert tx.dtype == ty.dtype


@pytest.mark.parametrize(
    "family_name",
    ["univariate_regression", "multivariate_regression", "bigram_lm"],
)
def test_materialized_tensors_match_generated_synthesizer(tmp_path, family_name: str) -> None:
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family(family_name)
    partial = family.create_instance(profile, 17)
    partial["params"]["train_size"] = 32
    partial["params"]["test_size"] = 16
    spec = family.build_spec_with_id(partial)
    materialized = {**partial, **spec}
    dataset_path = tmp_path / family_name

    family.materialize(materialized, dataset_path)

    module = load_synthesize_module(dataset_path / "synthesize.py")
    regenerated = module.synthesize()
    saved = family.load_tensors(dataset_path)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(saved, regenerated, strict=True)
    )


def test_bigram_transition_metadata_matches_generated_synthesizer(tmp_path) -> None:
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family("bigram_lm")
    partial = family.create_instance(profile, 23)
    partial["params"]["train_size"] = 16
    partial["params"]["test_size"] = 8
    spec = family.build_spec_with_id(partial)
    dataset_path = tmp_path / "bigram_lm"

    family.materialize({**partial, **spec}, dataset_path)

    module = load_synthesize_module(dataset_path / "synthesize.py")
    probs, pi = module.build_transition()
    with np.load(dataset_path / "transition.npz") as saved:
        assert np.array_equal(saved["probs"], probs)
        assert np.array_equal(saved["pi"], pi)


def test_bigram_materialize_executes_rendered_synthesizer(tmp_path, monkeypatch) -> None:
    sentinel_template = '''import numpy as np
import torch

def build_transition():
    return np.array([[0.25, 0.75], [0.5, 0.5]]), np.array([0.4, 0.6])

def synthesize():
    return (
        torch.tensor([[101]], dtype=torch.int64),
        torch.tensor([[102]], dtype=torch.int64),
        torch.tensor([[103]], dtype=torch.int64),
        torch.tensor([[104]], dtype=torch.int64),
    )
'''
    monkeypatch.setattr(bigram_family_module, "SYNTHESIZE_TEMPLATE", sentinel_template)
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family("bigram_lm")
    partial = family.create_instance(profile, 29)
    spec = family.build_spec_with_id(partial)
    dataset_path = tmp_path / "bigram_lm"

    family.materialize({**partial, **spec}, dataset_path)

    saved_tensors = family.load_tensors(dataset_path)
    assert [tensor.item() for tensor in saved_tensors] == [101, 102, 103, 104]
    with np.load(dataset_path / "transition.npz") as saved:
        assert np.array_equal(
            saved["probs"],
            np.array([[0.25, 0.75], [0.5, 0.5]]),
        )
        assert np.array_equal(saved["pi"], np.array([0.4, 0.6]))


def test_bigram_shared_transition_matrix() -> None:
    data1 = make_bigram_dataset(
        vocab_size=16,
        context_length=8,
        train_size=100,
        test_size=50,
        seed=1,
        table_seed=99,
    )
    data2 = make_bigram_dataset(
        vocab_size=16,
        context_length=8,
        train_size=100,
        test_size=50,
        seed=2,
        table_seed=99,
    )
    assert (data1["probs"] == data2["probs"]).all()
    assert not (data1["x_train"] == data2["x_train"]).all()


def test_compatible_models_by_family() -> None:
    ensure_registries()
    assert get_dataset_family("multivariate_regression").compatible_model_types() == ["mlp"]
    assert get_dataset_family("bigram_lm").compatible_model_types() == ["transformer_lm"]
