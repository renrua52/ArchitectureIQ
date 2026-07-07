from __future__ import annotations

import pytest

from architecture_iq.datasets import create_dataset, format_dataset_summary_lines
from architecture_iq.families.bigram_lm.bigram import make_bigram_dataset
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, list_dataset_families


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
