from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

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


def test_bigram_materialize_executes_generated_synthesize(tmp_path) -> None:
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family("bigram_lm")
    partial = family.create_instance(profile, 3)
    spec = family.build_spec_with_id(partial)
    expected = tuple(torch.full((2, 3), i, dtype=torch.int64) for i in range(4))
    module = SimpleNamespace(synthesize=lambda: expected)

    with patch(
        "architecture_iq.runtime.loader.load_synthesize_module",
        return_value=module,
    ):
        family.materialize(spec, tmp_path)

    actual = family.load_tensors(tmp_path)
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


@pytest.mark.parametrize("family_name", ["univariate_regression", "multivariate_regression"])
def test_label_noise_train_only_and_reproducible(family_name: str, tmp_path) -> None:
    """Noise perturbs train labels only; test stays the exact target; id is content-addressed."""
    ensure_registries()
    profile = load_profile("v1")
    family = get_dataset_family(family_name)

    clean_partial = family.create_instance(profile, 5, noise_std=0.0)
    noisy_partial = family.create_instance(profile, 5, noise_std=0.2)
    clean_spec = family.build_spec_with_id(clean_partial)
    noisy_spec = family.build_spec_with_id(noisy_partial)

    # Same seed + different noise => different content-addressed dataset_id.
    assert clean_spec["dataset_id"] != noisy_spec["dataset_id"]
    assert noisy_spec["params"]["noise"]["enabled"] is True
    assert clean_spec["params"]["noise"]["enabled"] is False

    out = tmp_path / "noisy"
    family.materialize({**noisy_partial, **noisy_spec}, out)
    tx, ty, vx, vy = family.load_tensors(out)

    # Reload the generated synthesize.py to recompute the true target.
    from architecture_iq.runtime.loader import load_synthesize_module

    module = load_synthesize_module(out / "synthesize.py")
    true_train_y = module.target(tx.squeeze(-1) if family_name == "univariate_regression" else tx)
    true_test_y = module.target(vx.squeeze(-1) if family_name == "univariate_regression" else vx)
    true_train_y = true_train_y.reshape(ty.shape)
    true_test_y = true_test_y.reshape(vy.shape)

    # Test labels are exactly the target; train labels are perturbed.
    assert torch.allclose(vy, true_test_y, atol=1e-6)
    assert not torch.allclose(ty, true_train_y, atol=1e-6)
    assert (ty - true_train_y).std().item() > 0.05  # ~0.2 requested


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
