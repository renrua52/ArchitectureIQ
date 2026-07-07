from __future__ import annotations

import random
from pathlib import Path

import pytest

from architecture_iq.candidates.generator import (
    candidate_matches_fixed,
    sample_variant_pool,
)
from architecture_iq.datasets import list_dataset_instances, resolve_dataset_family
from architecture_iq.interactive import (
    assemble_model_spec,
    assemble_optimizer_spec,
    prompt_choice,
    prompt_dataset_family,
    prompt_fixed_components,
    prompt_grid_value,
    prompt_int,
    prompt_model_spec,
)
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def test_prompt_grid_value_random() -> None:
    assert prompt_grid_value("x", [16, 32], input_fn=lambda _: "") is None


def test_prompt_grid_value_by_index() -> None:
    assert prompt_grid_value("x", [16, 32], input_fn=lambda _: "2") == 32


def test_prompt_grid_value_by_literal() -> None:
    assert prompt_grid_value("x", [0.001, 0.003], input_fn=lambda _: "0.001") == 0.001


def test_prompt_choice_random() -> None:
    assert prompt_choice("opt", ["Adam", "SGD"], input_fn=lambda _: "") is None


def test_prompt_choice_select() -> None:
    assert prompt_choice("opt", ["Adam", "SGD"], input_fn=lambda _: "1") == "Adam"


def test_assemble_optimizer_partial() -> None:
    profile = load_profile("v1")
    rng = random.Random(0)
    spec = assemble_optimizer_spec(profile, rng, opt_type="Adam", lr=0.001)
    assert spec["type"] == "Adam"
    assert spec["lr"] == 0.001
    assert "betas" in spec


def test_assemble_model_partial() -> None:
    profile = load_profile("v1")
    rng = random.Random(0)
    spec = assemble_model_spec(profile, rng, depth=2, width=32)
    assert spec["depth"] == 2
    assert spec["width"] == 32
    assert len(spec["activations"]) == 2
    assert len(spec["layer_norm"]) == 2


def test_assemble_model_with_layer_norm() -> None:
    profile = load_profile("v1")
    rng = random.Random(0)
    spec = assemble_model_spec(
        profile,
        rng,
        depth=2,
        width=32,
        activations=["relu", "gelu"],
        layer_norm=[True, False],
        residual=False,
    )
    assert spec["layer_norm"] == [True, False]
    assert spec["activations"] == ["relu", "gelu"]


def test_prompt_model_spec_per_layer() -> None:
    profile = load_profile("v1")
    rng = random.Random(0)
    inputs = iter(
        [
            "2",  # depth
            "2",  # width 32
            "1",  # residual false
            "1",  # layer 1 activation relu
            "3",  # layer 2 activation gelu
            "1",  # layer 1 layer norm true
            "2",  # layer 2 layer norm false
        ]
    )
    spec = prompt_model_spec(
        profile,
        rng,
        input_fn=lambda _: next(inputs),
        write=lambda _: None,
    )
    assert spec["depth"] == 2
    assert spec["activations"] == ["relu", "gelu"]
    assert spec["layer_norm"] == [True, False]


def test_prompt_fixed_components_architecture_only() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = random.Random(0)
    inputs = iter(
        [
            "1",  # batch_size 16
            "1",  # loss mse
            "1",  # optimizer SGD
            "3",  # lr
            "",  # weight_decay random
            "2",  # momentum
        ]
    )
    fixed = prompt_fixed_components(
        profile,
        question_type="architecture_only",
        family="univariate_regression",
        budget=1024,
        rng=rng,
        input_fn=lambda _: next(inputs),
        write=lambda _: None,
    )
    assert fixed["batch_size"] == 16
    assert fixed["loss"]["loss_id"] == "mse"
    assert fixed["optimizer"]["type"] == "SGD"


def test_sample_variant_pool_uses_fixed_shared() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = random.Random(0)
    fixed = {
        "batch_size": 16,
        "loss": {"loss_id": "mse"},
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
    }
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        question_type="architecture_only",
        pool_size=4,
        rng=rng,
        fixed_shared=fixed,
    )
    assert len(specs) == 4
    for spec in specs:
        assert spec["budget"]["batch_size"] == 16
        assert spec["loss"] == fixed["loss"]
        assert spec["optimizer"] == fixed["optimizer"]


def test_candidate_matches_fixed() -> None:
    spec = {
        "budget": {"batch_size": 16},
        "model": {"type": "mlp", "depth": 1},
        "optimizer": {"type": "Adam"},
        "loss": {"loss_id": "mse"},
    }
    assert candidate_matches_fixed(
        spec,
        {"batch_size": 16, "loss": {"loss_id": "mse"}},
    )
    assert not candidate_matches_fixed(
        spec,
        {"batch_size": 32, "loss": {"loss_id": "mse"}},
    )


def test_prompt_dataset_family_random() -> None:
    profile = load_profile("v1")
    rng = random.Random(0)
    family = prompt_dataset_family(
        profile,
        rng=rng,
        input_fn=lambda _: "",
        write=lambda _: None,
    )
    assert family in profile.pools["dataset_families"]


def test_prompt_int_default() -> None:
    assert prompt_int("seed", default=0, input_fn=lambda _: "") == 0


def test_infer_question_type_from_specs() -> None:
    from architecture_iq.candidates.axes import infer_question_type

    specs = [
        {
            "budget": {"batch_size": 16},
            "model": {"depth": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
        {
            "budget": {"batch_size": 16},
            "model": {"depth": 2},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
    ]
    assert infer_question_type(specs) == "architecture_only"


def test_resolve_dataset_family_requires_explicit_choice() -> None:
    profile = load_profile("v1")
    with pytest.raises(ValueError, match="required"):
        resolve_dataset_family(profile)
    assert resolve_dataset_family(profile, family="univariate_regression") == "univariate_regression"
    assert (
        resolve_dataset_family(profile, random_pick=True, rng=random.Random(0))
        == "univariate_regression"
    )


@pytest.mark.skipif(not (DATA / "datasets").is_dir(), reason="no datasets")
def test_list_dataset_instances() -> None:
    instances = list_dataset_instances(DATA)
    assert instances
    assert all(entry.path.is_dir() for entry in instances)
    assert all(entry.family for entry in instances)
