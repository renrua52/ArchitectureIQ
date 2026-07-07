from __future__ import annotations

import random

from architecture_iq.candidates.axes import infer_axes, infer_question_type
from architecture_iq.candidates.sets import make_set_name, parse_varying_axes
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries
from architecture_iq.candidates.sets import sample_candidate_set_pool


def test_parse_varying_axes() -> None:
    assert parse_varying_axes(["model"]) == frozenset({"model"})
    assert parse_varying_axes(["model", "optimizer"]) == frozenset({"model", "optimizer"})


def test_make_set_name_format() -> None:
    name = make_set_name(1024, frozenset({"model"}), salt=0)
    assert name.startswith("set_1024_var_fix_fix_")


def test_sample_candidate_set_pool_respects_varying_axes() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = random.Random(0)
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        count=8,
        varying_axes=frozenset({"model"}),
        rng=rng,
    )
    optimizers = {spec["optimizer"]["type"] for spec in specs}
    losses = {spec["loss"]["loss_id"] for spec in specs}
    models = {spec["model"]["depth"] for spec in specs}
    assert len(optimizers) == 1
    assert len(losses) == 1
    assert len(models) > 1


def test_infer_axes_mixed_budget() -> None:
    specs = [
        {
            "budget": {"batch_size": 16},
            "model": {"a": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
        {
            "budget": {"batch_size": 32},
            "model": {"a": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
    ]
    invariant, varying = infer_axes(specs)
    assert "batch_size" in varying
    assert infer_question_type(specs) == "mixed"
