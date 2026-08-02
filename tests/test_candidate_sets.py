from __future__ import annotations

import random
import pytest

from architecture_iq.candidates.axes import infer_axes, infer_question_type
from architecture_iq.candidates.generator import sample_loss
from architecture_iq.candidates.sets import (
    make_set_name,
    parse_model_type_counts,
    parse_varying_axes,
    sample_candidate_set_pool,
)
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries


def test_parse_varying_axes() -> None:
    assert parse_varying_axes(["model"]) == frozenset({"model"})
    assert parse_varying_axes(["model", "optimizer"]) == frozenset({"model", "optimizer"})


def test_parse_model_type_counts() -> None:
    assert parse_model_type_counts(["transformer_lm=16", "gru_lm=16"]) == {
        "transformer_lm": 16,
        "gru_lm": 16,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        parse_model_type_counts(["gru_lm=1", "gru_lm=2"])


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


def test_sample_language_regularized_loss_has_lambda() -> None:
    profile = load_profile("v2.3-gru-pilot")
    for loss_id in ("cross_entropy_l1", "cross_entropy_l2"):
        profile.pools["losses"]["bigram_lm"] = [loss_id]
        loss = sample_loss(profile, "bigram_lm", random.Random(0))
        assert loss["loss_id"] == loss_id
        assert loss["lambda"] in profile.loss_grids["lambda"]


def test_model_type_quotas_are_exact() -> None:
    ensure_registries()
    profile = load_profile("v2.4-gru-architecture-pilot")
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="bg_test",
        family="bigram_lm",
        budget=5120,
        count=4,
        varying_axes=frozenset({"model"}),
        rng=random.Random(0),
        model_type_counts={"transformer_lm": 2, "gru_lm": 2},
        dataset_params={"vocab_size": 32, "context_length": 16},
    )
    assert [spec["model"]["type"] for spec in specs].count("transformer_lm") == 2
    assert [spec["model"]["type"] for spec in specs].count("gru_lm") == 2


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
