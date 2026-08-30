from __future__ import annotations

import json
import random
import pytest

import architecture_iq.candidates.sets as sets_module
from architecture_iq.candidates.axes import infer_axes, infer_question_type
from architecture_iq.candidates.generator import (
    sample_loss,
    sample_model,
    trainable_parameter_count,
)
from architecture_iq.candidates.sets import (
    make_set_name,
    parse_model_type_counts,
    parse_varying_axes,
    realized_parameter_band,
    sample_candidate_set_pool,
    select_parameter_band,
    write_set_manifest,
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


def test_sample_loss_only_draws_basic_losses() -> None:
    # Only plain mse / cross_entropy are sampleable; regularisation lives
    # solely in optimizer weight_decay. The filter is an allowlist, so a
    # profile cannot reintroduce an exotic loss by naming it in a pool.
    profile = load_profile("v2.3-gru-pilot")
    rng = random.Random(0)
    for _ in range(50):
        loss = sample_loss(profile, "bigram_lm", rng)
        assert loss["loss_id"] == "cross_entropy"
        assert "lambda" not in loss
    for pool in (["cross_entropy_l1"], ["cross_entropy_huber"]):
        profile.pools["losses"]["bigram_lm"] = pool
        try:
            sample_loss(profile, "bigram_lm", rng)
        except ValueError as exc:
            assert "no sampleable loss" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for pool {pool}")


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


# --- generation-time parameter banding (v1.4) --------------------------------

BIGRAM_PARAMS = {"vocab_size": 32, "context_length": 24}
TABULAR_PARAMS = {"input_dim": 8, "num_classes": 2}


def _param_ratio(specs: list[dict]) -> float:
    counts = [int(spec["trainable_parameter_count"]) for spec in specs]
    return max(counts) / min(counts)


def test_parameter_ratio_is_opt_in_per_profile() -> None:
    # Banding must be opt-in: 13 profiles predate candidate_generation and have
    # to keep producing byte-identical sets.
    assert load_profile("v1.4").parameter_ratio_max() == 2.0
    assert load_profile("v1.4").parameter_band_probe() == 256
    for name in ("v1", "v1.3", "v2", "v2.5-xor-holdout"):
        assert load_profile(name).parameter_ratio_max() is None
        assert load_profile(name).parameter_band_probe() == 256


def test_model_varying_set_stays_inside_the_band() -> None:
    ensure_registries()
    profile = load_profile("v1.4")
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="sym_band",
        family="univariate_regression",
        budget=8192,
        count=8,
        varying_axes=frozenset({"model"}),
        rng=random.Random(0),
    )
    assert len(specs) == 8
    assert len({spec["model"]["width"] for spec in specs}) > 1
    assert _param_ratio(specs) <= profile.parameter_ratio_max()


def test_optimizer_only_set_has_ratio_one_and_skips_the_probe(monkeypatch) -> None:
    # One shared model means the ratio is 1 by construction, so no band is
    # located and no probe cost is paid.
    ensure_registries()
    profile = load_profile("v1.4")
    calls = 0

    def counting_sample_model(*args, **kwargs):
        nonlocal calls
        calls += 1
        return sample_model(*args, **kwargs)

    monkeypatch.setattr(sets_module, "sample_model", counting_sample_model)
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="sym_band",
        family="univariate_regression",
        budget=8192,
        count=6,
        varying_axes=frozenset({"optimizer"}),
        rng=random.Random(1),
    )

    assert len({spec["trainable_parameter_count"] for spec in specs}) == 1
    assert _param_ratio(specs) == 1.0
    # Locating a band would take profile.parameter_band_probe() draws; a
    # shared-model set needs exactly one.
    assert calls == 1


def test_selected_band_holds_at_least_count_distinct_specs() -> None:
    ensure_registries()
    profile = load_profile("v1.4")
    count = 10
    lo, hi = select_parameter_band(
        profile,
        family="univariate_regression",
        count=count,
        rng=random.Random(3),
        dataset_params=None,
    )
    assert hi == int(lo * profile.parameter_ratio_max())
    rng = random.Random(4)
    in_band = set()
    for _ in range(profile.parameter_band_probe()):
        spec = sample_model(profile, rng, family="univariate_regression")
        if lo <= trainable_parameter_count(spec) <= hi:
            in_band.add(json.dumps(spec, sort_keys=True))
    assert len(in_band) >= count


def test_band_placement_varies_with_seed() -> None:
    # Random placement per set is the point: bands must not all cluster at one
    # parameter scale, or every question compares models of the same size.
    ensure_registries()
    profile = load_profile("v1.4")
    bands = {
        select_parameter_band(
            profile,
            family="univariate_regression",
            count=6,
            rng=random.Random(seed),
            dataset_params=None,
        )
        for seed in range(8)
    }
    assert len(bands) > 1


def test_mixed_bigram_quotas_fit_one_band() -> None:
    ensure_registries()
    profile = load_profile("v1.4")
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="bg_band",
        family="bigram_lm",
        budget=8192,
        count=6,
        varying_axes=frozenset({"model"}),
        rng=random.Random(0),
        model_type_counts={"transformer_lm": 4, "gru_lm": 2},
        dataset_params=BIGRAM_PARAMS,
    )
    types = [spec["model"]["type"] for spec in specs]
    assert types.count("transformer_lm") == 4
    assert types.count("gru_lm") == 2
    assert _param_ratio(specs) <= profile.parameter_ratio_max()


def test_gru_only_quota_is_infeasible_and_says_why() -> None:
    # gru_lm's v1.4 grid has 4 points (8416 / 14752 / 29088 / 54048), so no 2x
    # window holds 3 of them. The sampler must name the binding constraint
    # instead of widening the band or silently shrinking the quota.
    ensure_registries()
    profile = load_profile("v1.4")
    with pytest.raises(RuntimeError) as excinfo:
        sample_candidate_set_pool(
            profile,
            dataset_id="bg_band",
            family="bigram_lm",
            budget=8192,
            count=3,
            varying_axes=frozenset({"model"}),
            rng=random.Random(0),
            model_type_counts={"gru_lm": 3},
            dataset_params=BIGRAM_PARAMS,
        )
    message = str(excinfo.value)
    assert "gru_lm" in message
    assert "parameter_ratio_max" in message


def test_set_manifest_records_the_realized_band(tmp_path) -> None:
    ensure_registries()
    profile = load_profile("v1.4")
    specs = sample_candidate_set_pool(
        profile,
        dataset_id="tab_band",
        family="synthetic_tabular_classification",
        budget=8192,
        count=6,
        varying_axes=frozenset({"model"}),
        rng=random.Random(2),
        dataset_params=TABULAR_PARAMS,
    )
    write_set_manifest(
        tmp_path,
        set_name="set_test",
        budget=8192,
        count=len(specs),
        varying_axes=frozenset({"model"}),
        fixed_shared={},
        model_type_counts=None,
        parameter_band=realized_parameter_band(
            specs, ratio_max=profile.parameter_ratio_max()
        ),
        seed=2,
        profile=profile,
        dataset_id="tab_band",
        family="synthetic_tabular_classification",
    )
    band = json.loads((tmp_path / "set.json").read_text())["parameter_band"]
    assert band["ratio_max"] == 2.0
    assert band["realized_min"] <= band["realized_max"]
    assert band["realized_ratio"] <= band["ratio_max"]
