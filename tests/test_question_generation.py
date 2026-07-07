from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import json

from architecture_iq.candidates.generator import choices_compatible, sample_variant_pool
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries
from architecture_iq.questions.generator import (
    _pick_distinct_subsets,
    eligible_candidate_paths,
    find_significant_subsets,
    select_significant_candidates,
)


def test_sample_variant_pool_unique() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(0)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        question_type="architecture_only",
        pool_size=8,
        rng=rng,
    )
    assert len(specs) == 8
    assert len({s["candidate_id"] for s in specs}) == 8


def test_architecture_only_shares_batch_size() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(0)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=1024,
        question_type="architecture_only",
        pool_size=8,
        rng=rng,
    )
    batch_sizes = {spec["budget"]["batch_size"] for spec in specs}
    assert len(batch_sizes) == 1
    optimizers = {spec["optimizer"]["type"] for spec in specs}
    assert len(optimizers) == 1
    assert choices_compatible(specs, "architecture_only")


def test_mixed_pool_varies_all_training_axes() -> None:
    ensure_registries()
    profile = load_profile("v1")
    rng = __import__("random").Random(42)
    specs = sample_variant_pool(
        profile,
        dataset_id="sym_test",
        family="univariate_regression",
        budget=8192,
        question_type="mixed",
        pool_size=24,
        rng=rng,
    )
    assert len(specs) == 24
    models = {json.dumps(spec["model"], sort_keys=True) for spec in specs}
    optimizers = {json.dumps(spec["optimizer"], sort_keys=True) for spec in specs}
    losses = {json.dumps(spec["loss"], sort_keys=True) for spec in specs}
    batch_sizes = {spec["budget"]["batch_size"] for spec in specs}
    assert len(models) > 1
    assert len(optimizers) > 1
    assert len(losses) > 1
    assert len(batch_sizes) == 1


def test_choices_compatible_rejects_mixed_batch_sizes() -> None:
    specs = [
        {
            "budget": {"batch_size": 16},
            "model": {"depth": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
        {
            "budget": {"batch_size": 32},
            "model": {"depth": 2},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
        },
    ]
    assert not choices_compatible(specs, "architecture_only")


def _fake_candidate_specs() -> dict[str, dict]:
    base = {
        "budget": {"batch_size": 16, "total_samples_seen": 1024, "training_steps": 64},
        "optimizer": {"type": "Adam"},
        "loss": {"loss_id": "mse"},
    }
    return {
        "good": {**base, "model": {"depth": 1}, "candidate_id": "good"},
        "mid": {**base, "model": {"depth": 2}, "candidate_id": "mid"},
        "bad1": {**base, "model": {"depth": 3}, "candidate_id": "bad1"},
        "bad2": {**base, "model": {"depth": 4}, "candidate_id": "bad2"},
    }


def test_profile_default_num_choices() -> None:
    profile = load_profile("v1")
    assert profile.num_choices == 2


def test_find_significant_subsets_returns_multiple() -> None:
    profile = load_profile("v1")

    def fake_load_summary(path: Path) -> dict:
        means = {
            "good": (0.1, 0.01, [0.09] * 10),
            "mid": (0.5, 0.02, [0.5] * 10),
            "bad1": (0.7, 0.02, [0.7] * 10),
            "bad2": (0.8, 0.02, [0.8] * 10),
        }
        key = path.name
        mean, std, finals = means[key]
        return {
            "excluded": False,
            "mean_test_mse": mean,
            "std_test_mse": std,
            "seed_results": [{"failed": False, "final_test_mse": v} for v in finals],
        }

    pool = [Path(p) for p in ("good", "mid", "bad1", "bad2")]
    rng = __import__("random").Random(0)
    specs = _fake_candidate_specs()

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            subsets = find_significant_subsets(pool, profile, rng, num_choices=2)

    assert len(subsets) >= 2
    picked = _pick_distinct_subsets(subsets, 2)
    assert len(picked) == 2
    assert _pick_distinct_subsets(subsets, 2)[0] != _pick_distinct_subsets(subsets, 2)[1]


def test_select_significant_candidates_exhaustive() -> None:
    profile = load_profile("v1")

    def fake_load_summary(path: Path) -> dict:
        means = {
            "good": (0.1, 0.01, [0.09] * 10),
            "mid": (0.5, 0.02, [0.5] * 10),
            "bad1": (0.7, 0.02, [0.7] * 10),
            "bad2": (0.8, 0.02, [0.8] * 10),
        }
        key = path.name
        mean, std, finals = means[key]
        return {
            "excluded": False,
            "mean_test_mse": mean,
            "std_test_mse": std,
            "seed_results": [{"failed": False, "final_test_mse": v} for v in finals],
        }

    pool = [Path(p) for p in ("good", "mid", "bad1", "bad2")]
    rng = __import__("random").Random(0)
    specs = _fake_candidate_specs()

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            selected = select_significant_candidates(pool, profile, rng)

    assert selected is not None
    assert len(selected) == 2


def test_eligible_candidate_paths_filters_excluded() -> None:
    paths = [Path("a"), Path("b")]

    def fake_load_summary(path: Path) -> dict:
        return {"excluded": path.name == "b"}

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        assert eligible_candidate_paths(paths) == [Path("a")]
