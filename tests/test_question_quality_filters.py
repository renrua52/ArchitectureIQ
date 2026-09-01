from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from architecture_iq.profile import load_profile
from architecture_iq.questions.generator import (
    candidate_admitted,
    eligible_candidate_paths,
    find_significant_subsets,
)
from architecture_iq.questions.quality import QuestionQualityFilters
from architecture_iq.significance.validator import validate_significance


def _summary(mean: float, std: float, finals: list[float], *, failed_seeds: int = 0) -> dict:
    n = len(finals)
    seed_results = []
    for i, value in enumerate(finals):
        failed = i < failed_seeds
        seed_results.append(
            {
                "failed": failed,
                "final_test_mse": float("inf") if failed else value,
            }
        )
    return {
        "excluded": False,
        "failed_seeds": failed_seeds,
        "mean_test_mse": mean,
        "std_test_mse": std,
        "seed_results": seed_results or [{"failed": False, "final_test_mse": mean}],
    }


def test_gap_max_rejects_large_winner_runner_gap() -> None:
    profile = load_profile("v1")
    summaries = [
        _summary(0.1, 0.01, [0.1] * 10),
        _summary(0.5, 0.02, [0.5] * 10),
        _summary(0.6, 0.02, [0.6] * 10),
    ]
    assert validate_significance(summaries, profile).passed
    result = validate_significance(summaries, profile, gap_max=0.2)
    assert not result.passed
    assert "gap_max" in result.reason


def test_gap_worst_max_rejects_weak_distractor() -> None:
    profile = load_profile("v1")
    # winner–runner gap small enough for gap_max=0.5, but worst is very far
    summaries = [
        _summary(0.10, 0.01, [0.10] * 10),
        _summary(0.20, 0.01, [0.20] * 10),
        _summary(0.90, 0.02, [0.90] * 10),
    ]
    assert validate_significance(summaries, profile, gap_max=0.5).passed
    result = validate_significance(summaries, profile, gap_max=0.5, gap_worst_max=0.5)
    assert not result.passed
    assert "gap_worst_max" in result.reason


def test_pool_require_finite_mean_and_max_failed_seeds() -> None:
    filters = QuestionQualityFilters(require_finite_mean=True, max_failed_seeds=0)
    ok = _summary(0.1, 0.01, [0.1] * 10)
    assert candidate_admitted(ok, filters=filters)
    bad_mean = dict(ok)
    bad_mean["mean_test_mse"] = float("inf")
    assert not candidate_admitted(bad_mean, filters=filters)
    flaky = _summary(0.1, 0.01, [0.1] * 10, failed_seeds=1)
    assert not candidate_admitted(flaky, filters=filters)
    # default filters keep historical behaviour (only excluded)
    assert candidate_admitted(flaky, filters=QuestionQualityFilters.disabled())


def test_find_significant_subsets_respects_gap_max() -> None:
    profile = load_profile("v1")
    pool = [Path("good"), Path("mid"), Path("bad")]

    def fake_load_summary(path: Path) -> dict:
        means = {
            "good": (0.1, 0.01, [0.1] * 10),
            "mid": (0.5, 0.02, [0.5] * 10),
            "bad": (0.7, 0.02, [0.7] * 10),
        }
        mean, std, finals = means[path.name]
        return _summary(mean, std, finals)

    specs = {
        "good": {
            "budget": {"batch_size": 16},
            "model": {"type": "mlp", "depth": 1},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
            "execution": {"device": "cpu"},
        },
        "mid": {
            "budget": {"batch_size": 16},
            "model": {"type": "mlp", "depth": 2},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
            "execution": {"device": "cpu"},
        },
        "bad": {
            "budget": {"batch_size": 16},
            "model": {"type": "mlp", "depth": 3},
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
            "execution": {"device": "cpu"},
        },
    }

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    rng = __import__("random").Random(0)
    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            without = find_significant_subsets(pool, profile, rng, num_choices=2)
            with_cap = find_significant_subsets(
                pool,
                profile,
                rng,
                num_choices=2,
                # mid–bad gap is 0.2; require strictly below that so all pairs fail
                quality=QuestionQualityFilters(gap_max=0.15),
            )
    assert without
    assert with_cap == []


def test_eligible_candidate_paths_optional_wash() -> None:
    paths = [Path("ok"), Path("inf"), Path("flaky")]

    def fake_load_summary(path: Path) -> dict:
        if path.name == "ok":
            return _summary(0.1, 0.01, [0.1] * 10)
        if path.name == "inf":
            row = _summary(0.1, 0.01, [0.1] * 10)
            row["mean_test_mse"] = float("inf")
            return row
        return _summary(0.1, 0.01, [0.1] * 10, failed_seeds=2)

    with patch("architecture_iq.questions.generator.load_summary", fake_load_summary):
        assert eligible_candidate_paths(paths) == paths  # defaults: only excluded
        washed = eligible_candidate_paths(
            paths,
            filters=QuestionQualityFilters(require_finite_mean=True, max_failed_seeds=0),
        )
    assert washed == [Path("ok")]


def test_quality_filters_from_profile_overlay() -> None:
    profile = load_profile("v1")
    base = QuestionQualityFilters.from_profile(profile)
    assert base == QuestionQualityFilters.disabled()
    over = base.overlay(gap_max=0.3, gap_max_provided=True, require_finite_mean=True)
    assert over.gap_max == 0.3
    assert over.require_finite_mean is True


def test_param_ratio_max_rejects_free_largest_model_win() -> None:
    # C4: a subset where the winner has many more parameters than the
    # runner-up is solvable by "pick the largest model" and must be rejected.
    profile = load_profile("v1")
    pool = [Path("tiny"), Path("huge")]
    summaries = {
        "tiny": _summary(0.5, 0.01, [0.5] * 10),
        "huge": _summary(0.1, 0.01, [0.1] * 10),
    }
    specs = {
        "tiny": {
            "budget": {"batch_size": 16},
            "model": {"type": "mlp", "depth": 1},
            "trainable_parameter_count": 1_000,
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
            "execution": {"device": "cpu"},
        },
        "huge": {
            "budget": {"batch_size": 16},
            "model": {"type": "mlp", "depth": 6},
            "trainable_parameter_count": 50_000,
            "optimizer": {"type": "Adam"},
            "loss": {"loss_id": "mse"},
            "execution": {"device": "cpu"},
        },
    }

    def fake_read_json(path: Path) -> dict:
        if path.name == "candidate_spec.json":
            return specs[path.parent.name]
        raise FileNotFoundError(path)

    rng = __import__("random").Random(0)
    with patch("architecture_iq.questions.generator.load_summary", lambda p: summaries[p.name]):
        with patch("architecture_iq.questions.generator.read_json", fake_read_json):
            without = find_significant_subsets(pool, profile, rng, num_choices=2)
            with_cap = find_significant_subsets(
                pool,
                profile,
                rng,
                num_choices=2,
                quality=QuestionQualityFilters(param_ratio_max=8.0),
            )
    assert without
    assert with_cap == []


def test_v13_profile_quality_and_budgets() -> None:
    # C3/C4/C5: v1.3 carries the floors agreed in the pack review.
    profile = load_profile("v1.3")
    assert profile.min_training_steps() == 256
    assert profile.raw["budgets"]["total_samples_seen"] == [4096, 8192, 16384, 32768]
    quality = QuestionQualityFilters.from_profile(profile)
    assert quality.gap_max is None
    assert quality.param_ratio_max == 10.0
    assert quality.max_questions_per_dataset == 1
    assert profile.significance["gap_min"] == 0.06
    assert profile.significance["win_rate_min"] == 1.0
    # A5: no regularized losses in any pool.
    for family, pool in profile.pools["losses"].items():
        assert not {"mse_l1", "mse_l2", "cross_entropy_l1", "cross_entropy_l2"} & set(pool), family
    # C1: bigram pools are lists; spiral turns is a list.
    assert isinstance(profile.family_config("bigram_lm")["vocab_size"], list)
    assert isinstance(profile.family_config("synthetic_tabular_classification")["spiral_turns"], list)


def test_valid_batch_sizes_respect_min_training_steps() -> None:
    from architecture_iq.candidates.generator import valid_batch_sizes

    profile = load_profile("v1.3")
    # 4096 / 16 = 256 steps ok; 4096 / 32 = 128 steps below the floor.
    assert valid_batch_sizes(profile, 4096) == [16]
    assert valid_batch_sizes(profile, 8192) == [16, 32]
    assert valid_batch_sizes(profile, 16384) == [16, 32, 64]
    # Profiles without the knob keep the old behaviour.
    legacy = load_profile("v1")
    assert valid_batch_sizes(legacy, 2048) == [16, 32, 64]
