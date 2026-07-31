"""Tests for the standalone, answer-safe surprise recommendation core."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))

import surprise_recommender as recommender  # noqa: E402


def _identity(
    number: int, *, release: str = "release_demo"
) -> recommender.QuestionIdentity:
    return recommender.QuestionIdentity(
        release_id=release,
        question_id=f"q_{number}",
        question_version=f"qv1_{number}",
    )


def _candidate(
    number: int,
    *,
    mean: float = 0.5,
    exposure: int = 0,
    family: str = "family_a",
    valid: bool = True,
    blocked: bool = False,
) -> recommender.RecommendationCandidate:
    return recommender.RecommendationCandidate(
        identity=_identity(number),
        family=family,
        posterior=recommender.BetaPosterior(alpha=mean, beta=1.0 - mean),
        exposure_count=exposure,
        valid=valid,
        blocked=blocked,
    )


def test_cold_start_prior_uses_all_available_weighted_signals() -> None:
    prior = recommender.cold_start_prior(
        anti_heuristic=0.8,
        ensemble_heuristic_wrong=False,
        exact_version_blind_error_rate=0.5,
    )

    # x = .5*.8 + .3*0 + .2*.5 = .5; p0 = .2 + .6*.5 = .5.
    assert prior == recommender.BetaPosterior(alpha=3.0, beta=3.0)
    assert prior.mean == pytest.approx(0.5)


def test_cold_start_prior_renormalizes_missing_signals() -> None:
    prior = recommender.cold_start_prior(
        anti_heuristic=0.6,
        ensemble_heuristic_wrong=True,
    )

    # x = (.5*.6 + .3*1) / .8 = .75; p0 = .65.
    assert prior.alpha == pytest.approx(1.0 + 4.0 * 0.65)
    assert prior.beta == pytest.approx(1.0 + 4.0 * 0.35)
    assert prior.mean == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("signals", "expected_probability"),
    [
        ({"anti_heuristic": 0.0}, 0.2),
        ({"ensemble_heuristic_wrong": True}, 0.8),
        ({"exact_version_blind_error_rate": 0.25}, 0.35),
    ],
)
def test_cold_start_prior_supports_each_signal_alone_and_clips_mapping(
    signals: dict[str, object],
    expected_probability: float,
) -> None:
    prior = recommender.cold_start_prior(**signals)

    assert prior.alpha == pytest.approx(1.0 + 4.0 * expected_probability)
    assert prior.beta == pytest.approx(1.0 + 4.0 * (1.0 - expected_probability))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"anti_heuristic": -0.1},
        {"anti_heuristic": 1.1},
        {"anti_heuristic": math.nan},
        {"anti_heuristic": True},
        {"ensemble_heuristic_wrong": 1},
        {"exact_version_blind_error_rate": math.inf},
        {"strength": 0.0, "anti_heuristic": 0.5},
    ],
)
def test_cold_start_prior_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(recommender.SurpriseValidationError):
        recommender.cold_start_prior(**kwargs)


def test_reactions_update_true_false_and_ignore_missing_feedback() -> None:
    prior = recommender.BetaPosterior(alpha=3.0, beta=2.0)

    posterior = recommender.update_posterior(
        prior,
        [True, None, False, True, None],
    )

    assert posterior == recommender.BetaPosterior(alpha=5.0, beta=3.0)
    assert recommender.update_posterior(prior, [None, None]) == prior
    assert prior == recommender.BetaPosterior(alpha=3.0, beta=2.0)


@pytest.mark.parametrize("reactions", [[1], [0], ["true"], [object()], "true"])
def test_reaction_updates_require_explicit_booleans_or_none(reactions: object) -> None:
    with pytest.raises(recommender.SurpriseValidationError):
        recommender.update_posterior(
            recommender.BetaPosterior(alpha=1.0, beta=1.0),
            reactions,  # type: ignore[arg-type]
        )


def test_exploit_filters_invalid_blocked_and_completed_candidates() -> None:
    completed = _candidate(1, mean=0.99)
    invalid = _candidate(2, mean=0.98, valid=False)
    blocked = _candidate(3, mean=0.97, blocked=True)
    winner = _candidate(4, mean=0.8)
    other = _candidate(5, mean=0.3)

    result = recommender.select_question(
        [other, blocked, winner, invalid, completed],
        {completed.identity},
        epsilon=0.0,
        seed=7,
    )

    assert result.mode == "exploit"
    assert result.question == winner.identity
    assert result.propensity == 1.0


def test_exploit_tie_break_is_stable_across_input_order() -> None:
    later = _candidate(20, mean=0.75)
    earlier = _candidate(10, mean=0.75)

    forward = recommender.select_question([later, earlier], set(), epsilon=0.0, seed=1)
    reverse = recommender.select_question(
        [earlier, later], set(), epsilon=0.0, seed=999
    )

    assert forward.question == reverse.question == earlier.identity


def test_last_family_is_excluded_when_another_family_is_available() -> None:
    high_same_family = _candidate(1, mean=0.99, family="last")
    lower_other_family = _candidate(2, mean=0.2, family="other")

    result = recommender.select_question(
        [high_same_family, lower_other_family],
        set(),
        epsilon=0.0,
        seed=3,
        last_family="last",
    )

    assert result.question == lower_other_family.identity


def test_last_family_remains_eligible_when_it_is_the_only_family() -> None:
    winner = _candidate(1, mean=0.8, family="last")
    other = _candidate(2, mean=0.2, family="last")

    result = recommender.select_question(
        [other, winner], set(), epsilon=0.0, seed=3, last_family="last"
    )

    assert result.question == winner.identity


def test_default_epsilon_explores_uniformly_at_lowest_exposure() -> None:
    exploit_only = _candidate(1, mean=0.9, exposure=5)
    least_a = _candidate(2, mean=0.6, exposure=1)
    least_b = _candidate(3, mean=0.4, exposure=1)

    # Random(1).random() < .2, so this decision samples the explore branch.
    result = recommender.select_question(
        [least_b, exploit_only, least_a], set(), seed=1
    )

    assert result.mode == "explore"
    assert result.question in {least_a.identity, least_b.identity}
    assert result.propensity == pytest.approx(0.2 / 2)

    seen = {
        recommender.select_question(
            [least_b, exploit_only, least_a], set(), epsilon=1.0, seed=seed
        ).question
        for seed in range(20)
    }
    assert seen == {least_a.identity, least_b.identity}


def test_propensity_is_full_mixture_probability_not_sampled_branch_only() -> None:
    winner_in_explore_pool = _candidate(1, mean=0.9, exposure=0)
    other_explore = _candidate(2, mean=0.4, exposure=0)
    exposed = _candidate(3, mean=0.7, exposure=2)

    exploit_result = recommender.select_question(
        [other_explore, exposed, winner_in_explore_pool], set(), seed=0
    )

    assert exploit_result.mode == "exploit"
    assert exploit_result.question == winner_in_explore_pool.identity
    assert exploit_result.propensity == pytest.approx(0.8 + 0.2 / 2)

    # Seed 1 enters exploration and selects the same policy winner.  Its total
    # probability remains the mixture probability, independent of sampled mode.
    explore_result = recommender.select_question(
        [other_explore, exposed, winner_in_explore_pool], set(), seed=1
    )
    if explore_result.question == winner_in_explore_pool.identity:
        assert explore_result.propensity == pytest.approx(0.8 + 0.2 / 2)
    else:
        assert explore_result.propensity == pytest.approx(0.2 / 2)


def test_fixed_seed_is_reproducible_and_result_has_no_private_fields() -> None:
    candidates = [
        _candidate(1, mean=0.9, exposure=0),
        _candidate(2, mean=0.4, exposure=0),
    ]

    first = recommender.select_question(candidates, set(), epsilon=1.0, seed=481)
    second = recommender.select_question(
        list(reversed(candidates)), set(), epsilon=1.0, seed=481
    )

    assert first == second
    public = first.to_public_dict()
    assert set(public) == {"mode", "question", "propensity"}
    assert set(public["question"]) == {
        "release_id",
        "question_id",
        "question_version",
    }
    serialized = repr(public)
    for private_name in (
        "correct_letter",
        "ground_truth",
        "alpha",
        "beta",
        "anti_heuristic",
        "blind_error",
        "exposure_count",
    ):
        assert private_name not in serialized


def test_no_eligible_question_fails_closed() -> None:
    candidate = _candidate(1, valid=False)

    with pytest.raises(recommender.NoEligibleQuestionError, match="no valid"):
        recommender.select_question([candidate], set(), seed=1)
    with pytest.raises(recommender.NoEligibleQuestionError, match="unfinished"):
        recommender.select_question(
            [_candidate(2)], {_identity(2)}, epsilon=0.2, seed=1
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: recommender.BetaPosterior(0.0, 1.0), "positive"),
        (lambda: recommender.BetaPosterior(math.nan, 1.0), "positive"),
        (lambda: recommender.BetaPosterior(1e308, 1e308), r"alpha \+ beta"),
        (lambda: recommender.QuestionIdentity("", "q", "v"), "non-empty"),
        (
            lambda: recommender.QuestionIdentity(" release", "q", "v"),
            "surrounding",
        ),
        (
            lambda: recommender.QuestionIdentity("release", "q\ud800", "v"),
            "surrogates",
        ),
        (
            lambda: recommender.RecommendationCandidate(
                _identity(1), "family", recommender.BetaPosterior(1, 1), True
            ),
            "exposure_count",
        ),
        (
            lambda: recommender.RecommendationCandidate(
                _identity(1),
                "family",
                recommender.BetaPosterior(1, 1),
                0,
                valid=1,
            ),
            "valid",
        ),
    ],
)
def test_value_objects_validate_strictly(factory: object, match: str) -> None:
    with pytest.raises(recommender.SurpriseValidationError, match=match):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": -0.01, "seed": 1},
        {"epsilon": 1.01, "seed": 1},
        {"epsilon": math.nan, "seed": 1},
        {"epsilon": True, "seed": 1},
        {"epsilon": 0.2, "seed": True},
        {"epsilon": 0.2, "seed": 1, "last_family": ""},
    ],
)
def test_selection_rejects_invalid_policy_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(recommender.SurpriseValidationError):
        recommender.select_question([_candidate(1)], set(), **kwargs)


def test_selection_rejects_bad_collections_and_duplicate_identities() -> None:
    candidate = _candidate(1)

    with pytest.raises(recommender.SurpriseValidationError, match="sequence"):
        recommender.select_question(iter([candidate]), set(), seed=1)
    with pytest.raises(recommender.SurpriseValidationError, match="completed"):
        recommender.select_question([candidate], [], seed=1)  # type: ignore[arg-type]
    with pytest.raises(recommender.SurpriseValidationError, match="duplicate"):
        recommender.select_question([candidate, candidate], set(), seed=1)
    with pytest.raises(recommender.SurpriseValidationError, match="QuestionIdentity"):
        recommender.select_question(
            [candidate],
            {"q_1"},
            seed=1,  # type: ignore[arg-type]
        )
