"""Auditable, side-effect-free recommendation primitives for surprising questions.

This module intentionally has no Streamlit, storage, or network dependency.  A
caller is responsible for loading private scoring features and reaction counts,
then passes only validated candidates to :func:`select_question`.

The selection result is deliberately narrow: it contains the sampled policy
branch, the immutable public question identity, and that decision's exact
mixture propensity.  Ground truth, answer keys, posterior parameters, and the
private signals used to construct the cold-start prior are never returned.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal


ANTI_HEURISTIC_WEIGHT = 0.5
ENSEMBLE_WRONG_WEIGHT = 0.3
BLIND_ERROR_WEIGHT = 0.2
DEFAULT_PRIOR_STRENGTH = 4.0
DEFAULT_EPSILON = 0.2
MIN_PRIOR_PROBABILITY = 0.2
MAX_PRIOR_PROBABILITY = 0.8


class SurpriseRecommendationError(ValueError):
    """Base class for invalid recommender inputs or an empty candidate pool."""


class SurpriseValidationError(SurpriseRecommendationError):
    """Raised when an input violates the recommender's strict contract."""


class NoEligibleQuestionError(SurpriseRecommendationError):
    """Raised when filtering leaves no question that may be recommended."""


def _unit_interval(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SurpriseValidationError(f"{field_name} must be a real number")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as exc:
        raise SurpriseValidationError(
            f"{field_name} must be representable as a finite float"
        ) from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise SurpriseValidationError(f"{field_name} must be finite and in [0, 1]")
    return resolved


def _positive_finite(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SurpriseValidationError(f"{field_name} must be a real number")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as exc:
        raise SurpriseValidationError(
            f"{field_name} must be representable as a finite float"
        ) from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise SurpriseValidationError(f"{field_name} must be finite and positive")
    return resolved


def _non_empty_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SurpriseValidationError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise SurpriseValidationError(
            f"{field_name} must be non-empty and have no surrounding whitespace"
        )
    if len(value) > 256 or any(
        ord(character) < 32
        or ord(character) == 127
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise SurpriseValidationError(
            f"{field_name} must be at most 256 characters with no control "
            "characters or Unicode surrogates"
        )
    return value


@dataclass(frozen=True, slots=True)
class BetaPosterior:
    """Parameters of a Beta distribution over ``P(user is surprised)``."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "alpha", _positive_finite(self.alpha, field_name="alpha")
        )
        object.__setattr__(self, "beta", _positive_finite(self.beta, field_name="beta"))
        if not math.isfinite(self.alpha + self.beta):
            raise SurpriseValidationError("alpha + beta must be finite")

    @property
    def mean(self) -> float:
        """Return the posterior mean surprise probability."""
        return self.alpha / (self.alpha + self.beta)


def cold_start_prior(
    *,
    anti_heuristic: float | None = None,
    ensemble_heuristic_wrong: bool | None = None,
    exact_version_blind_error_rate: float | None = None,
    strength: float = DEFAULT_PRIOR_STRENGTH,
) -> BetaPosterior:
    """Build the cold-start Beta prior from whichever private signals exist.

    Signal weights are ``0.5 / 0.3 / 0.2`` and are re-normalized over available
    signals.  The blind-evaluation argument is intentionally named
    ``exact_version_blind_error_rate``: callers must not attach evaluations from
    another question version.  ``strength`` is the amount of synthetic evidence
    added to a uniform ``Beta(1, 1)`` base prior.
    """
    resolved_strength = _positive_finite(strength, field_name="strength")
    weighted_signals: list[tuple[float, float]] = []

    if anti_heuristic is not None:
        weighted_signals.append(
            (
                ANTI_HEURISTIC_WEIGHT,
                _unit_interval(anti_heuristic, field_name="anti_heuristic"),
            )
        )
    if ensemble_heuristic_wrong is not None:
        if not isinstance(ensemble_heuristic_wrong, bool):
            raise SurpriseValidationError(
                "ensemble_heuristic_wrong must be a bool when provided"
            )
        weighted_signals.append(
            (ENSEMBLE_WRONG_WEIGHT, float(ensemble_heuristic_wrong))
        )
    if exact_version_blind_error_rate is not None:
        weighted_signals.append(
            (
                BLIND_ERROR_WEIGHT,
                _unit_interval(
                    exact_version_blind_error_rate,
                    field_name="exact_version_blind_error_rate",
                ),
            )
        )

    if not weighted_signals:
        raise SurpriseValidationError(
            "at least one cold-start surprise signal must be provided"
        )

    available_weight = sum(weight for weight, _ in weighted_signals)
    score = sum(weight * value for weight, value in weighted_signals) / available_weight
    probability = min(
        MAX_PRIOR_PROBABILITY,
        max(MIN_PRIOR_PROBABILITY, 0.2 + 0.6 * score),
    )
    return BetaPosterior(
        alpha=1.0 + resolved_strength * probability,
        beta=1.0 + resolved_strength * (1.0 - probability),
    )


def update_posterior(
    posterior: BetaPosterior,
    reactions: Iterable[bool | None],
) -> BetaPosterior:
    """Update a posterior with explicit reactions; ``None`` means no feedback."""
    if not isinstance(posterior, BetaPosterior):
        raise SurpriseValidationError("posterior must be a BetaPosterior")
    if isinstance(reactions, (str, bytes)) or not isinstance(reactions, Iterable):
        raise SurpriseValidationError("reactions must be an iterable of bool or None")

    surprised = 0
    expected = 0
    for index, reaction in enumerate(reactions):
        if reaction is None:
            continue
        if not isinstance(reaction, bool):
            raise SurpriseValidationError(f"reactions[{index}] must be a bool or None")
        if reaction:
            surprised += 1
        else:
            expected += 1
    return BetaPosterior(
        alpha=posterior.alpha + surprised,
        beta=posterior.beta + expected,
    )


@dataclass(frozen=True, order=True, slots=True)
class QuestionIdentity:
    """Exact published identity used for filtering and stable tie-breaking."""

    release_id: str
    question_id: str
    question_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "release_id",
            _non_empty_identifier(self.release_id, field_name="release_id"),
        )
        object.__setattr__(
            self,
            "question_id",
            _non_empty_identifier(self.question_id, field_name="question_id"),
        )
        object.__setattr__(
            self,
            "question_version",
            _non_empty_identifier(self.question_version, field_name="question_version"),
        )

    def to_public_dict(self) -> dict[str, str]:
        """Return the complete identity without any question-answer content."""
        return {
            "release_id": self.release_id,
            "question_id": self.question_id,
            "question_version": self.question_version,
        }


@dataclass(frozen=True, slots=True)
class RecommendationCandidate:
    """Private policy input for one immutable published question version."""

    identity: QuestionIdentity
    family: str
    posterior: BetaPosterior
    exposure_count: int
    valid: bool = True
    blocked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, QuestionIdentity):
            raise SurpriseValidationError("identity must be a QuestionIdentity")
        object.__setattr__(
            self, "family", _non_empty_identifier(self.family, field_name="family")
        )
        if not isinstance(self.posterior, BetaPosterior):
            raise SurpriseValidationError("posterior must be a BetaPosterior")
        if (
            isinstance(self.exposure_count, bool)
            or not isinstance(self.exposure_count, int)
            or self.exposure_count < 0
        ):
            raise SurpriseValidationError(
                "exposure_count must be a non-negative integer"
            )
        if not isinstance(self.valid, bool):
            raise SurpriseValidationError("valid must be a bool")
        if not isinstance(self.blocked, bool):
            raise SurpriseValidationError("blocked must be a bool")


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Public, answer-safe result of one epsilon-greedy decision."""

    mode: Literal["exploit", "explore"]
    question: QuestionIdentity
    propensity: float

    def __post_init__(self) -> None:
        if self.mode not in {"exploit", "explore"}:
            raise SurpriseValidationError("mode must be 'exploit' or 'explore'")
        if not isinstance(self.question, QuestionIdentity):
            raise SurpriseValidationError("question must be a QuestionIdentity")
        object.__setattr__(
            self,
            "propensity",
            _unit_interval(self.propensity, field_name="propensity"),
        )
        if self.propensity == 0.0:
            raise SurpriseValidationError("propensity must be greater than zero")

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize only fields safe to send to an unauthenticated quiz client."""
        return {
            "mode": self.mode,
            "question": self.question.to_public_dict(),
            "propensity": self.propensity,
        }


def _validate_selection_inputs(
    candidates: Sequence[RecommendationCandidate],
    completed: Set[QuestionIdentity],
    *,
    epsilon: float,
    seed: int,
    last_family: str | None,
) -> tuple[float, set[QuestionIdentity], str | None]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise SurpriseValidationError(
            "candidates must be a sequence of RecommendationCandidate values"
        )
    identities: set[QuestionIdentity] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, RecommendationCandidate):
            raise SurpriseValidationError(
                f"candidates[{index}] must be a RecommendationCandidate"
            )
        if candidate.identity in identities:
            raise SurpriseValidationError(
                f"duplicate candidate identity: {candidate.identity.question_id}"
            )
        identities.add(candidate.identity)

    if isinstance(completed, (str, bytes)) or not isinstance(completed, Set):
        raise SurpriseValidationError(
            "completed must be a set of QuestionIdentity values"
        )
    completed_identities: set[QuestionIdentity] = set()
    for index, identity in enumerate(completed):
        if not isinstance(identity, QuestionIdentity):
            raise SurpriseValidationError(
                f"completed item {index} must be a QuestionIdentity"
            )
        completed_identities.add(identity)

    resolved_epsilon = _unit_interval(epsilon, field_name="epsilon")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SurpriseValidationError("seed must be an integer")
    if last_family is not None:
        last_family = _non_empty_identifier(last_family, field_name="last_family")
    return resolved_epsilon, completed_identities, last_family


def select_question(
    candidates: Sequence[RecommendationCandidate],
    completed: Set[QuestionIdentity],
    *,
    epsilon: float = DEFAULT_EPSILON,
    seed: int,
    last_family: str | None = None,
) -> Recommendation:
    """Select one eligible question using an auditable epsilon-greedy policy.

    Eligibility requires ``valid``, not blocked, and an identity absent from the
    completed set.  If another family is available, questions from
    ``last_family`` are removed before policy probabilities are calculated.

    Exploitation chooses the highest posterior mean with a lexicographic exact-
    identity tie-break.  Exploration is uniform over questions at the minimum
    exposure count.  Consequently the returned mixture propensity is

    ``(1-epsilon) * I[q is exploit winner] + epsilon * I[q in explore pool] / N``.
    """
    resolved_epsilon, completed_identities, resolved_last_family = (
        _validate_selection_inputs(
            candidates,
            completed,
            epsilon=epsilon,
            seed=seed,
            last_family=last_family,
        )
    )
    eligible = [
        candidate
        for candidate in candidates
        if candidate.valid
        and not candidate.blocked
        and candidate.identity not in completed_identities
    ]
    if not eligible:
        raise NoEligibleQuestionError(
            "no valid, unblocked, unfinished question remains"
        )

    if resolved_last_family is not None:
        other_families = [
            candidate
            for candidate in eligible
            if candidate.family != resolved_last_family
        ]
        if other_families:
            eligible = other_families

    ordered = sorted(eligible, key=lambda candidate: candidate.identity)
    exploit_winner = min(
        ordered,
        key=lambda candidate: (-candidate.posterior.mean, candidate.identity),
    )
    minimum_exposure = min(candidate.exposure_count for candidate in ordered)
    explore_pool = [
        candidate
        for candidate in ordered
        if candidate.exposure_count == minimum_exposure
    ]

    rng = random.Random(seed)
    explore = resolved_epsilon == 1.0 or (
        resolved_epsilon > 0.0 and rng.random() < resolved_epsilon
    )
    if explore:
        selected = explore_pool[rng.randrange(len(explore_pool))]
        mode: Literal["exploit", "explore"] = "explore"
    else:
        selected = exploit_winner
        mode = "exploit"

    propensity = 0.0
    if selected.identity == exploit_winner.identity:
        propensity += 1.0 - resolved_epsilon
    if selected.exposure_count == minimum_exposure:
        propensity += resolved_epsilon / len(explore_pool)

    return Recommendation(
        mode=mode,
        question=selected.identity,
        propensity=propensity,
    )
