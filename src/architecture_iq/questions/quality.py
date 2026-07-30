"""Optional rule-based filters applied when assembling questions.

All filters default to off so large-gap / imperfect candidates remain usable
when the caller wants flexibility. Enable via profile
``question_generation.quality`` and/or ``generate-question`` CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from architecture_iq.profile import Profile


@dataclass(frozen=True)
class QuestionQualityFilters:
    """Optional post/pre filters for question assembly.

    Attributes:
        gap_max: If set, reject subsets where |mean_winner - mean_runner_up|
            exceeds this absolute value.
        gap_worst_max: If set, reject subsets where |mean_winner - mean_worst|
            exceeds this value.
        require_finite_mean: If true, drop candidates whose selection mean is
            missing or non-finite even when ``excluded`` is false.
        max_failed_seeds: If set, drop candidates with more than this many
            failed seeds. ``None`` means only honor ``summary.excluded``.
    """

    gap_max: float | None = None
    gap_worst_max: float | None = None
    require_finite_mean: bool = False
    max_failed_seeds: int | None = None

    @classmethod
    def disabled(cls) -> QuestionQualityFilters:
        return cls()

    @classmethod
    def from_profile(cls, profile: Profile) -> QuestionQualityFilters:
        raw = profile.question_generation.get("quality")
        if not raw:
            return cls.disabled()
        if not isinstance(raw, dict):
            raise ValueError("question_generation.quality must be a mapping when present")
        gap_max = raw.get("gap_max", None)
        gap_worst_max = raw.get("gap_worst_max", None)
        max_failed = raw.get("max_failed_seeds", None)
        return cls(
            gap_max=None if gap_max is None else float(gap_max),
            gap_worst_max=None if gap_worst_max is None else float(gap_worst_max),
            require_finite_mean=bool(raw.get("require_finite_mean", False)),
            max_failed_seeds=None if max_failed is None else int(max_failed),
        )

    def overlay(
        self,
        *,
        gap_max: float | None = None,
        gap_worst_max: float | None = None,
        require_finite_mean: bool | None = None,
        max_failed_seeds: int | None = None,
        gap_max_provided: bool = False,
        gap_worst_max_provided: bool = False,
        max_failed_seeds_provided: bool = False,
    ) -> QuestionQualityFilters:
        """Return a copy with CLI overrides applied when explicitly provided."""
        return QuestionQualityFilters(
            gap_max=gap_max if gap_max_provided else self.gap_max,
            gap_worst_max=gap_worst_max if gap_worst_max_provided else self.gap_worst_max,
            require_finite_mean=(
                self.require_finite_mean if require_finite_mean is None else require_finite_mean
            ),
            max_failed_seeds=(
                max_failed_seeds if max_failed_seeds_provided else self.max_failed_seeds
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gap_max": self.gap_max,
            "gap_worst_max": self.gap_worst_max,
            "require_finite_mean": self.require_finite_mean,
            "max_failed_seeds": self.max_failed_seeds,
        }

    @property
    def any_enabled(self) -> bool:
        return (
            self.gap_max is not None
            or self.gap_worst_max is not None
            or self.require_finite_mean
            or self.max_failed_seeds is not None
        )
