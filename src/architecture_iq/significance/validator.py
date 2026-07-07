from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from architecture_iq.profile import Profile
from architecture_iq.util import read_json


@dataclass
class SignificanceResult:
    passed: bool
    gap: float
    win_rate: float
    metric: str
    winner_index: int
    reason: str = ""


def mean_metric_key(metric: str) -> str:
    return f"mean_{metric}"


def final_metric_key(metric: str) -> str:
    return f"final_{metric}"


def validate_significance(
    summaries: list[dict[str, Any]],
    profile: Profile,
    *,
    metric: str = "test_mse",
    higher_is_better: bool = False,
    gap_min: float | None = None,
    win_rate_min: float | None = None,
    use_non_overlap: bool | None = None,
) -> SignificanceResult:
    sig = profile.significance
    gap_min = float(gap_min if gap_min is not None else sig["gap_min"])
    win_rate_min = float(win_rate_min if win_rate_min is not None else sig["win_rate_min"])
    use_non_overlap = bool(
        use_non_overlap if use_non_overlap is not None else sig.get("use_non_overlap", True)
    )
    mean_key = mean_metric_key(metric)
    final_key = final_metric_key(metric)

    if any(s.get("excluded") for s in summaries):
        return SignificanceResult(
            passed=False,
            gap=0.0,
            win_rate=0.0,
            metric=metric,
            winner_index=-1,
            reason="excluded candidate in pool",
        )

    means = np.array([s[mean_key] for s in summaries], dtype=np.float64)
    stds = np.array([s[f"std_{metric}"] for s in summaries], dtype=np.float64)
    if not np.all(np.isfinite(means)):
        return SignificanceResult(
            passed=False, gap=0.0, win_rate=0.0, metric=metric, winner_index=-1, reason="non-finite mean"
        )

    order = np.argsort(means)
    if higher_is_better:
        order = order[::-1]
    winner = int(order[0])
    if len(order) < 2:
        return SignificanceResult(
            passed=False, gap=0.0, win_rate=0.0, metric=metric, winner_index=winner, reason="too few choices"
        )
    runner_up = int(order[1])
    gap = float(abs(means[runner_up] - means[winner]))
    if gap < gap_min:
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=0.0,
            metric=metric,
            winner_index=winner,
            reason=f"gap {gap:.4f} < {gap_min}",
        )

    n_seeds = len(summaries[0]["seed_results"])
    wins = 0
    for seed_i in range(n_seeds):
        vals = []
        for s in summaries:
            sr = s["seed_results"][seed_i]
            vals.append(float("inf") if sr["failed"] else sr[final_key])
        vals_arr = np.array(vals)
        seed_order = np.argsort(vals_arr)
        if higher_is_better:
            seed_order = seed_order[::-1]
        if int(seed_order[0]) == winner:
            wins += 1
    win_rate = wins / n_seeds
    if win_rate < win_rate_min:
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=win_rate,
            metric=metric,
            winner_index=winner,
            reason=f"win_rate {win_rate:.2f} < {win_rate_min}",
        )

    if use_non_overlap:
        if means[winner] + stds[winner] >= means[runner_up] - stds[runner_up]:
            return SignificanceResult(
                passed=False,
                gap=gap,
                win_rate=win_rate,
                metric=metric,
                winner_index=winner,
                reason="non-overlap heuristic failed",
            )

    return SignificanceResult(
        passed=True,
        gap=gap,
        win_rate=win_rate,
        metric=metric,
        winner_index=winner,
    )


def load_summary(candidate_path: Path) -> dict[str, Any]:
    return read_json(candidate_path / "results" / "summary.json")
