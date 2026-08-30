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
    # min-over-rivals(best seed) - winner(worst seed), oriented so positive
    # means the winner's whole seed range sits clear of every rival's. Recorded
    # even when the separation check is off, so audits can see the headroom.
    separation_margin: float = float("nan")


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
    gap_max: float | None = None,
    gap_worst_max: float | None = None,
    win_rate_min: float | None = None,
    use_non_overlap: bool | None = None,
    require_full_separation: bool | None = None,
) -> SignificanceResult:
    """Validate that a choice set has a decisive winner.

    Optional ``gap_max`` / ``gap_worst_max`` are off unless the caller passes
    them (typically from question-quality options). They are not read from the
    profile significance block, so large-but-informative gaps stay usable by default.

    ``require_full_separation`` *is* read from the profile
    (``significance.require_full_separation``), defaulting to off so profiles
    written before the field keep their exact behaviour.
    """
    sig = profile.significance
    gap_min = float(gap_min if gap_min is not None else sig["gap_min"])
    win_rate_min = float(win_rate_min if win_rate_min is not None else sig["win_rate_min"])
    use_non_overlap = bool(
        use_non_overlap if use_non_overlap is not None else sig.get("use_non_overlap", True)
    )
    require_full_separation = bool(
        require_full_separation
        if require_full_separation is not None
        else sig.get("require_full_separation", False)
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
    worst = int(order[-1])
    gap = float(abs(means[runner_up] - means[winner]))
    gap_worst = float(abs(means[worst] - means[winner]))
    if gap < gap_min:
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=0.0,
            metric=metric,
            winner_index=winner,
            reason=f"gap {gap:.4f} < {gap_min}",
        )
    if gap_max is not None and gap > float(gap_max):
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=0.0,
            metric=metric,
            winner_index=winner,
            reason=f"gap {gap:.4f} > gap_max {float(gap_max)}",
        )
    if gap_worst_max is not None and gap_worst > float(gap_worst_max):
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=0.0,
            metric=metric,
            winner_index=winner,
            reason=f"worst_gap {gap_worst:.4f} > gap_worst_max {float(gap_worst_max)}",
        )

    # The per-seed comparison is paired: seed_results[i] of every choice must be
    # the same seed, or "won on seed i" compares unrelated runs.
    seed_lists = [
        [r.get("seed") for r in s["seed_results"]] for s in summaries
    ]
    if any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=0.0,
            metric=metric,
            winner_index=winner,
            reason="seed lists differ across choices; per-seed wins are not paired",
        )
    n_seeds = len(seed_lists[0])
    # A failed seed has to lose, so it takes the worst value the comparison
    # direction allows: +inf when lower is better, -inf when higher is.
    failed_value = float("-inf") if higher_is_better else float("inf")
    per_seed = [
        [
            failed_value if sr["failed"] else float(sr[final_key])
            for sr in s["seed_results"]
        ]
        for s in summaries
    ]
    wins = 0
    for seed_i in range(n_seeds):
        winner_val = per_seed[winner][seed_i]
        others = [vals[seed_i] for index, vals in enumerate(per_seed) if index != winner]
        # Strictly better, so a tie is not a win: with win_rate_min == 1.0 this
        # makes the criterion "the winner led on every single seed".
        won_this_seed = (
            all(winner_val > other for other in others)
            if higher_is_better
            else all(winner_val < other for other in others)
        )
        if won_this_seed:
            wins += 1
    win_rate = wins / n_seeds

    # Full separation: the winner's *worst* seed must still beat every rival's
    # *best* seed, so the two seed ranges do not overlap at all.
    winner_worst = min(per_seed[winner]) if higher_is_better else max(per_seed[winner])
    rival_bests = [
        max(vals) if higher_is_better else min(vals)
        for index, vals in enumerate(per_seed)
        if index != winner
    ]
    best_rival = max(rival_bests) if higher_is_better else min(rival_bests)
    separation_margin = (
        winner_worst - best_rival if higher_is_better else best_rival - winner_worst
    )

    if win_rate < win_rate_min:
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=win_rate,
            metric=metric,
            winner_index=winner,
            separation_margin=separation_margin,
            reason=f"win_rate {win_rate:.2f} < {win_rate_min}",
        )

    # Separation is strictly stronger than win_rate == 1.0, and the difference
    # matters because win_rate is a *paired* sign test. When two choices share
    # an identical architecture -- every optimizer_only set, by construction --
    # one torch.manual_seed(seed) gives both runs bit-identical initial weights
    # and the same minibatch order, so the paired difference carries almost no
    # variance and any systematic nudge sweeps every seed. Measured: a pair
    # differing only in weight_decay (1e-3 vs 0) took all 10 xor seeds at a mean
    # difference of 0.012%, i.e. 2.6% of one candidate's own seed-to-seed std.
    # Separation is unpaired, so it forces the effect past that spread.
    if require_full_separation and not separation_margin > 0.0:
        return SignificanceResult(
            passed=False,
            gap=gap,
            win_rate=win_rate,
            metric=metric,
            winner_index=winner,
            separation_margin=separation_margin,
            reason=(
                f"seed ranges overlap: winner's worst seed {winner_worst:.6g} "
                f"does not beat the best rival seed {best_rival:.6g}"
            ),
        )

    if use_non_overlap:
        if means[winner] + stds[winner] >= means[runner_up] - stds[runner_up]:
            return SignificanceResult(
                passed=False,
                gap=gap,
                win_rate=win_rate,
                metric=metric,
                winner_index=winner,
                separation_margin=separation_margin,
                reason="non-overlap heuristic failed",
            )

    return SignificanceResult(
        passed=True,
        gap=gap,
        win_rate=win_rate,
        metric=metric,
        winner_index=winner,
        separation_margin=separation_margin,
    )


def load_summary(candidate_path: Path) -> dict[str, Any]:
    return read_json(candidate_path / "results" / "summary.json")
