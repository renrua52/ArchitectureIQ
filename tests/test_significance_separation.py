"""Seed-range separation, the v1.4 significance criterion.

`win_rate_min: 1.0` is a *paired* sign test: seed i of the winner against seed i
of each rival. Two choices that share one model spec -- every candidate in an
optimizer_only set does -- get bit-identical initial weights from a single
`torch.manual_seed(seed)`, so the paired difference carries almost no variance
and any systematic nudge sweeps all 10 seeds no matter how small it is.
Separation is unpaired and cannot be passed that way.
"""

from __future__ import annotations

import pytest

from architecture_iq.profile import load_profile
from architecture_iq.significance.validator import validate_significance

# Measured, not invented: two xor candidates whose candidate_spec.json differ in
# exactly one field (weight_decay 1e-3 vs 0) and whose model.py files are
# byte-identical. Note the shape -- the two lists track each other to the fourth
# decimal while ranging over 0.6726..0.6853 across seeds. That is the paired
# structure: the gap is 0.012% of the mean and 2.6% of either candidate's own
# seed-to-seed std, yet the winner leads on every single seed.
WD_WINNER = [0.677092, 0.677200, 0.678565, 0.682179, 0.679841,
             0.677760, 0.678199, 0.685321, 0.672606, 0.680441]
WD_LOSER = [0.677172, 0.677296, 0.678622, 0.682288, 0.679913,
            0.677876, 0.678263, 0.685388, 0.672718, 0.680511]


def _summary(finals: list[float], *, metric: str = "test_mse", failed: list[bool] | None = None) -> dict:
    flags = failed or [False] * len(finals)
    mean = sum(v for v, f in zip(finals, flags, strict=True) if not f) / max(
        1, sum(1 for f in flags if not f)
    )
    return {
        "excluded": False,
        f"mean_{metric}": mean,
        f"std_{metric}": 0.0,
        "seed_results": [
            {"seed": i, "failed": f, f"final_{metric}": v}
            for i, (v, f) in enumerate(zip(finals, flags, strict=True))
        ],
    }


def test_v14_requires_separation_and_older_profiles_do_not():
    """Only v1.4 opts in, so the profiles that predate the field are unchanged."""
    assert load_profile("v1.4").significance["require_full_separation"] is True
    for name in ("v1", "v1.3"):
        assert "require_full_separation" not in load_profile(name).significance


def test_paired_sweep_with_overlapping_ranges_is_rejected():
    """The 10/10 paired win that motivated the criterion."""
    summaries = [_summary(WD_WINNER), _summary(WD_LOSER), _summary([0.9] * 10)]

    paired_only = validate_significance(
        summaries, load_profile("v1.4"), require_full_separation=False
    )
    assert paired_only.win_rate == 1.0, "the paired test really does pass here"
    assert paired_only.passed

    separated = validate_significance(summaries, load_profile("v1.4"))
    assert not separated.passed
    assert separated.win_rate == 1.0
    assert "seed ranges overlap" in separated.reason
    # Winner's worst seed 0.685321 vs rival's best 0.672718 -- overlapping by far
    # more than the 8.4e-5 mean difference.
    assert separated.separation_margin < 0.0


def test_separated_ranges_pass():
    summaries = [
        _summary([0.10 + 0.001 * i for i in range(10)]),   # 0.100 .. 0.109
        _summary([0.20 + 0.001 * i for i in range(10)]),   # 0.200 .. 0.209
        _summary([0.30] * 10),
    ]
    result = validate_significance(summaries, load_profile("v1.4"))
    assert result.passed
    assert result.winner_index == 0
    assert result.separation_margin == pytest.approx(0.200 - 0.109)


def test_separation_implies_a_perfect_paired_win_rate():
    """So win_rate_min: 1.0 is a cheap first gate, never the binding one."""
    summaries = [
        _summary([0.10, 0.11, 0.12, 0.10, 0.11, 0.12, 0.10, 0.11, 0.12, 0.10]),
        _summary([0.50, 0.20, 0.60, 0.30, 0.55, 0.25, 0.40, 0.35, 0.45, 0.50]),
        _summary([0.70] * 10),
    ]
    result = validate_significance(summaries, load_profile("v1.4"))
    assert result.passed
    assert result.win_rate == 1.0
    assert result.separation_margin == pytest.approx(0.20 - 0.12)


def test_margin_is_recorded_even_when_the_gate_is_off():
    """An audit needs the headroom of already-shipped questions."""
    summaries = [_summary(WD_WINNER), _summary(WD_LOSER), _summary([0.9] * 10)]
    result = validate_significance(
        summaries, load_profile("v1.4"), require_full_separation=False
    )
    assert result.passed
    assert result.separation_margin == pytest.approx(min(WD_LOSER) - max(WD_WINNER))


def test_separation_respects_higher_is_better():
    """Accuracy-style metrics separate in the opposite direction."""
    summaries = [
        _summary([0.90 + 0.001 * i for i in range(10)], metric="test_accuracy"),
        _summary([0.70 + 0.001 * i for i in range(10)], metric="test_accuracy"),
        _summary([0.50] * 10, metric="test_accuracy"),
    ]
    result = validate_significance(
        summaries, load_profile("v1.4"), metric="test_accuracy", higher_is_better=True
    )
    assert result.passed
    assert result.winner_index == 0
    assert result.separation_margin == pytest.approx(0.900 - 0.709)


def test_a_failed_seed_loses_in_both_directions():
    """A failed run took +inf regardless of direction, so under
    higher_is_better it used to beat every rival on that seed."""
    good = _summary([0.90] * 10, metric="test_accuracy")
    broken = _summary(
        [0.10] * 9 + [float("nan")],
        metric="test_accuracy",
        failed=[False] * 9 + [True],
    )
    result = validate_significance(
        [good, broken, _summary([0.05] * 10, metric="test_accuracy")],
        load_profile("v1.4"),
        metric="test_accuracy",
        higher_is_better=True,
    )
    assert result.passed
    assert result.win_rate == 1.0
