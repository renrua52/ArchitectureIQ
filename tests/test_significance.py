from __future__ import annotations

from architecture_iq.profile import load_profile
from architecture_iq.significance.validator import validate_significance


def _summary(mean: float, std: float, finals: list[float]) -> dict:
    return {
        "excluded": False,
        "mean_test_mse": mean,
        "std_test_mse": std,
        "seed_results": [{"failed": False, "final_test_mse": v} for v in finals],
    }


def test_significance_pass():
    profile = load_profile("v1")
    summaries = [
        _summary(0.1, 0.01, [0.09, 0.10, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
        _summary(0.5, 0.02, [0.5] * 10),
        _summary(0.6, 0.02, [0.6] * 10),
        _summary(0.7, 0.02, [0.7] * 10),
    ]
    result = validate_significance(summaries, profile)
    assert result.passed
    assert result.winner_index == 0


def test_significance_fail_gap():
    profile = load_profile("v1")
    summaries = [
        _summary(0.10, 0.01, [0.10] * 10),
        _summary(0.11, 0.01, [0.11] * 10),
        _summary(0.60, 0.02, [0.60] * 10),
        _summary(0.70, 0.02, [0.70] * 10),
    ]
    result = validate_significance(summaries, profile)
    assert not result.passed
