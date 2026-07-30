from __future__ import annotations

import pytest

from architecture_iq.profile import load_profile
from architecture_iq.significance.validator import validate_significance


def _summary(mean: float, std: float, *, seed_value: float) -> dict:
    return {
        "mean_test_ce": mean,
        "std_test_ce": std,
        "excluded": False,
        "seed_results": [
            {"seed": seed, "failed": False, "final_test_ce": seed_value}
            for seed in range(10)
        ],
    }


def test_xor_review_keeps_non_overlap_while_removing_absolute_gap() -> None:
    result = validate_significance(
        [
            _summary(0.100, 0.001, seed_value=0.100),
            _summary(0.110, 0.001, seed_value=0.110),
        ],
        load_profile("v2.4-xor-review"),
        metric="test_ce",
    )

    assert result.passed
    assert result.gap == pytest.approx(0.01)
    assert result.win_rate == 1.0
