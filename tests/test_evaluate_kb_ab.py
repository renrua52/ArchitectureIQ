from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_kb_ab import exact_mcnemar_p, select_records  # noqa: E402


def test_select_records_respects_family_quotas_and_seed() -> None:
    records = [
        {
            "item_id": f"{family}#{index:03d}",
            "family": family,
            "status": "ok",
        }
        for family in ("a", "b")
        for index in range(10)
    ]
    quotas = {"a": 3, "b": 2}

    selected = select_records(records, family_quotas=quotas, seed=7)

    assert len(selected) == 5
    assert sum(record["family"] == "a" for record in selected) == 3
    assert sum(record["family"] == "b" for record in selected) == 2
    assert selected == select_records(records, family_quotas=quotas, seed=7)


def test_exact_mcnemar_p() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 5) == 0.0625
    assert exact_mcnemar_p(3, 3) == 1.0
