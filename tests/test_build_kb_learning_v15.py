from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_kb_learning_v15 import LEARNING_SEED_BASE, build_epoch_plan  # noqa: E402


def test_learning_plan_has_balanced_fresh_questions() -> None:
    first = build_epoch_plan(1)
    second = build_epoch_plan(2)

    assert len(first) == len(second) == 50
    assert {item["item_id"] for item in first}.isdisjoint(
        item["item_id"] for item in second
    )
    assert {item["seed_base"] for item in first}.isdisjoint(
        item["seed_base"] for item in second
    )
    assert min(item["seed_base"] for item in first) >= min(LEARNING_SEED_BASE.values())

    family_counts = Counter(item["family"] for item in first)
    type_counts = Counter(item["question_type"] for item in first)
    budget_counts = Counter(item["budget"] for item in first)
    assert max(family_counts.values()) - min(family_counts.values()) <= 1
    assert max(type_counts.values()) - min(type_counts.values()) <= 1
    assert max(budget_counts.values()) - min(budget_counts.values()) <= 1
