"""Shared helpers for ArchitectureIQ ranking questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def count_inversions(predicted_order: list[str], true_order: list[str]) -> int:
    """Count pairwise order mistakes relative to ``true_order``."""
    true_rank = {label: i for i, label in enumerate(true_order)}
    missing = [label for label in true_order if label not in predicted_order]
    extra = [label for label in predicted_order if label not in true_rank]
    if missing or extra or len(predicted_order) != len(true_order):
        raise ValueError(
            "predicted_order must contain exactly the true labels "
            f"(missing={missing}, extra={extra})"
        )

    inversions = 0
    for i, left in enumerate(predicted_order):
        for right in predicted_order[i + 1 :]:
            if true_rank[left] > true_rank[right]:
                inversions += 1
    return inversions


def max_inversions(n_items: int) -> int:
    return n_items * (n_items - 1) // 2


def candidate_metric(summary: dict[str, Any]) -> tuple[str, float, float | None]:
    metric = summary.get("selection_metric", "test_mse")
    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    return metric, float(summary[mean_key]), (
        float(summary[std_key]) if std_key in summary else None
    )


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
