"""Aggregate sequential feedback experiment summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def block_label(block: dict[str, Any]) -> str:
    return str(block.get("block") or block.get("range") or block.get("through"))


def find_block_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("accuracy_by_block", "block_accuracy", "group_summary"):
        rows = summary.get(key)
        if isinstance(rows, list):
            return rows
    return []


def overall(summary: dict[str, Any]) -> dict[str, Any]:
    item = summary.get("overall")
    if isinstance(item, dict):
        return item
    correct = summary.get("correct_count", summary.get("total_correct"))
    total = summary.get("total_questions")
    accuracy = summary.get("overall_accuracy")
    if correct is not None and total is not None:
        return {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(accuracy)
            if accuracy is not None
            else int(correct) / int(total),
        }
    rows = find_block_rows(summary)
    correct = sum(int(row.get("correct", 0)) for row in rows)
    total = sum(int(row.get("total", 0)) for row in rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def aggregate(paths: list[Path]) -> dict[str, Any]:
    experiments = []
    by_block: dict[str, list[float]] = {}

    for path in paths:
        summary = load_json(path)
        total = overall(summary)
        blocks = find_block_rows(summary)
        for row in blocks:
            by_block.setdefault(block_label(row), []).append(float(row["accuracy"]))
        experiments.append(
            {
                "name": summary.get("experiment") or path.stem,
                "path": str(path),
                "correct": total.get("correct"),
                "total": total.get("total"),
                "accuracy": total.get("accuracy"),
                "blocks": [
                    {
                        "block": block_label(row),
                        "correct": row.get("correct"),
                        "total": row.get("total"),
                        "accuracy": row.get("accuracy"),
                    }
                    for row in blocks
                ],
            }
        )

    accuracies = [float(item["accuracy"]) for item in experiments]
    return {
        "experiments": experiments,
        "overall": {
            "n": len(experiments),
            "mean_accuracy": mean(accuracies),
            "stdev_accuracy": stdev(accuracies),
            "min_accuracy": min(accuracies) if accuracies else 0.0,
            "max_accuracy": max(accuracies) if accuracies else 0.0,
        },
        "by_block": [
            {
                "block": label,
                "n": len(values),
                "mean_accuracy": mean(values),
                "stdev_accuracy": stdev(values),
                "min_accuracy": min(values),
                "max_accuracy": max(values),
            }
            for label, values in sorted(by_block.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()

    report = aggregate(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
