#!/usr/bin/env python3
"""Score ArchitectureIQ ranking-question answers by inversion count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from common import count_inversions, max_inversions, read_json, write_json  # noqa: E402


def _normalize_answers(raw: dict[str, Any]) -> dict[str, list[str]]:
    if "answers" in raw and isinstance(raw["answers"], dict):
        raw = raw["answers"]
    normalized: dict[str, list[str]] = {}
    for qid, value in raw.items():
        if isinstance(value, str):
            normalized[qid] = [part.strip() for part in value.replace(">", ",").split(",") if part.strip()]
        elif isinstance(value, list):
            normalized[qid] = [str(part).strip() for part in value]
        else:
            raise ValueError(f"Unsupported answer for {qid}: {value!r}")
    return normalized


def _load_answer_key(answer_key_path: Path) -> dict[str, dict[str, list[str]]]:
    raw = read_json(answer_key_path)
    if "questions" in raw:
        return raw["questions"]
    if "answers" in raw:
        return {
            qid: {"true_order": order}
            for qid, order in raw["answers"].items()
        }
    raise ValueError(f"Unsupported answer key schema: {answer_key_path}")


def score_answers(answer_key_path: Path, answers_path: Path) -> dict[str, Any]:
    answer_key = _load_answer_key(answer_key_path)
    answers = _normalize_answers(json.loads(answers_path.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    total_inv = 0
    total_max = 0
    for qid, key in answer_key.items():
        true_order = key["true_order"]
        predicted = answers.get(qid)
        row: dict[str, Any] = {
            "question_id": qid,
            "true_order": true_order,
            "predicted_order": predicted,
            "max_inversions": max_inversions(len(true_order)),
        }
        if predicted is None:
            row["error"] = "missing answer"
            row["inversions"] = row["max_inversions"]
        else:
            try:
                row["inversions"] = count_inversions(predicted, true_order)
            except ValueError as exc:
                row["error"] = str(exc)
                row["inversions"] = row["max_inversions"]
        total_inv += int(row["inversions"])
        total_max += int(row["max_inversions"])
        rows.append(row)

    return {
        "num_questions": len(rows),
        "total_inversions": total_inv,
        "max_inversions": total_max,
        "normalized_score": 1.0 - (total_inv / total_max if total_max else 0.0),
        "unexpected_answers": sorted(set(answers) - set(answer_key)),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Ranking question run directory, or an answer key JSON path with --answer-key-file.",
    )
    parser.add_argument("answers", type=Path, help="JSON answers file.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--answer-key-file",
        action="store_true",
        help="Treat run_dir as the answer key JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    answer_key = run_dir if args.answer_key_file else run_dir / "answer_key.json"
    if not answer_key.is_file():
        print(f"Missing answer key: {answer_key}", file=sys.stderr)
        return 1
    if not args.answers.is_file():
        print(f"Missing answers: {args.answers}", file=sys.stderr)
        return 1
    try:
        result = score_answers(answer_key, args.answers)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Could not score answers: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        write_json(args.output, result)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
