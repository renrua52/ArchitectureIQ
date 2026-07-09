"""Score and aggregate ArchitectureIQ quiz attempts.

The script accepts answer files produced by human or agent runs. Each answer
item must include a question_id plus either an answer letter or a candidate_id.
It writes a JSON report with per-attempt scores and optional majority voting.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_attempt(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("answers", "items", "attempt", "responses", "predictions"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Attempt file must be a list or contain an answers/items list.")


def build_key(answer_key: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for item in answer_key:
        choices = item["choices"]
        candidate_to_letter = {choice["candidate_id"]: choice["letter"] for choice in choices}
        keyed[item["question_id"]] = {
            **item,
            "candidate_to_letter": candidate_to_letter,
        }
    return keyed


def extract_letter(answer: dict[str, Any], key_item: dict[str, Any]) -> str | None:
    raw = (
        answer.get("answer")
        or answer.get("predicted_letter")
        or answer.get("letter")
        or answer.get("choice")
    )
    if isinstance(raw, str):
        raw = raw.strip().upper()
        if raw in {"A", "B", "C", "D", "E"}:
            return raw
        candidate_letter = key_item["candidate_to_letter"].get(raw)
        if candidate_letter:
            return candidate_letter

    candidate_id = answer.get("candidate_id") or answer.get("predicted_candidate_id")
    if isinstance(candidate_id, str):
        return key_item["candidate_to_letter"].get(candidate_id.strip())

    return None


def score_attempt(path: Path, answer_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_answers = normalize_attempt(load_json(path))
    answers_by_id: dict[str, dict[str, Any]] = {}
    for answer in raw_answers:
        question_id = answer.get("question_id") if isinstance(answer, dict) else None
        if isinstance(question_id, str):
            answers_by_id[question_id] = answer

    rows: list[dict[str, Any]] = []
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    missing: list[str] = []
    invalid: list[str] = []

    for question_id, key_item in answer_key.items():
        answer = answers_by_id.get(question_id)
        predicted = extract_letter(answer, key_item) if answer else None
        if answer is None:
            missing.append(question_id)
        elif predicted is None:
            invalid.append(question_id)

        is_correct = predicted == key_item["correct_letter"]
        if is_correct:
            correct += 1

        family = key_item["family"]
        by_family[family]["total"] += 1
        by_family[family]["correct"] += int(is_correct)
        rows.append(
            {
                "question_id": question_id,
                "family": family,
                "predicted_letter": predicted,
                "correct_letter": key_item["correct_letter"],
                "is_correct": is_correct,
            }
        )

    total = len(answer_key)
    return {
        "name": path.stem,
        "path": str(path),
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "missing": missing,
        "invalid": invalid,
        "by_family": summarize_counters(by_family),
        "rows": rows,
    }


def summarize_counters(counters: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, counter in sorted(counters.items()):
        total = counter["total"]
        correct = counter["correct"]
        summary[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return summary


def majority_vote(attempts: list[dict[str, Any]], answer_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_maps = {
        attempt["name"]: {row["question_id"]: row for row in attempt["rows"]}
        for attempt in attempts
    }
    rows: list[dict[str, Any]] = []
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0

    for question_id, key_item in answer_key.items():
        votes = [
            row_map[question_id]["predicted_letter"]
            for row_map in row_maps.values()
            if row_map[question_id]["predicted_letter"]
        ]
        counts = Counter(votes)
        predicted = None
        tie = False
        if counts:
            top_count = counts.most_common(1)[0][1]
            winners = sorted(letter for letter, count in counts.items() if count == top_count)
            predicted = winners[0]
            tie = len(winners) > 1

        is_correct = predicted == key_item["correct_letter"]
        correct += int(is_correct)
        family = key_item["family"]
        by_family[family]["total"] += 1
        by_family[family]["correct"] += int(is_correct)
        rows.append(
            {
                "question_id": question_id,
                "family": family,
                "predicted_letter": predicted,
                "correct_letter": key_item["correct_letter"],
                "is_correct": is_correct,
                "tie": tie,
                "votes": dict(counts),
            }
        )

    total = len(answer_key)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "by_family": summarize_counters(by_family),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("attempts", type=Path, nargs="+")
    args = parser.parse_args()

    answer_key = build_key(load_json(args.answer_key))
    scored_attempts = [score_attempt(path, answer_key) for path in args.attempts]
    report = {
        "attempts": [
            {key: value for key, value in attempt.items() if key != "rows"}
            for attempt in scored_attempts
        ],
        "majority_vote": majority_vote(scored_attempts, answer_key)
        if len(scored_attempts) >= 2
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
