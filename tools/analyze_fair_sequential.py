"""Validate and aggregate matched full-sequential feedback runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODEL_DIRS = {
    "gpt54": "gpt-5.4",
    "gpt55": "gpt-5.5",
    "gpt56_sol": "gpt-5.6-sol",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ratio(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def validate_run(
    session_path: Path,
    questions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    state = load_json(session_path)
    records = state.get("records", [])
    errors: list[str] = []
    if state.get("total_questions") != len(questions):
        errors.append("total_questions does not match the clean question set")
    if state.get("next_index") != len(questions):
        errors.append(f"incomplete next_index={state.get('next_index')}")
    if len(records) != len(questions):
        errors.append(f"expected {len(questions)} records, found {len(records)}")
    if not state.get("protocol", {}).get("prediction_before_feedback"):
        errors.append("prediction_before_feedback is not enabled")

    seen_candidates: set[str] = set()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (question, record) in enumerate(zip(questions, records), start=1):
        if record.get("n") != index:
            errors.append(f"record {index}: wrong n={record.get('n')}")
        if record.get("question_id") != question.get("question_id"):
            errors.append(f"record {index}: question_id/order mismatch")
        choices = {choice["letter"]: choice for choice in question["choices"]}
        predicted_letter = record.get("predicted_letter")
        correct_letter = record.get("correct_letter")
        if predicted_letter not in choices:
            errors.append(f"record {index}: invalid predicted letter")
        elif record.get("predicted_candidate_id") != choices[predicted_letter]["candidate_id"]:
            errors.append(f"record {index}: predicted candidate/letter mismatch")
        if correct_letter not in choices:
            errors.append(f"record {index}: invalid correct letter")
            correct_candidate = None
        else:
            correct_candidate = choices[correct_letter]["candidate_id"]
            if record.get("correct_candidate_id") != correct_candidate:
                errors.append(f"record {index}: correct candidate/letter mismatch")
        expected_correct = predicted_letter == correct_letter
        if record.get("is_correct") is not expected_correct:
            errors.append(f"record {index}: is_correct mismatch")
        if not str(record.get("lesson", "")).strip():
            errors.append(f"record {index}: missing lesson")

        candidate_ids = [choice["candidate_id"] for choice in question["choices"]]
        known_count = sum(candidate_id in seen_candidates for candidate_id in candidate_ids)
        labels = ["none_seen" if known_count == 0 else "some_seen"]
        if correct_candidate in seen_candidates:
            labels.append("winner_seen")
        if known_count == len(candidate_ids):
            labels.append("all_seen")
        for label in labels:
            strata[label]["total"] += 1
            strata[label]["correct"] += int(expected_correct)
        seen_candidates.update(candidate_ids)

    correct = sum(int(record.get("is_correct", False)) for record in records)
    return (
        {
            "session": str(session_path),
            "valid": not errors,
            "errors": errors,
            **ratio(correct, len(records)),
        },
        strata,
    )


def aggregate(root: Path, expected_runs: int) -> dict[str, Any]:
    quiz_root = root.parents[0]
    questions = load_json(quiz_root / "questions_sanitized.json")
    output: dict[str, Any] = {
        "experiment": "fair full-sequential feedback evaluation v2",
        "expected_runs_per_model": expected_runs,
        "models": {},
        "valid": True,
        "errors": [],
    }
    for model_dir, model_name in MODEL_DIRS.items():
        run_paths = sorted((root / model_dir).glob("run_*/session.json"))
        if len(run_paths) != expected_runs:
            output["valid"] = False
            output["errors"].append(
                f"{model_name}: expected {expected_runs} runs, found {len(run_paths)}"
            )
        runs: list[dict[str, Any]] = []
        combined_strata: dict[str, Counter[str]] = defaultdict(Counter)
        family_counts: dict[str, Counter[str]] = defaultdict(Counter)
        block_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for session_path in run_paths:
            run, strata = validate_run(session_path, questions)
            runs.append(run)
            if not run["valid"]:
                output["valid"] = False
                output["errors"].append(f"{model_name}/{session_path.parent.name}: invalid")
            state = load_json(session_path)
            for label, counts in strata.items():
                combined_strata[label].update(counts)
            for record in state.get("records", []):
                family = str(record.get("family"))
                block = f"{((int(record['n']) - 1) // 10) * 10 + 1}-{((int(record['n']) - 1) // 10 + 1) * 10}"
                for key, bucket in ((family, family_counts), (block, block_counts)):
                    bucket[key]["total"] += 1
                    bucket[key]["correct"] += int(record.get("is_correct", False))
        accuracies = [run["accuracy"] for run in runs if run["accuracy"] is not None]
        output["models"][model_name] = {
            "runs": runs,
            "mean_accuracy": statistics.mean(accuracies) if accuracies else None,
            "sample_stdev_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else None,
            "by_family": {
                key: ratio(value["correct"], value["total"])
                for key, value in sorted(family_counts.items())
            },
            "by_block": {
                key: ratio(value["correct"], value["total"])
                for key, value in sorted(block_counts.items())
            },
            "candidate_history": {
                key: ratio(value["correct"], value["total"])
                for key, value in sorted(combined_strata.items())
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/quiz_attempt_60/fair_sequential_v2"),
    )
    parser.add_argument("--expected-runs", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = aggregate(args.root, args.expected_runs)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
