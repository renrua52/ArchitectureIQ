#!/usr/bin/env python3
"""Run support-feedback followed by no-feedback holdout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _feedback_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    items = payload.get("questions", payload) if isinstance(payload, dict) else payload
    return {str(item["question_id"]): item for item in items}


def _candidate_id(question: dict[str, Any], letter: str) -> str | None:
    for choice in question.get("choices", []):
        if str(choice.get("letter")).upper() == letter:
            return str(choice.get("candidate_id"))
    return None


def init_session(
    session_path: Path,
    collection_dir: Path,
    feedback_path: Path,
    experiment_name: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if session_path.exists() and not force:
        raise FileExistsError(f"Session already exists: {session_path}")
    collection_dir = collection_dir.resolve()
    support = load_json(collection_dir / "support.json")
    holdout = load_json(collection_dir / "holdout.json")
    feedback = _feedback_by_id(feedback_path)
    question_ids = [str(item["question_id"]) for item in [*support, *holdout]]
    missing = [question_id for question_id in question_ids if question_id not in feedback]
    if missing:
        raise ValueError("Missing private feedback for: " + ", ".join(missing))
    state = {
        "schema_version": "leakage_safe_feedback_session_v1",
        "experiment_name": experiment_name,
        "collection_dir": str(collection_dir),
        "feedback_path": str(feedback_path.resolve()),
        "support_index": 0,
        "holdout_index": 0,
        "support_records": [],
        "holdout_records": [],
        "frozen_lessons": [],
        "protocol": {
            "support_feedback": True,
            "holdout_feedback": False,
            "lessons_frozen_before_holdout": True,
        },
    }
    write_json(session_path, state)
    return state


def _phase_and_question(state: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    collection_dir = Path(state["collection_dir"])
    support = load_json(collection_dir / "support.json")
    holdout = load_json(collection_dir / "holdout.json")
    support_index = int(state["support_index"])
    if support_index < len(support):
        return "support", support[support_index]
    holdout_index = int(state["holdout_index"])
    if holdout_index < len(holdout):
        return "holdout", holdout[holdout_index]
    return "complete", None


def current_question(session_path: Path) -> dict[str, Any]:
    state = load_json(session_path)
    phase, question = _phase_and_question(state)
    if question is None:
        return {"done": True, "phase": "complete"}
    lessons = (
        [record.get("lesson", "") for record in state["support_records"]]
        if phase == "support"
        else state["frozen_lessons"]
    )
    return {
        "done": False,
        "phase": phase,
        "question": question,
        "prior_lessons": [lesson for lesson in lessons if lesson][-12:],
    }


def submit_answer(
    session_path: Path,
    predicted_letter: str,
    *,
    confidence: float | None = None,
    reason: str = "",
) -> dict[str, Any]:
    state = load_json(session_path)
    phase, question = _phase_and_question(state)
    if question is None:
        raise ValueError("Session is already complete")
    predicted_letter = predicted_letter.strip().upper()
    valid_letters = {str(choice["letter"]).upper() for choice in question["choices"]}
    if predicted_letter not in valid_letters:
        raise ValueError(f"Invalid letter {predicted_letter!r}; expected {sorted(valid_letters)}")
    question_id = str(question["question_id"])
    prediction = {
        "question_id": question_id,
        "predicted_letter": predicted_letter,
        "predicted_candidate_id": _candidate_id(question, predicted_letter),
        "confidence": confidence,
        "reason": reason,
    }

    if phase == "support":
        feedback = _feedback_by_id(Path(state["feedback_path"]))[question_id]
        correct_letter = str(feedback["correct_letter"]).upper()
        record = {
            **prediction,
            "correct_letter": correct_letter,
            "is_correct": predicted_letter == correct_letter,
            "metric": feedback.get("metric"),
            "choice_mean_metrics": feedback.get("choice_mean_metrics"),
            "lesson": "",
        }
        state["support_records"].append(record)
        state["support_index"] = int(state["support_index"]) + 1
        if _phase_and_question(state)[0] == "holdout":
            state["frozen_lessons"] = [
                item["lesson"] for item in state["support_records"] if item.get("lesson")
            ][-12:]
        write_json(session_path, state)
        return {
            "phase": "support",
            "recorded_prediction": prediction,
            "feedback": {
                "correct_letter": correct_letter,
                "is_correct": record["is_correct"],
                "metric": record["metric"],
                "choice_mean_metrics": record["choice_mean_metrics"],
            },
        }

    state["holdout_records"].append(prediction)
    state["holdout_index"] = int(state["holdout_index"]) + 1
    write_json(session_path, state)
    return {"phase": "holdout", "recorded_prediction": prediction}


def record_lesson(session_path: Path, lesson: str) -> dict[str, Any]:
    state = load_json(session_path)
    phase, _ = _phase_and_question(state)
    if state["holdout_records"] or phase == "complete":
        raise ValueError("Lessons are frozen once holdout begins")
    if not state["support_records"]:
        raise ValueError("No support answer has been submitted")
    state["support_records"][-1]["lesson"] = lesson
    state["frozen_lessons"] = [
        item["lesson"] for item in state["support_records"] if item.get("lesson")
    ][-12:]
    write_json(session_path, state)
    return {"question_id": state["support_records"][-1]["question_id"], "lesson": lesson}


def _score(records: list[dict[str, Any]], feedback: dict[str, dict[str, Any]]) -> dict[str, Any]:
    correct = sum(
        str(record["predicted_letter"]).upper()
        == str(feedback[record["question_id"]]["correct_letter"]).upper()
        for record in records
    )
    total = len(records)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def build_summary(session_path: Path, blind_score: float | None = None) -> dict[str, Any]:
    state = load_json(session_path)
    feedback = _feedback_by_id(Path(state["feedback_path"]))
    return {
        "experiment_name": state["experiment_name"],
        "protocol": state["protocol"],
        "blind_score": blind_score,
        "support_sequential_score": _score(state["support_records"], feedback),
        "post_feedback_holdout_score": _score(state["holdout_records"], feedback),
        "frozen_lessons": state["frozen_lessons"],
        "complete": _phase_and_question(state)[0] == "complete",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--session", type=Path, required=True)
    init.add_argument("--collection", type=Path, required=True)
    init.add_argument("--feedback", type=Path, required=True)
    init.add_argument("--experiment", required=True)
    init.add_argument("--force", action="store_true")
    current = subparsers.add_parser("current")
    current.add_argument("--session", type=Path, required=True)
    answer = subparsers.add_parser("answer")
    answer.add_argument("--session", type=Path, required=True)
    answer.add_argument("--letter", required=True)
    answer.add_argument("--confidence", type=float)
    answer.add_argument("--reason", default="")
    lesson = subparsers.add_parser("lesson")
    lesson.add_argument("--session", type=Path, required=True)
    lesson.add_argument("--text", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--session", type=Path, required=True)
    summary.add_argument("--blind-score", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init":
        result = init_session(
            args.session,
            args.collection,
            args.feedback,
            args.experiment,
            force=args.force,
        )
    elif args.command == "current":
        result = current_question(args.session)
    elif args.command == "answer":
        result = submit_answer(
            args.session,
            args.letter,
            confidence=args.confidence,
            reason=args.reason,
        )
    elif args.command == "lesson":
        result = record_lesson(args.session, args.text)
    else:
        result = build_summary(args.session, args.blind_score)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
