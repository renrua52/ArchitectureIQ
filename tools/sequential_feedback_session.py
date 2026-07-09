"""Auditable sequential feedback sessions for quiz attempts.

Normal use:

1. init a session file
2. show the current question
3. submit an answer, which records the prediction before revealing feedback
4. record the lesson learned from that feedback
5. repeat until complete, then write a summary
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def candidate_id_for_letter(question: dict[str, Any], letter: str) -> str | None:
    for choice in question.get("choices", []):
        if choice.get("letter") == letter:
            return choice.get("candidate_id")
    return None


def make_feedback_by_id(feedback_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["question_id"]: item for item in feedback_items}


def init_session(
    session_path: Path,
    questions_path: Path,
    feedback_path: Path,
    experiment_name: str,
    force: bool = False,
) -> dict[str, Any]:
    if session_path.exists() and not force:
        raise FileExistsError(f"Session already exists: {session_path}")
    questions = load_json(questions_path)
    feedback = load_json(feedback_path)
    state = {
        "experiment_name": experiment_name,
        "questions_path": str(questions_path),
        "feedback_path": str(feedback_path),
        "total_questions": len(questions),
        "next_index": 0,
        "records": [],
        "protocol": {
            "prediction_before_feedback": True,
            "tool": "tools/sequential_feedback_session.py",
        },
        "question_ids": [item["question_id"] for item in questions],
        "feedback_ids": [item["question_id"] for item in feedback],
    }
    write_json(session_path, state)
    return state


def load_state(session_path: Path) -> dict[str, Any]:
    return load_json(session_path)


def load_questions_for_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    return load_json(Path(state["questions_path"]))


def load_feedback_for_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return make_feedback_by_id(load_json(Path(state["feedback_path"])))


def current_question(session_path: Path) -> dict[str, Any]:
    state = load_state(session_path)
    questions = load_questions_for_state(state)
    index = int(state["next_index"])
    if index >= len(questions):
        return {"done": True, "total_questions": len(questions)}
    return {
        "done": False,
        "n": index + 1,
        "total_questions": len(questions),
        "question": questions[index],
        "prior_lessons": [
            {
                "n": record["n"],
                "question_id": record["question_id"],
                "family": record["family"],
                "predicted_letter": record["predicted_letter"],
                "correct_letter": record["correct_letter"],
                "is_correct": record["is_correct"],
                "lesson": record.get("lesson", ""),
            }
            for record in state["records"][-8:]
        ],
    }


def submit_answer(
    session_path: Path,
    predicted_letter: str,
    predicted_candidate_id: str | None,
    confidence: float | None,
    reason: str,
) -> dict[str, Any]:
    state = load_state(session_path)
    questions = load_questions_for_state(state)
    feedback_by_id = load_feedback_for_state(state)
    index = int(state["next_index"])
    if index >= len(questions):
        raise ValueError("Session is already complete.")

    predicted_letter = predicted_letter.strip().upper()
    question = questions[index]
    question_id = question["question_id"]
    feedback = feedback_by_id[question_id]
    correct_letter = feedback["correct_letter"]
    if predicted_candidate_id is None:
        predicted_candidate_id = candidate_id_for_letter(question, predicted_letter)
    correct_candidate_id = candidate_id_for_letter(question, correct_letter)
    is_correct = predicted_letter == correct_letter

    previous_correct = sum(1 for record in state["records"] if record["is_correct"])
    cumulative_correct = previous_correct + int(is_correct)
    record = {
        "n": index + 1,
        "question_id": question_id,
        "family": question.get("family"),
        "predicted_letter": predicted_letter,
        "predicted_candidate_id": predicted_candidate_id,
        "confidence": confidence,
        "reason": reason,
        "correct_letter": correct_letter,
        "correct_candidate_id": correct_candidate_id,
        "is_correct": is_correct,
        "metric": feedback.get("metric"),
        "choice_mean_metrics": feedback.get("choice_mean_metrics"),
        "lesson": "",
        "cumulative_correct": cumulative_correct,
        "cumulative_accuracy": cumulative_correct / (index + 1),
    }
    state["records"].append(record)
    state["next_index"] = index + 1
    write_json(session_path, state)
    return {
        "recorded_prediction": {
            "n": record["n"],
            "question_id": question_id,
            "predicted_letter": predicted_letter,
            "predicted_candidate_id": predicted_candidate_id,
            "confidence": confidence,
            "reason": reason,
        },
        "feedback": {
            "correct_letter": correct_letter,
            "correct_candidate_id": correct_candidate_id,
            "is_correct": is_correct,
            "metric": feedback.get("metric"),
            "choice_mean_metrics": feedback.get("choice_mean_metrics"),
            "cumulative_correct": cumulative_correct,
            "cumulative_accuracy": record["cumulative_accuracy"],
        },
        "next_n": state["next_index"] + 1
        if state["next_index"] < len(questions)
        else None,
    }


def record_lesson(session_path: Path, lesson: str) -> dict[str, Any]:
    state = load_state(session_path)
    if not state["records"]:
        raise ValueError("No answer has been submitted yet.")
    state["records"][-1]["lesson"] = lesson
    write_json(session_path, state)
    return {
        "updated_n": state["records"][-1]["n"],
        "question_id": state["records"][-1]["question_id"],
        "lesson": lesson,
    }


def block_name(n: int) -> str:
    start = ((n - 1) // 10) * 10 + 1
    end = min(start + 9, 65)
    return f"{start}-{end}"


def build_summary(state: dict[str, Any]) -> dict[str, Any]:
    records = state["records"]
    total = len(records)
    correct = sum(1 for record in records if record["is_correct"])
    by_block: dict[str, Counter[str]] = defaultdict(Counter)
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        block = block_name(int(record["n"]))
        by_block[block]["total"] += 1
        by_block[block]["correct"] += int(record["is_correct"])
        family = str(record.get("family"))
        by_family[family]["total"] += 1
        by_family[family]["correct"] += int(record["is_correct"])

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        item_total = counter["total"]
        item_correct = counter["correct"]
        return {
            "correct": item_correct,
            "total": item_total,
            "accuracy": item_correct / item_total if item_total else 0.0,
        }

    return {
        "experiment_name": state["experiment_name"],
        "protocol": state["protocol"],
        "total_questions": state["total_questions"],
        "answered_questions": total,
        "correct_count": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "block_accuracy": [
            {"block": key, **summarize(counter)}
            for key, counter in sorted(by_block.items())
        ],
        "by_family": {
            key: summarize(counter) for key, counter in sorted(by_family.items())
        },
        "cumulative_curve": [
            {
                "n": record["n"],
                "cumulative_correct": record["cumulative_correct"],
                "cumulative_accuracy": record["cumulative_accuracy"],
            }
            for record in records
        ],
        "final_lessons": [
            record["lesson"] for record in records if record.get("lesson")
        ][-12:],
    }


def write_summary(session_path: Path, output_path: Path) -> dict[str, Any]:
    summary = build_summary(load_state(session_path))
    write_json(output_path, summary)
    return summary


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--session", type=Path, required=True)
    init_parser.add_argument("--questions", type=Path, required=True)
    init_parser.add_argument("--feedback", type=Path, required=True)
    init_parser.add_argument("--experiment", required=True)
    init_parser.add_argument("--force", action="store_true")

    current_parser = subparsers.add_parser("current")
    current_parser.add_argument("--session", type=Path, required=True)

    answer_parser = subparsers.add_parser("answer")
    answer_parser.add_argument("--session", type=Path, required=True)
    answer_parser.add_argument("--letter", required=True)
    answer_parser.add_argument("--candidate-id")
    answer_parser.add_argument("--confidence", type=float)
    answer_parser.add_argument("--reason", required=True)

    lesson_parser = subparsers.add_parser("lesson")
    lesson_parser.add_argument("--session", type=Path, required=True)
    lesson_parser.add_argument("--lesson", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--session", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "init":
        print_json(
            init_session(
                args.session,
                args.questions,
                args.feedback,
                args.experiment,
                args.force,
            )
        )
    elif args.command == "current":
        print_json(current_question(args.session))
    elif args.command == "answer":
        print_json(
            submit_answer(
                args.session,
                args.letter,
                args.candidate_id,
                args.confidence,
                args.reason,
            )
        )
    elif args.command == "lesson":
        print_json(record_lesson(args.session, args.lesson))
    elif args.command == "summary":
        if args.output:
            print_json(write_summary(args.session, args.output))
        else:
            print_json(build_summary(load_state(args.session)))


if __name__ == "__main__":
    main()
