from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from architecture_iq.util import read_json


def score_response(question_path: Path, response: str, num_choices: int | None = None) -> dict:
    q = read_json(question_path / "question.json")
    letter = response.strip().upper()
    correct = q["correct_letter"].upper()
    return {
        "question_id": q["question_id"],
        "response": letter,
        "correct_letter": correct,
        "correct": letter == correct,
        "type": q["type"],
        "family": q["family"],
        "profile": q.get("profile", "v1"),
    }


def evaluate_directory(questions_root: Path) -> dict:
    """Evaluate questions from a directory with per-question response.txt files."""
    results = []
    for qdir in sorted(questions_root.iterdir()):
        if not qdir.is_dir():
            continue
        qfile = qdir / "question.json"
        if not qfile.exists():
            continue
        resp_file = qdir / "response.txt"
        if resp_file.exists():
            results.append(score_response(qdir, resp_file.read_text(encoding="utf-8")))

    return _build_report(results, protocol="blind_per_question")


def evaluate_batch(
    questions_path: Path,
    responses_path: Path,
    *,
    protocol: str = "blind_fullset",
) -> dict:
    """Evaluate from a batch JSON of questions and a batch JSON of responses.

    questions_path: JSON list of sanitized question dicts.
    responses_path: JSON list of response dicts with 'question_id' and 'predicted_letter'.
    """
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    responses = json.loads(responses_path.read_text(encoding="utf-8"))

    q_by_id = {q["question_id"]: q for q in questions}
    r_by_id = {r["question_id"]: r.get("predicted_letter", r.get("response", "")) for r in responses}

    results = []
    for qid, question in q_by_id.items():
        letter = r_by_id.get(qid, "").strip().upper()
        correct = question.get("correct_letter", "").upper()
        results.append({
            "question_id": qid,
            "response": letter,
            "correct_letter": correct,
            "correct": letter == correct,
            "type": question.get("type", "?"),
            "family": question.get("family", "?"),
            "profile": question.get("profile", "v1"),
        })

    return _build_report(results, protocol=protocol)


def evaluate_sequential(
    session_path: Path,
    questions_path: Path,
    *,
    protocol: str = "sequential",
) -> dict:
    """Evaluate from a sequential feedback session.json.

    session_path: Path to session.json from sequential_feedback_session.py.
    questions_path: JSON list of sanitized question dicts (for answer key).
    """
    session = json.loads(session_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    q_by_id = {q["question_id"]: q for q in questions}

    results = []
    for record in session.get("records", []):
        qid = record.get("question_id", "")
        question = q_by_id.get(qid, {})
        predicted = record.get("predicted_letter", "").strip().upper()
        correct = record.get("correct_letter", question.get("correct_letter", "")).strip().upper()
        results.append({
            "question_id": qid,
            "response": predicted,
            "correct_letter": correct,
            "correct": predicted == correct,
            "type": question.get("type", "?"),
            "family": record.get("family", question.get("family", "?")),
            "profile": question.get("profile", "v1"),
        })

    return _build_report(results, protocol=protocol)


def _build_report(results: list[dict[str, Any]], *, protocol: str) -> dict[str, Any]:
    if not results:
        return {"protocol": protocol, "count": 0, "accuracy": None, "results": []}

    acc = sum(1 for r in results if r["correct"]) / len(results)

    by_family: dict[str, dict[str, int]] = {}
    for r in results:
        fam = r["family"]
        by_family.setdefault(fam, {"correct": 0, "total": 0})
        by_family[fam]["total"] += 1
        if r["correct"]:
            by_family[fam]["correct"] += 1

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        qt = r["type"]
        by_type.setdefault(qt, {"correct": 0, "total": 0})
        by_type[qt]["total"] += 1
        if r["correct"]:
            by_type[qt]["correct"] += 1

    # Infer num_choices from first result's question if available
    num_choices = 2  # default for v2
    random_baseline = 1.0 / num_choices

    return {
        "protocol": protocol,
        "count": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "accuracy": acc,
        "random_baseline": random_baseline,
        "by_family": {
            fam: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0,
            }
            for fam, stats in sorted(by_family.items())
        },
        "by_type": {
            qt: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0,
            }
            for qt, stats in sorted(by_type.items())
        },
        "results": results,
    }