from __future__ import annotations

from pathlib import Path

from architecture_iq.util import read_json


def score_response(question_path: Path, response: str) -> dict:
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
    results = []
    for qdir in sorted(questions_root.iterdir()):
        if not qdir.is_dir():
            continue
        qfile = qdir / "question.json"
        if not qfile.exists():
            continue
        # Placeholder: no response file means skip in batch eval
        resp_file = qdir / "response.txt"
        if resp_file.exists():
            results.append(score_response(qdir, resp_file.read_text(encoding="utf-8")))

    if not results:
        return {"count": 0, "accuracy": None, "results": []}

    acc = sum(1 for r in results if r["correct"]) / len(results)
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["correct"])

    return {
        "count": len(results),
        "accuracy": acc,
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
        "results": results,
    }
