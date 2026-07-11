"""Export ArchitectureIQ artifacts as static JSON for the deployed worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app import app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "vanilla" / "src" / "generated" / "questions.json"


def main() -> None:
    client = TestClient(app)
    questions = client.get("/api/questions").json()["questions"]
    selected = _latest_sixty(questions)
    details: dict[str, Any] = {}
    answers: dict[str, Any] = {}

    for summary in selected:
        qid = summary["id"]
        detail = client.get(f"/api/questions/{qid}").json()
        details[qid] = detail
        first_letter = detail["choices"][0]["letter"]
        answer = client.post(f"/api/questions/{qid}/answer", json={"letter": first_letter}).json()
        answers[qid] = {
            "correctLetter": answer["correctLetter"],
            "ranked": answer["ranked"],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "questions": selected,
                "details": details,
                "answers": answers,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"exported {len(selected)} questions to {OUT}")


def _latest_sixty(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_60 = [q for q in questions if str(q["id"]).startswith("q_")]
    if len(run_60) <= 60:
        return run_60
    return run_60[-60:]


if __name__ == "__main__":
    main()
