"""Read-only collection API with answer reveal after an explicit POST.

The API reads the same BakeFile used by the React quiz.  It intentionally has
no training or telemetry side effects: question GETs strip ``reveal`` and the
answer endpoint returns it only for the requested question.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
BAKE_FILE = Path(
    os.environ.get(
        "QUIZ_BAKE_FILE",
        str(ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"),
    )
).expanduser()

app = FastAPI(title="ArchitectureIQ Quiz API", version="0.1.0")


class AnswerRequest(BaseModel):
    letter: str = Field(min_length=1, max_length=1)
    session_id: str | None = Field(default=None, min_length=8, max_length=128)


def _bake() -> dict[str, Any]:
    if not BAKE_FILE.is_file():
        raise HTTPException(status_code=503, detail=f"BakeFile is missing: {BAKE_FILE}")
    import json

    payload = json.loads(BAKE_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("byId"), dict):
        raise HTTPException(status_code=503, detail="BakeFile schema is invalid")
    return payload


def _collection_summary(payload: dict[str, Any]) -> dict[str, Any]:
    collection = payload.get("collection") or {}
    questions = payload.get("questions") or []
    return {
        "collection_id": collection.get("collection_id", "default"),
        "title": collection.get("title", "ArchitectureIQ Quiz"),
        "question_count": len(questions),
        "candidate_count": collection.get("candidate_count"),
        "profiles": collection.get("profiles", []),
        "tracks": collection.get("tracks", []),
        "ordered": bool(payload.get("ordered", False)),
    }


def _collection_id(payload: dict[str, Any]) -> str:
    return str(_collection_summary(payload)["collection_id"])


def _public_question(item: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in item.items() if key not in {"reveal"}}
    public["answer_locked"] = True
    return public


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        _bake()
    except HTTPException:
        return {"status": "misconfigured"}
    return {"status": "ok"}


@app.get("/api/collections")
def list_collections() -> list[dict[str, Any]]:
    return [_collection_summary(_bake())]


@app.get("/api/collections/{collection_id}")
def get_collection(collection_id: str) -> dict[str, Any]:
    payload = _bake()
    if collection_id != _collection_id(payload):
        raise HTTPException(status_code=404, detail="Unknown collection")
    return _collection_summary(payload)


@app.get("/api/collections/{collection_id}/questions")
def list_collection_questions(collection_id: str) -> dict[str, Any]:
    payload = _bake()
    if collection_id != _collection_id(payload):
        raise HTTPException(status_code=404, detail="Unknown collection")
    return {
        "collection": _collection_summary(payload),
        "questions": payload.get("questions", []),
    }


@app.get("/api/questions/{question_id}")
def get_question(question_id: str) -> dict[str, Any]:
    payload = _bake()
    item = payload["byId"].get(question_id)
    if not isinstance(item, dict):
        raise HTTPException(status_code=404, detail="Unknown question")
    return _public_question(item)


@app.post("/api/questions/{question_id}/answer")
def answer_question(question_id: str, body: AnswerRequest) -> dict[str, Any]:
    payload = _bake()
    item = payload["byId"].get(question_id)
    if not isinstance(item, dict):
        raise HTTPException(status_code=404, detail="Unknown question")
    correct_letter = str(item.get("reveal", {}).get("correctLetter", ""))
    if body.letter.upper() not in {choice.get("letter") for choice in item.get("detail", {}).get("choices", [])}:
        raise HTTPException(status_code=422, detail="Answer letter is not a valid choice")
    return {
        "question_id": question_id,
        "picked_letter": body.letter.upper(),
        "correct": body.letter.upper() == correct_letter,
        "reveal": item.get("reveal", {}),
    }
