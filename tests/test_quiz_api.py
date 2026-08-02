"""Lightweight HTTP contract smoke coverage for the collection-backed quiz API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from services.quiz_api import app as quiz_api


def test_quiz_api_hides_reveal_until_an_answer_is_submitted(
    tmp_path: Path, monkeypatch
) -> None:
    bake_file = tmp_path / "questions.json"
    bake_file.write_text(
        json.dumps(
            {
                "ordered": True,
                "collection": {
                    "collection_id": "smoke_collection",
                    "title": "Smoke collection",
                    "candidate_count": 2,
                    "profiles": ["v1"],
                    "tracks": ["smoke"],
                },
                "questions": [{"question_id": "q_smoke"}],
                "byId": {
                    "q_smoke": {
                        "question_id": "q_smoke",
                        "detail": {"choices": [{"letter": "A"}, {"letter": "B"}]},
                        "reveal": {
                            "correctLetter": "B",
                            "explanation": "smoke",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(quiz_api, "BAKE_FILE", bake_file)

    async def smoke() -> None:
        transport = httpx.ASGITransport(app=quiz_api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://quiz-api.test",
        ) as client:
            health = await client.get("/api/health")
            assert health.json() == {"status": "ok"}

            collection = await client.get("/api/collections/smoke_collection")
            assert collection.json()["question_count"] == 1

            question = await client.get("/api/questions/q_smoke")
            assert question.status_code == 200
            assert "reveal" not in question.json()
            assert question.json()["answer_locked"] is True

            answer = await client.post(
                "/api/questions/q_smoke/answer",
                json={"letter": "b"},
            )
            assert answer.status_code == 200
            assert answer.json() == {
                "question_id": "q_smoke",
                "picked_letter": "B",
                "correct": True,
                "reveal": {"correctLetter": "B", "explanation": "smoke"},
            }

    asyncio.run(smoke())
