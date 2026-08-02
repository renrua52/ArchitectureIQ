#!/usr/bin/env python3
"""In-process API smoke for the current baked collection."""

from __future__ import annotations

from fastapi.testclient import TestClient

from services.quiz_api.app import app


def main() -> int:
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    collections = client.get("/api/collections")
    assert collections.status_code == 200
    collection_meta = collections.json()[0]
    collection_id = collection_meta["collection_id"]
    questions = client.get(f"/api/collections/{collection_id}/questions")
    assert questions.status_code == 200 and len(questions.json()["questions"]) == collection_meta["question_count"]
    question_id = questions.json()["questions"][0]["id"]
    public = client.get(f"/api/questions/{question_id}")
    assert public.status_code == 200 and "reveal" not in public.json()
    letters = [choice["letter"] for choice in public.json()["detail"]["choices"]]
    answer = client.post(f"/api/questions/{question_id}/answer", json={"letter": letters[0]})
    assert answer.status_code == 200 and "reveal" in answer.json()
    print("quiz API smoke passed: collection/order/answer-lock/reveal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
