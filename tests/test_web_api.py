import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import app


def test_web_api_question_flow():
    client = TestClient(app)

    questions_response = client.get("/api/questions")
    assert questions_response.status_code == 200
    questions = questions_response.json()["questions"]
    assert questions

    question_id = questions[0]["id"]
    question_response = client.get(f"/api/questions/{question_id}")
    assert question_response.status_code == 200
    question = question_response.json()
    assert question["id"] == question_id
    assert question["title"]
    assert question["choices"]
    assert question["shared"]

    first_letter = question["choices"][0]["letter"]
    answer_response = client.post(
        f"/api/questions/{question_id}/answer",
        json={"letter": first_letter},
    )
    assert answer_response.status_code == 200
    answer = answer_response.json()
    assert answer["picked"] == first_letter
    assert answer["correctLetter"]
    assert answer["ranked"]
