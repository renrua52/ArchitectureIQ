from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "leakage_safe_feedback_session",
    ROOT / "tools" / "leakage_safe_feedback_session.py",
)
assert SPEC is not None and SPEC.loader is not None
SESSION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SESSION
SPEC.loader.exec_module(SESSION)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_support_feedback_then_holdout_without_feedback(tmp_path: Path) -> None:
    collection_dir = tmp_path / "collection"
    support = [
        {
            "question_id": "q_support",
            "family": "demo",
            "dataset_id": "d1",
            "prompt": "Support prompt",
            "choices": [
                {"letter": "A", "candidate_id": "c1"},
                {"letter": "B", "candidate_id": "c2"},
            ],
        }
    ]
    holdout = [
        {
            "question_id": "q_holdout",
            "family": "demo",
            "dataset_id": "d1",
            "prompt": "Holdout prompt",
            "choices": [
                {"letter": "A", "candidate_id": "c3"},
                {"letter": "B", "candidate_id": "c4"},
            ],
        }
    ]
    feedback = [
        {
            "question_id": "q_support",
            "correct_letter": "B",
            "metric": "loss",
            "choice_mean_metrics": {"A": 2.0, "B": 1.0},
        },
        {
            "question_id": "q_holdout",
            "correct_letter": "A",
            "metric": "loss",
            "choice_mean_metrics": {"A": 0.5, "B": 1.5},
        },
    ]
    _write_json(collection_dir / "support.json", support)
    _write_json(collection_dir / "holdout.json", holdout)
    feedback_path = tmp_path / "private_feedback.json"
    session_path = tmp_path / "session.json"
    _write_json(feedback_path, feedback)

    SESSION.init_session(session_path, collection_dir, feedback_path, "smoke")
    support_result = SESSION.submit_answer(session_path, "A", reason="first")
    SESSION.record_lesson(session_path, "Prefer the lower expected loss.")
    current = SESSION.current_question(session_path)
    holdout_result = SESSION.submit_answer(session_path, "A", reason="apply lesson")
    summary = SESSION.build_summary(session_path, blind_score=0.25)

    assert support_result["feedback"]["correct_letter"] == "B"
    assert current["phase"] == "holdout"
    assert current["prior_lessons"] == ["Prefer the lower expected loss."]
    assert "feedback" not in holdout_result
    assert "correct_letter" not in holdout_result["recorded_prediction"]
    assert summary["blind_score"] == 0.25
    assert summary["support_sequential_score"]["accuracy"] == 0.0
    assert summary["post_feedback_holdout_score"]["accuracy"] == 1.0
    assert summary["complete"] is True

    with pytest.raises(ValueError, match="frozen"):
        SESSION.record_lesson(session_path, "Too late")
