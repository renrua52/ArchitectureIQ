"""Contracts for auditable question exposure and recommendation decisions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import app as inspector_app  # noqa: E402
import feedback  # noqa: E402


def _question(question_id: str = "q_presented") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "question_id": question_id,
        "family": "univariate_regression",
        "choices": [
            {"letter": "A", "candidate_id": "c_a"},
            {"letter": "B", "candidate_id": "c_b"},
        ],
        "correct_letter": "A",
    }


def test_question_presentation_round_trips_and_summarizes_policy() -> None:
    trace = feedback.SessionTrace("anon_presented")
    event = trace.record_question_presented(
        _question(),
        attempt_id="attempt_one",
        release_id="release_one",
        decision_id="decision_one",
        policy_version="surprise_policy_v1",
        mode="exploit",
        propensity=0.85,
        source="next",
        position=2,
        occurred_at="2026-07-12T00:00:01Z",
    )

    assert event["event_type"] == "question_presented"
    assert event["payload"] == {
        "attempt_id": "attempt_one",
        "release_id": "release_one",
        "decision_id": "decision_one",
        "policy_version": "surprise_policy_v1",
        "mode": "exploit",
        "propensity": 0.85,
        "source": "next",
        "position": 2,
    }
    summary = feedback.summarize_session_events(trace.events)
    assert summary["presentations"] == 1
    assert summary["presentation_rows"] == [
        {
            "sequence": 1,
            "occurred_at": "2026-07-12T00:00:01Z",
            "question_id": "q_presented",
            "question_version": feedback.question_version(_question()),
            "policy_version": "surprise_policy_v1",
            "mode": "exploit",
            "propensity": 0.85,
            "source": "next",
            "position": 2,
        }
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "unknown"}, "mode"),
        ({"source": "unknown"}, "source"),
        ({"propensity": 0}, "propensity"),
        ({"propensity": 1.1}, "propensity"),
        ({"propensity": True}, "propensity"),
        ({"position": 0}, "position"),
        ({"position": 1.5}, "position"),
        ({"decision_id": ""}, "decision_id"),
        ({"release_id": None}, "release_id"),
    ],
)
def test_question_presentation_rejects_invalid_policy_facts(
    overrides: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "attempt_id": "attempt_one",
        "release_id": "release_one",
        "decision_id": "decision_one",
        "policy_version": "surprise_policy_v1",
        "mode": "explore",
        "propensity": 0.2,
        "source": "next",
        "position": 1,
    }
    values.update(overrides)

    with pytest.raises(feedback.FeedbackValidationError, match=message):
        feedback.SessionTrace("anon_invalid_presented").record_question_presented(
            _question(),
            **values,
        )


def test_presentation_decision_id_is_idempotent_and_immutable() -> None:
    trace = feedback.SessionTrace("anon_replay_presented")
    values = {
        "attempt_id": "attempt_one",
        "release_id": "release_one",
        "decision_id": "decision_one",
        "policy_version": "surprise_policy_v1",
        "mode": "exploit",
        "propensity": 0.8,
        "source": "next",
        "position": 1,
    }
    first = trace.record_question_presented(_question(), **values)
    replay = trace.record_question_presented(_question(), **values)

    assert replay == first
    assert len(trace.events) == 1
    with pytest.raises(feedback.EventConflictError):
        trace.record_question_presented(
            _question(),
            **{**values, "mode": "explore"},
        )


def test_streamlit_initial_and_next_navigation_record_propensity() -> None:
    import matplotlib
    from streamlit.testing.v1 import AppTest

    matplotlib.use("Agg", force=True)
    app = AppTest.from_file(str(INSPECTOR / "app.py")).run(timeout=30)
    assert not app.exception
    trace = app.session_state["feedback_trace"]
    initial = [
        event for event in trace.events if event["event_type"] == "question_presented"
    ]
    assert len(initial) == 1
    assert initial[0]["payload"]["source"] == "initial"
    assert initial[0]["payload"]["propensity"] == 1.0

    next(button for button in app.button if button.label == "Next").click().run(
        timeout=30
    )
    assert not app.exception
    trace = app.session_state["feedback_trace"]
    presented = [
        event for event in trace.events if event["event_type"] == "question_presented"
    ]
    assert len(presented) == 2
    decision = presented[-1]["payload"]
    assert decision["source"] == "next"
    assert decision["policy_version"] == inspector_app.SURPRISE_POLICY_VERSION
    assert decision["mode"] in {"explore", "exploit"}
    assert 0 < decision["propensity"] <= 1
    assert decision["position"] == 2
