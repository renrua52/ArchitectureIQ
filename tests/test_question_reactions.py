"""Contracts for explicit post-result surprise reactions."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import feedback  # noqa: E402
import feedback_recovery  # noqa: E402


def _question() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "question_id": "q_surprise",
        "family": "univariate_regression",
        "choices": [
            {"letter": "A", "candidate_id": "c_a"},
            {"letter": "B", "candidate_id": "c_b"},
        ],
        "correct_letter": "B",
    }


@pytest.mark.parametrize("value", [True, False])
def test_question_reaction_round_trips_and_is_summarized(value: bool) -> None:
    trace = feedback.SessionTrace("anon_surprise")
    event = trace.record_question_reaction(
        _question(),
        value=value,
        attempt_id="attempt_one",
        release_id="release_one",
        occurred_at="2026-07-12T00:00:01Z",
        extra={"family": "univariate_regression"},
    )

    assert event["event_type"] == "question_reaction_submitted"
    assert event["payload"] == {
        "reaction": "surprise",
        "value": value,
        "timing": "after_reveal",
        "attempt_id": "attempt_one",
        "release_id": "release_one",
        "family": "univariate_regression",
    }
    envelope = trace.to_envelope()
    assert envelope["events"][0] == event

    summary = feedback.summarize_session_events(trace.events)
    assert summary["reactions"] == {
        "total": 1,
        "surprised": int(value),
        "not_surprised": int(not value),
    }
    assert summary["reaction_rows"][0]["value"] is value


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_question_reaction_rejects_non_boolean_values(value: Any) -> None:
    trace = feedback.SessionTrace("anon_invalid_surprise")

    with pytest.raises(feedback.FeedbackValidationError, match="boolean"):
        trace.record_question_reaction(
            _question(),
            value=value,
            attempt_id="attempt_one",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "reaction": "like",
                "value": True,
                "timing": "after_reveal",
                "attempt_id": "attempt_one",
            },
            "reaction",
        ),
        (
            {
                "reaction": "surprise",
                "value": True,
                "timing": "before_reveal",
                "attempt_id": "attempt_one",
            },
            "timing",
        ),
        (
            {
                "reaction": "surprise",
                "value": True,
                "timing": "after_reveal",
            },
            "attempt_id",
        ),
    ],
)
def test_question_reaction_payload_contract_is_strict(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(feedback.FeedbackValidationError, match=message):
        feedback.build_event(
            "question_reaction_submitted",
            session_id="anon_payload",
            question=_question(),
            payload=payload,
        )


def test_question_reaction_has_one_stable_vote_per_attempt() -> None:
    trace = feedback.SessionTrace("anon_stable_surprise")
    first = trace.record_question_reaction(
        _question(),
        value=True,
        attempt_id="attempt_one",
        release_id="release_one",
        occurred_at="2026-07-12T00:00:01Z",
    )
    replay = trace.record_question_reaction(
        _question(),
        value=True,
        attempt_id="attempt_one",
        release_id="release_one",
        occurred_at="2026-07-12T00:00:02Z",
    )

    assert replay == first
    assert len(trace.events) == 1
    with pytest.raises(feedback.EventConflictError):
        trace.record_question_reaction(
            _question(),
            value=False,
            attempt_id="attempt_one",
            release_id="release_one",
        )

    second_attempt = trace.record_question_reaction(
        _question(),
        value=False,
        attempt_id="attempt_two",
        release_id="release_one",
    )
    assert second_attempt["event_id"] != first["event_id"]
    assert len(trace.events) == 2


def test_download_recovery_keeps_question_reaction_pending() -> None:
    trace = feedback.SessionTrace("anon_recover_surprise")
    event = trace.record_question_reaction(
        _question(),
        value=False,
        attempt_id="attempt_recover",
        release_id="release_one",
    )

    recovered = feedback_recovery.parse_recovered_trace(
        json.dumps(trace.to_envelope()).encode("utf-8")
    )

    assert recovered.events == (event,)
    assert recovered.events[0]["payload"]["value"] is False


def test_streamlit_only_offers_surprise_reaction_after_answer() -> None:
    import matplotlib
    from streamlit.testing.v1 import AppTest

    matplotlib.use("Agg", force=True)
    app = AppTest.from_file(str(INSPECTOR / "app.py")).run(timeout=30)
    assert not app.exception
    assert not [button for button in app.button if "出乎意料" in button.label]

    app.button(key="select_A").click().run(timeout=30)
    assert not app.exception
    reaction_buttons = [
        button
        for button in app.button
        if button.label in {"😮 Surprised / 出乎意料", "As expected / 符合预期"}
    ]
    assert len(reaction_buttons) == 2

    reaction_buttons[0].click().run(timeout=30)
    assert not app.exception
    trace = app.session_state["feedback_trace"]
    reactions = [
        event
        for event in trace.events
        if event["event_type"] == "question_reaction_submitted"
    ]
    assert len(reactions) == 1
    assert reactions[0]["payload"]["value"] is True
    assert not [
        button
        for button in app.button
        if button.label in {"😮 Surprised / 出乎意料", "As expected / 符合预期"}
    ]
