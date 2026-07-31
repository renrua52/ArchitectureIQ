"""Strict state and app integration tests for the browser live outbox."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
INSPECTOR = REPO / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import app as inspector_app  # noqa: E402
import feedback  # noqa: E402
import feedback_browser_outbox as browser_outbox  # noqa: E402


class _SessionState(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _trace() -> feedback.SessionTrace:
    trace = feedback.SessionTrace(
        "anon_browser_outbox",
        created_at="2026-07-12T00:00:00Z",
    )
    question = {"question_id": "q_browser_outbox", "type": "mixed"}
    trace.record_answer(
        question,
        selected_letter="B",
        selected_candidate_id="c_browser_b",
        event_id="evt_browser_answer",
        occurred_at="2026-07-12T00:00:01Z",
        extra={"attempt_id": "attempt_browser", "is_correct": True},
    )
    trace.record_comment(
        question,
        category="other",
        text="browser persisted comment",
        event_id="evt_browser_comment",
        occurred_at="2026-07-12T00:00:02Z",
        extra={"attempt_id": "attempt_browser"},
    )
    return trace


def _presentation_trace() -> tuple[
    feedback.SessionTrace,
    tuple[dict[str, Any], dict[str, Any]],
]:
    trace = feedback.SessionTrace(
        "anon_browser_presented",
        created_at="2026-07-12T01:00:00Z",
    )
    question = {"question_id": "q_browser_presented", "type": "mixed"}
    acknowledged = trace.record_question_presented(
        question,
        attempt_id="attempt_browser_presented",
        release_id="release_browser_presented",
        decision_id="decision_browser_exploit",
        policy_version="surprise_policy_v1",
        mode="exploit",
        propensity=0.625,
        source="next",
        position=3,
        event_id="evt_browser_presented_ack",
        occurred_at="2026-07-12T01:00:01Z",
    )
    quarantined = trace.record_question_presented(
        question,
        attempt_id="attempt_browser_presented",
        release_id="release_browser_presented",
        decision_id="decision_browser_picker",
        policy_version="manual_policy_v1",
        mode="manual",
        propensity=1.0,
        source="picker",
        position=4,
        event_id="evt_browser_presented_quarantine",
        occurred_at="2026-07-12T01:00:02Z",
    )
    return trace, (acknowledged, quarantined)


def _serialized() -> str:
    trace = _trace()
    return browser_outbox.serialize_browser_outbox(
        generation=7,
        current_attempt_id="attempt_browser",
        trace=trace,
        acknowledged_event_ids=["evt_browser_answer"],
        quarantined_events={
            "evt_browser_comment": {
                "event_id": "evt_browser_comment",
                "request_id": None,
                "error_code": "LOCAL_BODY_LIMIT",
            }
        },
    )


def _canonical_checksum(document: dict[str, Any]) -> str:
    core = {key: value for key, value in document.items() if key != "checksum"}
    encoded = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_empty_trace_roundtrip_preserves_session_and_attempt() -> None:
    trace = feedback.SessionTrace(
        "anon_empty_browser",
        created_at="2026-07-12T00:00:00Z",
    )

    raw = browser_outbox.serialize_browser_outbox(
        generation=0,
        current_attempt_id="attempt_empty_browser",
        trace=trace,
    )
    restored = browser_outbox.parse_browser_outbox(raw)

    assert restored.trace.session_id == trace.session_id
    assert restored.trace.created_at == trace.created_at
    assert restored.trace.events == ()
    assert restored.current_attempt_id == "attempt_empty_browser"
    assert restored.generation == 0


def test_roundtrip_preserves_events_acknowledgements_and_quarantine() -> None:
    restored = browser_outbox.parse_browser_outbox(_serialized())

    assert [event["event_id"] for event in restored.trace.events] == [
        "evt_browser_answer",
        "evt_browser_comment",
    ]
    assert [event["sequence"] for event in restored.trace.events] == [1, 2]
    assert restored.acknowledged_event_ids == ("evt_browser_answer",)
    assert restored.quarantined_by_id() == {
        "evt_browser_comment": {
            "event_id": "evt_browser_comment",
            "request_id": None,
            "error_code": "LOCAL_BODY_LIMIT",
        }
    }
    assert restored.encoded_size == len(_serialized().encode("utf-8"))


def test_roundtrip_preserves_question_reaction_for_ui_recovery() -> None:
    trace = _trace()
    question = {"question_id": "q_browser_outbox", "type": "mixed"}
    reaction = trace.record_question_reaction(
        question,
        value=True,
        attempt_id="attempt_browser",
        release_id="release_browser",
        occurred_at="2026-07-12T00:00:03Z",
    )

    raw = browser_outbox.serialize_browser_outbox(
        generation=8,
        current_attempt_id="attempt_browser",
        trace=trace,
    )
    restored = browser_outbox.parse_browser_outbox(raw)

    assert restored.trace.events[-1] == reaction
    assert inspector_app.feedback.summarize_session_events(restored.trace.events)[
        "reactions"
    ] == {"total": 1, "surprised": 1, "not_surprised": 0}


def test_question_presentations_survive_refresh_ack_and_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, presentations = _presentation_trace()
    raw = browser_outbox.serialize_browser_outbox(
        generation=9,
        current_attempt_id="attempt_browser_presented",
        trace=trace,
        acknowledged_event_ids=[presentations[0]["event_id"]],
        quarantined_events={
            presentations[1]["event_id"]: {
                "event_id": presentations[1]["event_id"],
                "request_id": "request_browser_presented",
                "error_code": "EVENT_ID_CONFLICT",
            }
        },
    )

    restored = browser_outbox.parse_browser_outbox(raw)
    assert restored.trace.events == presentations
    assert restored.acknowledged_event_ids == ("evt_browser_presented_ack",)
    assert restored.quarantined_by_id() == {
        "evt_browser_presented_quarantine": {
            "event_id": "evt_browser_presented_quarantine",
            "request_id": "request_browser_presented",
            "error_code": "EVENT_ID_CONFLICT",
        }
    }
    assert [event["payload"] for event in restored.trace.events] == [
        {
            "attempt_id": "attempt_browser_presented",
            "release_id": "release_browser_presented",
            "decision_id": "decision_browser_exploit",
            "policy_version": "surprise_policy_v1",
            "mode": "exploit",
            "propensity": 0.625,
            "source": "next",
            "position": 3,
        },
        {
            "attempt_id": "attempt_browser_presented",
            "release_id": "release_browser_presented",
            "decision_id": "decision_browser_picker",
            "policy_version": "manual_policy_v1",
            "mode": "manual",
            "propensity": 1.0,
            "source": "picker",
            "position": 4,
        },
    ]

    state = _SessionState(
        feedback_trace=feedback.SessionTrace.new(),
        quiz_attempt_id="attempt_new",
        feedback_uploaded_event_ids=[],
        feedback_quarantined_events={},
        feedback_browser_outbox_generation=0,
        feedback_browser_outbox_saved_checksum=None,
        quiz_answers={},
        quiz_results={},
    )
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    inspector_app._apply_browser_outbox_snapshot(restored)

    assert state.feedback_trace.events == presentations
    assert state.feedback_uploaded_event_ids == ["evt_browser_presented_ack"]
    assert set(state.feedback_quarantined_events) == {
        "evt_browser_presented_quarantine"
    }
    assert [
        {
            field: event["payload"][field]
            for field in (
                "policy_version",
                "decision_id",
                "mode",
                "source",
                "position",
                "propensity",
            )
        }
        for event in state.feedback_trace.events
    ] == [
        {
            "policy_version": "surprise_policy_v1",
            "decision_id": "decision_browser_exploit",
            "mode": "exploit",
            "source": "next",
            "position": 3,
            "propensity": 0.625,
        },
        {
            "policy_version": "manual_policy_v1",
            "decision_id": "decision_browser_picker",
            "mode": "manual",
            "source": "picker",
            "position": 4,
            "propensity": 1.0,
        },
    ]


def test_snapshot_never_contains_endpoint_or_token_fields() -> None:
    document = json.loads(_serialized())

    assert "endpoint" not in document
    assert "token" not in document
    assert "bearer" not in _serialized().lower()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("generation", 8),
        lambda value: value.__setitem__("unsupported", True),
        lambda value: value.__setitem__("checksum", "0" * 64),
    ],
)
def test_tampering_or_unknown_fields_fail_closed(mutation: Any) -> None:
    document = json.loads(_serialized())
    mutation(document)

    with pytest.raises(browser_outbox.BrowserOutboxError):
        browser_outbox.parse_browser_outbox(json.dumps(document))


def test_duplicate_json_keys_and_oversized_bytes_fail_closed() -> None:
    raw = _serialized()
    duplicate = raw.replace(
        "{",
        '{"schema_version":"1.0",',
        1,
    )

    with pytest.raises(browser_outbox.BrowserOutboxError, match="duplicate"):
        browser_outbox.parse_browser_outbox(duplicate)
    with pytest.raises(browser_outbox.BrowserOutboxError, match="limit"):
        browser_outbox.parse_browser_outbox(raw, max_bytes=10)


@pytest.mark.parametrize("relationship", ["unknown", "overlap"])
def test_status_relationships_fail_after_valid_checksum(relationship: str) -> None:
    document = json.loads(_serialized())
    if relationship == "unknown":
        document["acknowledged_event_ids"] = ["evt_not_in_trace"]
    else:
        document["acknowledged_event_ids"] = ["evt_browser_comment"]
    document["checksum"] = _canonical_checksum(document)

    with pytest.raises(browser_outbox.BrowserOutboxError):
        browser_outbox.parse_browser_outbox(
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        )


def test_noncontiguous_event_sequence_fails_after_valid_checksum() -> None:
    document = json.loads(_serialized())
    document["trace"]["events"][1]["sequence"] = 4
    document["checksum"] = _canonical_checksum(document)

    with pytest.raises(browser_outbox.BrowserOutboxError, match="contiguous"):
        browser_outbox.parse_browser_outbox(
            json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        )


def test_app_restore_reconstructs_current_attempt_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = browser_outbox.parse_browser_outbox(_serialized())
    state = _SessionState(
        feedback_trace=feedback.SessionTrace.new(),
        quiz_attempt_id="attempt_new",
        feedback_uploaded_event_ids=[],
        feedback_quarantined_events={},
        feedback_browser_outbox_generation=0,
        feedback_browser_outbox_saved_checksum=None,
        quiz_answers={},
        quiz_results={},
    )
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    inspector_app._apply_browser_outbox_snapshot(snapshot)

    assert state.feedback_trace.session_id == "anon_browser_outbox"
    assert state.quiz_attempt_id == "attempt_browser"
    assert state.feedback_uploaded_event_ids == ["evt_browser_answer"]
    assert state.quiz_results
    assert list(state.quiz_results.values()) == [True]
    assert list(state.quiz_answers.values())[0]["selected_letter"] == "B"


def test_start_new_session_clears_only_live_outbox_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_trace = _trace()
    state = _SessionState(
        feedback_trace=old_trace,
        feedback_uploaded_event_ids=["evt_browser_answer"],
        feedback_quarantined_events={"evt_browser_comment": {}},
        comment_notices={"q": {}},
        feedback_upload_notice="old",
        feedback_recovery_outboxes={"saved-recovery": {"keep": True}},
        quiz_results={"q": True},
        quiz_answers={"q": {}},
        quiz_attempt_id="attempt_browser",
        feedback_browser_outbox_generation=7,
        feedback_browser_outbox_invalid_raw="broken",
        feedback_browser_outbox_write_blocked=True,
        feedback_browser_outbox_notice=None,
        committed_letter="B",
        focus_letter="B",
        info_letter=None,
        inspect_file="candidate_spec.json",
        setting_notice=None,
    )
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    inspector_app._start_new_feedback_session()

    assert state.feedback_trace.session_id != old_trace.session_id
    assert state.feedback_trace.events == ()
    assert state.feedback_uploaded_event_ids == []
    assert state.feedback_quarantined_events == {}
    assert state.quiz_attempt_id != "attempt_browser"
    assert state.quiz_results == {}
    assert state.quiz_answers == {}
    assert state.feedback_browser_outbox_write_blocked is False
    assert state.feedback_recovery_outboxes == {"saved-recovery": {"keep": True}}


def test_component_uses_indexeddb_and_streamlit_protocol() -> None:
    html = (INSPECTOR / "browser_outbox_component" / "index.html").read_text(
        encoding="utf-8"
    )

    for marker in (
        "indexedDB.open",
        "streamlit:componentReady",
        "streamlit:render",
        "streamlit:setComponentValue",
        "streamlit:setFrameHeight",
        "isStreamlitMessage: true",
    ):
        assert marker in html
    assert "localStorage" not in html
