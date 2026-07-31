"""State-transition tests for the inspector's feedback outbox integration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
INSPECTOR = REPO / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import app as inspector_app  # noqa: E402
import feedback  # noqa: E402
import feedback_outbox  # noqa: E402


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
        "anon_app_outbox",
        created_at="2026-07-12T00:00:00Z",
    )
    question = {"question_id": "q_app_outbox", "type": "mixed"}
    for index in range(3):
        trace.record_comment(
            question,
            category="other",
            text=f"comment {index}",
            event_id=f"evt_app_outbox_{index}",
            occurred_at="2026-07-12T00:00:00Z",
        )
    return trace


def _state() -> _SessionState:
    return _SessionState(
        feedback_uploaded_event_ids=[],
        feedback_quarantined_events={},
        comment_notices={},
    )


def test_pending_count_excludes_acknowledged_and_quarantined(
    monkeypatch: Any,
) -> None:
    trace = _trace()
    state = _state()
    state.feedback_uploaded_event_ids = [trace.events[0]["event_id"]]
    state.feedback_quarantined_events = {
        trace.events[1]["event_id"]: {
            "event_id": trace.events[1]["event_id"],
            "request_id": "00000000-0000-4000-8000-000000000001",
            "error_code": "EVENT_ID_CONFLICT",
        }
    }
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    assert inspector_app._pending_upload_count(trace.events) == 1


def test_apply_outbox_result_updates_ack_quarantine_and_comment_notice(
    monkeypatch: Any,
) -> None:
    trace = _trace()
    question_key = trace.events[1]["question_version"]
    state = _state()
    state.comment_notices = {
        question_key: {
            "level": "warning",
            "message": "old failure",
            "event_id": trace.events[1]["event_id"],
        }
    }
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    result = feedback_outbox.OutboxUploadResult(
        pending_event_ids=tuple(event["event_id"] for event in trace.events),
        acknowledged_event_ids=(trace.events[0]["event_id"],),
        quarantined_events=(
            feedback_outbox.QuarantinedEvent(
                event_id=trace.events[1]["event_id"],
                request_id="00000000-0000-4000-8000-000000000002",
                error_code="EVENT_ID_CONFLICT",
            ),
        ),
        remaining_event_ids=(trace.events[2]["event_id"],),
        request_ids=("00000000-0000-4000-8000-000000000002",),
        batch_request_count=1,
        isolation_request_count=2,
    )

    inspector_app._apply_outbox_result(result)

    assert state.feedback_uploaded_event_ids == [trace.events[0]["event_id"]]
    assert set(state.feedback_quarantined_events) == {trace.events[1]["event_id"]}
    notice = state.comment_notices[question_key]
    assert notice["level"] == "error"
    assert "quarantined" in notice["message"]
    assert inspector_app._pending_upload_count(trace.events) == 1


def test_strict_mark_uploaded_does_not_accept_generic_2xx(
    monkeypatch: Any,
) -> None:
    trace = _trace()
    state = _state()
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    receipt = feedback.UploadReceipt(
        status_code=200,
        endpoint="https://collector.example/upload",
        response="ok",
    )

    assert not inspector_app._mark_events_uploaded([trace.events[0]], receipt)
    assert state.feedback_uploaded_event_ids == []


def test_single_comment_conflict_is_quarantined_immediately(
    monkeypatch: Any,
) -> None:
    trace = _trace()
    state = _state()
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    request_id = "00000000-0000-4000-8000-000000000003"
    error = feedback.FeedbackUploadConflictError(
        "content conflict",
        endpoint="https://ingest.example/feedback-ingest",
        status_code=409,
        request_id=request_id,
        response={
            "accepted": 0,
            "duplicate": 0,
            "conflict": 1,
            "rejected": 1,
            "request_id": request_id,
            "error": {"code": "EVENT_ID_CONFLICT"},
        },
    )

    inspector_app._quarantine_event(trace.events[0], error)

    record = state.feedback_quarantined_events[trace.events[0]["event_id"]]
    assert record["request_id"] == request_id
    assert record["error_code"] == "EVENT_ID_CONFLICT"
    assert inspector_app._pending_upload_count(trace.events) == 2
