"""Tests for pending-only feedback batching and conflict isolation."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
INSPECTOR = REPO / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import feedback  # noqa: E402
import feedback_outbox as outbox  # noqa: E402


def _question() -> dict[str, Any]:
    return {"question_id": "q_outbox", "type": "mixed"}


def _events(count: int, *, text_size: int = 8) -> tuple[dict[str, Any], ...]:
    trace = feedback.SessionTrace(
        "anon_outbox",
        created_at="2026-07-12T00:00:00Z",
    )
    for index in range(count):
        trace.record_comment(
            _question(),
            category="other",
            text=f"{index:04d}-" + ("x" * text_size),
            event_id=f"evt_outbox_{index:04d}",
            occurred_at="2026-07-12T00:00:00Z",
        )
    return trace.events


def _events_with_one_oversized(
    oversized_index: int,
) -> tuple[dict[str, Any], ...]:
    events = list(_events(3))
    events[oversized_index]["payload"]["text"] = "x" * 1_800
    return tuple(events)


def _request_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


def _receipt(
    *, accepted: int, duplicate: int = 0, index: int = 1
) -> feedback.UploadReceipt:
    request_id = _request_id(index)
    return feedback.UploadReceipt(
        status_code=200,
        endpoint="https://ingest.example/feedback-ingest",
        request_id=request_id,
        response={
            "accepted": accepted,
            "duplicate": duplicate,
            "conflict": 0,
            "rejected": 0,
            "request_id": request_id,
        },
    )


def _conflict(*, index: int) -> feedback.FeedbackUploadConflictError:
    request_id = _request_id(index)
    return feedback.FeedbackUploadConflictError(
        "feedback event ID conflicts with stored content",
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


class _SuccessClient:
    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.request_index = 1

    def post_trace(self, trace: dict[str, Any]) -> feedback.UploadReceipt:
        self.traces.append(trace)
        receipt = _receipt(
            accepted=len(trace["events"]),
            index=self.request_index,
        )
        self.request_index += 1
        return receipt

    def post_event(self, event: dict[str, Any]) -> feedback.UploadReceipt:
        self.events.append(event)
        receipt = _receipt(accepted=1, index=self.request_index)
        self.request_index += 1
        return receipt


def test_receiver_limit_constants_match_edge_contract() -> None:
    source = (
        REPO / "supabase" / "functions" / "feedback-ingest" / "index.ts"
    ).read_text(encoding="utf-8")
    events = re.search(r"const MAX_EVENTS_PER_REQUEST = (\d+);", source)
    body = re.search(r"const MAX_REQUEST_BYTES = ([\d_]+);", source)
    sequence = re.search(r"const MAX_EVENT_SEQUENCE = ([\d_]+);", source)

    assert events is not None
    assert body is not None
    assert sequence is not None
    assert outbox.MAX_EVENTS_PER_BATCH == int(events.group(1))
    assert outbox.MAX_BODY_BYTES == int(body.group(1).replace("_", ""))
    assert feedback.MAX_EVENT_SEQUENCE == int(sequence.group(1).replace("_", ""))


def test_pending_events_excludes_acknowledged_and_quarantined_ids() -> None:
    events = _events(4)
    selected = outbox.pending_events(
        events,
        acknowledged_event_ids={events[0]["event_id"]},
        quarantined_event_ids={events[2]["event_id"]},
    )

    assert [event["event_id"] for event in selected] == [
        events[1]["event_id"],
        events[3]["event_id"],
    ]
    assert [event["sequence"] for event in selected] == [2, 4]

    with pytest.raises(outbox.FeedbackOutboxError, match="both acknowledged"):
        outbox.pending_events(
            events,
            acknowledged_event_ids={events[0]["event_id"]},
            quarantined_event_ids={events[0]["event_id"]},
        )


def test_batches_enforce_event_count_and_preserve_global_sequence() -> None:
    batches = outbox.build_upload_batches(_events(501))

    assert [len(batch.event_ids) for batch in batches] == [500, 1]
    assert batches[0].envelope["events"][0]["sequence"] == 1
    assert batches[0].envelope["events"][-1]["sequence"] == 500
    assert batches[1].envelope["events"][0]["sequence"] == 501
    assert all(batch.encoded_size <= outbox.MAX_BODY_BYTES for batch in batches)


def test_batches_split_on_exact_encoded_bytes_and_reject_oversized_single_event() -> (
    None
):
    events = _events(3, text_size=1_800)
    batches = outbox.build_upload_batches(events, max_body_bytes=2_500)

    assert len(batches) == 3
    assert [event_id for batch in batches for event_id in batch.event_ids] == [
        event["event_id"] for event in events
    ]
    assert all(batch.encoded_size <= 2_500 for batch in batches)

    with pytest.raises(outbox.FeedbackOutboxError, match="exceeds the receiver"):
        outbox.build_upload_batches(events[:1], max_body_bytes=100)


@pytest.mark.parametrize("oversized_index", [0, 1, 2])
def test_upload_quarantines_oversized_event_and_continues_in_order(
    oversized_index: int,
) -> None:
    events = _events_with_one_oversized(oversized_index)
    oversized_id = events[oversized_index]["event_id"]
    expected_acknowledged = tuple(
        event["event_id"]
        for index, event in enumerate(events)
        if index != oversized_index
    )
    client = _SuccessClient()

    result = outbox.upload_pending_events(
        client,  # type: ignore[arg-type]
        events,
        max_body_bytes=1_000,
    )

    assert result.pending_event_ids == tuple(event["event_id"] for event in events)
    assert result.acknowledged_event_ids == expected_acknowledged
    assert [item.to_dict() for item in result.quarantined_events] == [
        {
            "event_id": oversized_id,
            "request_id": None,
            "error_code": outbox.LOCAL_BODY_LIMIT_ERROR_CODE,
        }
    ]
    assert result.remaining_event_ids == ()
    assert result.issue is None
    assert result.settled is True
    assert result.all_acknowledged is False
    assert result.batch_request_count == len(client.traces) > 0
    assert result.isolation_request_count == 0
    uploaded_ids = [
        event["event_id"] for envelope in client.traces for event in envelope["events"]
    ]
    assert uploaded_ids == list(expected_acknowledged)
    assert oversized_id not in uploaded_ids
    assert client.events == []


def test_only_oversized_event_is_locally_settled_without_network() -> None:
    event = _events_with_one_oversized(0)[0]
    client = _SuccessClient()

    result = outbox.upload_pending_events(
        client,  # type: ignore[arg-type]
        (event,),
        max_body_bytes=1_000,
    )

    assert result.pending_event_ids == (event["event_id"],)
    assert result.acknowledged_event_ids == ()
    assert [item.to_dict() for item in result.quarantined_events] == [
        {
            "event_id": event["event_id"],
            "request_id": None,
            "error_code": outbox.LOCAL_BODY_LIMIT_ERROR_CODE,
        }
    ]
    assert result.remaining_event_ids == ()
    assert result.request_ids == ()
    assert result.batch_request_count == 0
    assert result.isolation_request_count == 0
    assert result.issue is None
    assert result.settled is True
    assert result.all_acknowledged is False
    assert client.traces == []
    assert client.events == []


def test_successful_chunks_acknowledge_only_pending_events() -> None:
    events = _events(5)
    client = _SuccessClient()
    result = outbox.upload_pending_events(
        client,  # type: ignore[arg-type]
        events,
        acknowledged_event_ids={events[0]["event_id"]},
        max_events=2,
    )

    assert result.pending_event_ids == tuple(event["event_id"] for event in events[1:])
    assert result.acknowledged_event_ids == result.pending_event_ids
    assert result.quarantined_events == ()
    assert result.remaining_event_ids == ()
    assert result.settled is True
    assert result.all_acknowledged is True
    assert result.batch_request_count == 2
    assert result.isolation_request_count == 0
    assert [len(trace["events"]) for trace in client.traces] == [2, 2]


def test_conflicting_batch_isolates_bad_id_and_uploads_the_rest() -> None:
    events = _events(3)
    conflict_id = events[1]["event_id"]

    class ConflictClient(_SuccessClient):
        def post_trace(self, trace: dict[str, Any]) -> feedback.UploadReceipt:
            self.traces.append(trace)
            raise _conflict(index=10)

        def post_event(self, event: dict[str, Any]) -> feedback.UploadReceipt:
            self.events.append(event)
            if event["event_id"] == conflict_id:
                raise _conflict(index=20)
            return _receipt(accepted=1, index=30 + len(self.events))

    client = ConflictClient()
    result = outbox.upload_pending_events(client, events)  # type: ignore[arg-type]

    assert result.acknowledged_event_ids == (
        events[0]["event_id"],
        events[2]["event_id"],
    )
    assert [item.event_id for item in result.quarantined_events] == [conflict_id]
    assert result.quarantined_events[0].request_id == _request_id(20)
    assert result.remaining_event_ids == ()
    assert result.settled is True
    assert result.all_acknowledged is False
    assert result.batch_request_count == 1
    assert result.isolation_request_count == 3
    assert [event["event_id"] for event in client.events] == [
        event["event_id"] for event in events
    ]
    serialized = result.to_dict()
    assert "0001-" not in str(serialized)
    assert "payload" not in str(serialized)


def test_invalid_receipt_leaves_every_event_pending() -> None:
    events = _events(2)

    class InvalidReceiptClient(_SuccessClient):
        def post_trace(self, trace: dict[str, Any]) -> feedback.UploadReceipt:
            self.traces.append(trace)
            return feedback.UploadReceipt(
                status_code=200,
                endpoint="https://ingest.example/feedback-ingest",
                response={"accepted": len(trace["events"])},
            )

    result = outbox.upload_pending_events(  # type: ignore[arg-type]
        InvalidReceiptClient(),
        events,
    )

    assert result.acknowledged_event_ids == ()
    assert result.remaining_event_ids == tuple(event["event_id"] for event in events)
    assert result.issue is not None
    assert result.issue.kind == "invalid_receipt"
    assert result.issue.retryable is False


def test_network_failure_after_partial_isolation_preserves_safe_progress() -> None:
    events = _events(3)

    class InterruptedIsolationClient(_SuccessClient):
        def post_trace(self, trace: dict[str, Any]) -> feedback.UploadReceipt:
            self.traces.append(trace)
            raise _conflict(index=40)

        def post_event(self, event: dict[str, Any]) -> feedback.UploadReceipt:
            self.events.append(event)
            if len(self.events) == 1:
                return _receipt(accepted=1, index=41)
            raise feedback.FeedbackUploadError(
                "temporary network failure",
                endpoint="https://ingest.example/feedback-ingest",
            )

    result = outbox.upload_pending_events(  # type: ignore[arg-type]
        InterruptedIsolationClient(),
        events,
    )

    assert result.acknowledged_event_ids == (events[0]["event_id"],)
    assert result.remaining_event_ids == (
        events[1]["event_id"],
        events[2]["event_id"],
    )
    assert result.issue is not None
    assert result.issue.kind == "isolation_upload_failed"
    assert result.issue.retryable is True


def test_configuration_failure_is_non_retryable_and_no_event_is_acknowledged() -> None:
    events = _events(1)
    client = feedback.FeedbackClient(
        endpoint="https://ingest.example/feedback-ingest",
        environ={},
    )

    result = outbox.upload_pending_events(client, events)

    assert result.acknowledged_event_ids == ()
    assert result.remaining_event_ids == (events[0]["event_id"],)
    assert result.issue is not None
    assert result.issue.kind == "batch_upload_failed"
    assert result.issue.retryable is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_events": 0}, "max_events"),
        ({"max_events": True}, "max_events"),
        ({"max_body_bytes": 0}, "max_body_bytes"),
        ({"max_body_bytes": False}, "max_body_bytes"),
    ],
)
def test_batch_limits_fail_closed(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(outbox.FeedbackOutboxError, match=message):
        outbox.build_upload_batches(_events(1), **kwargs)
