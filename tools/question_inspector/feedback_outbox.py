"""Pure pending-upload batching and conflict isolation for feedback events.

The Streamlit layer owns session state and presentation.  This module owns the
reliable upload mechanics: acknowledged and quarantined IDs are excluded,
requests stay within the receiver limits, receipts are verified fail-closed,
and a conflicting batch is retried event-by-event so one poisoned ID cannot
block later answers or comments.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # Package import in tests/tools; top-level import in the Streamlit app.
    from . import feedback
except ImportError:  # pragma: no cover - exercised by the app's import style.
    import feedback


MAX_EVENTS_PER_BATCH = 500
MAX_BODY_BYTES = 1024 * 1024
LOCAL_BODY_LIMIT_ERROR_CODE = "LOCAL_BODY_LIMIT"


class FeedbackOutboxError(ValueError):
    """The local outbox cannot safely construct an upload request."""


@dataclass(frozen=True)
class UploadBatch:
    """One validated receiver-sized trace envelope."""

    event_ids: tuple[str, ...]
    envelope: Mapping[str, Any]
    encoded_size: int


@dataclass(frozen=True)
class _OversizedEvent:
    """One locally measured event that cannot fit a receiver request."""

    event_id: str
    encoded_size: int


@dataclass(frozen=True)
class _UploadPlan:
    """Receiver-sized batches plus events that cannot be sent individually."""

    batches: tuple[UploadBatch, ...]
    oversized_events: tuple[_OversizedEvent, ...]


@dataclass(frozen=True)
class QuarantinedEvent:
    """One event isolated by a local limit or a server content conflict."""

    event_id: str
    request_id: str | None
    error_code: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class UploadIssue:
    """Safe upload failure metadata; never contains an event payload."""

    kind: str
    message: str
    request_id: str | None
    retryable: bool

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "kind": self.kind,
            "message": self.message,
            "request_id": self.request_id,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class OutboxUploadResult:
    """Detached state transition returned after an upload attempt."""

    pending_event_ids: tuple[str, ...]
    acknowledged_event_ids: tuple[str, ...]
    quarantined_events: tuple[QuarantinedEvent, ...]
    remaining_event_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    batch_request_count: int
    isolation_request_count: int
    issue: UploadIssue | None = None

    @property
    def settled(self) -> bool:
        """Whether every pending ID was either acknowledged or quarantined."""
        return not self.remaining_event_ids and self.issue is None

    @property
    def all_acknowledged(self) -> bool:
        """Whether every pending event has positive storage acknowledgement."""
        return self.settled and not self.quarantined_events

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending_event_ids": list(self.pending_event_ids),
            "acknowledged_event_ids": list(self.acknowledged_event_ids),
            "quarantined_events": [item.to_dict() for item in self.quarantined_events],
            "remaining_event_ids": list(self.remaining_event_ids),
            "request_ids": list(self.request_ids),
            "batch_request_count": self.batch_request_count,
            "isolation_request_count": self.isolation_request_count,
            "settled": self.settled,
            "all_acknowledged": self.all_acknowledged,
            "issue": self.issue.to_dict() if self.issue is not None else None,
        }


def _identifier_set(values: Iterable[str], *, field: str) -> frozenset[str]:
    resolved: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise FeedbackOutboxError(f"{field} must contain non-empty strings")
        resolved.add(value)
    return frozenset(resolved)


def pending_events(
    events: Sequence[Mapping[str, Any]],
    *,
    acknowledged_event_ids: Iterable[str] = (),
    quarantined_event_ids: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Validate a trace and return only uploadable, unacknowledged events."""

    acknowledged = _identifier_set(
        acknowledged_event_ids,
        field="acknowledged_event_ids",
    )
    quarantined = _identifier_set(
        quarantined_event_ids,
        field="quarantined_event_ids",
    )
    overlap = acknowledged & quarantined
    if overlap:
        raise FeedbackOutboxError(
            "an event cannot be both acknowledged and quarantined: "
            f"{', '.join(sorted(overlap))}"
        )
    if not events:
        return ()
    first = events[0]
    session_id = first.get("session_id") if isinstance(first, Mapping) else None
    if not isinstance(session_id, str) or not session_id:
        raise FeedbackOutboxError("events must begin with a valid session_id")
    try:
        envelope = feedback.build_session_trace_envelope(
            session_id,
            events,
            created_at=str(first.get("occurred_at") or ""),
        )
    except feedback.FeedbackValidationError as exc:
        raise FeedbackOutboxError(f"invalid feedback trace: {exc}") from exc
    return tuple(
        event
        for event in envelope["events"]
        if event["event_id"] not in acknowledged
        and event["event_id"] not in quarantined
    )


def _encoded_envelope(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    if not events:
        raise FeedbackOutboxError("an upload batch cannot be empty")
    first = events[0]
    envelope = feedback.build_session_trace_envelope(
        str(first["session_id"]),
        events,
        created_at=str(first["occurred_at"]),
    )
    try:
        encoded = json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # Defensive after schema validation.
        raise FeedbackOutboxError(f"upload envelope is not strict JSON: {exc}") from exc
    return envelope, len(encoded)


def _split_to_size(
    events: Sequence[Mapping[str, Any]],
    *,
    max_body_bytes: int,
) -> _UploadPlan:
    envelope, encoded_size = _encoded_envelope(events)
    if encoded_size <= max_body_bytes:
        return _UploadPlan(
            batches=(
                UploadBatch(
                    event_ids=tuple(str(event["event_id"]) for event in events),
                    envelope=envelope,
                    encoded_size=encoded_size,
                ),
            ),
            oversized_events=(),
        )
    if len(events) == 1:
        return _UploadPlan(
            batches=(),
            oversized_events=(
                _OversizedEvent(
                    event_id=str(events[0]["event_id"]),
                    encoded_size=encoded_size,
                ),
            ),
        )
    midpoint = len(events) // 2
    left = _split_to_size(events[:midpoint], max_body_bytes=max_body_bytes)
    right = _split_to_size(events[midpoint:], max_body_bytes=max_body_bytes)
    return _UploadPlan(
        batches=(*left.batches, *right.batches),
        oversized_events=(*left.oversized_events, *right.oversized_events),
    )


def _validate_batch_limits(*, max_events: int, max_body_bytes: int) -> None:
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, int)
        or max_events < 1
    ):
        raise FeedbackOutboxError("max_events must be a positive integer")
    if (
        isinstance(max_body_bytes, bool)
        or not isinstance(max_body_bytes, int)
        or max_body_bytes < 1
    ):
        raise FeedbackOutboxError("max_body_bytes must be a positive integer")


def _plan_upload_batches(
    events: Sequence[Mapping[str, Any]],
    *,
    max_events: int,
    max_body_bytes: int,
) -> _UploadPlan:
    """Plan every event without allowing one oversized item to block others."""

    _validate_batch_limits(
        max_events=max_events,
        max_body_bytes=max_body_bytes,
    )
    batches: list[UploadBatch] = []
    oversized: list[_OversizedEvent] = []
    for offset in range(0, len(events), max_events):
        plan = _split_to_size(
            events[offset : offset + max_events],
            max_body_bytes=max_body_bytes,
        )
        batches.extend(plan.batches)
        oversized.extend(plan.oversized_events)
    return _UploadPlan(
        batches=tuple(batches),
        oversized_events=tuple(oversized),
    )


def build_upload_batches(
    events: Sequence[Mapping[str, Any]],
    *,
    acknowledged_event_ids: Iterable[str] = (),
    quarantined_event_ids: Iterable[str] = (),
    max_events: int = MAX_EVENTS_PER_BATCH,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> tuple[UploadBatch, ...]:
    """Return ordered batches within the receiver count and byte limits."""

    selected = pending_events(
        events,
        acknowledged_event_ids=acknowledged_event_ids,
        quarantined_event_ids=quarantined_event_ids,
    )
    plan = _plan_upload_batches(
        selected,
        max_events=max_events,
        max_body_bytes=max_body_bytes,
    )
    if plan.oversized_events:
        oversized = plan.oversized_events[0]
        raise FeedbackOutboxError(
            f"event {oversized.event_id!r} exceeds the receiver body limit "
            f"({oversized.encoded_size} > {max_body_bytes} bytes)"
        )
    return plan.batches


def _retryable_error(exc: feedback.FeedbackError) -> bool:
    if isinstance(
        exc,
        (feedback.FeedbackConfigurationError, feedback.FeedbackValidationError),
    ):
        return False
    status = getattr(exc, "status_code", None)
    return (
        status is None
        or status in {408, 425, 429}
        or (isinstance(status, int) and status >= 500)
    )


def _issue(
    kind: str,
    message: str,
    *,
    request_id: str | None = None,
    retryable: bool,
) -> UploadIssue:
    return UploadIssue(
        kind=kind,
        message=message,
        request_id=request_id,
        retryable=retryable,
    )


def upload_pending_events(
    client: feedback.FeedbackClient,
    events: Sequence[Mapping[str, Any]],
    *,
    acknowledged_event_ids: Iterable[str] = (),
    quarantined_event_ids: Iterable[str] = (),
    max_events: int = MAX_EVENTS_PER_BATCH,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> OutboxUploadResult:
    """Upload pending chunks, isolating content conflicts event-by-event."""

    initial_pending = pending_events(
        events,
        acknowledged_event_ids=acknowledged_event_ids,
        quarantined_event_ids=quarantined_event_ids,
    )
    pending_ids = tuple(str(event["event_id"]) for event in initial_pending)
    plan = _plan_upload_batches(
        initial_pending,
        max_events=max_events,
        max_body_bytes=max_body_bytes,
    )
    acknowledged: list[str] = []
    quarantined: list[QuarantinedEvent] = [
        QuarantinedEvent(
            event_id=item.event_id,
            request_id=None,
            error_code=LOCAL_BODY_LIMIT_ERROR_CODE,
        )
        for item in plan.oversized_events
    ]
    request_ids: list[str] = []
    batch_requests = 0
    isolation_requests = 0
    issue: UploadIssue | None = None

    def remember_request_id(value: str | None) -> None:
        if isinstance(value, str) and value and value not in request_ids:
            request_ids.append(value)

    for batch in plan.batches:
        batch_requests += 1
        try:
            receipt = client.post_trace(batch.envelope)
        except feedback.FeedbackUploadConflictError as exc:
            remember_request_id(exc.request_id)
            batch_events = tuple(batch.envelope["events"])
            for event in batch_events:
                isolation_requests += 1
                event_id = str(event["event_id"])
                try:
                    event_receipt = client.post_event(event)
                except feedback.FeedbackUploadConflictError as event_exc:
                    remember_request_id(event_exc.request_id)
                    quarantined.append(
                        QuarantinedEvent(
                            event_id=event_id,
                            request_id=event_exc.request_id,
                            error_code=event_exc.error_code or "EVENT_ID_CONFLICT",
                        )
                    )
                    continue
                except feedback.FeedbackError as event_exc:
                    remember_request_id(getattr(event_exc, "request_id", None))
                    issue = _issue(
                        "isolation_upload_failed",
                        "Conflict isolation stopped because an individual event "
                        "could not be confirmed.",
                        request_id=getattr(event_exc, "request_id", None),
                        retryable=_retryable_error(event_exc),
                    )
                    break
                remember_request_id(event_receipt.request_id)
                if not feedback.upload_receipt_acknowledges_all(
                    event_receipt,
                    sent_count=1,
                ):
                    issue = _issue(
                        "invalid_receipt",
                        "An individual event received a partial or invalid receipt.",
                        request_id=event_receipt.request_id,
                        retryable=False,
                    )
                    break
                acknowledged.append(event_id)
            if issue is not None:
                break
            continue
        except feedback.FeedbackError as exc:
            remember_request_id(getattr(exc, "request_id", None))
            issue = _issue(
                "batch_upload_failed",
                "A pending feedback batch could not be confirmed.",
                request_id=getattr(exc, "request_id", None),
                retryable=_retryable_error(exc),
            )
            break

        remember_request_id(receipt.request_id)
        if not feedback.upload_receipt_acknowledges_all(
            receipt,
            sent_count=len(batch.event_ids),
        ):
            issue = _issue(
                "invalid_receipt",
                "A pending feedback batch received a partial or invalid receipt.",
                request_id=receipt.request_id,
                retryable=False,
            )
            break
        acknowledged.extend(batch.event_ids)

    pending_order = {event_id: index for index, event_id in enumerate(pending_ids)}
    quarantined.sort(key=lambda item: pending_order[item.event_id])
    resolved_ids = set(acknowledged) | {item.event_id for item in quarantined}
    remaining = tuple(
        event_id for event_id in pending_ids if event_id not in resolved_ids
    )
    return OutboxUploadResult(
        pending_event_ids=pending_ids,
        acknowledged_event_ids=tuple(acknowledged),
        quarantined_events=tuple(quarantined),
        remaining_event_ids=remaining,
        request_ids=tuple(request_ids),
        batch_request_count=batch_requests,
        isolation_request_count=isolation_requests,
        issue=issue,
    )
