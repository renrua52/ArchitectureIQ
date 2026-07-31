"""Strict recovery upload for previously downloaded session traces.

The browser can lose Streamlit's in-memory session state after a refresh.  A
downloaded session trace is therefore a small durable client-side outbox.  This
module validates that file before any network call, identifies its *complete*
canonical content, and routes its events through the same bounded/conflict-
isolating uploader used by the live session.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:  # Package import in tests/tools; top-level import in the Streamlit app.
    from . import feedback, feedback_outbox
except ImportError:  # pragma: no cover - exercised by the app's import style.
    import feedback
    import feedback_outbox


MAX_RECOVERY_FILE_BYTES = 10 * 1024 * 1024


class FeedbackRecoveryError(ValueError):
    """A saved trace cannot be admitted to the recovery outbox."""


@dataclass(frozen=True)
class RecoveredTrace:
    """One validated saved trace and its canonical full-content identity."""

    recovery_id: str
    trace_id: str
    session_id: str
    created_at: str
    events: tuple[dict[str, Any], ...]
    canonical_size: int

    @property
    def event_count(self) -> int:
        return len(self.events)


def _canonical_bytes(envelope: dict[str, Any]) -> bytes:
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _recovery_id(canonical: bytes) -> str:
    return f"recovery_sha256_{hashlib.sha256(canonical).hexdigest()}"


def _raw_bytes(raw: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(raw, str):
        try:
            return raw.encode("utf-8")
        except UnicodeError as exc:
            raise FeedbackRecoveryError(
                "saved session trace must be UTF-8 JSON"
            ) from exc
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    raise FeedbackRecoveryError("saved session trace must be uploaded as JSON bytes")


def parse_recovered_trace(
    raw: bytes | bytearray | memoryview | str,
    *,
    max_file_bytes: int = MAX_RECOVERY_FILE_BYTES,
) -> RecoveredTrace:
    """Validate a saved trace locally and return a detached recovery record."""

    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes < 1
    ):
        raise FeedbackRecoveryError("max_file_bytes must be a positive integer")
    encoded = _raw_bytes(raw)
    if len(encoded) > max_file_bytes:
        raise FeedbackRecoveryError(
            "saved session trace exceeds the recovery file limit "
            f"({len(encoded)} > {max_file_bytes} bytes)"
        )
    try:
        envelope = feedback.parse_session_trace_json(encoded)
    except feedback.FeedbackValidationError as exc:
        raise FeedbackRecoveryError(f"invalid saved session trace: {exc}") from exc

    canonical = _canonical_bytes(envelope)
    return RecoveredTrace(
        recovery_id=_recovery_id(canonical),
        trace_id=str(envelope["trace_id"]),
        session_id=str(envelope["session_id"]),
        created_at=str(envelope["created_at"]),
        events=tuple(envelope["events"]),
        canonical_size=len(canonical),
    )


def upload_recovered_trace(
    client: feedback.FeedbackClient,
    recovered: RecoveredTrace,
    *,
    acknowledged_event_ids: Iterable[str] = (),
    quarantined_event_ids: Iterable[str] = (),
    max_events: int = feedback_outbox.MAX_EVENTS_PER_BATCH,
    max_body_bytes: int = feedback_outbox.MAX_BODY_BYTES,
) -> feedback_outbox.OutboxUploadResult:
    """Upload one validated recovery outbox using normal receiver limits."""

    if not isinstance(recovered, RecoveredTrace):
        raise FeedbackRecoveryError("recovered must be a validated RecoveredTrace")
    envelope = {
        "schema_version": feedback.TRACE_SCHEMA_VERSION,
        "envelope_type": "session_trace",
        "trace_id": recovered.trace_id,
        "session_id": recovered.session_id,
        "created_at": recovered.created_at,
        "event_count": recovered.event_count,
        "events": list(recovered.events),
    }
    try:
        validated = feedback.validate_session_trace_envelope(envelope)
    except feedback.FeedbackValidationError as exc:
        raise FeedbackRecoveryError(
            f"recovered trace changed after local validation: {exc}"
        ) from exc
    canonical = _canonical_bytes(validated)
    if (
        _recovery_id(canonical) != recovered.recovery_id
        or len(canonical) != recovered.canonical_size
    ):
        raise FeedbackRecoveryError(
            "recovered trace changed after local validation; parse the file again"
        )
    return feedback_outbox.upload_pending_events(
        client,
        validated["events"],
        acknowledged_event_ids=acknowledged_event_ids,
        quarantined_event_ids=quarantined_event_ids,
        max_events=max_events,
        max_body_bytes=max_body_bytes,
    )
