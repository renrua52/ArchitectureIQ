"""Strict browser-persisted state for the live feedback outbox.

The Streamlit app keeps its mutable UI state in memory, while a tiny custom
component stores this credential-free snapshot in the browser's IndexedDB.
This module owns the Python trust boundary: browser bytes are never restored
until their exact schema, checksum, trace, and acknowledgement/quarantine
relationships have all been validated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # Package import in tests/tools; top-level import in the Streamlit app.
    from . import feedback
except ImportError:  # pragma: no cover - exercised by the app's import style.
    import feedback


BROWSER_OUTBOX_SCHEMA_VERSION = "1.0"
BROWSER_OUTBOX_SNAPSHOT_TYPE = "architecture_iq_browser_feedback_outbox"
MAX_BROWSER_OUTBOX_BYTES = 10 * 1024 * 1024
CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

_CORE_KEYS = frozenset(
    {
        "schema_version",
        "snapshot_type",
        "generation",
        "current_attempt_id",
        "trace",
        "acknowledged_event_ids",
        "quarantined_events",
    }
)
_DOCUMENT_KEYS = _CORE_KEYS | {"checksum"}
_TRACE_KEYS = frozenset({"session_id", "created_at", "events"})
_QUARANTINE_KEYS = frozenset({"event_id", "request_id", "error_code"})


class BrowserOutboxError(ValueError):
    """Browser state is missing, corrupt, unsafe, or internally inconsistent."""


@dataclass(frozen=True)
class BrowserOutboxSnapshot:
    """One detached, checksum-verified live outbox snapshot."""

    generation: int
    current_attempt_id: str
    trace: feedback.SessionTrace
    acknowledged_event_ids: tuple[str, ...]
    quarantined_events: tuple[dict[str, str | None], ...]
    checksum: str
    encoded_size: int

    def quarantined_by_id(self) -> dict[str, dict[str, str | None]]:
        return {
            str(record["event_id"]): dict(record) for record in self.quarantined_events
        }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BrowserOutboxError(f"browser outbox is not strict JSON: {exc}") from exc


def _checksum(core: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise BrowserOutboxError(f"{label} keys must be strings")
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unsupported: {', '.join(unknown)}")
        raise BrowserOutboxError(
            f"{label} must use the exact schema ({'; '.join(details)})"
        )


def _identifier(value: Any, *, field_name: str) -> str:
    try:
        return feedback._validate_identifier(value, field_name=field_name)
    except feedback.FeedbackValidationError as exc:
        raise BrowserOutboxError(str(exc)) from exc


def _timestamp(value: Any, *, field_name: str) -> str:
    try:
        return feedback._validate_rfc3339(value, field_name=field_name)
    except feedback.FeedbackValidationError as exc:
        raise BrowserOutboxError(str(exc)) from exc


def _generation(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= feedback.MAX_SAFE_JSON_INTEGER
    ):
        raise BrowserOutboxError(
            "browser outbox generation must be a non-negative JavaScript-safe integer"
        )
    return value


def _trace_from_value(value: Any) -> feedback.SessionTrace:
    if not isinstance(value, Mapping):
        raise BrowserOutboxError("browser outbox trace must be a JSON object")
    _require_exact_keys(value, _TRACE_KEYS, label="browser outbox trace")
    session_id = _identifier(value["session_id"], field_name="trace.session_id")
    created_at = _timestamp(value["created_at"], field_name="trace.created_at")
    raw_events = value["events"]
    if not isinstance(raw_events, list):
        raise BrowserOutboxError("browser outbox trace.events must be a JSON array")

    trace = feedback.SessionTrace(session_id, created_at=created_at)
    if not raw_events:
        return trace
    try:
        envelope = feedback.build_session_trace_envelope(
            session_id,
            raw_events,
            created_at=created_at,
        )
    except feedback.FeedbackValidationError as exc:
        raise BrowserOutboxError(f"invalid browser outbox trace: {exc}") from exc
    for expected_sequence, event in enumerate(envelope["events"], start=1):
        if event["sequence"] != expected_sequence:
            raise BrowserOutboxError(
                "browser outbox event sequences must be contiguous from one"
            )
        trace.append_event(event)
    return trace


def _identifier_tuple(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BrowserOutboxError(f"{field_name} must be a JSON array")
    resolved = tuple(
        _identifier(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if tuple(sorted(set(resolved))) != resolved:
        raise BrowserOutboxError(f"{field_name} must be sorted and unique")
    return resolved


def _quarantined_tuple(value: Any) -> tuple[dict[str, str | None], ...]:
    if not isinstance(value, list):
        raise BrowserOutboxError("quarantined_events must be a JSON array")
    records: list[dict[str, str | None]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise BrowserOutboxError(
                f"quarantined_events[{index}] must be a JSON object"
            )
        _require_exact_keys(
            raw,
            _QUARANTINE_KEYS,
            label=f"quarantined_events[{index}]",
        )
        request_id = raw["request_id"]
        records.append(
            {
                "event_id": _identifier(
                    raw["event_id"],
                    field_name=f"quarantined_events[{index}].event_id",
                ),
                "request_id": (
                    _identifier(
                        request_id,
                        field_name=f"quarantined_events[{index}].request_id",
                    )
                    if request_id is not None
                    else None
                ),
                "error_code": _identifier(
                    raw["error_code"],
                    field_name=f"quarantined_events[{index}].error_code",
                ),
            }
        )
    event_ids = tuple(str(record["event_id"]) for record in records)
    if tuple(sorted(set(event_ids))) != event_ids:
        raise BrowserOutboxError(
            "quarantined_events must be sorted and unique by event_id"
        )
    return tuple(records)


def _core(
    *,
    generation: int,
    current_attempt_id: str,
    trace: feedback.SessionTrace,
    acknowledged_event_ids: Iterable[str],
    quarantined_events: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(trace, feedback.SessionTrace):
        raise BrowserOutboxError("trace must be a SessionTrace")
    resolved_generation = _generation(generation)
    resolved_attempt = _identifier(
        current_attempt_id,
        field_name="current_attempt_id",
    )
    event_ids = {str(event["event_id"]) for event in trace.events}
    acknowledged = tuple(
        sorted(
            {
                _identifier(item, field_name="acknowledged_event_ids[]")
                for item in acknowledged_event_ids
            }
        )
    )
    raw_quarantined = (
        list(quarantined_events.values())
        if isinstance(quarantined_events, Mapping)
        else list(quarantined_events)
    )
    quarantine_records = _quarantined_tuple(
        sorted(
            (dict(record) for record in raw_quarantined),
            key=lambda record: str(record.get("event_id", "")),
        )
    )
    quarantined_ids = {str(record["event_id"]) for record in quarantine_records}
    acknowledged_set = set(acknowledged)
    if acknowledged_set & quarantined_ids:
        raise BrowserOutboxError("an event cannot be both acknowledged and quarantined")
    unknown = (acknowledged_set | quarantined_ids) - event_ids
    if unknown:
        raise BrowserOutboxError(
            "browser outbox status references unknown event IDs: "
            + ", ".join(sorted(unknown))
        )
    return {
        "schema_version": BROWSER_OUTBOX_SCHEMA_VERSION,
        "snapshot_type": BROWSER_OUTBOX_SNAPSHOT_TYPE,
        "generation": resolved_generation,
        "current_attempt_id": resolved_attempt,
        "trace": {
            "session_id": trace.session_id,
            "created_at": _timestamp(trace.created_at, field_name="trace.created_at"),
            "events": list(trace.events),
        },
        "acknowledged_event_ids": list(acknowledged),
        "quarantined_events": list(quarantine_records),
    }


def serialize_browser_outbox(
    *,
    generation: int,
    current_attempt_id: str,
    trace: feedback.SessionTrace,
    acknowledged_event_ids: Iterable[str] = (),
    quarantined_events: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return canonical, checksummed JSON safe to hand to browser storage."""

    core = _core(
        generation=generation,
        current_attempt_id=current_attempt_id,
        trace=trace,
        acknowledged_event_ids=acknowledged_event_ids,
        quarantined_events=quarantined_events,
    )
    document = {**core, "checksum": _checksum(core)}
    encoded = _canonical_bytes(document)
    if len(encoded) > MAX_BROWSER_OUTBOX_BYTES:
        raise BrowserOutboxError(
            "browser outbox exceeds the local persistence limit "
            f"({len(encoded)} > {MAX_BROWSER_OUTBOX_BYTES} bytes)"
        )
    return encoded.decode("utf-8")


def parse_browser_outbox(
    raw: bytes | bytearray | memoryview | str,
    *,
    max_bytes: int = MAX_BROWSER_OUTBOX_BYTES,
) -> BrowserOutboxSnapshot:
    """Strictly parse browser-controlled bytes into a detached snapshot."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise BrowserOutboxError("max_bytes must be a positive integer")
    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError as exc:
            raise BrowserOutboxError("browser outbox must be UTF-8 JSON") from exc
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        encoded = bytes(raw)
    else:
        raise BrowserOutboxError("browser outbox must be bytes or text")
    if len(encoded) > max_bytes:
        raise BrowserOutboxError(
            "browser outbox exceeds the local persistence limit "
            f"({len(encoded)} > {max_bytes} bytes)"
        )
    try:
        document = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BrowserOutboxError(f"invalid browser outbox JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise BrowserOutboxError("browser outbox must be a JSON object")
    _require_exact_keys(document, _DOCUMENT_KEYS, label="browser outbox")
    if document["schema_version"] != BROWSER_OUTBOX_SCHEMA_VERSION:
        raise BrowserOutboxError("browser outbox schema_version is not supported")
    if document["snapshot_type"] != BROWSER_OUTBOX_SNAPSHOT_TYPE:
        raise BrowserOutboxError("browser outbox snapshot_type is not supported")
    checksum = document["checksum"]
    if not isinstance(checksum, str) or CHECKSUM_PATTERN.fullmatch(checksum) is None:
        raise BrowserOutboxError("browser outbox checksum is invalid")
    core = {key: document[key] for key in document if key != "checksum"}
    if _checksum(core) != checksum:
        raise BrowserOutboxError("browser outbox checksum does not match its content")

    generation = _generation(document["generation"])
    current_attempt_id = _identifier(
        document["current_attempt_id"],
        field_name="current_attempt_id",
    )
    trace = _trace_from_value(document["trace"])
    acknowledged = _identifier_tuple(
        document["acknowledged_event_ids"],
        field_name="acknowledged_event_ids",
    )
    quarantined = _quarantined_tuple(document["quarantined_events"])
    event_ids = {str(event["event_id"]) for event in trace.events}
    acknowledged_set = set(acknowledged)
    quarantined_ids = {str(record["event_id"]) for record in quarantined}
    if acknowledged_set & quarantined_ids:
        raise BrowserOutboxError("an event cannot be both acknowledged and quarantined")
    unknown = (acknowledged_set | quarantined_ids) - event_ids
    if unknown:
        raise BrowserOutboxError(
            "browser outbox status references unknown event IDs: "
            + ", ".join(sorted(unknown))
        )
    return BrowserOutboxSnapshot(
        generation=generation,
        current_attempt_id=current_attempt_id,
        trace=trace,
        acknowledged_event_ids=acknowledged,
        quarantined_events=quarantined,
        checksum=checksum,
        encoded_size=len(encoded),
    )


__all__ = [
    "BROWSER_OUTBOX_SCHEMA_VERSION",
    "BROWSER_OUTBOX_SNAPSHOT_TYPE",
    "MAX_BROWSER_OUTBOX_BYTES",
    "BrowserOutboxError",
    "BrowserOutboxSnapshot",
    "parse_browser_outbox",
    "serialize_browser_outbox",
]
