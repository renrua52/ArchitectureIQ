"""In-memory feedback events and a small HTTP upload client.

This module deliberately depends only on the Python standard library.  The
question inspector can keep a :class:`SessionTrace` in its own session state,
but this module neither knows about Streamlit nor writes traces to disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


EVENT_SCHEMA_VERSION = "1.0"
TRACE_SCHEMA_VERSION = "1.0"
QUESTION_VERSION_ALGORITHM = "qv1"

TRACE_KEYS = frozenset(
    {
        "schema_version",
        "envelope_type",
        "trace_id",
        "session_id",
        "created_at",
        "event_count",
        "events",
    }
)

EVENT_KEYS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_type",
        "occurred_at",
        "session_id",
        "question_id",
        "question_version",
        "payload",
        "sequence",
    }
)
# ``build_event`` produces a valid local event before it belongs to a trace.
# SessionTrace assigns its stable sequence, and every upload envelope therefore
# contains the complete EVENT_KEYS wire shape expected by feedback-ingest.
EVENT_REQUIRED_LOCAL_KEYS = EVENT_KEYS - {"sequence"}
LOGICAL_EVENT_KEYS = (
    "schema_version",
    "event_id",
    "event_type",
    "session_id",
    "question_id",
    "question_version",
    "payload",
)

EVENT_TYPES = frozenset(
    {
        "answer_submitted",
        "question_presented",
        "question_reaction_submitted",
        "custom_setting_proposed",
        "custom_setting_rejected",
        "custom_run_completed",
        "custom_run_failed",
        "comment_submitted",
    }
)

COMMENT_CATEGORIES = frozenset(
    {
        "question_quality",
        "answer_or_result",
        "custom_setting",
        "bug",
        "suggestion",
        "other",
    }
)
MIN_COMMENT_LENGTH = 1
MAX_COMMENT_LENGTH = 2_000

FEEDBACK_ENDPOINT_ENV = "ARCHITECTURE_IQ_FEEDBACK_ENDPOINT"
FEEDBACK_URL_ENV = "ARCHITECTURE_IQ_FEEDBACK_URL"
FEEDBACK_TOKEN_ENV = "ARCHITECTURE_IQ_FEEDBACK_TOKEN"
FEEDBACK_TIMEOUT_ENV = "ARCHITECTURE_IQ_FEEDBACK_TIMEOUT"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_EVENT_SEQUENCE = 2_147_483_647
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
RFC3339_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})\Z"
)


class FeedbackError(Exception):
    """Base exception for feedback validation, configuration, and transport."""


class FeedbackValidationError(FeedbackError, ValueError):
    """Raised when an event or trace does not satisfy the feedback schema."""


class EventConflictError(FeedbackValidationError):
    """Raised when one event id is reused for different event content."""


class FeedbackConfigurationError(FeedbackError, ValueError):
    """Raised when explicit or environment configuration is invalid."""


class FeedbackNotConfiguredError(FeedbackConfigurationError):
    """Raised when authenticated feedback upload is not fully configured."""


class FeedbackUploadError(FeedbackError):
    """A token-safe, diagnosable HTTP or network upload failure."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: int | None = None,
        response_excerpt: str | None = None,
        response: Any = None,
        request_id: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_excerpt = response_excerpt
        self.response = deepcopy(response)
        self.request_id = request_id
        self.error_code: str | None = None
        self.conflict_count: int | None = None
        if isinstance(response, Mapping):
            error = response.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("code"), str):
                self.error_code = error["code"]
            conflict = response.get("conflict")
            if isinstance(conflict, int) and not isinstance(conflict, bool):
                self.conflict_count = conflict
        details = [message, f"endpoint={endpoint}"]
        if status_code is not None:
            details.append(f"status={status_code}")
        if request_id:
            details.append(f"request_id={request_id}")
        if response_excerpt:
            details.append(f"response={response_excerpt}")
        super().__init__("; ".join(details))


class FeedbackUploadConflictError(FeedbackUploadError):
    """The server rejected reuse of an event ID with different content."""


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the ingestion Bearer token on the configured endpoint only."""

    def redirect_request(
        self,
        request: Any,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        del request, response, code, message, headers, new_url
        return None


def _open_feedback_request(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open one upload without forwarding credentials through a redirect."""
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def generate_session_id() -> str:
    """Return a random identifier containing no user or machine information."""
    return f"anon_{secrets.token_urlsafe(18)}"


def generate_event_id() -> str:
    """Return an opaque id suitable for upload idempotency."""
    return f"evt_{secrets.token_urlsafe(18)}"


def _is_inspector_temporary_key(key: str) -> bool:
    return key.startswith("_inspector_")


def _normalize_question_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_question_value(item)
            for key, item in value.items()
            if not _is_inspector_temporary_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_question_value(item) for item in value]
    return value


def normalize_question(question: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical JSON-compatible question used for versioning.

    Inspector-only keys (currently every key beginning with ``_inspector_``)
    are removed recursively.  Mapping key order is normalized by the JSON
    encoder in :func:`compute_question_version`; list order remains meaningful.
    """
    if not isinstance(question, Mapping):
        raise FeedbackValidationError("question must be a mapping")
    normalized = _normalize_question_value(question)
    try:
        # Round-tripping also rejects objects that were never part of question
        # JSON and gives callers an ordinary, detached JSON value.
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        copied = json.loads(encoded)
        _validate_json_interoperability(copied, field_name="question")
        return copied
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError(f"question is not canonical JSON: {exc}") from exc


def compute_question_version(question: Mapping[str, Any]) -> str:
    """Hash normalized question JSON into a stable, algorithm-tagged version."""
    normalized = normalize_question(question)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{QUESTION_VERSION_ALGORITHM}_{digest}"


# A short alias reads naturally at event construction call sites.
question_version = compute_question_version


def _contains_unicode_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_json_interoperability(value: Any, *, field_name: str) -> None:
    """Reject JSON values that cannot round-trip losslessly through JavaScript."""

    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str):
            if _contains_unicode_surrogate(item):
                raise FeedbackValidationError(
                    f"{field_name} cannot contain Unicode surrogate code points"
                )
            continue
        if isinstance(item, int):
            if not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER:
                raise FeedbackValidationError(
                    f"{field_name} integer-valued JSON numbers must be between "
                    f"{-MAX_SAFE_JSON_INTEGER} and {MAX_SAFE_JSON_INTEGER}"
                )
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise FeedbackValidationError(
                    f"{field_name} must contain only finite JSON values"
                )
            if item.is_integer() and not (
                -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER
            ):
                raise FeedbackValidationError(
                    f"{field_name} integer-valued JSON numbers must be between "
                    f"{-MAX_SAFE_JSON_INTEGER} and {MAX_SAFE_JSON_INTEGER}"
                )
            continue
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _copy_json(value: Any, *, field_name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError(
            f"{field_name} must contain only finite JSON values: {exc}"
        ) from exc
    _validate_json_interoperability(copied, field_name=field_name)
    return copied


def _validate_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeedbackValidationError(f"{field_name} must be a non-empty string")
    resolved = value.strip()
    if len(resolved) > 200:
        raise FeedbackValidationError(f"{field_name} must be at most 200 characters")
    if _contains_unicode_surrogate(resolved):
        raise FeedbackValidationError(
            f"{field_name} cannot contain Unicode surrogate code points"
        )
    if "\r" in resolved or "\n" in resolved:
        raise FeedbackValidationError(f"{field_name} cannot contain newlines")
    return resolved


def _validate_rfc3339(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or RFC3339_PATTERN.fullmatch(value) is None:
        raise FeedbackValidationError(f"{field_name} must be an RFC 3339 timestamp")
    zone_start = -1 if value.endswith("Z") else -6
    date_and_time = value[:zone_start]
    zone = value[zone_start:]
    if "." in date_and_time:
        whole_seconds, fractional_seconds = date_and_time.split(".", maxsplit=1)
        date_and_time = f"{whole_seconds}.{fractional_seconds[:6]}"
    normalized = date_and_time + ("+00:00" if zone == "Z" else zone)
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise FeedbackValidationError(
            f"{field_name} must be an RFC 3339 timestamp"
        ) from exc
    return value


def validate_comment(category: str, text: str) -> tuple[str, str]:
    """Validate and normalize one per-question comment."""
    if not isinstance(category, str) or category not in COMMENT_CATEGORIES:
        allowed = ", ".join(sorted(COMMENT_CATEGORIES))
        raise FeedbackValidationError(
            f"unsupported comment category {category!r}; choose one of: {allowed}"
        )
    if not isinstance(text, str):
        raise FeedbackValidationError("comment text must be a string")
    normalized = text.strip()
    if _contains_unicode_surrogate(normalized):
        raise FeedbackValidationError(
            "comment text cannot contain Unicode surrogate code points"
        )
    if len(normalized) < MIN_COMMENT_LENGTH:
        raise FeedbackValidationError("comment text cannot be empty")
    if len(normalized) > MAX_COMMENT_LENGTH:
        raise FeedbackValidationError(
            f"comment text must be at most {MAX_COMMENT_LENGTH} characters"
        )
    return category, normalized


def validate_event_payload(
    event_type: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the required payload shape for a supported event type."""
    if event_type not in EVENT_TYPES:
        allowed = ", ".join(sorted(EVENT_TYPES))
        raise FeedbackValidationError(
            f"unsupported event type {event_type!r}; choose one of: {allowed}"
        )
    if not isinstance(payload, Mapping):
        raise FeedbackValidationError("payload must be a mapping")
    resolved = _copy_json(payload, field_name="payload")

    if event_type == "answer_submitted":
        resolved["selected_letter"] = _validate_identifier(
            resolved.get("selected_letter"), field_name="payload.selected_letter"
        )
        if resolved.get("selected_candidate_id") is not None:
            resolved["selected_candidate_id"] = _validate_identifier(
                resolved["selected_candidate_id"],
                field_name="payload.selected_candidate_id",
            )
    elif event_type == "question_presented":
        for field_name in (
            "attempt_id",
            "release_id",
            "decision_id",
            "policy_version",
            "source",
        ):
            resolved[field_name] = _validate_identifier(
                resolved.get(field_name), field_name=f"payload.{field_name}"
            )
        if resolved.get("mode") not in {
            "exploit",
            "explore",
            "fallback",
            "manual",
        }:
            raise FeedbackValidationError(
                "payload.mode must be exploit, explore, fallback, or manual"
            )
        if resolved["source"] not in {"initial", "next", "random", "picker"}:
            raise FeedbackValidationError(
                "payload.source must be initial, next, random, or picker"
            )
        propensity = resolved.get("propensity")
        if (
            isinstance(propensity, bool)
            or not isinstance(propensity, (int, float))
            or not math.isfinite(propensity)
            or not 0 < propensity <= 1
        ):
            raise FeedbackValidationError(
                "payload.propensity must be a finite number in (0, 1]"
            )
        position = resolved.get("position")
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
            or not float(position).is_integer()
            or position <= 0
            or position > MAX_SAFE_JSON_INTEGER
        ):
            raise FeedbackValidationError(
                "payload.position must be a positive JavaScript-safe integer"
            )
    elif event_type == "question_reaction_submitted":
        if resolved.get("reaction") != "surprise":
            raise FeedbackValidationError(
                "payload.reaction must be 'surprise' for a question reaction"
            )
        if not isinstance(resolved.get("value"), bool):
            raise FeedbackValidationError(
                "payload.value must be a boolean for a question reaction"
            )
        if resolved.get("timing") != "after_reveal":
            raise FeedbackValidationError(
                "payload.timing must be 'after_reveal' for a question reaction"
            )
        resolved["attempt_id"] = _validate_identifier(
            resolved.get("attempt_id"), field_name="payload.attempt_id"
        )
        if resolved.get("release_id") is not None:
            resolved["release_id"] = _validate_identifier(
                resolved["release_id"], field_name="payload.release_id"
            )
    elif event_type in {"custom_setting_proposed", "custom_setting_rejected"}:
        if not isinstance(resolved.get("setting"), dict):
            raise FeedbackValidationError("payload.setting must be a JSON object")
    elif event_type in {"custom_run_completed", "custom_run_failed"}:
        if not isinstance(resolved.get("run"), dict):
            raise FeedbackValidationError("payload.run must be a JSON object")
        expected_status = (
            "completed" if event_type == "custom_run_completed" else "failed"
        )
        if resolved["run"].get("status") != expected_status:
            raise FeedbackValidationError(
                f"payload.run.status must be {expected_status!r} for {event_type}"
            )
    else:
        category, text = validate_comment(
            resolved.get("category"), resolved.get("text")
        )
        resolved["category"] = category
        resolved["text"] = text
    return resolved


def _merge_extra(
    payload: dict[str, Any],
    extra: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> None:
    if not extra:
        return
    copied = _copy_json(extra, field_name=field_name)
    overlap = sorted(payload.keys() & copied.keys())
    if overlap:
        raise FeedbackValidationError(
            f"{field_name} cannot replace reserved fields: {', '.join(overlap)}"
        )
    payload.update(copied)


def build_event(
    event_type: str,
    *,
    session_id: str,
    question: Mapping[str, Any],
    payload: Mapping[str, Any],
    event_id: str | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable-by-convention, versioned feedback event value."""
    if event_type not in EVENT_TYPES:
        allowed = ", ".join(sorted(EVENT_TYPES))
        raise FeedbackValidationError(
            f"unsupported event type {event_type!r}; choose one of: {allowed}"
        )
    resolved_session_id = _validate_identifier(session_id, field_name="session_id")
    resolved_event_id = _validate_identifier(
        event_id or generate_event_id(), field_name="event_id"
    )
    question_id = _validate_identifier(
        question.get("question_id"), field_name="question.question_id"
    )
    resolved_payload = validate_event_payload(event_type, payload)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": resolved_event_id,
        "event_type": event_type,
        "occurred_at": occurred_at or _utc_now(),
        "session_id": resolved_session_id,
        "question_id": question_id,
        "question_version": compute_question_version(question),
        "payload": resolved_payload,
    }


def _logical_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must match when an idempotency key is reused."""
    return {key: deepcopy(event[key]) for key in LOGICAL_EVENT_KEYS}


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare detached JSON values without Python's bool/number coercion.

    JSON has distinct boolean and number types, while Python considers
    ``True == 1``. JSON numbers retain their ordinary mathematical equality,
    so ``1 == 1.0`` and ``-0.0 == 0`` remain true. Object key order is not
    meaningful, but array order is.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and left == right
        )
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(
                _json_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_json_values_equal(left[key], right[key]) for key in left)
        )
    return False


def _validate_event_shape(event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise FeedbackValidationError("event must be a mapping")
    copied = _copy_json(event, field_name="event")
    unsupported = sorted(copied.keys() - EVENT_KEYS)
    if unsupported:
        raise FeedbackValidationError(
            f"event contains unsupported fields: {', '.join(unsupported)}"
        )
    missing = sorted(EVENT_REQUIRED_LOCAL_KEYS - copied.keys())
    if missing:
        raise FeedbackValidationError(
            f"event is missing required fields: {', '.join(missing)}"
        )
    if copied["schema_version"] != EVENT_SCHEMA_VERSION:
        raise FeedbackValidationError(
            f"unsupported event schema version: {copied['schema_version']!r}"
        )
    if copied["event_type"] not in EVENT_TYPES:
        raise FeedbackValidationError(
            f"unsupported event type: {copied['event_type']!r}"
        )
    for name in (
        "event_id",
        "session_id",
        "question_id",
        "question_version",
    ):
        copied[name] = _validate_identifier(copied[name], field_name=name)
    copied["occurred_at"] = _validate_rfc3339(
        copied["occurred_at"], field_name="occurred_at"
    )
    if not isinstance(copied["payload"], dict):
        raise FeedbackValidationError("event payload must be a JSON object")
    if "sequence" in copied:
        sequence = copied["sequence"]
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= 0
            or sequence > MAX_EVENT_SEQUENCE
        ):
            raise FeedbackValidationError(
                f"event sequence must be an integer between 1 and {MAX_EVENT_SEQUENCE}"
            )
    copied["payload"] = validate_event_payload(copied["event_type"], copied["payload"])
    return copied


def _validated_session_events(
    events: Sequence[Mapping[str, Any]],
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate and detach one append-only, single-session event sequence."""
    resolved_session_id = (
        _validate_identifier(session_id, field_name="session_id")
        if session_id is not None
        else None
    )
    resolved_events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_sequence = 0
    for expected_sequence, event in enumerate(events, start=1):
        copied = _validate_event_shape(event)
        if resolved_session_id is None:
            resolved_session_id = copied["session_id"]
        elif copied["session_id"] != resolved_session_id:
            raise FeedbackValidationError(
                "every event in a session must use the same session_id"
            )
        if copied["event_id"] in seen_ids:
            raise FeedbackValidationError(
                f"duplicate event_id in session: {copied['event_id']!r}"
            )
        seen_ids.add(copied["event_id"])
        sequence = copied.get("sequence", expected_sequence)
        if sequence <= previous_sequence:
            raise FeedbackValidationError(
                "event sequences must be strictly increasing within a session"
            )
        copied["sequence"] = sequence
        previous_sequence = sequence
        resolved_events.append(copied)
    return resolved_events


class SessionTrace:
    """An in-memory append-only event sequence for one anonymous session."""

    def __init__(
        self,
        session_id: str | None = None,
        *,
        created_at: str | None = None,
    ) -> None:
        self._session_id = _validate_identifier(
            session_id or generate_session_id(), field_name="session_id"
        )
        self._created_at = created_at or _utc_now()
        self._events: list[dict[str, Any]] = []
        self._event_indexes: dict[str, int] = {}

    @classmethod
    def new(cls) -> "SessionTrace":
        return cls()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def created_at(self) -> str:
        return self._created_at

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        # Callers cannot rewrite the trace through returned nested containers.
        return tuple(deepcopy(self._events))

    def __len__(self) -> int:
        return len(self._events)

    def append_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Append an event, or return the original for an idempotent replay.

        Reusing an ``event_id`` with the same logical content is a no-op.  A
        different payload under that id is rejected, protecting append-only
        traces from accidental replacement.
        """
        copied = _validate_event_shape(event)
        if copied["session_id"] != self._session_id:
            raise FeedbackValidationError(
                "event session_id does not match this session trace"
            )
        event_id = copied["event_id"]
        existing_index = self._event_indexes.get(event_id)
        if existing_index is not None:
            existing = self._events[existing_index]
            if not _json_values_equal(
                _logical_event(existing),
                _logical_event(copied),
            ):
                raise EventConflictError(
                    f"event_id {event_id!r} is already used by different content"
                )
            return deepcopy(existing)

        copied["sequence"] = len(self._events) + 1
        self._event_indexes[event_id] = len(self._events)
        self._events.append(copied)
        return deepcopy(copied)

    def _record(
        self,
        event_type: str,
        question: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        event_id: str | None,
        occurred_at: str | None,
    ) -> dict[str, Any]:
        return self.append_event(
            build_event(
                event_type,
                session_id=self._session_id,
                question=question,
                payload=payload,
                event_id=event_id,
                occurred_at=occurred_at,
            )
        )

    def record_answer(
        self,
        question: Mapping[str, Any],
        *,
        selected_letter: str,
        selected_candidate_id: str | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        letter = _validate_identifier(selected_letter, field_name="selected_letter")
        payload: dict[str, Any] = {"selected_letter": letter}
        if selected_candidate_id is not None:
            payload["selected_candidate_id"] = _validate_identifier(
                selected_candidate_id, field_name="selected_candidate_id"
            )
        _merge_extra(payload, extra, field_name="answer extra")
        return self._record(
            "answer_submitted",
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_question_reaction(
        self,
        question: Mapping[str, Any],
        *,
        value: bool,
        attempt_id: str,
        release_id: str | None = None,
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the first immutable surprise reaction for one quiz attempt.

        The deterministic event id gives the same session/attempt/question one
        idempotency key. Replaying the same value is a no-op; trying to replace
        it with the opposite value is an explicit content conflict.
        """
        if not isinstance(value, bool):
            raise FeedbackValidationError("question reaction value must be a boolean")
        resolved_attempt_id = _validate_identifier(attempt_id, field_name="attempt_id")
        resolved_release_id = (
            _validate_identifier(release_id, field_name="release_id")
            if release_id is not None
            else None
        )
        resolved_question_version = compute_question_version(question)
        if event_id is None:
            identity = "\0".join(
                (
                    EVENT_SCHEMA_VERSION,
                    self._session_id,
                    resolved_attempt_id,
                    resolved_release_id or "unreleased",
                    resolved_question_version,
                    "surprise",
                )
            ).encode("utf-8")
            event_id = f"evt_reaction_{hashlib.sha256(identity).hexdigest()}"
        payload: dict[str, Any] = {
            "reaction": "surprise",
            "value": value,
            "timing": "after_reveal",
            "attempt_id": resolved_attempt_id,
        }
        if resolved_release_id is not None:
            payload["release_id"] = resolved_release_id
        _merge_extra(payload, extra, field_name="question reaction extra")
        return self._record(
            "question_reaction_submitted",
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_question_presented(
        self,
        question: Mapping[str, Any],
        *,
        attempt_id: str,
        release_id: str,
        decision_id: str,
        policy_version: str,
        mode: str,
        propensity: float,
        source: str,
        position: int,
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        """Record one auditable question exposure/selection decision."""
        resolved_decision_id = _validate_identifier(
            decision_id, field_name="decision_id"
        )
        if event_id is None:
            identity = "\0".join(
                (
                    EVENT_SCHEMA_VERSION,
                    self._session_id,
                    resolved_decision_id,
                    compute_question_version(question),
                )
            ).encode("utf-8")
            event_id = f"evt_presented_{hashlib.sha256(identity).hexdigest()}"
        return self._record(
            "question_presented",
            question,
            {
                "attempt_id": attempt_id,
                "release_id": release_id,
                "decision_id": resolved_decision_id,
                "policy_version": policy_version,
                "mode": mode,
                "propensity": propensity,
                "source": source,
                "position": position,
            },
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_custom_setting(
        self,
        question: Mapping[str, Any],
        *,
        setting: Mapping[str, Any],
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "setting": _copy_json(setting, field_name="custom setting")
        }
        _merge_extra(payload, extra, field_name="custom setting extra")
        return self._record(
            "custom_setting_proposed",
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_custom_setting_rejected(
        self,
        question: Mapping[str, Any],
        *,
        setting: Mapping[str, Any],
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "setting": _copy_json(setting, field_name="rejected custom setting")
        }
        _merge_extra(payload, extra, field_name="rejected custom setting extra")
        return self._record(
            "custom_setting_rejected",
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_custom_run(
        self,
        question: Mapping[str, Any],
        *,
        run: Mapping[str, Any],
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"run": _copy_json(run, field_name="custom run")}
        _merge_extra(payload, extra, field_name="custom run extra")
        status = payload["run"].get("status")
        event_type_by_status = {
            "completed": "custom_run_completed",
            "failed": "custom_run_failed",
        }
        if not isinstance(status, str) or status not in event_type_by_status:
            raise FeedbackValidationError(
                "custom run status must be either 'completed' or 'failed'"
            )
        return self._record(
            event_type_by_status[status],
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def record_comment(
        self,
        question: Mapping[str, Any],
        *,
        category: str,
        text: str,
        event_id: str | None = None,
        occurred_at: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_category, resolved_text = validate_comment(category, text)
        payload: dict[str, Any] = {
            "category": resolved_category,
            "text": resolved_text,
        }
        _merge_extra(payload, extra, field_name="comment extra")
        return self._record(
            "comment_submitted",
            question,
            payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    def to_envelope(self) -> dict[str, Any]:
        return build_session_trace_envelope(
            self._session_id,
            self._events,
            created_at=self._created_at,
        )


def build_session_trace_envelope(
    session_id: str,
    events: Sequence[Mapping[str, Any]],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Construct a complete, uploadable envelope for a session trace."""
    resolved_session_id = _validate_identifier(session_id, field_name="session_id")
    resolved_events = _validated_session_events(
        events,
        session_id=resolved_session_id,
    )
    resolved_created_at = _validate_rfc3339(
        created_at or _utc_now(), field_name="created_at"
    )

    trace_fingerprint = json.dumps(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": resolved_session_id,
            "event_ids": [event["event_id"] for event in resolved_events],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    trace_id = f"trace_{hashlib.sha256(trace_fingerprint).hexdigest()}"
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "envelope_type": "session_trace",
        "trace_id": trace_id,
        "session_id": resolved_session_id,
        "created_at": resolved_created_at,
        "event_count": len(resolved_events),
        "events": resolved_events,
    }


def validate_session_trace_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach one complete downloaded/uploadable trace envelope."""

    if not isinstance(envelope, Mapping):
        raise FeedbackValidationError("trace envelope must be a mapping")
    copied = _copy_json(envelope, field_name="trace envelope")
    missing = sorted(TRACE_KEYS - copied.keys())
    unsupported = sorted(copied.keys() - TRACE_KEYS)
    if missing or unsupported:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unsupported:
            details.append(f"unsupported: {', '.join(unsupported)}")
        raise FeedbackValidationError(
            f"trace envelope must use the exact schema ({'; '.join(details)})"
        )
    if copied["schema_version"] != TRACE_SCHEMA_VERSION:
        raise FeedbackValidationError("trace envelope schema_version is not supported")
    if copied["envelope_type"] != "session_trace":
        raise FeedbackValidationError("trace envelope_type must be 'session_trace'")
    trace_id = _validate_identifier(copied["trace_id"], field_name="trace_id")
    session_id = _validate_identifier(copied["session_id"], field_name="session_id")
    created_at = _validate_identifier(copied["created_at"], field_name="created_at")
    event_count = copied["event_count"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
    ):
        raise FeedbackValidationError("trace event_count must be a positive integer")
    raw_events = copied["events"]
    if not isinstance(raw_events, list):
        raise FeedbackValidationError("trace events must be a JSON array")
    if event_count != len(raw_events):
        raise FeedbackValidationError("trace event_count does not match events")
    for index, event in enumerate(raw_events, start=1):
        if not isinstance(event, dict):
            raise FeedbackValidationError(f"trace event {index} must be a JSON object")
        missing_event_keys = sorted(EVENT_KEYS - event.keys())
        unsupported_event_keys = sorted(event.keys() - EVENT_KEYS)
        if missing_event_keys or unsupported_event_keys:
            details = []
            if missing_event_keys:
                details.append(f"missing: {', '.join(missing_event_keys)}")
            if unsupported_event_keys:
                details.append(f"unsupported: {', '.join(unsupported_event_keys)}")
            raise FeedbackValidationError(
                f"trace event {index} must use the exact wire schema "
                f"({'; '.join(details)})"
            )
    rebuilt = build_session_trace_envelope(
        session_id,
        raw_events,
        created_at=created_at,
    )
    if trace_id != rebuilt["trace_id"]:
        raise FeedbackValidationError(
            "trace_id does not match the session and ordered event IDs"
        )
    return rebuilt


def parse_session_trace_json(raw: bytes | str) -> dict[str, Any]:
    """Parse strict JSON from a downloaded trace and validate its full schema."""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise FeedbackValidationError("session trace must be UTF-8 JSON") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise FeedbackValidationError("session trace input must be bytes or text")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key!r}")
            value[key] = item
        return value

    try:
        document = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise FeedbackValidationError(f"invalid session trace JSON: {exc}") from exc
    return validate_session_trace_envelope(document)


def _summary_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _summary_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _nested_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _summary_row(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-sensitive fields shared by every session table row."""
    return {
        "sequence": event["sequence"],
        "occurred_at": event["occurred_at"],
        "question_id": event["question_id"],
        "question_version": event["question_version"],
    }


def summarize_session_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a safe, UI-ready summary of one append-only session.

    Accuracy uses only answer events whose ``payload.is_correct`` value is a
    boolean.  Rows retain their event order but expose only an explicit
    allowlist; arbitrary payload extras, event ids, and the session id are not
    copied into the result. Surprise rows retain only the explicit boolean and
    timing contract. Comment text is included intentionally so a user can
    review the message they submitted.
    """
    resolved_events = _validated_session_events(events)
    event_type_counts = {event_type: 0 for event_type in sorted(EVENT_TYPES)}
    answer_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    reaction_rows: list[dict[str, Any]] = []
    presentation_rows: list[dict[str, Any]] = []
    comment_rows: list[dict[str, Any]] = []
    answered_versions: set[str] = set()
    correct_answers = 0
    incorrect_answers = 0

    for event in resolved_events:
        event_type = event["event_type"]
        event_type_counts[event_type] += 1
        payload = event["payload"]

        if event_type == "answer_submitted":
            is_correct = payload.get("is_correct")
            known_correctness = is_correct if isinstance(is_correct, bool) else None
            answered_versions.add(event["question_version"])
            if known_correctness is True:
                correct_answers += 1
            elif known_correctness is False:
                incorrect_answers += 1
            answer_rows.append(
                {
                    **_summary_row(event),
                    "selected_letter": payload["selected_letter"],
                    "selected_candidate_id": _summary_string(
                        payload.get("selected_candidate_id")
                    ),
                    "is_correct": known_correctness,
                }
            )
        elif event_type == "question_presented":
            presentation_rows.append(
                {
                    **_summary_row(event),
                    "policy_version": payload["policy_version"],
                    "mode": payload["mode"],
                    "propensity": payload["propensity"],
                    "source": payload["source"],
                    "position": payload["position"],
                }
            )
        elif event_type == "question_reaction_submitted":
            reaction_rows.append(
                {
                    **_summary_row(event),
                    "reaction": payload["reaction"],
                    "value": payload["value"],
                    "timing": payload["timing"],
                    "attempt_id": payload["attempt_id"],
                }
            )
        elif event_type in {
            "custom_setting_proposed",
            "custom_setting_rejected",
        }:
            setting = payload["setting"]
            budget = _nested_mapping(setting, "budget")
            model = _nested_mapping(setting, "model")
            optimizer = _nested_mapping(setting, "optimizer")
            loss = _nested_mapping(setting, "loss")
            proposal_rows.append(
                {
                    **_summary_row(event),
                    "status": (
                        "proposed"
                        if event_type == "custom_setting_proposed"
                        else "rejected"
                    ),
                    "label": _summary_string(payload.get("label")),
                    "candidate_id": _summary_string(setting.get("candidate_id")),
                    "model_type": _summary_string(model.get("type")),
                    "optimizer_type": _summary_string(
                        optimizer.get("type") or optimizer.get("optimizer_type")
                    ),
                    "loss_id": _summary_string(loss.get("loss_id")),
                    "total_samples_seen": _summary_number(
                        budget.get("total_samples_seen")
                    ),
                    "batch_size": _summary_number(budget.get("batch_size")),
                    "error_type": _summary_string(payload.get("error_type")),
                }
            )
        elif event_type == "comment_submitted":
            comment_rows.append(
                {
                    **_summary_row(event),
                    "category": payload["category"],
                    "text": payload["text"],
                }
            )

    known_answers = correct_answers + incorrect_answers
    answer_count = event_type_counts["answer_submitted"]
    return {
        "event_count": len(resolved_events),
        "event_type_counts": event_type_counts,
        "answers": {
            "total": answer_count,
            "unique_question_versions": len(answered_versions),
            "known": known_answers,
            "correct": correct_answers,
            "incorrect": incorrect_answers,
            "unknown": answer_count - known_answers,
            "accuracy": (correct_answers / known_answers if known_answers else None),
        },
        "settings": {
            "proposed": event_type_counts["custom_setting_proposed"],
            "rejected": event_type_counts["custom_setting_rejected"],
        },
        "runs": {
            "completed": event_type_counts["custom_run_completed"],
            "failed": event_type_counts["custom_run_failed"],
        },
        "reactions": {
            "total": event_type_counts["question_reaction_submitted"],
            "surprised": sum(row["value"] is True for row in reaction_rows),
            "not_surprised": sum(row["value"] is False for row in reaction_rows),
        },
        "presentations": event_type_counts["question_presented"],
        "comments": event_type_counts["comment_submitted"],
        "answer_rows": answer_rows,
        "proposal_rows": proposal_rows,
        "reaction_rows": reaction_rows,
        "presentation_rows": presentation_rows,
        "comment_rows": comment_rows,
    }


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FeedbackConfigurationError(
            "feedback endpoint and token must be strings when configured"
        )
    cleaned = value.strip()
    return cleaned or None


def _validated_timeout(value: float | str | None) -> float:
    if value is None or value == "":
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise FeedbackConfigurationError(
            "feedback timeout must be a positive number"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise FeedbackConfigurationError("feedback timeout must be greater than zero")
    return timeout


def _validate_endpoint(endpoint: str | None) -> str | None:
    endpoint = _clean_optional_string(endpoint)
    if endpoint is None:
        return None
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FeedbackConfigurationError(
            "feedback endpoint must be an absolute http(s) URL"
        )
    if "\r" in endpoint or "\n" in endpoint:
        raise FeedbackConfigurationError("feedback endpoint cannot contain newlines")
    try:
        parsed.port
    except ValueError as exc:
        raise FeedbackConfigurationError(
            "feedback endpoint has an invalid port"
        ) from exc
    return endpoint


@dataclass(frozen=True)
class FeedbackConfig:
    """Upload configuration resolved from arguments and/or environment."""

    endpoint: str | None
    bearer_token: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        object.__setattr__(
            self, "bearer_token", _clean_optional_string(self.bearer_token)
        )
        if self.bearer_token and (
            "\r" in self.bearer_token or "\n" in self.bearer_token
        ):
            raise FeedbackConfigurationError("feedback token cannot contain newlines")
        object.__setattr__(
            self, "timeout_seconds", _validated_timeout(self.timeout_seconds)
        )

    @classmethod
    def from_sources(
        cls,
        *,
        endpoint: str | None = None,
        bearer_token: str | None = None,
        timeout_seconds: float | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "FeedbackConfig":
        env = os.environ if environ is None else environ
        resolved_endpoint = endpoint
        if resolved_endpoint is None:
            resolved_endpoint = env.get(FEEDBACK_ENDPOINT_ENV) or env.get(
                FEEDBACK_URL_ENV
            )
        resolved_token = (
            bearer_token if bearer_token is not None else env.get(FEEDBACK_TOKEN_ENV)
        )
        resolved_timeout: float | str | None = timeout_seconds
        if resolved_timeout is None:
            resolved_timeout = env.get(FEEDBACK_TIMEOUT_ENV)
        return cls(
            endpoint=resolved_endpoint,
            bearer_token=resolved_token,
            timeout_seconds=_validated_timeout(resolved_timeout),
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "FeedbackConfig":
        return cls.from_sources(environ=environ)

    @property
    def is_configured(self) -> bool:
        """Whether the authenticated STORE-001 upload contract is ready."""
        return self.endpoint is not None and self.bearer_token is not None

    def require_configured(self) -> str:
        if self.endpoint is None:
            raise FeedbackNotConfiguredError(
                "feedback upload endpoint is not configured; "
                f"set {FEEDBACK_ENDPOINT_ENV}"
            )
        if self.bearer_token is None:
            raise FeedbackNotConfiguredError(
                "feedback upload Bearer token is not configured; "
                f"set {FEEDBACK_TOKEN_ENV}"
            )
        return self.endpoint

    def __repr__(self) -> str:
        endpoint = self.endpoint
        if endpoint is not None:
            endpoint = _redact(_safe_endpoint(endpoint), self.bearer_token)
        return (
            "FeedbackConfig("
            f"endpoint={endpoint!r}, "
            f"timeout_seconds={self.timeout_seconds!r}"
            ")"
        )


def is_feedback_configured(
    *,
    endpoint: str | None = None,
    bearer_token: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Check availability without attempting a request."""
    return FeedbackConfig.from_sources(
        endpoint=endpoint,
        bearer_token=bearer_token,
        environ=environ,
    ).is_configured


def _safe_endpoint(endpoint: str) -> str:
    """Remove credentials, query parameters, and fragments from diagnostics."""
    parsed = urllib.parse.urlsplit(endpoint)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _redact(value: str, secret: str | None) -> str:
    redacted = value
    if secret:
        for candidate in {secret, urllib.parse.quote(secret, safe="")}:
            if candidate:
                redacted = redacted.replace(candidate, "[REDACTED]")
    return redacted


def _response_text(raw: bytes, secret: str | None) -> str:
    return _redact(raw.decode("utf-8", errors="replace"), secret)


def _parsed_response(text: str) -> Any:
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _response_request_id(response: Any, secret: str | None) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for name in ("X-Request-ID", "X-Correlation-ID", "Request-ID"):
        value = headers.get(name)
        if value:
            return _redact(str(value), secret)
    return None


@dataclass(frozen=True)
class UploadReceipt:
    """The non-sensitive receipt returned after a successful POST."""

    status_code: int
    endpoint: str
    response: Any = None
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "response": deepcopy(self.response),
        }


def upload_receipt_acknowledges_all(
    receipt: UploadReceipt,
    *,
    sent_count: int,
    allow_legacy_generic_2xx: bool = False,
) -> bool:
    """Return whether a receipt acknowledges every event in one upload.

    The STORE-001 contract is fail-closed: a successful response needs all four
    counters plus one canonical request UUID repeated exactly in the response
    body.  Legacy endpoints that return a generic 2xx without any counters may
    be accepted only through the explicit compatibility flag.  Once even one
    counter is present, legacy mode cannot relax the strict contract.
    """
    if (
        isinstance(sent_count, bool)
        or not isinstance(sent_count, int)
        or sent_count < 0
    ):
        raise FeedbackValidationError("sent_count must be a non-negative integer")
    if not receipt.ok:
        return False

    response = receipt.response
    counter_names = ("accepted", "duplicate", "rejected", "conflict")
    present_counters = (
        {name for name in counter_names if name in response}
        if isinstance(response, Mapping)
        else set()
    )
    if not present_counters:
        return allow_legacy_generic_2xx is True
    if not isinstance(response, Mapping) or present_counters != set(counter_names):
        return False

    request_id = receipt.request_id
    if not isinstance(request_id, str):
        return False
    try:
        parsed_request_id = uuid.UUID(request_id)
    except (ValueError, AttributeError):
        return False
    if (
        str(parsed_request_id) != request_id
        or parsed_request_id.variant != uuid.RFC_4122
        or parsed_request_id.version not in {1, 2, 3, 4, 5}
        or response.get("request_id") != request_id
    ):
        return False

    counters: dict[str, int] = {}
    for name in counter_names:
        value = response.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        counters[name] = value
    return (
        counters["conflict"] == 0
        and counters["rejected"] == 0
        and counters["accepted"] + counters["duplicate"] == sent_count
    )


class FeedbackClient:
    """POST feedback JSON using :mod:`urllib.request`."""

    def __init__(
        self,
        config: FeedbackConfig | None = None,
        *,
        endpoint: str | None = None,
        bearer_token: str | None = None,
        timeout_seconds: float | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if config is not None and any(
            value is not None for value in (endpoint, bearer_token, timeout_seconds)
        ):
            raise FeedbackConfigurationError(
                "pass either config or explicit feedback parameters, not both"
            )
        self._config = config or FeedbackConfig.from_sources(
            endpoint=endpoint,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
            environ=environ,
        )

    @property
    def config(self) -> FeedbackConfig:
        return self._config

    @property
    def is_configured(self) -> bool:
        return self._config.is_configured

    def post_json(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> UploadReceipt:
        endpoint = self._config.require_configured()
        token = self._config.bearer_token
        safe_endpoint = _redact(_safe_endpoint(endpoint), token)
        if not isinstance(payload, Mapping):
            raise FeedbackValidationError("upload payload must be a mapping")
        copied_payload = _copy_json(payload, field_name="upload payload")
        try:
            body = json.dumps(
                copied_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise FeedbackValidationError(
                f"upload payload must contain only finite JSON values: {exc}"
            ) from exc

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "ArchitectureIQ-question-inspector/1",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = _validate_identifier(
                idempotency_key, field_name="idempotency_key"
            )
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with _open_feedback_request(
                request, timeout=self._config.timeout_seconds
            ) as response:
                response_status = getattr(response, "status", None)
                if response_status is None:
                    response_status = response.getcode()
                status = int(response_status)
                raw = response.read()
                response_text = _response_text(raw, token)
                if not 200 <= status < 300:
                    parsed = _parsed_response(response_text)
                    error_code = (
                        parsed.get("error", {}).get("code")
                        if isinstance(parsed, Mapping)
                        and isinstance(parsed.get("error"), Mapping)
                        else None
                    )
                    error_type = (
                        FeedbackUploadConflictError
                        if status == 409 and error_code == "EVENT_ID_CONFLICT"
                        else FeedbackUploadError
                    )
                    raise error_type(
                        "feedback event ID conflicts with stored content"
                        if error_type is FeedbackUploadConflictError
                        else "feedback server rejected the upload",
                        endpoint=safe_endpoint,
                        status_code=status,
                        response_excerpt=response_text[:1_000],
                        response=parsed,
                        request_id=_response_request_id(response, token),
                    )
                return UploadReceipt(
                    status_code=status,
                    endpoint=safe_endpoint,
                    response=_parsed_response(response_text),
                    request_id=_response_request_id(response, token),
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            response_text = _response_text(raw, token)
            parsed = _parsed_response(response_text)
            error_code = (
                parsed.get("error", {}).get("code")
                if isinstance(parsed, Mapping)
                and isinstance(parsed.get("error"), Mapping)
                else None
            )
            error_type = (
                FeedbackUploadConflictError
                if exc.code == 409 and error_code == "EVENT_ID_CONFLICT"
                else FeedbackUploadError
            )
            raise error_type(
                "feedback event ID conflicts with stored content"
                if error_type is FeedbackUploadConflictError
                else "feedback server rejected the upload",
                endpoint=safe_endpoint,
                status_code=exc.code,
                response_excerpt=response_text[:1_000],
                response=parsed,
                request_id=_response_request_id(exc, token),
            ) from None
        except FeedbackUploadError:
            raise
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            reason = _redact(str(getattr(exc, "reason", exc)), token)
            raise FeedbackUploadError(
                f"feedback upload could not reach the server: {reason}",
                endpoint=safe_endpoint,
            ) from None

    def post_event(self, event: Mapping[str, Any]) -> UploadReceipt:
        copied = _validate_event_shape(event)
        # A one-event submission uses the exact same envelope contract as a
        # whole-session upload.  The server therefore needs only one endpoint
        # and one request schema; event_id remains the retry idempotency key.
        envelope = build_session_trace_envelope(
            copied["session_id"],
            [copied],
            created_at=copied["occurred_at"],
        )
        return self.post_json(envelope, idempotency_key=copied["event_id"])

    def post_trace(self, trace: SessionTrace | Mapping[str, Any]) -> UploadReceipt:
        envelope = trace.to_envelope() if isinstance(trace, SessionTrace) else trace
        validated = validate_session_trace_envelope(envelope)
        return self.post_json(validated, idempotency_key=validated["trace_id"])


def post_feedback_json(
    payload: Mapping[str, Any],
    *,
    config: FeedbackConfig | None = None,
    endpoint: str | None = None,
    bearer_token: str | None = None,
    timeout_seconds: float | str | None = None,
    environ: Mapping[str, str] | None = None,
    idempotency_key: str | None = None,
) -> UploadReceipt:
    """Functional convenience wrapper around :class:`FeedbackClient`."""
    client = FeedbackClient(
        config,
        endpoint=endpoint,
        bearer_token=bearer_token,
        timeout_seconds=timeout_seconds,
        environ=environ,
    )
    return client.post_json(payload, idempotency_key=idempotency_key)
