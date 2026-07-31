"""Read-only, standard-library access to ArchitectureIQ feedback reports."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


REPORTS_URL_ENV = "ARCHITECTURE_IQ_REPORTS_URL"
REPORTS_READ_TOKEN_ENV = "ARCHITECTURE_IQ_REPORTS_READ_TOKEN"
REPORTS_TIMEOUT_ENV = "ARCHITECTURE_IQ_REPORTS_TIMEOUT"

DEFAULT_TIMEOUT_SECONDS = 10.0
MIN_TIMEOUT_SECONDS = 0.1
MAX_TIMEOUT_SECONDS = 30.0
DEFAULT_LIMIT = 200
MAX_LIMIT = 1_000
MAX_OFFSET = 1_000_000
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_FILTER_LENGTH = 200
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
MIN_POSTGRES_INTEGER = -2_147_483_648
MAX_POSTGRES_INTEGER = 2_147_483_647

_COMMENT_CATEGORIES = frozenset(
    {
        "question_quality",
        "answer_or_result",
        "custom_setting",
        "bug",
        "suggestion",
        "other",
    }
)
_FEEDBACK_EVENT_TYPES = frozenset(
    {
        "answer_submitted",
        "custom_setting_proposed",
        "custom_setting_rejected",
        "custom_run_completed",
        "custom_run_failed",
        "comment_submitted",
        "question_reaction_submitted",
        "question_presented",
    }
)
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_RELEASE_ID_PATTERN = re.compile(r"^release_[0-9a-f]{64}$")
_REGISTRY_ID_PATTERN = re.compile(r"^registry_[0-9a-f]{64}$")
_QUESTION_VERSION_PATTERN = re.compile(r"^qv1_[0-9a-f]{64}$")


@dataclass(frozen=True)
class _ViewSpec:
    columns: tuple[str, ...]
    filters: frozenset[str]


_SUMMARY_COLUMNS = (
    "event_count",
    "first_event_at",
    "last_event_at",
    "session_count",
    "attempt_count",
    "solve_attempt_count",
    "answered_attempt_count",
    "question_count",
    "answer_count",
    "known_answer_count",
    "correct_answer_count",
    "incorrect_answer_count",
    "unknown_answer_count",
    "accuracy",
    "proposal_count",
    "rejected_setting_count",
    "completed_run_count",
    "failed_run_count",
    "comment_count",
    "attempts_with_proposal",
    "proposal_usage_rate",
    "ingestion_failure_rate",
    "ingestion_failure_rate_available",
)

_INGESTION_SUMMARY_COLUMNS = (
    "recorded_request_count",
    "first_started_at",
    "last_finished_at",
    "success_request_count",
    "client_rejection_count",
    "service_failure_count",
    "event_id_conflict_request_count",
    "accepted_event_count",
    "duplicate_event_count",
    "idempotent_duplicate_event_count",
    "unclassified_duplicate_event_count",
    "conflicting_event_count",
    "conflict_audit_event_count",
    "event_id_reuse_count",
    "classified_event_count",
    "known_event_result_count",
    "request_failure_rate",
    "duplicate_event_rate",
    "event_id_reuse_rate",
    "classified_conflicting_event_rate",
    "recorded_rate_available",
    "end_to_end_coverage_available",
)

_AUTHORITY_STATUS_COLUMNS = (
    "authority_revision",
    "business_reports_authoritative",
    "registered_release_count",
    "registered_question_count",
    "registered_choice_count",
    "detail_revision",
    "detail_reports_authoritative",
)

_REGISTRY_QUALITY_COLUMNS = (
    "registered_release_count",
    "registered_question_count",
    "registered_choice_count",
    "registry_available",
    "raw_event_count",
    "authoritative_event_count",
    "excluded_event_count",
    "missing_release_event_count",
    "unknown_release_event_count",
    "question_not_in_release_event_count",
    "raw_answer_count",
    "authoritative_answer_count",
    "unresolved_answer_count",
    "invalid_selected_letter_answer_count",
    "selected_candidate_mismatch_answer_count",
    "unmatched_comment_count",
    "unmatched_proposal_count",
    "client_context_mismatch_event_count",
    "client_correctness_mismatch_answer_count",
    "registry_match_rate",
    "answer_resolution_rate",
)

_SURPRISE_QUESTION_COLUMNS = (
    "question_id",
    "question_version",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "answered_attempt_count",
    "rating_count",
    "surprised_count",
    "not_surprised_count",
    "rating_coverage_rate",
    "observed_surprise_rate",
    "posterior_mean",
    "first_rating_at",
    "last_rating_at",
)

_SURPRISE_QUALITY_COLUMNS = (
    "raw_reaction_count",
    "valid_reaction_count",
    "orphan_reaction_count",
    "duplicate_reaction_count",
    "registry_unmatched_reaction_count",
    "invalid_payload_reaction_count",
    "missing_prior_answer_reaction_count",
    "unknown_release_reaction_count",
    "counts_conserved",
    "orphan_breakdown_conserved",
)

_EVENT_RESOLUTION_COLUMNS = (
    "event_id",
    "event_type",
    "occurred_at",
    "received_at",
    "session_id",
    "attempt_id",
    "client_release_id",
    "registry_status",
    "answer_status",
    "registry_id",
    "release_id",
    "question_id",
    "question_version",
    "family",
    "dataset_id",
    "question_type",
    "selected_letter",
    "client_selected_candidate_id",
    "selected_candidate_id",
    "authoritative_is_correct",
    "client_is_correct",
    "client_context_mismatch",
    "client_correctness_mismatch",
)


_SESSION_COLUMNS = (
    "session_id",
    "attempt_id",
    "started_at",
    "last_event_at",
    "first_received_at",
    "last_received_at",
    "release_ids",
    "families",
    "question_types",
    "event_count",
    "question_count",
    "answer_count",
    "known_answer_count",
    "correct_answer_count",
    "incorrect_answer_count",
    "unknown_answer_count",
    "accuracy",
    "proposal_count",
    "rejected_setting_count",
    "completed_run_count",
    "failed_run_count",
    "comment_count",
)
_QUESTION_COLUMNS = (
    "question_id",
    "question_version",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "first_event_at",
    "last_event_at",
    "event_count",
    "session_count",
    "attempt_count",
    "solve_attempt_count",
    "answered_attempt_count",
    "answer_count",
    "known_answer_count",
    "correct_answer_count",
    "incorrect_answer_count",
    "unknown_answer_count",
    "accuracy",
    "proposal_count",
    "rejected_setting_count",
    "completed_run_count",
    "failed_run_count",
    "comment_count",
    "attempts_with_proposal",
    "proposal_usage_rate",
)
_ANSWER_COLUMNS = (
    "event_id",
    "occurred_at",
    "received_at",
    "session_id",
    "attempt_id",
    "question_id",
    "question_version",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "selected_letter",
    "client_selected_candidate_id",
    "selected_candidate_id",
    "answer_status",
    "is_correct",
    "client_is_correct",
    "client_context_mismatch",
    "client_correctness_mismatch",
)
_PROPOSAL_COLUMNS = (
    "event_id",
    "occurred_at",
    "received_at",
    "session_id",
    "attempt_id",
    "question_id",
    "question_version",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "setting_status",
    "label",
    "setting_json",
    "inherited_from_json",
    "n_seeds",
    "base_seed",
    "error_type",
)
_COMMENT_COLUMNS = (
    "event_id",
    "occurred_at",
    "received_at",
    "session_id",
    "attempt_id",
    "question_id",
    "question_version",
    "release_id",
    "family",
    "question_type",
    "category",
    "comment_text",
)

BUSINESS_SNAPSHOT_VIEW = "feedback_report_business_snapshot"
SURPRISE_QUESTIONS_VIEW = "feedback_report_surprise_questions"
SURPRISE_QUALITY_VIEW = "feedback_report_surprise_quality"
BUSINESS_REPORT_VIEWS = (
    "feedback_report_summary",
    "feedback_report_sessions",
    "feedback_report_questions",
    "feedback_report_answers",
    "feedback_report_proposals",
    "feedback_report_comments",
)
_BUSINESS_SNAPSHOT_COLUMNS = (
    "snapshot_revision",
    "snapshot_at",
    "authority_revision",
    "business_reports_authoritative",
    "registered_release_count",
    "registered_question_count",
    "registered_choice_count",
    "detail_revision",
    "detail_reports_authoritative",
    "pages_json",
)

_COMMON_FILTERS = frozenset(
    {
        "release_id",
        "family",
        "question_type",
        "question_id",
        "session_id",
        "attempt_id",
        "from",
        "to",
    }
)

_VIEW_SPECS = {
    "feedback_report_summary": _ViewSpec(
        columns=_SUMMARY_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_ingestion_summary": _ViewSpec(
        columns=_INGESTION_SUMMARY_COLUMNS,
        filters=frozenset({"from", "to", "request_id"}),
    ),
    "feedback_report_authority_status": _ViewSpec(
        columns=_AUTHORITY_STATUS_COLUMNS,
        filters=frozenset(),
    ),
    "feedback_report_business_snapshot": _ViewSpec(
        columns=_BUSINESS_SNAPSHOT_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_registry_quality": _ViewSpec(
        columns=_REGISTRY_QUALITY_COLUMNS,
        filters=frozenset({"from", "to"}),
    ),
    "feedback_report_surprise_questions": _ViewSpec(
        columns=_SURPRISE_QUESTION_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_surprise_quality": _ViewSpec(
        columns=_SURPRISE_QUALITY_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_event_resolution": _ViewSpec(
        columns=_EVENT_RESOLUTION_COLUMNS,
        filters=frozenset({"event_id"}),
    ),
    "feedback_report_sessions": _ViewSpec(
        columns=_SESSION_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_questions": _ViewSpec(
        columns=_QUESTION_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_answers": _ViewSpec(
        columns=_ANSWER_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_proposals": _ViewSpec(
        columns=_PROPOSAL_COLUMNS,
        filters=_COMMON_FILTERS,
    ),
    "feedback_report_comments": _ViewSpec(
        columns=_COMMENT_COLUMNS,
        filters=_COMMON_FILTERS | {"category"},
    ),
}

REPORT_VIEWS = tuple(_VIEW_SPECS)
_SINGLE_ROW_VIEWS = frozenset(
    {
        "feedback_report_summary",
        "feedback_report_ingestion_summary",
        "feedback_report_authority_status",
        "feedback_report_business_snapshot",
        "feedback_report_registry_quality",
        "feedback_report_surprise_quality",
        "feedback_report_event_resolution",
    }
)


class ReportsError(Exception):
    """Base exception for report configuration, queries, and transport."""


class ReportsConfigurationError(ReportsError, ValueError):
    """Raised when report access configuration is invalid."""


class ReportsNotConfiguredError(ReportsConfigurationError):
    """Raised when a request is attempted without both URL and read token."""


class ReportsQueryError(ReportsError, ValueError):
    """Raised when a query falls outside the report allowlists."""


class ReportsResponseError(ReportsError, ValueError):
    """Raised when the report endpoint returns an invalid envelope or rows."""


class ReportsRequestError(ReportsError):
    """A token-safe HTTP or network failure."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: int | None = None,
        response_excerpt: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_excerpt = response_excerpt
        details = [message, f"endpoint={endpoint}"]
        if status_code is not None:
            details.append(f"status={status_code}")
        if response_excerpt:
            details.append(f"response={response_excerpt}")
        super().__init__("; ".join(details))


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the report Bearer token on the configured origin only."""

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


def _open_report_request(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open one request without following redirects that may leak auth."""
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout)


def _clean_optional_string(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReportsConfigurationError(f"{field_name} must be a string")
    cleaned = value.strip()
    return cleaned or None


def _validate_timeout(value: float | str | None) -> float:
    if value is None or value == "":
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, bool):
        raise ReportsConfigurationError("reports timeout must be a number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportsConfigurationError("reports timeout must be a number") from exc
    if (
        not math.isfinite(timeout)
        or timeout < MIN_TIMEOUT_SECONDS
        or timeout > MAX_TIMEOUT_SECONDS
    ):
        raise ReportsConfigurationError(
            "reports timeout must be between "
            f"{MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds"
        )
    return timeout


def _validate_base_url(value: str | None) -> str | None:
    url = _clean_optional_string(value, field_name="reports URL")
    if url is None:
        return None
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ReportsConfigurationError(
            "reports URL cannot contain whitespace or control characters"
        )
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise ReportsConfigurationError("reports URL is invalid") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ReportsConfigurationError("reports URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ReportsConfigurationError("reports URL cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ReportsConfigurationError(
            "reports URL cannot contain query parameters or a fragment"
        )
    netloc = hostname
    if ":" in hostname and not hostname.startswith("["):
        netloc = f"[{hostname}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((scheme, netloc, path, "", ""))


def _validate_read_token(value: str | None) -> str | None:
    token = _clean_optional_string(value, field_name="reports read token")
    if token is not None and any(character.isspace() for character in token):
        raise ReportsConfigurationError("reports read token cannot contain whitespace")
    return token


def _redact(value: str, secret: str | None) -> str:
    redacted = value
    if secret:
        for candidate in {secret, urllib.parse.quote(secret, safe="")}:
            if candidate:
                redacted = redacted.replace(candidate, "[REDACTED]")
    return redacted


def _safe_endpoint(url: str, token: str | None) -> str:
    parsed = urllib.parse.urlsplit(url)
    endpoint = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )
    return _redact(endpoint, token)


@dataclass(frozen=True)
class ReportsConfig:
    """Protected report Edge Function URL and server-side read token."""

    url: str | None
    read_token: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _validate_base_url(self.url))
        object.__setattr__(
            self,
            "read_token",
            _validate_read_token(self.read_token),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _validate_timeout(self.timeout_seconds),
        )

    @classmethod
    def from_sources(
        cls,
        *,
        url: str | None = None,
        read_token: str | None = None,
        timeout_seconds: float | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ReportsConfig":
        env = os.environ if environ is None else environ
        resolved_url = url if url is not None else env.get(REPORTS_URL_ENV)
        resolved_token = (
            read_token if read_token is not None else env.get(REPORTS_READ_TOKEN_ENV)
        )
        resolved_timeout: float | str | None = timeout_seconds
        if resolved_timeout is None:
            resolved_timeout = env.get(REPORTS_TIMEOUT_ENV)
        return cls(
            url=resolved_url,
            read_token=resolved_token,
            timeout_seconds=_validate_timeout(resolved_timeout),
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ReportsConfig":
        return cls.from_sources(environ=environ)

    @property
    def is_configured(self) -> bool:
        return self.url is not None and self.read_token is not None

    def require_configured(self) -> tuple[str, str]:
        if self.url is None or self.read_token is None:
            raise ReportsNotConfiguredError(
                "feedback reports are not configured; set "
                f"{REPORTS_URL_ENV} and {REPORTS_READ_TOKEN_ENV}"
            )
        return self.url, self.read_token

    def __repr__(self) -> str:
        safe_url = (
            _safe_endpoint(self.url, self.read_token) if self.url is not None else None
        )
        return (
            f"ReportsConfig(url={safe_url!r}, timeout_seconds={self.timeout_seconds!r})"
        )


def _view_spec(view: str) -> _ViewSpec:
    if not isinstance(view, str) or view not in _VIEW_SPECS:
        allowed = ", ".join(REPORT_VIEWS)
        raise ReportsQueryError(
            f"unsupported report view {view!r}; choose one of: {allowed}"
        )
    return _VIEW_SPECS[view]


def _contains_unicode_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_LIMIT
    ):
        raise ReportsQueryError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    return limit


def _validate_offset(offset: int) -> int:
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset > MAX_OFFSET
    ):
        raise ReportsQueryError(f"offset must be an integer between 0 and {MAX_OFFSET}")
    return offset


def _filter_value(column: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ReportsQueryError(f"filter {column!r} must be a non-empty string")
    if (
        value != value.strip()
        or len(value) > MAX_FILTER_LENGTH
        or _contains_unicode_surrogate(value)
        or any(ord(character) < 32 for character in value)
    ):
        raise ReportsQueryError(
            "filter strings must be trimmed, at most "
            f"{MAX_FILTER_LENGTH} characters, and contain no controls"
        )
    if column in {"from", "to"}:
        if not _RFC3339_PATTERN.fullmatch(value):
            raise ReportsQueryError(f"filter {column!r} must be an RFC 3339 timestamp")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReportsQueryError(
                f"filter {column!r} must be an RFC 3339 timestamp"
            ) from exc
    if column == "request_id" and not _UUID_PATTERN.fullmatch(value):
        raise ReportsQueryError("filter 'request_id' must be a UUID")
    if column == "category" and value not in _COMMENT_CATEGORIES:
        raise ReportsQueryError(f"unsupported comment category {value!r}")
    return value


def _query_parameters(
    view: str,
    *,
    filters: Mapping[str, Any] | None,
    limit: int,
    offset: int,
) -> list[tuple[str, str]]:
    spec = _view_spec(view)
    if filters is not None and not isinstance(filters, Mapping):
        raise ReportsQueryError("filters must be a mapping")

    resolved_filters = filters or {}
    for column in resolved_filters:
        if not isinstance(column, str) or column not in spec.filters:
            allowed = ", ".join(sorted(spec.filters))
            raise ReportsQueryError(
                f"unsupported filter {column!r} for {view}; choose one of: {allowed}"
            )
    if view == "feedback_report_event_resolution" and set(resolved_filters) != {
        "event_id"
    }:
        raise ReportsQueryError(
            "feedback_report_event_resolution requires exactly one event_id filter"
        )
    parameters = [("view", view)]
    for column in sorted(resolved_filters):
        parameters.append((column, _filter_value(column, resolved_filters[column])))
    if "from" in resolved_filters and "to" in resolved_filters:
        start = datetime.fromisoformat(resolved_filters["from"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(resolved_filters["to"].replace("Z", "+00:00"))
        if start >= end:
            raise ReportsQueryError("filter 'from' must be earlier than 'to'")
    parameters.append(("limit", str(_validate_limit(limit))))
    resolved_offset = _validate_offset(offset)
    if view in _SINGLE_ROW_VIEWS and resolved_offset != 0:
        raise ReportsQueryError("single-row report offset must be zero")
    parameters.append(("offset", str(resolved_offset)))
    return parameters


def _ingestion_timestamp(value: Any, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _RFC3339_PATTERN.fullmatch(value):
        raise ReportsResponseError(
            f"ingestion summary {field_name} must be RFC 3339 or null"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReportsResponseError(
            f"ingestion summary {field_name} must be RFC 3339 or null"
        ) from exc


def _ingestion_rate(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportsResponseError(
            f"ingestion summary {field_name} must be a ratio or null"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or not 0 <= resolved <= 1:
        raise ReportsResponseError(
            f"ingestion summary {field_name} must be between zero and one"
        )
    return resolved


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return float(
        (Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )
    )


def _validate_ingestion_summary_row(row: Mapping[str, Any], *, index: int) -> None:
    count_fields = (
        "recorded_request_count",
        "success_request_count",
        "client_rejection_count",
        "service_failure_count",
        "event_id_conflict_request_count",
        "accepted_event_count",
        "duplicate_event_count",
        "idempotent_duplicate_event_count",
        "unclassified_duplicate_event_count",
        "conflicting_event_count",
        "conflict_audit_event_count",
        "event_id_reuse_count",
        "classified_event_count",
        "known_event_result_count",
    )
    counts = {
        field_name: _response_integer(
            row[field_name],
            field_name=f"row {index} {field_name}",
            minimum=0,
        )
        for field_name in count_fields
    }
    recorded = counts["recorded_request_count"]
    categorized = (
        counts["success_request_count"]
        + counts["client_rejection_count"]
        + counts["service_failure_count"]
    )
    if categorized != recorded:
        raise ReportsResponseError(
            "ingestion summary request classes do not add up to recorded requests"
        )
    conflict_requests = counts["event_id_conflict_request_count"]
    if conflict_requests > counts["client_rejection_count"]:
        raise ReportsResponseError(
            "ingestion summary conflict requests exceed client rejections"
        )

    duplicate_events = counts["duplicate_event_count"]
    idempotent_duplicates = counts["idempotent_duplicate_event_count"]
    unclassified_duplicates = counts["unclassified_duplicate_event_count"]
    conflicting_events = counts["conflicting_event_count"]
    conflict_audit_events = counts["conflict_audit_event_count"]
    if duplicate_events != idempotent_duplicates + unclassified_duplicates:
        raise ReportsResponseError(
            "ingestion summary duplicate classifications do not add up"
        )
    if (conflict_requests == 0) != (conflicting_events == 0) or (
        conflicting_events < conflict_requests
    ):
        raise ReportsResponseError(
            "ingestion summary conflicting events are inconsistent with requests"
        )
    if conflict_audit_events != conflicting_events:
        raise ReportsResponseError(
            "ingestion summary conflict audit does not match classified conflicts"
        )
    expected_reuse = duplicate_events + conflicting_events
    if counts["event_id_reuse_count"] != expected_reuse:
        raise ReportsResponseError(
            "ingestion summary event-ID reuse counts do not add up"
        )

    known_events = counts["accepted_event_count"] + counts["duplicate_event_count"]
    if counts["known_event_result_count"] != known_events:
        raise ReportsResponseError("ingestion summary event counts do not add up")
    success_requests = counts["success_request_count"]
    classified_result_requests = success_requests + conflict_requests
    classified_events = counts["classified_event_count"]
    observed_classified_results = known_events + conflicting_events
    if (
        known_events < success_requests
        or ((classified_result_requests == 0) != (classified_events == 0))
        or classified_events < classified_result_requests
        or (classified_events < observed_classified_results)
    ):
        raise ReportsResponseError(
            "ingestion summary event results are inconsistent with classified requests"
        )

    first_started = _ingestion_timestamp(
        row["first_started_at"], field_name="first_started_at"
    )
    last_finished = _ingestion_timestamp(
        row["last_finished_at"], field_name="last_finished_at"
    )
    if recorded == 0:
        if first_started is not None or last_finished is not None:
            raise ReportsResponseError(
                "empty ingestion summary must not contain request timestamps"
            )
    elif (
        first_started is None or last_finished is None or first_started > last_finished
    ):
        raise ReportsResponseError(
            "non-empty ingestion summary has inconsistent request timestamps"
        )

    failure_rate = _ingestion_rate(
        row["request_failure_rate"], field_name="request_failure_rate"
    )
    duplicate_rate = _ingestion_rate(
        row["duplicate_event_rate"], field_name="duplicate_event_rate"
    )
    reuse_rate = _ingestion_rate(
        row["event_id_reuse_rate"], field_name="event_id_reuse_rate"
    )
    conflict_rate = _ingestion_rate(
        row["classified_conflicting_event_rate"],
        field_name="classified_conflicting_event_rate",
    )
    expected_failure_rate = (
        None
        if recorded == 0
        else _rounded_ratio(
            counts["client_rejection_count"] + counts["service_failure_count"],
            recorded,
        )
    )
    expected_duplicate_rate = (
        None if known_events == 0 else _rounded_ratio(duplicate_events, known_events)
    )
    expected_reuse_rate = (
        None
        if classified_events == 0
        else _rounded_ratio(expected_reuse, classified_events)
    )
    classified_duplicate_results = idempotent_duplicates + conflicting_events
    expected_conflict_rate = (
        None
        if classified_duplicate_results == 0
        else _rounded_ratio(conflicting_events, classified_duplicate_results)
    )
    for field_name, actual, expected in (
        ("request_failure_rate", failure_rate, expected_failure_rate),
        ("duplicate_event_rate", duplicate_rate, expected_duplicate_rate),
        ("event_id_reuse_rate", reuse_rate, expected_reuse_rate),
        (
            "classified_conflicting_event_rate",
            conflict_rate,
            expected_conflict_rate,
        ),
    ):
        if actual is None or expected is None:
            if actual is not expected:
                raise ReportsResponseError(
                    f"ingestion summary {field_name} has the wrong nullability"
                )
        elif not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ReportsResponseError(
                f"ingestion summary {field_name} does not match its counts"
            )

    recorded_available = row["recorded_rate_available"]
    end_to_end_available = row["end_to_end_coverage_available"]
    if not isinstance(recorded_available, bool) or recorded_available != (recorded > 0):
        raise ReportsResponseError(
            "ingestion summary recorded_rate_available does not match its denominator"
        )
    if end_to_end_available is not False:
        raise ReportsResponseError(
            "ingestion summary must not claim end-to-end coverage"
        )


def _registry_rate(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportsResponseError(
            f"registry quality {field_name} must be a ratio or null"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or not 0 <= resolved <= 1:
        raise ReportsResponseError(
            f"registry quality {field_name} must be between zero and one"
        )
    return resolved


def _validate_authority_status_row(row: Mapping[str, Any], *, index: int) -> None:
    if row["authority_revision"] != "registry_v1":
        raise ReportsResponseError(
            f"row {index} authority status revision is not registry_v1"
        )
    if row["business_reports_authoritative"] is not True:
        raise ReportsResponseError(
            f"row {index} authority status does not prove the business cutover"
        )
    if row["detail_revision"] != "detail_v1":
        raise ReportsResponseError(
            f"row {index} authority status revision is not detail_v1"
        )
    if row["detail_reports_authoritative"] is not True:
        raise ReportsResponseError(
            f"row {index} authority status does not prove the detail-report cutover"
        )
    counts = {
        field_name: _response_integer(
            row[field_name],
            field_name=f"row {index} {field_name}",
            minimum=0,
        )
        for field_name in (
            "registered_release_count",
            "registered_question_count",
            "registered_choice_count",
        )
    }
    releases = counts["registered_release_count"]
    questions = counts["registered_question_count"]
    choices = counts["registered_choice_count"]
    if releases == 0:
        if questions != 0 or choices != 0:
            raise ReportsResponseError(
                "empty authority status cannot contain questions or choices"
            )
    elif questions < releases or choices < 2 * questions:
        raise ReportsResponseError(
            "authority status release/question/choice counts are inconsistent"
        )


def _validate_registry_quality_row(row: Mapping[str, Any], *, index: int) -> None:
    count_fields = (
        "registered_release_count",
        "registered_question_count",
        "registered_choice_count",
        "raw_event_count",
        "authoritative_event_count",
        "excluded_event_count",
        "missing_release_event_count",
        "unknown_release_event_count",
        "question_not_in_release_event_count",
        "raw_answer_count",
        "authoritative_answer_count",
        "unresolved_answer_count",
        "invalid_selected_letter_answer_count",
        "selected_candidate_mismatch_answer_count",
        "unmatched_comment_count",
        "unmatched_proposal_count",
        "client_context_mismatch_event_count",
        "client_correctness_mismatch_answer_count",
    )
    counts = {
        field_name: _response_integer(
            row[field_name],
            field_name=f"row {index} {field_name}",
            minimum=0,
        )
        for field_name in count_fields
    }

    releases = counts["registered_release_count"]
    questions = counts["registered_question_count"]
    choices = counts["registered_choice_count"]
    available = row["registry_available"]
    if not isinstance(available, bool) or available != (releases > 0):
        raise ReportsResponseError(
            "registry quality availability does not match registered releases"
        )
    if releases == 0:
        if questions != 0 or choices != 0:
            raise ReportsResponseError(
                "empty registry quality cannot contain questions or choices"
            )
    elif questions < releases or choices < 2 * questions:
        raise ReportsResponseError(
            "registry quality release/question/choice counts are inconsistent"
        )

    raw_events = counts["raw_event_count"]
    authoritative_events = counts["authoritative_event_count"]
    excluded_events = counts["excluded_event_count"]
    if authoritative_events + excluded_events != raw_events:
        raise ReportsResponseError(
            "registry quality event classifications do not add up"
        )
    unresolved_registry_events = (
        counts["missing_release_event_count"]
        + counts["unknown_release_event_count"]
        + counts["question_not_in_release_event_count"]
    )
    if unresolved_registry_events != excluded_events:
        raise ReportsResponseError(
            "registry quality unresolved registry statuses do not add up"
        )

    raw_answers = counts["raw_answer_count"]
    authoritative_answers = counts["authoritative_answer_count"]
    unresolved_answers = counts["unresolved_answer_count"]
    if authoritative_answers + unresolved_answers != raw_answers:
        raise ReportsResponseError(
            "registry quality answer classifications do not add up"
        )
    if (
        counts["invalid_selected_letter_answer_count"]
        + counts["selected_candidate_mismatch_answer_count"]
        > unresolved_answers
    ):
        raise ReportsResponseError(
            "registry quality answer mismatch counts exceed unresolved answers"
        )
    if (
        counts["unmatched_comment_count"] + counts["unmatched_proposal_count"]
        > excluded_events
    ):
        raise ReportsResponseError(
            "registry quality unmatched feedback counts exceed excluded events"
        )
    if counts["client_context_mismatch_event_count"] > authoritative_events:
        raise ReportsResponseError(
            "registry quality context mismatches exceed authoritative events"
        )
    if counts["client_correctness_mismatch_answer_count"] > authoritative_answers:
        raise ReportsResponseError(
            "registry quality correctness mismatches exceed authoritative answers"
        )

    registry_rate = _registry_rate(
        row["registry_match_rate"], field_name="registry_match_rate"
    )
    answer_rate = _registry_rate(
        row["answer_resolution_rate"], field_name="answer_resolution_rate"
    )
    expected_registry_rate = (
        None if raw_events == 0 else _rounded_ratio(authoritative_events, raw_events)
    )
    expected_answer_rate = (
        None if raw_answers == 0 else _rounded_ratio(authoritative_answers, raw_answers)
    )
    for field_name, actual, expected in (
        ("registry_match_rate", registry_rate, expected_registry_rate),
        ("answer_resolution_rate", answer_rate, expected_answer_rate),
    ):
        if actual is None or expected is None:
            if actual is not expected:
                raise ReportsResponseError(
                    f"registry quality {field_name} has the wrong nullability"
                )
        elif not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ReportsResponseError(
                f"registry quality {field_name} does not match its counts"
            )


def _resolution_string(
    value: Any,
    *,
    field_name: str,
    optional: bool = False,
    context: str = "event resolution",
) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_FILTER_LENGTH
        or _contains_unicode_surrogate(value)
        or any(ord(character) < 32 for character in value)
    ):
        qualifier = "a safe string or null" if optional else "a safe string"
        raise ReportsResponseError(f"{context} {field_name} must be {qualifier}")
    return value


def _detail_timestamp(value: Any, *, field_name: str, context: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_PATTERN.fullmatch(value):
        raise ReportsResponseError(f"{context} {field_name} must be RFC 3339")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReportsResponseError(f"{context} {field_name} must be RFC 3339") from exc


def _validate_detail_identity(
    row: Mapping[str, Any],
    *,
    index: int,
    detail_name: str,
) -> None:
    context = f"{detail_name} row {index}"
    for field_name in (
        "event_id",
        "session_id",
        "question_id",
        "release_id",
        "question_version",
        "family",
        "dataset_id",
        "question_type",
    ):
        _resolution_string(
            row[field_name],
            field_name=field_name,
            context=context,
        )
    _resolution_string(
        row["attempt_id"],
        field_name="attempt_id",
        optional=True,
        context=context,
    )
    for field_name in ("occurred_at", "received_at"):
        _detail_timestamp(
            row[field_name],
            field_name=field_name,
            context=context,
        )
    if not _RELEASE_ID_PATTERN.fullmatch(row["release_id"]):
        raise ReportsResponseError(f"{context} release_id is not canonical")
    if not _QUESTION_VERSION_PATTERN.fullmatch(row["question_version"]):
        raise ReportsResponseError(f"{context} question_version is not canonical")


def _validate_answer_detail_row(row: Mapping[str, Any], *, index: int) -> None:
    _validate_detail_identity(row, index=index, detail_name="answer detail")
    context = f"answer detail row {index}"
    selected_letter = _resolution_string(
        row["selected_letter"],
        field_name="selected_letter",
        optional=True,
        context=context,
    )
    client_candidate = _resolution_string(
        row["client_selected_candidate_id"],
        field_name="client_selected_candidate_id",
        optional=True,
        context=context,
    )
    selected_candidate = _resolution_string(
        row["selected_candidate_id"],
        field_name="selected_candidate_id",
        optional=True,
        context=context,
    )
    answer_status = row["answer_status"]
    if answer_status not in {
        "resolved",
        "invalid_selected_letter",
        "selected_candidate_mismatch",
    }:
        raise ReportsResponseError(f"{context} answer_status is invalid")

    authoritative_correct = row["is_correct"]
    client_correct = row["client_is_correct"]
    if authoritative_correct is not None and not isinstance(
        authoritative_correct, bool
    ):
        raise ReportsResponseError(f"{context} is_correct must be boolean or null")
    if client_correct is not None and not isinstance(client_correct, bool):
        raise ReportsResponseError(
            f"{context} client_is_correct must be boolean or null"
        )
    for mismatch_field in (
        "client_context_mismatch",
        "client_correctness_mismatch",
    ):
        if not isinstance(row[mismatch_field], bool):
            raise ReportsResponseError(f"{context} {mismatch_field} must be a boolean")

    expected_correctness_mismatch = (
        isinstance(authoritative_correct, bool)
        and isinstance(client_correct, bool)
        and authoritative_correct != client_correct
    )
    if row["client_correctness_mismatch"] != expected_correctness_mismatch:
        raise ReportsResponseError(
            f"{context} correctness mismatch does not match its authority facts"
        )

    if answer_status == "resolved":
        if (
            selected_letter is None
            or selected_candidate is None
            or not isinstance(authoritative_correct, bool)
            or (client_candidate is not None and client_candidate != selected_candidate)
        ):
            raise ReportsResponseError(
                f"{context} resolved status has inconsistent authority facts"
            )
    elif answer_status == "invalid_selected_letter":
        if selected_candidate is not None or authoritative_correct is not None:
            raise ReportsResponseError(
                f"{context} invalid-letter status has authority-only facts"
            )
    elif (
        selected_letter is None
        or selected_candidate is None
        or client_candidate is None
        or client_candidate == selected_candidate
        or authoritative_correct is not None
    ):
        raise ReportsResponseError(
            f"{context} candidate-mismatch status has inconsistent facts"
        )


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError(f"duplicate JSON object key {key!r}")
        resolved[key] = value
    return resolved


def _validate_json_interoperability(value: Any, *, field_name: str) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str):
            if _contains_unicode_surrogate(item):
                raise ReportsResponseError(
                    f"{field_name} cannot contain Unicode surrogate code points"
                )
            continue
        if isinstance(item, int):
            if not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER:
                raise ReportsResponseError(
                    f"{field_name} contains an unsafe integer-valued JSON number"
                )
            continue
        if isinstance(item, float):
            if not math.isfinite(item) or (
                item.is_integer()
                and not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER
            ):
                raise ReportsResponseError(
                    f"{field_name} contains a non-finite or unsafe JSON number"
                )
            continue
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
            continue
        if isinstance(item, list):
            pending.extend(item)
            continue
        raise ReportsResponseError(f"{field_name} contains a non-JSON value")


def _validate_json_object_text(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReportsResponseError(f"{field_name} must be JSON object text or null")
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReportsResponseError(
            f"{field_name} must be strict JSON object text: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ReportsResponseError(f"{field_name} must encode a JSON object")
    _validate_json_interoperability(parsed, field_name=field_name)
    return parsed


def _validate_proposal_detail_row(row: Mapping[str, Any], *, index: int) -> None:
    _validate_detail_identity(row, index=index, detail_name="proposal detail")
    context = f"proposal detail row {index}"
    if row["setting_status"] not in {"proposed", "rejected"}:
        raise ReportsResponseError(f"{context} setting_status is invalid")

    for field_name in ("label", "error_type"):
        value = row[field_name]
        if value is not None and (
            not isinstance(value, str) or _contains_unicode_surrogate(value)
        ):
            raise ReportsResponseError(
                f"{context} {field_name} must be Unicode-safe text or null"
            )
    _validate_json_object_text(
        row["setting_json"],
        field_name=f"{context} setting_json",
    )
    _validate_json_object_text(
        row["inherited_from_json"],
        field_name=f"{context} inherited_from_json",
    )
    for field_name in ("n_seeds", "base_seed"):
        value = row[field_name]
        if value is not None:
            _response_integer(
                value,
                field_name=f"{context} {field_name}",
                minimum=MIN_POSTGRES_INTEGER,
                maximum=MAX_POSTGRES_INTEGER,
            )


def _surprise_ratio(
    value: Any,
    *,
    field_name: str,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "a ratio or null" if optional else "a ratio"
        raise ReportsResponseError(f"{field_name} must be {qualifier}")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0 <= resolved <= 1:
        raise ReportsResponseError(f"{field_name} must be between zero and one")
    return resolved


def _validate_surprise_question_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> None:
    context = f"surprise question row {index}"
    for field_name in (
        "question_id",
        "question_version",
        "release_id",
        "family",
        "dataset_id",
        "question_type",
    ):
        _resolution_string(
            row[field_name],
            field_name=field_name,
            context=context,
        )
    if not _RELEASE_ID_PATTERN.fullmatch(row["release_id"]):
        raise ReportsResponseError(f"{context} release_id is not canonical")
    if not _QUESTION_VERSION_PATTERN.fullmatch(row["question_version"]):
        raise ReportsResponseError(f"{context} question_version is not canonical")

    counts = {
        field_name: _response_integer(
            row[field_name],
            field_name=f"{context} {field_name}",
            minimum=1 if field_name == "answered_attempt_count" else 0,
        )
        for field_name in (
            "answered_attempt_count",
            "rating_count",
            "surprised_count",
            "not_surprised_count",
        )
    }
    answered = counts["answered_attempt_count"]
    ratings = counts["rating_count"]
    surprised = counts["surprised_count"]
    not_surprised = counts["not_surprised_count"]
    if ratings != surprised + not_surprised:
        raise ReportsResponseError(f"{context} rating counts do not add up")
    if ratings > answered:
        raise ReportsResponseError(f"{context} rating count exceeds answered attempts")

    coverage = _surprise_ratio(
        row["rating_coverage_rate"],
        field_name=f"{context} rating_coverage_rate",
    )
    observed = _surprise_ratio(
        row["observed_surprise_rate"],
        field_name=f"{context} observed_surprise_rate",
        optional=True,
    )
    posterior = _surprise_ratio(
        row["posterior_mean"],
        field_name=f"{context} posterior_mean",
    )
    expected_ratios = (
        ("rating_coverage_rate", coverage, _rounded_ratio(ratings, answered)),
        (
            "observed_surprise_rate",
            observed,
            None if ratings == 0 else _rounded_ratio(surprised, ratings),
        ),
        (
            "posterior_mean",
            posterior,
            _rounded_ratio(1 + surprised, 2 + ratings),
        ),
    )
    for field_name, actual, expected in expected_ratios:
        if actual is None or expected is None:
            if actual is not expected:
                raise ReportsResponseError(
                    f"{context} {field_name} has the wrong nullability"
                )
        elif not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ReportsResponseError(
                f"{context} {field_name} does not match its counts"
            )

    timestamps = {
        field_name: (
            None
            if row[field_name] is None
            else _detail_timestamp(
                row[field_name],
                field_name=field_name,
                context=context,
            )
        )
        for field_name in ("first_rating_at", "last_rating_at")
    }
    if ratings == 0:
        if any(value is not None for value in timestamps.values()):
            raise ReportsResponseError(
                f"{context} unrated row must not contain rating timestamps"
            )
    elif (
        timestamps["first_rating_at"] is None
        or timestamps["last_rating_at"] is None
        or timestamps["first_rating_at"] > timestamps["last_rating_at"]
    ):
        raise ReportsResponseError(
            f"{context} rating timestamps are missing or out of order"
        )


def _validate_surprise_quality_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> None:
    context = f"surprise quality row {index}"
    count_fields = (
        "raw_reaction_count",
        "valid_reaction_count",
        "orphan_reaction_count",
        "duplicate_reaction_count",
        "registry_unmatched_reaction_count",
        "invalid_payload_reaction_count",
        "missing_prior_answer_reaction_count",
        "unknown_release_reaction_count",
    )
    counts = {
        field_name: _response_integer(
            row[field_name],
            field_name=f"{context} {field_name}",
            minimum=0,
        )
        for field_name in count_fields
    }
    counts_conserved = row["counts_conserved"]
    orphan_conserved = row["orphan_breakdown_conserved"]
    if not isinstance(counts_conserved, bool) or not isinstance(orphan_conserved, bool):
        raise ReportsResponseError(f"{context} conservation flags must be booleans")
    expected_counts_conserved = counts["raw_reaction_count"] == (
        counts["valid_reaction_count"]
        + counts["orphan_reaction_count"]
        + counts["duplicate_reaction_count"]
    )
    expected_orphan_conserved = counts["orphan_reaction_count"] == (
        counts["registry_unmatched_reaction_count"]
        + counts["invalid_payload_reaction_count"]
        + counts["missing_prior_answer_reaction_count"]
    )
    if counts_conserved is not True or counts_conserved != expected_counts_conserved:
        raise ReportsResponseError(f"{context} raw classifications do not add up")
    if orphan_conserved is not True or orphan_conserved != expected_orphan_conserved:
        raise ReportsResponseError(f"{context} orphan classifications do not add up")
    if (
        counts["unknown_release_reaction_count"]
        > counts["registry_unmatched_reaction_count"]
    ):
        raise ReportsResponseError(
            f"{context} unknown releases exceed registry-unmatched reactions"
        )


def _validate_event_resolution_row(row: Mapping[str, Any], *, index: int) -> None:
    del index
    _resolution_string(row["event_id"], field_name="event_id")
    registry_status = row["registry_status"]
    answer_status = row["answer_status"]
    if registry_status == "not_found" or answer_status == "not_found":
        if registry_status != "not_found" or answer_status != "not_found":
            raise ReportsResponseError(
                "not-found event resolution statuses must appear together"
            )
        nullable_fields = set(_EVENT_RESOLUTION_COLUMNS) - {
            "event_id",
            "registry_status",
            "answer_status",
            "client_context_mismatch",
            "client_correctness_mismatch",
        }
        if any(row[field_name] is not None for field_name in nullable_fields):
            raise ReportsResponseError(
                "not-found event resolution cannot contain event or authority facts"
            )
        if (
            row["client_context_mismatch"] is not False
            or row["client_correctness_mismatch"] is not False
        ):
            raise ReportsResponseError(
                "not-found event resolution cannot claim client mismatches"
            )
        return

    event_type = _resolution_string(row["event_type"], field_name="event_type")
    if event_type not in _FEEDBACK_EVENT_TYPES:
        raise ReportsResponseError("event resolution event_type is invalid")
    _resolution_string(row["session_id"], field_name="session_id")
    _resolution_string(row["attempt_id"], field_name="attempt_id", optional=True)
    client_release = _resolution_string(
        row["client_release_id"],
        field_name="client_release_id",
        optional=True,
    )
    for timestamp_field in ("occurred_at", "received_at"):
        timestamp = row[timestamp_field]
        if not isinstance(timestamp, str) or not _RFC3339_PATTERN.fullmatch(timestamp):
            raise ReportsResponseError(
                f"event resolution {timestamp_field} must be RFC 3339"
            )
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReportsResponseError(
                f"event resolution {timestamp_field} must be RFC 3339"
            ) from exc

    if registry_status not in {
        "matched",
        "missing_release",
        "unknown_release",
        "question_not_in_release",
    }:
        raise ReportsResponseError("event resolution registry_status is invalid")
    if answer_status not in {
        "not_answer",
        "unresolved_registry",
        "invalid_selected_letter",
        "selected_candidate_mismatch",
        "resolved",
    }:
        raise ReportsResponseError("event resolution answer_status is invalid")

    registry_id = _resolution_string(
        row["registry_id"], field_name="registry_id", optional=True
    )
    if registry_id is not None and not _REGISTRY_ID_PATTERN.fullmatch(registry_id):
        raise ReportsResponseError("event resolution registry_id is invalid")
    canonical_fields = (
        "release_id",
        "question_id",
        "question_version",
        "family",
        "dataset_id",
        "question_type",
    )
    canonical = {
        field_name: _resolution_string(
            row[field_name], field_name=field_name, optional=True
        )
        for field_name in canonical_fields
    }
    if registry_status == "matched":
        if registry_id is None or any(value is None for value in canonical.values()):
            raise ReportsResponseError(
                "matched event resolution must contain authoritative identity"
            )
        if not _RELEASE_ID_PATTERN.fullmatch(str(canonical["release_id"])) or not (
            _QUESTION_VERSION_PATTERN.fullmatch(str(canonical["question_version"]))
        ):
            raise ReportsResponseError(
                "matched event resolution has an invalid authoritative version"
            )
    elif any(value is not None for value in canonical.values()):
        raise ReportsResponseError(
            "unmatched event resolution cannot contain authoritative identity"
        )
    if registry_status in {"missing_release", "unknown_release"} and (
        registry_id is not None
    ):
        raise ReportsResponseError(
            "unregistered release resolution cannot contain a registry id"
        )
    if registry_status == "missing_release" and client_release is not None:
        raise ReportsResponseError(
            "missing-release resolution cannot contain a client release id"
        )
    if registry_status != "missing_release" and client_release is None:
        raise ReportsResponseError(
            "non-missing release resolution must contain a client release id"
        )

    selected_letter = _resolution_string(
        row["selected_letter"], field_name="selected_letter", optional=True
    )
    _resolution_string(
        row["client_selected_candidate_id"],
        field_name="client_selected_candidate_id",
        optional=True,
    )
    selected_candidate = _resolution_string(
        row["selected_candidate_id"],
        field_name="selected_candidate_id",
        optional=True,
    )
    authoritative_correct = row["authoritative_is_correct"]
    client_correct = row["client_is_correct"]
    if authoritative_correct is not None and not isinstance(
        authoritative_correct, bool
    ):
        raise ReportsResponseError(
            "event resolution authoritative_is_correct must be boolean or null"
        )
    if client_correct is not None and not isinstance(client_correct, bool):
        raise ReportsResponseError(
            "event resolution client_is_correct must be boolean or null"
        )
    for mismatch_field in (
        "client_context_mismatch",
        "client_correctness_mismatch",
    ):
        if not isinstance(row[mismatch_field], bool):
            raise ReportsResponseError(
                f"event resolution {mismatch_field} must be a boolean"
            )
    if row["client_context_mismatch"] and registry_status != "matched":
        raise ReportsResponseError(
            "unmatched event resolution cannot claim a context mismatch"
        )
    expected_correctness_mismatch = (
        isinstance(authoritative_correct, bool)
        and isinstance(client_correct, bool)
        and authoritative_correct != client_correct
    )
    if row["client_correctness_mismatch"] != expected_correctness_mismatch:
        raise ReportsResponseError(
            "event resolution correctness mismatch does not match its facts"
        )

    is_answer = event_type == "answer_submitted"
    if not is_answer:
        if answer_status != "not_answer" or authoritative_correct is not None:
            raise ReportsResponseError(
                "non-answer event resolution has answer-only authority fields"
            )
    elif registry_status != "matched":
        if answer_status != "unresolved_registry" or authoritative_correct is not None:
            raise ReportsResponseError(
                "unmatched answer resolution must remain unresolved"
            )
    elif answer_status == "resolved":
        if (
            selected_letter is None
            or selected_candidate is None
            or not isinstance(authoritative_correct, bool)
        ):
            raise ReportsResponseError(
                "resolved answer must contain an authoritative choice and result"
            )
    elif answer_status == "invalid_selected_letter":
        if selected_candidate is not None or (authoritative_correct is not None):
            raise ReportsResponseError(
                "invalid-letter answer resolution has inconsistent authority fields"
            )
    elif answer_status == "selected_candidate_mismatch":
        if (
            selected_letter is None
            or selected_candidate is None
            or (authoritative_correct is not None)
        ):
            raise ReportsResponseError(
                "candidate-mismatch answer resolution has inconsistent fields"
            )
    else:
        raise ReportsResponseError(
            "matched answer resolution has an invalid answer status"
        )


def validate_report_rows(view: str, value: Any) -> list[dict[str, Any]]:
    """Validate and detach a JSON array returned for one allowlisted view."""
    spec = _view_spec(view)
    if not isinstance(value, list):
        raise ReportsResponseError("report response must be a JSON array")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        rows = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ReportsResponseError(
            f"report rows must contain only finite JSON values: {exc}"
        ) from exc

    allowed_columns = frozenset(spec.columns)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReportsResponseError(
                f"report response row {index} must be a JSON object"
            )
        missing = sorted(allowed_columns - row.keys())
        unknown = sorted(row.keys() - allowed_columns)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unknown:
                details.append(f"unexpected: {', '.join(unknown)}")
            raise ReportsResponseError(
                f"report response row {index} has invalid columns "
                f"({'; '.join(details)})"
            )
        if view == "feedback_report_ingestion_summary":
            _validate_ingestion_summary_row(row, index=index)
        elif view == "feedback_report_authority_status":
            _validate_authority_status_row(row, index=index)
        elif view == "feedback_report_registry_quality":
            _validate_registry_quality_row(row, index=index)
        elif view == SURPRISE_QUESTIONS_VIEW:
            _validate_surprise_question_row(row, index=index)
        elif view == SURPRISE_QUALITY_VIEW:
            _validate_surprise_quality_row(row, index=index)
        elif view == "feedback_report_event_resolution":
            _validate_event_resolution_row(row, index=index)
        elif view == "feedback_report_answers":
            _validate_answer_detail_row(row, index=index)
        elif view == "feedback_report_proposals":
            _validate_proposal_detail_row(row, index=index)
        elif view == BUSINESS_SNAPSHOT_VIEW:
            _validate_business_snapshot_row(row, index=index)
    return rows


@dataclass(frozen=True)
class ReportPage:
    """One validated page returned by the protected report endpoint."""

    view: str
    rows: tuple[dict[str, Any], ...]
    total: int
    limit: int
    offset: int
    request_id: str | None = None

    @property
    def is_complete(self) -> bool:
        """Whether this page contains every row matching the query."""
        return self.offset == 0 and self.total == len(self.rows)

    def rows_copy(self) -> list[dict[str, Any]]:
        return deepcopy(list(self.rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "rows": self.rows_copy(),
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class BusinessSnapshot:
    """One atomic, authority-attested snapshot of all six business views."""

    snapshot_revision: str
    snapshot_at: str
    authority_revision: str
    business_reports_authoritative: bool
    detail_revision: str
    detail_reports_authoritative: bool
    registered_release_count: int
    registered_question_count: int
    registered_choice_count: int
    pages: Mapping[str, ReportPage]
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        pages: dict[str, dict[str, Any]] = {}
        for view in BUSINESS_REPORT_VIEWS:
            page = self.pages[view]
            pages[view] = {
                "view": page.view,
                "rows": page.rows_copy(),
                "total": page.total,
                "limit": page.limit,
                "offset": page.offset,
            }
        return {
            "snapshot_revision": self.snapshot_revision,
            "snapshot_at": self.snapshot_at,
            "authority_revision": self.authority_revision,
            "business_reports_authoritative": self.business_reports_authoritative,
            "detail_revision": self.detail_revision,
            "detail_reports_authoritative": self.detail_reports_authoritative,
            "registered_release_count": self.registered_release_count,
            "registered_question_count": self.registered_question_count,
            "registered_choice_count": self.registered_choice_count,
            "pages": pages,
            "request_id": self.request_id,
        }


def _response_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = (
            f" between {minimum} and {maximum}"
            if maximum is not None
            else f" greater than or equal to {minimum}"
        )
        raise ReportsResponseError(
            f"report response {field_name} must be an integer{suffix}"
        )
    return value


def validate_report_response(
    view: str,
    value: Any,
    *,
    expected_limit: int | None = None,
    expected_offset: int | None = None,
) -> ReportPage:
    """Validate the protected Edge Function's report-page envelope."""
    _view_spec(view)
    if not isinstance(value, Mapping):
        raise ReportsResponseError("report response must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ReportsResponseError("report response fields must be strings")
    required = {"view", "rows", "total", "limit", "offset"}
    optional = {"request_id"}
    missing = sorted(required - value.keys())
    if missing:
        raise ReportsResponseError(
            f"report response is missing fields: {', '.join(missing)}"
        )
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        raise ReportsResponseError(
            f"report response contains unexpected fields: {', '.join(unknown)}"
        )
    if value["view"] != view:
        raise ReportsResponseError(
            f"report response view {value['view']!r} does not match {view!r}"
        )

    rows = validate_report_rows(view, value["rows"])
    total = _response_integer(value["total"], field_name="total", minimum=0)
    limit = _response_integer(
        value["limit"],
        field_name="limit",
        minimum=1,
        maximum=MAX_LIMIT,
    )
    offset = _response_integer(
        value["offset"],
        field_name="offset",
        minimum=0,
        maximum=MAX_OFFSET,
    )
    if len(rows) > limit:
        raise ReportsResponseError("report response contains more rows than limit")
    if total < len(rows):
        raise ReportsResponseError("report response total is smaller than rows")
    if view in _SINGLE_ROW_VIEWS and (len(rows) != 1 or total != 1 or offset != 0):
        raise ReportsResponseError(
            "single-row report must contain exactly one row with total one and offset zero"
        )
    if expected_limit is not None and limit != expected_limit:
        raise ReportsResponseError(
            "report response limit does not match the requested limit"
        )
    if expected_offset is not None and offset != expected_offset:
        raise ReportsResponseError(
            "report response offset does not match the requested offset"
        )

    request_id = value.get("request_id")
    if request_id is not None and (
        not isinstance(request_id, str)
        or not request_id
        or request_id.strip() != request_id
        or len(request_id) > 200
        or "\r" in request_id
        or "\n" in request_id
    ):
        raise ReportsResponseError(
            "report response request_id must be a safe non-empty string"
        )
    return ReportPage(
        view=view,
        rows=tuple(rows),
        total=total,
        limit=limit,
        offset=offset,
        request_id=request_id,
    )


def _validate_business_snapshot_metadata(
    row: Mapping[str, Any],
    *,
    index: int,
) -> None:
    if row["snapshot_revision"] != "business_snapshot_v1":
        raise ReportsResponseError(
            f"row {index} business snapshot revision is not business_snapshot_v1"
        )
    _detail_timestamp(
        row["snapshot_at"],
        field_name="snapshot_at",
        context=f"business snapshot row {index}",
    )
    _validate_authority_status_row(
        {
            "authority_revision": row["authority_revision"],
            "business_reports_authoritative": row["business_reports_authoritative"],
            "registered_release_count": row["registered_release_count"],
            "registered_question_count": row["registered_question_count"],
            "registered_choice_count": row["registered_choice_count"],
            "detail_revision": row["detail_revision"],
            "detail_reports_authoritative": row["detail_reports_authoritative"],
        },
        index=index,
    )


def _decode_business_snapshot_pages(value: Any) -> dict[str, ReportPage]:
    if not isinstance(value, str):
        raise ReportsResponseError(
            "business snapshot pages_json must be strict JSON object text"
        )
    try:
        decoded = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReportsResponseError(
            f"business snapshot pages_json must be strict JSON object text: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ReportsResponseError(
            "business snapshot pages_json must encode a JSON object"
        )
    _validate_json_interoperability(decoded, field_name="business snapshot pages_json")

    expected_views = frozenset(BUSINESS_REPORT_VIEWS)
    actual_views = frozenset(decoded)
    if actual_views != expected_views:
        missing = sorted(expected_views - actual_views)
        unknown = sorted(actual_views - expected_views)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unexpected: {', '.join(unknown)}")
        raise ReportsResponseError(
            f"business snapshot pages_json has invalid views ({'; '.join(details)})"
        )

    required_page_fields = {"view", "rows", "total", "limit", "offset"}
    pages: dict[str, ReportPage] = {}
    for view in BUSINESS_REPORT_VIEWS:
        envelope = decoded[view]
        if not isinstance(envelope, dict):
            raise ReportsResponseError(
                f"business snapshot page {view} must be a JSON object"
            )
        if set(envelope) != required_page_fields:
            missing = sorted(required_page_fields - envelope.keys())
            unknown = sorted(envelope.keys() - required_page_fields)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unknown:
                details.append(f"unexpected: {', '.join(unknown)}")
            raise ReportsResponseError(
                f"business snapshot page {view} has invalid fields "
                f"({'; '.join(details)})"
            )
        pages[view] = validate_report_response(
            view,
            envelope,
            expected_offset=0,
        )

    if len({page.limit for page in pages.values()}) != 1:
        raise ReportsResponseError(
            "business snapshot pages must use one common page limit"
        )
    return pages


def _validate_business_snapshot_conservation(
    pages: Mapping[str, ReportPage],
    *,
    release_filter_active: bool,
) -> None:
    summary_page = pages["feedback_report_summary"]
    summary = summary_page.rows[0]
    counts = {
        field_name: _response_integer(
            summary[field_name],
            field_name=f"business snapshot summary {field_name}",
            minimum=0,
        )
        for field_name in (
            "answer_count",
            "proposal_count",
            "rejected_setting_count",
            "comment_count",
            "attempt_count",
            "question_count",
        )
    }
    exact_totals = {
        "feedback_report_answers": counts["answer_count"],
        "feedback_report_proposals": (
            counts["proposal_count"] + counts["rejected_setting_count"]
        ),
        "feedback_report_comments": counts["comment_count"],
        "feedback_report_sessions": counts["attempt_count"],
    }
    for view, expected_total in exact_totals.items():
        if pages[view].total != expected_total:
            raise ReportsResponseError(
                f"business snapshot {view} total does not match summary counts"
            )

    question_total = pages["feedback_report_questions"].total
    question_count = counts["question_count"]
    if question_total < question_count or (
        release_filter_active and question_total != question_count
    ):
        qualifier = "equal" if release_filter_active else "not smaller than"
        raise ReportsResponseError(
            "business snapshot question total must be "
            f"{qualifier} the summary question count"
        )


def _validate_business_snapshot_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> None:
    _validate_business_snapshot_metadata(row, index=index)
    pages = _decode_business_snapshot_pages(row["pages_json"])
    _validate_business_snapshot_conservation(
        pages,
        release_filter_active=False,
    )


def _business_snapshot_from_page(
    page: ReportPage,
    *,
    release_filter_active: bool,
) -> BusinessSnapshot:
    if page.view != BUSINESS_SNAPSHOT_VIEW or len(page.rows) != 1:
        raise ReportsResponseError(
            "business snapshot outer page must contain exactly one snapshot row"
        )
    row = page.rows[0]
    pages = _decode_business_snapshot_pages(row["pages_json"])
    if any(inner.limit != page.limit for inner in pages.values()):
        raise ReportsResponseError(
            "business snapshot inner page limits do not match the outer response"
        )
    _validate_business_snapshot_conservation(
        pages,
        release_filter_active=release_filter_active,
    )
    return BusinessSnapshot(
        snapshot_revision=row["snapshot_revision"],
        snapshot_at=row["snapshot_at"],
        authority_revision=row["authority_revision"],
        business_reports_authoritative=row["business_reports_authoritative"],
        detail_revision=row["detail_revision"],
        detail_reports_authoritative=row["detail_reports_authoritative"],
        registered_release_count=row["registered_release_count"],
        registered_question_count=row["registered_question_count"],
        registered_choice_count=row["registered_choice_count"],
        pages=dict(pages),
        request_id=page.request_id,
    )


def validate_business_snapshot_response(
    value: Any,
    *,
    expected_limit: int | None = None,
    release_filter_active: bool = False,
) -> BusinessSnapshot:
    """Validate and detach one atomic six-view business snapshot envelope."""
    if not isinstance(release_filter_active, bool):
        raise TypeError("release_filter_active must be a boolean")
    page = validate_report_response(
        BUSINESS_SNAPSHOT_VIEW,
        value,
        expected_limit=expected_limit,
        expected_offset=0,
    )
    return _business_snapshot_from_page(
        page,
        release_filter_active=release_filter_active,
    )


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"invalid JSON constant {constant}")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def report_rows_to_csv(
    view: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Serialize validated rows with the view's stable, complete column order."""
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise ReportsResponseError("report rows must be a sequence")
    resolved_rows = validate_report_rows(view, list(rows))
    columns = _view_spec(view).columns
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in resolved_rows:
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return output.getvalue()


class ReportsClient:
    """GET allowlisted views from the protected feedback-report endpoint."""

    def __init__(
        self,
        config: ReportsConfig | None = None,
        *,
        url: str | None = None,
        read_token: str | None = None,
        timeout_seconds: float | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if config is not None and any(
            value is not None for value in (url, read_token, timeout_seconds)
        ):
            raise ReportsConfigurationError(
                "pass either config or explicit report parameters, not both"
            )
        self._config = config or ReportsConfig.from_sources(
            url=url,
            read_token=read_token,
            timeout_seconds=timeout_seconds,
            environ=environ,
        )

    @property
    def config(self) -> ReportsConfig:
        return self._config

    @property
    def is_configured(self) -> bool:
        return self._config.is_configured

    def build_query_url(
        self,
        view: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> str:
        """Return a safe GET URL containing no authentication material."""
        base_url, _ = self._config.require_configured()
        parameters = _query_parameters(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        return f"{base_url}?{urllib.parse.urlencode(parameters)}"

    def fetch_page(
        self,
        view: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> ReportPage:
        """Fetch and validate one metadata-preserving report page."""
        _, token = self._config.require_configured()
        endpoint = self.build_query_url(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        safe_endpoint = _safe_endpoint(endpoint, token)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "ArchitectureIQ-feedback-reports/1",
        }
        request = urllib.request.Request(
            endpoint,
            headers=headers,
            method="GET",
        )
        try:
            with _open_report_request(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ReportsResponseError(
                        f"report response exceeds {MAX_RESPONSE_BYTES} bytes"
                    )
                text = raw.decode("utf-8")
                if not 200 <= int(status) < 300:
                    raise ReportsRequestError(
                        "report server rejected the request",
                        endpoint=safe_endpoint,
                        status_code=int(status),
                        response_excerpt=_redact(text[:1_000], token),
                    )
                try:
                    value = json.loads(
                        text,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ReportsResponseError(
                        f"report response is not valid JSON: {exc}"
                    ) from exc
                return validate_report_response(
                    view,
                    value,
                    expected_limit=limit,
                    expected_offset=offset,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            response_text = _redact(
                raw.decode("utf-8", errors="replace")[:1_000],
                token,
            )
            raise ReportsRequestError(
                "report server rejected the request",
                endpoint=safe_endpoint,
                status_code=exc.code,
                response_excerpt=response_text,
            ) from None
        except (ReportsRequestError, ReportsResponseError):
            raise
        except UnicodeDecodeError as exc:
            raise ReportsResponseError(f"report response is not UTF-8: {exc}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            reason = _redact(str(getattr(exc, "reason", exc)), token)
            raise ReportsRequestError(
                f"report request could not reach the server: {reason}",
                endpoint=safe_endpoint,
            ) from None

    def fetch_business_snapshot(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> BusinessSnapshot:
        """Fetch all six business views in one validated, atomic GET response."""
        page = self.fetch_page(
            BUSINESS_SNAPSHOT_VIEW,
            filters=filters,
            limit=limit,
            offset=0,
        )
        return _business_snapshot_from_page(
            page,
            release_filter_active=filters is not None and "release_id" in filters,
        )

    def fetch_surprise_questions(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> ReportPage:
        """Fetch one strict, registry-authoritative surprise question page."""
        return self.fetch_page(
            SURPRISE_QUESTIONS_VIEW,
            filters=filters,
            limit=limit,
            offset=offset,
        )

    def fetch_surprise_quality(
        self,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> ReportPage:
        """Fetch the independent single-row surprise quality snapshot."""
        return self.fetch_page(
            SURPRISE_QUALITY_VIEW,
            filters=filters,
            limit=limit,
            offset=0,
        )

    def fetch_rows(
        self,
        view: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch a page and return a detached rows-only convenience value."""
        page = self.fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        return page.rows_copy()

    def fetch_csv(
        self,
        view: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> str:
        """Fetch and serialize only when the response is a complete result."""
        page = self.fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        if not page.is_complete:
            raise ReportsResponseError("cannot export CSV from a partial report page")
        return report_rows_to_csv(view, page.rows)


def copy_view_columns() -> dict[str, tuple[str, ...]]:
    """Return a detached public inventory for UI column selection."""
    return deepcopy({view: spec.columns for view, spec in _VIEW_SPECS.items()})
