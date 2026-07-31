"""Pure presentation helpers for the internal feedback Reports app."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import math
from typing import Any, Literal

from .client import REPORT_VIEWS, report_rows_to_csv


MISSING_VALUE = "—"
DataQualitySeverity = Literal["info", "warning", "error"]
_ARRAY_DISPLAY_COLUMNS = frozenset({"release_ids", "families", "question_types"})
_JSON_DISPLAY_COLUMNS = frozenset({"setting_json", "inherited_from_json"})
_MAX_JSON_DISPLAY_CHARS = 1_000
_PERCENTAGE_DISPLAY_COLUMNS = frozenset(
    {
        "accuracy",
        "proposal_usage_rate",
        "ingestion_failure_rate",
        "request_failure_rate",
        "duplicate_event_rate",
        "event_id_reuse_rate",
        "classified_conflicting_event_rate",
        "registry_match_rate",
        "answer_resolution_rate",
        "rating_coverage_rate",
        "observed_surprise_rate",
        "posterior_mean",
    }
)
_VIEW_EXPORT_NAMES = {
    "feedback_report_summary": "summary",
    "feedback_report_ingestion_summary": "ingestion-summary",
    "feedback_report_registry_quality": "registry-quality",
    "feedback_report_surprise_questions": "surprise-questions",
    "feedback_report_surprise_quality": "surprise-quality",
    "feedback_report_event_resolution": "event-resolution",
    "feedback_report_sessions": "sessions",
    "feedback_report_questions": "questions",
    "feedback_report_answers": "answers",
    "feedback_report_proposals": "proposals",
    "feedback_report_comments": "comments",
}


@dataclass(frozen=True)
class DataQualitySignal:
    """One deterministic, presentation-ready feedback data-quality finding."""

    code: str
    severity: DataQualitySeverity
    title: str
    body: str


def _quality_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _quality_count(
    row: Mapping[str, Any],
    key: str,
    *,
    row_name: str,
) -> int:
    if key not in row:
        raise ValueError(f"{row_name} is missing {key}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{row_name} {key} must be a non-negative integer")
    return value


def _quality_boolean(
    row: Mapping[str, Any],
    key: str,
    *,
    row_name: str,
) -> bool:
    if key not in row:
        raise ValueError(f"{row_name} is missing {key}")
    value = row[key]
    if not isinstance(value, bool):
        raise ValueError(f"{row_name} {key} must be a boolean")
    return value


def _quality_fraction(numerator: int, denominator: int, *, label: str) -> float | None:
    if numerator > denominator:
        raise ValueError(f"{label} numerator cannot exceed its denominator")
    return None if denominator == 0 else numerator / denominator


def build_data_quality_signals(
    business_summary: Mapping[str, Any],
    *,
    ingestion_summary: Mapping[str, Any] | None = None,
    ingestion_unavailable_reason: str | None = None,
    registry_summary: Mapping[str, Any] | None = None,
    registry_unavailable_reason: str | None = None,
    question_registry_available: bool | None = None,
) -> tuple[DataQualitySignal, ...]:
    """Derive evidence-based quality signals from the two summary rows.

    The function intentionally uses only already reported counts.  In particular,
    it does not classify release identifiers or infer whether a release is known.
    The helper never emits a green health signal: the atomic business snapshot
    and the separately fetched auxiliary rows do not share one snapshot or time
    semantics.
    """
    business = _quality_mapping(business_summary, field_name="business_summary")
    if ingestion_summary is not None and ingestion_unavailable_reason is not None:
        raise ValueError(
            "ingestion_summary and ingestion_unavailable_reason are mutually exclusive"
        )
    if question_registry_available is not None and not isinstance(
        question_registry_available, bool
    ):
        raise TypeError("question_registry_available must be a boolean or None")
    if registry_summary is not None and registry_unavailable_reason is not None:
        raise ValueError(
            "registry_summary and registry_unavailable_reason are mutually exclusive"
        )
    if registry_summary is not None and question_registry_available is not None:
        raise ValueError(
            "registry_summary and question_registry_available are mutually exclusive"
        )
    if ingestion_unavailable_reason is not None:
        if not isinstance(ingestion_unavailable_reason, str):
            raise TypeError("ingestion_unavailable_reason must be a string or None")
        if ingestion_unavailable_reason != ingestion_unavailable_reason.strip() or not (
            ingestion_unavailable_reason
        ):
            raise ValueError(
                "ingestion_unavailable_reason must be a trimmed non-empty string"
            )
    if registry_unavailable_reason is not None:
        if not isinstance(registry_unavailable_reason, str):
            raise TypeError("registry_unavailable_reason must be a string or None")
        if registry_unavailable_reason != registry_unavailable_reason.strip() or not (
            registry_unavailable_reason
        ):
            raise ValueError(
                "registry_unavailable_reason must be a trimmed non-empty string"
            )

    answer_count = _quality_count(
        business,
        "answer_count",
        row_name="business summary",
    )
    unknown_answer_count = _quality_count(
        business,
        "unknown_answer_count",
        row_name="business summary",
    )
    unknown_answer_rate = _quality_fraction(
        unknown_answer_count,
        answer_count,
        label="unknown answer count",
    )

    registry_available = bool(question_registry_available)
    registry_issue_count = 0
    signals: list[DataQualitySignal] = []
    if registry_summary is not None:
        registry = _quality_mapping(
            registry_summary,
            field_name="registry_summary",
        )
        registry_available = _quality_boolean(
            registry,
            "registry_available",
            row_name="registry summary",
        )
        registered_release_count = _quality_count(
            registry,
            "registered_release_count",
            row_name="registry summary",
        )
        if registry_available != (registered_release_count > 0):
            raise ValueError(
                "registry summary availability must match registered releases"
            )
        excluded_event_count = _quality_count(
            registry,
            "excluded_event_count",
            row_name="registry summary",
        )
        missing_release_event_count = _quality_count(
            registry,
            "missing_release_event_count",
            row_name="registry summary",
        )
        unknown_release_event_count = _quality_count(
            registry,
            "unknown_release_event_count",
            row_name="registry summary",
        )
        question_not_in_release_event_count = _quality_count(
            registry,
            "question_not_in_release_event_count",
            row_name="registry summary",
        )
        invalid_selected_letter_answer_count = _quality_count(
            registry,
            "invalid_selected_letter_answer_count",
            row_name="registry summary",
        )
        selected_candidate_mismatch_answer_count = _quality_count(
            registry,
            "selected_candidate_mismatch_answer_count",
            row_name="registry summary",
        )
        unmatched_comment_count = _quality_count(
            registry,
            "unmatched_comment_count",
            row_name="registry summary",
        )
        unmatched_proposal_count = _quality_count(
            registry,
            "unmatched_proposal_count",
            row_name="registry summary",
        )
        client_context_mismatch_event_count = _quality_count(
            registry,
            "client_context_mismatch_event_count",
            row_name="registry summary",
        )
        client_correctness_mismatch_answer_count = _quality_count(
            registry,
            "client_correctness_mismatch_answer_count",
            row_name="registry summary",
        )
        registry_issue_count = (
            excluded_event_count
            + invalid_selected_letter_answer_count
            + selected_candidate_mismatch_answer_count
            + client_context_mismatch_event_count
            + client_correctness_mismatch_answer_count
        )
        if excluded_event_count != (
            missing_release_event_count
            + unknown_release_event_count
            + question_not_in_release_event_count
        ):
            raise ValueError(
                "registry summary excluded-event classifications must add up"
            )
        if not registry_available:
            signals.append(
                DataQualitySignal(
                    code="question_registry_unavailable",
                    severity="error",
                    title="No attested question registry",
                    body=(
                        "No immutable quiz release is registered on the server. Raw "
                        "events are retained, but none can enter authoritative "
                        "business dimensions or accuracy until a reviewed registry "
                        "data migration is applied."
                    ),
                )
            )
        if excluded_event_count:
            signals.append(
                DataQualitySignal(
                    code="unresolved_question_registry_events",
                    severity="warning",
                    title="Raw events excluded by the registry",
                    body=(
                        f"{excluded_event_count:,} raw event(s) lack exact registered "
                        "release/question membership: "
                        f"{missing_release_event_count:,} missing release, "
                        f"{unknown_release_event_count:,} unknown release, and "
                        f"{question_not_in_release_event_count:,} question/version "
                        "not in the claimed release. They remain auditable raw facts "
                        "but are excluded from business reports."
                    ),
                )
            )
        answer_identity_mismatch_count = (
            invalid_selected_letter_answer_count
            + selected_candidate_mismatch_answer_count
        )
        if answer_identity_mismatch_count:
            signals.append(
                DataQualitySignal(
                    code="unresolved_registered_answers",
                    severity="error",
                    title="Registered answers could not be resolved",
                    body=(
                        f"{answer_identity_mismatch_count:,} answer event(s) matched "
                        "a registered question but not a canonical choice: "
                        f"{invalid_selected_letter_answer_count:,} invalid letter and "
                        f"{selected_candidate_mismatch_answer_count:,} "
                        "letter/candidate mismatch. They are excluded from "
                        "authoritative accuracy."
                    ),
                )
            )
        if client_context_mismatch_event_count:
            signals.append(
                DataQualitySignal(
                    code="client_registry_context_mismatch",
                    severity="warning",
                    title="Client context disagrees with the registry",
                    body=(
                        f"{client_context_mismatch_event_count:,} registry-matched "
                        "event(s) supplied a different family, dataset, or question "
                        "type. Reports use the server registry value."
                    ),
                )
            )
        if client_correctness_mismatch_answer_count:
            signals.append(
                DataQualitySignal(
                    code="client_registry_correctness_mismatch",
                    severity="error",
                    title="Client correctness disagrees with the registry",
                    body=(
                        f"{client_correctness_mismatch_answer_count:,} resolved answer "
                        "event(s) supplied the opposite is_correct value. Reports "
                        "ignore that client field and use the registered answer."
                    ),
                )
            )
        if unmatched_comment_count or unmatched_proposal_count:
            signals.append(
                DataQualitySignal(
                    code="unmatched_feedback_footprints",
                    severity="warning",
                    title="Comments or proposals lack registry attribution",
                    body=(
                        f"{unmatched_comment_count:,} comment(s) and "
                        f"{unmatched_proposal_count:,} proposal/rejection event(s) "
                        "remain in raw storage but cannot be assigned to an "
                        "authoritative release/question dimension."
                    ),
                )
            )
    elif registry_unavailable_reason is not None:
        signals.append(
            DataQualitySignal(
                code="question_registry_quality_unavailable",
                severity="warning",
                title="Registry quality unavailable",
                body=(
                    f"{registry_unavailable_reason} Authoritative business reports "
                    "do not fall back to client-reported dimensions or correctness."
                ),
            )
        )
    elif answer_count and not registry_available:
        signals.append(
            DataQualitySignal(
                code="question_registry_unavailable",
                severity="warning",
                title="Question registry unavailable",
                body=(
                    "No server registry evidence is available for this snapshot. "
                    "Correctness and release/family/question-type dimensions must not "
                    "fall back to the uploaded client payload."
                ),
            )
        )
    if unknown_answer_count:
        assert unknown_answer_rate is not None
        signals.append(
            DataQualitySignal(
                code="unknown_answer_correctness",
                severity="warning",
                title="Unknown answer correctness",
                body=(
                    f"{unknown_answer_count:,} of {answer_count:,} answer event(s) "
                    f"({format_percentage(unknown_answer_rate)}) could not be "
                    "resolved to a canonical registered choice and are excluded "
                    "from authoritative accuracy."
                ),
            )
        )

    if ingestion_summary is None:
        reason = ingestion_unavailable_reason or (
            "No ingestion observability snapshot is available."
        )
        signals.append(
            DataQualitySignal(
                code="ingestion_unavailable",
                severity="warning",
                title="Ingestion quality unavailable",
                body=(
                    f"{reason} No healthy ingestion conclusion can be drawn from "
                    "the business event counts alone."
                ),
            )
        )
        return tuple(signals)

    ingestion = _quality_mapping(
        ingestion_summary,
        field_name="ingestion_summary",
    )
    recorded_request_count = _quality_count(
        ingestion,
        "recorded_request_count",
        row_name="ingestion summary",
    )
    client_rejection_count = _quality_count(
        ingestion,
        "client_rejection_count",
        row_name="ingestion summary",
    )
    service_failure_count = _quality_count(
        ingestion,
        "service_failure_count",
        row_name="ingestion summary",
    )
    event_id_conflict_request_count = _quality_count(
        ingestion,
        "event_id_conflict_request_count",
        row_name="ingestion summary",
    )
    duplicate_event_count = _quality_count(
        ingestion,
        "duplicate_event_count",
        row_name="ingestion summary",
    )
    idempotent_duplicate_event_count = _quality_count(
        ingestion,
        "idempotent_duplicate_event_count",
        row_name="ingestion summary",
    )
    unclassified_duplicate_event_count = _quality_count(
        ingestion,
        "unclassified_duplicate_event_count",
        row_name="ingestion summary",
    )
    conflicting_event_count = _quality_count(
        ingestion,
        "conflicting_event_count",
        row_name="ingestion summary",
    )
    conflict_audit_event_count = _quality_count(
        ingestion,
        "conflict_audit_event_count",
        row_name="ingestion summary",
    )
    event_id_reuse_count = _quality_count(
        ingestion,
        "event_id_reuse_count",
        row_name="ingestion summary",
    )
    classified_event_count = _quality_count(
        ingestion,
        "classified_event_count",
        row_name="ingestion summary",
    )
    known_event_result_count = _quality_count(
        ingestion,
        "known_event_result_count",
        row_name="ingestion summary",
    )
    end_to_end_coverage_available = _quality_boolean(
        ingestion,
        "end_to_end_coverage_available",
        row_name="ingestion summary",
    )

    recorded_problem_count = client_rejection_count + service_failure_count
    if recorded_problem_count > recorded_request_count:
        raise ValueError(
            "ingestion summary rejection and service-failure counts cannot exceed "
            "recorded requests"
        )
    if event_id_conflict_request_count > client_rejection_count:
        raise ValueError(
            "ingestion summary conflict requests cannot exceed client rejections"
        )
    if duplicate_event_count != (
        idempotent_duplicate_event_count + unclassified_duplicate_event_count
    ):
        raise ValueError("ingestion summary duplicate classifications must add up")
    if event_id_reuse_count != duplicate_event_count + conflicting_event_count:
        raise ValueError("ingestion summary event-ID reuse counts must add up")
    _quality_fraction(
        event_id_reuse_count,
        classified_event_count,
        label="event-ID reuse count",
    )
    if (event_id_conflict_request_count == 0) != (
        conflicting_event_count == 0
    ) or conflicting_event_count < event_id_conflict_request_count:
        raise ValueError(
            "ingestion summary conflicts are inconsistent with conflict requests"
        )
    if conflict_audit_event_count != conflicting_event_count:
        raise ValueError(
            "ingestion summary conflict audit does not match classified conflicts"
        )

    ordinary_client_rejection_count = (
        client_rejection_count - event_id_conflict_request_count
    )
    ordinary_client_rejection_rate = _quality_fraction(
        ordinary_client_rejection_count,
        recorded_request_count,
        label="ordinary client rejection count",
    )
    service_failure_rate = _quality_fraction(
        service_failure_count,
        recorded_request_count,
        label="service failure count",
    )
    _quality_fraction(
        duplicate_event_count,
        known_event_result_count,
        label="duplicate event count",
    )
    classified_event_id_reuse_count = (
        idempotent_duplicate_event_count + conflicting_event_count
    )
    classified_conflicting_event_rate = _quality_fraction(
        conflicting_event_count,
        classified_event_id_reuse_count,
        label="conflicting event count",
    )

    if service_failure_count:
        assert service_failure_rate is not None
        signals.append(
            DataQualitySignal(
                code="recorded_service_failures",
                severity="error",
                title="Recorded service failures",
                body=(
                    f"{service_failure_count:,} of {recorded_request_count:,} recorded "
                    f"authenticated POST request(s) "
                    f"({format_percentage(service_failure_rate)}) ended in a service "
                    "failure and may require a safe retry."
                ),
            )
        )
    if conflicting_event_count:
        assert classified_conflicting_event_rate is not None
        signals.append(
            DataQualitySignal(
                code="conflicting_event_ids",
                severity="error",
                title="Conflicting event IDs detected",
                body=(
                    f"{conflicting_event_count:,} of "
                    f"{classified_event_id_reuse_count:,} classified event-ID "
                    f"reuse(s) ({format_percentage(classified_conflicting_event_rate)}) "
                    "had different logical content across "
                    f"{event_id_conflict_request_count:,} recorded request(s), with "
                    f"{conflict_audit_event_count:,} correlated private-audit row(s). "
                    "The "
                    "stored first write was preserved and each conflicting batch was "
                    "rejected."
                ),
            )
        )
    if ordinary_client_rejection_count:
        assert ordinary_client_rejection_rate is not None
        signals.append(
            DataQualitySignal(
                code="recorded_client_rejections",
                severity="warning",
                title="Recorded client rejections",
                body=(
                    f"{ordinary_client_rejection_count:,} of "
                    f"{recorded_request_count:,} recorded authenticated POST request(s) "
                    f"({format_percentage(ordinary_client_rejection_rate)}) failed "
                    "request or envelope validation before event storage. Event-ID "
                    "conflict requests are reported separately."
                ),
            )
        )
    if unclassified_duplicate_event_count:
        signals.append(
            DataQualitySignal(
                code="unclassified_duplicate_event_ids",
                severity="warning",
                title="Legacy duplicate IDs are unclassified",
                body=(
                    f"{unclassified_duplicate_event_count:,} duplicate event ID(s) "
                    "came from legacy outcome rows recorded before conflict "
                    "classification was available. They are neither verified "
                    "idempotent retries nor confirmed conflicts."
                ),
            )
        )
    if idempotent_duplicate_event_count:
        signals.append(
            DataQualitySignal(
                code="verified_idempotent_duplicates",
                severity="info",
                title="Verified idempotent retries",
                body=(
                    f"{idempotent_duplicate_event_count:,} duplicate event(s) matched "
                    "the stored logical content and were safely classified as "
                    "idempotent retries."
                ),
            )
        )

    if recorded_request_count == 0:
        signals.append(
            DataQualitySignal(
                code="no_recorded_ingestion_outcomes",
                severity="info",
                title="No recorded ingestion outcomes",
                body=(
                    "The selected server-time window has no persisted authenticated "
                    "POST outcomes, so failure and event-ID reuse rates are N/A."
                ),
            )
        )

    if not end_to_end_coverage_available:
        signals.append(
            DataQualitySignal(
                code="ingestion_coverage_incomplete",
                severity="warning",
                title="Ingestion coverage is incomplete",
                body=(
                    "Ingestion findings cover only authenticated POST outcomes that "
                    "reached the Edge Function and were persisted; they are not an "
                    "end-to-end health measure."
                ),
            )
        )
    elif (
        registry_available
        and registry_issue_count == 0
        and answer_count > 0
        and recorded_request_count > 0
        and unknown_answer_count == 0
        and recorded_problem_count == 0
        and event_id_reuse_count == 0
    ):
        signals.append(
            DataQualitySignal(
                code="no_observed_quality_issues",
                severity="info",
                title="No issues observed in the current snapshots",
                body=(
                    f"All {answer_count:,} answer event(s) have known correctness, and "
                    f"all {recorded_request_count:,} covered ingestion request(s) have "
                    "no recorded rejection, service failure, or event-ID reuse. "
                    "This is not an overall health guarantee because the business "
                    "and auxiliary snapshots are not mutually atomic and use "
                    "different time semantics."
                ),
            )
        )

    return tuple(signals)


def _validate_view(view: str) -> str:
    if not isinstance(view, str) or view not in REPORT_VIEWS:
        allowed = ", ".join(REPORT_VIEWS)
        raise ValueError(f"unsupported report view {view!r}; choose one of: {allowed}")
    return view


def _optional_filter(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    cleaned = value.strip()
    return cleaned or None


def _validate_date(value: date | None, *, field_name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be a date or None")
    return value


def _utc_midnight(value: date) -> str:
    return datetime.combine(value, time.min, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def build_global_filters(
    *,
    release_id: str | None = None,
    family: str | None = None,
    question_type: str | None = None,
    question_id: str | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, str]:
    """Build scalar report filters, using a half-open UTC date interval.

    ``start_date`` is included from 00:00 UTC. ``end_date`` is a user-facing
    inclusive calendar date and is converted to the exclusive midnight at the
    start of the following day.
    """
    filters: dict[str, str] = {}
    scalar_values = {
        "release_id": _optional_filter(release_id, field_name="release_id"),
        "family": _optional_filter(family, field_name="family"),
        "question_type": _optional_filter(question_type, field_name="question_type"),
        "question_id": _optional_filter(question_id, field_name="question_id"),
        "session_id": _optional_filter(session_id, field_name="session_id"),
        "attempt_id": _optional_filter(attempt_id, field_name="attempt_id"),
    }
    filters.update(
        (field_name, value)
        for field_name, value in scalar_values.items()
        if value is not None
    )

    resolved_start = _validate_date(start_date, field_name="start_date")
    resolved_end = _validate_date(end_date, field_name="end_date")
    if resolved_start is not None:
        filters["from"] = _utc_midnight(resolved_start)
    if resolved_end is not None:
        try:
            exclusive_end = resolved_end + timedelta(days=1)
        except OverflowError as exc:
            raise ValueError("end_date is too large to form an exclusive end") from exc
        filters["to"] = _utc_midnight(exclusive_end)
    if "from" in filters and "to" in filters and filters["from"] >= filters["to"]:
        raise ValueError("start_date must not be after end_date")
    return filters


def format_percentage(
    value: int | float | None,
    *,
    decimals: int = 1,
    missing: str = MISSING_VALUE,
) -> str:
    """Format a ratio as a percentage while preserving missing values."""
    if value is None:
        return missing
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("percentage value must be a number or None")
    if not math.isfinite(float(value)):
        raise ValueError("percentage value must be finite")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    resolved = 0.0 if value == 0 else float(value)
    return f"{resolved:.{decimals}%}"


def format_optional_kpi(
    value: Any,
    *,
    missing: str = MISSING_VALUE,
) -> str:
    """Format one KPI value without conflating ``None`` and numeric zero."""
    if value is None:
        return missing
    if isinstance(value, bool):
        raise TypeError("KPI value must not be a boolean")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("KPI value must be finite")
        return f"{value:,}"
    return str(value)


def _flatten_array(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    if not isinstance(value, (list, tuple)):
        raise TypeError("report array display value must be a list or tuple")
    return ", ".join(str(item) for item in value) if value else MISSING_VALUE


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value}")


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError(f"duplicate JSON object key {key!r}")
        resolved[key] = value
    return resolved


def _format_json_display(value: Any) -> str:
    if value is None:
        return MISSING_VALUE
    if not isinstance(value, str):
        raise TypeError("report JSON display value must be text or None")
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("report JSON display text must be strict JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("report JSON display text must contain an object")
    try:
        rendered = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        rendered.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("report JSON display text must contain safe JSON") from exc
    if len(rendered) <= _MAX_JSON_DISPLAY_CHARS:
        return rendered
    return f"{rendered[: _MAX_JSON_DISPLAY_CHARS - 1]}…"


def format_report_table_rows(
    view: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return detached rows with arrays and accuracy formatted for display."""
    _validate_view(view)
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    formatted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} must be a mapping")
        display_row = deepcopy(dict(row))
        for column in _ARRAY_DISPLAY_COLUMNS & display_row.keys():
            display_row[column] = _flatten_array(display_row[column])
        for column in _JSON_DISPLAY_COLUMNS & display_row.keys():
            display_row[column] = _format_json_display(display_row[column])
        for column in _PERCENTAGE_DISPLAY_COLUMNS & display_row.keys():
            display_row[column] = format_percentage(display_row[column])
        formatted.append(display_row)
    return formatted


def report_csv_download_bytes(
    view: str,
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize report rows as UTF-8-BOM CSV bytes for a download button."""
    _validate_view(view)
    return report_rows_to_csv(view, rows).encode("utf-8-sig")


def report_csv_filename(view: str, *, generated_at: datetime) -> str:
    """Build a deterministic UTC CSV filename from an aware timestamp."""
    _validate_view(view)
    if not isinstance(generated_at, datetime):
        raise TypeError("generated_at must be a datetime")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    timestamp = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"architectureiq-report-{_VIEW_EXPORT_NAMES[view]}-{timestamp}.csv"


def report_page_is_truncated(*, total: int, row_count: int) -> bool:
    """Return whether a report page omits matching rows."""
    for value, field_name in ((total, "total"), (row_count, "row_count")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    if row_count > total:
        raise ValueError("row_count cannot exceed total")
    return row_count < total
