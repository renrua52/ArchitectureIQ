"""Internal, read-only Streamlit dashboard for feedback reports."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from feedback_reports import (  # noqa: E402
    MAX_LIMIT,
    REPORTS_READ_TOKEN_ENV,
    REPORTS_TIMEOUT_ENV,
    REPORTS_URL_ENV,
    SURPRISE_QUALITY_VIEW,
    SURPRISE_QUESTIONS_VIEW,
    ReportPage,
    ReportsClient,
    ReportsConfig,
    ReportsConfigurationError,
    ReportsError,
    ReportsResponseError,
    validate_report_response,
)
from feedback_reports.ui import (  # noqa: E402
    DataQualitySignal,
    build_global_filters,
    build_data_quality_signals,
    format_optional_kpi,
    format_percentage,
    format_report_table_rows,
    report_csv_download_bytes,
    report_csv_filename,
    report_page_is_truncated,
)


SUMMARY_VIEW = "feedback_report_summary"
INGESTION_VIEW = "feedback_report_ingestion_summary"
REGISTRY_VIEW = "feedback_report_registry_quality"
SESSIONS_VIEW = "feedback_report_sessions"
QUESTIONS_VIEW = "feedback_report_questions"
ANSWERS_VIEW = "feedback_report_answers"
PROPOSALS_VIEW = "feedback_report_proposals"
COMMENTS_VIEW = "feedback_report_comments"
REPORT_VIEW_ORDER = (
    SUMMARY_VIEW,
    SESSIONS_VIEW,
    QUESTIONS_VIEW,
    ANSWERS_VIEW,
    PROPOSALS_VIEW,
    COMMENTS_VIEW,
)
DEFAULT_PAGE_LIMIT = 500
CONTENT_FILTER_NAMES = frozenset(
    {
        "release_id",
        "family",
        "question_type",
        "question_id",
        "session_id",
        "attempt_id",
    }
)


st.set_page_config(
    page_title="ArchitectureIQ Reports",
    page_icon="📊",
    layout="wide",
)


def _init_state() -> None:
    defaults = {
        "reports_snapshot": None,
        "reports_snapshot_metadata": None,
        "reports_ingestion_snapshot": None,
        "reports_ingestion_error": None,
        "reports_ingestion_loaded_at": None,
        "reports_registry_snapshot": None,
        "reports_registry_error": None,
        "reports_registry_loaded_at": None,
        "reports_surprise_questions_snapshot": None,
        "reports_surprise_questions_error": None,
        "reports_surprise_questions_loaded_at": None,
        "reports_surprise_quality_snapshot": None,
        "reports_surprise_quality_error": None,
        "reports_surprise_quality_loaded_at": None,
        "reports_loaded_at": None,
        "reports_error": None,
        "reports_filters": {},
        "reports_filter_release": "",
        "reports_filter_family": "",
        "reports_filter_question_type": "",
        "reports_filter_question_id": "",
        "reports_filter_session_id": "",
        "reports_filter_attempt_id": "",
        "reports_use_dates": False,
        "reports_start_date": date.today() - timedelta(days=29),
        "reports_end_date": date.today(),
        "reports_limit": DEFAULT_PAGE_LIMIT,
        "reports_applied_limit": DEFAULT_PAGE_LIMIT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reports_config() -> ReportsConfig:
    values: Mapping[str, Any] = {}
    try:
        configured = st.secrets.get("reports", {})
    except FileNotFoundError:
        configured = {}
    if isinstance(configured, Mapping):
        values = configured
    return ReportsConfig.from_sources(
        url=values.get("endpoint") or values.get("url"),
        read_token=values.get("token") or values.get("read_token"),
        timeout_seconds=values.get("timeout_seconds"),
    )


def _refresh_surprise_reports(
    client: ReportsClient,
    *,
    filters: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> None:
    requested_filters = dict(
        st.session_state.reports_filters if filters is None else filters
    )
    requested_limit = st.session_state.reports_applied_limit if limit is None else limit
    requests = (
        (
            "surprise_questions",
            client.fetch_surprise_questions,
        ),
        (
            "surprise_quality",
            client.fetch_surprise_quality,
        ),
    )
    for state_prefix, fetch_page in requests:
        try:
            page = fetch_page(
                filters=requested_filters,
                limit=requested_limit,
            )
        except ReportsError as exc:
            st.session_state[f"reports_{state_prefix}_snapshot"] = None
            st.session_state[f"reports_{state_prefix}_error"] = str(exc)
            st.session_state[f"reports_{state_prefix}_loaded_at"] = None
        else:
            st.session_state[f"reports_{state_prefix}_snapshot"] = page.to_dict()
            st.session_state[f"reports_{state_prefix}_error"] = None
            st.session_state[f"reports_{state_prefix}_loaded_at"] = datetime.now(
                timezone.utc
            )


def _refresh_reports(
    client: ReportsClient,
    *,
    filters: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> None:
    requested_filters = dict(
        st.session_state.reports_filters if filters is None else filters
    )
    requested_limit = st.session_state.reports_applied_limit if limit is None else limit
    try:
        with st.spinner("Loading feedback reports…"):
            business_snapshot = client.fetch_business_snapshot(
                filters=requested_filters,
                limit=requested_limit,
            )
            next_pages = {
                view: business_snapshot.pages[view].to_dict()
                for view in REPORT_VIEW_ORDER
            }
            next_metadata = {
                "snapshot_at": business_snapshot.snapshot_at,
                "snapshot_revision": business_snapshot.snapshot_revision,
                "authority_revision": business_snapshot.authority_revision,
                "detail_revision": business_snapshot.detail_revision,
                "registered_release_count": (
                    business_snapshot.registered_release_count
                ),
                "registered_question_count": (
                    business_snapshot.registered_question_count
                ),
                "registered_choice_count": business_snapshot.registered_choice_count,
            }
            next_loaded_at = datetime.now(timezone.utc)
    except ReportsError as exc:
        st.session_state.reports_error = str(exc)
    else:
        # Commit all six business pages and their authority metadata together only
        # after the strict BusinessSnapshot parser has accepted the whole response.
        st.session_state.reports_snapshot = next_pages
        st.session_state.reports_snapshot_metadata = next_metadata
        st.session_state.reports_loaded_at = next_loaded_at
        st.session_state.reports_filters = requested_filters
        st.session_state.reports_applied_limit = requested_limit
        st.session_state.reports_error = None
        if CONTENT_FILTER_NAMES & requested_filters.keys():
            st.session_state.reports_ingestion_snapshot = None
            st.session_state.reports_ingestion_error = None
            st.session_state.reports_ingestion_loaded_at = None
            st.session_state.reports_registry_snapshot = None
            st.session_state.reports_registry_error = None
            st.session_state.reports_registry_loaded_at = None
        else:
            auxiliary_filters = {
                name: value
                for name, value in requested_filters.items()
                if name in {"from", "to"}
            }
            for state_prefix, view in (
                ("ingestion", INGESTION_VIEW),
                ("registry", REGISTRY_VIEW),
            ):
                try:
                    auxiliary_page = client.fetch_page(
                        view,
                        filters=auxiliary_filters,
                        limit=requested_limit,
                        offset=0,
                    )
                except ReportsError as exc:
                    st.session_state[f"reports_{state_prefix}_snapshot"] = None
                    st.session_state[f"reports_{state_prefix}_error"] = str(exc)
                    st.session_state[f"reports_{state_prefix}_loaded_at"] = None
                else:
                    st.session_state[f"reports_{state_prefix}_snapshot"] = (
                        auxiliary_page.to_dict()
                    )
                    st.session_state[f"reports_{state_prefix}_error"] = None
                    st.session_state[f"reports_{state_prefix}_loaded_at"] = (
                        datetime.now(timezone.utc)
                    )
        _refresh_surprise_reports(
            client,
            filters=requested_filters,
            limit=requested_limit,
        )


def _reset_filters() -> None:
    st.session_state.reports_filter_release = ""
    st.session_state.reports_filter_family = ""
    st.session_state.reports_filter_question_type = ""
    st.session_state.reports_filter_question_id = ""
    st.session_state.reports_filter_session_id = ""
    st.session_state.reports_filter_attempt_id = ""
    st.session_state.reports_use_dates = False
    st.session_state.reports_start_date = date.today() - timedelta(days=29)
    st.session_state.reports_end_date = date.today()
    st.session_state.reports_limit = DEFAULT_PAGE_LIMIT
    st.session_state.reports_applied_limit = DEFAULT_PAGE_LIMIT
    st.session_state.reports_filters = {}
    st.session_state.reports_snapshot = None
    st.session_state.reports_snapshot_metadata = None
    st.session_state.reports_ingestion_snapshot = None
    st.session_state.reports_ingestion_error = None
    st.session_state.reports_ingestion_loaded_at = None
    st.session_state.reports_registry_snapshot = None
    st.session_state.reports_registry_error = None
    st.session_state.reports_registry_loaded_at = None
    st.session_state.reports_surprise_questions_snapshot = None
    st.session_state.reports_surprise_questions_error = None
    st.session_state.reports_surprise_questions_loaded_at = None
    st.session_state.reports_surprise_quality_snapshot = None
    st.session_state.reports_surprise_quality_error = None
    st.session_state.reports_surprise_quality_loaded_at = None
    st.session_state.reports_loaded_at = None
    st.session_state.reports_error = None


def _render_sidebar(client: ReportsClient | None) -> None:
    st.header("Report filters")
    if client is None or not client.is_configured:
        st.warning("Report endpoint is not configured.")
    else:
        st.success("Protected report endpoint configured.")

    with st.form("reports_filters_form"):
        st.text_input("Release ID", key="reports_filter_release")
        st.text_input("Family", key="reports_filter_family")
        st.text_input("Question type", key="reports_filter_question_type")
        st.text_input("Question ID", key="reports_filter_question_id")
        st.text_input("Session ID", key="reports_filter_session_id")
        st.text_input("Attempt ID", key="reports_filter_attempt_id")
        st.checkbox("Limit by UTC date", key="reports_use_dates")
        date_columns = st.columns(2)
        date_columns[0].date_input(
            "Start",
            key="reports_start_date",
        )
        date_columns[1].date_input(
            "End",
            key="reports_end_date",
        )
        st.caption("Start/end are ignored unless UTC date filtering is enabled.")
        st.selectbox(
            "Maximum rows per table",
            (100, 250, 500, MAX_LIMIT),
            key="reports_limit",
        )
        apply_filters = st.form_submit_button(
            "Apply filters",
            type="primary",
            width="stretch",
        )

    button_columns = st.columns(2)
    refresh = button_columns[0].button(
        "Refresh",
        disabled=client is None or not client.is_configured,
        width="stretch",
    )
    reset = button_columns[1].button("Reset", width="stretch")

    if reset:
        _reset_filters()
        st.rerun()

    if apply_filters:
        try:
            use_dates = st.session_state.reports_use_dates
            filters = build_global_filters(
                release_id=st.session_state.reports_filter_release,
                family=st.session_state.reports_filter_family,
                question_type=st.session_state.reports_filter_question_type,
                question_id=st.session_state.reports_filter_question_id,
                session_id=st.session_state.reports_filter_session_id,
                attempt_id=st.session_state.reports_filter_attempt_id,
                start_date=(st.session_state.reports_start_date if use_dates else None),
                end_date=st.session_state.reports_end_date if use_dates else None,
            )
        except (TypeError, ValueError) as exc:
            st.session_state.reports_error = str(exc)
        else:
            if client is not None and client.is_configured:
                _refresh_reports(
                    client,
                    filters=filters,
                    limit=st.session_state.reports_limit,
                )

    if refresh and client is not None:
        _refresh_reports(client)


def _page(view: str) -> ReportPage | None:
    snapshot = st.session_state.reports_snapshot
    if not isinstance(snapshot, Mapping):
        return None
    raw_page = snapshot.get(view)
    if not isinstance(raw_page, Mapping):
        return None
    try:
        return validate_report_response(view, raw_page)
    except ReportsResponseError:
        return None


def _ingestion_page() -> ReportPage | None:
    raw_page = st.session_state.reports_ingestion_snapshot
    if not isinstance(raw_page, Mapping):
        return None
    try:
        return validate_report_response(INGESTION_VIEW, raw_page)
    except ReportsResponseError:
        return None


def _registry_page() -> ReportPage | None:
    raw_page = st.session_state.reports_registry_snapshot
    if not isinstance(raw_page, Mapping):
        return None
    try:
        return validate_report_response(REGISTRY_VIEW, raw_page)
    except ReportsResponseError:
        return None


def _surprise_questions_page() -> ReportPage | None:
    raw_page = st.session_state.reports_surprise_questions_snapshot
    if not isinstance(raw_page, Mapping):
        return None
    try:
        return validate_report_response(SURPRISE_QUESTIONS_VIEW, raw_page)
    except ReportsResponseError:
        return None


def _surprise_quality_page() -> ReportPage | None:
    raw_page = st.session_state.reports_surprise_quality_snapshot
    if not isinstance(raw_page, Mapping):
        return None
    try:
        return validate_report_response(SURPRISE_QUALITY_VIEW, raw_page)
    except ReportsResponseError:
        return None


def _filters_caption() -> str:
    filters = st.session_state.reports_filters
    if not filters:
        return "All events"
    return " · ".join(f"{name}={value}" for name, value in filters.items())


def _render_csv_download(view: str, page: ReportPage) -> None:
    rows = page.rows_copy()
    truncated = report_page_is_truncated(total=page.total, row_count=len(rows))
    if truncated:
        st.warning(
            f"Showing {len(rows):,} of {page.total:,} matching rows. "
            "Narrow the filters before exporting; partial CSV export is disabled."
        )
        data = b""
    else:
        data = report_csv_download_bytes(view, rows)
    loaded_at_key = {
        INGESTION_VIEW: "reports_ingestion_loaded_at",
        REGISTRY_VIEW: "reports_registry_loaded_at",
        SURPRISE_QUESTIONS_VIEW: "reports_surprise_questions_loaded_at",
        SURPRISE_QUALITY_VIEW: "reports_surprise_quality_loaded_at",
    }.get(view, "reports_loaded_at")
    generated_at = st.session_state[loaded_at_key] or datetime.now(timezone.utc)
    st.download_button(
        "Download filtered CSV",
        data=data,
        file_name=report_csv_filename(view, generated_at=generated_at),
        mime="text/csv; charset=utf-8",
        disabled=truncated,
        key=f"download_{view}",
    )


def _render_summary(page: ReportPage, registry_page: ReportPage | None) -> None:
    rows = page.rows_copy()
    if len(rows) != 1:
        st.error("The summary endpoint did not return exactly one summary row.")
        return
    summary = rows[0]
    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Sessions", format_optional_kpi(summary.get("session_count"))
    )
    metric_columns[1].metric(
        "Attempts", format_optional_kpi(summary.get("attempt_count"))
    )
    metric_columns[2].metric(
        "Answers", format_optional_kpi(summary.get("answer_count"))
    )
    metric_columns[3].metric(
        "Authoritative accuracy", format_percentage(summary.get("accuracy"))
    )
    metric_columns[4].metric(
        "Propose usage",
        format_percentage(summary.get("proposal_usage_rate")),
    )
    metric_columns[5].metric(
        "Comments", format_optional_kpi(summary.get("comment_count"))
    )
    st.caption(
        f"{format_optional_kpi(summary.get('question_count'))} question version(s) · "
        f"{format_optional_kpi(summary.get('solve_attempt_count'))} solve attempt(s) · "
        f"{format_optional_kpi(summary.get('correct_answer_count'))} correct / "
        f"{format_optional_kpi(summary.get('incorrect_answer_count'))} incorrect / "
        f"{format_optional_kpi(summary.get('unknown_answer_count'))} unknown · "
        f"{format_optional_kpi(summary.get('proposal_count'))} proposed / "
        f"{format_optional_kpi(summary.get('rejected_setting_count'))} rejected · "
        f"{format_optional_kpi(summary.get('completed_run_count'))} run(s) completed / "
        f"{format_optional_kpi(summary.get('failed_run_count'))} failed"
    )
    if registry_page is None:
        st.warning(
            "Registry coverage could not be loaded. Business rows still use only "
            "server-side registry attribution, but this snapshot cannot show how "
            "many raw events were excluded."
        )
    else:
        registry_rows = registry_page.rows_copy()
        registry_available = bool(
            len(registry_rows) == 1
            and registry_rows[0].get("registry_available") is True
        )
        if registry_available:
            st.caption(
                "Correctness and release/family/question-type dimensions are "
                "derived from the immutable, attested server registry. Unmatched "
                "raw events are excluded from these business KPIs."
            )
        else:
            st.warning(
                "No attested quiz release is registered. Business KPIs cannot "
                "attribute raw events or calculate authoritative accuracy."
            )
    if summary.get("ingestion_failure_rate_available") is not True:
        st.info(
            "End-to-end ingestion failure rate is not available. See the Ingestion "
            "observability tab for the narrower set of persisted endpoint outcomes."
        )
    else:
        st.caption(
            "Ingestion failure rate: "
            f"{format_percentage(summary.get('ingestion_failure_rate'))}"
        )
    _render_csv_download(SUMMARY_VIEW, page)


def _render_ingestion(page: ReportPage) -> None:
    rows = page.rows_copy()
    if len(rows) != 1:
        st.error("The ingestion endpoint did not return exactly one summary row.")
        return
    summary = rows[0]
    metric_columns = st.columns(7)
    metric_columns[0].metric(
        "Recorded requests",
        format_optional_kpi(summary.get("recorded_request_count")),
    )
    metric_columns[1].metric(
        "Success",
        format_optional_kpi(summary.get("success_request_count")),
    )
    metric_columns[2].metric(
        "Client rejected (total)",
        format_optional_kpi(summary.get("client_rejection_count")),
    )
    metric_columns[3].metric(
        "Event-ID conflict requests",
        format_optional_kpi(summary.get("event_id_conflict_request_count")),
    )
    metric_columns[4].metric(
        "Service failures",
        format_optional_kpi(summary.get("service_failure_count")),
    )
    metric_columns[5].metric(
        "Recorded failure rate",
        format_percentage(summary.get("request_failure_rate")),
    )
    metric_columns[6].metric(
        "Duplicate rate (known results)",
        format_percentage(summary.get("duplicate_event_rate")),
    )

    reuse_columns = st.columns(6)
    reuse_columns[0].metric(
        "Verified idempotent duplicates",
        format_optional_kpi(summary.get("idempotent_duplicate_event_count")),
    )
    reuse_columns[1].metric(
        "Legacy unclassified duplicates",
        format_optional_kpi(summary.get("unclassified_duplicate_event_count")),
    )
    reuse_columns[2].metric(
        "Conflicting events",
        format_optional_kpi(summary.get("conflicting_event_count")),
    )
    reuse_columns[3].metric(
        "Conflict audit rows",
        format_optional_kpi(summary.get("conflict_audit_event_count")),
    )
    reuse_columns[4].metric(
        "Event-ID reuse rate",
        format_percentage(summary.get("event_id_reuse_rate")),
    )
    reuse_columns[5].metric(
        "Classified conflict rate",
        format_percentage(summary.get("classified_conflicting_event_rate")),
    )
    st.caption(
        f"{format_optional_kpi(summary.get('accepted_event_count'))} accepted event(s) · "
        f"{format_optional_kpi(summary.get('duplicate_event_count'))} non-conflicting "
        "duplicate result(s) · "
        f"{format_optional_kpi(summary.get('event_id_reuse_count'))} total event-ID "
        "reuse(s) across "
        f"{format_optional_kpi(summary.get('classified_event_count'))} classified "
        "event(s) · "
        f"server window {summary.get('first_started_at') or '—'} to "
        f"{summary.get('last_finished_at') or '—'}"
    )
    if summary.get("recorded_rate_available") is not True:
        st.info(
            "No persisted authenticated POST outcomes match the server-time window."
        )
    st.warning(
        "Coverage is incomplete: these rates include only authenticated POST outcomes "
        "that reached the Edge Function and were persisted. They exclude 401/405, "
        "missing configuration, requests that never reached Edge, and any outcome "
        "write lost to timeout, schema/HTTP failure, or database outage. Verified "
        "idempotent duplicates matched stored logical content; legacy duplicate IDs "
        "remain unclassified, while conflicting content is rejected under "
        "first-write-wins."
    )
    _render_csv_download(INGESTION_VIEW, page)


def _render_registry_quality(page: ReportPage) -> None:
    rows = page.rows_copy()
    if len(rows) != 1:
        st.error("The registry endpoint did not return exactly one quality row.")
        return
    summary = rows[0]
    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Registered releases",
        format_optional_kpi(summary.get("registered_release_count")),
    )
    metric_columns[1].metric(
        "Registered questions",
        format_optional_kpi(summary.get("registered_question_count")),
    )
    metric_columns[2].metric(
        "Raw events",
        format_optional_kpi(summary.get("raw_event_count")),
    )
    metric_columns[3].metric(
        "Registry match rate",
        format_percentage(summary.get("registry_match_rate")),
    )
    metric_columns[4].metric(
        "Resolved answers",
        format_optional_kpi(summary.get("authoritative_answer_count")),
    )
    metric_columns[5].metric(
        "Answer resolution rate",
        format_percentage(summary.get("answer_resolution_rate")),
    )
    st.caption(
        f"{format_optional_kpi(summary.get('authoritative_event_count'))} "
        "registry-matched / "
        f"{format_optional_kpi(summary.get('excluded_event_count'))} excluded · "
        f"{format_optional_kpi(summary.get('missing_release_event_count'))} "
        "missing release · "
        f"{format_optional_kpi(summary.get('unknown_release_event_count'))} "
        "unknown release · "
        f"{format_optional_kpi(summary.get('question_not_in_release_event_count'))} "
        "question/version not in release"
    )
    st.caption(
        f"{format_optional_kpi(summary.get('invalid_selected_letter_answer_count'))} "
        "invalid selected letter · "
        f"{format_optional_kpi(summary.get('selected_candidate_mismatch_answer_count'))} "
        "letter/candidate mismatch · "
        f"{format_optional_kpi(summary.get('client_context_mismatch_event_count'))} "
        "client context mismatch · "
        f"{format_optional_kpi(summary.get('client_correctness_mismatch_answer_count'))} "
        "client correctness mismatch"
    )
    if summary.get("registry_available") is not True:
        st.error(
            "No attested release is registered; all raw events remain outside "
            "authoritative business reports."
        )
    elif summary.get("excluded_event_count"):
        st.warning(
            "Some raw events could not be attributed to an exact registered "
            "release/question version and are excluded from business reports."
        )
    else:
        st.info(
            "Every raw event in this event-time window has exact registered "
            "release/question membership."
        )
    _render_csv_download(REGISTRY_VIEW, page)


def _render_surprise_questions(page: ReportPage) -> None:
    rows = page.rows_copy()
    st.caption(f"{len(rows):,} of {page.total:,} matching question version(s)")
    st.caption(
        "A player who does not click either post-result reaction is missing, not "
        "a ‘no’. The observed rate is therefore unavailable with zero ratings. "
        "The smoothed posterior uses Beta(1,1) and never substitutes answer "
        "correctness, comments, or offline model signals for a player reaction."
    )
    if rows:
        top = rows[0]
        st.caption(
            "Highest posterior row on this ordered page: "
            f"{top['release_id']} / {top['question_id']} / "
            f"{top['question_version']}"
        )
        metric_columns = st.columns(6)
        metric_columns[0].metric(
            "Ratings",
            format_optional_kpi(top.get("rating_count")),
        )
        metric_columns[1].metric(
            "Surprised (yes)",
            format_optional_kpi(top.get("surprised_count")),
        )
        metric_columns[2].metric(
            "Not surprised (no)",
            format_optional_kpi(top.get("not_surprised_count")),
        )
        metric_columns[3].metric(
            "Rating coverage",
            format_percentage(top.get("rating_coverage_rate")),
        )
        metric_columns[4].metric(
            "Observed surprise",
            format_percentage(top.get("observed_surprise_rate")),
        )
        metric_columns[5].metric(
            "Beta(1,1) posterior",
            format_percentage(top.get("posterior_mean")),
        )
        st.dataframe(
            format_report_table_rows(SURPRISE_QUESTIONS_VIEW, rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info(
            "No registry-matched answered question versions match the current filters."
        )
    _render_csv_download(SURPRISE_QUESTIONS_VIEW, page)


def _render_surprise_quality(page: ReportPage) -> None:
    rows = page.rows_copy()
    if len(rows) != 1:
        st.error("The surprise quality endpoint did not return exactly one row.")
        return
    quality = rows[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Raw reactions",
        format_optional_kpi(quality.get("raw_reaction_count")),
    )
    metric_columns[1].metric(
        "Valid first ratings",
        format_optional_kpi(quality.get("valid_reaction_count")),
    )
    metric_columns[2].metric(
        "Orphan reactions",
        format_optional_kpi(quality.get("orphan_reaction_count")),
    )
    metric_columns[3].metric(
        "Duplicate reactions",
        format_optional_kpi(quality.get("duplicate_reaction_count")),
    )
    orphan_columns = st.columns(4)
    orphan_columns[0].metric(
        "Registry unmatched",
        format_optional_kpi(quality.get("registry_unmatched_reaction_count")),
    )
    orphan_columns[1].metric(
        "Invalid reaction payload",
        format_optional_kpi(quality.get("invalid_payload_reaction_count")),
    )
    orphan_columns[2].metric(
        "Missing prior answer",
        format_optional_kpi(quality.get("missing_prior_answer_reaction_count")),
    )
    orphan_columns[3].metric(
        "Unknown release",
        format_optional_kpi(quality.get("unknown_release_reaction_count")),
    )
    st.info(
        "Validated conservation: raw reactions = valid first ratings + orphans + "
        "duplicates; orphans = registry unmatched + invalid payload + missing "
        "prior answer. Unknown-release reactions are a subset of registry-unmatched "
        "reactions."
    )
    _render_csv_download(SURPRISE_QUALITY_VIEW, page)


def _render_quality_signal(signal: DataQualitySignal) -> None:
    message = f"**{signal.title}**\n\n{signal.body}"
    renderers = {
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }
    renderers[signal.severity](message)


def _render_data_quality(
    summary_page: ReportPage,
    ingestion_page: ReportPage | None,
    registry_page: ReportPage | None,
    *,
    ingestion_unavailable_reason: str | None = None,
    registry_unavailable_reason: str | None = None,
) -> None:
    summary_rows = summary_page.rows_copy()
    if len(summary_rows) != 1:
        st.error("The business summary is unavailable for data-quality checks.")
        return
    ingestion_summary: Mapping[str, Any] | None = None
    if ingestion_page is not None:
        ingestion_rows = ingestion_page.rows_copy()
        if len(ingestion_rows) != 1:
            st.error("The ingestion summary is unavailable for data-quality checks.")
            return
        ingestion_summary = ingestion_rows[0]
    registry_summary: Mapping[str, Any] | None = None
    if registry_page is not None:
        registry_rows = registry_page.rows_copy()
        if len(registry_rows) != 1:
            st.error("The registry summary is unavailable for data-quality checks.")
            return
        registry_summary = registry_rows[0]

    try:
        signals = build_data_quality_signals(
            summary_rows[0],
            ingestion_summary=ingestion_summary,
            ingestion_unavailable_reason=ingestion_unavailable_reason,
            registry_summary=registry_summary,
            registry_unavailable_reason=registry_unavailable_reason,
        )
    except (TypeError, ValueError) as exc:
        st.error(f"The saved summaries contain invalid data-quality facts. {exc}")
        return

    st.caption(
        "Derived from the current business, registry, and ingestion snapshots; this "
        "tab makes no additional report request. Business/registry event time and "
        "ingestion server time remain separate, non-atomic windows."
    )
    for signal in signals:
        _render_quality_signal(signal)
    st.caption(
        "Verified duplicates matched stored logical content; legacy duplicates remain "
        "unclassified, and confirmed conflicts are reported separately. This view "
        "uses the server registry to classify release/question claims but does not "
        "claim end-to-end ingestion health."
    )


def _render_table(view: str, page: ReportPage, *, empty_message: str) -> None:
    rows = page.rows_copy()
    st.caption(f"{len(rows):,} of {page.total:,} matching row(s)")
    if rows:
        st.dataframe(
            format_report_table_rows(view, rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info(empty_message)
    _render_csv_download(view, page)


def _render_configuration_help() -> None:
    st.info(
        "Deploy the protected `feedback-report` Edge Function, then configure "
        "this app with a dedicated report token. The ingest token and Supabase "
        "service-role key are not valid client configuration."
    )
    st.code(
        """[reports]
endpoint = "https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-report"
token = "DEDICATED_REPORT_TOKEN"
timeout_seconds = 10""",
        language="toml",
    )
    st.caption(
        f"Environment alternatives: {REPORTS_URL_ENV}, {REPORTS_READ_TOKEN_ENV}, "
        f"and {REPORTS_TIMEOUT_ENV}."
    )


def main() -> None:
    _init_state()
    try:
        config = _reports_config()
        client = ReportsClient(config)
        config_error = None
    except ReportsConfigurationError as exc:
        client = None
        config_error = str(exc)

    with st.sidebar:
        _render_sidebar(client)

    st.title("ArchitectureIQ Reports")
    st.caption(
        "Read-only internal feedback analytics. Business tabs use content filters "
        "and client event time, but dimensions and correctness come only from the "
        "server registry. Ingestion observability is independent and uses server "
        "request time."
    )

    if config_error:
        st.error(f"Report configuration is invalid: {config_error}")
        return
    if client is None or not client.is_configured:
        _render_configuration_help()
        return

    if (
        st.session_state.reports_snapshot is None
        and st.session_state.reports_error is None
    ):
        _refresh_reports(client)

    error = st.session_state.reports_error
    if error:
        st.error(
            "Reports could not be refreshed. The previous successful snapshot, "
            f"if any, is kept. {error}"
        )
    loaded_at = st.session_state.reports_loaded_at
    snapshot_metadata = st.session_state.reports_snapshot_metadata
    if loaded_at is not None:
        server_snapshot_at = (
            snapshot_metadata.get("snapshot_at")
            if isinstance(snapshot_metadata, Mapping)
            else "unavailable"
        )
        st.caption(
            f"Loaded {loaded_at.astimezone(timezone.utc).isoformat()} · "
            f"Server snapshot_at={server_snapshot_at} · "
            f"{_filters_caption()}"
        )

    if not isinstance(snapshot_metadata, Mapping):
        st.error(
            "No validated atomic business snapshot is available. Business KPIs and "
            "tables remain hidden until the server returns business_snapshot_v1 with "
            "the embedded registry_v1/detail_v1 authority contract."
        )
        return
    st.caption(
        "All six business tabs share one PostgreSQL MVCC snapshot at "
        f"{snapshot_metadata['snapshot_at']}. Verified server revisions "
        f"{snapshot_metadata['snapshot_revision']}/"
        f"{snapshot_metadata['authority_revision']}/"
        f"{snapshot_metadata['detail_revision']}; registry contains "
        f"{snapshot_metadata['registered_release_count']:,} release(s), "
        f"{snapshot_metadata['registered_question_count']:,} question(s), and "
        f"{snapshot_metadata['registered_choice_count']:,} choice(s)."
    )

    pages = {view: _page(view) for view in REPORT_VIEW_ORDER}
    if any(page is None for page in pages.values()):
        if st.session_state.reports_snapshot is None:
            st.warning("No report snapshot is available yet.")
        else:
            st.error("The saved report snapshot is incomplete. Refresh to retry.")
        return

    ingestion_page = _ingestion_page()
    registry_page = _registry_page()
    surprise_questions_page = _surprise_questions_page()
    surprise_quality_page = _surprise_quality_page()
    content_filters_active = bool(
        CONTENT_FILTER_NAMES & st.session_state.reports_filters.keys()
    )
    if content_filters_active:
        ingestion_unavailable_reason = (
            "Content filters intentionally exclude ingestion observability."
        )
    elif st.session_state.reports_ingestion_error:
        ingestion_unavailable_reason = (
            "The independent ingestion observability request failed."
        )
    elif ingestion_page is None:
        ingestion_unavailable_reason = (
            "No valid ingestion observability snapshot is available."
        )
    else:
        ingestion_unavailable_reason = None
    if content_filters_active:
        registry_unavailable_reason = (
            "Content filters intentionally exclude the all-event registry quality "
            "snapshot."
        )
    elif st.session_state.reports_registry_error:
        registry_unavailable_reason = "The independent registry quality request failed."
    elif registry_page is None:
        registry_unavailable_reason = "No valid registry quality snapshot is available."
    else:
        registry_unavailable_reason = None

    (
        summary_tab,
        sessions_tab,
        questions_tab,
        answers_tab,
        proposals_tab,
        comments_tab,
        ingestion_tab,
        registry_tab,
        surprise_tab,
        data_quality_tab,
    ) = st.tabs(
        [
            "Summary",
            "Sessions",
            "Questions",
            "Answers",
            "Proposals",
            "Comments",
            "Ingestion observability",
            "Registry quality",
            "Surprise",
            "Data quality",
        ]
    )
    with summary_tab:
        _render_summary(
            pages[SUMMARY_VIEW],  # type: ignore[arg-type]
            registry_page if registry_unavailable_reason is None else None,
        )
    with sessions_tab:
        _render_table(
            SESSIONS_VIEW,
            pages[SESSIONS_VIEW],  # type: ignore[arg-type]
            empty_message="No sessions match the current filters.",
        )
    with questions_tab:
        _render_table(
            QUESTIONS_VIEW,
            pages[QUESTIONS_VIEW],  # type: ignore[arg-type]
            empty_message="No questions match the current filters.",
        )
    with answers_tab:
        _render_table(
            ANSWERS_VIEW,
            pages[ANSWERS_VIEW],  # type: ignore[arg-type]
            empty_message="No answers match the current filters.",
        )
    with proposals_tab:
        _render_table(
            PROPOSALS_VIEW,
            pages[PROPOSALS_VIEW],  # type: ignore[arg-type]
            empty_message="No proposals match the current filters.",
        )
    with comments_tab:
        _render_table(
            COMMENTS_VIEW,
            pages[COMMENTS_VIEW],  # type: ignore[arg-type]
            empty_message="No comments match the current filters.",
        )
    with ingestion_tab:
        if content_filters_active:
            st.info(
                "Ingestion observability is unavailable while release/family/type/"
                "question/session/attempt filters are active because rejected "
                "requests have no trustworthy business dimension. Clear those "
                "filters to query it."
            )
        elif st.session_state.reports_ingestion_error:
            st.error(
                "Ingestion observability could not be loaded independently of the "
                f"business reports. {st.session_state.reports_ingestion_error}"
            )
        elif ingestion_page is None:
            st.warning("No ingestion observability snapshot is available yet.")
        else:
            _render_ingestion(ingestion_page)
    with registry_tab:
        if content_filters_active:
            st.info(
                "Registry quality is an all-event coverage surface and is unavailable "
                "while release/family/type/question/session/attempt filters are "
                "active. Clear those filters to query it."
            )
        elif st.session_state.reports_registry_error:
            st.error(
                "Registry quality could not be loaded independently of the business "
                f"reports. {st.session_state.reports_registry_error}"
            )
        elif registry_page is None:
            st.warning("No registry quality snapshot is available yet.")
        else:
            _render_registry_quality(registry_page)
    with surprise_tab:
        st.caption(
            "Surprise questions and surprise quality are fetched by two independent "
            "RPC statements. They are not part of the six-page business MVCC "
            "snapshot and are not guaranteed to share a snapshot with each other. "
            "Both use the currently applied eight identity/time filters."
        )
        surprise_loaded = st.session_state.reports_surprise_questions_loaded_at
        quality_loaded = st.session_state.reports_surprise_quality_loaded_at
        if surprise_loaded is not None or quality_loaded is not None:
            st.caption(
                "Question page loaded="
                f"{surprise_loaded.astimezone(timezone.utc).isoformat() if surprise_loaded else 'unavailable'}"
                " · Quality loaded="
                f"{quality_loaded.astimezone(timezone.utc).isoformat() if quality_loaded else 'unavailable'}"
            )
        if st.button("Refresh surprise", key="refresh_surprise_reports"):
            _refresh_surprise_reports(client)
            st.rerun()

        st.subheader("Observed surprise by question")
        if st.session_state.reports_surprise_questions_error:
            st.error(
                "The independent surprise question page could not be loaded. "
                f"{st.session_state.reports_surprise_questions_error}"
            )
        elif surprise_questions_page is None:
            st.warning("No validated surprise question page is available yet.")
        else:
            _render_surprise_questions(surprise_questions_page)

        st.subheader("Reaction quality")
        if st.session_state.reports_surprise_quality_error:
            st.error(
                "The independent surprise quality row could not be loaded. "
                f"{st.session_state.reports_surprise_quality_error}"
            )
        elif surprise_quality_page is None:
            st.warning("No validated surprise quality row is available yet.")
        else:
            _render_surprise_quality(surprise_quality_page)
    with data_quality_tab:
        _render_data_quality(
            pages[SUMMARY_VIEW],  # type: ignore[arg-type]
            ingestion_page if ingestion_unavailable_reason is None else None,
            registry_page if registry_unavailable_reason is None else None,
            ingestion_unavailable_reason=ingestion_unavailable_reason,
            registry_unavailable_reason=registry_unavailable_reason,
        )


if __name__ == "__main__":
    main()
