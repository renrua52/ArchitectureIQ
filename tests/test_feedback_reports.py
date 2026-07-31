"""Tests for the protected feedback Reports data-access package."""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.parse
from email.message import Message
from typing import Any

import pytest

from tools import feedback_reports as reports
from tools.feedback_reports import client as reports_client


REPORT_URL = "https://project.supabase.co/functions/v1/feedback-report"
INGEST_REQUEST_ID = "72aee12d-7742-44ea-b3d9-f056ae5c8ac2"
RELEASE_ID = "release_" + "a" * 64
QUESTION_VERSION = "qv1_" + "c" * 64


def _row(view: str, **values: Any) -> dict[str, Any]:
    row = {column: None for column in reports.copy_view_columns()[view]}
    unknown = values.keys() - row.keys()
    assert not unknown, f"test provided unknown columns: {sorted(unknown)}"
    row.update(values)
    return row


def _ingestion_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_ingestion_summary",
        recorded_request_count=3,
        first_started_at="2026-07-12T00:00:00Z",
        last_finished_at="2026-07-12T00:01:00Z",
        success_request_count=2,
        client_rejection_count=1,
        service_failure_count=0,
        event_id_conflict_request_count=0,
        accepted_event_count=2,
        duplicate_event_count=1,
        idempotent_duplicate_event_count=1,
        unclassified_duplicate_event_count=0,
        conflicting_event_count=0,
        conflict_audit_event_count=0,
        event_id_reuse_count=1,
        classified_event_count=3,
        known_event_result_count=3,
        request_failure_rate=0.3333,
        duplicate_event_rate=0.3333,
        event_id_reuse_rate=0.3333,
        classified_conflicting_event_rate=0.0,
        recorded_rate_available=True,
        end_to_end_coverage_available=False,
    )
    row.update(values)
    return row


def _authority_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_authority_status",
        authority_revision="registry_v1",
        business_reports_authoritative=True,
        registered_release_count=1,
        registered_question_count=60,
        registered_choice_count=180,
        detail_revision="detail_v1",
        detail_reports_authoritative=True,
    )
    row.update(values)
    return row


def _registry_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_registry_quality",
        registered_release_count=1,
        registered_question_count=60,
        registered_choice_count=180,
        registry_available=True,
        raw_event_count=10,
        authoritative_event_count=8,
        excluded_event_count=2,
        missing_release_event_count=1,
        unknown_release_event_count=1,
        question_not_in_release_event_count=0,
        raw_answer_count=5,
        authoritative_answer_count=4,
        unresolved_answer_count=1,
        invalid_selected_letter_answer_count=1,
        selected_candidate_mismatch_answer_count=0,
        unmatched_comment_count=1,
        unmatched_proposal_count=0,
        client_context_mismatch_event_count=1,
        client_correctness_mismatch_answer_count=1,
        registry_match_rate=0.8,
        answer_resolution_rate=0.8,
    )
    row.update(values)
    return row


def _surprise_question_row(**values: Any) -> dict[str, Any]:
    row = _row(
        reports.SURPRISE_QUESTIONS_VIEW,
        question_id="q_1",
        question_version=QUESTION_VERSION,
        release_id=RELEASE_ID,
        family="bigram_lm",
        dataset_id="bg_1",
        question_type="mixed",
        answered_attempt_count=3,
        rating_count=2,
        surprised_count=1,
        not_surprised_count=1,
        rating_coverage_rate=0.6667,
        observed_surprise_rate=0.5,
        posterior_mean=0.5,
        first_rating_at="2026-07-12T00:00:00Z",
        last_rating_at="2026-07-12T00:01:00Z",
    )
    row.update(values)
    return row


def _surprise_quality_row(**values: Any) -> dict[str, Any]:
    row = _row(
        reports.SURPRISE_QUALITY_VIEW,
        raw_reaction_count=8,
        valid_reaction_count=3,
        orphan_reaction_count=3,
        duplicate_reaction_count=2,
        registry_unmatched_reaction_count=1,
        invalid_payload_reaction_count=1,
        missing_prior_answer_reaction_count=1,
        unknown_release_reaction_count=1,
        counts_conserved=True,
        orphan_breakdown_conserved=True,
    )
    row.update(values)
    return row


def _resolution_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_event_resolution",
        event_id="evt_1",
        event_type="answer_submitted",
        occurred_at="2026-07-12T00:00:00Z",
        received_at="2026-07-12T00:00:01Z",
        session_id="anon_1",
        attempt_id="attempt_1",
        client_release_id="release_" + "a" * 64,
        registry_status="matched",
        answer_status="resolved",
        registry_id="registry_" + "b" * 64,
        release_id="release_" + "a" * 64,
        question_id="q_1",
        question_version="qv1_" + "c" * 64,
        family="bigram_lm",
        dataset_id="bg_1",
        question_type="mixed",
        selected_letter="A",
        client_selected_candidate_id="c_1",
        selected_candidate_id="c_1",
        authoritative_is_correct=True,
        client_is_correct=False,
        client_context_mismatch=False,
        client_correctness_mismatch=True,
    )
    row.update(values)
    return row


def _answer_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_answers",
        event_id="evt_answer_1",
        occurred_at="2026-07-12T00:00:00Z",
        received_at="2026-07-12T00:00:01Z",
        session_id="anon_1",
        attempt_id="attempt_1",
        question_id="q_1",
        question_version=QUESTION_VERSION,
        release_id=RELEASE_ID,
        family="bigram_lm",
        dataset_id="bg_1",
        question_type="mixed",
        selected_letter="A",
        client_selected_candidate_id="c_1",
        selected_candidate_id="c_1",
        answer_status="resolved",
        is_correct=True,
        client_is_correct=False,
        client_context_mismatch=False,
        client_correctness_mismatch=True,
    )
    row.update(values)
    return row


def _proposal_row(**values: Any) -> dict[str, Any]:
    row = _row(
        "feedback_report_proposals",
        event_id="evt_proposal_1",
        occurred_at="2026-07-12T00:00:00Z",
        received_at="2026-07-12T00:00:01Z",
        session_id="anon_1",
        attempt_id="attempt_1",
        question_id="q_1",
        question_version=QUESTION_VERSION,
        release_id=RELEASE_ID,
        family="bigram_lm",
        dataset_id="bg_1",
        question_type="mixed",
        setting_status="proposed",
        label="我的设置 😀",
        setting_json='{"budget":{"total_samples_seen":5120},"model":{"type":"mlp"}}',
        inherited_from_json='{"candidate_id":"c_1","exact_spec_match":false}',
        n_seeds=3,
        base_seed=0,
        error_type=None,
    )
    row.update(values)
    return row


def _envelope(
    view: str,
    rows: list[dict[str, Any]],
    *,
    total: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    return {
        "view": view,
        "rows": rows,
        "total": len(rows) if total is None else total,
        "limit": limit,
        "offset": offset,
        "request_id": "request-123",
    }


def _business_snapshot_envelope(
    *,
    limit: int = 10,
    question_total: int = 1,
) -> dict[str, Any]:
    summary = _row(
        "feedback_report_summary",
        answer_count=1,
        proposal_count=1,
        rejected_setting_count=1,
        comment_count=1,
        attempt_count=1,
        question_count=1,
    )
    rows = {
        "feedback_report_summary": [summary],
        "feedback_report_sessions": [
            _row(
                "feedback_report_sessions",
                session_id="anon_1",
                attempt_id="attempt_1",
            )
        ],
        "feedback_report_questions": [
            _row(
                "feedback_report_questions",
                question_id="q_1",
                question_version=QUESTION_VERSION,
                release_id=RELEASE_ID,
            )
        ],
        "feedback_report_answers": [_answer_row()],
        "feedback_report_proposals": [
            _proposal_row(),
            _proposal_row(
                event_id="evt_proposal_2",
                setting_status="rejected",
            ),
        ],
        "feedback_report_comments": [
            _row(
                "feedback_report_comments",
                event_id="evt_comment_1",
                comment_text="Useful",
            )
        ],
    }
    totals = {view: len(view_rows) for view, view_rows in rows.items()}
    totals["feedback_report_questions"] = question_total
    pages = {
        view: {
            "view": view,
            "rows": rows[view],
            "total": totals[view],
            "limit": limit,
            "offset": 0,
        }
        for view in reports.BUSINESS_REPORT_VIEWS
    }
    snapshot_row = {
        "snapshot_revision": "business_snapshot_v1",
        "snapshot_at": "2026-07-12T12:34:56.123456+00:00",
        "authority_revision": "registry_v1",
        "business_reports_authoritative": True,
        "detail_revision": "detail_v1",
        "detail_reports_authoritative": True,
        "registered_release_count": 1,
        "registered_question_count": 60,
        "registered_choice_count": 180,
        "pages_json": json.dumps(
            pages,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return _envelope(
        reports.BUSINESS_SNAPSHOT_VIEW,
        [snapshot_row],
        total=1,
        limit=limit,
        offset=0,
    )


def _client(**kwargs: Any) -> reports.ReportsClient:
    return reports.ReportsClient(
        url=REPORT_URL,
        read_token="read-only-secret",
        environ={},
        **kwargs,
    )


def test_public_view_inventory_uses_only_edge_report_keys() -> None:
    assert reports.REPORT_VIEWS == (
        "feedback_report_summary",
        "feedback_report_ingestion_summary",
        "feedback_report_authority_status",
        "feedback_report_business_snapshot",
        "feedback_report_registry_quality",
        "feedback_report_surprise_questions",
        "feedback_report_surprise_quality",
        "feedback_report_event_resolution",
        "feedback_report_sessions",
        "feedback_report_questions",
        "feedback_report_answers",
        "feedback_report_proposals",
        "feedback_report_comments",
    )
    columns = reports.copy_view_columns()
    assert tuple(columns) == reports.REPORT_VIEWS
    assert "known_answer_count" in columns["feedback_report_summary"]
    assert (
        "event_id_conflict_request_count"
        in columns["feedback_report_ingestion_summary"]
    )
    assert (
        "classified_conflicting_event_rate"
        in columns["feedback_report_ingestion_summary"]
    )
    assert "conflict_audit_event_count" in columns["feedback_report_ingestion_summary"]
    assert columns["feedback_report_ingestion_summary"][-2:] == (
        "recorded_rate_available",
        "end_to_end_coverage_available",
    )
    assert columns["feedback_report_authority_status"][:2] == (
        "authority_revision",
        "business_reports_authoritative",
    )
    assert columns["feedback_report_authority_status"][-2:] == (
        "detail_revision",
        "detail_reports_authoritative",
    )
    assert columns[reports.BUSINESS_SNAPSHOT_VIEW] == (
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
    assert columns["feedback_report_registry_quality"][-2:] == (
        "registry_match_rate",
        "answer_resolution_rate",
    )
    assert columns[reports.SURPRISE_QUESTIONS_VIEW] == (
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
    assert columns[reports.SURPRISE_QUALITY_VIEW] == (
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
    assert columns["feedback_report_event_resolution"][-2:] == (
        "client_context_mismatch",
        "client_correctness_mismatch",
    )
    assert "incorrect_answer_count" in columns["feedback_report_sessions"]
    assert "unknown_answer_count" in columns["feedback_report_questions"]
    assert columns["feedback_report_answers"][-5:] == (
        "answer_status",
        "is_correct",
        "client_is_correct",
        "client_context_mismatch",
        "client_correctness_mismatch",
    )
    assert columns["feedback_report_proposals"][-7:] == (
        "setting_status",
        "label",
        "setting_json",
        "inherited_from_json",
        "n_seeds",
        "base_seed",
        "error_type",
    )
    assert columns["feedback_report_comments"][-2:] == (
        "category",
        "comment_text",
    )


def test_config_explicit_values_override_environment_and_hide_token() -> None:
    environment = {
        reports.REPORTS_URL_ENV: "https://environment.example/report",
        reports.REPORTS_READ_TOKEN_ENV: "environment-secret",
        reports.REPORTS_TIMEOUT_ENV: "12.5",
    }
    config = reports.ReportsConfig.from_sources(
        url=f" {REPORT_URL}/ ",
        read_token="explicit-secret",
        timeout_seconds=2.5,
        environ=environment,
    )

    assert config.url == REPORT_URL
    assert config.read_token == "explicit-secret"
    assert config.timeout_seconds == 2.5
    assert config.is_configured
    assert "explicit-secret" not in repr(config)
    assert "environment-secret" not in repr(config)


def test_config_loads_environment_and_supports_unconfigured_detection() -> None:
    config = reports.ReportsConfig.from_env(
        {
            reports.REPORTS_URL_ENV: REPORT_URL,
            reports.REPORTS_READ_TOKEN_ENV: "server-read-token",
            reports.REPORTS_TIMEOUT_ENV: "8",
        }
    )
    assert config.is_configured
    assert config.timeout_seconds == 8

    client = reports.ReportsClient(environ={})
    assert not client.is_configured
    with pytest.raises(reports.ReportsNotConfiguredError, match="not configured"):
        client.build_query_url("feedback_report_summary")


@pytest.mark.parametrize(
    "url",
    [
        "relative/report",
        "ftp://example.test/report",
        "https://user:password@example.test/report",
        "https://example.test/report?token=secret",
        "https://example.test/report#fragment",
        "https://example.test:bad/report",
        "https://example.test/report\nInjected",
    ],
)
def test_config_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(reports.ReportsConfigurationError, match="URL"):
        reports.ReportsConfig(url=url, read_token="token")


@pytest.mark.parametrize(
    "timeout",
    [
        True,
        0,
        reports.MIN_TIMEOUT_SECONDS / 2,
        reports.MAX_TIMEOUT_SECONDS + 0.1,
        float("inf"),
        "not-a-number",
    ],
)
def test_config_enforces_bounded_timeout(timeout: Any) -> None:
    with pytest.raises(reports.ReportsConfigurationError, match="timeout"):
        reports.ReportsConfig(
            url=REPORT_URL,
            read_token="token",
            timeout_seconds=timeout,
        )


def test_config_rejects_token_newlines_and_mixed_constructor_sources() -> None:
    for token in ("token\r\ninjected", "token with spaces"):
        with pytest.raises(reports.ReportsConfigurationError, match="token"):
            reports.ReportsConfig(url=REPORT_URL, read_token=token)
    with pytest.raises(reports.ReportsConfigurationError, match="either config"):
        reports.ReportsClient(
            reports.ReportsConfig(url=REPORT_URL, read_token="token"),
            url=REPORT_URL,
        )


def test_query_url_uses_raw_encoded_edge_filters_and_safe_pagination() -> None:
    client = _client()
    injected_value = "release_1&limit=999"

    url = client.build_query_url(
        "feedback_report_questions",
        filters={
            "release_id": injected_value,
            "family": "bigram_lm",
            "from": "2026-07-01T00:00:00Z",
            "to": "2026-07-13T00:00:00+00:00",
        },
        limit=50,
        offset=100,
    )

    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
    assert (
        urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        == REPORT_URL
    )
    assert query == {
        "view": ["feedback_report_questions"],
        "release_id": [injected_value],
        "family": ["bigram_lm"],
        "from": ["2026-07-01T00:00:00Z"],
        "to": ["2026-07-13T00:00:00+00:00"],
        "limit": ["50"],
        "offset": ["100"],
    }
    assert all(not value[0].startswith("eq.") for value in query.values())
    assert "read-only-secret" not in url


def test_summary_query_has_no_order_expression() -> None:
    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(
            _client().build_query_url(
                "feedback_report_summary",
                filters={"question_id": "q_1"},
                limit=1,
            )
        ).query
    )
    assert query["view"] == ["feedback_report_summary"]
    assert query["question_id"] == ["q_1"]
    assert "order" not in query

    ingestion_query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(
            _client().build_query_url(
                "feedback_report_ingestion_summary",
                filters={
                    "from": "2026-07-01T00:00:00Z",
                    "to": "2026-07-02T00:00:00Z",
                    "request_id": INGEST_REQUEST_ID,
                },
                limit=1,
            )
        ).query
    )
    assert ingestion_query == {
        "view": ["feedback_report_ingestion_summary"],
        "from": ["2026-07-01T00:00:00Z"],
        "to": ["2026-07-02T00:00:00Z"],
        "request_id": [INGEST_REQUEST_ID],
        "limit": ["1"],
        "offset": ["0"],
    }


def test_registry_and_exact_event_query_allowlists_are_disjoint() -> None:
    authority_url = _client().build_query_url(
        "feedback_report_authority_status",
    )
    authority_query = urllib.parse.parse_qs(urllib.parse.urlsplit(authority_url).query)
    assert authority_query == {
        "view": ["feedback_report_authority_status"],
        "limit": ["200"],
        "offset": ["0"],
    }

    registry_url = _client().build_query_url(
        "feedback_report_registry_quality",
        filters={
            "from": "2026-07-01T00:00:00Z",
            "to": "2026-07-02T00:00:00Z",
        },
    )
    registry_query = urllib.parse.parse_qs(urllib.parse.urlsplit(registry_url).query)
    assert registry_query["view"] == ["feedback_report_registry_quality"]
    assert set(registry_query) == {"view", "from", "to", "limit", "offset"}

    resolution_url = _client().build_query_url(
        "feedback_report_event_resolution",
        filters={"event_id": "evt_exact_1"},
    )
    resolution_query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(resolution_url).query
    )
    assert resolution_query == {
        "view": ["feedback_report_event_resolution"],
        "event_id": ["evt_exact_1"],
        "limit": ["200"],
        "offset": ["0"],
    }

    for view, filters in (
        ("feedback_report_authority_status", {"from": "2026-07-01T00:00:00Z"}),
        ("feedback_report_authority_status", {"session_id": "anon_1"}),
        ("feedback_report_registry_quality", {"release_id": "release_1"}),
        ("feedback_report_registry_quality", {"attempt_id": "attempt_1"}),
        ("feedback_report_ingestion_summary", {"session_id": "anon_1"}),
        ("feedback_report_event_resolution", {}),
        ("feedback_report_event_resolution", {"from": "2026-07-01T00:00:00Z"}),
        (
            "feedback_report_event_resolution",
            {"event_id": "evt_1", "attempt_id": "attempt_1"},
        ),
        ("feedback_report_summary", {"event_id": "evt_1"}),
    ):
        with pytest.raises(reports.ReportsQueryError):
            _client().build_query_url(view, filters=filters)


def test_detail_queries_use_all_eight_common_filters() -> None:
    filters = {
        "release_id": RELEASE_ID,
        "family": "bigram_lm",
        "question_type": "mixed",
        "question_id": "q_1",
        "session_id": "anon_1",
        "attempt_id": "attempt_1",
        "from": "2026-07-01T00:00:00Z",
        "to": "2026-07-02T00:00:00Z",
    }
    for view in ("feedback_report_answers", "feedback_report_proposals"):
        url = _client().build_query_url(view, filters=filters, limit=17, offset=4)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        assert query == {
            "view": [view],
            **{key: [value] for key, value in filters.items()},
            "limit": ["17"],
            "offset": ["4"],
        }
        with pytest.raises(reports.ReportsQueryError):
            _client().build_query_url(view, filters={"category": "bug"})


def test_surprise_queries_use_eight_filters_with_distinct_pagination() -> None:
    filters = {
        "release_id": RELEASE_ID,
        "family": "bigram_lm",
        "question_type": "mixed",
        "question_id": "q_1",
        "session_id": "anon_1",
        "attempt_id": "attempt_1",
        "from": "2026-07-01T00:00:00Z",
        "to": "2026-07-02T00:00:00Z",
    }
    question_url = _client().build_query_url(
        reports.SURPRISE_QUESTIONS_VIEW,
        filters=filters,
        limit=17,
        offset=4,
    )
    question_query = urllib.parse.parse_qs(urllib.parse.urlsplit(question_url).query)
    assert question_query == {
        "view": [reports.SURPRISE_QUESTIONS_VIEW],
        **{key: [value] for key, value in filters.items()},
        "limit": ["17"],
        "offset": ["4"],
    }

    quality_url = _client().build_query_url(
        reports.SURPRISE_QUALITY_VIEW,
        filters=filters,
        limit=17,
    )
    quality_query = urllib.parse.parse_qs(urllib.parse.urlsplit(quality_url).query)
    assert quality_query == {
        "view": [reports.SURPRISE_QUALITY_VIEW],
        **{key: [value] for key, value in filters.items()},
        "limit": ["17"],
        "offset": ["0"],
    }
    with pytest.raises(reports.ReportsQueryError, match="offset"):
        _client().build_query_url(reports.SURPRISE_QUALITY_VIEW, offset=1)
    for view in (reports.SURPRISE_QUESTIONS_VIEW, reports.SURPRISE_QUALITY_VIEW):
        with pytest.raises(reports.ReportsQueryError, match="unsupported filter"):
            _client().build_query_url(view, filters={"category": "bug"})


def test_business_snapshot_query_uses_common_filters_and_zero_offset() -> None:
    filters = {
        "release_id": RELEASE_ID,
        "family": "bigram_lm",
        "question_type": "mixed",
        "question_id": "q_1",
        "session_id": "anon_1",
        "attempt_id": "attempt_1",
        "from": "2026-07-01T00:00:00Z",
        "to": "2026-07-02T00:00:00Z",
    }
    url = _client().build_query_url(
        reports.BUSINESS_SNAPSHOT_VIEW,
        filters=filters,
        limit=17,
    )
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query == {
        "view": [reports.BUSINESS_SNAPSHOT_VIEW],
        **{key: [value] for key, value in filters.items()},
        "limit": ["17"],
        "offset": ["0"],
    }
    with pytest.raises(reports.ReportsQueryError, match="offset"):
        _client().build_query_url(reports.BUSINESS_SNAPSHOT_VIEW, offset=1)
    with pytest.raises(reports.ReportsQueryError, match="unsupported filter"):
        _client().build_query_url(
            reports.BUSINESS_SNAPSHOT_VIEW,
            filters={"category": "suggestion"},
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view": "feedback_question_stats"},
        {
            "view": "feedback_report_questions",
            "filters": {"category": "bug"},
        },
        {
            "view": "feedback_report_comments",
            "filters": {"category": "private"},
        },
        {
            "view": "feedback_report_questions",
            "filters": {"from": "2026-07-01"},
        },
        {
            "view": "feedback_report_questions",
            "filters": {
                "from": "2026-07-02T00:00:00Z",
                "to": "2026-07-01T00:00:00Z",
            },
        },
        {
            "view": "feedback_report_questions",
            "filters": {"family": " untrimmed "},
        },
        {
            "view": "feedback_report_questions",
            "filters": {"family": True},
        },
        {
            "view": "feedback_report_questions",
            "filters": {"session_id": ""},
        },
        {
            "view": "feedback_report_questions",
            "filters": {"attempt_id": "x" * 201},
        },
        {
            "view": "feedback_report_answers",
            "filters": {"family": "\ud800"},
        },
        {
            "view": "feedback_report_questions",
            "filters": {1: "invalid-key"},
        },
        {"view": "feedback_report_questions", "limit": 0},
        {"view": "feedback_report_questions", "limit": True},
        {"view": "feedback_report_questions", "limit": reports.MAX_LIMIT + 1},
        {"view": "feedback_report_questions", "offset": -1},
        {"view": "feedback_report_questions", "offset": True},
        {"view": "feedback_report_questions", "offset": reports.MAX_OFFSET + 1},
        {"view": "feedback_report_summary", "offset": 1},
        {"view": "feedback_report_ingestion_summary", "offset": 1},
        {"view": "feedback_report_authority_status", "offset": 1},
        {
            "view": "feedback_report_ingestion_summary",
            "filters": {"release_id": "release_1"},
        },
        {
            "view": "feedback_report_summary",
            "filters": {"request_id": INGEST_REQUEST_ID},
        },
        {
            "view": "feedback_report_ingestion_summary",
            "filters": {"request_id": "not-a-uuid"},
        },
        {
            "view": "feedback_report_ingestion_summary",
            "filters": {"request_id": "72aee12d-7742-04ea-b3d9-f056ae5c8ac2"},
        },
        {
            "view": "feedback_report_ingestion_summary",
            "filters": {"request_id": "72aee12d-7742-44ea-73d9-f056ae5c8ac2"},
        },
    ],
)
def test_query_builder_rejects_values_outside_allowlists(
    kwargs: dict[str, Any],
) -> None:
    view = kwargs.pop("view")
    with pytest.raises(reports.ReportsQueryError):
        _client().build_query_url(view, **kwargs)


def test_row_validation_requires_exact_view_schema_and_detaches() -> None:
    source = [
        _row(
            "feedback_report_sessions",
            session_id="anon_1",
            release_ids=["release_1"],
            event_count=3,
        )
    ]
    resolved = reports.validate_report_rows("feedback_report_sessions", source)

    assert resolved == source
    resolved[0]["release_ids"].append("release_2")
    assert source[0]["release_ids"] == ["release_1"]

    missing = dict(source[0])
    missing.pop("event_count")
    with pytest.raises(reports.ReportsResponseError, match="missing: event_count"):
        reports.validate_report_rows("feedback_report_sessions", [missing])

    unexpected = {**source[0], "service_role_key": "must-not-pass"}
    with pytest.raises(reports.ReportsResponseError, match="unexpected"):
        reports.validate_report_rows("feedback_report_sessions", [unexpected])


def test_ingestion_summary_validation_enforces_counts_rates_and_coverage() -> None:
    valid = _ingestion_row()
    assert reports.validate_report_rows(
        "feedback_report_ingestion_summary", [valid]
    ) == [valid]

    empty = _ingestion_row(
        recorded_request_count=0,
        first_started_at=None,
        last_finished_at=None,
        success_request_count=0,
        client_rejection_count=0,
        service_failure_count=0,
        event_id_conflict_request_count=0,
        accepted_event_count=0,
        duplicate_event_count=0,
        idempotent_duplicate_event_count=0,
        unclassified_duplicate_event_count=0,
        conflicting_event_count=0,
        conflict_audit_event_count=0,
        event_id_reuse_count=0,
        classified_event_count=0,
        known_event_result_count=0,
        request_failure_rate=None,
        duplicate_event_rate=None,
        event_id_reuse_rate=None,
        classified_conflicting_event_rate=None,
        recorded_rate_available=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_ingestion_summary", [empty]
    ) == [empty]

    invalid_rows = (
        _ingestion_row(recorded_request_count=True),
        _ingestion_row(success_request_count=1),
        _ingestion_row(event_id_conflict_request_count=2),
        _ingestion_row(known_event_result_count=2),
        _ingestion_row(idempotent_duplicate_event_count=0),
        _ingestion_row(unclassified_duplicate_event_count=1),
        _ingestion_row(conflicting_event_count=1),
        _ingestion_row(conflict_audit_event_count=1),
        _ingestion_row(event_id_reuse_count=2),
        _ingestion_row(classified_event_count=0),
        _ingestion_row(
            accepted_event_count=0,
            duplicate_event_count=0,
            idempotent_duplicate_event_count=0,
            event_id_reuse_count=0,
            known_event_result_count=0,
            duplicate_event_rate=None,
            event_id_reuse_rate=None,
            classified_conflicting_event_rate=None,
        ),
        _ingestion_row(request_failure_rate=0.5),
        _ingestion_row(duplicate_event_rate=float("nan")),
        _ingestion_row(event_id_reuse_rate=0.5),
        _ingestion_row(classified_conflicting_event_rate=None),
        _ingestion_row(first_started_at=None),
        _ingestion_row(recorded_rate_available=False),
        _ingestion_row(end_to_end_coverage_available=True),
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows("feedback_report_ingestion_summary", [invalid])


def test_registry_quality_validation_enforces_conservation_and_rates() -> None:
    valid = _registry_row()
    assert reports.validate_report_rows(
        "feedback_report_registry_quality", [valid]
    ) == [valid]

    empty = _registry_row(
        registered_release_count=0,
        registered_question_count=0,
        registered_choice_count=0,
        registry_available=False,
        raw_event_count=0,
        authoritative_event_count=0,
        excluded_event_count=0,
        missing_release_event_count=0,
        unknown_release_event_count=0,
        raw_answer_count=0,
        authoritative_answer_count=0,
        unresolved_answer_count=0,
        invalid_selected_letter_answer_count=0,
        unmatched_comment_count=0,
        client_context_mismatch_event_count=0,
        client_correctness_mismatch_answer_count=0,
        registry_match_rate=None,
        answer_resolution_rate=None,
    )
    assert reports.validate_report_rows(
        "feedback_report_registry_quality", [empty]
    ) == [empty]

    invalid_rows = (
        _registry_row(registry_available=False),
        _registry_row(registered_choice_count=100),
        _registry_row(excluded_event_count=1),
        _registry_row(missing_release_event_count=0),
        _registry_row(unresolved_answer_count=0),
        _registry_row(invalid_selected_letter_answer_count=2),
        _registry_row(unmatched_comment_count=3),
        _registry_row(client_context_mismatch_event_count=9),
        _registry_row(client_correctness_mismatch_answer_count=5),
        _registry_row(registry_match_rate=0.7),
        _registry_row(answer_resolution_rate=None),
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows("feedback_report_registry_quality", [invalid])


def test_surprise_question_validation_enforces_exact_counts_rates_and_times() -> None:
    valid = _surprise_question_row()
    assert reports.validate_report_rows(reports.SURPRISE_QUESTIONS_VIEW, [valid]) == [
        valid
    ]

    unrated = _surprise_question_row(
        answered_attempt_count=1,
        rating_count=0,
        surprised_count=0,
        not_surprised_count=0,
        rating_coverage_rate=0.0,
        observed_surprise_rate=None,
        posterior_mean=0.5,
        first_rating_at=None,
        last_rating_at=None,
    )
    assert reports.validate_report_rows(reports.SURPRISE_QUESTIONS_VIEW, [unrated]) == [
        unrated
    ]

    huge_count = 9_007_199_254_740_993
    huge = _surprise_question_row(
        answered_attempt_count=huge_count,
        rating_count=huge_count,
        surprised_count=huge_count,
        not_surprised_count=0,
        rating_coverage_rate=1.0,
        observed_surprise_rate=1.0,
        posterior_mean=1.0,
    )
    assert (
        reports.validate_report_rows(reports.SURPRISE_QUESTIONS_VIEW, [huge])[0][
            "rating_count"
        ]
        == huge_count
    )

    invalid_rows = (
        _surprise_question_row(answered_attempt_count=True),
        _surprise_question_row(answered_attempt_count=0),
        _surprise_question_row(rating_count=3),
        _surprise_question_row(surprised_count=2),
        _surprise_question_row(rating_coverage_rate=0.5),
        _surprise_question_row(observed_surprise_rate=None),
        _surprise_question_row(posterior_mean=0.6),
        _surprise_question_row(first_rating_at=None),
        _surprise_question_row(last_rating_at="2026-07-11T23:59:59Z"),
        _surprise_question_row(release_id="release_client_claim"),
        _surprise_question_row(question_version="qv0_client_claim"),
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows(reports.SURPRISE_QUESTIONS_VIEW, [invalid])


def test_surprise_quality_validation_enforces_both_conservation_equations() -> None:
    valid = _surprise_quality_row()
    assert reports.validate_report_rows(reports.SURPRISE_QUALITY_VIEW, [valid]) == [
        valid
    ]
    empty = _surprise_quality_row(
        raw_reaction_count=0,
        valid_reaction_count=0,
        orphan_reaction_count=0,
        duplicate_reaction_count=0,
        registry_unmatched_reaction_count=0,
        invalid_payload_reaction_count=0,
        missing_prior_answer_reaction_count=0,
        unknown_release_reaction_count=0,
    )
    assert reports.validate_report_rows(reports.SURPRISE_QUALITY_VIEW, [empty]) == [
        empty
    ]

    invalid_rows = (
        _surprise_quality_row(raw_reaction_count=True),
        _surprise_quality_row(raw_reaction_count=7),
        _surprise_quality_row(orphan_reaction_count=2),
        _surprise_quality_row(counts_conserved=False),
        _surprise_quality_row(orphan_breakdown_conserved=1),
        _surprise_quality_row(unknown_release_reaction_count=2),
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows(reports.SURPRISE_QUALITY_VIEW, [invalid])


def test_authority_status_validation_proves_revision_and_inventory() -> None:
    valid = _authority_row()
    assert reports.validate_report_rows(
        "feedback_report_authority_status", [valid]
    ) == [valid]

    empty = _authority_row(
        registered_release_count=0,
        registered_question_count=0,
        registered_choice_count=0,
    )
    assert reports.validate_report_rows(
        "feedback_report_authority_status", [empty]
    ) == [empty]

    for invalid in (
        _authority_row(authority_revision="registry_v0"),
        _authority_row(business_reports_authoritative=False),
        _authority_row(detail_revision="detail_v0"),
        _authority_row(detail_reports_authoritative=False),
        _authority_row(registered_release_count=True),
        _authority_row(registered_question_count=0),
        _authority_row(registered_choice_count=100),
        _authority_row(
            registered_release_count=0,
            registered_question_count=1,
        ),
    ):
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows("feedback_report_authority_status", [invalid])


def test_exact_event_resolution_validation_is_fail_closed() -> None:
    valid = _resolution_row()
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [valid]
    ) == [valid]

    missing_client_correctness = _resolution_row(
        client_is_correct=None,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [missing_client_correctness]
    ) == [missing_client_correctness]

    presentation = _resolution_row(
        event_type="question_presented",
        answer_status="not_answer",
        selected_letter=None,
        client_selected_candidate_id=None,
        selected_candidate_id=None,
        authoritative_is_correct=None,
        client_is_correct=None,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [presentation]
    ) == [presentation]

    unmatched = _resolution_row(
        client_release_id="release_unknown",
        registry_status="unknown_release",
        answer_status="unresolved_registry",
        registry_id=None,
        release_id=None,
        question_id=None,
        question_version=None,
        family=None,
        dataset_id=None,
        question_type=None,
        selected_candidate_id=None,
        authoritative_is_correct=None,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [unmatched]
    ) == [unmatched]

    not_found = _row(
        "feedback_report_event_resolution",
        event_id="evt_absent",
        registry_status="not_found",
        answer_status="not_found",
        client_context_mismatch=False,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [not_found]
    ) == [not_found]

    legacy_missing_letter = _resolution_row(
        answer_status="invalid_selected_letter",
        selected_letter=None,
        selected_candidate_id=None,
        authoritative_is_correct=None,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_event_resolution", [legacy_missing_letter]
    ) == [legacy_missing_letter]

    invalid_rows = (
        _resolution_row(registry_status="trusted_by_client"),
        _resolution_row(answer_status="client_says_correct"),
        _resolution_row(release_id=None),
        _resolution_row(authoritative_is_correct=None),
        _resolution_row(selected_candidate_id=None),
        _resolution_row(client_correctness_mismatch=None),
        unmatched | {"registry_id": "registry_bad"},
        unmatched | {"answer_status": "resolved"},
        not_found | {"event_type": "comment_submitted"},
        not_found | {"client_context_mismatch": True},
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows("feedback_report_event_resolution", [invalid])


def test_answer_detail_validation_enforces_authoritative_status_facts() -> None:
    resolved = _answer_row()
    assert reports.validate_report_rows("feedback_report_answers", [resolved]) == [
        resolved
    ]

    missing_client_correctness = _answer_row(
        client_is_correct=None,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_answers", [missing_client_correctness]
    ) == [missing_client_correctness]

    invalid_letter = _answer_row(
        selected_letter="Z",
        client_selected_candidate_id="client_guess",
        selected_candidate_id=None,
        answer_status="invalid_selected_letter",
        is_correct=None,
        client_is_correct=False,
        client_correctness_mismatch=False,
    )
    mismatch = _answer_row(
        client_selected_candidate_id="client_wrong_candidate",
        selected_candidate_id="registry_candidate",
        answer_status="selected_candidate_mismatch",
        is_correct=None,
        client_is_correct=True,
        client_correctness_mismatch=False,
    )
    assert reports.validate_report_rows(
        "feedback_report_answers", [invalid_letter, mismatch]
    ) == [invalid_letter, mismatch]

    invalid_rows = (
        _answer_row(answer_status="client_reported"),
        _answer_row(release_id="release_bad"),
        _answer_row(question_version="qv1_bad"),
        _answer_row(selected_candidate_id=None),
        _answer_row(client_selected_candidate_id="different"),
        _answer_row(is_correct=None),
        invalid_letter | {"selected_candidate_id": "c_1"},
        mismatch | {"client_selected_candidate_id": "registry_candidate"},
        mismatch | {"is_correct": False},
        _answer_row(client_correctness_mismatch=False),
        _answer_row(client_context_mismatch=None),
        _answer_row(event_id="evt_\ud800"),
    )
    for invalid in invalid_rows:
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_report_rows("feedback_report_answers", [invalid])


def test_proposal_detail_validation_keeps_strict_json_text_unchanged() -> None:
    valid = _proposal_row()
    resolved = reports.validate_report_rows("feedback_report_proposals", [valid])
    assert resolved == [valid]
    assert resolved[0]["setting_json"] == valid["setting_json"]
    assert resolved[0]["inherited_from_json"] == valid["inherited_from_json"]
    assert isinstance(resolved[0]["setting_json"], str)

    legacy = _proposal_row(
        setting_status="rejected",
        setting_json=None,
        inherited_from_json=None,
        n_seeds=None,
        base_seed=None,
        error_type=" arbitrary legacy error\n ",
    )
    assert reports.validate_report_rows("feedback_report_proposals", [legacy]) == [
        legacy
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.__setitem__("setting_status", "completed"),
        lambda row: row.__setitem__("setting_json", {}),
        lambda row: row.__setitem__("setting_json", "[]"),
        lambda row: row.__setitem__("setting_json", '{"x":1,"x":2}'),
        lambda row: row.__setitem__("setting_json", '{"unsafe":9007199254740992}'),
        lambda row: row.__setitem__("setting_json", '{"unsafe":9007199254740992.0}'),
        lambda row: row.__setitem__("setting_json", '{"huge":1e999}'),
        lambda row: row.__setitem__("setting_json", '{"bad":NaN}'),
        lambda row: row.__setitem__("setting_json", '{"bad":"\\ud800"}'),
        lambda row: row.__setitem__("inherited_from_json", '"candidate"'),
        lambda row: row.__setitem__("n_seeds", True),
        lambda row: row.__setitem__("n_seeds", 1.5),
        lambda row: row.__setitem__("n_seeds", 2_147_483_648),
        lambda row: row.__setitem__("base_seed", -2_147_483_649),
        lambda row: row.__setitem__("label", "bad\ud800label"),
        lambda row: row.__setitem__("error_type", "bad\ud800error"),
    ],
)
def test_proposal_detail_validation_rejects_unsafe_rows(mutation: Any) -> None:
    row = _proposal_row()
    mutation(row)
    with pytest.raises(reports.ReportsResponseError):
        reports.validate_report_rows("feedback_report_proposals", [row])


def test_conflict_reuse_rate_includes_atomically_rejected_new_events() -> None:
    conflict = _ingestion_row(
        recorded_request_count=1,
        first_started_at="2026-07-12T00:00:00Z",
        last_finished_at="2026-07-12T00:00:01Z",
        success_request_count=0,
        client_rejection_count=1,
        service_failure_count=0,
        event_id_conflict_request_count=1,
        accepted_event_count=0,
        duplicate_event_count=1,
        idempotent_duplicate_event_count=1,
        unclassified_duplicate_event_count=0,
        conflicting_event_count=1,
        conflict_audit_event_count=1,
        event_id_reuse_count=2,
        classified_event_count=3,
        known_event_result_count=1,
        request_failure_rate=1.0,
        duplicate_event_rate=1.0,
        event_id_reuse_rate=0.6667,
        classified_conflicting_event_rate=0.5,
        recorded_rate_available=True,
        end_to_end_coverage_available=False,
    )

    assert reports.validate_report_rows(
        "feedback_report_ingestion_summary", [conflict]
    ) == [conflict]


def test_ingestion_summary_preserves_unclassified_legacy_duplicates() -> None:
    legacy = _ingestion_row(
        recorded_request_count=1,
        first_started_at="2026-07-11T00:00:00Z",
        last_finished_at="2026-07-11T00:00:01Z",
        success_request_count=1,
        client_rejection_count=0,
        service_failure_count=0,
        event_id_conflict_request_count=0,
        accepted_event_count=0,
        duplicate_event_count=1,
        idempotent_duplicate_event_count=0,
        unclassified_duplicate_event_count=1,
        conflicting_event_count=0,
        conflict_audit_event_count=0,
        event_id_reuse_count=1,
        classified_event_count=1,
        known_event_result_count=1,
        request_failure_rate=0.0,
        duplicate_event_rate=1.0,
        event_id_reuse_rate=1.0,
        classified_conflicting_event_rate=None,
        recorded_rate_available=True,
        end_to_end_coverage_available=False,
    )

    assert reports.validate_report_rows(
        "feedback_report_ingestion_summary", [legacy]
    ) == [legacy]


def test_ingestion_summary_allows_pure_conflict_without_known_results() -> None:
    conflict = _ingestion_row(
        recorded_request_count=1,
        first_started_at="2026-07-12T00:00:00Z",
        last_finished_at="2026-07-12T00:00:01Z",
        success_request_count=0,
        client_rejection_count=1,
        service_failure_count=0,
        event_id_conflict_request_count=1,
        accepted_event_count=0,
        duplicate_event_count=0,
        idempotent_duplicate_event_count=0,
        unclassified_duplicate_event_count=0,
        conflicting_event_count=1,
        conflict_audit_event_count=1,
        event_id_reuse_count=1,
        classified_event_count=1,
        known_event_result_count=0,
        request_failure_rate=1.0,
        duplicate_event_rate=None,
        event_id_reuse_rate=1.0,
        classified_conflicting_event_rate=1.0,
        recorded_rate_available=True,
        end_to_end_coverage_available=False,
    )

    assert reports.validate_report_rows(
        "feedback_report_ingestion_summary", [conflict]
    ) == [conflict]


@pytest.mark.parametrize(
    "rows",
    [
        {},
        ["not-an-object"],
        [_row("feedback_report_questions", accuracy=float("nan"))],
    ],
)
def test_row_validation_rejects_non_json_array_rows(rows: Any) -> None:
    with pytest.raises(reports.ReportsResponseError):
        reports.validate_report_rows("feedback_report_questions", rows)


def test_response_validation_preserves_page_metadata() -> None:
    view = "feedback_report_questions"
    value = _envelope(
        view,
        [_row(view, question_id="q_1", answer_count=3)],
        total=4,
        limit=2,
        offset=0,
    )
    page = reports.validate_report_response(
        view,
        value,
        expected_limit=2,
        expected_offset=0,
    )

    assert page.view == view
    assert page.total == 4
    assert page.limit == 2
    assert page.offset == 0
    assert page.request_id == "request-123"
    assert not page.is_complete
    copied = page.rows_copy()
    copied[0]["question_id"] = "changed"
    assert page.rows[0]["question_id"] == "q_1"
    assert page.to_dict()["rows"][0]["question_id"] == "q_1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("total"), "missing fields"),
        (lambda value: value.__setitem__("debug", "secret"), "unexpected fields"),
        (
            lambda value: value.__setitem__("view", "feedback_report_sessions"),
            "does not match",
        ),
        (lambda value: value.__setitem__("total", True), "total"),
        (lambda value: value.__setitem__("total", 0), "smaller than rows"),
        (lambda value: value.__setitem__("limit", 0), "limit"),
        (lambda value: value.__setitem__("offset", -1), "offset"),
        (lambda value: value.__setitem__("request_id", " bad "), "request_id"),
    ],
)
def test_response_validation_rejects_invalid_envelopes(
    mutation: Any,
    message: str,
) -> None:
    view = "feedback_report_questions"
    value = _envelope(view, [_row(view, question_id="q_1")])
    mutation(value)
    with pytest.raises(reports.ReportsResponseError, match=message):
        reports.validate_report_response(view, value)


def test_summary_response_always_requires_exactly_one_row() -> None:
    view = "feedback_report_summary"
    valid = _envelope(view, [_row(view, event_count=0)], total=1, limit=1)
    page = reports.validate_report_response(view, valid)
    assert len(page.rows) == 1
    assert page.is_complete

    for rows in ([], [_row(view), _row(view)]):
        with pytest.raises(reports.ReportsResponseError, match="exactly one"):
            reports.validate_report_response(
                view,
                _envelope(view, rows, total=len(rows), limit=max(1, len(rows))),
            )

    ingestion_view = "feedback_report_ingestion_summary"
    ingestion_page = reports.validate_report_response(
        ingestion_view,
        _envelope(ingestion_view, [_ingestion_row()], total=1, limit=1),
    )
    assert len(ingestion_page.rows) == 1
    for rows in ([], [_ingestion_row(), _ingestion_row()]):
        with pytest.raises(reports.ReportsResponseError, match="exactly one"):
            reports.validate_report_response(
                ingestion_view,
                _envelope(
                    ingestion_view,
                    rows,
                    total=len(rows),
                    limit=max(1, len(rows)),
                ),
            )
    for total, offset in ((2, 0), (1, 1)):
        with pytest.raises(
            reports.ReportsResponseError, match="total one and offset zero"
        ):
            reports.validate_report_response(
                ingestion_view,
                _envelope(
                    ingestion_view,
                    [_ingestion_row()],
                    total=total,
                    limit=1,
                    offset=offset,
                ),
            )

    surprise_quality_page = reports.validate_report_response(
        reports.SURPRISE_QUALITY_VIEW,
        _envelope(
            reports.SURPRISE_QUALITY_VIEW,
            [_surprise_quality_row()],
            total=1,
            limit=1,
        ),
    )
    assert len(surprise_quality_page.rows) == 1
    with pytest.raises(reports.ReportsResponseError, match="exactly one"):
        reports.validate_report_response(
            reports.SURPRISE_QUALITY_VIEW,
            _envelope(
                reports.SURPRISE_QUALITY_VIEW,
                [],
                total=0,
                limit=1,
            ),
        )


def test_business_snapshot_validation_preserves_atomic_pages_and_metadata() -> None:
    snapshot = reports.validate_business_snapshot_response(
        _business_snapshot_envelope(),
        expected_limit=10,
        release_filter_active=True,
    )

    assert isinstance(snapshot, reports.BusinessSnapshot)
    assert snapshot.snapshot_revision == "business_snapshot_v1"
    assert snapshot.snapshot_at == "2026-07-12T12:34:56.123456+00:00"
    assert snapshot.authority_revision == "registry_v1"
    assert snapshot.detail_revision == "detail_v1"
    assert snapshot.request_id == "request-123"
    assert tuple(snapshot.pages) == reports.BUSINESS_REPORT_VIEWS
    assert snapshot.pages["feedback_report_answers"].total == 1
    assert snapshot.pages["feedback_report_proposals"].total == 2

    detached = snapshot.to_dict()
    assert set(detached["pages"]["feedback_report_answers"]) == {
        "view",
        "rows",
        "total",
        "limit",
        "offset",
    }
    detached["pages"]["feedback_report_answers"]["rows"][0]["event_id"] = "changed"
    assert snapshot.pages["feedback_report_answers"].rows[0]["event_id"] == (
        "evt_answer_1"
    )


def test_business_snapshot_preserves_exact_postgresql_bigint_authority_counts() -> None:
    response = _business_snapshot_envelope()
    response["rows"][0].update(
        {
            "registered_release_count": 9_007_199_254_740_993,
            "registered_question_count": 9_007_199_254_740_995,
            "registered_choice_count": 18_014_398_509_481_990,
        }
    )

    snapshot = reports.validate_business_snapshot_response(response)

    assert snapshot.registered_release_count == 9_007_199_254_740_993
    assert snapshot.registered_question_count == 9_007_199_254_740_995
    assert snapshot.registered_choice_count == 18_014_398_509_481_990


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("snapshot_revision", "snapshot_v1", "business_snapshot_v1"),
        ("snapshot_at", "not-a-timestamp", "snapshot_at"),
        ("authority_revision", "registry_v0", "registry_v1"),
        ("business_reports_authoritative", False, "business cutover"),
        ("detail_revision", "detail_v0", "detail_v1"),
        ("detail_reports_authoritative", False, "detail-report cutover"),
        ("registered_question_count", -1, "registered_question_count"),
    ],
)
def test_business_snapshot_rejects_invalid_outer_authority_metadata(
    field: str,
    value: Any,
    message: str,
) -> None:
    response = _business_snapshot_envelope()
    response["rows"][0][field] = value
    with pytest.raises(reports.ReportsResponseError, match=message):
        reports.validate_business_snapshot_response(response)


def test_business_snapshot_rejects_missing_or_unknown_outer_row_fields() -> None:
    missing = _business_snapshot_envelope()
    missing["rows"][0].pop("pages_json")
    with pytest.raises(reports.ReportsResponseError, match="missing"):
        reports.validate_business_snapshot_response(missing)

    unknown = _business_snapshot_envelope()
    unknown["rows"][0]["debug"] = "not allowed"
    with pytest.raises(reports.ReportsResponseError, match="unexpected"):
        reports.validate_business_snapshot_response(unknown)


def test_business_snapshot_pages_json_is_strict_and_interoperable() -> None:
    valid = _business_snapshot_envelope()["rows"][0]["pages_json"]
    decoded = json.loads(valid)
    duplicate = (
        valid[:-1]
        + ',"feedback_report_summary":'
        + json.dumps(decoded["feedback_report_summary"], separators=(",", ":"))
        + "}"
    )
    invalid_values: list[Any] = [
        None,
        {},
        "[]",
        duplicate,
        valid.replace('"offset":0', '"offset":NaN', 1),
        valid.replace('"offset":0', '"offset":1e999', 1),
        valid.replace('"offset":0', '"offset":9007199254740992', 1),
        valid.replace('"offset":0', '"offset":9007199254740992.0', 1),
        valid.replace(
            '"view":"feedback_report_summary"',
            '"view":"\\ud800"',
            1,
        ),
    ]

    for pages_json in invalid_values:
        response = _business_snapshot_envelope()
        response["rows"][0]["pages_json"] = pages_json
        with pytest.raises(reports.ReportsResponseError):
            reports.validate_business_snapshot_response(response)


def test_business_snapshot_requires_exact_six_page_envelopes() -> None:
    missing_view = _business_snapshot_envelope()
    pages = json.loads(missing_view["rows"][0]["pages_json"])
    pages.pop("feedback_report_comments")
    missing_view["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="invalid views"):
        reports.validate_business_snapshot_response(missing_view)

    extra_view = _business_snapshot_envelope()
    pages = json.loads(extra_view["rows"][0]["pages_json"])
    pages["feedback_report_debug"] = pages["feedback_report_summary"]
    extra_view["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="invalid views"):
        reports.validate_business_snapshot_response(extra_view)

    extra_field = _business_snapshot_envelope()
    pages = json.loads(extra_field["rows"][0]["pages_json"])
    pages["feedback_report_questions"]["request_id"] = "nested-request"
    extra_field["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="invalid fields"):
        reports.validate_business_snapshot_response(extra_field)

    invalid_row = _business_snapshot_envelope()
    pages = json.loads(invalid_row["rows"][0]["pages_json"])
    pages["feedback_report_answers"]["rows"][0].pop("event_id")
    invalid_row["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="invalid columns"):
        reports.validate_business_snapshot_response(invalid_row)


@pytest.mark.parametrize(
    ("field", "value", "view"),
    [
        ("answer_count", 2, "feedback_report_answers"),
        ("proposal_count", 0, "feedback_report_proposals"),
        ("comment_count", 0, "feedback_report_comments"),
        ("attempt_count", 0, "feedback_report_sessions"),
    ],
)
def test_business_snapshot_enforces_cross_page_conservation(
    field: str,
    value: int,
    view: str,
) -> None:
    response = _business_snapshot_envelope()
    pages = json.loads(response["rows"][0]["pages_json"])
    pages["feedback_report_summary"]["rows"][0][field] = value
    response["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match=view):
        reports.validate_business_snapshot_response(response)


def test_business_snapshot_question_total_respects_release_filter_semantics() -> None:
    multi_release = _business_snapshot_envelope(question_total=2)
    snapshot = reports.validate_business_snapshot_response(multi_release)
    assert snapshot.pages["feedback_report_questions"].total == 2

    with pytest.raises(reports.ReportsResponseError, match="equal"):
        reports.validate_business_snapshot_response(
            multi_release,
            release_filter_active=True,
        )

    too_small = _business_snapshot_envelope()
    pages = json.loads(too_small["rows"][0]["pages_json"])
    pages["feedback_report_summary"]["rows"][0]["question_count"] = 2
    too_small["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="not smaller"):
        reports.validate_business_snapshot_response(too_small)


def test_business_snapshot_requires_common_requested_limit_and_zero_offsets() -> None:
    inconsistent = _business_snapshot_envelope()
    pages = json.loads(inconsistent["rows"][0]["pages_json"])
    pages["feedback_report_comments"]["limit"] = 9
    inconsistent["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="common page limit"):
        reports.validate_business_snapshot_response(inconsistent)

    wrong_outer_limit = _business_snapshot_envelope()
    pages = json.loads(wrong_outer_limit["rows"][0]["pages_json"])
    for nested in pages.values():
        nested["limit"] = 9
    wrong_outer_limit["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="outer response"):
        reports.validate_business_snapshot_response(wrong_outer_limit)

    nonzero_offset = _business_snapshot_envelope()
    pages = json.loads(nonzero_offset["rows"][0]["pages_json"])
    pages["feedback_report_comments"]["offset"] = 1
    nonzero_offset["rows"][0]["pages_json"] = json.dumps(pages)
    with pytest.raises(reports.ReportsResponseError, match="requested offset"):
        reports.validate_business_snapshot_response(nonzero_offset)


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.headers = Message()

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_client_fetch_page_uses_get_bearer_timeout_and_validates_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    view = "feedback_report_comments"
    body = json.dumps(
        _envelope(
            view,
            [_row(view, event_id="evt_1", comment_text="Useful")],
            limit=25,
        )
    ).encode()

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(body)

    monkeypatch.setattr(reports_client, "_open_report_request", fake_urlopen)
    client = _client(timeout_seconds=3.5)
    page = client.fetch_page(
        view,
        filters={"category": "suggestion"},
        limit=25,
    )

    request = captured["request"]
    assert request.method == "GET"
    assert request.data is None
    assert request.get_header("Authorization") == "Bearer read-only-secret"
    assert request.get_header("Apikey") is None
    assert captured["timeout"] == 3.5
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
        "view": [view],
        "category": ["suggestion"],
        "limit": ["25"],
        "offset": ["0"],
    }
    assert page.total == 1
    assert page.is_complete
    assert page.rows[0]["event_id"] == "evt_1"


def test_client_fetch_business_snapshot_uses_exactly_one_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[Any, float]] = []
    payload = _business_snapshot_envelope(limit=25)
    payload["rows"][0].update(
        {
            "registered_release_count": 9_007_199_254_740_993,
            "registered_question_count": 9_007_199_254_740_995,
            "registered_choice_count": 18_014_398_509_481_990,
        }
    )
    body = json.dumps(payload).encode()

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        captured.append((request, timeout))
        return _FakeResponse(body)

    monkeypatch.setattr(reports_client, "_open_report_request", fake_urlopen)
    snapshot = _client(timeout_seconds=4).fetch_business_snapshot(
        filters={"release_id": RELEASE_ID},
        limit=25,
    )

    assert len(captured) == 1
    request, timeout = captured[0]
    assert request.method == "GET"
    assert request.get_header("Authorization") == "Bearer read-only-secret"
    assert timeout == 4
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {
        "view": [reports.BUSINESS_SNAPSHOT_VIEW],
        "release_id": [RELEASE_ID],
        "limit": ["25"],
        "offset": ["0"],
    }
    assert snapshot.request_id == "request-123"
    assert snapshot.registered_release_count == 9_007_199_254_740_993
    assert snapshot.registered_question_count == 9_007_199_254_740_995
    assert snapshot.registered_choice_count == 18_014_398_509_481_990
    assert snapshot.pages["feedback_report_questions"].total == 1


def test_client_fetches_surprise_pages_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        del timeout
        captured.append(request)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
        view = query["view"][0]
        row = (
            _surprise_question_row()
            if view == reports.SURPRISE_QUESTIONS_VIEW
            else _surprise_quality_row()
        )
        return _FakeResponse(
            json.dumps(
                _envelope(view, [row], total=1, limit=int(query["limit"][0]))
            ).encode()
        )

    monkeypatch.setattr(reports_client, "_open_report_request", fake_urlopen)
    client = _client()
    filters = {"release_id": RELEASE_ID, "attempt_id": "attempt_1"}
    questions = client.fetch_surprise_questions(
        filters=filters,
        limit=25,
    )
    quality = client.fetch_surprise_quality(
        filters=filters,
        limit=25,
    )

    assert len(captured) == 2
    assert questions.view == reports.SURPRISE_QUESTIONS_VIEW
    assert quality.view == reports.SURPRISE_QUALITY_VIEW
    assert questions.rows[0]["rating_count"] == 2
    assert quality.rows[0]["counts_conserved"] is True
    assert [
        urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["view"][0]
        for request in captured
    ] == [reports.SURPRISE_QUESTIONS_VIEW, reports.SURPRISE_QUALITY_VIEW]


def test_rows_convenience_returns_detached_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = "feedback_report_questions"
    body = json.dumps(
        _envelope(view, [_row(view, question_id="q_1")], limit=10)
    ).encode()
    monkeypatch.setattr(
        reports_client,
        "_open_report_request",
        lambda request, *, timeout: _FakeResponse(body),
    )

    rows = _client().fetch_rows(view, limit=10)
    assert isinstance(rows, list)
    rows[0]["question_id"] = "local-change"
    assert rows[0]["question_id"] == "local-change"


def test_http_and_network_errors_are_diagnostic_without_token_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "server-token-that-must-not-leak"
    client = reports.ReportsClient(
        url=f"https://reports.example/{secret}",
        read_token=secret,
        environ={},
    )

    def fail_http(request: Any, *, timeout: float) -> Any:
        del request, timeout
        raise urllib.error.HTTPError(
            REPORT_URL,
            503,
            "Unavailable",
            Message(),
            io.BytesIO(f'{{"message":"retry with {secret}"}}'.encode()),
        )

    monkeypatch.setattr(reports_client, "_open_report_request", fail_http)
    with pytest.raises(reports.ReportsRequestError) as caught:
        client.fetch_page("feedback_report_summary", limit=1)
    assert caught.value.status_code == 503
    assert "status=503" in str(caught.value)
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)

    def fail_network(request: Any, *, timeout: float) -> Any:
        del request, timeout
        raise urllib.error.URLError(f"connection reset near {secret}")

    monkeypatch.setattr(reports_client, "_open_report_request", fail_network)
    with pytest.raises(reports.ReportsRequestError) as caught:
        client.fetch_page("feedback_report_summary", limit=1)
    assert "connection reset" in str(caught.value)
    assert secret not in str(caught.value)


def test_report_transport_refuses_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpener:
        def open(self, request: Any, *, timeout: float) -> _FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(b"{}")

    def fake_build_opener(*handlers: Any) -> FakeOpener:
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    request = urllib.request.Request(REPORT_URL, method="GET")
    response = reports_client._open_report_request(request, timeout=2.5)

    assert isinstance(response, _FakeResponse)
    assert captured["request"] is request
    assert captured["timeout"] == 2.5
    handler = captured["handlers"][0]
    assert isinstance(handler, urllib.request.HTTPRedirectHandler)
    assert (
        handler.redirect_request(
            request,
            response,
            302,
            "Found",
            Message(),
            "https://other.example/reports",
        )
        is None
    )


def test_client_rejects_invalid_json_and_raw_array_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(
        reports_client,
        "_open_report_request",
        lambda request, *, timeout: _FakeResponse(b"not json"),
    )
    with pytest.raises(reports.ReportsResponseError, match="valid JSON"):
        client.fetch_page("feedback_report_questions")

    monkeypatch.setattr(
        reports_client,
        "_open_report_request",
        lambda request, *, timeout: _FakeResponse(b"[]"),
    )
    with pytest.raises(reports.ReportsResponseError, match="JSON object"):
        client.fetch_page("feedback_report_questions")


def test_csv_serialization_uses_stable_columns_and_blocks_formulas() -> None:
    view = "feedback_report_comments"
    rows = [
        _row(
            view,
            event_id="evt_1",
            question_id="问题一",
            category="suggestion",
            comment_text='=HYPERLINK("https://example.invalid")',
        )
    ]
    serialized = reports.report_rows_to_csv(view, rows)
    parsed = list(csv.DictReader(io.StringIO(serialized)))

    assert tuple(parsed[0]) == reports.copy_view_columns()[view]
    assert parsed[0]["question_id"] == "问题一"
    assert parsed[0]["comment_text"].startswith("'=HYPERLINK")
    assert reports.report_rows_to_csv(view, []).splitlines() == [
        ",".join(reports.copy_view_columns()[view])
    ]


def test_csv_serializes_nested_arrays_as_canonical_json() -> None:
    view = "feedback_report_sessions"
    row = _row(
        view,
        session_id="anon_1",
        release_ids=["release_2", "release_1"],
        families=["多元回归"],
    )
    parsed = next(csv.DictReader(io.StringIO(reports.report_rows_to_csv(view, [row]))))
    assert parsed["release_ids"] == '["release_2","release_1"]'
    assert parsed["families"] == '["多元回归"]'


def test_fetch_csv_refuses_partial_page(monkeypatch: pytest.MonkeyPatch) -> None:
    view = "feedback_report_questions"
    page = reports.ReportPage(
        view=view,
        rows=(_row(view, question_id="q_1"),),
        total=2,
        limit=1,
        offset=0,
    )
    monkeypatch.setattr(
        reports.ReportsClient,
        "fetch_page",
        lambda self, *args, **kwargs: page,
    )

    with pytest.raises(reports.ReportsResponseError, match="partial"):
        _client().fetch_csv(view, limit=1)
