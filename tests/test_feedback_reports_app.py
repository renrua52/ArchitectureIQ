"""Streamlit smoke tests for the internal feedback Reports app."""

from __future__ import annotations

import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from tools.feedback_reports import (
    REPORTS_READ_TOKEN_ENV,
    REPORTS_TIMEOUT_ENV,
    REPORTS_URL_ENV,
    copy_view_columns,
)


REPO = Path(__file__).resolve().parents[1]
APP = REPO / "tools" / "feedback_reports" / "app.py"
READ_TOKEN = "internal-report-read-token"
DETAIL_RELEASE_ID = f"release_{'a' * 64}"
DETAIL_QUESTION_VERSION = f"qv1_{'b' * 64}"
BUSINESS_SNAPSHOT_VIEW = "feedback_report_business_snapshot"
SURPRISE_QUESTIONS_VIEW = "feedback_report_surprise_questions"
SURPRISE_QUALITY_VIEW = "feedback_report_surprise_quality"
BUSINESS_SNAPSHOT_AT = "2026-07-12T00:10:02Z"
BUSINESS_VIEWS = (
    "feedback_report_summary",
    "feedback_report_sessions",
    "feedback_report_questions",
    "feedback_report_answers",
    "feedback_report_proposals",
    "feedback_report_comments",
)


def _complete_row(view: str, **overrides: Any) -> dict[str, Any]:
    arrays = {"release_ids", "families", "question_types"}
    values: dict[str, Any] = {}
    for column in copy_view_columns()[view]:
        if column in arrays:
            values[column] = []
        elif column.endswith("_count"):
            values[column] = 0
        elif column.endswith("_rate") or column == "accuracy":
            values[column] = None
        elif column.endswith("_available"):
            values[column] = False
        else:
            values[column] = None
    values.update(overrides)
    return values


REPORT_ROWS = {
    "feedback_report_summary": _complete_row(
        "feedback_report_summary",
        event_count=8,
        first_event_at="2026-07-12T00:00:00Z",
        last_event_at="2026-07-12T00:10:00Z",
        session_count=1,
        attempt_count=1,
        solve_attempt_count=1,
        answered_attempt_count=1,
        question_count=1,
        answer_count=2,
        known_answer_count=2,
        correct_answer_count=1,
        incorrect_answer_count=1,
        accuracy=0.5,
        proposal_count=1,
        completed_run_count=1,
        comment_count=1,
        attempts_with_proposal=1,
        proposal_usage_rate=1.0,
    ),
    "feedback_report_ingestion_summary": _complete_row(
        "feedback_report_ingestion_summary",
        recorded_request_count=4,
        first_started_at="2026-07-12T00:00:00Z",
        last_finished_at="2026-07-12T00:10:00Z",
        success_request_count=2,
        client_rejection_count=1,
        service_failure_count=1,
        event_id_conflict_request_count=1,
        accepted_event_count=3,
        duplicate_event_count=2,
        idempotent_duplicate_event_count=1,
        unclassified_duplicate_event_count=1,
        conflicting_event_count=1,
        conflict_audit_event_count=1,
        event_id_reuse_count=3,
        classified_event_count=7,
        known_event_result_count=5,
        request_failure_rate=0.5,
        duplicate_event_rate=0.4,
        event_id_reuse_rate=0.4286,
        classified_conflicting_event_rate=0.5,
        recorded_rate_available=True,
        end_to_end_coverage_available=False,
    ),
    "feedback_report_registry_quality": _complete_row(
        "feedback_report_registry_quality",
        registered_release_count=1,
        registered_question_count=60,
        registered_choice_count=180,
        registry_available=True,
        raw_event_count=8,
        authoritative_event_count=8,
        excluded_event_count=0,
        raw_answer_count=2,
        authoritative_answer_count=2,
        unresolved_answer_count=0,
        registry_match_rate=1.0,
        answer_resolution_rate=1.0,
    ),
    SURPRISE_QUESTIONS_VIEW: _complete_row(
        SURPRISE_QUESTIONS_VIEW,
        question_id="q_test",
        question_version=DETAIL_QUESTION_VERSION,
        release_id=DETAIL_RELEASE_ID,
        family="univariate_regression",
        dataset_id="sym_test",
        question_type="mixed",
        answered_attempt_count=1,
        rating_count=1,
        surprised_count=1,
        not_surprised_count=0,
        rating_coverage_rate=1.0,
        observed_surprise_rate=1.0,
        posterior_mean=0.6667,
        first_rating_at="2026-07-12T00:08:00Z",
        last_rating_at="2026-07-12T00:08:00Z",
    ),
    SURPRISE_QUALITY_VIEW: _complete_row(
        SURPRISE_QUALITY_VIEW,
        raw_reaction_count=3,
        valid_reaction_count=1,
        orphan_reaction_count=1,
        duplicate_reaction_count=1,
        registry_unmatched_reaction_count=1,
        invalid_payload_reaction_count=0,
        missing_prior_answer_reaction_count=0,
        unknown_release_reaction_count=1,
        counts_conserved=True,
        orphan_breakdown_conserved=True,
    ),
    "feedback_report_sessions": _complete_row(
        "feedback_report_sessions",
        session_id="anon_test",
        attempt_id="attempt_test",
        started_at="2026-07-12T00:00:00Z",
        last_event_at="2026-07-12T00:10:00Z",
        first_received_at="2026-07-12T00:00:01Z",
        last_received_at="2026-07-12T00:10:01Z",
        release_ids=[DETAIL_RELEASE_ID],
        families=["univariate_regression"],
        question_types=["mixed"],
        event_count=8,
        question_count=1,
        answer_count=2,
        known_answer_count=2,
        correct_answer_count=1,
        incorrect_answer_count=1,
        accuracy=0.5,
        proposal_count=1,
        completed_run_count=1,
        comment_count=1,
    ),
    "feedback_report_questions": _complete_row(
        "feedback_report_questions",
        question_id="q_test",
        question_version=DETAIL_QUESTION_VERSION,
        release_id=DETAIL_RELEASE_ID,
        family="univariate_regression",
        dataset_id="sym_test",
        question_type="mixed",
        first_event_at="2026-07-12T00:00:00Z",
        last_event_at="2026-07-12T00:10:00Z",
        event_count=8,
        session_count=1,
        attempt_count=1,
        solve_attempt_count=1,
        answered_attempt_count=1,
        answer_count=2,
        known_answer_count=2,
        correct_answer_count=1,
        incorrect_answer_count=1,
        accuracy=0.5,
        proposal_count=1,
        completed_run_count=1,
        comment_count=1,
        attempts_with_proposal=1,
        proposal_usage_rate=1.0,
    ),
    "feedback_report_answers": _complete_row(
        "feedback_report_answers",
        event_id="evt_answer",
        occurred_at="2026-07-12T00:05:00Z",
        received_at="2026-07-12T00:05:01Z",
        session_id="anon_test",
        attempt_id="attempt_test",
        question_id="q_test",
        question_version=DETAIL_QUESTION_VERSION,
        release_id=DETAIL_RELEASE_ID,
        family="univariate_regression",
        dataset_id="sym_test",
        question_type="mixed",
        selected_letter="A",
        client_selected_candidate_id="candidate_a",
        selected_candidate_id="candidate_a",
        answer_status="resolved",
        is_correct=True,
        client_is_correct=False,
        client_context_mismatch=True,
        client_correctness_mismatch=True,
    ),
    "feedback_report_proposals": _complete_row(
        "feedback_report_proposals",
        event_id="evt_proposal",
        occurred_at="2026-07-12T00:06:00Z",
        received_at="2026-07-12T00:06:01Z",
        session_id="anon_test",
        attempt_id="attempt_test",
        question_id="q_test",
        question_version=DETAIL_QUESTION_VERSION,
        release_id=DETAIL_RELEASE_ID,
        family="univariate_regression",
        dataset_id="sym_test",
        question_type="mixed",
        setting_status="proposed",
        label="Try a wider model",
        setting_json='{"model":{"hidden_dims":[64,64]}}',
        inherited_from_json='{"choice":"A"}',
        n_seeds=3,
        base_seed=42,
    ),
    "feedback_report_comments": _complete_row(
        "feedback_report_comments",
        event_id="evt_comment",
        occurred_at="2026-07-12T00:09:00Z",
        received_at="2026-07-12T00:09:01Z",
        session_id="anon_test",
        attempt_id="attempt_test",
        question_id="q_test",
        question_version=DETAIL_QUESTION_VERSION,
        release_id=DETAIL_RELEASE_ID,
        family="univariate_regression",
        question_type="mixed",
        category="suggestion",
        comment_text="Please clarify this question.",
    ),
}


def _business_snapshot_row(*, limit: int) -> dict[str, Any]:
    summary = REPORT_ROWS["feedback_report_summary"]
    totals = {
        "feedback_report_summary": 1,
        "feedback_report_sessions": summary["attempt_count"],
        "feedback_report_questions": summary["question_count"],
        "feedback_report_answers": summary["answer_count"],
        "feedback_report_proposals": (
            summary["proposal_count"] + summary["rejected_setting_count"]
        ),
        "feedback_report_comments": summary["comment_count"],
    }
    pages = {
        view: {
            "view": view,
            "rows": [REPORT_ROWS[view]],
            "total": totals[view],
            "limit": limit,
            "offset": 0,
        }
        for view in BUSINESS_VIEWS
    }
    return {
        "snapshot_revision": "business_snapshot_v1",
        "snapshot_at": BUSINESS_SNAPSHOT_AT,
        "authority_revision": "registry_v1",
        "business_reports_authoritative": True,
        "detail_revision": "detail_v1",
        "detail_reports_authoritative": True,
        "registered_release_count": 1,
        "registered_question_count": 60,
        "registered_choice_count": 180,
        "pages_json": json.dumps(pages, separators=(",", ":"), sort_keys=True),
    }


class _ReportHandler(BaseHTTPRequestHandler):
    records: list[dict[str, Any]] = []
    fail_ingestion = False
    fail_business_snapshot = False
    fail_surprise_questions = False
    surprise_question_total = 1

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        view = query.get("view", [""])[0]
        self.records.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "view": view,
            }
        )
        if self.headers.get("Authorization") != f"Bearer {READ_TOKEN}":
            self.send_response(401)
            self.end_headers()
            return
        limit = int(query["limit"][0])
        row = (
            _business_snapshot_row(limit=limit)
            if view == BUSINESS_SNAPSHOT_VIEW
            else REPORT_ROWS.get(view)
        )
        if row is None:
            self.send_response(400)
            self.end_headers()
            return
        if view == "feedback_report_ingestion_summary" and self.fail_ingestion:
            self.send_response(503)
            self.end_headers()
            return
        if view == BUSINESS_SNAPSHOT_VIEW and self.fail_business_snapshot:
            self.send_response(503)
            self.end_headers()
            return
        if view == SURPRISE_QUESTIONS_VIEW and self.fail_surprise_questions:
            self.send_response(503)
            self.end_headers()
            return
        total = self.surprise_question_total if view == SURPRISE_QUESTIONS_VIEW else 1
        body = json.dumps(
            {
                "view": view,
                "rows": [row],
                "total": total,
                "limit": limit,
                "offset": int(query["offset"][0]),
                "request_id": f"request-{view}",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def report_endpoint() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    _ReportHandler.records = []
    _ReportHandler.fail_ingestion = False
    _ReportHandler.fail_business_snapshot = False
    _ReportHandler.fail_surprise_questions = False
    _ReportHandler.surprise_question_total = 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReportHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/feedback-report", _ReportHandler.records
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_reports_app_explains_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REPORTS_URL_ENV, raising=False)
    monkeypatch.delenv(REPORTS_READ_TOKEN_ENV, raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert [title.value for title in app.title] == ["ArchitectureIQ Reports"]
    assert any("not configured" in warning.value for warning in app.warning)
    assert any("service-role key" in info.value for info in app.info)


def test_reports_app_loads_business_and_ingestion_views_and_renders_kpis(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)
    monkeypatch.setenv(REPORTS_TIMEOUT_ENV, "5")

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert {record["view"] for record in records} == {
        BUSINESS_SNAPSHOT_VIEW,
        "feedback_report_ingestion_summary",
        "feedback_report_registry_quality",
        SURPRISE_QUESTIONS_VIEW,
        SURPRISE_QUALITY_VIEW,
    }
    assert len(records) == 5
    assert [record["view"] for record in records].count(BUSINESS_SNAPSHOT_VIEW) == 1
    assert not (
        {"feedback_report_authority_status", *BUSINESS_VIEWS}
        & {record["view"] for record in records}
    )
    assert all(record["authorization"] == f"Bearer {READ_TOKEN}" for record in records)
    assert all(READ_TOKEN not in record["path"] for record in records)
    assert [tab.label for tab in app.tabs] == [
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
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics == {
        "Sessions": "1",
        "Attempts": "1",
        "Answers": "2",
        "Authoritative accuracy": "50.0%",
        "Propose usage": "100.0%",
        "Comments": "1",
        "Recorded requests": "4",
        "Success": "2",
        "Client rejected (total)": "1",
        "Event-ID conflict requests": "1",
        "Service failures": "1",
        "Recorded failure rate": "50.0%",
        "Duplicate rate (known results)": "40.0%",
        "Verified idempotent duplicates": "1",
        "Legacy unclassified duplicates": "1",
        "Conflicting events": "1",
        "Conflict audit rows": "1",
        "Event-ID reuse rate": "42.9%",
        "Classified conflict rate": "50.0%",
        "Registered releases": "1",
        "Registered questions": "60",
        "Raw events": "8",
        "Registry match rate": "100.0%",
        "Resolved answers": "2",
        "Answer resolution rate": "100.0%",
        "Ratings": "1",
        "Surprised (yes)": "1",
        "Not surprised (no)": "0",
        "Rating coverage": "100.0%",
        "Observed surprise": "100.0%",
        "Beta(1,1) posterior": "66.7%",
        "Raw reactions": "3",
        "Valid first ratings": "1",
        "Orphan reactions": "1",
        "Duplicate reactions": "1",
        "Registry unmatched": "1",
        "Invalid reaction payload": "0",
        "Missing prior answer": "0",
        "Unknown release": "1",
    }
    assert len(app.dataframe) == 6
    assert len(app.download_button) == 10
    proposal_frame = app.dataframe[3].value
    assert proposal_frame.loc[0, "setting_json"] == (
        '{"model":{"hidden_dims":[64,64]}}'
    )
    assert proposal_frame.loc[0, "inherited_from_json"] == '{"choice":"A"}'
    assert any(
        "3 total event-ID reuse(s) across 7 classified event(s)" in caption.value
        for caption in app.caption
    )
    assert any("not available" in info.value for info in app.info)
    assert any("Verified idempotent retries" in info.value for info in app.info)
    assert any("Coverage is incomplete" in warning.value for warning in app.warning)
    assert not any(
        "Treat them as provisional" in warning.value for warning in app.warning
    )
    assert any(
        "derived from the immutable, attested server registry" in caption.value
        for caption in app.caption
    )
    assert any(
        "All six business tabs share one PostgreSQL MVCC snapshot" in caption.value
        for caption in app.caption
    )
    assert any(
        f"Server snapshot_at={BUSINESS_SNAPSHOT_AT}" in caption.value
        for caption in app.caption
    )
    assert any(
        "business_snapshot_v1/registry_v1/detail_v1" in caption.value
        for caption in app.caption
    )
    assert any("missing, not a ‘no’" in caption.value for caption in app.caption)
    assert any("Beta(1,1)" in caption.value for caption in app.caption)
    assert any(
        "not part of the six-page business MVCC" in caption.value
        for caption in app.caption
    )
    assert any("Validated conservation" in info.value for info in app.info)
    assert any(
        "Legacy duplicate IDs are unclassified" in warning.value
        for warning in app.warning
    )
    assert any("Recorded service failures" in error.value for error in app.error)
    assert any("Conflicting event IDs detected" in error.value for error in app.error)
    assert not any(
        "Recorded client rejections" in warning.value for warning in app.warning
    )


def test_reports_app_skips_ingestion_query_when_content_filters_are_active(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)
    release_input = next(
        field for field in app.text_input if field.label == "Release ID"
    )
    session_input = next(
        field for field in app.text_input if field.label == "Session ID"
    )
    attempt_input = next(
        field for field in app.text_input if field.label == "Attempt ID"
    )
    apply_button = next(
        button for button in app.button if button.label == "Apply filters"
    )
    release_input.set_value("release_test")
    session_input.set_value("anon_test")
    attempt_input.set_value("attempt_test")
    app = apply_button.click().run(timeout=30)

    assert not app.exception
    views = [record["view"] for record in records]
    assert views.count(BUSINESS_SNAPSHOT_VIEW) == 2
    assert views.count("feedback_report_ingestion_summary") == 1
    assert views.count("feedback_report_registry_quality") == 1
    assert views.count(SURPRISE_QUESTIONS_VIEW) == 2
    assert views.count(SURPRISE_QUALITY_VIEW) == 2
    assert not ({"feedback_report_authority_status", *BUSINESS_VIEWS} & set(views))
    snapshot_requests = [
        record for record in records if record["view"] == BUSINESS_SNAPSHOT_VIEW
    ]
    applied_query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(snapshot_requests[-1]["path"]).query
    )
    assert applied_query["release_id"] == ["release_test"]
    assert applied_query["session_id"] == ["anon_test"]
    assert applied_query["attempt_id"] == ["attempt_test"]
    for view in (SURPRISE_QUESTIONS_VIEW, SURPRISE_QUALITY_VIEW):
        surprise_request = [record for record in records if record["view"] == view][-1]
        surprise_query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(surprise_request["path"]).query
        )
        assert surprise_query["release_id"] == ["release_test"]
        assert surprise_query["session_id"] == ["anon_test"]
        assert surprise_query["attempt_id"] == ["attempt_test"]
    assert any(
        "question/session/attempt filters are active" in info.value for info in app.info
    )
    assert any(
        "Ingestion quality unavailable" in warning.value for warning in app.warning
    )


def test_surprise_partial_page_disables_only_its_csv(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, _ = report_endpoint
    _ReportHandler.surprise_question_total = 2
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    downloads = {button.key: button for button in app.download_button}
    surprise_questions = downloads["download_feedback_report_surprise_questions"]
    surprise_quality = downloads["download_feedback_report_surprise_quality"]
    assert surprise_questions.disabled
    assert not surprise_quality.disabled
    assert any(
        "Showing 1 of 2 matching rows" in warning.value for warning in app.warning
    )


def test_surprise_question_failure_keeps_independent_quality_row(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    _ReportHandler.fail_surprise_questions = True
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert {record["view"] for record in records} == {
        BUSINESS_SNAPSHOT_VIEW,
        "feedback_report_ingestion_summary",
        "feedback_report_registry_quality",
        SURPRISE_QUESTIONS_VIEW,
        SURPRISE_QUALITY_VIEW,
    }
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Ratings" not in metrics
    assert metrics["Raw reactions"] == "3"
    assert metrics["Orphan reactions"] == "1"
    assert any(
        "independent surprise question page could not be loaded" in error.value
        for error in app.error
    )
    assert len(app.download_button) == 9


@pytest.mark.parametrize(
    ("filter_values", "expected_query"),
    [
        (
            {"Session ID": "anon_session_only"},
            {"session_id": ["anon_session_only"]},
        ),
        (
            {"Attempt ID": "attempt_only"},
            {"attempt_id": ["attempt_only"]},
        ),
        (
            {
                "Session ID": "anon_identity_pair",
                "Attempt ID": "attempt_identity_pair",
            },
            {
                "session_id": ["anon_identity_pair"],
                "attempt_id": ["attempt_identity_pair"],
            },
        ),
    ],
    ids=("session-only", "attempt-only", "session-and-attempt"),
)
def test_identity_filters_explain_why_auxiliary_reports_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
    filter_values: dict[str, str],
    expected_query: dict[str, list[str]],
) -> None:
    endpoint, records = report_endpoint
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)
    text_inputs = {field.label: field for field in app.text_input}
    for label, value in filter_values.items():
        text_inputs[label].set_value(value)
    apply_button = next(
        button for button in app.button if button.label == "Apply filters"
    )
    app = apply_button.click().run(timeout=30)

    assert not app.exception
    views = [record["view"] for record in records]
    assert views.count(BUSINESS_SNAPSHOT_VIEW) == 2
    assert views.count("feedback_report_ingestion_summary") == 1
    assert views.count("feedback_report_registry_quality") == 1
    assert views.count(SURPRISE_QUESTIONS_VIEW) == 2
    assert views.count(SURPRISE_QUALITY_VIEW) == 2
    snapshot_requests = [
        record for record in records if record["view"] == BUSINESS_SNAPSHOT_VIEW
    ]
    applied_query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(snapshot_requests[-1]["path"]).query
    )
    for name, values in expected_query.items():
        assert applied_query[name] == values
        for view in (SURPRISE_QUESTIONS_VIEW, SURPRISE_QUALITY_VIEW):
            surprise_request = [record for record in records if record["view"] == view][
                -1
            ]
            surprise_query = urllib.parse.parse_qs(
                urllib.parse.urlsplit(surprise_request["path"]).query
            )
            assert surprise_query[name] == values

    infos = [info.value for info in app.info]
    warnings = [warning.value for warning in app.warning]
    assert any(
        "question/session/attempt filters are active" in value for value in infos
    )
    assert any(
        "Content filters intentionally exclude ingestion observability." in value
        for value in warnings
    )
    assert any(
        "Content filters intentionally exclude the all-event registry quality "
        "snapshot." in value
        for value in warnings
    )
    assert not any(
        "No valid ingestion observability snapshot is available." in value
        for value in warnings
    )
    assert not any(
        "No valid registry quality snapshot is available." in value
        for value in warnings
    )


def test_ingestion_failure_does_not_discard_business_report_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    _ReportHandler.fail_ingestion = True
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert {record["view"] for record in records} == {
        BUSINESS_SNAPSHOT_VIEW,
        "feedback_report_ingestion_summary",
        "feedback_report_registry_quality",
        SURPRISE_QUESTIONS_VIEW,
        SURPRISE_QUALITY_VIEW,
    }
    assert [tab.label for tab in app.tabs][-4:] == [
        "Ingestion observability",
        "Registry quality",
        "Surprise",
        "Data quality",
    ]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Sessions"] == "1"
    assert "Recorded requests" not in metrics
    assert len(app.dataframe) == 6
    assert len(app.download_button) == 9
    assert any(
        "Ingestion observability could not be loaded independently" in error.value
        for error in app.error
    )
    assert any(
        "Ingestion quality unavailable" in warning.value for warning in app.warning
    )


def test_initial_business_snapshot_failure_hides_all_business_kpis(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    _ReportHandler.fail_business_snapshot = True
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert [record["view"] for record in records] == [BUSINESS_SNAPSHOT_VIEW]
    assert not app.metric
    assert not app.dataframe
    assert not app.tabs
    assert any(
        "No validated atomic business snapshot is available" in error.value
        for error in app.error
    )


def test_failed_refresh_keeps_previous_atomic_business_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Sessions"] == "1"

    _ReportHandler.fail_business_snapshot = True
    refresh_button = next(button for button in app.button if button.label == "Refresh")
    app = refresh_button.click().run(timeout=30)

    assert not app.exception
    assert [record["view"] for record in records] == [
        BUSINESS_SNAPSHOT_VIEW,
        "feedback_report_ingestion_summary",
        "feedback_report_registry_quality",
        SURPRISE_QUESTIONS_VIEW,
        SURPRISE_QUALITY_VIEW,
        BUSINESS_SNAPSHOT_VIEW,
    ]
    assert {metric.label: metric.value for metric in app.metric}["Sessions"] == "1"
    assert any(
        f"Server snapshot_at={BUSINESS_SNAPSHOT_AT}" in caption.value
        for caption in app.caption
    )
    assert any("previous successful snapshot" in error.value for error in app.error)


def test_data_quality_keeps_business_signal_when_ingestion_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    report_endpoint: tuple[str, list[dict[str, Any]]],
) -> None:
    endpoint, records = report_endpoint
    _ReportHandler.fail_ingestion = True
    summary = REPORT_ROWS["feedback_report_summary"]
    monkeypatch.setitem(summary, "answer_count", 3)
    monkeypatch.setitem(summary, "unknown_answer_count", 1)
    monkeypatch.setenv(REPORTS_URL_ENV, endpoint)
    monkeypatch.setenv(REPORTS_READ_TOKEN_ENV, READ_TOKEN)

    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert len(records) == 5
    warnings = [warning.value for warning in app.warning]
    assert any("Unknown answer correctness" in value for value in warnings)
    assert any("Ingestion quality unavailable" in value for value in warnings)
    assert not any("No observed data-quality issues" in value for value in warnings)
