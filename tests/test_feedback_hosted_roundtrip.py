"""Offline orchestration tests for the opt-in hosted feedback verifier."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from typing import Any

import pytest

from tools.feedback_reports import verify_hosted_roundtrip as verifier


INGEST_REQUEST_ID = "72aee12d-7742-44ea-b3d9-f056ae5c8ac2"
SUCCESSFUL_BATCH_FIRST_REQUEST_ID = "6d77b67b-06ff-44a5-b23d-a342719d71f7"
SUCCESSFUL_BATCH_REPLAY_REQUEST_ID = "80174a98-dd3d-4c52-a852-bcb39b744d87"
CONFLICT_REQUEST_ID = "bdbb2f4b-87a0-44c9-83f1-fdc5c596c36d"
MIXED_BATCH_REQUEST_ID = "f1e8ef10-f49d-4c6b-bb30-a414367c38ad"
MANIFEST_SHA256 = "c" * 64
VERIFIED_AT = "2026-07-12T12:34:56.789Z"


def _page(
    view: str,
    rows: list[dict[str, Any]],
    *,
    total: int | None = None,
) -> verifier.ReportPage:
    return verifier.ReportPage(
        view=view,
        rows=tuple(rows),
        total=len(rows) if total is None else total,
        limit=2,
        offset=0,
        request_id="report-request",
    )


def _empty_pages() -> dict[str, verifier.ReportPage]:
    return {
        verifier.SUMMARY_VIEW: _page(
            verifier.SUMMARY_VIEW,
            [{"event_count": 0}],
            total=1,
        ),
        verifier.SESSIONS_VIEW: _page(verifier.SESSIONS_VIEW, []),
        verifier.QUESTIONS_VIEW: _page(verifier.QUESTIONS_VIEW, []),
        verifier.COMMENTS_VIEW: _page(verifier.COMMENTS_VIEW, []),
    }


def _snapshot_page(
    view: str,
    rows: list[dict[str, Any]],
    *,
    total: int | None = None,
    limit: int = 1,
) -> verifier.ReportPage:
    return verifier.ReportPage(
        view=view,
        rows=tuple(deepcopy(rows)),
        total=len(rows) if total is None else total,
        limit=limit,
        offset=0,
        request_id=None,
    )


def _empty_business_snapshot(
    probe: verifier.AuthoritativeProbeQuestion,
) -> verifier.BusinessSnapshot:
    summary = {
        "event_count": 0,
        "first_event_at": None,
        "last_event_at": None,
        "session_count": 0,
        "attempt_count": 0,
        "solve_attempt_count": 0,
        "answered_attempt_count": 0,
        "question_count": 0,
        "answer_count": 0,
        "known_answer_count": 0,
        "correct_answer_count": 0,
        "incorrect_answer_count": 0,
        "unknown_answer_count": 0,
        "accuracy": None,
        "proposal_count": 0,
        "rejected_setting_count": 0,
        "completed_run_count": 0,
        "failed_run_count": 0,
        "comment_count": 0,
        "attempts_with_proposal": 0,
        "proposal_usage_rate": None,
        "ingestion_failure_rate": None,
        "ingestion_failure_rate_available": False,
    }
    pages = {
        view: (
            _snapshot_page(view, [summary], total=1)
            if view == verifier.SUMMARY_VIEW
            else _snapshot_page(view, [])
        )
        for view in verifier.BUSINESS_REPORT_VIEWS
    }
    return verifier.BusinessSnapshot(
        snapshot_revision="business_snapshot_v1",
        snapshot_at="2026-07-12T00:00:00.123456Z",
        authority_revision="registry_v1",
        business_reports_authoritative=True,
        detail_revision="detail_v1",
        detail_reports_authoritative=True,
        registered_release_count=1,
        registered_question_count=probe.registry_question_count,
        registered_choice_count=probe.registry_choice_count,
        pages=pages,
        request_id="business-snapshot-request",
    )


def _identity_business_snapshot(
    probe: verifier.AuthoritativeProbeQuestion,
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> verifier.BusinessSnapshot:
    answer = next(
        event for event in events if event["event_type"] == "answer_submitted"
    )
    proposal = next(
        event for event in events if event["event_type"] == "custom_setting_proposed"
    )
    comments = [event for event in events if event["event_type"] == "comment_submitted"]
    session_id = events[0]["session_id"]
    attempt_id = events[0]["payload"]["attempt_id"]
    summary = {
        "event_count": 4,
        "session_count": 1,
        "attempt_count": 1,
        "solve_attempt_count": 1,
        "answered_attempt_count": 1,
        "question_count": 1,
        "answer_count": 1,
        "known_answer_count": 1,
        "correct_answer_count": int(probe.authoritative_is_correct),
        "incorrect_answer_count": int(not probe.authoritative_is_correct),
        "unknown_answer_count": 0,
        "accuracy": 1.0 if probe.authoritative_is_correct else 0.0,
        "proposal_count": 1,
        "rejected_setting_count": 0,
        "completed_run_count": 0,
        "failed_run_count": 0,
        "comment_count": 2,
        "attempts_with_proposal": 1,
        "proposal_usage_rate": 1.0,
    }
    pages = {
        verifier.SUMMARY_VIEW: _snapshot_page(
            verifier.SUMMARY_VIEW, [summary], total=1, limit=limit
        ),
        verifier.SESSIONS_VIEW: _snapshot_page(
            verifier.SESSIONS_VIEW,
            [
                {
                    "session_id": session_id,
                    "attempt_id": attempt_id,
                    "event_count": 4,
                }
            ],
            total=1,
            limit=limit,
        ),
        verifier.QUESTIONS_VIEW: _snapshot_page(
            verifier.QUESTIONS_VIEW,
            [
                {
                    "question_id": probe.question_id,
                    "question_version": probe.question_version,
                    "event_count": 4,
                }
            ],
            total=1,
            limit=limit,
        ),
        verifier.ANSWER_DETAILS_VIEW: _snapshot_page(
            verifier.ANSWER_DETAILS_VIEW,
            [
                {
                    "event_id": answer["event_id"],
                    "session_id": session_id,
                    "attempt_id": attempt_id,
                }
            ],
            total=1,
            limit=limit,
        ),
        verifier.PROPOSAL_DETAILS_VIEW: _snapshot_page(
            verifier.PROPOSAL_DETAILS_VIEW,
            [
                {
                    "event_id": proposal["event_id"],
                    "session_id": session_id,
                    "attempt_id": attempt_id,
                }
            ],
            total=1,
            limit=limit,
        ),
        verifier.COMMENTS_VIEW: _snapshot_page(
            verifier.COMMENTS_VIEW,
            [
                {
                    "event_id": event["event_id"],
                    "session_id": session_id,
                    "attempt_id": attempt_id,
                }
                for event in comments
            ],
            total=2,
            limit=limit,
        ),
    }
    return verifier.BusinessSnapshot(
        snapshot_revision="business_snapshot_v1",
        snapshot_at="2026-07-12T00:00:02.123456Z",
        authority_revision="registry_v1",
        business_reports_authoritative=True,
        detail_revision="detail_v1",
        detail_reports_authoritative=True,
        registered_release_count=1,
        registered_question_count=probe.registry_question_count,
        registered_choice_count=probe.registry_choice_count,
        pages=pages,
        request_id="identity-business-snapshot-request",
    )


def _visible_pages(
    identity: verifier.SmokeIdentity,
) -> dict[str, verifier.ReportPage]:
    return {
        verifier.SUMMARY_VIEW: _page(
            verifier.SUMMARY_VIEW,
            [
                {
                    "event_count": 1,
                    "session_count": 1,
                    "attempt_count": 1,
                    "question_count": 1,
                    "answer_count": 0,
                    "proposal_count": 0,
                    "comment_count": 1,
                    "ingestion_failure_rate": None,
                    "ingestion_failure_rate_available": False,
                }
            ],
            total=1,
        ),
        verifier.SESSIONS_VIEW: _page(
            verifier.SESSIONS_VIEW,
            [
                {
                    "session_id": identity.session_id,
                    "attempt_id": identity.attempt_id,
                    "release_ids": [identity.release_id],
                    "families": [verifier.FAMILY],
                    "question_types": [verifier.QUESTION_TYPE],
                    "event_count": 1,
                    "question_count": 1,
                    "answer_count": 0,
                    "comment_count": 1,
                }
            ],
        ),
        verifier.QUESTIONS_VIEW: _page(
            verifier.QUESTIONS_VIEW,
            [
                {
                    "question_id": identity.question_id,
                    "question_version": identity.question_version,
                    "release_id": identity.release_id,
                    "family": verifier.FAMILY,
                    "dataset_id": identity.dataset_id,
                    "question_type": verifier.QUESTION_TYPE,
                    "event_count": 1,
                    "session_count": 1,
                    "answer_count": 0,
                    "comment_count": 1,
                }
            ],
        ),
        verifier.COMMENTS_VIEW: _page(
            verifier.COMMENTS_VIEW,
            [
                {
                    "event_id": identity.event_id,
                    "session_id": identity.session_id,
                    "attempt_id": identity.attempt_id,
                    "question_id": identity.question_id,
                    "question_version": identity.question_version,
                    "release_id": identity.release_id,
                    "family": verifier.FAMILY,
                    "question_type": verifier.QUESTION_TYPE,
                    "category": verifier.COMMENT_CATEGORY,
                    "comment_text": identity.comment_text,
                }
            ],
        ),
    }


def _successful_batch_pages(
    identity: verifier.SuccessfulBatchIdentity,
) -> dict[str, verifier.ReportPage]:
    return {
        verifier.SUMMARY_VIEW: _page(
            verifier.SUMMARY_VIEW,
            [
                {
                    "event_count": 2,
                    "session_count": 1,
                    "attempt_count": 1,
                    "question_count": 1,
                    "answer_count": 1,
                    "proposal_count": 0,
                    "comment_count": 1,
                    "ingestion_failure_rate": None,
                    "ingestion_failure_rate_available": False,
                }
            ],
            total=1,
        ),
        verifier.SESSIONS_VIEW: _page(
            verifier.SESSIONS_VIEW,
            [
                {
                    "session_id": identity.session_id,
                    "attempt_id": identity.attempt_id,
                    "release_ids": [identity.release_id],
                    "families": [verifier.FAMILY],
                    "question_types": [verifier.QUESTION_TYPE],
                    "event_count": 2,
                    "question_count": 1,
                    "answer_count": 1,
                    "comment_count": 1,
                }
            ],
        ),
        verifier.QUESTIONS_VIEW: _page(
            verifier.QUESTIONS_VIEW,
            [
                {
                    "question_id": identity.question_id,
                    "question_version": identity.question_version,
                    "release_id": identity.release_id,
                    "family": verifier.FAMILY,
                    "dataset_id": identity.dataset_id,
                    "question_type": verifier.QUESTION_TYPE,
                    "event_count": 2,
                    "session_count": 1,
                    "answer_count": 1,
                    "comment_count": 1,
                }
            ],
        ),
        verifier.COMMENTS_VIEW: _page(
            verifier.COMMENTS_VIEW,
            [
                {
                    "event_id": identity.comment_event_id,
                    "session_id": identity.session_id,
                    "attempt_id": identity.attempt_id,
                    "question_id": identity.question_id,
                    "question_version": identity.question_version,
                    "release_id": identity.release_id,
                    "family": verifier.FAMILY,
                    "question_type": verifier.QUESTION_TYPE,
                    "category": verifier.COMMENT_CATEGORY,
                    "comment_text": identity.comment_text,
                }
            ],
        ),
    }


def _ingestion_page(
    *,
    accepted: int,
    duplicate: int,
    event_count: int = 1,
) -> verifier.ReportPage:
    duplicate_rate = duplicate / event_count
    return _page(
        verifier.INGESTION_VIEW,
        [
            {
                "recorded_request_count": 1,
                "first_started_at": "2026-07-12T00:00:00Z",
                "last_finished_at": "2026-07-12T00:00:01Z",
                "success_request_count": 1,
                "client_rejection_count": 0,
                "service_failure_count": 0,
                "event_id_conflict_request_count": 0,
                "accepted_event_count": accepted,
                "duplicate_event_count": duplicate,
                "idempotent_duplicate_event_count": duplicate,
                "unclassified_duplicate_event_count": 0,
                "conflicting_event_count": 0,
                "conflict_audit_event_count": 0,
                "event_id_reuse_count": duplicate,
                "classified_event_count": event_count,
                "known_event_result_count": event_count,
                "request_failure_rate": 0,
                "duplicate_event_rate": duplicate_rate,
                "event_id_reuse_rate": duplicate_rate,
                "classified_conflicting_event_rate": 0 if duplicate else None,
                "recorded_rate_available": True,
                "end_to_end_coverage_available": False,
            }
        ],
        total=1,
    )


def _empty_ingestion_page() -> verifier.ReportPage:
    return _page(
        verifier.INGESTION_VIEW,
        [
            {
                "recorded_request_count": 0,
                "first_started_at": None,
                "last_finished_at": None,
                "success_request_count": 0,
                "client_rejection_count": 0,
                "service_failure_count": 0,
                "event_id_conflict_request_count": 0,
                "accepted_event_count": 0,
                "duplicate_event_count": 0,
                "idempotent_duplicate_event_count": 0,
                "unclassified_duplicate_event_count": 0,
                "conflicting_event_count": 0,
                "conflict_audit_event_count": 0,
                "event_id_reuse_count": 0,
                "classified_event_count": 0,
                "known_event_result_count": 0,
                "request_failure_rate": None,
                "duplicate_event_rate": None,
                "event_id_reuse_rate": None,
                "classified_conflicting_event_rate": None,
                "recorded_rate_available": False,
                "end_to_end_coverage_available": False,
            }
        ],
        total=1,
    )


def _conflict_ingestion_page() -> verifier.ReportPage:
    return _page(
        verifier.INGESTION_VIEW,
        [
            {
                "recorded_request_count": 1,
                "first_started_at": "2026-07-12T00:00:00Z",
                "last_finished_at": "2026-07-12T00:00:01Z",
                "success_request_count": 0,
                "client_rejection_count": 1,
                "service_failure_count": 0,
                "event_id_conflict_request_count": 1,
                "accepted_event_count": 0,
                "duplicate_event_count": 0,
                "idempotent_duplicate_event_count": 0,
                "unclassified_duplicate_event_count": 0,
                "conflicting_event_count": 1,
                "conflict_audit_event_count": 1,
                "event_id_reuse_count": 1,
                "classified_event_count": 1,
                "known_event_result_count": 0,
                "request_failure_rate": 1,
                "duplicate_event_rate": None,
                "event_id_reuse_rate": 1,
                "classified_conflicting_event_rate": 1,
                "recorded_rate_available": True,
                "end_to_end_coverage_available": False,
            }
        ],
        total=1,
    )


def _mixed_batch_ingestion_page() -> verifier.ReportPage:
    return _page(
        verifier.INGESTION_VIEW,
        [
            {
                "recorded_request_count": 1,
                "first_started_at": "2026-07-12T00:00:00Z",
                "last_finished_at": "2026-07-12T00:00:01Z",
                "success_request_count": 0,
                "client_rejection_count": 1,
                "service_failure_count": 0,
                "event_id_conflict_request_count": 1,
                "accepted_event_count": 0,
                "duplicate_event_count": 0,
                "idempotent_duplicate_event_count": 0,
                "unclassified_duplicate_event_count": 0,
                "conflicting_event_count": 1,
                "conflict_audit_event_count": 1,
                "event_id_reuse_count": 1,
                "classified_event_count": 2,
                "known_event_result_count": 0,
                "request_failure_rate": 1,
                "duplicate_event_rate": None,
                "event_id_reuse_rate": 0.5,
                "classified_conflicting_event_rate": 1,
                "recorded_rate_available": True,
                "end_to_end_coverage_available": False,
            }
        ],
        total=1,
    )


def _probe_question() -> verifier.AuthoritativeProbeQuestion:
    question = {
        "question_id": "q_authoritative",
        "family": "multivariate_regression",
        "dataset_id": "mvar_authoritative",
        "type": "mixed",
        "correct_letter": "A",
        "choices": [
            {"letter": "A", "candidate_id": "c_authoritative_a"},
            {"letter": "B", "candidate_id": "c_authoritative_b"},
        ],
    }
    return verifier.AuthoritativeProbeQuestion(
        registry_id="registry_" + "b" * 64,
        release_id="release_" + "a" * 64,
        question_id=question["question_id"],
        question_version=verifier.feedback.compute_question_version(question),
        family=question["family"],
        dataset_id=question["dataset_id"],
        question_type=question["type"],
        selected_letter="A",
        selected_candidate_id="c_authoritative_a",
        authoritative_is_correct=True,
        registry_question_count=1,
        registry_choice_count=2,
        manifest_sha256=MANIFEST_SHA256,
        question=question,
    )


def _assert_authoritative_evidence(
    result: verifier.RoundtripResult,
    probe: verifier.AuthoritativeProbeQuestion,
    *,
    verified_at: str | None = None,
) -> dict[str, Any]:
    payload = result.to_dict()
    assert payload["schema_version"] == verifier.EVIDENCE_SCHEMA_VERSION
    assert payload["evidence_type"] == verifier.EVIDENCE_TYPE
    assert payload["authority_mode"] == verifier.AUTHORITY_MODE_AUTHORITATIVE
    assert payload["manifest_sha256"] == probe.manifest_sha256
    assert payload["registry_id"] == probe.registry_id
    assert payload["registry_question_count"] == probe.registry_question_count
    assert payload["registry_choice_count"] == probe.registry_choice_count
    observed_at = payload["verified_at"]
    assert isinstance(observed_at, str)
    assert verifier._is_utc_rfc3339(observed_at)
    if verified_at is not None:
        assert observed_at == verified_at
    assert result.to_dict() == payload
    return payload


def _not_found_resolution_row(event_id: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": None,
        "occurred_at": None,
        "received_at": None,
        "session_id": None,
        "attempt_id": None,
        "client_release_id": None,
        "registry_status": "not_found",
        "answer_status": "not_found",
        "registry_id": None,
        "release_id": None,
        "question_id": None,
        "question_version": None,
        "family": None,
        "dataset_id": None,
        "question_type": None,
        "selected_letter": None,
        "client_selected_candidate_id": None,
        "selected_candidate_id": None,
        "authoritative_is_correct": None,
        "client_is_correct": None,
        "client_context_mismatch": False,
        "client_correctness_mismatch": False,
    }


def _authoritative_resolution_row(
    event: dict[str, Any],
    probe: verifier.AuthoritativeProbeQuestion,
) -> dict[str, Any]:
    payload = event["payload"]
    is_answer = event["event_type"] == "answer_submitted"
    return {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "occurred_at": event["occurred_at"],
        "received_at": "2026-07-12T00:00:01Z",
        "session_id": event["session_id"],
        "attempt_id": payload["attempt_id"],
        "client_release_id": payload["release_id"],
        "registry_status": "matched",
        "answer_status": "resolved" if is_answer else "not_answer",
        "registry_id": probe.registry_id,
        "release_id": probe.release_id,
        "question_id": probe.question_id,
        "question_version": probe.question_version,
        "family": probe.family,
        "dataset_id": probe.dataset_id,
        "question_type": probe.question_type,
        "selected_letter": payload.get("selected_letter") if is_answer else None,
        "client_selected_candidate_id": (
            payload.get("selected_candidate_id") if is_answer else None
        ),
        "selected_candidate_id": (probe.selected_candidate_id if is_answer else None),
        "authoritative_is_correct": (
            probe.authoritative_is_correct if is_answer else None
        ),
        "client_is_correct": payload.get("is_correct") if is_answer else None,
        "client_context_mismatch": True,
        "client_correctness_mismatch": is_answer,
    }


def _answer_detail_row(
    event: dict[str, Any],
    resolution: dict[str, Any],
    probe: verifier.AuthoritativeProbeQuestion,
) -> dict[str, Any]:
    payload = event["payload"]
    return {
        "event_id": event["event_id"],
        "occurred_at": resolution["occurred_at"],
        "received_at": resolution["received_at"],
        "session_id": event["session_id"],
        "attempt_id": payload["attempt_id"],
        "question_id": probe.question_id,
        "question_version": probe.question_version,
        "release_id": probe.release_id,
        "family": probe.family,
        "dataset_id": probe.dataset_id,
        "question_type": probe.question_type,
        "selected_letter": probe.selected_letter,
        "client_selected_candidate_id": probe.selected_candidate_id,
        "selected_candidate_id": probe.selected_candidate_id,
        "answer_status": "resolved",
        "is_correct": probe.authoritative_is_correct,
        "client_is_correct": not probe.authoritative_is_correct,
        "client_context_mismatch": True,
        "client_correctness_mismatch": True,
    }


def _proposal_detail_row(
    event: dict[str, Any],
    resolution: dict[str, Any],
    probe: verifier.AuthoritativeProbeQuestion,
) -> dict[str, Any]:
    payload = event["payload"]
    return {
        "event_id": event["event_id"],
        "occurred_at": resolution["occurred_at"],
        "received_at": resolution["received_at"],
        "session_id": event["session_id"],
        "attempt_id": payload["attempt_id"],
        "question_id": probe.question_id,
        "question_version": probe.question_version,
        "release_id": probe.release_id,
        "family": probe.family,
        "dataset_id": probe.dataset_id,
        "question_type": probe.question_type,
        "setting_status": "proposed",
        "label": payload["label"],
        "setting_json": json.dumps(
            payload["setting"], ensure_ascii=False, sort_keys=True
        ),
        "inherited_from_json": json.dumps(
            payload["inherited_from"], ensure_ascii=False, sort_keys=True
        ),
        "n_seeds": payload["n_seeds"],
        "base_seed": payload["base_seed"],
        "error_type": None,
    }


class AuthoritativeReportsClient:
    def __init__(self, probe: verifier.AuthoritativeProbeQuestion) -> None:
        self.probe = probe
        self.resolutions: dict[str, dict[str, Any]] = {}
        self.outcomes: dict[str, verifier.ReportPage] = {}
        self.authority_status = {
            "authority_revision": "registry_v1",
            "business_reports_authoritative": True,
            "registered_release_count": 1,
            "registered_question_count": probe.registry_question_count,
            "registered_choice_count": probe.registry_choice_count,
            "detail_revision": "detail_v1",
            "detail_reports_authoritative": True,
        }
        self.detail_pages: dict[str, tuple[tuple[dict[str, Any], ...], int]] = {}
        self.detail_rows: dict[str, list[dict[str, Any]]] = {
            view: [] for view in verifier.DETAIL_REPORT_VIEWS
        }
        self.business_events: list[dict[str, Any]] = []
        self.business_snapshot = _empty_business_snapshot(probe)
        self.calls: list[tuple[str, dict[str, Any], int, int]] = []

    def fetch_business_snapshot(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int,
    ) -> verifier.BusinessSnapshot:
        self.calls.append(
            (
                verifier.BUSINESS_SNAPSHOT_VIEW,
                dict(filters or {}),
                limit,
                0,
            )
        )
        resolved_filters = dict(filters or {})
        if set(resolved_filters) == {"session_id", "attempt_id"}:
            matching = [
                event
                for event in self.business_events
                if event["session_id"] == resolved_filters["session_id"]
                and event["payload"]["attempt_id"] == resolved_filters["attempt_id"]
            ]
            if matching:
                return _identity_business_snapshot(
                    self.probe,
                    matching,
                    limit=limit,
                )
        return deepcopy(self.business_snapshot)

    def fetch_page(
        self,
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        self.calls.append((view, dict(filters), limit, offset))
        if view == verifier.AUTHORITY_STATUS_VIEW:
            return verifier.ReportPage(
                view=view,
                rows=(deepcopy(self.authority_status),),
                total=1,
                limit=limit,
                offset=offset,
                request_id="authority-status-request",
            )
        if view == verifier.EVENT_RESOLUTION_VIEW:
            event_id = str(filters["event_id"])
            row = self.resolutions.get(
                event_id,
                _not_found_resolution_row(event_id),
            )
            return verifier.ReportPage(
                view=view,
                rows=(deepcopy(row),),
                total=1,
                limit=limit,
                offset=offset,
                request_id="resolution-request",
            )
        if view in verifier.DETAIL_REPORT_VIEWS:
            if view in self.detail_pages:
                rows, total = self.detail_pages[view]
                return verifier.ReportPage(
                    view=view,
                    rows=deepcopy(rows),
                    total=total,
                    limit=limit,
                    offset=offset,
                    request_id="detail-override-request",
                )
            rows = self.detail_rows[view]
            filtered = []
            for row in rows:
                if any(
                    key in filters and row[key] != value
                    for key, value in filters.items()
                    if key not in {"from", "to"}
                ):
                    continue
                occurred_at = datetime.fromisoformat(
                    str(row["occurred_at"]).replace("Z", "+00:00")
                )
                if "from" in filters and occurred_at < datetime.fromisoformat(
                    str(filters["from"]).replace("Z", "+00:00")
                ):
                    continue
                if "to" in filters and occurred_at >= datetime.fromisoformat(
                    str(filters["to"]).replace("Z", "+00:00")
                ):
                    continue
                filtered.append(row)
            filtered.sort(key=lambda row: str(row["event_id"]))
            filtered.sort(key=lambda row: str(row["occurred_at"]), reverse=True)
            total = len(filtered)
            return verifier.ReportPage(
                view=view,
                rows=tuple(deepcopy(filtered[offset : offset + limit])),
                total=total,
                limit=limit,
                offset=offset,
                request_id="detail-request",
            )
        if view == verifier.INGESTION_VIEW:
            return self.outcomes.get(
                str(filters["request_id"]),
                _empty_ingestion_page(),
            )
        raise AssertionError(f"authoritative verifier queried aggregate view {view}")


class AuthoritativeFeedbackClient:
    def __init__(
        self,
        reports: AuthoritativeReportsClient,
        probe: verifier.AuthoritativeProbeQuestion,
    ) -> None:
        self.reports = reports
        self.probe = probe
        self.raw_events: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self.successful_traces: list[dict[str, Any]] = []

    @staticmethod
    def _logical(event: dict[str, Any]) -> dict[str, Any]:
        return {
            key: deepcopy(event[key]) for key in verifier.feedback.LOGICAL_EVENT_KEYS
        }

    def seed_event(self, event: dict[str, Any]) -> None:
        copied = deepcopy(event)
        self.raw_events[copied["event_id"]] = copied
        self.reports.business_events.append(deepcopy(copied))
        resolution = _authoritative_resolution_row(copied, self.probe)
        self.reports.resolutions[copied["event_id"]] = resolution
        if copied["event_type"] == "answer_submitted":
            self.reports.detail_rows[verifier.ANSWER_DETAILS_VIEW].append(
                _answer_detail_row(copied, resolution, self.probe)
            )
        elif copied["event_type"] == "custom_setting_proposed":
            self.reports.detail_rows[verifier.PROPOSAL_DETAILS_VIEW].append(
                _proposal_detail_row(copied, resolution, self.probe)
            )

    def post_event(self, event: dict[str, Any]) -> verifier.feedback.UploadReceipt:
        copied = deepcopy(event)
        self.events.append(copied)
        existing = self.raw_events.get(copied["event_id"])
        if existing is None:
            self.seed_event(copied)
            accepted, duplicate = 1, 0
            self.reports.outcomes[INGEST_REQUEST_ID] = _ingestion_page(
                accepted=accepted,
                duplicate=duplicate,
            )
            return verifier.feedback.UploadReceipt(
                status_code=200,
                endpoint="https://ingest.example/feedback-ingest",
                request_id=INGEST_REQUEST_ID,
                response={
                    "accepted": accepted,
                    "duplicate": duplicate,
                    "conflict": 0,
                    "rejected": 0,
                    "request_id": INGEST_REQUEST_ID,
                },
            )
        if self._logical(existing) == self._logical(copied):
            self.reports.outcomes[INGEST_REQUEST_ID] = _ingestion_page(
                accepted=0,
                duplicate=1,
            )
            return verifier.feedback.UploadReceipt(
                status_code=200,
                endpoint="https://ingest.example/feedback-ingest",
                request_id=INGEST_REQUEST_ID,
                response={
                    "accepted": 0,
                    "duplicate": 1,
                    "conflict": 0,
                    "rejected": 0,
                    "request_id": INGEST_REQUEST_ID,
                },
            )
        self.reports.outcomes[CONFLICT_REQUEST_ID] = _conflict_ingestion_page()
        raise _structured_conflict_error(
            header_request_id=CONFLICT_REQUEST_ID,
            body_request_id=CONFLICT_REQUEST_ID,
        )

    def post_trace(
        self,
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        envelope = (
            trace.to_envelope()
            if isinstance(trace, verifier.feedback.SessionTrace)
            else deepcopy(trace)
        )
        self.traces.append(envelope)
        if any(
            event["event_id"].endswith(".mixed-withheld")
            for event in envelope["events"]
        ):
            self.reports.outcomes[MIXED_BATCH_REQUEST_ID] = (
                _mixed_batch_ingestion_page()
            )
            raise _structured_conflict_error(
                header_request_id=MIXED_BATCH_REQUEST_ID,
                body_request_id=MIXED_BATCH_REQUEST_ID,
                rejected=2,
            )

        self.successful_traces.append(envelope)
        exists = [event["event_id"] in self.raw_events for event in envelope["events"]]
        assert len(set(exists)) == 1
        event_count = len(envelope["events"])
        accepted, duplicate = (0, event_count) if exists[0] else (event_count, 0)
        if accepted:
            for event in envelope["events"]:
                self.seed_event(event)
        request_id = (
            SUCCESSFUL_BATCH_FIRST_REQUEST_ID
            if len(self.successful_traces) == 1
            else SUCCESSFUL_BATCH_REPLAY_REQUEST_ID
        )
        self.reports.outcomes[request_id] = _ingestion_page(
            accepted=accepted,
            duplicate=duplicate,
            event_count=event_count,
        )
        return verifier.feedback.UploadReceipt(
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


class FakeReportsClient:
    def __init__(
        self,
        identity: verifier.SmokeIdentity,
        *,
        visible: bool = False,
        successful_batch_visible: bool = False,
    ):
        self.identity = identity
        self.batch_identity, _ = verifier.build_successful_batch_trace(identity)
        self.visible = visible
        self.successful_batch_visible = successful_batch_visible
        self.outcome_visible = False
        self.outcome_request_id: str | None = None
        self.accepted = 0
        self.duplicate = 0
        self.outcome_delay_calls = 0
        self.successful_batch_outcomes: dict[str, tuple[int, int]] = {}
        self.successful_batch_outcome_delay_calls = 0
        self.conflict_outcome_visible = False
        self.conflict_outcome_delay_calls = 0
        self.mixed_outcome_visible = False
        self.mixed_outcome_delay_calls = 0
        self.calls: list[tuple[str, dict[str, Any], int, int]] = []

    def fetch_page(
        self,
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        self.calls.append((view, dict(filters), limit, offset))
        if view == verifier.INGESTION_VIEW:
            if self.outcome_visible and filters == {
                "request_id": self.outcome_request_id
            }:
                if self.outcome_delay_calls > 0:
                    self.outcome_delay_calls -= 1
                    return _empty_ingestion_page()
                return _ingestion_page(
                    accepted=self.accepted,
                    duplicate=self.duplicate,
                )
            batch_counts = self.successful_batch_outcomes.get(
                str(filters.get("request_id"))
            )
            if batch_counts is not None:
                if self.successful_batch_outcome_delay_calls > 0:
                    self.successful_batch_outcome_delay_calls -= 1
                    return _empty_ingestion_page()
                accepted, duplicate = batch_counts
                return _ingestion_page(
                    accepted=accepted,
                    duplicate=duplicate,
                    event_count=2,
                )
            if self.conflict_outcome_visible and filters == {
                "request_id": CONFLICT_REQUEST_ID
            }:
                if self.conflict_outcome_delay_calls > 0:
                    self.conflict_outcome_delay_calls -= 1
                    return _empty_ingestion_page()
                return _conflict_ingestion_page()
            if self.mixed_outcome_visible and filters == {
                "request_id": MIXED_BATCH_REQUEST_ID
            }:
                if self.mixed_outcome_delay_calls > 0:
                    self.mixed_outcome_delay_calls -= 1
                    return _empty_ingestion_page()
                return _mixed_batch_ingestion_page()
            return _empty_ingestion_page()
        if filters == verifier._filters(self.batch_identity):
            pages = (
                _successful_batch_pages(self.batch_identity)
                if self.successful_batch_visible
                else _empty_pages()
            )
        else:
            pages = _visible_pages(self.identity) if self.visible else _empty_pages()
        return pages[view]


class FakeFeedbackClient:
    def __init__(
        self,
        reports: FakeReportsClient,
        *,
        duplicate: bool = False,
    ) -> None:
        self.reports = reports
        self.duplicate = duplicate
        self.events: list[dict[str, Any]] = []
        self.traces: list[dict[str, Any]] = []
        self.successful_batch_traces: list[dict[str, Any]] = []
        self.conflict_started_after_report_calls: int | None = None
        self.mixed_started_after_report_calls: int | None = None

    def post_event(self, event: dict[str, Any]) -> verifier.feedback.UploadReceipt:
        self.events.append(event)
        if len(self.events) == 2:
            self.conflict_started_after_report_calls = len(self.reports.calls)
            self.reports.conflict_outcome_visible = True
            response = {
                "accepted": 0,
                "duplicate": 0,
                "conflict": 1,
                "rejected": 1,
                "request_id": CONFLICT_REQUEST_ID,
                "error": {
                    "code": "EVENT_ID_CONFLICT",
                    "message": "the event ID stores different logical content",
                },
            }
            raise verifier.feedback.FeedbackUploadConflictError(
                "feedback event ID conflicts with stored content",
                endpoint="https://ingest.example/feedback-ingest",
                status_code=409,
                response=response,
                request_id=CONFLICT_REQUEST_ID,
            )
        self.reports.visible = True
        self.reports.outcome_visible = True
        self.reports.outcome_request_id = INGEST_REQUEST_ID
        self.reports.accepted = 0 if self.duplicate else 1
        self.reports.duplicate = 1 if self.duplicate else 0
        return verifier.feedback.UploadReceipt(
            status_code=200,
            endpoint="https://ingest.example/feedback-ingest",
            request_id=INGEST_REQUEST_ID,
            response={
                "accepted": 0 if self.duplicate else 1,
                "duplicate": 1 if self.duplicate else 0,
                "conflict": 0,
                "rejected": 0,
                "request_id": INGEST_REQUEST_ID,
            },
        )

    def post_trace(
        self,
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        envelope = (
            trace.to_envelope()
            if isinstance(trace, verifier.feedback.SessionTrace)
            else trace
        )
        self.traces.append(envelope)
        event_ids = [event["event_id"] for event in envelope["events"]]
        if all(".batch-" in event_id for event_id in event_ids):
            was_visible = self.reports.successful_batch_visible
            self.reports.successful_batch_visible = True
            self.successful_batch_traces.append(envelope)
            request_id = (
                SUCCESSFUL_BATCH_FIRST_REQUEST_ID
                if len(self.successful_batch_traces) == 1
                else SUCCESSFUL_BATCH_REPLAY_REQUEST_ID
            )
            accepted = 0 if was_visible else 2
            duplicate = 2 if was_visible else 0
            self.reports.successful_batch_outcomes[request_id] = (
                accepted,
                duplicate,
            )
            return verifier.feedback.UploadReceipt(
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
        self.mixed_started_after_report_calls = len(self.reports.calls)
        self.reports.mixed_outcome_visible = True
        raise _structured_conflict_error(
            header_request_id=MIXED_BATCH_REQUEST_ID,
            body_request_id=MIXED_BATCH_REQUEST_ID,
            rejected=2,
        )


def _structured_conflict_error(
    *,
    status_code: int = 409,
    error_code: str = "EVENT_ID_CONFLICT",
    header_request_id: str | None = CONFLICT_REQUEST_ID,
    body_request_id: str = CONFLICT_REQUEST_ID,
    accepted: int = 0,
    duplicate: int = 0,
    conflict: int = 1,
    rejected: int = 1,
) -> verifier.feedback.FeedbackUploadConflictError:
    return verifier.feedback.FeedbackUploadConflictError(
        "feedback event ID conflicts with stored content",
        endpoint="https://ingest.example/feedback-ingest",
        status_code=status_code,
        response={
            "accepted": accepted,
            "duplicate": duplicate,
            "conflict": conflict,
            "rejected": rejected,
            "request_id": body_request_id,
            "error": {
                "code": error_code,
                "message": "the event ID stores different logical content",
            },
        },
        request_id=header_request_id,
    )


def test_default_bundle_selects_deterministic_attested_registry_membership() -> None:
    probe = verifier.load_authoritative_probe_question(verifier.DEFAULT_BUNDLE_PATH)
    registry = verifier.build_feedback_registry(verifier.DEFAULT_BUNDLE_PATH)
    manifest = verifier.load_quiz_manifest(verifier.DEFAULT_BUNDLE_PATH)
    first = registry["questions"][0]

    assert manifest is not None
    assert probe.registry_id == registry["registry_id"]
    assert probe.release_id == registry["release_id"]
    assert probe.manifest_sha256 == manifest.manifest_sha256
    assert probe.registry_question_count == registry["question_count"]
    assert probe.registry_choice_count == registry["choice_count"]
    assert probe.question_id == first["question_id"]
    assert probe.question_version == first["question_version"]
    assert probe.family == first["family"]
    assert probe.dataset_id == first["dataset_id"]
    assert probe.question_type == first["question_type"]
    assert probe.selected_letter == sorted(first["choices"])[0]
    assert probe.selected_candidate_id == first["choices"][probe.selected_letter]
    assert probe.authoritative_is_correct is (
        probe.selected_letter == first["correct_letter"]
    )
    assert (
        verifier.feedback.compute_question_version(probe.question)
        == probe.question_version
    )


@pytest.mark.parametrize(
    "verified_at",
    [
        "2026-07-12T12:34:56+00:00",
        "2026-07-12 12:34:56Z",
        "2026-02-30T12:34:56Z",
    ],
)
def test_authoritative_evidence_requires_real_utc_rfc3339(
    verified_at: str,
) -> None:
    probe = _probe_question()
    identity, _ = verifier.build_smoke_event(
        "bad_evidence_time",
        probe_question=probe,
    )

    with pytest.raises(
        verifier.RoundtripConfigurationError,
        match="UTC RFC3339",
    ):
        verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=1,
            receipt=None,
            registry_id=probe.registry_id,
            verified_at=verified_at,
            manifest_sha256=probe.manifest_sha256,
            registry_question_count=probe.registry_question_count,
            registry_choice_count=probe.registry_choice_count,
            authority_mode=verifier.AUTHORITY_MODE_AUTHORITATIVE,
        )


def test_authoritative_evidence_rejects_nullable_ledger_fields() -> None:
    probe = _probe_question()
    identity, _ = verifier.build_smoke_event(
        "missing_evidence",
        probe_question=probe,
    )

    with pytest.raises(
        verifier.RoundtripConfigurationError,
        match="authoritative evidence is missing",
    ):
        verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=1,
            receipt=None,
            authority_mode=verifier.AUTHORITY_MODE_AUTHORITATIVE,
        )


def test_authoritative_events_use_real_membership_and_false_client_claims() -> None:
    probe = _probe_question()
    identity, event = verifier.build_smoke_event(
        "authoritative_build",
        probe_question=probe,
    )
    batch_identity, trace = verifier.build_successful_batch_trace(
        identity,
        probe_question=probe,
    )

    assert identity.release_id == probe.release_id
    assert identity.question_id == probe.question_id
    assert identity.question_version == probe.question_version
    assert event["payload"]["release_id"] == probe.release_id
    assert event["payload"]["dataset_id"] == probe.dataset_id
    assert event["payload"]["family"] != probe.family
    assert event["payload"]["question_type"] != probe.question_type
    assert batch_identity.release_id == probe.release_id
    assert batch_identity.question_id == probe.question_id
    answer, proposal, comment = trace["events"]
    assert answer["question_version"] == probe.question_version
    assert answer["payload"]["selected_letter"] == probe.selected_letter
    assert answer["payload"]["selected_candidate_id"] == probe.selected_candidate_id
    assert answer["payload"]["is_correct"] is not probe.authoritative_is_correct
    assert answer["payload"]["family"] != probe.family
    assert answer["payload"]["question_type"] != probe.question_type
    assert proposal["event_id"] == batch_identity.proposal_event_id
    assert proposal["event_type"] == "custom_setting_proposed"
    assert proposal["question_version"] == probe.question_version
    assert proposal["payload"]["setting"]["run_id"] == "authoritative_build"
    assert proposal["payload"]["label"] == (
        f"{verifier.DETAIL_PROBE_LABEL_PREFIX} authoritative_build"
    )
    assert proposal["payload"]["n_seeds"] == verifier.DETAIL_PROBE_N_SEEDS
    assert proposal["payload"]["base_seed"] == verifier.DETAIL_PROBE_BASE_SEED
    assert proposal["payload"]["family"] != probe.family
    assert proposal["payload"]["question_type"] != probe.question_type
    assert "is_correct" not in comment["payload"]


def test_authoritative_roundtrip_uses_exact_resolution_not_shared_aggregates() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    evidence_clock_calls = 0

    def evidence_clock() -> str:
        nonlocal evidence_clock_calls
        evidence_clock_calls += 1
        return VERIFIED_AT

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="authoritative_success",
        probe_question=probe,
        include_mixed_batch_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        utc_now=evidence_clock,
    )

    assert evidence_clock_calls == 1
    assert result.registry_id == probe.registry_id
    assert result.authority_status_verified is True
    assert result.detail_reports_verified is True
    assert result.business_snapshot_verified is True
    assert result.session_attempt_filters_verified is True
    assert result.identity.release_id == probe.release_id
    assert result.identity.question_id == probe.question_id
    assert result.identity.question_version == probe.question_version
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is True
    assert result.conflict_verified is True
    assert result.mixed_batch_verified is True
    assert result.polls == 7
    assert {call[0] for call in reports.calls} == {
        verifier.AUTHORITY_STATUS_VIEW,
        verifier.BUSINESS_SNAPSHOT_VIEW,
        verifier.ANSWER_DETAILS_VIEW,
        verifier.PROPOSAL_DETAILS_VIEW,
        verifier.EVENT_RESOLUTION_VIEW,
        verifier.INGESTION_VIEW,
    }
    assert len(ingestion.raw_events) == 4
    assert len(ingestion.events) == 2
    assert len(ingestion.traces) == 3
    answer = next(
        event
        for event in ingestion.raw_events.values()
        if event["event_type"] == "answer_submitted"
    )
    assert answer["payload"]["is_correct"] is not probe.authoritative_is_correct
    assert answer["payload"]["family"] != probe.family
    assert answer["payload"]["question_type"] != probe.question_type
    proposal = next(
        event
        for event in ingestion.raw_events.values()
        if event["event_type"] == "custom_setting_proposed"
    )
    assert proposal["payload"]["setting"]["run_id"] == "authoritative_success"
    assert proposal["payload"]["n_seeds"] == verifier.DETAIL_PROBE_N_SEEDS
    assert proposal["payload"]["base_seed"] == verifier.DETAIL_PROBE_BASE_SEED
    withheld_event_id = f"{result.identity.event_id}.mixed-withheld"
    assert withheld_event_id not in ingestion.raw_events
    withheld_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.EVENT_RESOLUTION_VIEW
        and call[1] == {"event_id": withheld_event_id}
    ]
    assert len(withheld_calls) == 2
    exact_detail_calls = [
        call
        for call in reports.calls
        if call[0] in verifier.DETAIL_REPORT_VIEWS
        and call[1].get("question_id") == probe.question_id
    ]
    assert {call[0] for call in exact_detail_calls} == {
        verifier.ANSWER_DETAILS_VIEW,
        verifier.PROPOSAL_DETAILS_VIEW,
    }
    assert len(exact_detail_calls) == 2
    for _, filters, limit, offset in exact_detail_calls:
        assert filters.keys() == {
            "release_id",
            "family",
            "question_type",
            "question_id",
            "from",
            "to",
        }
        assert filters["release_id"] == probe.release_id
        assert filters["family"] == probe.family
        assert filters["question_type"] == probe.question_type
        assert filters["question_id"] == probe.question_id
        start = datetime.fromisoformat(filters["from"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(filters["to"].replace("Z", "+00:00"))
        assert end - start == timedelta(milliseconds=1)
        assert (limit, offset) == (verifier.DETAIL_PAGE_LIMIT, 0)
    payload = result.to_dict()
    _assert_authoritative_evidence(result, probe, verified_at=VERIFIED_AT)
    assert payload["registry_id"] == probe.registry_id
    assert payload["authority_status_verified"] is True
    assert payload["business_snapshot_verified"] is True
    assert payload["session_attempt_filters_verified"] is True
    snapshot_calls = [
        call for call in reports.calls if call[0] == verifier.BUSINESS_SNAPSHOT_VIEW
    ]
    assert len(snapshot_calls) == 4
    _, snapshot_filters, snapshot_limit, snapshot_offset = snapshot_calls[0]
    assert set(snapshot_filters) == {"question_id"}
    assert snapshot_filters["question_id"].startswith("verifier_snapshot_missing_")
    assert (snapshot_limit, snapshot_offset) == (1, 0)
    identity_snapshot_calls = snapshot_calls[1:]
    assert identity_snapshot_calls[0][1] == {
        "session_id": result.identity.session_id,
        "attempt_id": result.identity.attempt_id,
    }
    assert identity_snapshot_calls[0][2:] == (10, 0)
    assert identity_snapshot_calls[1][1]["attempt_id"] == result.identity.attempt_id
    assert identity_snapshot_calls[1][1]["session_id"].startswith(
        "verifier_wrong_session_"
    )
    assert identity_snapshot_calls[2][1]["session_id"] == result.identity.session_id
    assert identity_snapshot_calls[2][1]["attempt_id"].startswith(
        "verifier_wrong_attempt_"
    )
    assert all(call[2:] == (1, 0) for call in identity_snapshot_calls[1:])


def test_authoritative_skip_successful_batch_never_claims_real_detail_rows() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="authoritative_skip_details",
        probe_question=probe,
        skip_successful_batch_probe=True,
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        utc_now=lambda: VERIFIED_AT,
    )

    _assert_authoritative_evidence(result, probe, verified_at=VERIFIED_AT)
    assert result.authority_status_verified is True
    assert result.business_snapshot_verified is True
    assert result.session_attempt_filters_verified is False
    assert result.successful_batch_verified is False
    assert result.detail_reports_verified is False
    assert [event["event_type"] for event in ingestion.raw_events.values()] == [
        "comment_submitted"
    ]
    assert all(
        call[1].get("question_id") != probe.question_id
        for call in reports.calls
        if call[0] in verifier.DETAIL_REPORT_VIEWS
    )


@pytest.mark.parametrize("ignored_filter", ["session_id", "attempt_id"])
def test_authoritative_roundtrip_rejects_ignored_identity_filter(
    ignored_filter: str,
) -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    original_fetch = reports.fetch_business_snapshot

    def fetch_with_ignored_filter(
        *, filters: dict[str, Any] | None = None, limit: int
    ) -> verifier.BusinessSnapshot:
        resolved = dict(filters or {})
        wrong_prefix = f"verifier_wrong_{ignored_filter.removesuffix('_id')}_"
        if str(resolved.get(ignored_filter, "")).startswith(wrong_prefix):
            return _identity_business_snapshot(
                probe,
                reports.business_events,
                limit=limit,
            )
        return original_fetch(filters=filters, limit=limit)

    reports.fetch_business_snapshot = fetch_with_ignored_filter  # type: ignore[method-assign]
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 1.0 if clock_calls > 20 else 0.0

    with pytest.raises(verifier.RoundtripReportError, match="did not converge"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id=f"ignored_{ignored_filter}",
            probe_question=probe,
            skip_conflict_probe=True,
            timeout_seconds=0.5,
            poll_interval=0.1,
            monotonic=clock,
            sleep=lambda seconds: None,
            utc_now=lambda: VERIFIED_AT,
        )


def test_authoritative_conflict_only_branch_keeps_same_evidence_metadata() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="authoritative_conflict_only",
        probe_question=probe,
        skip_successful_batch_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        utc_now=lambda: VERIFIED_AT,
    )

    _assert_authoritative_evidence(result, probe, verified_at=VERIFIED_AT)
    assert result.conflict_verified is True
    assert result.mixed_batch_verified is False
    assert result.successful_batch_verified is False


@pytest.mark.parametrize(
    ("view", "field", "value"),
    [
        (verifier.ANSWER_DETAILS_VIEW, "family", verifier.WRONG_CLIENT_FAMILY),
        (verifier.ANSWER_DETAILS_VIEW, "is_correct", False),
        (verifier.ANSWER_DETAILS_VIEW, "client_context_mismatch", False),
        (verifier.ANSWER_DETAILS_VIEW, "client_correctness_mismatch", False),
        (verifier.PROPOSAL_DETAILS_VIEW, "family", verifier.WRONG_CLIENT_FAMILY),
        (verifier.PROPOSAL_DETAILS_VIEW, "setting_status", "rejected"),
        (verifier.PROPOSAL_DETAILS_VIEW, "setting_json", "{}"),
        (verifier.PROPOSAL_DETAILS_VIEW, "base_seed", 1702),
    ],
)
def test_authoritative_detail_proof_rejects_tampered_real_row_semantics(
    view: str,
    field: str,
    value: Any,
) -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    original_seed_event = ingestion.seed_event

    def seed_event(event: dict[str, Any]) -> None:
        original_seed_event(event)
        target_type = (
            "answer_submitted"
            if view == verifier.ANSWER_DETAILS_VIEW
            else "custom_setting_proposed"
        )
        if event["event_type"] == target_type:
            reports.detail_rows[view][-1][field] = value

    ingestion.seed_event = seed_event  # type: ignore[method-assign]
    monotonic_values = iter((0.0, 0.0, 0.0, 1.0))
    with pytest.raises(
        verifier.RoundtripReportError,
        match="authoritative detail reports did not converge",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id=f"tampered_{field}",
            probe_question=probe,
            skip_conflict_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: next(monotonic_values),
            sleep=lambda seconds: None,
        )


def test_authoritative_detail_proof_binds_timestamps_to_uploaded_event() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    original_seed_event = ingestion.seed_event

    def seed_event(event: dict[str, Any]) -> None:
        original_seed_event(event)
        if event["event_type"] != "answer_submitted":
            return
        shifted = (
            (
                datetime.fromisoformat(event["occurred_at"].replace("Z", "+00:00"))
                + timedelta(seconds=1)
            )
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        reports.resolutions[event["event_id"]]["occurred_at"] = shifted
        reports.detail_rows[verifier.ANSWER_DETAILS_VIEW][-1]["occurred_at"] = shifted

    ingestion.seed_event = seed_event  # type: ignore[method-assign]
    monotonic_values = iter((0.0, 0.0, 0.0, 1.0))
    with pytest.raises(
        verifier.RoundtripReportError,
        match="occurred_at does not match the uploaded probe event",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="tampered_joint_timestamp",
            probe_question=probe,
            skip_conflict_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: next(monotonic_values),
            sleep=lambda seconds: None,
        )


def test_authoritative_detail_proof_scans_later_pages_for_exact_event() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    original_seed_event = ingestion.seed_event

    def seed_event(event: dict[str, Any]) -> None:
        original_seed_event(event)
        if event["event_type"] != "answer_submitted":
            return
        target = reports.detail_rows[verifier.ANSWER_DETAILS_VIEW][-1]
        for index in range(verifier.DETAIL_PAGE_LIMIT):
            noise = deepcopy(target)
            noise["event_id"] = f"detail_noise_{index:04d}"
            reports.detail_rows[verifier.ANSWER_DETAILS_VIEW].append(noise)

    ingestion.seed_event = seed_event  # type: ignore[method-assign]
    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="detail_second_page",
        probe_question=probe,
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.detail_reports_verified is True
    answer_detail_calls = [
        (limit, offset)
        for view, filters, limit, offset in reports.calls
        if view == verifier.ANSWER_DETAILS_VIEW
        and filters.get("question_id") == probe.question_id
    ]
    assert answer_detail_calls == [
        (verifier.DETAIL_PAGE_LIMIT, 0),
        (verifier.DETAIL_PAGE_LIMIT, verifier.DETAIL_PAGE_LIMIT),
    ]


def test_authoritative_resume_requires_and_preserves_exact_identity() -> None:
    probe = _probe_question()
    identity, event = verifier.build_smoke_event(
        "authoritative_resume",
        probe_question=probe,
    )
    batch_identity, trace = verifier.build_successful_batch_trace(
        identity,
        probe_question=probe,
    )
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    ingestion.seed_event(event)
    for batch_event in trace["events"]:
        ingestion.seed_event(batch_event)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="authoritative_resume",
        probe_question=probe,
        resume=True,
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        utc_now=lambda: VERIFIED_AT,
    )

    _assert_authoritative_evidence(result, probe, verified_at=VERIFIED_AT)
    assert result.identity == identity
    assert result.receipt is not None
    assert result.receipt["duplicate"] == 1
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is False
    assert len(ingestion.raw_events) == 4
    assert reports.resolutions[batch_identity.answer_event_id]["answer_status"] == (
        "resolved"
    )


def test_authoritative_probe_evidence_is_validated_before_any_hosted_request() -> None:
    probe = replace(_probe_question(), manifest_sha256="not-a-sha")
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripConfigurationError,
        match="manifest SHA-256",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="invalid_probe_evidence",
            probe_question=probe,
        )

    assert reports.calls == []
    assert ingestion.events == []
    assert ingestion.traces == []


def test_authoritative_roundtrip_requires_authority_status_before_post() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    reports.authority_status["authority_revision"] = "client_payload_v0"
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="authority status revision",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="missing_15000_authority",
            probe_question=probe,
        )

    assert reports.calls == [
        (verifier.AUTHORITY_STATUS_VIEW, {}, 1, 0),
    ]
    assert ingestion.events == []
    assert ingestion.traces == []


def test_authoritative_roundtrip_requires_detail_v1_before_post() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    reports.authority_status["detail_revision"] = "detail_v0"
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="detail_v1",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="missing_16000_authority",
            probe_question=probe,
        )

    assert reports.calls == [
        (verifier.AUTHORITY_STATUS_VIEW, {}, 1, 0),
    ]
    assert ingestion.events == []
    assert ingestion.traces == []


def test_authoritative_roundtrip_requires_detail_edge_views_before_post() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    reports.detail_pages[verifier.ANSWER_DETAILS_VIEW] = (({"unexpected": True},), 1)
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="negative-control page",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="missing_detail_edge",
            probe_question=probe,
        )

    assert [call[0] for call in reports.calls] == [
        verifier.AUTHORITY_STATUS_VIEW,
        verifier.ANSWER_DETAILS_VIEW,
    ]
    assert ingestion.events == []
    assert ingestion.traces == []


@pytest.mark.parametrize(
    "failure_mode",
    [
        "missing_view",
        "old_snapshot_revision",
        "old_authority_revision",
        "old_detail_revision",
        "nonempty_summary",
        "nonempty_page",
        "insufficient_registry_counts",
    ],
)
def test_authoritative_roundtrip_requires_empty_business_snapshot_before_post(
    failure_mode: str,
) -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    snapshot = reports.business_snapshot
    if failure_mode == "missing_view":
        pages = dict(snapshot.pages)
        pages.pop(verifier.COMMENTS_VIEW)
        reports.business_snapshot = replace(snapshot, pages=pages)
    elif failure_mode == "old_snapshot_revision":
        reports.business_snapshot = replace(
            snapshot,
            snapshot_revision="business_snapshot_v0",
        )
    elif failure_mode == "old_authority_revision":
        reports.business_snapshot = replace(
            snapshot,
            authority_revision="registry_v0",
        )
    elif failure_mode == "old_detail_revision":
        reports.business_snapshot = replace(
            snapshot,
            detail_revision="detail_v0",
        )
    elif failure_mode == "nonempty_summary":
        pages = dict(snapshot.pages)
        summary_page = pages[verifier.SUMMARY_VIEW]
        summary = dict(summary_page.rows[0])
        summary["event_count"] = 1
        summary["first_event_at"] = "2026-07-12T00:00:00Z"
        summary["last_event_at"] = "2026-07-12T00:00:00Z"
        pages[verifier.SUMMARY_VIEW] = verifier.ReportPage(
            view=summary_page.view,
            rows=(summary,),
            total=summary_page.total,
            limit=summary_page.limit,
            offset=summary_page.offset,
            request_id=summary_page.request_id,
        )
        reports.business_snapshot = replace(snapshot, pages=pages)
    elif failure_mode == "nonempty_page":
        pages = dict(snapshot.pages)
        comments_page = pages[verifier.COMMENTS_VIEW]
        pages[verifier.COMMENTS_VIEW] = verifier.ReportPage(
            view=comments_page.view,
            rows=({"event_id": "unexpected_snapshot_event"},),
            total=1,
            limit=comments_page.limit,
            offset=comments_page.offset,
            request_id=comments_page.request_id,
        )
        reports.business_snapshot = replace(snapshot, pages=pages)
    else:
        reports.business_snapshot = replace(
            snapshot,
            registered_question_count=probe.registry_question_count - 1,
        )
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="business snapshot negative control",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id=f"snapshot_{failure_mode}",
            probe_question=probe,
        )

    assert [call[0] for call in reports.calls] == [
        verifier.AUTHORITY_STATUS_VIEW,
        verifier.ANSWER_DETAILS_VIEW,
        verifier.PROPOSAL_DETAILS_VIEW,
        verifier.BUSINESS_SNAPSHOT_VIEW,
    ]
    assert ingestion.events == []
    assert ingestion.traces == []


def test_authoritative_resume_refuses_not_found_without_posting() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="requires the exact original event",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="authoritative_missing_resume",
            probe_question=probe,
            resume=True,
        )

    assert ingestion.events == []
    assert ingestion.traces == []


def test_authoritative_roundtrip_rejects_client_claim_as_canonical_readback() -> None:
    probe = _probe_question()
    reports = AuthoritativeReportsClient(probe)
    ingestion = AuthoritativeFeedbackClient(reports, probe)
    original_fetch_page = reports.fetch_page

    def fetch_page(
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        page = original_fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        if (
            view == verifier.EVENT_RESOLUTION_VIEW
            and page.rows[0]["registry_status"] == "matched"
        ):
            row = dict(page.rows[0])
            row["family"] = verifier.WRONG_CLIENT_FAMILY
            return verifier.ReportPage(
                view=page.view,
                rows=(row,),
                total=page.total,
                limit=page.limit,
                offset=page.offset,
                request_id=page.request_id,
            )
        return page

    reports.fetch_page = fetch_page  # type: ignore[method-assign]
    monotonic_values = iter((0.0, 1.0))
    with pytest.raises(
        verifier.RoundtripReportError,
        match="authoritative comment reports.*family",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="authoritative_client_claim",
            probe_question=probe,
            skip_successful_batch_probe=True,
            skip_conflict_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: next(monotonic_values),
            sleep=lambda seconds: None,
        )


def test_smoke_event_is_deterministic_and_uses_production_schema() -> None:
    before = datetime.now(timezone.utc) - timedelta(seconds=1)
    first_identity, first_event = verifier.build_smoke_event("run_123")
    second_identity, second_event = verifier.build_smoke_event("run_123")
    other_identity, _ = verifier.build_smoke_event("run_456")
    after = datetime.now(timezone.utc) + timedelta(seconds=1)

    assert first_identity == second_identity
    assert {
        key: value for key, value in first_event.items() if key != "occurred_at"
    } == {key: value for key, value in second_event.items() if key != "occurred_at"}
    assert first_identity != other_identity
    occurred_at = datetime.fromisoformat(
        first_event["occurred_at"].replace("Z", "+00:00")
    )
    assert before <= occurred_at <= after
    assert first_event["event_type"] == "comment_submitted"
    assert first_event["event_id"] == first_identity.event_id
    assert first_event["payload"]["release_id"] == first_identity.release_id
    assert first_event["payload"]["attempt_id"] == first_identity.attempt_id
    assert (
        verifier.feedback.build_session_trace_envelope(
            first_identity.session_id,
            [first_event],
        )["event_count"]
        == 1
    )


def test_successful_batch_trace_is_answer_comment_same_session_and_replayable() -> None:
    smoke_identity, _ = verifier.build_smoke_event("batch_trace")

    first_identity, first = verifier.build_successful_batch_trace(smoke_identity)
    second_identity, second = verifier.build_successful_batch_trace(smoke_identity)

    assert first_identity == second_identity
    assert first["trace_id"] == second["trace_id"]
    assert first["session_id"] == smoke_identity.session_id
    assert first["event_count"] == 2
    assert [event["sequence"] for event in first["events"]] == [1, 2]
    answer, comment = first["events"]
    assert [answer["event_type"], comment["event_type"]] == [
        "answer_submitted",
        "comment_submitted",
    ]
    assert answer["event_id"] == first_identity.answer_event_id
    assert comment["event_id"] == first_identity.comment_event_id
    assert first_identity.proposal_event_id is None
    assert answer["question_id"] == comment["question_id"]
    assert answer["question_version"] == comment["question_version"]
    assert answer["payload"]["selected_letter"] == "A"
    assert comment["payload"]["text"] == first_identity.comment_text
    assert first_identity.question_id != smoke_identity.question_id
    assert first_identity.release_id != smoke_identity.release_id
    for first_event, second_event in zip(
        first["events"], second["events"], strict=True
    ):
        assert {
            key: value for key, value in first_event.items() if key != "occurred_at"
        } == {key: value for key, value in second_event.items() if key != "occurred_at"}


def test_mixed_batch_trace_is_deterministic_discoverable_and_strictly_ordered() -> None:
    identity, event = verifier.build_smoke_event("mixed_trace")

    first = verifier._mixed_batch_trace(identity, event)
    second = verifier._mixed_batch_trace(identity, event)

    assert first == second
    assert set(first) == {
        "schema_version",
        "envelope_type",
        "trace_id",
        "session_id",
        "created_at",
        "event_count",
        "events",
    }
    assert first["trace_id"].startswith("trace_")
    assert len(first["trace_id"]) == len("trace_") + 64
    assert first["session_id"] == identity.session_id
    assert first["event_count"] == 2
    assert [item["sequence"] for item in first["events"]] == [1, 2]
    conflict, withheld = first["events"]
    assert conflict["event_id"] == identity.event_id
    assert conflict["payload"]["text"] == (
        f"{identity.comment_text}{verifier.CONFLICT_COMMENT_SUFFIX}"
    )
    assert withheld["event_id"] == "evt_e2e_mixed_trace.mixed-withheld"
    assert withheld["event_id"] != conflict["event_id"]
    assert withheld["payload"]["text"] == (
        f"{verifier.MIXED_BATCH_COMMENT_PREFIX} mixed_trace"
    )
    for field in (
        "attempt_id",
        "release_id",
        "family",
        "dataset_id",
        "question_type",
    ):
        assert withheld["payload"][field] == event["payload"][field]
    assert withheld["question_id"] == identity.question_id
    assert withheld["question_version"] == identity.question_version

    collision_identity, _ = verifier.build_smoke_event("mixed_mixed_trace")
    assert withheld["event_id"] != collision_identity.event_id
    assert len(withheld["event_id"]) <= 200
    assert "\n" not in withheld["event_id"]


@pytest.mark.parametrize("run_id", ["", "has space", "x" * 49, "slash/value"])
def test_smoke_run_id_rejects_unsafe_values(run_id: str) -> None:
    with pytest.raises(verifier.RoundtripConfigurationError, match="run id"):
        verifier.build_smoke_event(run_id)


def test_roundtrip_requires_empty_preflight_then_all_four_exact_reports() -> None:
    identity, _ = verifier.build_smoke_event("success")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="success",
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
        utc_now=lambda: VERIFIED_AT,
    )

    assert result.identity == identity
    assert result.request_id == INGEST_REQUEST_ID
    assert result.conflict_request_id == CONFLICT_REQUEST_ID
    assert result.conflict_verified is True
    assert result.successful_batch_first_request_id == (
        SUCCESSFUL_BATCH_FIRST_REQUEST_ID
    )
    assert result.successful_batch_replay_request_id == (
        SUCCESSFUL_BATCH_REPLAY_REQUEST_ID
    )
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is True
    assert result.business_snapshot_verified is False
    assert result.mixed_batch_request_id is None
    assert result.mixed_batch_verified is False
    payload = result.to_dict()
    assert payload["schema_version"] == verifier.EVIDENCE_SCHEMA_VERSION
    assert payload["evidence_type"] == verifier.EVIDENCE_TYPE
    assert payload["verified_at"] == VERIFIED_AT
    assert payload["authority_mode"] == verifier.AUTHORITY_MODE_LEGACY
    assert payload["manifest_sha256"] is None
    assert payload["registry_question_count"] is None
    assert payload["registry_choice_count"] is None
    assert result.to_dict() == payload
    safe_result = json.dumps(result.to_dict(), sort_keys=True)
    assert verifier.CONFLICT_COMMENT_SUFFIX not in safe_result
    assert "EVENT_ID_CONFLICT" not in safe_result
    assert result.polls == 4
    assert len(ingestion.events) == 2
    assert len(ingestion.traces) == 2
    assert ingestion.traces[0] == ingestion.traces[1]
    assert ingestion.events[1]["event_id"] == ingestion.events[0]["event_id"]
    assert ingestion.events[1]["payload"]["text"] == (
        f"{identity.comment_text}{verifier.CONFLICT_COMMENT_SUFFIX}"
    )
    assert ingestion.events[0]["payload"]["text"] == identity.comment_text
    assert ingestion.conflict_started_after_report_calls == 25
    assert len(reports.calls) == 30
    assert {call[0] for call in reports.calls} == {
        *verifier.REPORT_VIEWS,
        verifier.INGESTION_VIEW,
    }
    batch_identity, _ = verifier.build_successful_batch_trace(identity)
    business_filters = {
        tuple(sorted(call[1].items()))
        for call in reports.calls
        if call[0] in verifier.REPORT_VIEWS
    }
    assert business_filters == {
        tuple(sorted(verifier._filters(identity).items())),
        tuple(sorted(verifier._filters(batch_identity).items())),
    }
    assert reports.calls[13] == (
        verifier.INGESTION_VIEW,
        {"request_id": INGEST_REQUEST_ID},
        1,
        0,
    )
    assert reports.calls[14] == (
        verifier.INGESTION_VIEW,
        {"request_id": verifier._negative_control_request_id(INGEST_REQUEST_ID)},
        1,
        0,
    )
    assert reports.calls[-1] == (
        verifier.INGESTION_VIEW,
        {"request_id": CONFLICT_REQUEST_ID},
        1,
        0,
    )
    preflight_view, preflight_filters, preflight_limit, preflight_offset = (
        reports.calls[8]
    )
    assert preflight_view == verifier.INGESTION_VIEW
    assert verifier.UUID_PATTERN.fullmatch(preflight_filters["request_id"])
    assert preflight_filters["request_id"] != INGEST_REQUEST_ID
    assert (preflight_limit, preflight_offset) == (1, 0)


def test_roundtrip_opt_in_mixed_batch_proves_all_or_none_and_first_write_wins() -> None:
    identity, _ = verifier.build_smoke_event("mixed_success")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="mixed_success",
        include_mixed_batch_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.request_id == INGEST_REQUEST_ID
    assert result.conflict_request_id == CONFLICT_REQUEST_ID
    assert result.conflict_verified is True
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is True
    assert result.mixed_batch_request_id == MIXED_BATCH_REQUEST_ID
    assert result.mixed_batch_verified is True
    assert result.polls == 5
    assert len(ingestion.events) == 2
    assert len(ingestion.traces) == 3
    trace = ingestion.traces[-1]
    assert trace["session_id"] == identity.session_id
    assert trace["event_count"] == 2
    assert [item["sequence"] for item in trace["events"]] == [1, 2]
    assert trace["events"][0]["event_id"] == identity.event_id
    assert trace["events"][1]["event_id"] != identity.event_id
    assert trace["events"][1]["payload"]["text"] == (
        f"{verifier.MIXED_BATCH_COMMENT_PREFIX} mixed_success"
    )
    assert (
        len(
            {
                result.request_id,
                result.successful_batch_first_request_id,
                result.successful_batch_replay_request_id,
                result.conflict_request_id,
                result.mixed_batch_request_id,
            }
        )
        == 5
    )
    assert reports.calls[-1] == (
        verifier.INGESTION_VIEW,
        {"request_id": MIXED_BATCH_REQUEST_ID},
        1,
        0,
    )
    mixed_start = ingestion.mixed_started_after_report_calls
    assert mixed_start is not None
    mixed_business_calls = [
        call for call in reports.calls[mixed_start:] if call[0] in verifier.REPORT_VIEWS
    ]
    assert len(mixed_business_calls) == 4
    payload = result.to_dict()
    assert payload["mixed_batch_request_id"] == MIXED_BATCH_REQUEST_ID
    assert payload["mixed_batch_verified"] is True


def test_roundtrip_rejects_skip_plus_mixed_before_preflight_or_any_post() -> None:
    identity, _ = verifier.build_smoke_event("incompatible_probes")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    with pytest.raises(
        verifier.RoundtripConfigurationError,
        match="cannot be combined",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="incompatible_probes",
            skip_conflict_probe=True,
            include_mixed_batch_probe=True,
        )

    assert ingestion.events == []
    assert ingestion.traces == []
    assert reports.calls == []


def test_roundtrip_polls_mixed_outcome_before_marking_batch_verified() -> None:
    identity, _ = verifier.build_smoke_event("delayed_mixed")
    reports = FakeReportsClient(identity)
    reports.mixed_outcome_delay_calls = 1
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="delayed_mixed",
        include_mixed_batch_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.polls == 6
    assert result.mixed_batch_verified is True
    mixed_outcome_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.INGESTION_VIEW
        and call[1] == {"request_id": MIXED_BATCH_REQUEST_ID}
    ]
    assert len(mixed_outcome_calls) == 2


@pytest.mark.parametrize("failure_mode", ["fresh_stored", "original_overwritten"])
def test_roundtrip_mixed_probe_rejects_any_partial_business_write(
    failure_mode: str,
) -> None:
    identity, _ = verifier.build_smoke_event(f"mixed_{failure_mode}")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_fetch_page = reports.fetch_page

    def fetch_page(
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        page = original_fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        if view == verifier.COMMENTS_VIEW and reports.mixed_outcome_visible:
            if failure_mode == "original_overwritten":
                page.rows[0]["comment_text"] = "overwritten by mixed conflict"
            else:
                fresh = dict(page.rows[0])
                fresh["event_id"] = f"{identity.event_id}.mixed-withheld"
                fresh["comment_text"] = (
                    f"{verifier.MIXED_BATCH_COMMENT_PREFIX} {identity.run_id}"
                )
                page = _page(
                    verifier.COMMENTS_VIEW,
                    [dict(page.rows[0]), fresh],
                    total=2,
                )
        return page

    reports.fetch_page = fetch_page  # type: ignore[method-assign]
    monotonic_values = iter((0.0, 0.0, 0.0, 1.0))

    with pytest.raises(verifier.RoundtripReportError, match="mixed-batch reports"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id=f"mixed_{failure_mode}",
            skip_successful_batch_probe=True,
            include_mixed_batch_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: next(monotonic_values),
            sleep=lambda seconds: None,
        )


def test_roundtrip_resume_replays_same_event_as_duplicate() -> None:
    identity, _ = verifier.build_smoke_event("resume")
    reports = FakeReportsClient(identity, visible=True)
    ingestion = FakeFeedbackClient(reports, duplicate=True)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="resume",
        resume=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.receipt == {
        "accepted": 0,
        "duplicate": 1,
        "conflict": 0,
        "rejected": 0,
        "request_id": INGEST_REQUEST_ID,
    }
    assert ingestion.events[0]["event_id"] == identity.event_id
    assert ingestion.events[1]["event_id"] == identity.event_id
    assert result.conflict_verified is True
    assert result.conflict_request_id == CONFLICT_REQUEST_ID
    assert reports.calls[-1][1] == {"request_id": CONFLICT_REQUEST_ID}


def test_roundtrip_resume_existing_batch_proves_replay_but_not_first_write() -> None:
    identity, _ = verifier.build_smoke_event("resume_existing_batch")
    reports = FakeReportsClient(
        identity,
        visible=True,
        successful_batch_visible=True,
    )
    ingestion = FakeFeedbackClient(reports, duplicate=True)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="resume_existing_batch",
        resume=True,
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is False
    assert result.successful_batch_first_request_id == (
        SUCCESSFUL_BATCH_FIRST_REQUEST_ID
    )
    assert result.successful_batch_replay_request_id == (
        SUCCESSFUL_BATCH_REPLAY_REQUEST_ID
    )
    assert len(ingestion.successful_batch_traces) == 2
    assert ingestion.successful_batch_traces[0] == ingestion.successful_batch_traces[1]
    assert reports.successful_batch_outcomes == {
        SUCCESSFUL_BATCH_FIRST_REQUEST_ID: (0, 2),
        SUCCESSFUL_BATCH_REPLAY_REQUEST_ID: (0, 2),
    }


def test_roundtrip_fresh_refuses_preexisting_batch_before_any_post() -> None:
    identity, _ = verifier.build_smoke_event("preexisting_batch")
    reports = FakeReportsClient(identity, successful_batch_visible=True)
    ingestion = FakeFeedbackClient(reports)

    with pytest.raises(
        verifier.RoundtripPreflightError,
        match="successful-batch namespace already contains",
    ):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="preexisting_batch",
        )

    assert ingestion.events == []
    assert ingestion.traces == []


def test_roundtrip_can_explicitly_skip_successful_batch_without_claiming_proof() -> (
    None
):
    identity, _ = verifier.build_smoke_event("skip_successful_batch")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="skip_successful_batch",
        skip_successful_batch_probe=True,
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert ingestion.traces == []
    assert result.successful_batch_first_request_id is None
    assert result.successful_batch_replay_request_id is None
    assert result.successful_batch_verified is False
    assert result.successful_batch_first_write_verified is False
    assert result.to_dict()["successful_batch_verified"] is False


def test_roundtrip_polls_each_successful_batch_outcome_and_readback() -> None:
    identity, _ = verifier.build_smoke_event("delayed_successful_batch")
    reports = FakeReportsClient(identity)
    reports.successful_batch_outcome_delay_calls = 2
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="delayed_successful_batch",
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.polls == 5
    first_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.INGESTION_VIEW
        and call[1] == {"request_id": SUCCESSFUL_BATCH_FIRST_REQUEST_ID}
    ]
    replay_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.INGESTION_VIEW
        and call[1] == {"request_id": SUCCESSFUL_BATCH_REPLAY_REQUEST_ID}
    ]
    assert len(first_calls) == 3
    assert len(replay_calls) == 1


def test_roundtrip_rejects_inexact_successful_batch_first_receipt() -> None:
    identity, _ = verifier.build_smoke_event("bad_batch_first_receipt")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    def post_trace(
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        del trace
        return verifier.feedback.UploadReceipt(
            status_code=200,
            endpoint="https://ingest.example/feedback-ingest",
            request_id=SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
            response={
                "accepted": 1,
                "duplicate": 1,
                "conflict": 0,
                "rejected": 0,
                "request_id": SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
            },
        )

    ingestion.post_trace = post_trace  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="accepted=2"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="bad_batch_first_receipt",
            skip_conflict_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


def test_roundtrip_rejects_successful_batch_request_id_reuse() -> None:
    identity, _ = verifier.build_smoke_event("batch_request_reuse")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_post_trace = ingestion.post_trace

    def post_trace(
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        receipt = original_post_trace(trace)
        if len(ingestion.successful_batch_traces) == 2:
            return verifier.feedback.UploadReceipt(
                status_code=receipt.status_code,
                endpoint=receipt.endpoint,
                request_id=SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
                response={
                    **(receipt.response or {}),
                    "request_id": SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
                },
            )
        return receipt

    ingestion.post_trace = post_trace  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="replay reused"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="batch_request_reuse",
            skip_conflict_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


def test_roundtrip_resume_can_opt_in_to_mixed_batch_probe() -> None:
    identity, _ = verifier.build_smoke_event("resume_mixed")
    reports = FakeReportsClient(identity, visible=True)
    ingestion = FakeFeedbackClient(reports, duplicate=True)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="resume_mixed",
        resume=True,
        include_mixed_batch_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.receipt is not None
    assert result.receipt["duplicate"] == 1
    assert result.conflict_verified is True
    assert result.mixed_batch_request_id == MIXED_BATCH_REQUEST_ID
    assert result.mixed_batch_verified is True
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is True
    assert len(ingestion.traces) == 3


def test_roundtrip_polls_until_asynchronous_outcome_is_persisted() -> None:
    identity, _ = verifier.build_smoke_event("delayed_outcome")
    reports = FakeReportsClient(identity)
    reports.outcome_delay_calls = 1
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="delayed_outcome",
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.polls == 5
    actual_outcome_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.INGESTION_VIEW
        and call[1] == {"request_id": INGEST_REQUEST_ID}
    ]
    assert len(actual_outcome_calls) == 2


def test_roundtrip_polls_conflict_outcome_then_preserves_original_comment() -> None:
    identity, _ = verifier.build_smoke_event("delayed_conflict")
    reports = FakeReportsClient(identity)
    reports.conflict_outcome_delay_calls = 1
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="delayed_conflict",
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert result.polls == 5
    conflict_outcome_calls = [
        call
        for call in reports.calls
        if call[0] == verifier.INGESTION_VIEW
        and call[1] == {"request_id": CONFLICT_REQUEST_ID}
    ]
    assert len(conflict_outcome_calls) == 2
    comment_calls_after_conflict = [
        call
        for call in reports.calls[ingestion.conflict_started_after_report_calls or 0 :]
        if call[0] == verifier.COMMENTS_VIEW
    ]
    assert len(comment_calls_after_conflict) == 2


def test_roundtrip_explicit_skip_marks_conflict_unverified_without_second_post() -> (
    None
):
    identity, _ = verifier.build_smoke_event("skip_conflict")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    result = verifier.run_roundtrip(
        ingestion,
        reports,
        run_id="skip_conflict",
        skip_conflict_probe=True,
        timeout_seconds=1,
        poll_interval=0.1,
        monotonic=lambda: 0.0,
        sleep=lambda seconds: None,
    )

    assert len(ingestion.events) == 1
    assert len(ingestion.successful_batch_traces) == 2
    assert result.successful_batch_verified is True
    assert result.successful_batch_first_write_verified is True
    assert result.conflict_request_id is None
    assert result.conflict_verified is False
    assert result.to_dict()["conflict_verified"] is False
    assert result.polls == 3


def test_roundtrip_rejects_conflict_if_business_view_no_longer_has_first_write() -> (
    None
):
    identity, _ = verifier.build_smoke_event("overwritten_comment")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_fetch_page = reports.fetch_page

    def fetch_page(
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        page = original_fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )
        if view == verifier.COMMENTS_VIEW and reports.conflict_outcome_visible:
            page.rows[0]["comment_text"] = "overwritten by conflict probe"
        return page

    reports.fetch_page = fetch_page  # type: ignore[method-assign]
    monotonic_values = iter((0.0, 0.0, 1.0))

    with pytest.raises(verifier.RoundtripReportError, match="comment_text"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="overwritten_comment",
            skip_successful_batch_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: next(monotonic_values),
            sleep=lambda seconds: None,
        )


@pytest.mark.parametrize(
    ("generic_response", "message"),
    [(False, "duplicate-only"), (True, "matching UUID")],
)
def test_roundtrip_resume_rejects_new_accept_or_generic_2xx(
    generic_response: bool,
    message: str,
) -> None:
    identity, _ = verifier.build_smoke_event("resume_wrong_store")
    reports = FakeReportsClient(identity, visible=True)
    ingestion = FakeFeedbackClient(reports)
    if generic_response:
        original_post_event = ingestion.post_event

        def post_event(event: dict[str, Any]) -> verifier.feedback.UploadReceipt:
            receipt = original_post_event(event)
            return verifier.feedback.UploadReceipt(
                status_code=receipt.status_code,
                endpoint=receipt.endpoint,
                request_id=receipt.request_id,
                response=None,
            )

        ingestion.post_event = post_event  # type: ignore[method-assign]

    with pytest.raises(verifier.RoundtripUploadError, match=message):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="resume_wrong_store",
            resume=True,
            timeout_seconds=1,
            poll_interval=0.1,
        )


def test_roundtrip_refuses_existing_namespace_without_resume() -> None:
    identity, _ = verifier.build_smoke_event("collision")
    reports = FakeReportsClient(identity, visible=True)
    ingestion = FakeFeedbackClient(reports)

    with pytest.raises(verifier.RoundtripPreflightError, match="already contains"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="collision",
            timeout_seconds=1,
            poll_interval=0.1,
        )
    assert ingestion.events == []


def test_roundtrip_checks_request_id_report_path_before_permanent_write() -> None:
    identity, _ = verifier.build_smoke_event("stale_report_backend")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_fetch_page = reports.fetch_page

    def fetch_page(
        view: str,
        *,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> verifier.ReportPage:
        if view == verifier.INGESTION_VIEW:
            raise verifier.ReportsRequestError(
                "unsupported query",
                endpoint="https://reports.example/feedback-report",
                status_code=400,
            )
        return original_fetch_page(
            view,
            filters=filters,
            limit=limit,
            offset=offset,
        )

    reports.fetch_page = fetch_page  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripPreflightError, match="request-id preflight"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="stale_report_backend",
            timeout_seconds=1,
            poll_interval=0.1,
        )
    assert ingestion.events == []


def test_roundtrip_rejects_wrong_comment_identity() -> None:
    identity, _ = verifier.build_smoke_event("wrong")
    pages = _visible_pages(identity)
    pages[verifier.COMMENTS_VIEW].rows[0]["comment_text"] = "different"

    with pytest.raises(verifier.RoundtripReportError, match="comment_text"):
        verifier.assert_roundtrip_visible(pages, identity)


def test_roundtrip_rejects_non_uuid_or_mismatched_receipt_request_id() -> None:
    for header_id, body_id in (
        ("not-a-uuid", "not-a-uuid"),
        (INGEST_REQUEST_ID, "bdbb2f4b-87a0-44c9-83f1-fdc5c596c36d"),
    ):
        receipt = verifier.feedback.UploadReceipt(
            status_code=200,
            endpoint="https://ingest.example/feedback-ingest",
            request_id=header_id,
            response={
                "accepted": 1,
                "duplicate": 0,
                "conflict": 0,
                "rejected": 0,
                "request_id": body_id,
            },
        )
        with pytest.raises(verifier.RoundtripUploadError, match="matching UUID"):
            verifier._receipt_expectation(receipt, resume=False)


def test_roundtrip_rejects_non_200_receipt_even_with_valid_counters() -> None:
    receipt = verifier.feedback.UploadReceipt(
        status_code=201,
        endpoint="https://ingest.example/feedback-ingest",
        request_id=INGEST_REQUEST_ID,
        response={
            "accepted": 1,
            "duplicate": 0,
            "conflict": 0,
            "rejected": 0,
            "request_id": INGEST_REQUEST_ID,
        },
    )
    with pytest.raises(verifier.RoundtripUploadError, match="HTTP 200"):
        verifier._receipt_expectation(receipt, resume=False)


def test_conflict_expectation_requires_exact_structured_409_contract() -> None:
    assert (
        verifier._conflict_expectation(_structured_conflict_error())
        == CONFLICT_REQUEST_ID
    )


def test_mixed_conflict_expectation_requires_two_rejected_events() -> None:
    assert (
        verifier._conflict_expectation(
            _structured_conflict_error(
                header_request_id=MIXED_BATCH_REQUEST_ID,
                body_request_id=MIXED_BATCH_REQUEST_ID,
                rejected=2,
            ),
            rejected=2,
        )
        == MIXED_BATCH_REQUEST_ID
    )
    with pytest.raises(verifier.RoundtripUploadError, match="rejected=2"):
        verifier._conflict_expectation(
            _structured_conflict_error(
                header_request_id=MIXED_BATCH_REQUEST_ID,
                body_request_id=MIXED_BATCH_REQUEST_ID,
                rejected=1,
            ),
            rejected=2,
        )


@pytest.mark.parametrize(
    "error",
    [
        _structured_conflict_error(
            header_request_id=MIXED_BATCH_REQUEST_ID.upper(),
            body_request_id=MIXED_BATCH_REQUEST_ID.upper(),
            rejected=2,
        ),
        _structured_conflict_error(
            header_request_id=MIXED_BATCH_REQUEST_ID,
            body_request_id=CONFLICT_REQUEST_ID,
            rejected=2,
        ),
    ],
)
def test_mixed_conflict_expectation_rejects_noncanonical_or_mismatched_uuid(
    error: verifier.feedback.FeedbackUploadConflictError,
) -> None:
    with pytest.raises(verifier.RoundtripUploadError, match="canonical UUID"):
        verifier._conflict_expectation(error, rejected=2)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"accepted": 1}, "mixed-batch probe requires"),
        ({"duplicate": 1}, "mixed-batch probe requires"),
        ({"conflict": 0}, "mixed-batch probe requires"),
        ({"rejected": 1}, "rejected=2"),
        ({"duplicate": True}, "non-negative integer"),
    ],
)
def test_mixed_conflict_expectation_rejects_any_wrong_counter(
    counts: dict[str, Any],
    message: str,
) -> None:
    parameters: dict[str, Any] = {
        "header_request_id": MIXED_BATCH_REQUEST_ID,
        "body_request_id": MIXED_BATCH_REQUEST_ID,
        "rejected": 2,
    }
    parameters.update(counts)
    with pytest.raises(verifier.RoundtripUploadError, match=message):
        verifier._conflict_expectation(
            _structured_conflict_error(**parameters),
            rejected=2,
        )


@pytest.mark.parametrize(
    "error",
    [
        _structured_conflict_error(status_code=400),
        _structured_conflict_error(error_code="OTHER_CONFLICT"),
        _structured_conflict_error(header_request_id=None),
        _structured_conflict_error(
            header_request_id=CONFLICT_REQUEST_ID.upper(),
            body_request_id=CONFLICT_REQUEST_ID.upper(),
        ),
        _structured_conflict_error(body_request_id=INGEST_REQUEST_ID),
    ],
)
def test_conflict_expectation_rejects_status_code_or_request_identity(
    error: verifier.feedback.FeedbackUploadConflictError,
) -> None:
    with pytest.raises(verifier.RoundtripUploadError, match="canonical UUID"):
        verifier._conflict_expectation(error)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"accepted": 1}, "requires accepted=0"),
        ({"duplicate": 1}, "requires accepted=0"),
        ({"conflict": 0}, "requires accepted=0"),
        ({"rejected": 0}, "requires accepted=0"),
        ({"conflict": True}, "non-negative integer"),
    ],
)
def test_conflict_expectation_rejects_any_wrong_counter(
    counts: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(verifier.RoundtripUploadError, match=message):
        verifier._conflict_expectation(_structured_conflict_error(**counts))


def test_roundtrip_rejects_2xx_instead_of_structured_conflict() -> None:
    identity, _ = verifier.build_smoke_event("accepted_conflict")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_post_event = ingestion.post_event

    def post_event(event: dict[str, Any]) -> verifier.feedback.UploadReceipt:
        if ingestion.events:
            ingestion.events.append(event)
            return verifier.feedback.UploadReceipt(
                status_code=200,
                endpoint="https://ingest.example/feedback-ingest",
                request_id=CONFLICT_REQUEST_ID,
                response={
                    "accepted": 1,
                    "duplicate": 0,
                    "conflict": 0,
                    "rejected": 0,
                    "request_id": CONFLICT_REQUEST_ID,
                },
            )
        return original_post_event(event)

    ingestion.post_event = post_event  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="was not rejected"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="accepted_conflict",
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


@pytest.mark.parametrize(
    "reused_request_id",
    [
        INGEST_REQUEST_ID,
        SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
        SUCCESSFUL_BATCH_REPLAY_REQUEST_ID,
    ],
)
def test_roundtrip_rejects_prior_success_request_id_for_conflict(
    reused_request_id: str,
) -> None:
    identity, _ = verifier.build_smoke_event("reused_request_id")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_post_event = ingestion.post_event

    def post_event(event: dict[str, Any]) -> verifier.feedback.UploadReceipt:
        if ingestion.events:
            ingestion.events.append(event)
            raise _structured_conflict_error(
                header_request_id=reused_request_id,
                body_request_id=reused_request_id,
            )
        return original_post_event(event)

    ingestion.post_event = post_event  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="reused the successful"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="reused_request_id",
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


@pytest.mark.parametrize(
    "reused_request_id",
    [
        INGEST_REQUEST_ID,
        SUCCESSFUL_BATCH_FIRST_REQUEST_ID,
        SUCCESSFUL_BATCH_REPLAY_REQUEST_ID,
        CONFLICT_REQUEST_ID,
    ],
)
def test_roundtrip_rejects_mixed_request_id_reused_from_either_prior_post(
    reused_request_id: str,
) -> None:
    identity, _ = verifier.build_smoke_event("mixed_reused_request_id")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)
    original_post_trace = ingestion.post_trace

    def post_trace(
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        envelope = (
            trace.to_envelope()
            if isinstance(trace, verifier.feedback.SessionTrace)
            else trace
        )
        if all(".batch-" in event["event_id"] for event in envelope["events"]):
            return original_post_trace(envelope)
        raise _structured_conflict_error(
            header_request_id=reused_request_id,
            body_request_id=reused_request_id,
            rejected=2,
        )

    ingestion.post_trace = post_trace  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="reused an earlier"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="mixed_reused_request_id",
            include_mixed_batch_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


def test_roundtrip_rejects_2xx_instead_of_structured_mixed_batch_conflict() -> None:
    identity, _ = verifier.build_smoke_event("mixed_accepted")
    reports = FakeReportsClient(identity)
    ingestion = FakeFeedbackClient(reports)

    def post_trace(
        trace: verifier.feedback.SessionTrace | dict[str, Any],
    ) -> verifier.feedback.UploadReceipt:
        del trace
        return verifier.feedback.UploadReceipt(
            status_code=200,
            endpoint="https://ingest.example/feedback-ingest",
            request_id=MIXED_BATCH_REQUEST_ID,
            response={
                "accepted": 1,
                "duplicate": 0,
                "conflict": 1,
                "rejected": 1,
                "request_id": MIXED_BATCH_REQUEST_ID,
            },
        )

    ingestion.post_trace = post_trace  # type: ignore[method-assign]
    with pytest.raises(verifier.RoundtripUploadError, match="was not rejected"):
        verifier.run_roundtrip(
            ingestion,
            reports,
            run_id="mixed_accepted",
            skip_successful_batch_probe=True,
            include_mixed_batch_probe=True,
            timeout_seconds=1,
            poll_interval=0.1,
            monotonic=lambda: 0.0,
            sleep=lambda seconds: None,
        )


def test_ingestion_outcome_must_match_exact_receipt_counts() -> None:
    page = _ingestion_page(accepted=1, duplicate=0)
    verifier.assert_ingestion_outcome(page, accepted=1, duplicate=0)

    page.rows[0]["duplicate_event_rate"] = 1
    with pytest.raises(verifier.RoundtripReportError, match="duplicate_event_rate"):
        verifier.assert_ingestion_outcome(page, accepted=1, duplicate=0)

    page = _ingestion_page(accepted=1, duplicate=0)
    page.rows[0]["classified_event_count"] = 0
    with pytest.raises(verifier.RoundtripReportError, match="classified_event_count"):
        verifier.assert_ingestion_outcome(page, accepted=1, duplicate=0)

    empty = _empty_ingestion_page()
    verifier.assert_ingestion_outcome_absent(empty)
    empty.rows[0]["recorded_request_count"] = 1
    with pytest.raises(verifier.RoundtripReportError, match="negative control"):
        verifier.assert_ingestion_outcome_absent(empty)

    accepted_batch = _ingestion_page(accepted=2, duplicate=0, event_count=2)
    verifier.assert_ingestion_outcome(
        accepted_batch,
        accepted=2,
        duplicate=0,
        event_count=2,
    )
    duplicate_batch = _ingestion_page(accepted=0, duplicate=2, event_count=2)
    verifier.assert_ingestion_outcome(
        duplicate_batch,
        accepted=0,
        duplicate=2,
        event_count=2,
    )
    duplicate_batch.rows[0]["classified_event_count"] = 1
    with pytest.raises(verifier.RoundtripReportError, match="classified_event_count"):
        verifier.assert_ingestion_outcome(
            duplicate_batch,
            accepted=0,
            duplicate=2,
            event_count=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success_request_count", 1),
        ("client_rejection_count", 0),
        ("event_id_conflict_request_count", 0),
        ("conflicting_event_count", 0),
        ("conflict_audit_event_count", 0),
        ("event_id_reuse_count", 0),
        ("classified_event_count", 0),
        ("request_failure_rate", 0),
        ("duplicate_event_rate", 0),
        ("event_id_reuse_rate", 0),
        ("classified_conflicting_event_rate", 0),
    ],
)
def test_conflict_outcome_requires_unique_rejection_audit_and_exact_rates(
    field: str,
    value: Any,
) -> None:
    page = _conflict_ingestion_page()
    page.rows[0][field] = value
    with pytest.raises(verifier.RoundtripReportError, match=field):
        verifier.assert_conflict_ingestion_outcome(page)


def test_conflict_outcome_accepts_exact_single_rejection_and_audit() -> None:
    verifier.assert_conflict_ingestion_outcome(_conflict_ingestion_page())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recorded_request_count", 0),
        ("success_request_count", 1),
        ("client_rejection_count", 0),
        ("service_failure_count", 1),
        ("event_id_conflict_request_count", 0),
        ("accepted_event_count", 1),
        ("duplicate_event_count", 1),
        ("idempotent_duplicate_event_count", 1),
        ("unclassified_duplicate_event_count", 1),
        ("conflicting_event_count", 0),
        ("conflict_audit_event_count", 0),
        ("event_id_reuse_count", 0),
        ("classified_event_count", 1),
        ("known_event_result_count", 1),
        ("request_failure_rate", 0),
        ("duplicate_event_rate", 0),
        ("event_id_reuse_rate", 1),
        ("classified_conflicting_event_rate", 0.5),
        ("recorded_rate_available", False),
        ("end_to_end_coverage_available", True),
    ],
)
def test_mixed_outcome_requires_exact_all_or_none_counts_and_rates(
    field: str,
    value: Any,
) -> None:
    page = _mixed_batch_ingestion_page()
    page.rows[0][field] = value
    with pytest.raises(verifier.RoundtripReportError, match=field):
        verifier.assert_mixed_batch_ingestion_outcome(page)


def test_mixed_outcome_accepts_exact_two_event_rejection_and_one_audit() -> None:
    verifier.assert_mixed_batch_ingestion_outcome(_mixed_batch_ingestion_page())


def test_mixed_outcome_requires_exact_22_columns_and_timestamps() -> None:
    missing = _mixed_batch_ingestion_page()
    missing.rows[0].pop("first_started_at")
    with pytest.raises(verifier.RoundtripReportError, match="22-column"):
        verifier.assert_mixed_batch_ingestion_outcome(missing)

    extra = _mixed_batch_ingestion_page()
    extra.rows[0]["unexpected"] = 1
    with pytest.raises(verifier.RoundtripReportError, match="22-column"):
        verifier.assert_mixed_batch_ingestion_outcome(extra)

    invalid_timestamp = _mixed_batch_ingestion_page()
    invalid_timestamp.rows[0]["last_finished_at"] = None
    with pytest.raises(verifier.RoundtripReportError, match="timestamps"):
        verifier.assert_mixed_batch_ingestion_outcome(invalid_timestamp)


def test_cli_requires_explicit_permanent_write_without_leaking_tokens(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    monkeypatch.setattr(
        verifier,
        "load_authoritative_probe_question",
        lambda path: pytest.fail(
            f"bundle should not be loaded before confirmation: {path}"
        ),
    )

    assert verifier.main(["--run-id", "dry_check"]) == 2
    captured = capsys.readouterr()
    assert "--confirm-permanent-write" in captured.err
    assert "ingest-secret" not in captured.err
    assert "report-secret" not in captured.err


def test_cli_defaults_to_the_fully_attested_demo_bundle() -> None:
    args = verifier._parser().parse_args([])
    assert args.bundle == str(verifier.DEFAULT_BUNDLE_PATH)


def test_cli_announces_generated_run_id_before_roundtrip_and_keeps_json_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    probe = _probe_question()
    loaded_paths: list[str] = []

    def load_probe(path: str) -> verifier.AuthoritativeProbeQuestion:
        loaded_paths.append(path)
        return probe

    monkeypatch.setattr(verifier, "load_authoritative_probe_question", load_probe)
    monkeypatch.setattr(verifier.secrets, "token_hex", lambda length: "generated_run")

    def fail_after_write(*args: Any, **kwargs: Any) -> verifier.RoundtripResult:
        del args
        assert kwargs["probe_question"] is probe
        captured_before_call = capsys.readouterr()
        assert captured_before_call.out == ""
        assert "generated_run" in captured_before_call.err
        assert probe.registry_id in captured_before_call.err
        assert "--resume" in captured_before_call.err
        assert "private conflict-audit" in captured_before_call.err
        raise verifier.RoundtripReportError("polling interrupted")

    monkeypatch.setattr(verifier, "run_roundtrip", fail_after_write)

    assert (
        verifier.main(
            [
                "--confirm-permanent-write",
                "--bundle",
                "/attested/custom-bundle",
                "--json",
                "--timeout",
                "0.1",
            ]
        )
        == 4
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "polling interrupted" in captured.err
    assert loaded_paths == ["/attested/custom-bundle"]


def test_cli_refuses_legacy_result_as_authoritative_ledger_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    probe = _probe_question()
    monkeypatch.setattr(
        verifier,
        "load_authoritative_probe_question",
        lambda path: probe,
    )
    identity, _ = verifier.build_smoke_event(
        "cli_legacy_result",
        probe_question=probe,
    )
    monkeypatch.setattr(
        verifier,
        "run_roundtrip",
        lambda *args, **kwargs: verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=1,
            receipt=None,
            verified_at=VERIFIED_AT,
        ),
    )

    assert (
        verifier.main(
            [
                "--confirm-permanent-write",
                "--run-id",
                "cli_legacy_result",
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "authoritative result does not match" in captured.err
    assert "ingest-secret" not in captured.err
    assert "report-secret" not in captured.err


@pytest.mark.parametrize("skip_conflict_probe", [False, True])
def test_cli_defaults_to_full_conflict_probe_and_only_explicit_flag_skips_it(
    skip_conflict_probe: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    probe = _probe_question()
    monkeypatch.setattr(
        verifier,
        "load_authoritative_probe_question",
        lambda path: probe,
    )
    identity, _ = verifier.build_smoke_event(
        "cli_probe",
        probe_question=probe,
    )
    called: dict[str, Any] = {}

    def fake_roundtrip(*args: Any, **kwargs: Any) -> verifier.RoundtripResult:
        del args
        called.update(kwargs)
        return verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=1 if skip_conflict_probe else 2,
            receipt={
                "accepted": 1,
                "duplicate": 0,
                "conflict": 0,
                "rejected": 0,
                "request_id": INGEST_REQUEST_ID,
            },
            conflict_request_id=(None if skip_conflict_probe else CONFLICT_REQUEST_ID),
            conflict_verified=not skip_conflict_probe,
            successful_batch_first_request_id=(SUCCESSFUL_BATCH_FIRST_REQUEST_ID),
            successful_batch_replay_request_id=(SUCCESSFUL_BATCH_REPLAY_REQUEST_ID),
            successful_batch_verified=True,
            successful_batch_first_write_verified=True,
            registry_id=probe.registry_id,
            authority_status_verified=True,
            business_snapshot_verified=True,
            session_attempt_filters_verified=True,
            verified_at=VERIFIED_AT,
            manifest_sha256=probe.manifest_sha256,
            registry_question_count=probe.registry_question_count,
            registry_choice_count=probe.registry_choice_count,
            authority_mode=verifier.AUTHORITY_MODE_AUTHORITATIVE,
        )

    monkeypatch.setattr(verifier, "run_roundtrip", fake_roundtrip)
    argv = ["--confirm-permanent-write", "--run-id", "cli_probe", "--json"]
    if skip_conflict_probe:
        argv.append("--skip-conflict-probe")

    assert verifier.main(argv) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {
        "schema_version",
        "evidence_type",
        "verified_at",
        "manifest_sha256",
        "registry_question_count",
        "registry_choice_count",
        "authority_mode",
        "ok",
        "run_id",
        "release_id",
        "question_id",
        "event_id",
        "request_id",
        "conflict_request_id",
        "conflict_verified",
        "mixed_batch_request_id",
        "mixed_batch_verified",
        "successful_batch_first_request_id",
        "successful_batch_replay_request_id",
        "successful_batch_verified",
        "successful_batch_first_write_verified",
        "registry_id",
        "authority_status_verified",
        "detail_reports_verified",
        "business_snapshot_verified",
        "session_attempt_filters_verified",
        "polls",
        "receipt",
    }
    assert payload["schema_version"] == verifier.EVIDENCE_SCHEMA_VERSION
    assert payload["evidence_type"] == verifier.EVIDENCE_TYPE
    assert payload["verified_at"] == VERIFIED_AT
    assert payload["manifest_sha256"] == probe.manifest_sha256
    assert payload["registry_question_count"] == probe.registry_question_count
    assert payload["registry_choice_count"] == probe.registry_choice_count
    assert payload["authority_mode"] == verifier.AUTHORITY_MODE_AUTHORITATIVE
    assert called["probe_question"] is probe
    assert called["skip_successful_batch_probe"] is False
    assert called["skip_conflict_probe"] is skip_conflict_probe
    assert called["include_mixed_batch_probe"] is False
    assert payload["conflict_verified"] is not skip_conflict_probe
    assert payload["conflict_request_id"] == (
        None if skip_conflict_probe else CONFLICT_REQUEST_ID
    )
    assert payload["mixed_batch_request_id"] is None
    assert payload["mixed_batch_verified"] is False
    assert payload["successful_batch_verified"] is True
    assert payload["successful_batch_first_write_verified"] is True
    assert payload["registry_id"] == probe.registry_id
    assert payload["authority_status_verified"] is True
    assert payload["business_snapshot_verified"] is True
    assert payload["session_attempt_filters_verified"] is True
    assert "ingest-secret" not in captured.out
    assert "report-secret" not in captured.out
    assert "ingest-secret" not in captured.err
    assert "report-secret" not in captured.err
    assert "answer/proposed-setting/comment trace" in captured.err
    assert "both detail rows" in captured.err
    if skip_conflict_probe:
        assert "explicitly skipped" in captured.err
    else:
        assert "409 outcome is permanent" in captured.err
        assert "private conflict-audit record" in captured.err


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_opt_in_forwards_mixed_probe_and_reports_permanent_result(
    json_output: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    probe = _probe_question()
    monkeypatch.setattr(
        verifier,
        "load_authoritative_probe_question",
        lambda path: probe,
    )
    identity, _ = verifier.build_smoke_event(
        "cli_mixed",
        probe_question=probe,
    )
    called: dict[str, Any] = {}

    def fake_roundtrip(*args: Any, **kwargs: Any) -> verifier.RoundtripResult:
        del args
        called.update(kwargs)
        return verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=3,
            receipt={
                "accepted": 1,
                "duplicate": 0,
                "conflict": 0,
                "rejected": 0,
                "request_id": INGEST_REQUEST_ID,
            },
            conflict_request_id=CONFLICT_REQUEST_ID,
            conflict_verified=True,
            mixed_batch_request_id=MIXED_BATCH_REQUEST_ID,
            mixed_batch_verified=True,
            successful_batch_first_request_id=(SUCCESSFUL_BATCH_FIRST_REQUEST_ID),
            successful_batch_replay_request_id=(SUCCESSFUL_BATCH_REPLAY_REQUEST_ID),
            successful_batch_verified=True,
            successful_batch_first_write_verified=True,
            registry_id=probe.registry_id,
            authority_status_verified=True,
            business_snapshot_verified=True,
            session_attempt_filters_verified=True,
            verified_at=VERIFIED_AT,
            manifest_sha256=probe.manifest_sha256,
            registry_question_count=probe.registry_question_count,
            registry_choice_count=probe.registry_choice_count,
            authority_mode=verifier.AUTHORITY_MODE_AUTHORITATIVE,
        )

    monkeypatch.setattr(verifier, "run_roundtrip", fake_roundtrip)
    argv = [
        "--confirm-permanent-write",
        "--run-id",
        "cli_mixed",
        "--include-mixed-batch-probe",
    ]
    if json_output:
        argv.append("--json")

    assert verifier.main(argv) == 0
    captured = capsys.readouterr()
    assert called["probe_question"] is probe
    assert called["skip_successful_batch_probe"] is False
    assert called["skip_conflict_probe"] is False
    assert called["include_mixed_batch_probe"] is True
    assert "mixed-batch probe permanently records another 409" in captured.err
    assert "fresh second event must be withheld" in captured.err
    if json_output:
        payload = json.loads(captured.out)
        assert payload["registry_id"] == probe.registry_id
        assert payload["authority_status_verified"] is True
        assert payload["business_snapshot_verified"] is True
        assert payload["session_attempt_filters_verified"] is True
        assert payload["mixed_batch_request_id"] == MIXED_BATCH_REQUEST_ID
        assert payload["mixed_batch_verified"] is True
    else:
        assert f"Registry id: {probe.registry_id}" in captured.out
        assert "Authority status verified: True" in captured.out
        assert "Business snapshot verified: True" in captured.out
        assert "Session/attempt filters verified: True" in captured.out
        assert (
            f"Successful-batch first request id: "
            f"{SUCCESSFUL_BATCH_FIRST_REQUEST_ID}" in captured.out
        )
        assert "Successful-batch probe verified: True" in captured.out
        assert "Successful-batch first write verified: True" in captured.out
        assert f"Mixed-batch request id: {MIXED_BATCH_REQUEST_ID}" in captured.out
        assert "Mixed-batch probe verified: True" in captured.out


def test_cli_explicit_successful_batch_skip_is_forwarded_and_not_claimed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    feedback_config = verifier.feedback.FeedbackConfig(
        endpoint="https://ingest.example/feedback-ingest",
        bearer_token="ingest-secret",
    )
    reports_config = verifier.ReportsConfig(
        url="https://reports.example/feedback-report",
        read_token="report-secret",
    )
    monkeypatch.setattr(
        verifier,
        "_require_safe_configs",
        lambda: (feedback_config, reports_config),
    )
    probe = _probe_question()
    monkeypatch.setattr(
        verifier,
        "load_authoritative_probe_question",
        lambda path: probe,
    )
    identity, _ = verifier.build_smoke_event(
        "cli_skip_batch",
        probe_question=probe,
    )
    called: dict[str, Any] = {}

    def fake_roundtrip(*args: Any, **kwargs: Any) -> verifier.RoundtripResult:
        del args
        called.update(kwargs)
        return verifier.RoundtripResult(
            identity=identity,
            request_id=INGEST_REQUEST_ID,
            polls=1,
            receipt=None,
            registry_id=probe.registry_id,
            authority_status_verified=True,
            business_snapshot_verified=True,
            verified_at=VERIFIED_AT,
            manifest_sha256=probe.manifest_sha256,
            registry_question_count=probe.registry_question_count,
            registry_choice_count=probe.registry_choice_count,
            authority_mode=verifier.AUTHORITY_MODE_AUTHORITATIVE,
        )

    monkeypatch.setattr(verifier, "run_roundtrip", fake_roundtrip)
    assert (
        verifier.main(
            [
                "--confirm-permanent-write",
                "--run-id",
                "cli_skip_batch",
                "--skip-successful-batch-probe",
                "--skip-conflict-probe",
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert called["probe_question"] is probe
    assert payload["authority_status_verified"] is True
    assert payload["business_snapshot_verified"] is True
    assert called["skip_successful_batch_probe"] is True
    assert payload["successful_batch_verified"] is False
    assert payload["successful_batch_first_write_verified"] is False
    assert "explicitly skipped" in captured.err


def test_cli_mixed_probe_and_skip_conflict_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exc_info:
        verifier._parser().parse_args(
            ["--skip-conflict-probe", "--include-mixed-batch-probe"]
        )
    assert exc_info.value.code == 2
