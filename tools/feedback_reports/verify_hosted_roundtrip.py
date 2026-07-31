#!/usr/bin/env python3
"""Verify hosted ingestion and registry-derived feedback authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import re
import secrets
import sys
import time
import urllib.parse
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from feedback_reports import (  # noqa: E402
    BUSINESS_REPORT_VIEWS,
    BusinessSnapshot,
    ReportPage,
    ReportsClient,
    ReportsConfig,
    ReportsError,
    ReportsRequestError,
)
from question_inspector import feedback  # noqa: E402
from question_inspector.release_manifest import (  # noqa: E402
    ReleaseManifestError,
    load_quiz_manifest,
)
from quiz_bundle import (  # noqa: E402
    FeedbackRegistryError,
    build_feedback_registry,
)


SUMMARY_VIEW = "feedback_report_summary"
SESSIONS_VIEW = "feedback_report_sessions"
QUESTIONS_VIEW = "feedback_report_questions"
COMMENTS_VIEW = "feedback_report_comments"
REPORT_VIEWS = (SUMMARY_VIEW, SESSIONS_VIEW, QUESTIONS_VIEW, COMMENTS_VIEW)
INGESTION_VIEW = "feedback_report_ingestion_summary"
EVENT_RESOLUTION_VIEW = "feedback_report_event_resolution"
AUTHORITY_STATUS_VIEW = "feedback_report_authority_status"
BUSINESS_SNAPSHOT_VIEW = "feedback_report_business_snapshot"
ANSWER_DETAILS_VIEW = "feedback_report_answers"
PROPOSAL_DETAILS_VIEW = "feedback_report_proposals"
DETAIL_REPORT_VIEWS = (ANSWER_DETAILS_VIEW, PROPOSAL_DETAILS_VIEW)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = REPO_ROOT / "examples" / "quiz_demo" / "bundle"
EVIDENCE_SCHEMA_VERSION = "1.0"
EVIDENCE_TYPE = "architecture_iq_hosted_feedback_roundtrip"
AUTHORITY_MODE_AUTHORITATIVE = "authoritative"
AUTHORITY_MODE_LEGACY = "legacy"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UTC_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
FAMILY = "architectureiq_e2e"
QUESTION_TYPE = "e2e_smoke"
COMMENT_CATEGORY = "other"
CONFLICT_COMMENT_SUFFIX = " [intentional event-ID conflict probe]"
MIXED_BATCH_COMMENT_PREFIX = "ArchitectureIQ hosted feedback mixed-batch withheld"
SUCCESSFUL_BATCH_COMMENT_PREFIX = (
    "ArchitectureIQ hosted feedback successful multi-event batch"
)
DETAIL_PROBE_LABEL_PREFIX = "ArchitectureIQ hosted detail probe"
DETAIL_PROBE_N_SEEDS = 3
DETAIL_PROBE_BASE_SEED = 1701
DETAIL_PAGE_LIMIT = 1_000
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
WRONG_CLIENT_FAMILY = "architectureiq_e2e_wrong_family"
WRONG_CLIENT_QUESTION_TYPE = "architectureiq_e2e_wrong_question_type"


class RoundtripError(RuntimeError):
    """Base error for a safe hosted roundtrip check."""


class RoundtripConfigurationError(RoundtripError):
    """Configuration or explicit-write confirmation is invalid."""


class RoundtripPreflightError(RoundtripError):
    """The deterministic smoke namespace is not safe to write."""


class RoundtripUploadError(RoundtripError):
    """The ingestion endpoint did not acknowledge the synthetic event."""


class RoundtripReportError(RoundtripError):
    """The event did not become consistently visible in all reports."""


@dataclass(frozen=True)
class SmokeIdentity:
    """Deterministic identifiers for one append-only smoke event."""

    run_id: str
    session_id: str
    attempt_id: str
    event_id: str
    question_id: str
    release_id: str
    dataset_id: str
    question_version: str
    comment_text: str


@dataclass(frozen=True)
class SuccessfulBatchIdentity:
    """Deterministic identity for the independent successful-batch probe."""

    run_id: str
    session_id: str
    attempt_id: str
    answer_event_id: str
    comment_event_id: str
    proposal_event_id: str | None
    question_id: str
    release_id: str
    dataset_id: str
    question_version: str
    comment_text: str


@dataclass(frozen=True)
class AuthoritativeProbeQuestion:
    """One deterministic, fully attested feedback-registry membership."""

    registry_id: str
    release_id: str
    question_id: str
    question_version: str
    family: str
    dataset_id: str
    question_type: str
    selected_letter: str
    selected_candidate_id: str
    authoritative_is_correct: bool
    registry_question_count: int
    registry_choice_count: int
    manifest_sha256: str
    question: Mapping[str, Any]


@dataclass(frozen=True)
class RoundtripResult:
    """Stable evidence metadata; never includes endpoint credentials.

    ``authoritative`` results are complete ledger evidence bound to an attested
    manifest and registry.  ``legacy`` results remain useful to the internal
    synthetic harness, but their nullable authority fields prevent them from
    masquerading as hosted registry-authority evidence.
    """

    identity: SmokeIdentity
    request_id: str | None
    polls: int
    receipt: Mapping[str, Any] | None
    conflict_request_id: str | None = None
    conflict_verified: bool = False
    mixed_batch_request_id: str | None = None
    mixed_batch_verified: bool = False
    successful_batch_first_request_id: str | None = None
    successful_batch_replay_request_id: str | None = None
    successful_batch_verified: bool = False
    successful_batch_first_write_verified: bool = False
    registry_id: str | None = None
    authority_status_verified: bool = False
    detail_reports_verified: bool = False
    business_snapshot_verified: bool = False
    session_attempt_filters_verified: bool = False
    verified_at: str | None = None
    manifest_sha256: str | None = None
    registry_question_count: int | None = None
    registry_choice_count: int | None = None
    authority_mode: str = AUTHORITY_MODE_LEGACY

    def __post_init__(self) -> None:
        if not isinstance(self.authority_mode, str) or self.authority_mode not in {
            AUTHORITY_MODE_AUTHORITATIVE,
            AUTHORITY_MODE_LEGACY,
        }:
            raise RoundtripConfigurationError("unsupported evidence authority mode")
        if self.verified_at is not None and not _is_utc_rfc3339(self.verified_at):
            raise RoundtripConfigurationError(
                "verified_at must be a UTC RFC3339 timestamp"
            )
        if self.manifest_sha256 is not None and (
            not isinstance(self.manifest_sha256, str)
            or SHA256_PATTERN.fullmatch(self.manifest_sha256) is None
        ):
            raise RoundtripConfigurationError(
                "manifest_sha256 must be 64 lowercase hexadecimal digits"
            )
        for name, value in (
            ("registry_question_count", self.registry_question_count),
            ("registry_choice_count", self.registry_choice_count),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise RoundtripConfigurationError(f"{name} must be a positive integer")
        if self.authority_mode == AUTHORITY_MODE_AUTHORITATIVE:
            missing = [
                name
                for name, value in (
                    ("verified_at", self.verified_at),
                    ("manifest_sha256", self.manifest_sha256),
                    ("registry_id", self.registry_id),
                    ("registry_question_count", self.registry_question_count),
                    ("registry_choice_count", self.registry_choice_count),
                )
                if value is None
            ]
            if missing:
                raise RoundtripConfigurationError(
                    "authoritative evidence is missing: " + ", ".join(missing)
                )
            if not isinstance(self.registry_id, str) or not self.registry_id:
                raise RoundtripConfigurationError(
                    "authoritative evidence registry_id must be a non-empty string"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence_type": EVIDENCE_TYPE,
            "verified_at": self.verified_at,
            "manifest_sha256": self.manifest_sha256,
            "registry_question_count": self.registry_question_count,
            "registry_choice_count": self.registry_choice_count,
            "authority_mode": self.authority_mode,
            "ok": True,
            "run_id": self.identity.run_id,
            "release_id": self.identity.release_id,
            "question_id": self.identity.question_id,
            "event_id": self.identity.event_id,
            "request_id": self.request_id,
            "conflict_request_id": self.conflict_request_id,
            "conflict_verified": self.conflict_verified,
            "mixed_batch_request_id": self.mixed_batch_request_id,
            "mixed_batch_verified": self.mixed_batch_verified,
            "successful_batch_first_request_id": (
                self.successful_batch_first_request_id
            ),
            "successful_batch_replay_request_id": (
                self.successful_batch_replay_request_id
            ),
            "successful_batch_verified": self.successful_batch_verified,
            "successful_batch_first_write_verified": (
                self.successful_batch_first_write_verified
            ),
            "registry_id": self.registry_id,
            "authority_status_verified": self.authority_status_verified,
            "detail_reports_verified": self.detail_reports_verified,
            "business_snapshot_verified": self.business_snapshot_verified,
            "session_attempt_filters_verified": (self.session_attempt_filters_verified),
            "polls": self.polls,
            "receipt": dict(self.receipt) if self.receipt is not None else None,
        }


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RoundtripConfigurationError(
            "run id must contain 1-48 ASCII letters, digits, underscores, or hyphens"
        )
    return run_id


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _is_utc_rfc3339(value: object) -> bool:
    """Return whether ``value`` is a real UTC RFC3339 timestamp."""
    if not isinstance(value, str) or UTC_RFC3339_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _validate_authoritative_probe_metadata(
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Reject incomplete authority metadata before any permanent probe write."""
    if (
        not isinstance(probe_question.manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(probe_question.manifest_sha256) is None
    ):
        raise RoundtripConfigurationError(
            "authoritative probe manifest SHA-256 is invalid"
        )
    for name, value in (
        ("registry_question_count", probe_question.registry_question_count),
        ("registry_choice_count", probe_question.registry_choice_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RoundtripConfigurationError(
                f"authoritative probe {name} must be a positive integer"
            )
    if (
        not isinstance(probe_question.registry_id, str)
        or not probe_question.registry_id
    ):
        raise RoundtripConfigurationError(
            "authoritative probe registry_id must be non-empty"
        )


def _validate_authoritative_result(
    result: RoundtripResult,
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require a result to remain bound to its attested probe metadata."""
    expected = {
        "authority_mode": AUTHORITY_MODE_AUTHORITATIVE,
        "release_id": probe_question.release_id,
        "question_id": probe_question.question_id,
        "registry_id": probe_question.registry_id,
        "manifest_sha256": probe_question.manifest_sha256,
        "registry_question_count": probe_question.registry_question_count,
        "registry_choice_count": probe_question.registry_choice_count,
    }
    observed = {
        "authority_mode": result.authority_mode,
        "release_id": result.identity.release_id,
        "question_id": result.identity.question_id,
        "registry_id": result.registry_id,
        "manifest_sha256": result.manifest_sha256,
        "registry_question_count": result.registry_question_count,
        "registry_choice_count": result.registry_choice_count,
    }
    mismatched = [name for name, value in expected.items() if observed[name] != value]
    if mismatched:
        raise RoundtripConfigurationError(
            "authoritative result does not match its attested probe: "
            + ", ".join(mismatched)
        )


def load_authoritative_probe_question(
    bundle_path: Path | str,
) -> AuthoritativeProbeQuestion:
    """Select the first canonical registry row from a fully attested bundle."""
    root = Path(bundle_path).expanduser().resolve()
    try:
        registry = build_feedback_registry(root)
        manifest = load_quiz_manifest(root)
    except (FeedbackRegistryError, ReleaseManifestError, OSError, ValueError) as exc:
        raise RoundtripConfigurationError(
            f"authoritative probe bundle attestation failed for {root}: {exc}"
        ) from exc
    if manifest is None:
        raise RoundtripConfigurationError(
            f"authoritative probe bundle has no quiz_manifest.json: {root}"
        )
    if manifest.release_id != registry["release_id"]:
        raise RoundtripConfigurationError(
            "runtime manifest and feedback registry disagree on release_id"
        )
    raw_questions = registry.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise RoundtripConfigurationError(
            "feedback registry contains no authoritative question membership"
        )
    membership = raw_questions[0]
    if not isinstance(membership, Mapping):
        raise RoundtripConfigurationError(
            "feedback registry selected question is not an object"
        )
    manifest_question = next(
        (
            question
            for question in manifest.questions
            if question.question_id == membership.get("question_id")
            and question.version == membership.get("question_version")
        ),
        None,
    )
    if manifest_question is None:
        raise RoundtripConfigurationError(
            "selected registry membership is absent from the runtime manifest"
        )

    question_path = (
        root / PurePosixPath(manifest_question.path) / "question.json"
    ).resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key!r}")
            value[key] = item
        return value

    try:
        question = json.loads(
            question_path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RoundtripConfigurationError(
            f"cannot load selected runtime question {question_path}: {exc}"
        ) from exc
    if not isinstance(question, dict):
        raise RoundtripConfigurationError(
            "selected runtime question must be a JSON object"
        )
    try:
        observed_version = feedback.compute_question_version(question)
    except feedback.FeedbackValidationError as exc:
        raise RoundtripConfigurationError(
            f"selected runtime question is not interoperable JSON: {exc}"
        ) from exc
    expected_fields = {
        "question_id": membership.get("question_id"),
        "question_version": membership.get("question_version"),
        "family": membership.get("family"),
        "dataset_id": membership.get("dataset_id"),
        "question_type": membership.get("question_type"),
    }
    observed_fields = {
        "question_id": question.get("question_id"),
        "question_version": observed_version,
        "family": question.get("family"),
        "dataset_id": question.get("dataset_id"),
        "question_type": question.get("type"),
    }
    if observed_fields != expected_fields:
        raise RoundtripConfigurationError(
            "selected runtime question does not match its registry membership"
        )
    choices = membership.get("choices")
    if not isinstance(choices, Mapping) or not choices:
        raise RoundtripConfigurationError("selected registry membership has no choices")
    selected_letter = sorted(choices)[0]
    selected_candidate_id = choices[selected_letter]
    if not isinstance(selected_letter, str) or not isinstance(
        selected_candidate_id, str
    ):
        raise RoundtripConfigurationError(
            "selected registry choice identity is invalid"
        )
    correct_letter = membership.get("correct_letter")
    if not isinstance(correct_letter, str):
        raise RoundtripConfigurationError(
            "selected registry membership has no correct letter"
        )
    return AuthoritativeProbeQuestion(
        registry_id=str(registry["registry_id"]),
        release_id=str(registry["release_id"]),
        question_id=str(membership["question_id"]),
        question_version=str(membership["question_version"]),
        family=str(membership["family"]),
        dataset_id=str(membership["dataset_id"]),
        question_type=str(membership["question_type"]),
        selected_letter=selected_letter,
        selected_candidate_id=selected_candidate_id,
        authoritative_is_correct=selected_letter == correct_letter,
        registry_question_count=int(registry["question_count"]),
        registry_choice_count=int(registry["choice_count"]),
        manifest_sha256=manifest.manifest_sha256,
        question=deepcopy(question),
    )


def _wrong_client_value(authoritative: str, preferred: str) -> str:
    return preferred if preferred != authoritative else f"{preferred}_other"


def _feedback_context(
    *,
    attempt_id: str,
    release_id: str,
    family: str,
    dataset_id: str,
    question_type: str,
    authoritative: bool,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "release_id": release_id,
        "family": (
            _wrong_client_value(family, WRONG_CLIENT_FAMILY)
            if authoritative
            else family
        ),
        "dataset_id": dataset_id,
        "question_type": (
            _wrong_client_value(question_type, WRONG_CLIENT_QUESTION_TYPE)
            if authoritative
            else question_type
        ),
    }


def build_smoke_event(
    run_id: str,
    *,
    probe_question: AuthoritativeProbeQuestion | None = None,
) -> tuple[SmokeIdentity, dict[str, Any]]:
    """Build one deterministic comment through the production event schema."""
    resolved = _validate_run_id(run_id)
    session_id = f"anon_e2e_{resolved}"
    attempt_id = f"attempt_e2e_{resolved}"
    event_id = f"evt_e2e_{resolved}"
    question_id = (
        probe_question.question_id
        if probe_question is not None
        else f"q_e2e_{resolved}"
    )
    release_id = (
        probe_question.release_id
        if probe_question is not None
        else f"release_e2e_{resolved}"
    )
    dataset_id = (
        probe_question.dataset_id
        if probe_question is not None
        else f"dataset_e2e_{resolved}"
    )
    comment_text = f"ArchitectureIQ hosted feedback roundtrip {resolved}"
    occurred_at = _utc_now()
    question = (
        deepcopy(dict(probe_question.question))
        if probe_question is not None
        else {
            "question_id": question_id,
            "kind": "hosted_feedback_roundtrip",
            "run_id": resolved,
        }
    )
    family = probe_question.family if probe_question is not None else FAMILY
    question_type = (
        probe_question.question_type if probe_question is not None else QUESTION_TYPE
    )
    trace = feedback.SessionTrace(
        session_id=session_id,
        created_at=occurred_at,
    )
    event = trace.record_comment(
        question,
        category=COMMENT_CATEGORY,
        text=comment_text,
        event_id=event_id,
        occurred_at=occurred_at,
        extra=_feedback_context(
            attempt_id=attempt_id,
            release_id=release_id,
            family=family,
            dataset_id=dataset_id,
            question_type=question_type,
            authoritative=probe_question is not None,
        ),
    )
    identity = SmokeIdentity(
        run_id=resolved,
        session_id=session_id,
        attempt_id=attempt_id,
        event_id=event_id,
        question_id=question_id,
        release_id=release_id,
        dataset_id=dataset_id,
        question_version=event["question_version"],
        comment_text=comment_text,
    )
    if (
        probe_question is not None
        and identity.question_version != probe_question.question_version
    ):
        raise RoundtripConfigurationError(
            "authoritative smoke event changed the attested question version"
        )
    return identity, event


def build_successful_batch_trace(
    identity: SmokeIdentity,
    *,
    probe_question: AuthoritativeProbeQuestion | None = None,
) -> tuple[SuccessfulBatchIdentity, dict[str, Any]]:
    """Build a deterministic successful trace in the smoke session.

    Authoritative mode uses the same real registry membership for an answer,
    proposed setting, and comment.  The legacy harness keeps its historical
    two-event answer/comment trace in a disjoint namespace.  Every event is
    built through :class:`SessionTrace`, and callers replay the returned
    envelope unchanged to exercise the production trace client.
    """
    occurred_at = _utc_now()
    question_id = (
        probe_question.question_id
        if probe_question is not None
        else f"{identity.question_id}.batch-success"
    )
    release_id = (
        probe_question.release_id
        if probe_question is not None
        else f"{identity.release_id}.batch-success"
    )
    dataset_id = (
        probe_question.dataset_id
        if probe_question is not None
        else f"{identity.dataset_id}.batch-success"
    )
    answer_event_id = f"{identity.event_id}.batch-answer"
    comment_event_id = f"{identity.event_id}.batch-comment"
    proposal_event_id = (
        f"{identity.event_id}.batch-proposal" if probe_question is not None else None
    )
    comment_text = f"{SUCCESSFUL_BATCH_COMMENT_PREFIX} {identity.run_id}"
    question = (
        deepcopy(dict(probe_question.question))
        if probe_question is not None
        else {
            "question_id": question_id,
            "kind": "hosted_feedback_successful_batch_roundtrip",
            "run_id": identity.run_id,
        }
    )
    family = probe_question.family if probe_question is not None else FAMILY
    question_type = (
        probe_question.question_type if probe_question is not None else QUESTION_TYPE
    )
    extra = _feedback_context(
        attempt_id=identity.attempt_id,
        release_id=release_id,
        family=family,
        dataset_id=dataset_id,
        question_type=question_type,
        authoritative=probe_question is not None,
    )
    answer_extra = dict(extra)
    if probe_question is not None:
        answer_extra["is_correct"] = not probe_question.authoritative_is_correct
    trace = feedback.SessionTrace(
        session_id=identity.session_id,
        created_at=occurred_at,
    )
    answer = trace.record_answer(
        question,
        selected_letter=(
            probe_question.selected_letter if probe_question is not None else "A"
        ),
        selected_candidate_id=(
            probe_question.selected_candidate_id
            if probe_question is not None
            else "candidate_e2e_batch_A"
        ),
        event_id=answer_event_id,
        occurred_at=occurred_at,
        extra=answer_extra,
    )
    if probe_question is not None:
        assert proposal_event_id is not None
        trace.record_custom_setting(
            question,
            setting={
                "probe": "architectureiq_hosted_detail_v1",
                "run_id": identity.run_id,
                "model": {"type": "detail_probe", "width": 17},
                "optimizer": {"type": "adamw", "learning_rate_micros": 300},
                "loss": {"type": "mse"},
                "budget": {"batch_size": 32, "total_samples_seen": 4096},
            },
            event_id=proposal_event_id,
            occurred_at=occurred_at,
            extra={
                **extra,
                "label": f"{DETAIL_PROBE_LABEL_PREFIX} {identity.run_id}",
                "inherited_from": {
                    "candidate_id": probe_question.selected_candidate_id,
                    "exact_spec_match": False,
                },
                "n_seeds": DETAIL_PROBE_N_SEEDS,
                "base_seed": DETAIL_PROBE_BASE_SEED,
            },
        )
    trace.record_comment(
        question,
        category=COMMENT_CATEGORY,
        text=comment_text,
        event_id=comment_event_id,
        occurred_at=occurred_at,
        extra=extra,
    )
    if any(
        event["question_version"] != answer["question_version"]
        for event in trace.events
    ):
        raise RoundtripUploadError(
            "successful-batch events unexpectedly use different question versions"
        )
    batch_identity = SuccessfulBatchIdentity(
        run_id=identity.run_id,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        answer_event_id=answer_event_id,
        comment_event_id=comment_event_id,
        proposal_event_id=proposal_event_id,
        question_id=question_id,
        release_id=release_id,
        dataset_id=dataset_id,
        question_version=answer["question_version"],
        comment_text=comment_text,
    )
    if (
        probe_question is not None
        and batch_identity.question_version != probe_question.question_version
    ):
        raise RoundtripConfigurationError(
            "authoritative successful batch changed the attested question version"
        )
    return batch_identity, trace.to_envelope()


def _successful_receipt_expectation(
    receipt: feedback.UploadReceipt,
    *,
    accepted: int,
    duplicate: int,
    probe_name: str,
) -> str:
    """Require one exact HTTP 200 success receipt and return its UUID."""
    response = receipt.response
    request_id = receipt.request_id
    if (
        receipt.status_code != 200
        or not isinstance(response, Mapping)
        or not isinstance(request_id, str)
        or UUID_PATTERN.fullmatch(request_id) is None
        or str(uuid.UUID(request_id)) != request_id
        or response.get("request_id") != request_id
    ):
        raise RoundtripUploadError(
            f"{probe_name} receipt must be HTTP 200 with one matching UUID "
            "request id (canonical) and counters"
        )

    counts: dict[str, int] = {}
    for name in ("accepted", "duplicate", "conflict", "rejected"):
        value = response.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RoundtripUploadError(
                f"{probe_name} receipt must include non-negative integer counters"
            )
        counts[name] = value
    expected = {
        "accepted": accepted,
        "duplicate": duplicate,
        "conflict": 0,
        "rejected": 0,
    }
    if counts != expected:
        raise RoundtripUploadError(
            f"{probe_name} requires accepted={accepted}, duplicate={duplicate}, "
            "conflict=0, rejected=0"
        )
    return request_id


def _receipt_expectation(
    receipt: feedback.UploadReceipt,
    *,
    resume: bool,
) -> tuple[str, int, int]:
    """Return the exact request/outcome identity acknowledged by ingestion."""
    accepted, duplicate = (0, 1) if resume else (1, 0)
    request_id = _successful_receipt_expectation(
        receipt,
        accepted=accepted,
        duplicate=duplicate,
        probe_name=(
            "resume duplicate-only ingestion"
            if resume
            else "new-run accepted-only ingestion"
        ),
    )
    return request_id, accepted, duplicate


def _conflicting_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a valid event that reuses the ID with different logical content."""
    conflict_event = deepcopy(dict(event))
    payload = conflict_event.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise RoundtripUploadError(
            "the synthetic comment cannot be converted into a conflict probe"
        )
    payload["text"] = f"{payload['text']}{CONFLICT_COMMENT_SUFFIX}"
    return conflict_event


def _mixed_batch_trace(
    identity: SmokeIdentity,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a conflict plus a discoverable new event in one trace.

    The second event intentionally uses the same question/report dimensions as
    the stored comment.  If a broken backend partially commits the batch, the
    exact business-view assertions will therefore observe a second comment.
    """
    conflict_event = _conflicting_event(event)
    conflict_event["sequence"] = 1

    withheld_event = deepcopy(dict(event))
    # A dot is outside RUN_ID_PATTERN, making this namespace disjoint from
    # every normal ``evt_e2e_{run_id}`` smoke event.
    withheld_event["event_id"] = f"{identity.event_id}.mixed-withheld"
    withheld_event["sequence"] = 2
    payload = withheld_event.get("payload")
    if not isinstance(payload, dict):
        raise RoundtripUploadError(
            "the synthetic comment cannot be converted into a mixed-batch probe"
        )
    payload["text"] = f"{MIXED_BATCH_COMMENT_PREFIX} {identity.run_id}"
    return feedback.build_session_trace_envelope(
        identity.session_id,
        [conflict_event, withheld_event],
        created_at=str(event.get("occurred_at") or _utc_now()),
    )


def _conflict_expectation(
    error: feedback.FeedbackUploadConflictError,
    *,
    rejected: int = 1,
) -> str:
    """Require the exact structured 409 contract and return its request UUID."""
    response = error.response
    request_id = error.request_id
    error_body = response.get("error") if isinstance(response, Mapping) else None
    if (
        error.status_code != 409
        or error.error_code != "EVENT_ID_CONFLICT"
        or not isinstance(response, Mapping)
        or not isinstance(error_body, Mapping)
        or error_body.get("code") != "EVENT_ID_CONFLICT"
        or not isinstance(request_id, str)
        or UUID_PATTERN.fullmatch(request_id) is None
        or str(uuid.UUID(request_id)) != request_id
        or response.get("request_id") != request_id
    ):
        raise RoundtripUploadError(
            "conflict probe must return HTTP 409 / EVENT_ID_CONFLICT with one "
            "matching canonical UUID request id"
        )

    counts: dict[str, int] = {}
    for name in ("accepted", "duplicate", "conflict", "rejected"):
        value = response.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RoundtripUploadError(
                "conflict response must include non-negative integer counters"
            )
        counts[name] = value
    if counts != {
        "accepted": 0,
        "duplicate": 0,
        "conflict": 1,
        "rejected": rejected,
    }:
        probe_name = "conflict probe" if rejected == 1 else "mixed-batch probe"
        raise RoundtripUploadError(
            f"{probe_name} requires accepted=0, duplicate=0, conflict=1, "
            f"rejected={rejected}"
        )
    return request_id


def _negative_control_request_id(request_id: str) -> str:
    """Return a distinct canonical UUID while preserving version/variant bits."""
    replacement = "0" if request_id[-1] != "0" else "1"
    return f"{request_id[:-1]}{replacement}"


def _filters(
    identity: SmokeIdentity | SuccessfulBatchIdentity,
) -> dict[str, str]:
    return {
        "question_id": identity.question_id,
        "release_id": identity.release_id,
    }


def _fetch_pages(
    client: ReportsClient,
    identity: SmokeIdentity | SuccessfulBatchIdentity,
) -> dict[str, ReportPage]:
    return {
        view: client.fetch_page(view, filters=_filters(identity), limit=2, offset=0)
        for view in REPORT_VIEWS
    }


def _fetch_ingestion_outcome(
    client: ReportsClient,
    request_id: str,
) -> ReportPage:
    return client.fetch_page(
        INGESTION_VIEW,
        filters={"request_id": request_id},
        limit=1,
        offset=0,
    )


def _fetch_event_resolution(
    client: ReportsClient,
    event_id: str,
) -> ReportPage:
    return client.fetch_page(
        EVENT_RESOLUTION_VIEW,
        filters={"event_id": event_id},
        limit=1,
        offset=0,
    )


def _fetch_authority_status(client: ReportsClient) -> ReportPage:
    return client.fetch_page(
        AUTHORITY_STATUS_VIEW,
        filters={},
        limit=1,
        offset=0,
    )


def assert_authority_status(
    page: ReportPage,
    *,
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require the combined 15000/16000 authority revision marker."""
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "authority status did not return one complete exact row"
        )
    row = page.rows[0]
    expected_columns = {
        "authority_revision",
        "business_reports_authoritative",
        "registered_release_count",
        "registered_question_count",
        "registered_choice_count",
        "detail_revision",
        "detail_reports_authoritative",
    }
    if set(row) != expected_columns:
        raise RoundtripReportError(
            "authority status does not use the exact seven-column contract"
        )
    if row["authority_revision"] != "registry_v1":
        raise RoundtripReportError("authority status revision is not 'registry_v1'")
    if row["business_reports_authoritative"] is not True:
        raise RoundtripReportError(
            "authority status does not mark business reports authoritative"
        )
    if row["detail_revision"] != "detail_v1":
        raise RoundtripReportError("authority status revision is not 'detail_v1'")
    if row["detail_reports_authoritative"] is not True:
        raise RoundtripReportError(
            "authority status does not mark detail reports authoritative"
        )
    minima = {
        "registered_release_count": 1,
        "registered_question_count": probe_question.registry_question_count,
        "registered_choice_count": probe_question.registry_choice_count,
    }
    for field, minimum in minima.items():
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RoundtripReportError(
                f"authority status {field} is {value!r}, expected at least {minimum}"
            )


def assert_detail_report_surfaces(client: ReportsClient) -> None:
    """Prove the protected Edge allowlist and both 16000 RPCs are reachable."""
    missing_question_id = f"verifier_missing_{secrets.token_hex(16)}"
    for view in DETAIL_REPORT_VIEWS:
        page = client.fetch_page(
            view,
            filters={"question_id": missing_question_id},
            limit=1,
            offset=0,
        )
        if not page.is_complete or page.total != 0 or page.rows or page.offset != 0:
            raise RoundtripReportError(
                f"{view} did not return a complete empty negative-control page"
            )


def _report_datetime(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise RoundtripReportError(f"{field_name} is not an RFC 3339 timestamp")
    try:
        resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RoundtripReportError(f"{field_name} is not RFC 3339") from exc
    if resolved.tzinfo is None:
        raise RoundtripReportError(f"{field_name} has no timezone")
    return resolved


def _detail_time_window(occurred_at: Any) -> tuple[str, str]:
    """Return a one-millisecond UTC window containing one resolved event."""
    resolved = _report_datetime(
        occurred_at,
        field_name="authoritative event resolution occurred_at",
    )

    def utc_text(value: datetime) -> str:
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    return utc_text(resolved), utc_text(resolved + timedelta(milliseconds=1))


def _assert_detail_timestamps(
    row: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
    resolution: Mapping[str, Any],
    view: str,
) -> None:
    uploaded = _report_datetime(
        event.get("occurred_at"),
        field_name="uploaded probe occurred_at",
    )
    resolved = _report_datetime(
        resolution.get("occurred_at"),
        field_name="authoritative event resolution occurred_at",
    )
    detailed = _report_datetime(
        row.get("occurred_at"),
        field_name=f"{view} occurred_at",
    )
    if uploaded != resolved or uploaded != detailed:
        raise RoundtripReportError(
            f"{view} occurred_at does not match the uploaded probe event"
        )
    resolution_received = _report_datetime(
        resolution.get("received_at"),
        field_name="authoritative event resolution received_at",
    )
    detail_received = _report_datetime(
        row.get("received_at"),
        field_name=f"{view} received_at",
    )
    if resolution_received != detail_received:
        raise RoundtripReportError(
            f"{view} received_at does not match exact event resolution"
        )


def assert_empty_business_snapshot(
    snapshot: BusinessSnapshot,
    *,
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require one authority-attested, fully empty six-view negative control."""
    if not isinstance(snapshot, BusinessSnapshot):
        raise RoundtripReportError(
            "business snapshot negative control did not return a validated snapshot"
        )
    expected_metadata = {
        "snapshot_revision": "business_snapshot_v1",
        "authority_revision": "registry_v1",
        "business_reports_authoritative": True,
        "detail_revision": "detail_v1",
        "detail_reports_authoritative": True,
    }
    for field, expected in expected_metadata.items():
        if getattr(snapshot, field, None) != expected:
            raise RoundtripReportError(
                "business snapshot negative control "
                f"{field} is {getattr(snapshot, field, None)!r}, "
                f"expected {expected!r}"
            )
    _report_datetime(
        snapshot.snapshot_at,
        field_name="business snapshot negative control snapshot_at",
    )
    minima = {
        "registered_release_count": 1,
        "registered_question_count": probe_question.registry_question_count,
        "registered_choice_count": probe_question.registry_choice_count,
    }
    for field, minimum in minima.items():
        value = getattr(snapshot, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RoundtripReportError(
                "business snapshot negative control "
                f"{field} is {value!r}, expected at least {minimum}"
            )
    if set(snapshot.pages) != set(BUSINESS_REPORT_VIEWS):
        raise RoundtripReportError(
            "business snapshot negative control does not contain exactly six views"
        )

    summary_page = snapshot.pages[SUMMARY_VIEW]
    if (
        not summary_page.is_complete
        or summary_page.view != SUMMARY_VIEW
        or summary_page.total != 1
        or len(summary_page.rows) != 1
        or summary_page.limit != 1
        or summary_page.offset != 0
    ):
        raise RoundtripReportError(
            "business snapshot negative control summary is not one complete row"
        )
    expected_summary = {
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
    if dict(summary_page.rows[0]) != expected_summary:
        raise RoundtripReportError(
            "business snapshot negative control summary is not exactly empty"
        )

    for view in BUSINESS_REPORT_VIEWS:
        if view == SUMMARY_VIEW:
            continue
        page = snapshot.pages[view]
        if (
            not page.is_complete
            or page.view != view
            or page.rows
            or page.total != 0
            or page.limit != 1
            or page.offset != 0
        ):
            raise RoundtripReportError(
                f"business snapshot negative control {view} is not complete and empty"
            )


def assert_session_attempt_business_snapshot(
    snapshot: BusinessSnapshot,
    *,
    identity: SmokeIdentity,
    batch_identity: SuccessfulBatchIdentity,
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require the exact four-event trace under one session/attempt filter."""
    if not isinstance(snapshot, BusinessSnapshot):
        raise RoundtripReportError(
            "session/attempt filter did not return a validated business snapshot"
        )
    expected_metadata = {
        "snapshot_revision": "business_snapshot_v1",
        "authority_revision": "registry_v1",
        "business_reports_authoritative": True,
        "detail_revision": "detail_v1",
        "detail_reports_authoritative": True,
    }
    if any(
        getattr(snapshot, field, None) != expected
        for field, expected in expected_metadata.items()
    ):
        raise RoundtripReportError(
            "session/attempt business snapshot authority metadata is invalid"
        )
    if set(snapshot.pages) != set(BUSINESS_REPORT_VIEWS):
        raise RoundtripReportError(
            "session/attempt business snapshot does not contain exactly six views"
        )

    expected_totals = {
        SUMMARY_VIEW: 1,
        SESSIONS_VIEW: 1,
        QUESTIONS_VIEW: 1,
        ANSWER_DETAILS_VIEW: 1,
        PROPOSAL_DETAILS_VIEW: 1,
        COMMENTS_VIEW: 2,
    }
    for view, expected_total in expected_totals.items():
        page = snapshot.pages[view]
        if (
            not page.is_complete
            or page.view != view
            or page.total != expected_total
            or len(page.rows) != expected_total
            or page.offset != 0
        ):
            raise RoundtripReportError(
                f"session/attempt business snapshot {view} has the wrong total"
            )

    summary = snapshot.pages[SUMMARY_VIEW].rows[0]
    expected_summary_counts = {
        "event_count": 4,
        "session_count": 1,
        "attempt_count": 1,
        "solve_attempt_count": 1,
        "answered_attempt_count": 1,
        "question_count": 1,
        "answer_count": 1,
        "known_answer_count": 1,
        "correct_answer_count": int(probe_question.authoritative_is_correct),
        "incorrect_answer_count": int(not probe_question.authoritative_is_correct),
        "unknown_answer_count": 0,
        "proposal_count": 1,
        "rejected_setting_count": 0,
        "completed_run_count": 0,
        "failed_run_count": 0,
        "comment_count": 2,
        "attempts_with_proposal": 1,
    }
    if any(
        summary.get(field) != value for field, value in expected_summary_counts.items()
    ):
        raise RoundtripReportError(
            "session/attempt business summary does not match the uploaded trace"
        )
    expected_accuracy = 1.0 if probe_question.authoritative_is_correct else 0.0
    if summary.get("accuracy") != expected_accuracy:
        raise RoundtripReportError(
            "session/attempt business summary accuracy is not registry-derived"
        )
    if summary.get("proposal_usage_rate") != 1.0:
        raise RoundtripReportError(
            "session/attempt business summary proposal usage is inconsistent"
        )

    session = snapshot.pages[SESSIONS_VIEW].rows[0]
    if (
        session.get("session_id") != identity.session_id
        or session.get("attempt_id") != identity.attempt_id
        or session.get("event_count") != 4
    ):
        raise RoundtripReportError(
            "session/attempt snapshot session row has the wrong identity"
        )
    question = snapshot.pages[QUESTIONS_VIEW].rows[0]
    if (
        question.get("question_id") != probe_question.question_id
        or question.get("question_version") != probe_question.question_version
        or question.get("event_count") != 4
    ):
        raise RoundtripReportError(
            "session/attempt snapshot question row has the wrong identity"
        )

    expected_event_ids = {
        ANSWER_DETAILS_VIEW: {batch_identity.answer_event_id},
        PROPOSAL_DETAILS_VIEW: {batch_identity.proposal_event_id},
        COMMENTS_VIEW: {identity.event_id, batch_identity.comment_event_id},
    }
    for view, expected_ids in expected_event_ids.items():
        rows = snapshot.pages[view].rows
        if {row.get("event_id") for row in rows} != expected_ids or any(
            row.get("session_id") != identity.session_id
            or row.get("attempt_id") != identity.attempt_id
            for row in rows
        ):
            raise RoundtripReportError(
                f"session/attempt snapshot {view} rows have the wrong identity"
            )


def _poll_session_attempt_business_snapshots(
    reports_client: ReportsClient,
    *,
    identity: SmokeIdentity,
    batch_identity: SuccessfulBatchIdentity,
    probe_question: AuthoritativeProbeQuestion,
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """Prove both identity filters with one positive and two negative snapshots."""
    wrong_session_id = f"verifier_wrong_session_{secrets.token_hex(16)}"
    wrong_attempt_id = f"verifier_wrong_attempt_{secrets.token_hex(16)}"
    deadline = monotonic() + timeout_seconds
    polls = 0
    last_error: str | None = None
    while True:
        polls += 1
        try:
            assert_session_attempt_business_snapshot(
                reports_client.fetch_business_snapshot(
                    filters={
                        "session_id": identity.session_id,
                        "attempt_id": identity.attempt_id,
                    },
                    limit=10,
                ),
                identity=identity,
                batch_identity=batch_identity,
                probe_question=probe_question,
            )
            assert_empty_business_snapshot(
                reports_client.fetch_business_snapshot(
                    filters={
                        "session_id": wrong_session_id,
                        "attempt_id": identity.attempt_id,
                    },
                    limit=1,
                ),
                probe_question=probe_question,
            )
            assert_empty_business_snapshot(
                reports_client.fetch_business_snapshot(
                    filters={
                        "session_id": identity.session_id,
                        "attempt_id": wrong_attempt_id,
                    },
                    limit=1,
                ),
                probe_question=probe_question,
            )
            return polls
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        if monotonic() >= deadline:
            raise RoundtripReportError(
                "session/attempt business snapshots did not converge before the "
                f"timeout: {last_error}"
            )
        sleep(poll_interval)


def _detail_filters(
    probe_question: AuthoritativeProbeQuestion,
    resolution: Mapping[str, Any],
) -> dict[str, str]:
    start, end = _detail_time_window(resolution.get("occurred_at"))
    return {
        "release_id": probe_question.release_id,
        "family": probe_question.family,
        "question_type": probe_question.question_type,
        "question_id": probe_question.question_id,
        "from": start,
        "to": end,
    }


def _fetch_detail_event_row(
    client: ReportsClient,
    *,
    view: str,
    event_id: str,
    probe_question: AuthoritativeProbeQuestion,
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Find one event by scanning every page in its narrow authority window."""
    filters = _detail_filters(probe_question, resolution)
    offset = 0
    expected_total: int | None = None
    matches: list[dict[str, Any]] = []
    while True:
        page = client.fetch_page(
            view,
            filters=filters,
            limit=DETAIL_PAGE_LIMIT,
            offset=offset,
        )
        if (
            page.view != view
            or page.limit != DETAIL_PAGE_LIMIT
            or page.offset != offset
        ):
            raise RoundtripReportError(
                f"{view} returned pagination metadata that does not match the query"
            )
        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise RoundtripReportError(
                f"{view} total changed while scanning the exact event window"
            )
        matches.extend(
            dict(row) for row in page.rows if row.get("event_id") == event_id
        )
        next_offset = offset + len(page.rows)
        if next_offset >= page.total:
            break
        if not page.rows:
            raise RoundtripReportError(
                f"{view} returned an empty page before its declared total"
            )
        offset = next_offset
    if len(matches) != 1:
        raise RoundtripReportError(
            f"{view} returned {len(matches)} rows for exact event {event_id!r}"
        )
    return matches[0]


def _strict_json_object_text(value: Any, *, field_name: str) -> dict[str, Any]:
    """Decode interoperable JSON object text without lossy duplicate handling."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for key, item in pairs:
            if key in resolved:
                raise ValueError(f"duplicate key {key!r}")
            resolved[key] = item
        return resolved

    def reject_constant(constant: str) -> Any:
        raise ValueError(f"invalid numeric constant {constant}")

    if not isinstance(value, str):
        raise RoundtripReportError(f"{field_name} is not JSON object text")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RoundtripReportError(f"{field_name} is not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RoundtripReportError(f"{field_name} does not encode a JSON object")

    pending = [parsed]
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in item):
                raise RoundtripReportError(
                    f"{field_name} contains a Unicode surrogate code point"
                )
            continue
        if isinstance(item, int):
            if not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER:
                raise RoundtripReportError(
                    f"{field_name} contains an unsafe integer-valued number"
                )
            continue
        if isinstance(item, float):
            if not math.isfinite(item) or (
                item.is_integer()
                and not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER
            ):
                raise RoundtripReportError(
                    f"{field_name} contains a non-finite or unsafe number"
                )
            continue
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
            continue
        if isinstance(item, list):
            pending.extend(item)
            continue
        raise RoundtripReportError(f"{field_name} contains a non-JSON value")
    return parsed


def _assert_exact_detail_facts(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    view: str,
) -> None:
    for field, value in expected.items():
        if row.get(field) != value:
            raise RoundtripReportError(
                f"{view} {field} is {row.get(field)!r}, expected {value!r}"
            )


def assert_answer_detail_row(
    row: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
    resolution: Mapping[str, Any],
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require canonical identity, answer resolution, and mismatch evidence."""
    expected_columns = {
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
    }
    if set(row) != expected_columns:
        raise RoundtripReportError(
            f"{ANSWER_DETAILS_VIEW} does not use the exact detail_v1 columns"
        )
    _assert_detail_timestamps(
        row,
        event=event,
        resolution=resolution,
        view=ANSWER_DETAILS_VIEW,
    )
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise RoundtripReportError("answer detail probe event has no payload")
    expected = {
        "event_id": event.get("event_id"),
        "occurred_at": resolution.get("occurred_at"),
        "received_at": resolution.get("received_at"),
        "session_id": event.get("session_id"),
        "attempt_id": payload.get("attempt_id"),
        "question_id": probe_question.question_id,
        "question_version": probe_question.question_version,
        "release_id": probe_question.release_id,
        "family": probe_question.family,
        "dataset_id": probe_question.dataset_id,
        "question_type": probe_question.question_type,
        "selected_letter": probe_question.selected_letter,
        "client_selected_candidate_id": probe_question.selected_candidate_id,
        "selected_candidate_id": probe_question.selected_candidate_id,
        "answer_status": "resolved",
        "is_correct": probe_question.authoritative_is_correct,
        "client_is_correct": not probe_question.authoritative_is_correct,
        "client_context_mismatch": True,
        "client_correctness_mismatch": True,
    }
    _assert_exact_detail_facts(row, expected, view=ANSWER_DETAILS_VIEW)


def assert_proposal_detail_row(
    row: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
    resolution: Mapping[str, Any],
    probe_question: AuthoritativeProbeQuestion,
) -> None:
    """Require one canonical proposed-setting row with exact structured facts."""
    expected_columns = {
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
    }
    if set(row) != expected_columns:
        raise RoundtripReportError(
            f"{PROPOSAL_DETAILS_VIEW} does not use the exact detail_v1 columns"
        )
    _assert_detail_timestamps(
        row,
        event=event,
        resolution=resolution,
        view=PROPOSAL_DETAILS_VIEW,
    )
    payload = event.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("setting"), Mapping
    ):
        raise RoundtripReportError("proposal detail probe event has no setting")
    expected = {
        "event_id": event.get("event_id"),
        "occurred_at": resolution.get("occurred_at"),
        "received_at": resolution.get("received_at"),
        "session_id": event.get("session_id"),
        "attempt_id": payload.get("attempt_id"),
        "question_id": probe_question.question_id,
        "question_version": probe_question.question_version,
        "release_id": probe_question.release_id,
        "family": probe_question.family,
        "dataset_id": probe_question.dataset_id,
        "question_type": probe_question.question_type,
        "setting_status": "proposed",
        "label": payload.get("label"),
        "n_seeds": DETAIL_PROBE_N_SEEDS,
        "base_seed": DETAIL_PROBE_BASE_SEED,
        "error_type": None,
    }
    _assert_exact_detail_facts(row, expected, view=PROPOSAL_DETAILS_VIEW)
    for field, payload_field in (
        ("setting_json", "setting"),
        ("inherited_from_json", "inherited_from"),
    ):
        observed = _strict_json_object_text(
            row.get(field),
            field_name=f"{PROPOSAL_DETAILS_VIEW} {field}",
        )
        expected_json = payload.get(payload_field)
        if not isinstance(expected_json, Mapping):
            raise RoundtripReportError(
                f"proposal detail probe payload has no {payload_field} object"
            )
        observed_canonical = json.dumps(
            observed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_canonical = json.dumps(
            dict(expected_json),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if observed_canonical != expected_canonical:
            raise RoundtripReportError(
                f"{PROPOSAL_DETAILS_VIEW} {field} does not match the uploaded object"
            )


def _event_resolution_row(page: ReportPage, *, event_id: str) -> Mapping[str, Any]:
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "event resolution did not return one complete exact row"
        )
    row = page.rows[0]
    if row.get("event_id") != event_id:
        raise RoundtripReportError(
            f"event resolution returned {row.get('event_id')!r}, expected {event_id!r}"
        )
    return row


def assert_event_resolution_not_found(
    page: ReportPage,
    *,
    event_id: str,
) -> None:
    """Require the exact-event negative-control sentinel."""
    row = _event_resolution_row(page, event_id=event_id)
    expected = {
        "registry_status": "not_found",
        "answer_status": "not_found",
        "client_context_mismatch": False,
        "client_correctness_mismatch": False,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RoundtripReportError(
                f"event resolution {field} is {row.get(field)!r}, expected {value!r}"
            )
    fact_fields = (
        "event_type",
        "occurred_at",
        "received_at",
        "session_id",
        "attempt_id",
        "client_release_id",
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
    )
    populated = [field for field in fact_fields if row.get(field) is not None]
    if populated:
        raise RoundtripReportError(
            "not-found event resolution unexpectedly contains facts: "
            + ", ".join(populated)
        )


def assert_authoritative_event_resolution(
    page: ReportPage,
    *,
    event_id: str,
    event_type: str,
    session_id: str,
    attempt_id: str,
    probe_question: AuthoritativeProbeQuestion,
) -> dict[str, Any]:
    """Require server-derived identity and correctness for one exact event."""
    row = _event_resolution_row(page, event_id=event_id)
    is_answer = event_type == "answer_submitted"
    expected = {
        "event_type": event_type,
        "session_id": session_id,
        "attempt_id": attempt_id,
        "client_release_id": probe_question.release_id,
        "registry_status": "matched",
        "answer_status": "resolved" if is_answer else "not_answer",
        "registry_id": probe_question.registry_id,
        "release_id": probe_question.release_id,
        "question_id": probe_question.question_id,
        "question_version": probe_question.question_version,
        "family": probe_question.family,
        "dataset_id": probe_question.dataset_id,
        "question_type": probe_question.question_type,
        "selected_letter": probe_question.selected_letter if is_answer else None,
        "client_selected_candidate_id": (
            probe_question.selected_candidate_id if is_answer else None
        ),
        "selected_candidate_id": (
            probe_question.selected_candidate_id if is_answer else None
        ),
        "authoritative_is_correct": (
            probe_question.authoritative_is_correct if is_answer else None
        ),
        "client_is_correct": (
            not probe_question.authoritative_is_correct if is_answer else None
        ),
        "client_context_mismatch": True,
        "client_correctness_mismatch": is_answer,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise RoundtripReportError(
                "authoritative event resolution "
                f"{field} is {row.get(field)!r}, expected {value!r}"
            )
    return dict(row)


def _summary_row(page: ReportPage) -> Mapping[str, Any]:
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError("summary report did not return one complete row")
    return page.rows[0]


def _preflight_is_empty(pages: Mapping[str, ReportPage]) -> bool:
    summary = _summary_row(pages[SUMMARY_VIEW])
    return summary.get("event_count") == 0 and all(
        pages[view].is_complete and pages[view].total == 0
        for view in (SESSIONS_VIEW, QUESTIONS_VIEW, COMMENTS_VIEW)
    )


def _assert_single_row(
    pages: Mapping[str, ReportPage],
    view: str,
) -> Mapping[str, Any]:
    page = pages[view]
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(f"{view} did not return one complete row")
    return page.rows[0]


def assert_roundtrip_visible(
    pages: Mapping[str, ReportPage],
    identity: SmokeIdentity,
) -> None:
    """Require exact identity and aggregate evidence in every report view."""
    summary = _summary_row(pages[SUMMARY_VIEW])
    expected_summary = {
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
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise RoundtripReportError(
                f"summary {field} is {summary.get(field)!r}, expected {expected!r}"
            )

    session = _assert_single_row(pages, SESSIONS_VIEW)
    expected_session = {
        "session_id": identity.session_id,
        "attempt_id": identity.attempt_id,
        "release_ids": [identity.release_id],
        "families": [FAMILY],
        "question_types": [QUESTION_TYPE],
        "event_count": 1,
        "question_count": 1,
        "answer_count": 0,
        "comment_count": 1,
    }
    for field, expected in expected_session.items():
        if session.get(field) != expected:
            raise RoundtripReportError(
                f"session {field} is {session.get(field)!r}, expected {expected!r}"
            )

    question = _assert_single_row(pages, QUESTIONS_VIEW)
    expected_question = {
        "question_id": identity.question_id,
        "question_version": identity.question_version,
        "release_id": identity.release_id,
        "family": FAMILY,
        "dataset_id": identity.dataset_id,
        "question_type": QUESTION_TYPE,
        "event_count": 1,
        "session_count": 1,
        "answer_count": 0,
        "comment_count": 1,
    }
    for field, expected in expected_question.items():
        if question.get(field) != expected:
            raise RoundtripReportError(
                f"question {field} is {question.get(field)!r}, expected {expected!r}"
            )

    comment = _assert_single_row(pages, COMMENTS_VIEW)
    expected_comment = {
        "event_id": identity.event_id,
        "session_id": identity.session_id,
        "attempt_id": identity.attempt_id,
        "question_id": identity.question_id,
        "question_version": identity.question_version,
        "release_id": identity.release_id,
        "family": FAMILY,
        "question_type": QUESTION_TYPE,
        "category": COMMENT_CATEGORY,
        "comment_text": identity.comment_text,
    }
    for field, expected in expected_comment.items():
        if comment.get(field) != expected:
            raise RoundtripReportError(
                f"comment {field} is {comment.get(field)!r}, expected {expected!r}"
            )


def assert_successful_batch_visible(
    pages: Mapping[str, ReportPage],
    identity: SuccessfulBatchIdentity,
) -> None:
    """Require the answer/comment batch to be visible under its own filters."""
    summary = _summary_row(pages[SUMMARY_VIEW])
    expected_summary = {
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
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise RoundtripReportError(
                "successful-batch summary "
                f"{field} is {summary.get(field)!r}, expected {expected!r}"
            )

    session = _assert_single_row(pages, SESSIONS_VIEW)
    expected_session = {
        "session_id": identity.session_id,
        "attempt_id": identity.attempt_id,
        "release_ids": [identity.release_id],
        "families": [FAMILY],
        "question_types": [QUESTION_TYPE],
        "event_count": 2,
        "question_count": 1,
        "answer_count": 1,
        "comment_count": 1,
    }
    for field, expected in expected_session.items():
        if session.get(field) != expected:
            raise RoundtripReportError(
                "successful-batch session "
                f"{field} is {session.get(field)!r}, expected {expected!r}"
            )

    question = _assert_single_row(pages, QUESTIONS_VIEW)
    expected_question = {
        "question_id": identity.question_id,
        "question_version": identity.question_version,
        "release_id": identity.release_id,
        "family": FAMILY,
        "dataset_id": identity.dataset_id,
        "question_type": QUESTION_TYPE,
        "event_count": 2,
        "session_count": 1,
        "answer_count": 1,
        "comment_count": 1,
    }
    for field, expected in expected_question.items():
        if question.get(field) != expected:
            raise RoundtripReportError(
                "successful-batch question "
                f"{field} is {question.get(field)!r}, expected {expected!r}"
            )

    comment = _assert_single_row(pages, COMMENTS_VIEW)
    expected_comment = {
        "event_id": identity.comment_event_id,
        "session_id": identity.session_id,
        "attempt_id": identity.attempt_id,
        "question_id": identity.question_id,
        "question_version": identity.question_version,
        "release_id": identity.release_id,
        "family": FAMILY,
        "question_type": QUESTION_TYPE,
        "category": COMMENT_CATEGORY,
        "comment_text": identity.comment_text,
    }
    for field, expected in expected_comment.items():
        if comment.get(field) != expected:
            raise RoundtripReportError(
                "successful-batch comment "
                f"{field} is {comment.get(field)!r}, expected {expected!r}"
            )


def assert_ingestion_outcome(
    page: ReportPage,
    *,
    accepted: int,
    duplicate: int,
    event_count: int = 1,
) -> None:
    """Require one successful persisted outcome for the exact POST request."""
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "ingestion outcome did not return one complete aggregate row"
        )
    if accepted + duplicate != event_count:
        raise RoundtripReportError(
            "successful ingestion outcome expectation must classify every event"
        )
    duplicate_rate = duplicate / event_count
    expected = {
        "recorded_request_count": 1,
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
    for field, expected_value in expected.items():
        if page.rows[0].get(field) != expected_value:
            raise RoundtripReportError(
                "ingestion outcome "
                f"{field} is {page.rows[0].get(field)!r}, "
                f"expected {expected_value!r}"
            )


def assert_ingestion_outcome_absent(page: ReportPage) -> None:
    """Require the negative-control request UUID to match no persisted outcome."""
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "ingestion negative control did not return one complete aggregate row"
        )
    expected = {
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
    for field, expected_value in expected.items():
        if page.rows[0].get(field) != expected_value:
            raise RoundtripReportError(
                "ingestion negative control "
                f"{field} is {page.rows[0].get(field)!r}, "
                f"expected {expected_value!r}"
            )


def assert_conflict_ingestion_outcome(page: ReportPage) -> None:
    """Require one exact rejection and correlated private-audit count."""
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "conflict ingestion outcome did not return one complete aggregate row"
        )
    expected = {
        "recorded_request_count": 1,
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
    for field, expected_value in expected.items():
        if page.rows[0].get(field) != expected_value:
            raise RoundtripReportError(
                "conflict ingestion outcome "
                f"{field} is {page.rows[0].get(field)!r}, "
                f"expected {expected_value!r}"
            )


def assert_mixed_batch_ingestion_outcome(page: ReportPage) -> None:
    """Require the exact all-or-none mixed-batch rejection evidence."""
    if not page.is_complete or page.total != 1 or len(page.rows) != 1:
        raise RoundtripReportError(
            "mixed-batch ingestion outcome did not return one complete aggregate row"
        )
    row = page.rows[0]
    expected = {
        "recorded_request_count": 1,
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
    timestamp_fields = {"first_started_at", "last_finished_at"}
    if set(row) != set(expected) | timestamp_fields:
        raise RoundtripReportError(
            "mixed-batch ingestion outcome must contain the exact 22-column schema"
        )
    if any(
        not isinstance(row.get(field), str) or not row[field]
        for field in timestamp_fields
    ):
        raise RoundtripReportError(
            "mixed-batch ingestion outcome must contain persisted request timestamps"
        )
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise RoundtripReportError(
                "mixed-batch ingestion outcome "
                f"{field} is {row.get(field)!r}, "
                f"expected {expected_value!r}"
            )


def _poll_successful_batch_request(
    reports_client: ReportsClient,
    identity: SuccessfulBatchIdentity,
    *,
    request_id: str,
    accepted: int,
    duplicate: int,
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    phase_name: str,
) -> int:
    """Poll one batch POST until both its outcome and readback are exact."""
    deadline = monotonic() + timeout_seconds
    polls = 0
    last_error = f"{phase_name} outcome has not appeared yet"
    while True:
        polls += 1
        try:
            pages = _fetch_pages(reports_client, identity)
            assert_successful_batch_visible(pages, identity)
            outcome = _fetch_ingestion_outcome(reports_client, request_id)
            assert_ingestion_outcome(
                outcome,
                accepted=accepted,
                duplicate=duplicate,
                event_count=2,
            )
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    "report endpoint rejected the successful-batch verification "
                    f"query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            return polls
        if monotonic() >= deadline:
            raise RoundtripReportError(
                f"{phase_name} reports did not converge before the timeout: "
                f"{last_error}"
            )
        sleep(poll_interval)


def _run_successful_batch_probe(
    feedback_client: feedback.FeedbackClient,
    reports_client: ReportsClient,
    *,
    identity: SuccessfulBatchIdentity,
    trace: Mapping[str, Any],
    preexisting: bool,
    prior_request_ids: set[str],
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, str, bool, int]:
    """Post an answer/comment batch, replay it unchanged, and verify both."""
    first_accepted, first_duplicate = (0, 2) if preexisting else (2, 0)
    try:
        first_receipt = feedback_client.post_trace(trace)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"successful-batch first upload failed: {exc}"
        ) from exc
    first_request_id = _successful_receipt_expectation(
        first_receipt,
        accepted=first_accepted,
        duplicate=first_duplicate,
        probe_name=(
            "successful-batch resume" if preexisting else "successful-batch first write"
        ),
    )
    if first_request_id in prior_request_ids:
        raise RoundtripUploadError(
            "successful-batch first upload reused an earlier request id instead "
            "of issuing a new UUID"
        )
    polls = _poll_successful_batch_request(
        reports_client,
        identity,
        request_id=first_request_id,
        accepted=first_accepted,
        duplicate=first_duplicate,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="successful-batch first upload",
    )

    try:
        replay_receipt = feedback_client.post_trace(trace)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"successful-batch replay upload failed: {exc}"
        ) from exc
    replay_request_id = _successful_receipt_expectation(
        replay_receipt,
        accepted=0,
        duplicate=2,
        probe_name="successful-batch unchanged replay",
    )
    if replay_request_id in prior_request_ids | {first_request_id}:
        raise RoundtripUploadError(
            "successful-batch replay reused an earlier request id instead of "
            "issuing a new UUID"
        )
    polls += _poll_successful_batch_request(
        reports_client,
        identity,
        request_id=replay_request_id,
        accepted=0,
        duplicate=2,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="successful-batch replay",
    )
    return first_request_id, replay_request_id, not preexisting, polls


def _poll_authoritative_state(
    reports_client: ReportsClient,
    *,
    probe_question: AuthoritativeProbeQuestion,
    session_id: str,
    attempt_id: str,
    expected_events: Mapping[str, str],
    absent_event_ids: tuple[str, ...] = (),
    unchanged_rows: Mapping[str, Mapping[str, Any]] | None = None,
    request_id: str,
    outcome_assertion: Callable[[ReportPage], None],
    negative_request_id: str | None = None,
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    phase_name: str,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Poll exact event authority plus one persisted request outcome."""
    deadline = monotonic() + timeout_seconds
    polls = 0
    last_error = f"{phase_name} authority/outcome has not appeared yet"
    while True:
        polls += 1
        try:
            rows: dict[str, dict[str, Any]] = {}
            for event_id, event_type in expected_events.items():
                page = _fetch_event_resolution(reports_client, event_id)
                row = assert_authoritative_event_resolution(
                    page,
                    event_id=event_id,
                    event_type=event_type,
                    session_id=session_id,
                    attempt_id=attempt_id,
                    probe_question=probe_question,
                )
                expected_unchanged = (unchanged_rows or {}).get(event_id)
                if expected_unchanged is not None and row != expected_unchanged:
                    raise RoundtripReportError(
                        f"{phase_name} changed exact resolution for {event_id!r}"
                    )
                rows[event_id] = row
            for event_id in absent_event_ids:
                assert_event_resolution_not_found(
                    _fetch_event_resolution(reports_client, event_id),
                    event_id=event_id,
                )
            outcome_assertion(_fetch_ingestion_outcome(reports_client, request_id))
            if negative_request_id is not None:
                assert_ingestion_outcome_absent(
                    _fetch_ingestion_outcome(
                        reports_client,
                        negative_request_id,
                    )
                )
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    "report endpoint rejected the authoritative verification "
                    f"query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            return polls, rows
        if monotonic() >= deadline:
            raise RoundtripReportError(
                f"{phase_name} reports did not converge before the timeout: "
                f"{last_error}"
            )
        sleep(poll_interval)


def _poll_authoritative_detail_reports(
    reports_client: ReportsClient,
    *,
    identity: SuccessfulBatchIdentity,
    trace: Mapping[str, Any],
    resolution_rows: Mapping[str, Mapping[str, Any]],
    probe_question: AuthoritativeProbeQuestion,
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """Poll until this batch's answer and proposal detail rows are both exact."""
    if identity.proposal_event_id is None:
        raise RoundtripReportError(
            "authoritative detail verification requires a proposal event"
        )
    raw_events = trace.get("events")
    if not isinstance(raw_events, list):
        raise RoundtripReportError(
            "authoritative successful-batch trace has no event list"
        )
    events = {
        event.get("event_id"): event
        for event in raw_events
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }
    answer = events.get(identity.answer_event_id)
    proposal = events.get(identity.proposal_event_id)
    answer_resolution = resolution_rows.get(identity.answer_event_id)
    proposal_resolution = resolution_rows.get(identity.proposal_event_id)
    if any(
        value is None
        for value in (answer, proposal, answer_resolution, proposal_resolution)
    ):
        raise RoundtripReportError(
            "authoritative detail verification is missing its exact probe facts"
        )
    assert answer is not None
    assert proposal is not None
    assert answer_resolution is not None
    assert proposal_resolution is not None

    deadline = monotonic() + timeout_seconds
    polls = 0
    last_error = "authoritative answer/proposal details have not appeared yet"
    while True:
        polls += 1
        try:
            answer_row = _fetch_detail_event_row(
                reports_client,
                view=ANSWER_DETAILS_VIEW,
                event_id=identity.answer_event_id,
                probe_question=probe_question,
                resolution=answer_resolution,
            )
            assert_answer_detail_row(
                answer_row,
                event=answer,
                resolution=answer_resolution,
                probe_question=probe_question,
            )
            proposal_row = _fetch_detail_event_row(
                reports_client,
                view=PROPOSAL_DETAILS_VIEW,
                event_id=identity.proposal_event_id,
                probe_question=probe_question,
                resolution=proposal_resolution,
            )
            assert_proposal_detail_row(
                proposal_row,
                event=proposal,
                resolution=proposal_resolution,
                probe_question=probe_question,
            )
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    "report endpoint rejected the authoritative detail-row "
                    f"verification query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            return polls
        if monotonic() >= deadline:
            raise RoundtripReportError(
                "authoritative detail reports did not converge before the "
                f"timeout: {last_error}"
            )
        sleep(poll_interval)


def _run_authoritative_successful_batch_probe(
    feedback_client: feedback.FeedbackClient,
    reports_client: ReportsClient,
    *,
    identity: SuccessfulBatchIdentity,
    trace: Mapping[str, Any],
    probe_question: AuthoritativeProbeQuestion,
    preexisting: bool,
    prior_request_ids: set[str],
    timeout_seconds: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[str, str, bool, bool, int]:
    """Prove batch outcomes, exact authority, and both detail row semantics."""
    if identity.proposal_event_id is None:
        raise RoundtripUploadError(
            "authoritative successful batch has no deterministic proposal event"
        )
    event_count = 3
    first_accepted, first_duplicate = (
        (0, event_count) if preexisting else (event_count, 0)
    )
    try:
        first_receipt = feedback_client.post_trace(trace)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"authoritative successful-batch first upload failed: {exc}"
        ) from exc
    first_request_id = _successful_receipt_expectation(
        first_receipt,
        accepted=first_accepted,
        duplicate=first_duplicate,
        probe_name=(
            "authoritative successful-batch resume"
            if preexisting
            else "authoritative successful-batch first write"
        ),
    )
    if first_request_id in prior_request_ids:
        raise RoundtripUploadError(
            "authoritative successful-batch first upload reused an earlier "
            "request id instead of issuing a new UUID"
        )

    def assert_first_outcome(page: ReportPage) -> None:
        assert_ingestion_outcome(
            page,
            accepted=first_accepted,
            duplicate=first_duplicate,
            event_count=event_count,
        )

    polls, first_rows = _poll_authoritative_state(
        reports_client,
        probe_question=probe_question,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        expected_events={
            identity.answer_event_id: "answer_submitted",
            identity.proposal_event_id: "custom_setting_proposed",
            identity.comment_event_id: "comment_submitted",
        },
        request_id=first_request_id,
        outcome_assertion=assert_first_outcome,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="authoritative successful-batch first upload",
    )
    detail_polls = _poll_authoritative_detail_reports(
        reports_client,
        identity=identity,
        trace=trace,
        resolution_rows=first_rows,
        probe_question=probe_question,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
    )

    try:
        replay_receipt = feedback_client.post_trace(trace)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"authoritative successful-batch replay upload failed: {exc}"
        ) from exc
    replay_request_id = _successful_receipt_expectation(
        replay_receipt,
        accepted=0,
        duplicate=event_count,
        probe_name="authoritative successful-batch unchanged replay",
    )
    if replay_request_id in prior_request_ids | {first_request_id}:
        raise RoundtripUploadError(
            "authoritative successful-batch replay reused an earlier request id "
            "instead of issuing a new UUID"
        )

    def assert_replay_outcome(page: ReportPage) -> None:
        assert_ingestion_outcome(
            page,
            accepted=0,
            duplicate=event_count,
            event_count=event_count,
        )

    replay_polls, _ = _poll_authoritative_state(
        reports_client,
        probe_question=probe_question,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        expected_events={
            identity.answer_event_id: "answer_submitted",
            identity.proposal_event_id: "custom_setting_proposed",
            identity.comment_event_id: "comment_submitted",
        },
        unchanged_rows=first_rows,
        request_id=replay_request_id,
        outcome_assertion=assert_replay_outcome,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="authoritative successful-batch replay",
    )
    return (
        first_request_id,
        replay_request_id,
        not preexisting,
        True,
        polls + detail_polls + replay_polls,
    )


def _run_legacy_roundtrip(
    feedback_client: feedback.FeedbackClient,
    reports_client: ReportsClient,
    *,
    run_id: str,
    resume: bool = False,
    skip_successful_batch_probe: bool = False,
    skip_conflict_probe: bool = False,
    include_mixed_batch_probe: bool = False,
    timeout_seconds: float = 90.0,
    poll_interval: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], str] = _utc_now,
) -> RoundtripResult:
    """Prove single-event and multi-event ingestion plus conflict handling.

    The successful-batch phase is on by default and posts the same two-event
    answer/comment trace twice.  Unless explicitly skipped, the later conflict
    request reuses the comment event ID with different text.  Every outcome and
    correlated private conflict-audit record is permanent.
    """
    if timeout_seconds <= 0 or poll_interval <= 0:
        raise RoundtripConfigurationError("timeout and poll interval must be positive")
    if skip_conflict_probe and include_mixed_batch_probe:
        raise RoundtripConfigurationError(
            "--include-mixed-batch-probe cannot be combined with --skip-conflict-probe"
        )
    identity, event = build_smoke_event(run_id)
    batch_identity: SuccessfulBatchIdentity | None = None
    batch_trace: Mapping[str, Any] | None = None
    batch_preexisting = False
    if not skip_successful_batch_probe:
        batch_identity, batch_trace = build_successful_batch_trace(identity)
    try:
        initial_pages = _fetch_pages(reports_client, identity)
        if batch_identity is not None:
            initial_batch_pages = _fetch_pages(reports_client, batch_identity)
            batch_preexisting = not _preflight_is_empty(initial_batch_pages)
    except (ReportsError, RoundtripReportError) as exc:
        raise RoundtripPreflightError(f"report preflight failed: {exc}") from exc

    if not _preflight_is_empty(initial_pages):
        if not resume:
            raise RoundtripPreflightError(
                "the deterministic smoke namespace already contains report data; "
                "choose another run id or pass --resume"
            )
        try:
            assert_roundtrip_visible(initial_pages, identity)
        except RoundtripReportError as exc:
            raise RoundtripPreflightError(
                f"existing resume data is not the expected smoke event: {exc}"
            ) from exc

    if batch_identity is not None and batch_preexisting:
        if not resume:
            raise RoundtripPreflightError(
                "the deterministic successful-batch namespace already contains "
                "report data; choose another run id or pass --resume"
            )
        try:
            assert_successful_batch_visible(initial_batch_pages, batch_identity)
        except RoundtripReportError as exc:
            raise RoundtripPreflightError(
                "existing successful-batch resume data is not the expected "
                f"answer/comment trace: {exc}"
            ) from exc

    try:
        request_filter_preflight = _fetch_ingestion_outcome(
            reports_client,
            str(uuid.uuid4()),
        )
        assert_ingestion_outcome_absent(request_filter_preflight)
    except (ReportsError, RoundtripReportError) as exc:
        raise RoundtripPreflightError(
            f"ingestion request-id preflight failed: {exc}"
        ) from exc

    try:
        receipt = feedback_client.post_event(event)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(f"comment upload failed: {exc}") from exc
    request_id, accepted, duplicate = _receipt_expectation(receipt, resume=resume)
    negative_request_id = _negative_control_request_id(request_id)

    deadline = monotonic() + timeout_seconds
    polls = 0
    last_error = "event has not appeared yet"
    while True:
        polls += 1
        try:
            pages = _fetch_pages(reports_client, identity)
            assert_roundtrip_visible(pages, identity)
            outcome = _fetch_ingestion_outcome(reports_client, request_id)
            assert_ingestion_outcome(
                outcome,
                accepted=accepted,
                duplicate=duplicate,
            )
            negative_outcome = _fetch_ingestion_outcome(
                reports_client,
                negative_request_id,
            )
            assert_ingestion_outcome_absent(negative_outcome)
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    f"report endpoint rejected the verification query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            response = receipt.response
            safe_receipt = None
            if isinstance(response, Mapping):
                safe_receipt = {
                    name: response.get(name)
                    for name in (
                        "accepted",
                        "duplicate",
                        "conflict",
                        "rejected",
                        "request_id",
                    )
                    if name in response
                }
            break
        if monotonic() >= deadline:
            raise RoundtripReportError(
                f"reports did not converge before the timeout: {last_error}"
            )
        sleep(poll_interval)

    successful_batch_first_request_id: str | None = None
    successful_batch_replay_request_id: str | None = None
    successful_batch_verified = False
    successful_batch_first_write_verified = False
    if batch_identity is not None and batch_trace is not None:
        (
            successful_batch_first_request_id,
            successful_batch_replay_request_id,
            successful_batch_first_write_verified,
            batch_polls,
        ) = _run_successful_batch_probe(
            feedback_client,
            reports_client,
            identity=batch_identity,
            trace=batch_trace,
            preexisting=batch_preexisting,
            prior_request_ids={request_id},
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            monotonic=monotonic,
            sleep=sleep,
        )
        polls += batch_polls
        successful_batch_verified = True

    def result(
        *,
        conflict_request_id: str | None = None,
        conflict_verified: bool = False,
        mixed_batch_request_id: str | None = None,
        mixed_batch_verified: bool = False,
    ) -> RoundtripResult:
        return RoundtripResult(
            identity=identity,
            request_id=request_id,
            polls=polls,
            receipt=safe_receipt,
            conflict_request_id=conflict_request_id,
            conflict_verified=conflict_verified,
            mixed_batch_request_id=mixed_batch_request_id,
            mixed_batch_verified=mixed_batch_verified,
            successful_batch_first_request_id=(successful_batch_first_request_id),
            successful_batch_replay_request_id=(successful_batch_replay_request_id),
            successful_batch_verified=successful_batch_verified,
            successful_batch_first_write_verified=(
                successful_batch_first_write_verified
            ),
            verified_at=utc_now(),
            authority_mode=AUTHORITY_MODE_LEGACY,
        )

    if skip_conflict_probe:
        return result()

    conflict_event = _conflicting_event(event)
    try:
        feedback_client.post_event(conflict_event)
    except feedback.FeedbackUploadConflictError as exc:
        conflict_request_id = _conflict_expectation(exc)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(f"conflict probe upload failed: {exc}") from exc
    else:
        raise RoundtripUploadError(
            "conflict probe was not rejected with FeedbackUploadConflictError"
        )
    prior_success_request_ids = {
        request_id,
        successful_batch_first_request_id,
        successful_batch_replay_request_id,
    }
    if conflict_request_id in prior_success_request_ids:
        raise RoundtripUploadError(
            "conflict probe reused the successful request id from an earlier "
            "phase instead of issuing a new UUID"
        )

    conflict_deadline = monotonic() + timeout_seconds
    last_error = "conflict outcome has not appeared yet"
    while True:
        polls += 1
        try:
            # Exact business-view parity after the rejected write proves that
            # the original comment remains the sole stored logical event.
            pages = _fetch_pages(reports_client, identity)
            assert_roundtrip_visible(pages, identity)
            conflict_outcome = _fetch_ingestion_outcome(
                reports_client,
                conflict_request_id,
            )
            assert_conflict_ingestion_outcome(conflict_outcome)
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    f"report endpoint rejected the conflict verification query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            break
        if monotonic() >= conflict_deadline:
            raise RoundtripReportError(
                f"conflict reports did not converge before the timeout: {last_error}"
            )
        sleep(poll_interval)

    if not include_mixed_batch_probe:
        return result(
            conflict_request_id=conflict_request_id,
            conflict_verified=True,
        )

    mixed_batch_trace = _mixed_batch_trace(identity, event)
    try:
        feedback_client.post_trace(mixed_batch_trace)
    except feedback.FeedbackUploadConflictError as exc:
        mixed_batch_request_id = _conflict_expectation(exc, rejected=2)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(f"mixed-batch probe upload failed: {exc}") from exc
    else:
        raise RoundtripUploadError(
            "mixed-batch probe was not rejected with FeedbackUploadConflictError"
        )
    if mixed_batch_request_id in prior_success_request_ids | {conflict_request_id}:
        raise RoundtripUploadError(
            "mixed-batch probe reused an earlier request id instead of issuing "
            "a new UUID"
        )

    mixed_deadline = monotonic() + timeout_seconds
    last_error = "mixed-batch outcome has not appeared yet"
    while True:
        polls += 1
        try:
            # One unchanged business row proves both that the conflicting event
            # did not overwrite the first write and that the new event was
            # withheld when its batch peer conflicted.
            pages = _fetch_pages(reports_client, identity)
            assert_roundtrip_visible(pages, identity)
            mixed_outcome = _fetch_ingestion_outcome(
                reports_client,
                mixed_batch_request_id,
            )
            assert_mixed_batch_ingestion_outcome(mixed_outcome)
        except ReportsRequestError as exc:
            if exc.status_code in {400, 401, 403}:
                raise RoundtripReportError(
                    "report endpoint rejected the mixed-batch verification "
                    f"query: {exc}"
                ) from exc
            last_error = str(exc)
        except (ReportsError, RoundtripReportError) as exc:
            last_error = str(exc)
        else:
            return result(
                conflict_request_id=conflict_request_id,
                conflict_verified=True,
                mixed_batch_request_id=mixed_batch_request_id,
                mixed_batch_verified=True,
            )
        if monotonic() >= mixed_deadline:
            raise RoundtripReportError(
                f"mixed-batch reports did not converge before the timeout: {last_error}"
            )
        sleep(poll_interval)


def _run_authoritative_roundtrip(
    feedback_client: feedback.FeedbackClient,
    reports_client: ReportsClient,
    *,
    run_id: str,
    probe_question: AuthoritativeProbeQuestion,
    resume: bool = False,
    skip_successful_batch_probe: bool = False,
    skip_conflict_probe: bool = False,
    include_mixed_batch_probe: bool = False,
    timeout_seconds: float = 90.0,
    poll_interval: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], str] = _utc_now,
) -> RoundtripResult:
    """Verify hosted ingestion against server-derived registry authority."""
    if timeout_seconds <= 0 or poll_interval <= 0:
        raise RoundtripConfigurationError("timeout and poll interval must be positive")
    if skip_conflict_probe and include_mixed_batch_probe:
        raise RoundtripConfigurationError(
            "--include-mixed-batch-probe cannot be combined with --skip-conflict-probe"
        )
    _validate_authoritative_probe_metadata(probe_question)
    identity, event = build_smoke_event(
        run_id,
        probe_question=probe_question,
    )
    batch_identity: SuccessfulBatchIdentity | None = None
    batch_trace: Mapping[str, Any] | None = None
    if not skip_successful_batch_probe:
        batch_identity, batch_trace = build_successful_batch_trace(
            identity,
            probe_question=probe_question,
        )
    withheld_event_id = f"{identity.event_id}.mixed-withheld"

    def preflight_event(
        event_id: str, event_type: str
    ) -> tuple[bool, dict[str, Any] | None]:
        page = _fetch_event_resolution(reports_client, event_id)
        row = _event_resolution_row(page, event_id=event_id)
        if row.get("registry_status") == "not_found":
            assert_event_resolution_not_found(page, event_id=event_id)
            return False, None
        return True, assert_authoritative_event_resolution(
            page,
            event_id=event_id,
            event_type=event_type,
            session_id=identity.session_id,
            attempt_id=identity.attempt_id,
            probe_question=probe_question,
        )

    try:
        assert_authority_status(
            _fetch_authority_status(reports_client),
            probe_question=probe_question,
        )
        assert_detail_report_surfaces(reports_client)
        snapshot_missing_question_id = (
            f"verifier_snapshot_missing_{secrets.token_hex(16)}"
        )
        assert_empty_business_snapshot(
            reports_client.fetch_business_snapshot(
                filters={"question_id": snapshot_missing_question_id},
                limit=1,
            ),
            probe_question=probe_question,
        )
        main_preexisting, _ = preflight_event(
            identity.event_id,
            "comment_submitted",
        )
        if main_preexisting and not resume:
            raise RoundtripPreflightError(
                "the deterministic authoritative event already exists; choose "
                "another run id or pass --resume"
            )
        if resume and not main_preexisting:
            raise RoundtripPreflightError(
                "authoritative --resume requires the exact original event to exist"
            )
        batch_preexisting = False
        if batch_identity is not None:
            if batch_identity.proposal_event_id is None:
                raise RoundtripPreflightError(
                    "authoritative successful batch has no proposal identity"
                )
            answer_exists, answer_preflight_row = preflight_event(
                batch_identity.answer_event_id,
                "answer_submitted",
            )
            proposal_exists, proposal_preflight_row = preflight_event(
                batch_identity.proposal_event_id,
                "custom_setting_proposed",
            )
            comment_exists, comment_preflight_row = preflight_event(
                batch_identity.comment_event_id,
                "comment_submitted",
            )
            if len({answer_exists, proposal_exists, comment_exists}) != 1:
                raise RoundtripPreflightError(
                    "authoritative successful-batch preflight found a partial batch"
                )
            batch_preexisting = answer_exists
            if batch_preexisting and not resume:
                raise RoundtripPreflightError(
                    "the deterministic authoritative successful batch already "
                    "exists; choose another run id or pass --resume"
                )
            if batch_preexisting:
                if batch_trace is None or any(
                    row is None
                    for row in (
                        answer_preflight_row,
                        proposal_preflight_row,
                        comment_preflight_row,
                    )
                ):
                    raise RoundtripPreflightError(
                        "authoritative successful-batch resume has no stored times"
                    )
                stored_rows = {
                    batch_identity.answer_event_id: answer_preflight_row,
                    batch_identity.proposal_event_id: proposal_preflight_row,
                    batch_identity.comment_event_id: comment_preflight_row,
                }
                aligned_trace = deepcopy(dict(batch_trace))
                aligned_events = aligned_trace.get("events")
                if not isinstance(aligned_events, list):
                    raise RoundtripPreflightError(
                        "authoritative successful-batch trace has no event list"
                    )
                for aligned_event in aligned_events:
                    if not isinstance(aligned_event, dict):
                        raise RoundtripPreflightError(
                            "authoritative successful-batch trace has invalid events"
                        )
                    stored = stored_rows.get(aligned_event.get("event_id"))
                    if not isinstance(stored, Mapping) or not isinstance(
                        stored.get("occurred_at"), str
                    ):
                        raise RoundtripPreflightError(
                            "authoritative successful-batch resume is missing an "
                            "exact stored occurrence time"
                        )
                    aligned_event["occurred_at"] = stored["occurred_at"]
                batch_trace = aligned_trace
        if include_mixed_batch_probe:
            withheld_page = _fetch_event_resolution(
                reports_client,
                withheld_event_id,
            )
            assert_event_resolution_not_found(
                withheld_page,
                event_id=withheld_event_id,
            )
        request_filter_preflight = _fetch_ingestion_outcome(
            reports_client,
            str(uuid.uuid4()),
        )
        assert_ingestion_outcome_absent(request_filter_preflight)
    except RoundtripPreflightError:
        raise
    except (ReportsError, RoundtripReportError) as exc:
        raise RoundtripPreflightError(
            f"authoritative exact-event preflight failed: {exc}"
        ) from exc

    try:
        receipt = feedback_client.post_event(event)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"authoritative comment upload failed: {exc}"
        ) from exc
    request_id, accepted, duplicate = _receipt_expectation(receipt, resume=resume)
    negative_request_id = _negative_control_request_id(request_id)

    def assert_main_outcome(page: ReportPage) -> None:
        assert_ingestion_outcome(
            page,
            accepted=accepted,
            duplicate=duplicate,
        )

    polls, main_rows = _poll_authoritative_state(
        reports_client,
        probe_question=probe_question,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        expected_events={identity.event_id: "comment_submitted"},
        request_id=request_id,
        outcome_assertion=assert_main_outcome,
        negative_request_id=negative_request_id,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="authoritative comment",
    )
    response = receipt.response
    safe_receipt = None
    if isinstance(response, Mapping):
        safe_receipt = {
            name: response.get(name)
            for name in (
                "accepted",
                "duplicate",
                "conflict",
                "rejected",
                "request_id",
            )
            if name in response
        }

    successful_batch_first_request_id: str | None = None
    successful_batch_replay_request_id: str | None = None
    successful_batch_verified = False
    successful_batch_first_write_verified = False
    detail_reports_verified = False
    session_attempt_filters_verified = False
    if batch_identity is not None and batch_trace is not None:
        (
            successful_batch_first_request_id,
            successful_batch_replay_request_id,
            successful_batch_first_write_verified,
            detail_reports_verified,
            batch_polls,
        ) = _run_authoritative_successful_batch_probe(
            feedback_client,
            reports_client,
            identity=batch_identity,
            trace=batch_trace,
            probe_question=probe_question,
            preexisting=batch_preexisting,
            prior_request_ids={request_id},
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            monotonic=monotonic,
            sleep=sleep,
        )
        polls += batch_polls
        successful_batch_verified = True
        polls += _poll_session_attempt_business_snapshots(
            reports_client,
            identity=identity,
            batch_identity=batch_identity,
            probe_question=probe_question,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            monotonic=monotonic,
            sleep=sleep,
        )
        session_attempt_filters_verified = True

    def result(
        *,
        conflict_request_id: str | None = None,
        conflict_verified: bool = False,
        mixed_batch_request_id: str | None = None,
        mixed_batch_verified: bool = False,
    ) -> RoundtripResult:
        return RoundtripResult(
            identity=identity,
            request_id=request_id,
            polls=polls,
            receipt=safe_receipt,
            conflict_request_id=conflict_request_id,
            conflict_verified=conflict_verified,
            mixed_batch_request_id=mixed_batch_request_id,
            mixed_batch_verified=mixed_batch_verified,
            successful_batch_first_request_id=(successful_batch_first_request_id),
            successful_batch_replay_request_id=(successful_batch_replay_request_id),
            successful_batch_verified=successful_batch_verified,
            successful_batch_first_write_verified=(
                successful_batch_first_write_verified
            ),
            registry_id=probe_question.registry_id,
            authority_status_verified=True,
            detail_reports_verified=detail_reports_verified,
            business_snapshot_verified=True,
            session_attempt_filters_verified=session_attempt_filters_verified,
            verified_at=utc_now(),
            manifest_sha256=probe_question.manifest_sha256,
            registry_question_count=probe_question.registry_question_count,
            registry_choice_count=probe_question.registry_choice_count,
            authority_mode=AUTHORITY_MODE_AUTHORITATIVE,
        )

    if skip_conflict_probe:
        return result()

    conflict_event = _conflicting_event(event)
    try:
        feedback_client.post_event(conflict_event)
    except feedback.FeedbackUploadConflictError as exc:
        conflict_request_id = _conflict_expectation(exc)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"authoritative conflict probe upload failed: {exc}"
        ) from exc
    else:
        raise RoundtripUploadError(
            "authoritative conflict probe was not rejected with "
            "FeedbackUploadConflictError"
        )
    prior_request_ids = {
        request_id,
        successful_batch_first_request_id,
        successful_batch_replay_request_id,
    }
    if conflict_request_id in prior_request_ids:
        raise RoundtripUploadError(
            "authoritative conflict probe reused an earlier successful request id"
        )

    def assert_conflict_outcome(page: ReportPage) -> None:
        assert_conflict_ingestion_outcome(page)

    conflict_polls, _ = _poll_authoritative_state(
        reports_client,
        probe_question=probe_question,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        expected_events={identity.event_id: "comment_submitted"},
        unchanged_rows=main_rows,
        request_id=conflict_request_id,
        outcome_assertion=assert_conflict_outcome,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="authoritative conflict",
    )
    polls += conflict_polls
    if not include_mixed_batch_probe:
        return result(
            conflict_request_id=conflict_request_id,
            conflict_verified=True,
        )

    mixed_batch_trace = _mixed_batch_trace(identity, event)
    try:
        feedback_client.post_trace(mixed_batch_trace)
    except feedback.FeedbackUploadConflictError as exc:
        mixed_batch_request_id = _conflict_expectation(exc, rejected=2)
    except feedback.FeedbackError as exc:
        raise RoundtripUploadError(
            f"authoritative mixed-batch probe upload failed: {exc}"
        ) from exc
    else:
        raise RoundtripUploadError(
            "authoritative mixed-batch probe was not rejected with "
            "FeedbackUploadConflictError"
        )
    if mixed_batch_request_id in prior_request_ids | {conflict_request_id}:
        raise RoundtripUploadError(
            "authoritative mixed-batch probe reused an earlier request id"
        )

    def assert_mixed_outcome(page: ReportPage) -> None:
        assert_mixed_batch_ingestion_outcome(page)

    mixed_polls, _ = _poll_authoritative_state(
        reports_client,
        probe_question=probe_question,
        session_id=identity.session_id,
        attempt_id=identity.attempt_id,
        expected_events={identity.event_id: "comment_submitted"},
        absent_event_ids=(withheld_event_id,),
        unchanged_rows=main_rows,
        request_id=mixed_batch_request_id,
        outcome_assertion=assert_mixed_outcome,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        monotonic=monotonic,
        sleep=sleep,
        phase_name="authoritative mixed-batch",
    )
    polls += mixed_polls
    return result(
        conflict_request_id=conflict_request_id,
        conflict_verified=True,
        mixed_batch_request_id=mixed_batch_request_id,
        mixed_batch_verified=True,
    )


def run_roundtrip(
    feedback_client: feedback.FeedbackClient,
    reports_client: ReportsClient,
    *,
    run_id: str,
    probe_question: AuthoritativeProbeQuestion | None = None,
    resume: bool = False,
    skip_successful_batch_probe: bool = False,
    skip_conflict_probe: bool = False,
    include_mixed_batch_probe: bool = False,
    timeout_seconds: float = 90.0,
    poll_interval: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], str] = _utc_now,
) -> RoundtripResult:
    """Run hosted verification and capture one stable completion timestamp."""
    common = {
        "run_id": run_id,
        "resume": resume,
        "skip_successful_batch_probe": skip_successful_batch_probe,
        "skip_conflict_probe": skip_conflict_probe,
        "include_mixed_batch_probe": include_mixed_batch_probe,
        "timeout_seconds": timeout_seconds,
        "poll_interval": poll_interval,
        "monotonic": monotonic,
        "sleep": sleep,
        "utc_now": utc_now,
    }
    if probe_question is None:
        return _run_legacy_roundtrip(
            feedback_client,
            reports_client,
            **common,
        )
    result = _run_authoritative_roundtrip(
        feedback_client,
        reports_client,
        probe_question=probe_question,
        **common,
    )
    _validate_authoritative_result(result, probe_question)
    return result


def _host_is_loopback(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _require_safe_configs() -> tuple[feedback.FeedbackConfig, ReportsConfig]:
    feedback_config = feedback.FeedbackConfig.from_env()
    reports_config = ReportsConfig.from_env()
    if not feedback_config.is_configured or not feedback_config.bearer_token:
        raise RoundtripConfigurationError(
            "feedback endpoint and Bearer token must both be configured"
        )
    reports_config.require_configured()
    assert feedback_config.endpoint is not None
    assert reports_config.url is not None
    if feedback_config.bearer_token == reports_config.read_token:
        raise RoundtripConfigurationError("ingest and report tokens must be different")
    for name, url in (
        ("feedback", feedback_config.endpoint),
        ("reports", reports_config.url),
    ):
        if urllib.parse.urlsplit(url).scheme != "https" and not _host_is_loopback(url):
            raise RoundtripConfigurationError(
                f"{name} endpoint must use HTTPS except for loopback testing"
            )
    return feedback_config, reports_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-permanent-write",
        action="store_true",
        help=(
            "Acknowledge that the probe events, ingestion outcomes, and "
            "private conflict-audit records cannot be deleted by this tool."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Deterministic smoke identifier; generated when omitted.",
    )
    parser.add_argument(
        "--bundle",
        default=str(DEFAULT_BUNDLE_PATH),
        help=(
            "Fully attested quiz bundle whose deterministic registry membership "
            "will be used (default: examples/quiz_demo/bundle)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retry an existing deterministic run after validating its exact identity.",
    )
    parser.add_argument(
        "--skip-successful-batch-probe",
        action="store_true",
        help=(
            "Recovery/minimal-footprint only: skip the accepted/replay batch "
            "probe and mark successful-batch and detail-row proofs false."
        ),
    )
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--skip-conflict-probe",
        action="store_true",
        help=(
            "Compatibility/recovery only: skip the intentional 409 probe and mark "
            "conflict verification false."
        ),
    )
    probe_group.add_argument(
        "--include-mixed-batch-probe",
        action="store_true",
        help=(
            "After the normal and single-conflict checks, permanently submit a "
            "two-event conflict/new-event trace to verify all-or-none ingestion."
        ),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _redacted_message(error: BaseException, secrets_to_hide: tuple[str, ...]) -> str:
    message = str(error)
    for secret in secrets_to_hide:
        if secret:
            message = message.replace(secret, "[REDACTED]")
            message = message.replace(
                urllib.parse.quote(secret, safe=""),
                "[REDACTED]",
            )
    return message


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    secrets_to_hide: tuple[str, ...] = ()
    try:
        feedback_config, reports_config = _require_safe_configs()
        secrets_to_hide = (
            feedback_config.bearer_token or "",
            reports_config.read_token or "",
        )
        if args.resume and not args.run_id:
            raise RoundtripConfigurationError("--resume requires --run-id")
        run_id = _validate_run_id(args.run_id or secrets.token_hex(8))
        if not args.confirm_permanent_write:
            raise RoundtripConfigurationError(
                "configuration is valid; rerun with --confirm-permanent-write "
                "to append registered-membership probe events and, by default, a "
                "permanent conflict rejection/audit footprint (prefer staging)"
            )
        probe_question = load_authoritative_probe_question(args.bundle)
        print(
            f"Hosted roundtrip run id (save for --resume): {run_id}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Authoritative registry membership: "
            f"registry={probe_question.registry_id} "
            f"release={probe_question.release_id} "
            f"question={probe_question.question_id}",
            file=sys.stderr,
            flush=True,
        )
        if args.skip_successful_batch_probe:
            print(
                "Successful multi-event batch probe explicitly skipped; both "
                "successful-batch verification fields will be false.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "The default successful-batch probe submits an answer/proposed-"
                "setting/comment trace and replays the identical three-event "
                "envelope. Both permanent request outcomes and both detail rows "
                "must be independently visible.",
                file=sys.stderr,
                flush=True,
            )
        if args.skip_conflict_probe:
            print(
                "Conflict probe explicitly skipped; conflict_verified will be false.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "The default probe intentionally submits different content under "
                "the same event ID. Its 409 outcome is permanent and the backend "
                "retains a private conflict-audit record.",
                file=sys.stderr,
                flush=True,
            )
        if args.include_mixed_batch_probe:
            print(
                "The opt-in mixed-batch probe permanently records another 409 "
                "outcome and private conflict-audit row. Its fresh second event "
                "must be withheld by all-or-none ingestion.",
                file=sys.stderr,
                flush=True,
            )
        result = run_roundtrip(
            feedback.FeedbackClient(feedback_config),
            ReportsClient(reports_config),
            run_id=run_id,
            probe_question=probe_question,
            resume=args.resume,
            skip_successful_batch_probe=args.skip_successful_batch_probe,
            skip_conflict_probe=args.skip_conflict_probe,
            include_mixed_batch_probe=args.include_mixed_batch_probe,
            timeout_seconds=args.timeout,
            poll_interval=args.poll_interval,
        )
        _validate_authoritative_result(result, probe_question)
    except RoundtripConfigurationError as exc:
        print(
            f"configuration/preflight error: {_redacted_message(exc, secrets_to_hide)}",
            file=sys.stderr,
        )
        return 2
    except RoundtripPreflightError as exc:
        print(
            f"preflight error: {_redacted_message(exc, secrets_to_hide)}",
            file=sys.stderr,
        )
        return 2
    except RoundtripUploadError as exc:
        print(
            f"upload error: {_redacted_message(exc, secrets_to_hide)}", file=sys.stderr
        )
        return 3
    except RoundtripReportError as exc:
        print(
            f"report error: {_redacted_message(exc, secrets_to_hide)}", file=sys.stderr
        )
        return 4
    except (feedback.FeedbackError, ReportsError, ValueError) as exc:
        print(
            f"verification error: {_redacted_message(exc, secrets_to_hide)}",
            file=sys.stderr,
        )
        return 2

    payload = result.to_dict()
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "Hosted feedback roundtrip passed: "
            f"run={payload['run_id']} registry={payload['registry_id']} "
            f"release={payload['release_id']} "
            f"question={payload['question_id']} polls={payload['polls']}"
        )
        if payload["registry_id"]:
            print(f"Registry id: {payload['registry_id']}")
        print(f"Authority status verified: {payload['authority_status_verified']}")
        print(f"Detail reports verified: {payload['detail_reports_verified']}")
        print(f"Business snapshot verified: {payload['business_snapshot_verified']}")
        print(
            "Session/attempt filters verified: "
            f"{payload['session_attempt_filters_verified']}"
        )
        if payload["request_id"]:
            print(f"Ingestion request id: {payload['request_id']}")
        if payload["successful_batch_first_request_id"]:
            print(
                "Successful-batch first request id: "
                f"{payload['successful_batch_first_request_id']}"
            )
        if payload["successful_batch_replay_request_id"]:
            print(
                "Successful-batch replay request id: "
                f"{payload['successful_batch_replay_request_id']}"
            )
        print(
            f"Successful-batch probe verified: {payload['successful_batch_verified']}"
        )
        print(
            "Successful-batch first write verified: "
            f"{payload['successful_batch_first_write_verified']}"
        )
        if payload["conflict_request_id"]:
            print(f"Conflict request id: {payload['conflict_request_id']}")
        print(f"Conflict probe verified: {payload['conflict_verified']}")
        if payload["mixed_batch_request_id"]:
            print(f"Mixed-batch request id: {payload['mixed_batch_request_id']}")
        print(f"Mixed-batch probe verified: {payload['mixed_batch_verified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
