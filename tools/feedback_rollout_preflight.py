#!/usr/bin/env python3
"""Offline, read-only preflight for the staged feedback-data rollout.

This tool intentionally proves only repository-local static contracts.  It does
not accept credentials or endpoints, contact a hosted service, mutate files, or
deploy anything.  Hosted acceptance therefore always remains UNVERIFIED.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:
    from quiz_bundle.feedback_registry import (
        FeedbackRegistryError,
        build_feedback_registry,
        render_feedback_registry_sql,
        serialize_feedback_registry,
    )
except ModuleNotFoundError:
    from tools.quiz_bundle.feedback_registry import (
        FeedbackRegistryError,
        build_feedback_registry,
        render_feedback_registry_sql,
        serialize_feedback_registry,
    )


PASS: Final = "PASS"
FAIL: Final = "FAIL"
UNVERIFIED: Final = "UNVERIFIED"

PHASES: Final = (
    "expand",
    "ingest-cutover",
    "lockdown-report",
    "report-app",
)

PREFLIGHT_TOOL: Final = "tools/feedback_rollout_preflight.py"
BASELINE_EVENT_MIGRATION: Final = (
    "supabase/migrations/20260711000000_feedback_events.sql"
)
BASELINE_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712000000_feedback_reports.sql"
)
BASELINE_OBSERVABILITY_MIGRATION: Final = (
    "supabase/migrations/20260712010000_feedback_ingest_observability.sql"
)
BASELINE_OBSERVABILITY_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712011000_feedback_ingest_observability_report.sql"
)
EXPAND_MIGRATION: Final = (
    "supabase/migrations/20260712012000_feedback_event_conflicts.sql"
)
QUESTION_REACTION_MIGRATION: Final = (
    "supabase/migrations/20260712019000_question_reactions.sql"
)
SURPRISE_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712020000_feedback_surprise_report.sql"
)
INGEST_EDGE: Final = "supabase/functions/feedback-ingest/index.ts"
INSPECTOR_FEEDBACK: Final = "tools/question_inspector/feedback.py"
INSPECTOR_RECOMMENDER: Final = "tools/question_inspector/surprise_recommender.py"
INSPECTOR_SURPRISE_CATALOG: Final = "tools/question_inspector/surprise_catalog.py"
HOSTED_ROUNDTRIP_VERIFIER: Final = "tools/feedback_reports/verify_hosted_roundtrip.py"
POSTGRES_ACCEPTANCE_VERIFIER: Final = "tools/feedback_postgres_acceptance.py"
DEPLOYMENT_LEDGER_TOOL: Final = "tools/deployment_ledger.py"
DEPLOYMENT_LEDGER_README: Final = "deployments/README.md"
DEPLOYMENT_LEDGER_JOURNAL: Final = "deployments/ledger.jsonl"
DEPLOYMENT_LEDGER_EVIDENCE_PREFIX: Final = "deployments/evidence/"
LOCKDOWN_MIGRATION: Final = (
    "supabase/migrations/20260712012500_feedback_event_writer_lockdown.sql"
)
CONFLICT_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712013000_feedback_conflict_observability_report.sql"
)
RAW_VIEW_HARDENING_MIGRATION: Final = (
    "supabase/migrations/20260712013500_feedback_raw_view_hardening.sql"
)
QUESTION_REGISTRY_MIGRATION: Final = (
    "supabase/migrations/20260712014000_feedback_question_registry.sql"
)
QUESTION_REGISTRY_DATA_MIGRATION: Final = (
    "supabase/migrations/20260712014500_feedback_question_registry_release_4e752a.sql"
)
AUTHORITATIVE_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712015000_feedback_authoritative_reports.sql"
)
DETAIL_REPORT_MIGRATION: Final = (
    "supabase/migrations/20260712016000_feedback_detail_reports.sql"
)
BUSINESS_SNAPSHOT_MIGRATION: Final = (
    "supabase/migrations/20260712017000_feedback_business_snapshot.sql"
)
SESSION_ATTEMPT_FILTER_MIGRATION: Final = (
    "supabase/migrations/20260712018000_feedback_session_attempt_filters.sql"
)
QUESTION_REGISTRY_JSON: Final = (
    "supabase/registries/"
    "release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f.json"
)
QUESTION_REGISTRY_EXPORTER: Final = "tools/quiz_bundle/feedback_registry.py"
QUESTION_REGISTRY_CLI: Final = "tools/export_feedback_registry.py"
QUIZ_BUNDLE_PUBLISHER: Final = "tools/quiz_bundle/publisher.py"
QUIZ_BUNDLE_VERSIONING: Final = "tools/quiz_bundle/versioning.py"
QUIZ_BUNDLE_MANIFEST: Final = "examples/quiz_demo/bundle/quiz_manifest.json"
QUIZ_BUNDLE_ROOT: Final = "examples/quiz_demo/bundle"
REPORT_EDGE: Final = "supabase/functions/feedback-report/index.ts"
REPORT_QUERY: Final = "supabase/functions/feedback-report/report_query.ts"
REPORT_CLIENT: Final = "tools/feedback_reports/client.py"
REPORT_PACKAGE: Final = "tools/feedback_reports/__init__.py"
REPORT_APP: Final = "tools/feedback_reports/app.py"
REPORT_UI: Final = "tools/feedback_reports/ui.py"
INSPECTOR_APP: Final = "tools/question_inspector/app.py"
INSPECTOR_OUTBOX: Final = "tools/question_inspector/feedback_outbox.py"
INSPECTOR_RECOVERY: Final = "tools/question_inspector/feedback_recovery.py"
INSPECTOR_RELEASE_MANIFEST: Final = "tools/question_inspector/release_manifest.py"
REQUIREMENTS: Final = "requirements.txt"
PROJECT_METADATA: Final = "pyproject.toml"
PROJECT_README: Final = "README.md"

_PHASE_ADDITIONS: Final[dict[str, tuple[str, ...]]] = {
    # Baselines are compatibility inputs, not an instruction to reapply them
    # to an existing receiver.  Hosted prerequisite state remains unverified.
    "expand": (
        PREFLIGHT_TOOL,
        BASELINE_EVENT_MIGRATION,
        BASELINE_REPORT_MIGRATION,
        BASELINE_OBSERVABILITY_MIGRATION,
        BASELINE_OBSERVABILITY_REPORT_MIGRATION,
        EXPAND_MIGRATION,
    ),
    "ingest-cutover": (
        INGEST_EDGE,
        INSPECTOR_FEEDBACK,
        HOSTED_ROUNDTRIP_VERIFIER,
    ),
    # The strict Python client is a compatibility input for the 13000 schema
    # boundary even though the complete report application is checked next.
    "lockdown-report": (
        LOCKDOWN_MIGRATION,
        CONFLICT_REPORT_MIGRATION,
        RAW_VIEW_HARDENING_MIGRATION,
        QUESTION_REGISTRY_MIGRATION,
        QUESTION_REGISTRY_EXPORTER,
        QUESTION_REGISTRY_CLI,
        QUIZ_BUNDLE_PUBLISHER,
        QUIZ_BUNDLE_VERSIONING,
        QUIZ_BUNDLE_MANIFEST,
        QUESTION_REGISTRY_JSON,
        QUESTION_REGISTRY_DATA_MIGRATION,
        AUTHORITATIVE_REPORT_MIGRATION,
        DETAIL_REPORT_MIGRATION,
        BUSINESS_SNAPSHOT_MIGRATION,
        SESSION_ATTEMPT_FILTER_MIGRATION,
        QUESTION_REACTION_MIGRATION,
        SURPRISE_REPORT_MIGRATION,
        REPORT_CLIENT,
    ),
    "report-app": (
        POSTGRES_ACCEPTANCE_VERIFIER,
        # The verifier and its contract are deploy inputs.  Runtime JSONL and
        # evidence are retrospective outputs and must never fingerprint the
        # candidate they later describe.
        DEPLOYMENT_LEDGER_TOOL,
        DEPLOYMENT_LEDGER_README,
        REPORT_EDGE,
        REPORT_QUERY,
        REPORT_PACKAGE,
        REPORT_APP,
        REPORT_UI,
        INSPECTOR_APP,
        INSPECTOR_OUTBOX,
        INSPECTOR_RECOVERY,
        INSPECTOR_RELEASE_MANIFEST,
        INSPECTOR_RECOMMENDER,
        INSPECTOR_SURPRISE_CATALOG,
        REQUIREMENTS,
        PROJECT_METADATA,
        PROJECT_README,
    ),
}

EXPECTED_MIGRATION_INVENTORY: Final = (
    Path(BASELINE_EVENT_MIGRATION).name,
    Path(BASELINE_REPORT_MIGRATION).name,
    Path(BASELINE_OBSERVABILITY_MIGRATION).name,
    Path(BASELINE_OBSERVABILITY_REPORT_MIGRATION).name,
    Path(EXPAND_MIGRATION).name,
    Path(LOCKDOWN_MIGRATION).name,
    Path(CONFLICT_REPORT_MIGRATION).name,
    Path(RAW_VIEW_HARDENING_MIGRATION).name,
    Path(QUESTION_REGISTRY_MIGRATION).name,
    Path(QUESTION_REGISTRY_DATA_MIGRATION).name,
    Path(AUTHORITATIVE_REPORT_MIGRATION).name,
    Path(DETAIL_REPORT_MIGRATION).name,
    Path(BUSINESS_SNAPSHOT_MIGRATION).name,
    Path(SESSION_ATTEMPT_FILTER_MIGRATION).name,
    Path(QUESTION_REACTION_MIGRATION).name,
    Path(SURPRISE_REPORT_MIGRATION).name,
)

_INGEST_RPC_COLUMNS: Final = (
    "requested_event_count",
    "new_event_count",
    "accepted_event_count",
    "duplicate_event_count",
    "conflicting_event_count",
    "rejected_event_count",
    "committed",
)

_EXPECTED_REPORT_VIEWS: Final = (
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

_REGISTRY_QUALITY_COLUMNS: Final = (
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

_AUTHORITY_STATUS_COLUMNS: Final = (
    "authority_revision",
    "business_reports_authoritative",
    "registered_release_count",
    "registered_question_count",
    "registered_choice_count",
)

_FINAL_AUTHORITY_STATUS_COLUMNS: Final = (
    *_AUTHORITY_STATUS_COLUMNS,
    "detail_revision",
    "detail_reports_authoritative",
)

_BUSINESS_SNAPSHOT_COLUMNS: Final = (
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

_EVENT_RESOLUTION_COLUMNS: Final = (
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

_ANSWER_COLUMNS: Final = (
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

_PROPOSAL_COLUMNS: Final = (
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

_RAW_SESSION_VIEW_COLUMNS: Final = (
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
    "correct_answer_count",
    "accuracy",
    "proposal_count",
    "rejected_setting_count",
    "completed_run_count",
    "failed_run_count",
    "comment_count",
    "known_answer_count",
    "incorrect_answer_count",
    "unknown_answer_count",
)
_RAW_QUESTION_VIEW_COLUMNS: Final = (
    "question_id",
    "question_version",
    "release_id",
    "family",
    "dataset_id",
    "question_type",
    "first_event_at",
    "last_event_at",
    "session_count",
    "attempt_count",
    "answer_count",
    "correct_answer_count",
    "accuracy",
    "proposal_count",
    "rejected_setting_count",
    "completed_run_count",
    "failed_run_count",
    "comment_count",
    "known_answer_count",
    "incorrect_answer_count",
    "unknown_answer_count",
)
_RAW_PROPOSAL_VIEW_COLUMNS: Final = (
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
    "setting_status",
    "label",
    "setting",
    "inherited_from",
    "n_seeds",
    "base_seed",
)


@dataclass(frozen=True)
class Check:
    """One non-secret preflight finding."""

    code: str
    status: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "status": self.status,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class GitCommandResult:
    """Minimal, injectable result for a read-only Git invocation."""

    returncode: int
    stdout: bytes


GitRunner = Callable[[Path, tuple[str, ...]], GitCommandResult]


@dataclass(frozen=True)
class RegistryAttestationEvidence:
    """Non-secret result of rebuilding the registry from the full quiz bundle."""

    release_id: str | None
    json_sha256: str | None
    sql_sha256: str | None
    failed: bool


@dataclass(frozen=True)
class GitEvidence:
    """Sanitized repository evidence used by the pure evaluator."""

    sha: str | None
    tracked_paths: frozenset[str]
    dirty_paths: frozenset[str]
    head_blob_sha256: tuple[tuple[str, str], ...] = ()
    migration_inventory: tuple[str, ...] = ()
    failed_queries: frozenset[str] = frozenset()
    registry_attestation: RegistryAttestationEvidence | None = None


@dataclass(frozen=True)
class PreflightResult:
    """Complete local-static result; it never claims hosted readiness."""

    phase: str
    checks: tuple[Check, ...]
    git_sha: str | None
    checked_rollout_input_sha256: str | None
    require_hosted: bool

    @property
    def static_overall(self) -> str:
        return FAIL if any(check.status == FAIL for check in self.checks) else PASS

    @property
    def overall(self) -> str:
        if self.static_overall == FAIL:
            return FAIL
        if any(check.status == UNVERIFIED for check in self.checks):
            return UNVERIFIED
        return PASS

    @property
    def exit_code(self) -> int:
        if self.static_overall == FAIL:
            return 1
        if self.require_hosted and self.overall == UNVERIFIED:
            return 2
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "scope": "local_static",
            "rollout_contract": "staged_upgrade",
            "fingerprint_scope": "enumerated_repository_rollout_inputs",
            "baseline_migrations_are_compatibility_inputs": True,
            "phase": self.phase,
            "static_overall": self.static_overall,
            "overall": self.overall,
            "hosted_verified": False,
            "deploy_ready": False,
            "require_hosted": self.require_hosted,
            "git_sha": self.git_sha,
            "checked_rollout_input_paths": list(deployment_paths_for_phase(self.phase)),
            "checked_rollout_input_sha256": self.checked_rollout_input_sha256,
            "checks": [check.to_dict() for check in self.checks],
        }


def deployment_paths_for_phase(phase: str) -> tuple[str, ...]:
    """Return ordered, cumulative deployment/compatibility inputs."""

    if phase not in PHASES:
        raise ValueError(f"unsupported rollout phase: {phase}")
    paths: list[str] = []
    for candidate in PHASES:
        paths.extend(_PHASE_ADDITIONS[candidate])
        if candidate == phase:
            break
    return tuple(paths)


def _git_subprocess_environment() -> dict[str, str]:
    """Return a minimal Git environment with no inherited credentials."""

    return {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _default_git_runner(repo_root: Path, args: tuple[str, ...]) -> GitCommandResult:
    completed = subprocess.run(  # noqa: S603 - fixed executable, no shell.
        (
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *args,
        ),
        cwd=repo_root,
        env=_git_subprocess_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return GitCommandResult(completed.returncode, completed.stdout)


def collect_registry_attestation(repo_root: Path) -> RegistryAttestationEvidence:
    """Rebuild both registry artifacts after full bundle/runtime attestation."""

    try:
        registry = build_feedback_registry(repo_root / QUIZ_BUNDLE_ROOT)
        serialized = serialize_feedback_registry(registry).encode("utf-8")
        rendered_sql = render_feedback_registry_sql(registry).encode("utf-8")
    except (FeedbackRegistryError, OSError, UnicodeError, ValueError):
        return RegistryAttestationEvidence(
            release_id=None,
            json_sha256=None,
            sql_sha256=None,
            failed=True,
        )
    return RegistryAttestationEvidence(
        release_id=str(registry["release_id"]),
        json_sha256=hashlib.sha256(serialized).hexdigest(),
        sql_sha256=hashlib.sha256(rendered_sql).hexdigest(),
        failed=False,
    )


def collect_git_evidence(
    repo_root: Path,
    paths: Sequence[str],
    *,
    runner: GitRunner | None = None,
) -> GitEvidence:
    """Run only read-only Git queries and return sanitized evidence."""

    run = runner or _default_git_runner
    failed: set[str] = set()

    sha_result = run(repo_root, ("rev-parse", "--verify", "HEAD^{commit}"))
    sha: str | None = None
    if sha_result.returncode == 0:
        candidate = sha_result.stdout.decode("ascii", errors="ignore").strip().lower()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", candidate):
            sha = candidate
        else:
            failed.add("sha")
    else:
        failed.add("sha")

    tracked_result = run(
        repo_root,
        ("ls-files", "--cached", "--full-name", "-z", "--", *paths),
    )
    tracked_paths: frozenset[str] = frozenset()
    if tracked_result.returncode == 0:
        tracked_paths = frozenset(
            item.decode("utf-8", errors="replace")
            for item in tracked_result.stdout.split(b"\0")
            if item
        )
    else:
        failed.add("tracked")

    head_blob_sha256: list[tuple[str, str]] = []
    if sha is None:
        failed.add("head_blobs")
    else:
        for path in paths:
            blob_result = run(repo_root, ("cat-file", "blob", f"{sha}:{path}"))
            if blob_result.returncode != 0:
                failed.add("head_blobs")
                continue
            head_blob_sha256.append(
                (path, hashlib.sha256(blob_result.stdout).hexdigest())
            )

    status_result = run(
        repo_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *paths),
    )
    dirty_paths: frozenset[str] = frozenset()
    if status_result.returncode == 0:
        # The status query is already path-scoped.  Exact rename parsing is not
        # needed for the fail-closed aggregate: any output dirties this input set.
        if status_result.stdout:
            dirty_paths = frozenset(paths)
    else:
        failed.add("clean")

    migration_inventory: tuple[str, ...] = ()
    migrations_dir = repo_root / "supabase" / "migrations"
    try:
        if migrations_dir.is_symlink() or not migrations_dir.is_dir():
            raise OSError("migration directory is unavailable")
        migration_inventory = tuple(
            sorted(
                candidate.name
                for candidate in migrations_dir.iterdir()
                if candidate.name.endswith(".sql")
            )
        )
    except OSError:
        failed.add("migration_inventory")

    registry_attestation = (
        collect_registry_attestation(repo_root)
        if QUESTION_REGISTRY_JSON in paths
        else None
    )
    return GitEvidence(
        sha=sha,
        tracked_paths=tracked_paths,
        dirty_paths=dirty_paths,
        head_blob_sha256=tuple(head_blob_sha256),
        migration_inventory=migration_inventory,
        failed_queries=frozenset(failed),
        registry_attestation=registry_attestation,
    )


def load_sources(
    repo_root: Path,
    paths: Sequence[str],
) -> tuple[dict[str, bytes], frozenset[str]]:
    """Read checked rollout inputs without escaping the repository."""

    root = repo_root.resolve()
    sources: dict[str, bytes] = {}
    missing: set[str] = set()
    for relative in paths:
        relative_path = Path(relative)
        try:
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("rollout input path must stay relative")
            candidate = root
            for part in relative_path.parts:
                candidate /= part
                if candidate.is_symlink():
                    raise ValueError("rollout inputs must not be symlinks")
            metadata = candidate.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("rollout inputs must be regular files")
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(root)
            payload = candidate.read_bytes()
            payload.decode("utf-8", errors="strict")
        except (OSError, UnicodeError, ValueError):
            missing.add(relative)
        else:
            sources[relative] = payload
    return sources, frozenset(missing)


def deployment_fingerprint(
    sources: Mapping[str, bytes | str],
    paths: Sequence[str],
) -> str | None:
    """Hash exact ordered path/content pairs with unambiguous length framing."""

    digest = hashlib.sha256()
    digest.update(b"ArchitectureIQ feedback rollout inputs v1\0")
    for path in paths:
        if path not in sources:
            return None
        path_bytes = path.encode("utf-8")
        value = sources[path]
        content = value.encode("utf-8") if isinstance(value, str) else value
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _text(sources: Mapping[str, bytes | str], path: str) -> str | None:
    value = sources.get(path)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeError:
        return None


def _strict_json_object(source: str) -> dict[str, object] | None:
    """Parse one interoperable JSON object while rejecting duplicate keys."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            source,
            object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {constant}")
            ),
        )
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _check(code: str, passed: bool, success: str, failure: str) -> Check:
    return Check(code, PASS if passed else FAIL, success if passed else failure)


def _strip_sql_comments(source: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", without_blocks)


def _balanced_region(source: str, opening: int, left: str, right: str) -> str | None:
    closing = _balanced_close_index(source, opening, left, right)
    return None if closing is None else source[opening + 1 : closing]


def _balanced_close_index(
    source: str,
    opening: int,
    left: str,
    right: str,
) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == left:
            depth += 1
        elif character == right:
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _split_top_level_commas(source: str) -> tuple[str, ...] | None:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(source):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            parts.append(source[start:index].strip())
            start = index + 1
    if depth != 0:
        return None
    parts.append(source[start:].strip())
    return tuple(part for part in parts if part)


def extract_sql_function_parameters(
    source: str, function_name: str
) -> tuple[str, ...] | None:
    """Extract one function's ordered, normalized parameter declarations."""

    source = _strip_sql_comments(source)
    matches = tuple(
        re.finditer(
            rf"\bcreate\s+(?:or\s+replace\s+)?function\s+public\."
            rf"{re.escape(function_name)}\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) != 1:
        return None
    opening = matches[0].end() - 1
    region = _balanced_region(source, opening, "(", ")")
    if region is None:
        return None
    parameters = _split_top_level_commas(region)
    if parameters is None:
        return None
    return tuple(re.sub(r"\s+", " ", item).strip().lower() for item in parameters)


def extract_sql_return_columns(
    source: str, function_name: str
) -> tuple[str, ...] | None:
    """Extract RETURNS TABLE column names without assuming a column count."""

    source = _strip_sql_comments(source)
    function = re.search(
        rf"\bcreate\s+(?:or\s+replace\s+)?function\s+public\."
        rf"{re.escape(function_name)}\s*\(",
        source,
        flags=re.IGNORECASE,
    )
    if function is None:
        return None
    parameters_open = function.end() - 1
    parameters_close = _balanced_close_index(source, parameters_open, "(", ")")
    if parameters_close is None:
        return None
    declaration_tail = source[parameters_close + 1 :]
    boundaries = [
        match.start()
        for pattern in (
            r"\blanguage\b",
            r"\bas\s+\$[a-z0-9_]*\$",
            r"\bcreate\s+(?:or\s+replace\s+)?function\b",
        )
        if (match := re.search(pattern, declaration_tail, flags=re.IGNORECASE))
        is not None
    ]
    declaration_end = min(boundaries, default=len(declaration_tail))
    declaration = declaration_tail[:declaration_end]
    returns = re.search(
        r"\breturns\s+table\s*\(",
        declaration,
        flags=re.IGNORECASE,
    )
    if returns is None:
        return None
    opening = parameters_close + 1 + returns.end() - 1
    region = _balanced_region(source, opening, "(", ")")
    if region is None:
        return None
    definitions = _split_top_level_commas(region)
    if definitions is None:
        return None
    columns: list[str] = []
    for definition in definitions:
        match = re.match(r"([a-z_][a-z0-9_]*)\s+\S+", definition, re.IGNORECASE)
        if match is None:
            return None
        columns.append(match.group(1).lower())
    if not columns or len(columns) != len(set(columns)):
        return None
    return tuple(columns)


def extract_sql_function_body(source: str, function_name: str) -> str | None:
    """Extract one SQL function body delimited by a PostgreSQL dollar tag."""

    source = _strip_sql_comments(source)
    matches = tuple(
        re.finditer(
            rf"\bcreate\s+(?:or\s+replace\s+)?function\s+public\."
            rf"{re.escape(function_name)}\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) != 1:
        return None
    delimiter = re.search(
        r"\bas\s+(\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$)",
        source[matches[0].end() :],
        flags=re.IGNORECASE,
    )
    if delimiter is None:
        return None
    tag = delimiter.group(1)
    body_start = matches[0].end() + delimiter.end()
    body_end = source.find(tag, body_start)
    if body_end < 0:
        return None
    return source[body_start:body_end]


def extract_sql_view_columns(source: str, view_name: str) -> tuple[str, ...] | None:
    """Extract one simple reporting view's ordered SELECT output names."""

    source = _strip_sql_comments(source)
    matches = tuple(
        re.finditer(
            rf"\bcreate\s+(?:or\s+replace\s+)?view\s+public\."
            rf"{re.escape(view_name)}\b",
            source,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) != 1:
        return None
    tail = source[matches[0].end() :]
    select = re.search(r"\bas\s+select\b", tail, flags=re.IGNORECASE)
    if select is None:
        return None
    select_tail = tail[select.end() :]
    source_table = re.search(
        r"\bfrom\s+public\.feedback_events\b",
        select_tail,
        flags=re.IGNORECASE,
    )
    if source_table is None:
        return None
    expressions = _split_top_level_commas(select_tail[: source_table.start()])
    if expressions is None:
        return None

    columns: list[str] = []
    for expression in expressions:
        alias = re.search(
            r"\bas\s+([a-z_][a-z0-9_]*)\s*$",
            expression,
            flags=re.IGNORECASE,
        )
        direct = re.fullmatch(
            r"([a-z_][a-z0-9_]*)",
            expression.strip(),
            flags=re.IGNORECASE,
        )
        match = alias or direct
        if match is None:
            return None
        columns.append(match.group(1).lower())
    if not columns or len(columns) != len(set(columns)):
        return None
    return tuple(columns)


def extract_python_string_constant(
    source: str,
    name: str,
    *,
    allow_unordered: bool = False,
) -> tuple[str, ...] | None:
    """Extract an ordered string sequence, optionally accepting a set."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value_node = node.value
        if value_node is None:
            return None
        try:
            if (
                allow_unordered
                and isinstance(value_node, ast.Call)
                and isinstance(value_node.func, ast.Name)
                and value_node.func.id in {"set", "frozenset"}
                and len(value_node.args) == 1
                and not value_node.keywords
            ):
                value = ast.literal_eval(value_node.args[0])
            else:
                value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return None
        if not isinstance(
            value,
            (tuple, list, set, frozenset) if allow_unordered else (tuple, list),
        ) or not all(isinstance(item, str) for item in value):
            return None
        resolved = tuple(value)
        if len(resolved) != len(set(resolved)):
            return None
        return resolved
    return None


def _python_dict_keys(source: str, name: str) -> tuple[str, ...] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return None
        keys: list[str] = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            keys.append(key.value)
        return tuple(keys)
    return None


def _python_ingestion_spec_uses_columns(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_VIEW_SPECS"
            for target in node.targets
        ):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if not (
                isinstance(key, ast.Constant)
                and key.value == "feedback_report_ingestion_summary"
                and isinstance(value, ast.Call)
            ):
                continue
            return any(
                keyword.arg == "columns"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "_INGESTION_SUMMARY_COLUMNS"
                for keyword in value.keywords
            )
    return False


def _strict_report_client(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    validator = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "validate_report_rows"
        ),
        None,
    )
    if validator is None:
        return False
    calls_ingestion_validator = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_ingestion_summary_row"
        for node in ast.walk(validator)
    )
    differences = [node for node in ast.walk(validator) if isinstance(node, ast.Sub)]
    return (
        calls_ingestion_validator
        and len(differences) >= 2
        and _python_ingestion_spec_uses_columns(source)
    )


def _extract_ts_string_collection(source: str, name: str) -> tuple[str, ...] | None:
    assignment = re.search(
        rf"\b(?:export\s+)?const\s+{re.escape(name)}\s*=",
        source,
    )
    if assignment is None:
        return None
    opening = source.find("[", assignment.end())
    if opening < 0:
        return None
    region = _balanced_region(source, opening, "[", "]")
    if region is None:
        return None
    scrubbed = re.sub(r"//[^\n]*|/\*.*?\*/", "", region, flags=re.DOTALL)
    values = tuple(
        match.group(2) for match in re.finditer(r"(['\"])([^'\"]+)\1", scrubbed)
    )
    if not values or len(values) != len(set(values)):
        return None
    return values


def _expand_checks(sources: Mapping[str, bytes | str]) -> list[Check]:
    sql = _text(sources, EXPAND_MIGRATION) or ""
    normalized = re.sub(r"\s+", " ", _strip_sql_comments(sql)).lower()
    columns = extract_sql_return_columns(sql, "feedback_ingest_events")
    rpc_ok = (
        columns == _INGEST_RPC_COLUMNS
        and "security definer" in normalized
        and "pg_catalog.pg_advisory_xact_lock" in normalized
        and "insert into public.feedback_event_conflicts" in normalized
        and "if v_inserted <> v_new then" in normalized
        and re.search(
            r"grant\s+execute\s+on\s+function\s+public\.feedback_ingest_events\s*\(",
            normalized,
        )
        is not None
    )
    revoke_event_table = re.search(
        r"\brevoke\b[^;]*\b(?:insert|all)\b[^;]*\bon\s+table\s+"
        r"public\.feedback_events\b",
        normalized,
    )
    return [
        _check(
            "contract.expand.atomic_rpc",
            rpc_ok,
            "12000 defines the service-role atomic conflict-aware ingest RPC.",
            "12000 does not match the required atomic ingest RPC contract.",
        ),
        _check(
            "contract.expand.direct_insert_preserved",
            revoke_event_table is None,
            "12000 leaves the transition-period direct event INSERT grant untouched.",
            "12000 revokes direct event-table INSERT during the expand phase.",
        ),
    ]


def _ingest_checks(sources: Mapping[str, bytes | str]) -> list[Check]:
    edge = _text(sources, INGEST_EDGE) or ""
    client = _text(sources, INSPECTOR_FEEDBACK) or ""
    verifier = _text(sources, HOSTED_ROUNDTRIP_VERIFIER) or ""
    rpc_columns = extract_sql_return_columns(
        _text(sources, EXPAND_MIGRATION) or "",
        "feedback_ingest_events",
    )
    edge_columns = _extract_ts_string_collection(edge, "expectedKeys")
    event_routes = tuple(
        match.group(2)
        for match in re.finditer(r"new\s+URL\(\s*(['\"])(/rest/v1/[^'\"]+)\1", edge)
    )
    rpc_only = (
        "/rest/v1/rpc/feedback_ingest_events" in edge
        and "/rest/v1/feedback_events" not in edge
        and "feedback_events" not in edge.replace("feedback_ingest_events", "")
        and not any(route == "/rest/v1/feedback_events" for route in event_routes)
    )
    obs2 = (
        re.search(r'const\s+OBSERVER_REVISION\s*=\s*["\']obs2["\']', edge) is not None
        and re.search(r'schema_version\s*:\s*["\']1\.1["\']', edge) is not None
        and "conflicting_event_count: details.counts.conflicting" in edge
    )
    conflict = all(
        marker in edge
        for marker in (
            "ingest.conflicting > 0",
            'outcomeCode: "event_id_conflict"',
            "httpStatus: 409",
            'code: "EVENT_ID_CONFLICT"',
            "retryable: false",
        )
    )
    interoperable_json = all(
        marker in client
        for marker in (
            "MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991",
            "def _validate_json_interoperability(",
            "item.is_integer()",
            "_contains_unicode_surrogate",
            "_validate_json_interoperability(copied, field_name=field_name)",
        )
    ) and all(
        marker in edge
        for marker in (
            "const MAX_SAFE_JSON_INTEGER = Number.MAX_SAFE_INTEGER",
            "function validateJsonInteroperability(",
            "validateJsonInteroperability(value);",
            "function unicodeCodePointLength(",
            "function hasLoneSurrogate(",
        )
    )
    production_client_reused = all(
        marker in verifier
        for marker in (
            "from question_inspector import feedback",
            "feedback.FeedbackClient(",
        )
    )
    question_reaction_wire = all(
        marker in client
        for marker in (
            '"question_reaction_submitted"',
            "def record_question_reaction(",
            '"reaction": "surprise"',
            '"timing": "after_reveal"',
            "question reaction value must be a boolean",
            'event_id = f"evt_reaction_',
            "def record_question_presented(",
            '"question_presented"',
            'event_id = f"evt_presented_',
        )
    ) and all(
        marker in edge
        for marker in (
            '"question_reaction_submitted"',
            'value.reaction !== "surprise"',
            'typeof value.value !== "boolean"',
            'value.timing !== "after_reveal"',
            "requireIdentifier(value.attempt_id",
            'case "question_presented"',
            'typeof value.propensity !== "number"',
            '"event.payload.position"',
        )
    )
    return [
        _check(
            "contract.ingest.rpc_only",
            rpc_only,
            "feedback-ingest sends event writes only through the ingest RPC.",
            "feedback-ingest does not use the required RPC-only event-write path.",
        ),
        _check(
            "contract.ingest.observer_obs2",
            obs2,
            "feedback-ingest records outcome schema 1.1 with observer revision obs2.",
            "feedback-ingest does not match the schema 1.1/obs2 outcome contract.",
        ),
        _check(
            "contract.ingest.conflict_response",
            conflict,
            "feedback-ingest exposes the non-retryable HTTP 409 conflict receipt.",
            "feedback-ingest does not match the required conflict response contract.",
        ),
        _check(
            "contract.ingest.rpc_shape",
            edge_columns is not None and edge_columns == rpc_columns,
            "feedback-ingest strictly validates the current RPC result columns.",
            "feedback-ingest and the ingest RPC result columns differ.",
        ),
        _check(
            "contract.ingest.interoperable_client",
            interoperable_json and production_client_reused,
            (
                "Inspector, hosted verifier, and Edge share the fail-closed "
                "safe-number/Unicode ingestion contract."
            ),
            (
                "The deployed client/verifier and Edge no longer share the "
                "required interoperable JSON contract."
            ),
        ),
        _check(
            "contract.ingest.question_reaction_wire",
            question_reaction_wire,
            (
                "Inspector and Edge share the strict, deterministic post-result "
                "surprise-reaction event contract."
            ),
            (
                "Inspector and Edge no longer agree on the strict post-result "
                "surprise-reaction event contract."
            ),
        ),
    ]


def _lockdown_report_checks(
    sources: Mapping[str, bytes | str],
    registry_attestation: RegistryAttestationEvidence | None,
) -> list[Check]:
    lockdown = _text(sources, LOCKDOWN_MIGRATION) or ""
    statements = tuple(
        re.sub(r"\s+", " ", statement).strip().lower()
        for statement in _strip_sql_comments(lockdown).split(";")
        if statement.strip()
    )
    only_lockdown = statements == (
        "begin",
        "revoke insert on table public.feedback_events from service_role",
        "grant select on table public.feedback_events to service_role",
        "commit",
    )
    report_sql = _text(sources, CONFLICT_REPORT_MIGRATION) or ""
    baseline_view_sql = _text(sources, BASELINE_EVENT_MIGRATION) or ""
    raw_view_sql = _text(sources, RAW_VIEW_HARDENING_MIGRATION) or ""
    registry_sql = _text(sources, QUESTION_REGISTRY_MIGRATION) or ""
    registry_data_sql = _text(sources, QUESTION_REGISTRY_DATA_MIGRATION) or ""
    authoritative_sql = _text(sources, AUTHORITATIVE_REPORT_MIGRATION) or ""
    detail_sql = _text(sources, DETAIL_REPORT_MIGRATION) or ""
    snapshot_sql = _text(sources, BUSINESS_SNAPSHOT_MIGRATION) or ""
    session_attempt_sql = _text(sources, SESSION_ATTEMPT_FILTER_MIGRATION) or ""
    reaction_sql = _text(sources, QUESTION_REACTION_MIGRATION) or ""
    surprise_report_sql = _text(sources, SURPRISE_REPORT_MIGRATION) or ""
    normalized_reaction = re.sub(r"\s+", " ", _strip_sql_comments(reaction_sql)).lower()
    normalized_surprise_report = (
        re.sub(r"\s+", " ", _strip_sql_comments(surprise_report_sql)).strip().lower()
    )
    hosted_verifier = _text(sources, HOSTED_ROUNDTRIP_VERIFIER) or ""
    registry_json = _text(sources, QUESTION_REGISTRY_JSON) or ""
    registry_exporter = _text(sources, QUESTION_REGISTRY_EXPORTER) or ""
    registry_cli = _text(sources, QUESTION_REGISTRY_CLI) or ""
    bundle_manifest = _text(sources, QUIZ_BUNDLE_MANIFEST) or ""
    raw_view_contract = (
        re.sub(r"\s+", " ", _strip_sql_comments(raw_view_sql)).strip().lower()
    )
    baseline_session_columns = extract_sql_view_columns(
        baseline_view_sql,
        "feedback_session_summary",
    )
    baseline_question_columns = extract_sql_view_columns(
        baseline_view_sql,
        "feedback_question_stats",
    )
    baseline_proposal_columns = extract_sql_view_columns(
        baseline_view_sql,
        "feedback_proposals",
    )
    raw_session_columns = extract_sql_view_columns(
        raw_view_sql,
        "feedback_session_summary",
    )
    raw_question_columns = extract_sql_view_columns(
        raw_view_sql,
        "feedback_question_stats",
    )
    raw_proposal_columns = extract_sql_view_columns(
        raw_view_sql,
        "feedback_proposals",
    )
    appended_answer_columns = (
        "known_answer_count",
        "incorrect_answer_count",
        "unknown_answer_count",
    )
    raw_views_hardened = (
        baseline_session_columns is not None
        and baseline_question_columns is not None
        and baseline_proposal_columns is not None
        and raw_session_columns == _RAW_SESSION_VIEW_COLUMNS
        and raw_question_columns == _RAW_QUESTION_VIEW_COLUMNS
        and raw_proposal_columns == _RAW_PROPOSAL_VIEW_COLUMNS
        and raw_session_columns[: len(baseline_session_columns)]
        == baseline_session_columns
        and raw_question_columns[: len(baseline_question_columns)]
        == baseline_question_columns
        and raw_proposal_columns == baseline_proposal_columns
        and raw_session_columns[len(baseline_session_columns) :]
        == appended_answer_columns
        and raw_question_columns[len(baseline_question_columns) :]
        == appended_answer_columns
        and raw_view_contract.count("create or replace view public.") == 3
        and raw_view_contract.count(
            "with (security_invoker = true, security_barrier = true)"
        )
        == 3
        and "count(distinct (question_id, question_version)) as question_count"
        in raw_view_contract
        and "coalesce(nullif(payload ->> 'attempt_id', ''), '')" in raw_view_contract
        and raw_view_contract.count("jsonb_typeof(payload -> 'is_correct') = 'boolean'")
        >= 4
        and raw_view_contract.count("between -2147483648 and 2147483647") == 2
        and "create or replace view public.feedback_comments" not in raw_view_contract
        and "from public, anon, authenticated" in raw_view_contract
        and "to service_role" in raw_view_contract
    )
    client = _text(sources, REPORT_CLIENT) or ""
    sql_columns = extract_sql_return_columns(
        report_sql,
        "feedback_report_ingestion_summary",
    )
    client_columns = extract_python_string_constant(
        client,
        "_INGESTION_SUMMARY_COLUMNS",
    )
    registry_quality_columns = extract_sql_return_columns(
        registry_sql,
        "feedback_report_registry_quality",
    )
    event_resolution_columns = extract_sql_return_columns(
        registry_sql,
        "feedback_report_event_resolution",
    )
    client_registry_columns = extract_python_string_constant(
        client,
        "_REGISTRY_QUALITY_COLUMNS",
    )
    client_resolution_columns = extract_python_string_constant(
        client,
        "_EVENT_RESOLUTION_COLUMNS",
    )
    authority_status_columns = extract_sql_return_columns(
        authoritative_sql,
        "feedback_report_authority_status",
    )
    client_authority_status_columns = extract_python_string_constant(
        client,
        "_AUTHORITY_STATUS_COLUMNS",
    )
    authoritative_columns_match = all(
        extract_sql_return_columns(authoritative_sql, function_name)
        == extract_python_string_constant(client, client_constant)
        for function_name, client_constant in (
            ("feedback_report_summary", "_SUMMARY_COLUMNS"),
            ("feedback_report_sessions", "_SESSION_COLUMNS"),
            ("feedback_report_questions", "_QUESTION_COLUMNS"),
            ("feedback_report_comments", "_COMMENT_COLUMNS"),
        )
    )
    answer_columns = extract_sql_return_columns(
        detail_sql,
        "feedback_report_answers",
    )
    proposal_columns = extract_sql_return_columns(
        detail_sql,
        "feedback_report_proposals",
    )
    client_answer_columns = extract_python_string_constant(client, "_ANSWER_COLUMNS")
    client_proposal_columns = extract_python_string_constant(
        client,
        "_PROPOSAL_COLUMNS",
    )
    detail_authority_status_columns = extract_sql_return_columns(
        detail_sql,
        "feedback_report_authority_status",
    )
    snapshot_columns = extract_sql_return_columns(
        snapshot_sql,
        "feedback_report_business_snapshot",
    )
    client_snapshot_columns = extract_python_string_constant(
        client,
        "_BUSINESS_SNAPSHOT_COLUMNS",
    )
    normalized_registry = (
        re.sub(r"\s+", " ", _strip_sql_comments(registry_sql)).strip().lower()
    )
    normalized_registry_data = (
        re.sub(r"\s+", " ", _strip_sql_comments(registry_data_sql)).strip().lower()
    )
    normalized_authoritative = (
        re.sub(r"\s+", " ", _strip_sql_comments(authoritative_sql)).strip().lower()
    )
    normalized_detail = (
        re.sub(r"\s+", " ", _strip_sql_comments(detail_sql)).strip().lower()
    )
    normalized_snapshot = (
        re.sub(r"\s+", " ", _strip_sql_comments(snapshot_sql)).strip().lower()
    )
    normalized_session_attempt = (
        re.sub(r"\s+", " ", _strip_sql_comments(session_attempt_sql)).strip().lower()
    )
    answer_body_source = extract_sql_function_body(
        detail_sql,
        "feedback_report_answers",
    )
    proposal_body_source = extract_sql_function_body(
        detail_sql,
        "feedback_report_proposals",
    )
    normalized_answer_body = (
        re.sub(r"\s+", " ", answer_body_source).strip().lower()
        if answer_body_source is not None
        else ""
    )
    normalized_proposal_body = (
        re.sub(r"\s+", " ", proposal_body_source).strip().lower()
        if proposal_body_source is not None
        else ""
    )
    snapshot_body_source = extract_sql_function_body(
        snapshot_sql,
        "feedback_report_business_snapshot",
    )
    normalized_snapshot_body = (
        re.sub(r"\s+", " ", snapshot_body_source).strip().lower()
        if snapshot_body_source is not None
        else ""
    )
    answer_where = (
        normalized_answer_body.split(" where ", maxsplit=1)[1].rsplit(
            " order by ", maxsplit=1
        )[0]
        if " where " in normalized_answer_body
        and " order by " in normalized_answer_body
        else None
    )
    proposal_where = (
        normalized_proposal_body.split(" where ", maxsplit=1)[1].rsplit(
            " order by ", maxsplit=1
        )[0]
        if " where " in normalized_proposal_body
        and " order by " in normalized_proposal_body
        else None
    )
    expected_answer_where = (
        "events.registry_status = 'matched' "
        "and events.event_type = 'answer_submitted' "
        "and ( p_release_id is null or events.authoritative_release_id = "
        "p_release_id ) and ( p_family is null or events.authoritative_family = "
        "p_family ) and ( p_question_type is null or "
        "events.authoritative_question_type = p_question_type ) and ( "
        "p_question_id is null or events.authoritative_question_id = "
        "p_question_id ) and (p_from is null or events.occurred_at >= p_from) "
        "and (p_to is null or events.occurred_at < p_to) and (p_from is null "
        "or p_to is null or p_from < p_to)"
    )
    expected_proposal_where = (
        "events.registry_status = 'matched' and events.event_type in ( "
        "'custom_setting_proposed', 'custom_setting_rejected' ) and ( "
        "p_release_id is null or events.authoritative_release_id = p_release_id ) "
        "and ( p_family is null or events.authoritative_family = p_family ) and "
        "( p_question_type is null or events.authoritative_question_type = "
        "p_question_type ) and ( p_question_id is null or "
        "events.authoritative_question_id = p_question_id ) and (p_from is null "
        "or events.occurred_at >= p_from) and (p_to is null or "
        "events.occurred_at < p_to) and (p_from is null or p_to is null or "
        "p_from < p_to)"
    )
    authoritative_dimension_projections = (
        "events.authoritative_question_id as question_id",
        "events.authoritative_question_version as question_version",
        "events.authoritative_release_id as release_id",
        "events.authoritative_family as family",
        "events.authoritative_dataset_id as dataset_id",
        "events.authoritative_question_type as question_type",
    )
    detail_acl_statements = tuple(
        statement
        for function_name in (
            "feedback_report_answers",
            "feedback_report_proposals",
        )
        for statement in (
            f"revoke all on function public.{function_name}( text, text, text, "
            "text, timestamptz, timestamptz ) from public, anon, authenticated, "
            "service_role;",
            f"grant execute on function public.{function_name}( text, text, text, "
            "text, timestamptz, timestamptz ) to service_role;",
        )
    ) + (
        "revoke all on function public.feedback_report_authority_status() from "
        "public, anon, authenticated, service_role;",
        "grant execute on function public.feedback_report_authority_status() to "
        "service_role;",
    )

    registry_document = _strict_json_object(registry_json)
    manifest_document = _strict_json_object(bundle_manifest)

    registry_artifact_matches = False
    if registry_document is not None and manifest_document is not None:
        questions = registry_document.get("questions")
        manifest_questions = manifest_document.get("questions")
        if isinstance(questions, list) and isinstance(manifest_questions, list):
            registry_questions = {
                (
                    item.get("question_id"),
                    item.get("question_version"),
                    item.get("family"),
                    item.get("dataset_id"),
                )
                for item in questions
                if isinstance(item, Mapping)
            }
            manifest_question_set = {
                (
                    item.get("question_id"),
                    item.get("version"),
                    item.get("family"),
                    item.get("dataset_id"),
                )
                for item in manifest_questions
                if isinstance(item, Mapping)
            }
            try:
                canonical_registry = serialize_feedback_registry(registry_document)
                rendered_registry_sql = render_feedback_registry_sql(registry_document)
            except FeedbackRegistryError:
                pass
            else:
                registry_artifact_matches = (
                    canonical_registry == registry_json
                    and rendered_registry_sql == registry_data_sql
                    and registry_document.get("release_id")
                    == manifest_document.get("release_id")
                    and registry_document.get("manifest_sha256")
                    == hashlib.sha256(bundle_manifest.encode("utf-8")).hexdigest()
                    and registry_document.get("question_count") == len(questions) == 60
                    and registry_document.get("choice_count") == 180
                    and registry_questions == manifest_question_set
                )

    registry_contract = (
        normalized_registry.count("create table public.feedback_quiz_") == 3
        and normalized_registry.count("enable row level security") == 3
        and normalized_registry.count("force row level security") == 3
        and normalized_registry.count("before update or delete or truncate") == 3
        and "grant select on table" in normalized_registry
        and "to service_role" in normalized_registry
        and "grant insert" not in normalized_registry
        and "deferrable initially deferred" in normalized_registry
        and "feedback_quiz_choice_inventory_complete" in normalized_registry
        and "feedback_quiz_question_release_inventory_complete" in normalized_registry
        and "feedback_quiz_choice_release_inventory_complete" in normalized_registry
        and "feedback_quiz_question_version_lock" in normalized_registry
        and "pg_catalog.pg_advisory_xact_lock" in normalized_registry
        and normalized_registry.count("after insert on public.feedback_quiz_choices")
        >= 2
        and "create view public.feedback_authoritative_events" in normalized_registry
        and "releases.release_id = nullif(events.payload ->> 'release_id', '')"
        in normalized_registry
        and "questions.question_id = events.question_id" in normalized_registry
        and "questions.question_version = events.question_version"
        in normalized_registry
        and "choices.letter = nullif(events.payload ->> 'selected_letter', '')"
        in normalized_registry
        and "coalesce( resolved.authoritative_is_correct is not null"
        in normalized_registry
        and "pg_catalog.coalesce(" not in normalized_registry
        and "false ) as client_correctness_mismatch" in normalized_registry
        and "'not_found'::text" in normalized_registry
        and registry_quality_columns == _REGISTRY_QUALITY_COLUMNS
        and event_resolution_columns == _EVENT_RESOLUTION_COLUMNS
        and client_registry_columns == registry_quality_columns
        and client_resolution_columns == event_resolution_columns
    )
    registry_data_contract = (
        normalized_registry_data.startswith("begin; insert into")
        and normalized_registry_data.endswith("commit;")
        and normalized_registry_data.count("insert into") == 3
        and all(
            forbidden not in normalized_registry_data
            for forbidden in (
                " update ",
                " delete ",
                " truncate ",
                " on conflict ",
                " service_role ",
            )
        )
        and registry_artifact_matches
        and registry_attestation is not None
        and not registry_attestation.failed
        and registry_attestation.release_id
        == (
            str(registry_document.get("release_id"))
            if registry_document is not None
            else None
        )
        and registry_attestation.json_sha256
        == hashlib.sha256(registry_json.encode("utf-8")).hexdigest()
        and registry_attestation.sql_sha256
        == hashlib.sha256(registry_data_sql.encode("utf-8")).hexdigest()
        and all(
            marker in registry_exporter
            for marker in (
                "load_quiz_manifest(root)",
                "build_bundle_manifest(",
                "manifest_sha256",
                "def _resolved_output_paths(",
                "def render_feedback_registry_sql(",
            )
        )
        and '"--check"' in registry_cli
    )
    authoritative_contract = (
        authoritative_columns_match
        and authority_status_columns == _AUTHORITY_STATUS_COLUMNS
        and "pg_catalog.coalesce(" not in normalized_authoritative
        and normalized_authoritative.count(
            "from public.feedback_authoritative_events as events"
        )
        == 4
        and normalized_authoritative.count("events.registry_status = 'matched'") == 4
        and "payload ->> 'release_id'" not in normalized_authoritative
        and "payload ->> 'family'" not in normalized_authoritative
        and "payload ->> 'dataset_id'" not in normalized_authoritative
        and "payload ->> 'question_type'" not in normalized_authoritative
        and "payload -> 'is_correct'" not in normalized_authoritative
        and "'registry_v1'::text as authority_revision" in normalized_authoritative
        and "true as business_reports_authoritative" in normalized_authoritative
        and "grant execute on function public.feedback_report_authority_status()"
        in normalized_authoritative
        and "def _validate_authority_status_row(" in client
        and "def _validate_registry_quality_row(" in client
        and "def _validate_event_resolution_row(" in client
    )
    detail_contract = (
        answer_columns == _ANSWER_COLUMNS
        and proposal_columns == _PROPOSAL_COLUMNS
        and client_answer_columns == answer_columns
        and client_proposal_columns == proposal_columns
        and detail_authority_status_columns == _FINAL_AUTHORITY_STATUS_COLUMNS
        and client_authority_status_columns == detail_authority_status_columns
        and normalized_detail.startswith(
            "begin; drop function public.feedback_report_authority_status(); "
            "create function"
        )
        and normalized_detail.endswith("commit;")
        and normalized_detail.count(
            "from public.feedback_authoritative_events as events"
        )
        == 2
        and normalized_detail.count("events.registry_status = 'matched'") == 2
        and answer_where == expected_answer_where
        and proposal_where == expected_proposal_where
        and all(
            projection in normalized_answer_body
            and projection in normalized_proposal_body
            for projection in authoritative_dimension_projections
        )
        and normalized_detail.count("events.authoritative_release_id = p_release_id")
        == 2
        and normalized_detail.count("events.authoritative_family = p_family") == 2
        and normalized_detail.count(
            "events.authoritative_question_type = p_question_type"
        )
        == 2
        and normalized_detail.count("events.authoritative_question_id = p_question_id")
        == 2
        and "events.authoritative_is_correct as is_correct" in normalized_detail
        and "payload ->> 'release_id'" not in normalized_detail
        and "payload ->> 'family'" not in normalized_detail
        and "payload ->> 'dataset_id'" not in normalized_detail
        and "payload ->> 'question_type'" not in normalized_detail
        and "when 'custom_setting_proposed' then 'proposed'" in normalized_detail
        and "when 'custom_setting_rejected' then 'rejected'" in normalized_detail
        and "and events.event_type = 'answer_submitted' and ( p_release_id is null"
        in normalized_detail
        and (
            "and events.event_type in ( 'custom_setting_proposed', "
            "'custom_setting_rejected' ) and ( p_release_id is null"
        )
        in normalized_detail
        and "custom_run_completed" not in normalized_detail
        and "custom_run_failed" not in normalized_detail
        and "comment_submitted" not in normalized_detail
        and "'detail_v1'::text as detail_revision" in normalized_detail
        and "true as detail_reports_authoritative" in normalized_detail
        and "jsonb_typeof(events.payload -> 'setting') = 'object'" in normalized_detail
        and "then (events.payload -> 'setting')::text else null end as setting_json"
        in normalized_detail
        and "then (events.payload -> 'inherited_from')::text else null end as inherited_from_json"
        in normalized_detail
        and normalized_detail.count("between -2147483648 and 2147483647") == 2
        and normalized_detail.count("revoke all on function public.feedback_report_")
        == 3
        and normalized_detail.count("from public, anon, authenticated, service_role")
        == 3
        and normalized_detail.count("grant execute on function public.feedback_report_")
        == 3
        and normalized_detail.count("to service_role") == 3
        and all(statement in normalized_detail for statement in detail_acl_statements)
        and "def _validate_answer_detail_row(" in client
        and "def _validate_proposal_detail_row(" in client
        and "_validate_answer_detail_row(row, index=index)" in client
        and "_validate_proposal_detail_row(row, index=index)" in client
    )
    snapshot_rpc_calls = (
        "feedback_report_authority_status",
        "feedback_report_summary",
        "feedback_report_sessions",
        "feedback_report_questions",
        "feedback_report_answers",
        "feedback_report_proposals",
        "feedback_report_comments",
    )
    snapshot_page_views = (
        "feedback_report_summary",
        "feedback_report_sessions",
        "feedback_report_questions",
        "feedback_report_answers",
        "feedback_report_proposals",
        "feedback_report_comments",
    )
    snapshot_page_byte_budgets = {
        65536: 1,
        262144: 3,
        2621440: 1,
        131072: 1,
    }
    hosted_snapshot_position = _function_first_call_position(
        hosted_verifier,
        "_run_authoritative_roundtrip",
        "fetch_business_snapshot",
    )
    hosted_first_post_position = _function_first_call_position(
        hosted_verifier,
        "_run_authoritative_roundtrip",
        "post_event",
    )
    snapshot_contract = (
        snapshot_columns == _BUSINESS_SNAPSHOT_COLUMNS
        and client_snapshot_columns == snapshot_columns
        and "pg_catalog.coalesce(" not in normalized_snapshot
        and normalized_snapshot.startswith(
            "begin; create function public.feedback_report_business_snapshot("
        )
        and normalized_snapshot.endswith("commit;")
        and "language sql stable security invoker set search_path = ''"
        in normalized_snapshot
        and "where p_limit between 1 and 1000" in normalized_snapshot_body
        and (
            "from parameters cross join lateral ( select status.* from "
            "public.feedback_report_authority_status() as status limit "
            "parameters.page_limit ) as status"
        )
        in normalized_snapshot_body
        and normalized_snapshot_body.count(
            "from parameters cross join lateral public.feedback_report_"
        )
        == 6
        and all(
            normalized_snapshot_body.count(f"public.{function_name}(") == 1
            for function_name in snapshot_rpc_calls
        )
        and "public.feedback_authoritative_events" not in normalized_snapshot_body
        and "'business_snapshot_v1'::text as snapshot_revision"
        in normalized_snapshot_body
        and "pg_catalog.statement_timestamp() as snapshot_at"
        in normalized_snapshot_body
        and "authority.authority_revision" in normalized_snapshot_body
        and "authority.detail_revision" in normalized_snapshot_body
        and "from parameters cross join authority" in normalized_snapshot_body
        and "::text as pages_json" in normalized_snapshot_body
        and "report_rows as not materialized" in normalized_snapshot_body
        and "sized_rows as not materialized" in normalized_snapshot_body
        and "budgeted_rows as not materialized" in normalized_snapshot_body
        and "page_results as materialized" in normalized_snapshot_body
        and "pages_document as materialized" in normalized_snapshot_body
        and all(
            f"{page_name}_page_rows as materialized" in normalized_snapshot_body
            for page_name in (
                "summary",
                "session",
                "question",
                "answer",
                "proposal",
                "comment",
            )
        )
        and all(
            removed_wide_cte not in normalized_snapshot_body
            for removed_wide_cte in (
                "summary_rows as materialized",
                "session_rows as materialized",
                "question_rows as materialized",
                "answer_rows as materialized",
                "proposal_rows as materialized",
                "comment_rows as materialized",
            )
        )
        and sum(
            byte_budget * expected_count
            for byte_budget, expected_count in snapshot_page_byte_budgets.items()
        )
        == 3_604_480
        and all(
            normalized_snapshot_body.count(f"{byte_budget}::bigint") == expected_count
            for byte_budget, expected_count in snapshot_page_byte_budgets.items()
        )
        and "4194304::bigint as snapshot_pages_bytes" in normalized_snapshot_body
        and normalized_snapshot_body.count("pg_catalog.to_jsonb(") == 6
        and normalized_snapshot_body.count(
            "- 'snapshot_page_rank' - 'snapshot_exact_total'"
        )
        == 6
        and normalized_snapshot_body.count("pg_catalog.row_number() over") == 6
        and normalized_snapshot_body.count("pg_catalog.count(*) over ()") == 6
        and "pg_catalog.convert_to(rows.row_json::text, 'utf8')"
        in normalized_snapshot_body
        and (
            "partition by rows.view_name order by rows.page_rank rows between "
            "unbounded preceding and current row"
        )
        in normalized_snapshot_body
        and normalized_snapshot_body.count(
            "where ranked.snapshot_page_rank <= ( select parameters.page_limit "
            "from parameters )"
        )
        == 6
        and ("rows.cumulative_page_bytes + 1 <= definitions.page_byte_budget")
        in normalized_snapshot_body
        and "pg_catalog.jsonb_agg( rows.row_json order by rows.page_rank )"
        in normalized_snapshot_body
        and (
            "pg_catalog.convert_to(pages_document.document::text, 'utf8') ) <= "
            "byte_budgets.snapshot_pages_bytes"
        )
        in normalized_snapshot_body
        and all(
            normalized_snapshot_body.count(f"'{view}'") == 6
            for view in snapshot_page_views
        )
        and normalized_snapshot_body.count("'rows',") == 6
        and normalized_snapshot_body.count("'total',") == 6
        and normalized_snapshot_body.count("'offset', 0") == 6
        and normalized_snapshot_body.count("'limit', parameters.page_limit") == 6
        and (
            "questions.answer_count desc, questions.question_id asc, "
            "questions.question_version asc, questions.release_id asc nulls first, "
            "questions.family asc nulls first, questions.dataset_id asc nulls first, "
            "questions.question_type asc nulls first"
        )
        in normalized_snapshot_body
        and (
            "revoke all on function public.feedback_report_business_snapshot( "
            "text, text, text, text, timestamptz, timestamptz, integer ) from "
            "public, anon, authenticated, service_role;"
        )
        in normalized_snapshot
        and (
            "grant execute on function public.feedback_report_business_snapshot( "
            "text, text, text, text, timestamptz, timestamptz, integer ) to "
            "service_role;"
        )
        in normalized_snapshot
        and "class BusinessSnapshot:" in client
        and "def validate_business_snapshot_response(" in client
        and "def fetch_business_snapshot(" in client
        and 'row["snapshot_revision"] != "business_snapshot_v1"' in client
        and "object_pairs_hook=_reject_duplicate_json_object" in client
        and "_validate_business_snapshot_conservation(" in client
        and all(
            marker in hosted_verifier
            for marker in (
                'BUSINESS_SNAPSHOT_VIEW = "feedback_report_business_snapshot"',
                "reports_client.fetch_business_snapshot(",
                "assert_empty_business_snapshot(",
                "business_snapshot_verified=True",
            )
        )
        and hosted_snapshot_position is not None
        and hosted_first_post_position is not None
        and hosted_snapshot_position < hosted_first_post_position
    )

    # 18000 is deliberately a forward migration rather than a rewrite of the
    # reviewed 15000/16000/17000 chain.  The final bodies must be byte-semantic
    # copies of those functions with only the two exact predicates and their
    # snapshot forwarding added.
    common_parameters = (
        "p_release_id text default null",
        "p_family text default null",
        "p_question_type text default null",
        "p_question_id text default null",
        "p_from timestamptz default null",
        "p_to timestamptz default null",
    )
    final_parameter_contract = {
        "feedback_report_summary": common_parameters
        + (
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        "feedback_report_sessions": common_parameters
        + (
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        "feedback_report_questions": common_parameters
        + (
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        # Preserve the historical seventh positional argument.
        "feedback_report_comments": common_parameters
        + (
            "p_category text default null",
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        "feedback_report_answers": common_parameters
        + (
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        "feedback_report_proposals": common_parameters
        + (
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
        # Preserve the historical seventh positional argument.
        "feedback_report_business_snapshot": common_parameters
        + (
            "p_limit integer default 200",
            "p_session_id text default null",
            "p_attempt_id text default null",
        ),
    }
    final_column_contract = {
        "feedback_report_summary": extract_python_string_constant(
            client, "_SUMMARY_COLUMNS"
        ),
        "feedback_report_sessions": extract_python_string_constant(
            client, "_SESSION_COLUMNS"
        ),
        "feedback_report_questions": extract_python_string_constant(
            client, "_QUESTION_COLUMNS"
        ),
        "feedback_report_comments": extract_python_string_constant(
            client, "_COMMENT_COLUMNS"
        ),
        "feedback_report_answers": client_answer_columns,
        "feedback_report_proposals": client_proposal_columns,
        "feedback_report_business_snapshot": client_snapshot_columns,
    }
    legacy_body_sources = {
        name: extract_sql_function_body(
            detail_sql
            if name in {"feedback_report_answers", "feedback_report_proposals"}
            else authoritative_sql,
            name,
        )
        for name in (
            "feedback_report_summary",
            "feedback_report_sessions",
            "feedback_report_questions",
            "feedback_report_comments",
            "feedback_report_answers",
            "feedback_report_proposals",
        )
    }
    final_body_sources = {
        name: extract_sql_function_body(session_attempt_sql, name)
        for name in final_parameter_contract
    }

    def normalized_body(value: str | None) -> str | None:
        return re.sub(r"\s+", " ", value).strip().lower() if value is not None else None

    def expected_filtered_body(value: str | None) -> str | None:
        body = normalized_body(value)
        needle = "and (p_from is null or events.occurred_at >= p_from)"
        if body is None or body.count(needle) != 1:
            return None
        return body.replace(
            needle,
            "and ( p_session_id is null or events.session_id = p_session_id ) "
            "and ( p_attempt_id is null or events.report_attempt_id = "
            f"p_attempt_id ) {needle}",
            1,
        )

    expected_snapshot_body = normalized_snapshot_body
    snapshot_parameter_prefix = "select p_limit as page_limit, p_release_id"
    if expected_snapshot_body.count(snapshot_parameter_prefix) == 1:
        expected_snapshot_body = expected_snapshot_body.replace(
            snapshot_parameter_prefix,
            "select p_limit as page_limit, p_session_id as session_id, "
            "p_attempt_id as attempt_id, p_release_id",
            1,
        )
    else:
        expected_snapshot_body = ""
    for alias in ("summary", "sessions", "questions", "answers", "proposals"):
        call_suffix = f"parameters.from_at, parameters.to_at ) as {alias}"
        if expected_snapshot_body.count(call_suffix) != 1:
            expected_snapshot_body = ""
            break
        expected_snapshot_body = expected_snapshot_body.replace(
            call_suffix,
            "parameters.from_at, parameters.to_at, parameters.session_id, "
            f"parameters.attempt_id ) as {alias}",
            1,
        )
    comments_call_suffix = "parameters.from_at, parameters.to_at, null ) as comments"
    if expected_snapshot_body.count(comments_call_suffix) == 1:
        expected_snapshot_body = expected_snapshot_body.replace(
            comments_call_suffix,
            "parameters.from_at, parameters.to_at, null, "
            "parameters.session_id, parameters.attempt_id ) as comments",
            1,
        )
    else:
        expected_snapshot_body = ""

    final_function_names = tuple(final_parameter_contract)
    old_signature_types = {
        "feedback_report_summary": "text,text,text,text,timestamptz,timestamptz",
        "feedback_report_sessions": "text,text,text,text,timestamptz,timestamptz",
        "feedback_report_questions": "text,text,text,text,timestamptz,timestamptz",
        "feedback_report_comments": (
            "text,text,text,text,timestamptz,timestamptz,text"
        ),
        "feedback_report_answers": "text,text,text,text,timestamptz,timestamptz",
        "feedback_report_proposals": "text,text,text,text,timestamptz,timestamptz",
        "feedback_report_business_snapshot": (
            "text,text,text,text,timestamptz,timestamptz,integer"
        ),
    }
    new_signature_types = {
        "feedback_report_summary": (
            "text,text,text,text,timestamptz,timestamptz,text,text"
        ),
        "feedback_report_sessions": (
            "text,text,text,text,timestamptz,timestamptz,text,text"
        ),
        "feedback_report_questions": (
            "text,text,text,text,timestamptz,timestamptz,text,text"
        ),
        "feedback_report_comments": (
            "text,text,text,text,timestamptz,timestamptz,text,text,text"
        ),
        "feedback_report_answers": (
            "text,text,text,text,timestamptz,timestamptz,text,text"
        ),
        "feedback_report_proposals": (
            "text,text,text,text,timestamptz,timestamptz,text,text"
        ),
        "feedback_report_business_snapshot": (
            "text,text,text,text,timestamptz,timestamptz,integer,text,text"
        ),
    }
    compact_session_attempt = re.sub(
        r"\s*([(),;])\s*", r"\1", normalized_session_attempt
    )
    exact_old_drops = all(
        f"drop function public.{name}({old_signature_types[name]});"
        in compact_session_attempt
        for name in final_function_names
    )
    exact_new_acls = all(
        (
            f"revoke all on function public.{name}({new_signature_types[name]})"
            "from public,anon,authenticated,service_role;"
        )
        in compact_session_attempt
        and (
            f"grant execute on function public.{name}({new_signature_types[name]})"
            "to service_role;"
        )
        in compact_session_attempt
        for name in final_function_names
    )
    snapshot_drop_position = normalized_session_attempt.find(
        "drop function public.feedback_report_business_snapshot("
    )
    dependent_drop_positions = tuple(
        normalized_session_attempt.find(f"drop function public.{name}(")
        for name in final_function_names
        if name != "feedback_report_business_snapshot"
    )
    session_attempt_contract = (
        normalized_session_attempt.startswith("begin; drop function")
        and normalized_session_attempt.endswith("commit;")
        and normalized_session_attempt.count("create function public.feedback_report_")
        == 7
        and normalized_session_attempt.count("drop function public.feedback_report_")
        == 7
        and "create or replace function" not in normalized_session_attempt
        and " cascade" not in normalized_session_attempt
        and "if exists" not in normalized_session_attempt
        and "create function public.feedback_report_authority_status"
        not in (normalized_session_attempt)
        and "drop function public.feedback_report_authority_status"
        not in (normalized_session_attempt)
        and exact_old_drops
        and exact_new_acls
        and snapshot_drop_position >= 0
        and all(
            position > snapshot_drop_position for position in dependent_drop_positions
        )
        and all(
            extract_sql_function_parameters(session_attempt_sql, name) == parameters
            for name, parameters in final_parameter_contract.items()
        )
        and all(
            columns is not None
            and extract_sql_return_columns(session_attempt_sql, name) == columns
            for name, columns in final_column_contract.items()
        )
        and all(
            normalized_body(final_body_sources[name])
            == expected_filtered_body(legacy_body_sources[name])
            for name in legacy_body_sources
        )
        and normalized_body(final_body_sources["feedback_report_business_snapshot"])
        == expected_snapshot_body
        and "'business_snapshot_v1'::text as snapshot_revision"
        in expected_snapshot_body
        and normalized_session_attempt.count("events.session_id = p_session_id") == 6
        and normalized_session_attempt.count("events.report_attempt_id = p_attempt_id")
        == 6
        and normalized_body(
            final_body_sources["feedback_report_business_snapshot"]
        ).count("parameters.session_id")
        == 6
        and normalized_body(
            final_body_sources["feedback_report_business_snapshot"]
        ).count("parameters.attempt_id")
        == 6
        and all(
            marker in hosted_verifier
            for marker in (
                "session_attempt_filters_verified = False",
                "session_attempt_filters_verified = True",
                "session_attempt_filters_verified=session_attempt_filters_verified",
            )
        )
    )
    reaction_store_contract = all(
        marker in normalized_reaction
        for marker in (
            "add constraint feedback_events_event_type_check",
            "'question_reaction_submitted'",
            "feedback_events_question_reaction_payload_check",
            "feedback_events_question_presented_payload_check",
            "payload ->> 'reaction' = 'surprise'",
            "jsonb_typeof(payload -> 'value') = 'boolean'",
            "payload ->> 'timing' = 'after_reveal'",
            "event_type <> 'question_presented'",
            "jsonb_typeof(payload -> 'propensity') = 'number'",
            "create or replace function public.feedback_ingest_events",
            "pg_catalog.pg_advisory_xact_lock",
            "insert into public.feedback_event_conflicts",
            "grant execute on function public.feedback_ingest_events",
        )
    ) and all(
        forbidden not in normalized_reaction
        for forbidden in (
            "disable row level security",
            "drop trigger",
            "grant insert on table public.feedback_events",
        )
    )
    surprise_report_contract = (
        normalized_surprise_report.startswith("begin; create function")
        and normalized_surprise_report.endswith("commit;")
        and normalized_surprise_report.count(
            "create function public.feedback_report_surprise_"
        )
        == 2
        and all(
            marker in normalized_surprise_report
            for marker in (
                "create function public.feedback_report_surprise_questions(",
                "create function public.feedback_report_surprise_quality(",
                "from public.feedback_authoritative_events",
                "registry_status = 'matched'",
                "event_type = 'question_reaction_submitted'",
                "event_type = 'answer_submitted'",
                "row_number() over",
                "reactions.reaction_rank = 1",
                "as observed_surprise_rate",
                "as posterior_mean",
                "as counts_conserved",
                "as orphan_breakdown_conserved",
                "grant execute on function public.feedback_report_surprise_questions(",
                "grant execute on function public.feedback_report_surprise_quality(",
            )
        )
        and all(
            forbidden not in normalized_surprise_report
            for forbidden in (
                "create or replace",
                "drop function",
                "alter table",
                "create function public.feedback_report_business_snapshot",
                "p_limit",
                "security definer",
                "disable row level security",
            )
        )
    )
    return [
        _check(
            "contract.lockdown.only_writer_lockdown",
            only_lockdown,
            "12500 contains only the direct-writer lockdown transaction.",
            "12500 contains statements outside the exact writer-lockdown boundary.",
        ),
        _check(
            "contract.report.exact_columns",
            sql_columns is not None and sql_columns == client_columns,
            (
                "13000 report columns exactly match the dynamically extracted "
                "Python client columns."
            ),
            "13000 report columns and Python client columns do not exactly match.",
        ),
        _check(
            "contract.report.raw_views_hardened",
            raw_views_hardened,
            (
                "13500 preserves private raw views while fixing version counts, "
                "attempt identity, known-answer accuracy, and safe optional integers."
            ),
            "13500 no longer matches the required private raw-view hardening contract.",
        ),
        _check(
            "contract.report.strict_client",
            _strict_report_client(client),
            "The Python client rejects missing/unknown columns and validates ingestion facts.",
            "The Python client no longer provides strict ingestion-row validation.",
        ),
        _check(
            "contract.report.authoritative_registry_schema",
            registry_contract,
            (
                "14000 defines a private append-only release/question/choice "
                "registry, dynamic authority projection, and exact quality surfaces."
            ),
            "14000 no longer matches the authoritative registry/ACL contract.",
        ),
        _check(
            "contract.report.authoritative_registry_data",
            registry_data_contract,
            (
                "The reviewed 60-question registry JSON and insert-only data "
                "migration exactly match a fresh full-bundle attested rebuild."
            ),
            (
                "The registry export, full-bundle attestation, artifact, manifest, "
                "or data migration disagree."
            ),
        ),
        _check(
            "contract.report.authoritative_business_cutover",
            authoritative_contract,
            (
                "15000 preserves business RPC schemas while deriving all dimensions "
                "and correctness from registry-matched events."
            ),
            "15000 or the strict client no longer matches the authority cutover.",
        ),
        _check(
            "contract.report.authoritative_details",
            detail_contract,
            (
                "16000 exposes strict paginated answer/proposal details using only "
                "registry-matched dimensions and server-derived answer facts."
            ),
            (
                "16000, its ACLs, or the strict detail-report client no longer "
                "matches the authoritative detail contract."
            ),
        ),
        _check(
            "contract.report.atomic_business_snapshot",
            snapshot_contract,
            (
                "17000 exposes one service-role-only MVCC snapshot RPC for all "
                "six authoritative business pages with a strict Python parser."
            ),
            (
                "17000, its ACL/single-statement page assembly, or the strict "
                "business-snapshot client contract no longer matches."
            ),
        ),
        _check(
            "contract.report.session_attempt_filters",
            session_attempt_contract,
            (
                "18000 forward-replaces the seven business RPC signatures with "
                "exact session/attempt predicates while preserving historical "
                "positional prefixes and business_snapshot_v1 semantics."
            ),
            (
                "18000 has a signature, predicate, forwarding, ACL, body-parity, "
                "or hosted-evidence contract mismatch."
            ),
        ),
        _check(
            "contract.report.question_reaction_store",
            reaction_store_contract,
            (
                "19000 adds the strict surprise-reaction payload and RPC enum "
                "without weakening append-only storage or conflict handling."
            ),
            (
                "19000 no longer matches the required reaction payload, RPC, "
                "append-only, or conflict-handling contract."
            ),
        ),
        _check(
            "contract.report.surprise_aggregate",
            surprise_report_contract,
            (
                "20000 adds service-role-only authoritative surprise and quality "
                "RPCs without changing business_snapshot_v1."
            ),
            (
                "20000 no longer matches the authoritative first-valid-reaction, "
                "quality-conservation, ACL, or snapshot-compatibility contract."
            ),
        ),
    ]


def _function_has_calls(source: str, function_name: str, forbidden: set[str]) -> bool:
    """Return whether a Python function contains any forbidden call names."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        return True
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
            return True
    return False


def _function_call_count(source: str, function_name: str, call_name: str) -> int:
    """Count direct/name-or-attribute calls inside one top-level function."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        return -1
    return sum(
        1
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    )


def _function_finally_call_count(
    source: str,
    function_name: str,
    call_name: str,
) -> int:
    """Count matching calls made from a top-level function's finally blocks."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        return -1
    return sum(
        1
        for try_node in ast.walk(function)
        if isinstance(try_node, ast.Try)
        for statement in try_node.finalbody
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    )


def _function_first_call_position(
    source: str,
    function_name: str,
    call_name: str,
) -> tuple[int, int] | None:
    """Return the first matching call position inside one top-level function."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        return None
    positions = [
        (node.lineno, node.col_offset)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == call_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == call_name)
        )
    ]
    return min(positions) if positions else None


def _function_has_leading_negative_guard(
    source: str,
    function_name: str,
    flag_name: str,
    exception_name: str,
) -> bool:
    """Require ``if not flag: raise Exception(...)`` before other statements."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        return False
    statements = list(function.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements.pop(0)
    if not statements or not isinstance(statements[0], ast.If):
        return False
    guard = statements[0]
    if not (
        isinstance(guard.test, ast.UnaryOp)
        and isinstance(guard.test.op, ast.Not)
        and isinstance(guard.test.operand, ast.Name)
        and guard.test.operand.id == flag_name
        and not guard.orelse
        and len(guard.body) == 1
        and isinstance(guard.body[0], ast.Raise)
    ):
        return False
    raised = guard.body[0].exc
    return (
        isinstance(raised, ast.Call)
        and isinstance(raised.func, ast.Name)
        and raised.func.id == exception_name
    )


def _report_app_checks(sources: Mapping[str, bytes | str]) -> list[Check]:
    edge = _text(sources, REPORT_EDGE) or ""
    query = _text(sources, REPORT_QUERY) or ""
    client = _text(sources, REPORT_CLIENT) or ""
    package = _text(sources, REPORT_PACKAGE) or ""
    app = _text(sources, REPORT_APP) or ""
    ui = _text(sources, REPORT_UI) or ""
    requirements = _text(sources, REQUIREMENTS) or ""
    project_metadata = _text(sources, PROJECT_METADATA) or ""
    postgres_acceptance = _text(sources, POSTGRES_ACCEPTANCE_VERIFIER) or ""
    deployment_ledger = _text(sources, DEPLOYMENT_LEDGER_TOOL) or ""
    deployment_ledger_readme = _text(sources, DEPLOYMENT_LEDGER_README) or ""
    inspector_app = _text(sources, INSPECTOR_APP) or ""
    inspector_outbox = _text(sources, INSPECTOR_OUTBOX) or ""
    inspector_recovery = _text(sources, INSPECTOR_RECOVERY) or ""
    inspector_release = _text(sources, INSPECTOR_RELEASE_MANIFEST) or ""

    ts_views = _extract_ts_string_collection(query, "REPORT_VIEWS")
    python_views = _python_dict_keys(client, "_VIEW_SPECS")
    edge_single = _extract_ts_string_collection(edge, "SINGLE_ROW_REPORT_VIEWS")
    python_single = extract_python_string_constant(
        client,
        "_SINGLE_ROW_VIEWS",
        allow_unordered=True,
    )
    views_match = (
        ts_views == _EXPECTED_REPORT_VIEWS
        and python_views == ts_views
        and edge_single is not None
        and frozenset(edge_single)
        == frozenset(
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
        and python_single is not None
        and frozenset(python_single) == frozenset(edge_single)
    )
    query_contract = (
        all(
            marker in query
            for marker in (
                'query.view === "feedback_report_ingestion_summary"',
                'query.view === "feedback_report_authority_status"',
                'query.view === "feedback_report_business_snapshot"',
                'query.view === "feedback_report_registry_quality"',
                'query.view === "feedback_report_event_resolution"',
                "p_from: query.from",
                "p_to: query.to",
                "p_request_id: query.requestId",
                "p_event_id: query.eventId",
                "p_limit: query.limit",
                "const codePointLength = Array.from(value).length",
            )
        )
        and query.count("p_limit: query.limit") == 1
    )
    edge_contract = (
        'request.method !== "GET"' in edge
        and "`/rest/v1/rpc/${query.view}`" in edge
        and "/rest/v1/feedback_events" not in edge
        and "parseReportQuery(request.url)" in edge
        and query_contract
        and edge.count("await fetch(endpoint") == 1
        and 'query.view === "feedback_report_business_snapshot"' in edge
        and "rawRowsJson = await response.text()" in edge
        and "value = JSON.parse(rawRowsJson)" in edge
        and "report.rawRowsJson ?? JSON.stringify(report.rows)" in edge
        and '"rows":${rowsJson}' in edge
        and "value.length" not in query
    )
    package_contract = all(
        marker in package
        for marker in (
            "ReportsClient",
            "BusinessSnapshot",
            "BUSINESS_SNAPSHOT_VIEW",
            "validate_business_snapshot_response",
            "validate_report_response",
            "REPORT_VIEWS",
            "SURPRISE_QUESTIONS_VIEW",
            "SURPRISE_QUALITY_VIEW",
        )
    )
    surprise_client_contract = all(
        marker in client
        for marker in (
            "_SURPRISE_QUESTION_COLUMNS = (",
            "_SURPRISE_QUALITY_COLUMNS = (",
            "def _validate_surprise_question_row(",
            "def _validate_surprise_quality_row(",
            "def fetch_surprise_questions(",
            "def fetch_surprise_quality(",
        )
    )
    app_contract = (
        all(
            marker in app
            for marker in (
                'INGESTION_VIEW = "feedback_report_ingestion_summary"',
                'REGISTRY_VIEW = "feedback_report_registry_quality"',
                'ANSWERS_VIEW = "feedback_report_answers"',
                'PROPOSALS_VIEW = "feedback_report_proposals"',
                "SURPRISE_QUESTIONS_VIEW",
                "SURPRISE_QUALITY_VIEW",
                "business_snapshot = client.fetch_business_snapshot(",
                "client.fetch_surprise_questions",
                "client.fetch_surprise_quality",
                '"Surprise"',
                '"Observed surprise by question"',
                '"Reaction quality"',
                '"snapshot_revision": business_snapshot.snapshot_revision',
                '"reports_snapshot_metadata": None',
                '"All six business tabs share one PostgreSQL MVCC snapshot',
                "business_snapshot_v1 with",
                '("ingestion", INGESTION_VIEW)',
                '("registry", REGISTRY_VIEW)',
                "validate_report_response(INGESTION_VIEW",
                "validate_report_response(REGISTRY_VIEW",
                '"Ingestion observability"',
                '"Registry quality"',
                '"Data quality"',
                '"Answers"',
                '"Proposals"',
                "build_data_quality_signals(",
            )
        )
        and "AUTHORITY_STATUS_VIEW" not in app
        and "_authority_page" not in app
        and _function_call_count(app, "_refresh_reports", "fetch_business_snapshot")
        == 1
        and _function_call_count(app, "_refresh_reports", "fetch_page") == 1
        and _function_call_count(
            app,
            "_refresh_reports",
            "_refresh_surprise_reports",
        )
        == 1
    )
    data_quality_is_local = not _function_has_calls(
        app,
        "_render_data_quality",
        {"fetch", "fetch_page", "urlopen", "_fetch_snapshot", "_refresh_reports"},
    )
    ui_contract = all(
        marker in ui
        for marker in (
            '"conflicting_event_count"',
            '"conflict_audit_event_count"',
            '"excluded_event_count"',
            '"client_correctness_mismatch_answer_count"',
            '"feedback_report_answers": "answers"',
            '"feedback_report_proposals": "proposals"',
            'severity="error"',
        )
    )
    runtime_contract = (
        "streamlit>=1.32" in requirements
        and re.search(r"(?m)^-e\s+\.\s*$", requirements) is not None
        and re.search(
            r'(?m)^readme\s*=\s*["\']README\.md["\']\s*$',
            project_metadata,
        )
        is not None
    )
    postgres_migration_versions = extract_python_string_constant(
        postgres_acceptance,
        "EXPECTED_MIGRATION_VERSIONS",
    )
    postgres_production_tokens = extract_python_string_constant(
        postgres_acceptance,
        "_PRODUCTION_TARGET_TOKENS",
        allow_unordered=True,
    )
    postgres_non_production_tokens = extract_python_string_constant(
        postgres_acceptance,
        "_NON_PRODUCTION_TARGET_TOKENS",
        allow_unordered=True,
    )
    expected_postgres_versions = tuple(
        migration.split("_", maxsplit=1)[0]
        for migration in EXPECTED_MIGRATION_INVENTORY
    )
    postgres_acceptance_markers = (
        'DSN_ENV: Final = "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN"',
        'TARGET_ENV: Final = "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_TARGET"',
        'parser.add_argument("--confirm-staging", action="store_true")',
        "if not args.confirm_staging:",
        "dsn = environ.get(DSN_ENV)",
        'token.startswith(("live", "main", "prod"))',
        "production-like target labels are forbidden",
        "target label must explicitly identify staging",
        "EXPECTED_FUNCTIONS: Final = (",
        "EXPECTED_TABLE_GRANTS: Final = {",
        "EXPECTED_TRIGGERS: Final = {",
        "EXPECTED_CONSTRAINTS: Final = {",
        "architecture_iq_acceptance:migrations",
        "architecture_iq_acceptance:functions",
        "architecture_iq_acceptance:rls",
        "architecture_iq_acceptance:table_grants",
        "architecture_iq_acceptance:triggers",
        "architecture_iq_acceptance:constraints",
        "architecture_iq_acceptance:registry_authority",
        "architecture_iq_acceptance:registry_content",
        "architecture_iq_acceptance:probe_residue",
        "pg_catalog.has_function_privilege('anon'",
        "pg_catalog.has_function_privilege('authenticated'",
        "pg_catalog.has_function_privilege('service_role'",
        "tuple(row[8:11]) == (False, False, True)",
        "relations.relrowsecurity",
        "relations.relforcerowsecurity",
        "observed_rls[name] == (True, True, 0)",
        "pg_catalog.has_table_privilege",
        'roles = ("anon", "authenticated", "service_role")',
        "triggers.tgenabled",
        "constraints.condeferrable",
        "constraints.convalidated",
        "set(observed_triggers) == set(EXPECTED_TRIGGERS)",
        "set(observed_constraints) == set(EXPECTED_CONSTRAINTS)",
        "feedback_events_append_only",
        "feedback_quiz_choice_inventory_complete",
        "feedback_quiz_question_release_inventory_complete",
        "feedback_quiz_choice_release_inventory_complete",
        "feedback_event_conflicts_event_id_fkey",
        "feedback_quiz_questions_correct_choice_fkey",
        "feedback_quiz_choices_letter_candidate_key",
        "SAVEPOINT {savepoint}",
        "ROLLBACK TO SAVEPOINT {savepoint}",
        "RELEASE SAVEPOINT {savepoint}",
        'expected_sqlstate="55000"',
        'expected_sqlstate="23514"',
        "set constraints all immediate",
        "release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f",
        "9fa3c9e28aa81dffd7ea751be40245d1f62f01c252b91024e62de0d8bb230005",
        "registry_db3f1a166af0b526e08d4eff49539c6a2150653d1940b0fcccbdbfbe0b525131",
        "CURRENT_QUESTION_COUNT: Final = 60",
        "CURRENT_CHOICE_COUNT: Final = 180",
        "def _registry_content_id(",
        "hashlib.sha256(canonical).hexdigest()",
        '"registry_v1"',
        '"detail_v1"',
    )
    postgres_acceptance_checks = (
        "postgres.identity",
        "postgres.migrations",
        "postgres.functions",
        "postgres.function_grants",
        "postgres.rls",
        "postgres.table_grants",
        "postgres.triggers",
        "postgres.constraints",
        "postgres.registry_authority",
        "postgres.append_only_probes",
        "postgres.registry_counterexamples",
        "postgres.rollback",
        "postgres.close",
    )
    postgres_acceptance_tables = (
        "feedback_events",
        "feedback_ingest_request_outcomes",
        "feedback_event_conflicts",
        "feedback_quiz_releases",
        "feedback_quiz_questions",
        "feedback_quiz_choices",
    )
    postgres_acceptance_contract = (
        postgres_migration_versions == expected_postgres_versions
        and frozenset(postgres_production_tokens or ())
        == {"live", "main", "prod", "production"}
        and frozenset(postgres_non_production_tokens or ())
        == {
            "dev",
            "development",
            "preview",
            "qa",
            "sandbox",
            "stage",
            "staging",
            "test",
            "testing",
        }
        and all(marker in postgres_acceptance for marker in postgres_acceptance_markers)
        and all(
            f'"{name}"' in postgres_acceptance
            for name in (
                "feedback_ingest_events",
                "feedback_logical_event_v1",
                *_EXPECTED_REPORT_VIEWS,
                *postgres_acceptance_tables,
            )
        )
        and all(
            f'"{code}"' in postgres_acceptance for code in postgres_acceptance_checks
        )
        and 'parser.add_argument("--dsn"' not in postgres_acceptance
        and "connection.commit(" not in postgres_acceptance
        and "pg_catalog.coalesce(" not in postgres_acceptance
        and "hosted_verified" not in postgres_acceptance
        and _function_call_count(postgres_acceptance, "run_acceptance", "rollback") == 1
        and _function_call_count(postgres_acceptance, "run_acceptance", "close") == 1
        and _function_call_count(postgres_acceptance, "run_acceptance", "commit") == 0
        and _function_finally_call_count(
            postgres_acceptance,
            "run_acceptance",
            "rollback",
        )
        == 1
        and _function_finally_call_count(
            postgres_acceptance,
            "run_acceptance",
            "close",
        )
        == 1
        and 'postgres-acceptance = ["psycopg[binary]>=3.1"]' in project_metadata
    )
    deployment_event_types = extract_python_string_constant(
        deployment_ledger,
        "EVENT_TYPES",
        allow_unordered=True,
    )
    deployment_evidence_events = extract_python_string_constant(
        deployment_ledger,
        "EVIDENCE_EVENTS",
        allow_unordered=True,
    )
    deployment_terminal_events = extract_python_string_constant(
        deployment_ledger,
        "TERMINAL_EVENTS",
        allow_unordered=True,
    )
    deployment_mapping_keys = extract_python_string_constant(
        deployment_ledger,
        "MAPPING_KEYS",
        allow_unordered=True,
    )
    confirmation_position = _function_first_call_position(
        deployment_ledger,
        "append_event",
        "ConfirmationRequired",
    )
    first_mutation_positions = tuple(
        _function_first_call_position(deployment_ledger, "append_event", call_name)
        for call_name in ("mkdir", "open", "_atomic_replace")
    )
    report_app_paths = deployment_paths_for_phase("report-app")
    deployment_ledger_contract = (
        frozenset(deployment_event_types or ())
        == {
            "candidate_attested",
            "deployment_declared",
            "postgres_accepted",
            "roundtrip_accepted",
            "source_mapping_attested",
            "activated",
            "superseded",
            "rolled_back",
        }
        and frozenset(deployment_evidence_events or ())
        == {"postgres_accepted", "roundtrip_accepted", "source_mapping_attested"}
        and frozenset(deployment_terminal_events or ()) == {"superseded", "rolled_back"}
        and frozenset(deployment_mapping_keys or ())
        == {
            "schema_version",
            "evidence_type",
            "captured_at",
            "mapping_authority",
            "provider_export",
            "provider",
            "project_id",
            "deploy_id",
            "site_url",
            "environment",
            "target_label",
            "backend_project_id",
            "ingest_origin_sha256",
            "report_origin_sha256",
            "deployment_context_id",
            "deployed_at",
            "deployment_status",
            "repo_url",
            "branch",
            "source_commit",
            "entrypoint",
            "release_id",
            "manifest_sha256",
            "registry_id",
            "rollout_input_fingerprint",
        }
        and all(
            marker in deployment_ledger
            for marker in (
                'RECORD_TYPE = "architecture_iq_deployment_event"',
                'POSTGRES_EVIDENCE_TYPE = "architecture_iq_postgres_staging_acceptance"',
                'ROUNDTRIP_EVIDENCE_TYPE = "architecture_iq_hosted_feedback_roundtrip"',
                'MAPPING_EVIDENCE_TYPE = "architecture_iq_provider_deployment_mapping"',
                'return "ACTIVATED_REVIEWED"',
                'return "READY_FOR_REVIEWED_ACTIVATION"',
                'return "DEPLOYMENT_DECLARED_SOURCE_MAPPING_UNVERIFIED"',
                'document.get("authority_mode") != "authoritative"',
                'document.get("mapping_authority")\n'
                '        != "reviewed_provider_control_plane_capture"',
                '"source mapping must be a reviewed provider-control-plane capture"',
                'identifier = f"deployment_context_{hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()}"',
                '"backend_project_id": str(declaration["backend_project_id"])',
                '"ingest_origin_sha256": str(declaration["ingest_origin_sha256"])',
                '"report_origin_sha256": str(declaration["report_origin_sha256"])',
                'frozenset({"path", "sha256", "media_type"})',
                "if hashlib.sha256(raw).hexdigest() != expected_hash:",
                'document.get("deployment_status") != "ready"',
                "export_path, export_sha256, export_raw = _evidence_bytes(",
                "if export_path == _path:",
                "if not export_raw:",
                '"provider export media type is unsupported"',
                '_git(repo_root, ("cat-file", "-e", f"{commit}^{{commit}}"))',
                "_rollout_fingerprint_from_commit(repo_root, commit, preflight_paths)",
                '"preflight fingerprint does not match source commit blobs"',
                '_git_blob(repo_root, commit, manifest_path, label="candidate.manifest.path")',
                '_git_blob(repo_root, commit, registry_path, label="candidate.registry.path")',
                '_git_blob(repo_root, commit, entrypoint, label="candidate.source.entrypoint")',
                'if document["previous_record_sha256"] != previous_hash:',
                'if document["record_sha256"] != expected_hash:',
                'if _canonical_json_bytes(document) + b"\\n" != line:',
                "if not path.exists():",
                'raise ConfirmationRequired("append requires --confirm-append")',
                'append.add_argument("--confirm-append", action="store_true")',
                "if state.evidence != EVIDENCE_EVENTS:",
                '"activation requires PostgreSQL, roundtrip, and source-mapping evidence"',
                'reviewed_events = {"source_mapping_attested", "activated", *TERMINAL_EVENTS}',
                "if event_type in reviewed_events and reviewed_by is None:",
                'if event_type in reviewed_events and reviewed_by == record.get("recorded_by"):',
                '"source mapping, activation, and terminal events require reviewed_by"',
                '"reviewed decision requires a distinct reviewer"',
                '"hosted evidence SHA-256 is reused by another deployment"',
                '"roundtrip run, event, or request identity is reused by another deployment"',
                '"provider export SHA-256 is reused by another deployment"',
                '"hosted roundtrip evidence predates the provider-reported deployment"',
                "def _git_environment() -> dict[str, str]:",
                '"PATH": os.defpath',
                '"HOME": os.devnull',
                '"GIT_CONFIG_NOSYSTEM": "1"',
                '"GIT_NO_REPLACE_OBJECTS": "1"',
                '"GIT_OPTIONAL_LOCKS": "0"',
                '"GIT_TERMINAL_PROMPT": "0"',
                "env=_git_environment()",
            )
        )
        and _function_call_count(
            deployment_ledger,
            "append_event",
            "ConfirmationRequired",
        )
        == 1
        and _function_has_leading_negative_guard(
            deployment_ledger,
            "append_event",
            "confirm",
            "ConfirmationRequired",
        )
        and confirmation_position is not None
        and all(
            position is not None and confirmation_position < position
            for position in first_mutation_positions
        )
        and _function_call_count(
            deployment_ledger,
            "append_event",
            "_verify_ledger_bytes",
        )
        == 2
        and _function_call_count(
            deployment_ledger,
            "append_event",
            "_atomic_replace",
        )
        == 1
        and _function_call_count(
            deployment_ledger,
            "_validate_postgres",
            "_context_summary",
        )
        == 1
        and _function_call_count(
            deployment_ledger,
            "_validate_roundtrip",
            "_context_summary",
        )
        == 1
        and _function_call_count(
            deployment_ledger,
            "_validate_mapping",
            "_context_summary",
        )
        == 1
        and _function_call_count(
            deployment_ledger,
            "_git",
            "_git_environment",
        )
        == 1
        and "os.environ" not in deployment_ledger
        and DEPLOYMENT_LEDGER_TOOL in report_app_paths
        and DEPLOYMENT_LEDGER_README in report_app_paths
        and tuple(path for path in report_app_paths if path.startswith("deployments/"))
        == (DEPLOYMENT_LEDGER_README,)
        and "deployments/ledger.jsonl" not in report_app_paths
        and not any(
            path.startswith("deployments/evidence/") for path in report_app_paths
        )
        and all(
            marker in deployment_ledger_readme
            for marker in (
                "retrospective audit trail",
                "not runtime self-attestation",
                "does not manufacture provider proof",
                "successful operational state is named `ACTIVATED_REVIEWED`",
                "`READY_FOR_REVIEWED_ACTIVATION`; “ready” means ready for a maintainer decision",
                "distinct from the recorder",
                "raw provider-export hashes",
                "independently recomputed; dirty working-tree bytes and self-reported PASS",
                "remains an operator-reviewed claim rather than provider cryptographic proof",
                "A hand-written mapping",
                "envelope without the separately hash-bound capture is rejected",
                "`ledger.jsonl` need not exist before the first real event",
                "treat a missing ledger as an empty ledger",
                "external head pin needed to",
                "detect removal of a suffix; a hash chain by itself cannot prove that its last",
                "Without `--confirm-append`, no ledger, lock, directory, or placeholder record is",
            )
        )
    )
    inspector_runtime_git_env_names = extract_python_string_constant(
        inspector_app,
        "RUNTIME_GIT_SHA_ENV_NAMES",
    )
    inspector_runtime_git_contract = (
        inspector_runtime_git_env_names
        == (
            "ARCHITECTURE_IQ_GIT_SHA",
            "GIT_COMMIT",
            "COMMIT_SHA",
            "SOURCE_VERSION",
        )
        and all(
            marker in inspector_app
            for marker in (
                'GIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\\Z")',
                '["git", "rev-parse", "--verify", "HEAD^{commit}"]',
                "def _checkout_git_environment() -> dict[str, str]:",
                '"PATH": os.defpath',
                '"HOME": os.devnull',
                '"LANG": "C"',
                '"LC_ALL": "C"',
                '"GIT_CONFIG_NOSYSTEM": "1"',
                '"GIT_NO_REPLACE_OBJECTS": "1"',
                '"GIT_OPTIONAL_LOCKS": "0"',
                '"GIT_TERMINAL_PROMPT": "0"',
                "env=_checkout_git_environment()",
                "if result.returncode != 0 or GIT_SHA_PATTERN.fullmatch(candidate) is None:",
                "for name in RUNTIME_GIT_SHA_ENV_NAMES",
                "if any(GIT_SHA_PATTERN.fullmatch(raw) is None for raw in configured):",
                "checkout_sha = (git_reader or _checkout_git_sha)(root)",
                "if checkout_sha is None or GIT_SHA_PATTERN.fullmatch(checkout_sha) is None:",
                "candidates.add(checkout_sha.lower())",
                "if len(candidates) != 1:",
                "return next(iter(candidates))",
            )
        )
        and _function_call_count(inspector_app, "_checkout_git_sha", "run") == 1
        and _function_call_count(
            inspector_app,
            "_checkout_git_sha",
            "_checkout_git_environment",
        )
        == 1
        and "{**os.environ" not in inspector_app
    )
    inspector_feedback_contract = (
        all(
            marker in inspector_app
            for marker in (
                '"Upload pending session events"',
                "feedback_outbox.upload_pending_events(",
                "feedback_recovery.upload_recovered_trace(",
                '"Upload comment"',
                "release_manifest.load_quiz_manifest(root)",
            )
        )
        and all(
            marker in inspector_outbox
            for marker in (
                "MAX_EVENTS_PER_BATCH = 500",
                "MAX_BODY_BYTES = 1024 * 1024",
                "except feedback.FeedbackUploadConflictError",
            )
        )
        and all(
            marker in inspector_recovery
            for marker in (
                "MAX_RECOVERY_FILE_BYTES = 10 * 1024 * 1024",
                "feedback.parse_session_trace_json(encoded)",
                "feedback_outbox.upload_pending_events(",
            )
        )
        and all(
            marker in inspector_release
            for marker in (
                "def load_quiz_manifest(",
                "expected_release =",
                "def _parse_artifacts(",
                "artifact inventory does not match physical files",
            )
        )
    )
    return [
        _check(
            "contract.report_app.view_inventory",
            views_match,
            "Report SQL consumers share the exact allowlisted view/single-row inventory.",
            "Report Edge and Python view inventories do not match.",
        ),
        _check(
            "contract.report_app.read_only_edge",
            edge_contract,
            "feedback-report is a GET-only allowlisted RPC proxy with request correlation.",
            "feedback-report does not match the protected read-only RPC contract.",
        ),
        _check(
            "contract.report_app.strict_client_app",
            package_contract
            and surprise_client_contract
            and app_contract
            and runtime_contract,
            "The strict report client is wired into the Streamlit report application.",
            "The report client/application/runtime wiring is incomplete.",
        ),
        _check(
            "contract.report_app.local_quality",
            data_quality_is_local and ui_contract,
            "Data quality derives conflict signals locally without another report request.",
            "Data quality no longer matches the local conflict-signal contract.",
        ),
        _check(
            "contract.postgres.staging_acceptance",
            postgres_acceptance_contract,
            (
                "The rollback-only PostgreSQL staging verifier covers catalog, ACL, "
                "RLS, registry, and mutation-probe contracts."
            ),
            (
                "The PostgreSQL staging verifier, safety boundary, catalog coverage, "
                "or optional driver contract is incomplete."
            ),
        ),
        _check(
            "contract.deployment_ledger",
            deployment_ledger_contract,
            (
                "The retrospective deployment ledger binds source, evidence, "
                "reviewed transitions, and a canonical hash chain without "
                "fingerprinting post-deploy records."
            ),
            (
                "The deployment ledger source, documentation, state/evidence "
                "boundary, confirmation guard, or fingerprint exclusions are "
                "incomplete."
            ),
        ),
        _check(
            "contract.inspector.runtime_git_identity",
            inspector_runtime_git_contract,
            (
                "The inspector requires the checkout commit, uses only allowlisted "
                "full-SHA declarations as cross-checks, and fails closed on malformed "
                "or conflicting identity."
            ),
            (
                "The inspector runtime Git identity no longer matches the checkout, "
                "allowlist, or fail-closed conflict contract."
            ),
        ),
        _check(
            "contract.inspector.feedback_ui",
            inspector_feedback_contract,
            (
                "The deployed inspector wires pending upload, recovery, comments, "
                "conflict isolation, and release attestation together."
            ),
            (
                "The inspector deployment inputs no longer match the required "
                "feedback/recovery/release contract."
            ),
        ),
    ]


def evaluate_preflight(
    sources: Mapping[str, bytes | str],
    *,
    phase: str,
    git_evidence: GitEvidence,
    require_hosted: bool = False,
) -> PreflightResult:
    """Purely evaluate a phase from source bytes and sanitized Git facts."""

    paths = deployment_paths_for_phase(phase)
    present = tuple(path for path in paths if _text(sources, path) is not None)
    fingerprint = deployment_fingerprint(sources, paths)
    head_hashes = dict(git_evidence.head_blob_sha256)
    head_match = (
        "head_blobs" not in git_evidence.failed_queries
        and len(head_hashes) == len(paths)
        and all(
            path in sources
            and head_hashes.get(path)
            == hashlib.sha256(
                sources[path].encode("utf-8")
                if isinstance(sources[path], str)
                else sources[path]
            ).hexdigest()
            for path in paths
        )
    )
    checks: list[Check] = [
        _check(
            "inputs.present_utf8",
            len(present) == len(paths),
            "Every cumulative rollout/compatibility input exists and is valid UTF-8.",
            "One or more rollout/compatibility inputs are missing or invalid.",
        ),
        _check(
            "migrations.inventory_order",
            "migration_inventory" not in git_evidence.failed_queries
            and git_evidence.migration_inventory == EXPECTED_MIGRATION_INVENTORY,
            "The migration inventory exactly matches the classified staged order.",
            "The migration inventory is missing, reordered, or contains an unclassified SQL file.",
        ),
        _check(
            "git.sha",
            "sha" not in git_evidence.failed_queries and git_evidence.sha is not None,
            "The current Git commit SHA was resolved.",
            "The current Git commit SHA could not be resolved safely.",
        ),
        _check(
            "git.inputs_tracked",
            "tracked" not in git_evidence.failed_queries
            and all(path in git_evidence.tracked_paths for path in paths),
            "Every cumulative rollout/compatibility input is tracked by Git.",
            "One or more rollout/compatibility inputs are untracked or tracking is unknown.",
        ),
        _check(
            "git.inputs_clean",
            "clean" not in git_evidence.failed_queries
            and not (set(paths) & git_evidence.dirty_paths),
            "Every rollout/compatibility input is clean against the Git commit.",
            "One or more rollout/compatibility inputs are dirty or cleanliness is unknown.",
        ),
        _check(
            "git.inputs_match_head",
            head_match,
            "Every input byte exactly matches its blob in the reported Git commit.",
            "One or more input bytes differ from the reported Git commit or are unknown.",
        ),
        _check(
            "inputs.fingerprint",
            fingerprint is not None,
            "The exact checked rollout-input SHA-256 fingerprint was computed.",
            "The checked rollout-input fingerprint could not be computed.",
        ),
    ]

    selected_index = PHASES.index(phase)
    checks.extend(_expand_checks(sources))
    if selected_index >= PHASES.index("ingest-cutover"):
        checks.extend(_ingest_checks(sources))
    if selected_index >= PHASES.index("lockdown-report"):
        checks.extend(
            _lockdown_report_checks(sources, git_evidence.registry_attestation)
        )
    if selected_index >= PHASES.index("report-app"):
        checks.extend(_report_app_checks(sources))

    checks.append(
        Check(
            "hosted.acceptance",
            UNVERIFIED,
            (
                "Hosted migration state, Edge revisions, secrets, ACL/RLS, RPC behavior, "
                "and roundtrip/conflict preservation were not contacted or verified."
            ),
        )
    )
    return PreflightResult(
        phase=phase,
        checks=tuple(checks),
        git_sha=git_evidence.sha,
        checked_rollout_input_sha256=fingerprint,
        require_hosted=require_hosted,
    )


def run_preflight(
    repo_root: Path,
    *,
    phase: str,
    require_hosted: bool = False,
    git_runner: GitRunner | None = None,
) -> PreflightResult:
    """Collect local read-only evidence, then invoke the pure evaluator."""

    paths = deployment_paths_for_phase(phase)
    sources, _missing = load_sources(repo_root, paths)
    git_evidence = collect_git_evidence(repo_root, paths, runner=git_runner)
    return evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=git_evidence,
        require_hosted=require_hosted,
    )


def render_json(result: PreflightResult) -> str:
    """Render the deliberately non-secret machine-readable envelope."""

    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_text(result: PreflightResult) -> str:
    """Render a human summary that cannot be mistaken for hosted readiness."""

    fingerprint = result.checked_rollout_input_sha256 or "UNAVAILABLE"
    git_sha = result.git_sha or "UNAVAILABLE"
    lines = [
        "ArchitectureIQ feedback rollout preflight (LOCAL STATIC ONLY)",
        "Rollout contract: staged upgrade",
        (
            "Baseline migrations are fingerprinted compatibility inputs; this is not "
            "an instruction to reapply them."
        ),
        f"Phase: {result.phase}",
        f"Git SHA: {git_sha}",
        f"Checked rollout input SHA-256: {fingerprint}",
        "",
    ]
    lines.extend(
        f"[{check.status}] {check.code}: {check.summary}" for check in result.checks
    )
    lines.extend(
        (
            "",
            f"Static overall: {result.static_overall}",
            f"Overall: {result.overall}",
            "Hosted verified: false",
            "Deploy ready: false",
        )
    )
    if result.static_overall == PASS:
        lines.append("STATIC PASS — HOSTED UNVERIFIED — NOT DEPLOY-READY")
    else:
        lines.append("STATIC FAIL — HOSTED UNVERIFIED — NOT DEPLOY-READY")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline, read-only static preflight for the staged feedback rollout; "
            "it never accepts hosted credentials or endpoints."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--require-hosted",
        action="store_true",
        help="return 2 when the intentionally offline hosted check is UNVERIFIED",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    git_runner: GitRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root or Path(__file__).resolve().parents[1]
    result = run_preflight(
        root,
        phase=args.phase,
        require_hosted=args.require_hosted,
        git_runner=git_runner,
    )
    print(render_json(result) if args.json_output else render_text(result))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
