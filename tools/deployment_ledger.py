"""Retrospective, append-only deployment evidence ledger for ArchitectureIQ.

The ledger records reviewed deployment facts after they exist.  It deliberately
does not turn an operator claim, a local preflight, or runtime self-reporting
into provider attestation.  Provider/source mapping requires a separately
hash-bound raw control-plane capture plus distinct maintainer review, and the
result remains reviewed evidence rather than a provider-signed proof.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence, TextIO
from urllib.parse import urlsplit, urlunsplit
import uuid


SCHEMA_VERSION = "1.0"
RECORD_TYPE = "architecture_iq_deployment_event"
POSTGRES_EVIDENCE_TYPE = "architecture_iq_postgres_staging_acceptance"
ROUNDTRIP_EVIDENCE_TYPE = "architecture_iq_hosted_feedback_roundtrip"
MAPPING_EVIDENCE_TYPE = "architecture_iq_provider_deployment_mapping"

EVENT_TYPES = frozenset(
    {
        "candidate_attested",
        "deployment_declared",
        "postgres_accepted",
        "roundtrip_accepted",
        "source_mapping_attested",
        "activated",
        "superseded",
        "rolled_back",
    }
)
EVIDENCE_EVENTS = frozenset(
    {"postgres_accepted", "roundtrip_accepted", "source_mapping_attested"}
)
TERMINAL_EVENTS = frozenset({"superseded", "rolled_back"})

MAX_LEDGER_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_DRAFT_BYTES = 1024 * 1024
MAX_RECORDS = 100_000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 500_000
MAX_STRING_CHARS = 4 * 1024 * 1024
MAX_SAFE_INTEGER = 2**53 - 1

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RELEASE_PATTERN = re.compile(r"^release_[0-9a-f]{64}$")
REGISTRY_PATTERN = re.compile(r"^registry_[0-9a-f]{64}$")
DEPLOYMENT_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SAFE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
SAFE_BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
RFC3339_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)

RECORD_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "event_type",
        "deployment_key",
        "recorded_at",
        "recorded_by",
        "reviewed_by",
        "facts",
        "previous_record_sha256",
        "record_sha256",
    }
)
DRAFT_KEYS = RECORD_KEYS - {"previous_record_sha256", "record_sha256"}

MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "generated_at",
        "source_runs",
        "questions",
        "counts",
        "artifacts",
    }
)
REGISTRY_KEYS = frozenset(
    {
        "schema_version",
        "registry_id",
        "release_id",
        "manifest_sha256",
        "question_count",
        "choice_count",
        "questions",
    }
)
PREFLIGHT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "rollout_contract",
        "fingerprint_scope",
        "baseline_migrations_are_compatibility_inputs",
        "phase",
        "static_overall",
        "overall",
        "hosted_verified",
        "deploy_ready",
        "require_hosted",
        "git_sha",
        "checked_rollout_input_paths",
        "checked_rollout_input_sha256",
        "checks",
    }
)
POSTGRES_KEYS = frozenset(
    {
        "schema_version",
        "evidence_type",
        "accepted",
        "target_label",
        "database_contacted",
        "transaction_rolled_back",
        "server",
        "registry",
        "checks",
        "summary",
    }
)
ROUNDTRIP_KEYS = frozenset(
    {
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
)
MAPPING_KEYS = frozenset(
    {
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
)


class LedgerError(ValueError):
    """Base class for safe, user-facing ledger validation failures."""


class LedgerFormatError(LedgerError):
    """The JSONL or draft encoding is invalid."""


class EvidenceValidationError(LedgerError):
    """A referenced immutable evidence artifact is invalid or inconsistent."""


class StateTransitionError(LedgerError):
    """An event is invalid for the deployment's replayed state."""


class ConfirmationRequired(LedgerError):
    """A mutating command did not receive explicit confirmation."""


@dataclass
class _DeploymentState:
    deployment_key: str
    release_id: str
    candidate: Mapping[str, Any]
    declaration: Mapping[str, Any] | None = None
    evidence: set[str] = field(default_factory=set)
    evidence_metadata: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    activated: bool = False
    terminal_event: str | None = None

    @property
    def status(self) -> str:
        if self.terminal_event is not None:
            return self.terminal_event.upper()
        if self.activated:
            return "ACTIVATED_REVIEWED"
        if self.declaration is None:
            return "CANDIDATE_ATTESTED"
        if "source_mapping_attested" not in self.evidence:
            return "DEPLOYMENT_DECLARED_SOURCE_MAPPING_UNVERIFIED"
        if self.evidence == EVIDENCE_EVENTS:
            return "READY_FOR_REVIEWED_ACTIVATION"
        if self.evidence:
            return "EVIDENCE_PARTIAL"
        return "DEPLOYMENT_DECLARED_SOURCE_MAPPING_UNVERIFIED"


@dataclass(frozen=True)
class DeploymentSummary:
    deployment_key: str
    release_id: str
    status: str
    evidence: tuple[str, ...]
    provider: str | None
    project_id: str | None
    deploy_id: str | None
    site_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_key": self.deployment_key,
            "release_id": self.release_id,
            "status": self.status,
            "evidence": list(self.evidence),
            "provider": self.provider,
            "project_id": self.project_id,
            "deploy_id": self.deploy_id,
            "site_url": self.site_url,
        }


@dataclass(frozen=True)
class LedgerSnapshot:
    records: tuple[Mapping[str, Any], ...]
    deployments: tuple[DeploymentSummary, ...]
    head_record_sha256: str | None

    def verification_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "record_count": len(self.records),
            "deployment_count": len(self.deployments),
            "head_record_sha256": self.head_record_sha256,
        }

    def listing_dict(self) -> dict[str, Any]:
        return {
            **self.verification_dict(),
            "deployments": [item.to_dict() for item in self.deployments],
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    del value
    raise LedgerFormatError("JSON non-finite numeric constants are forbidden")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerFormatError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def _validate_json_subset(value: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise LedgerFormatError("JSON document exceeds the safe node limit")
        if depth > MAX_JSON_DEPTH:
            raise LedgerFormatError("JSON document exceeds the safe nesting limit")
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, int):
            if abs(current) > MAX_SAFE_INTEGER:
                raise LedgerFormatError(
                    "JSON integer exceeds the interoperable safe range"
                )
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise LedgerFormatError("JSON number must be finite")
            continue
        if isinstance(current, str):
            if len(current) > MAX_STRING_CHARS:
                raise LedgerFormatError("JSON string exceeds the safe length limit")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in current):
                raise LedgerFormatError(
                    "JSON strings must not contain unpaired surrogates"
                )
            continue
        if isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
            continue
        if isinstance(current, dict):
            stack.extend((key, depth + 1) for key in current)
            stack.extend((item, depth + 1) for item in current.values())
            continue
        raise LedgerFormatError("JSON contains a non-interoperable value")


def _parse_json_bytes(raw: bytes, *, label: str, limit: int) -> Any:
    if len(raw) > limit:
        raise LedgerFormatError(f"{label} exceeds the safe byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerFormatError(f"{label} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except LedgerFormatError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LedgerFormatError(f"{label} is not strict JSON") from exc
    _validate_json_subset(value)
    return value


def _require_object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LedgerFormatError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise LedgerFormatError(f"{label} does not use the exact schema")


def _safe_text(value: Any, *, label: str, maximum: int = 255, minimum: int = 1) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise LedgerFormatError(f"{label} must be a bounded non-empty string")
    if value != value.strip() or any(ord(character) < 0x20 for character in value):
        raise LedgerFormatError(
            f"{label} contains unsafe whitespace or control characters"
        )
    return value


def _nullable_safe_text(value: Any, *, label: str, maximum: int = 255) -> str | None:
    if value is None:
        return None
    return _safe_text(value, label=label, maximum=maximum)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise LedgerFormatError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LedgerFormatError(f"{label} must be an integer of at least {minimum}")
    return value


def _utc_datetime(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or RFC3339_UTC_PATTERN.fullmatch(value) is None:
        raise LedgerFormatError(
            f"{label} must be an RFC 3339 UTC timestamp ending in Z"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerFormatError(f"{label} is not a valid timestamp") from exc


def _utc_timestamp(value: Any, *, label: str) -> str:
    _utc_datetime(value, label=label)
    assert isinstance(value, str)
    return value


def _safe_repo_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SAFE_PATH_PATTERN.fullmatch(value) is None:
        raise LedgerFormatError(f"{label} must be a safe repository-relative path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise LedgerFormatError(
            f"{label} must be a normalized repository-relative path"
        )
    if parsed.as_posix() != value:
        raise LedgerFormatError(f"{label} must use normalized POSIX separators")
    return value


def _https_url(value: Any, *, label: str) -> str:
    value = _safe_text(value, label=label, maximum=2048)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise LedgerFormatError(f"{label} is not a valid URL") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise LedgerFormatError(
            f"{label} must be a credential-free HTTPS URL without query or fragment"
        )
    hostname = parsed.hostname.lower()
    if any(ord(character) > 0x7F for character in hostname):
        raise LedgerFormatError(f"{label} hostname must be ASCII")
    netloc = hostname if port is None else f"{hostname}:443"
    path = parsed.path or "/"
    if any(ord(character) < 0x20 for character in path):
        raise LedgerFormatError(f"{label} path contains control characters")
    normalized = urlunsplit(("https", netloc, path, "", ""))
    if normalized != value:
        raise LedgerFormatError(f"{label} must be normalized")
    return value


def _canonical_uuid(value: Any, *, label: str) -> str:
    value = _safe_text(value, label=label, maximum=36)
    try:
        canonical = str(uuid.UUID(value))
    except ValueError as exc:
        raise LedgerFormatError(f"{label} must be a canonical UUID") from exc
    if canonical != value:
        raise LedgerFormatError(f"{label} must be a canonical UUID")
    return value


def _feedback_identifier(value: Any, *, label: str) -> str:
    """Validate the production feedback identifier contract."""
    if not isinstance(value, str):
        raise LedgerFormatError(f"{label} must be a non-empty string")
    resolved = value.strip()
    if not resolved or len(resolved) > 200:
        raise LedgerFormatError(f"{label} must contain 1-200 characters after trimming")
    if "\r" in resolved or "\n" in resolved:
        raise LedgerFormatError(f"{label} must not contain newlines")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in resolved):
        raise LedgerFormatError(f"{label} must not contain Unicode surrogates")
    return resolved


def _repo_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise LedgerFormatError("repository root is not a directory")
    return root


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LedgerFormatError("path escapes the repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise LedgerFormatError("repository paths must not contain symbolic links")


def _existing_repo_file(
    repo_root: Path, relative: str, *, label: str
) -> tuple[Path, bytes]:
    relative = _safe_repo_path(relative, label=label)
    path = repo_root.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_components(repo_root, path)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise EvidenceValidationError(f"{label} does not exist") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise EvidenceValidationError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > MAX_EVIDENCE_BYTES:
        raise EvidenceValidationError(f"{label} exceeds the safe evidence size")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceValidationError(f"{label} could not be read") from exc
    return path, raw


def _evidence_document(
    reference: Any, *, repo_root: Path, label: str
) -> tuple[str, str, bytes, Mapping[str, Any]]:
    relative, expected_hash, raw = _evidence_bytes(
        reference, repo_root=repo_root, label=label
    )
    document = _require_object(
        _parse_json_bytes(raw, label=f"{label} JSON", limit=MAX_EVIDENCE_BYTES),
        label=f"{label} JSON",
    )
    return relative, expected_hash, raw, document


def _evidence_bytes(
    reference: Any, *, repo_root: Path, label: str
) -> tuple[str, str, bytes]:
    reference = _require_object(reference, label=label)
    _require_exact_keys(reference, frozenset({"path", "sha256"}), label=label)
    relative = _safe_repo_path(reference["path"], label=f"{label}.path")
    expected_hash = _sha256(reference["sha256"], label=f"{label}.sha256")
    _path, raw = _existing_repo_file(repo_root, relative, label=f"{label}.path")
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise EvidenceValidationError(f"{label} raw SHA-256 does not match")
    return relative, expected_hash, raw


def _git_environment() -> dict[str, str]:
    """Use local Git without inherited redirection or replacement objects."""
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


def _git(repo_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceValidationError("Git evidence could not be inspected") from exc
    if result.returncode != 0:
        raise EvidenceValidationError(
            "Git evidence does not exist at the declared commit"
        )
    return result.stdout


def _git_blob(repo_root: Path, commit: str, relative: str, *, label: str) -> bytes:
    relative = _safe_repo_path(relative, label=label)
    listing = _git(repo_root, ("ls-tree", "-z", commit, "--", relative))
    entries = [entry for entry in listing.split(b"\0") if entry]
    if len(entries) != 1:
        raise EvidenceValidationError(
            f"{label} is not one Git blob at the source commit"
        )
    try:
        metadata, observed_path = entries[0].split(b"\t", 1)
        mode, object_type, _object_id = metadata.split(b" ", 2)
        decoded_path = observed_path.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvidenceValidationError(f"{label} has invalid Git tree metadata") from exc
    if (
        decoded_path != relative
        or object_type != b"blob"
        or mode not in {b"100644", b"100755"}
    ):
        raise EvidenceValidationError(f"{label} is not a regular Git file")
    return _git(repo_root, ("show", f"{commit}:{relative}"))


def _rollout_fingerprint_from_commit(
    repo_root: Path, commit: str, paths: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"ArchitectureIQ feedback rollout inputs v1\0")
    for relative in paths:
        content = _git_blob(
            repo_root, commit, relative, label="preflight checked rollout input"
        )
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise EvidenceValidationError(
                "preflight checked rollout input is not valid UTF-8"
            ) from exc
        path_bytes = relative.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_manifest(document: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    _require_exact_keys(document, MANIFEST_KEYS, label="manifest")
    if document["schema_version"] != SCHEMA_VERSION:
        raise EvidenceValidationError("manifest schema version is unsupported")
    release_id = document["release_id"]
    if not isinstance(release_id, str) or RELEASE_PATTERN.fullmatch(release_id) is None:
        raise EvidenceValidationError("manifest release_id is invalid")
    core = {
        key: document[key]
        for key in ("schema_version", "source_runs", "questions", "counts", "artifacts")
    }
    expected_release = (
        f"release_{hashlib.sha256(_canonical_json_bytes(core)).hexdigest()}"
    )
    if release_id != expected_release:
        raise EvidenceValidationError(
            "manifest release_id does not match its canonical core"
        )
    counts = _require_object(document["counts"], label="manifest.counts")
    questions = document["questions"]
    artifacts = document["artifacts"]
    if not isinstance(questions, list) or not isinstance(artifacts, list):
        raise EvidenceValidationError("manifest questions and artifacts must be arrays")
    question_count = _positive_int(
        counts.get("questions"), label="manifest question count"
    )
    if question_count != len(questions):
        raise EvidenceValidationError("manifest question count does not match its rows")
    artifact_count = _positive_int(
        counts.get("artifact_files"), label="manifest artifact count", allow_zero=True
    )
    if artifact_count != len(artifacts):
        raise EvidenceValidationError("manifest artifact count does not match its rows")
    return {
        "release_id": release_id,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "question_count": question_count,
    }


def _validate_registry(document: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(document, REGISTRY_KEYS, label="registry")
    if document["schema_version"] != SCHEMA_VERSION:
        raise EvidenceValidationError("registry schema version is unsupported")
    release_id = document["release_id"]
    registry_id = document["registry_id"]
    manifest_sha256 = document["manifest_sha256"]
    if not isinstance(release_id, str) or RELEASE_PATTERN.fullmatch(release_id) is None:
        raise EvidenceValidationError("registry release_id is invalid")
    if (
        not isinstance(registry_id, str)
        or REGISTRY_PATTERN.fullmatch(registry_id) is None
    ):
        raise EvidenceValidationError("registry registry_id is invalid")
    _sha256(manifest_sha256, label="registry.manifest_sha256")
    question_count = _positive_int(
        document["question_count"], label="registry question_count"
    )
    choice_count = _positive_int(
        document["choice_count"], label="registry choice_count"
    )
    questions = document["questions"]
    if not isinstance(questions, list) or len(questions) != question_count:
        raise EvidenceValidationError("registry question count does not match its rows")
    observed_choices = 0
    for question in questions:
        if not isinstance(question, dict) or not isinstance(
            question.get("choices"), dict
        ):
            raise EvidenceValidationError("registry question choices are invalid")
        observed_choices += len(question["choices"])
    if observed_choices != choice_count:
        raise EvidenceValidationError("registry choice count does not match its rows")
    core = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "question_count": question_count,
        "choice_count": choice_count,
        "questions": questions,
    }
    expected_registry = (
        f"registry_{hashlib.sha256(_canonical_json_bytes(core)).hexdigest()}"
    )
    if registry_id != expected_registry:
        raise EvidenceValidationError("registry_id does not match its canonical core")
    return {
        "release_id": release_id,
        "registry_id": registry_id,
        "manifest_sha256": manifest_sha256,
        "question_count": question_count,
        "choice_count": choice_count,
    }


def _validate_preflight(
    document: Mapping[str, Any], *, source_commit: str, fingerprint: str
) -> tuple[str, ...]:
    _require_exact_keys(document, PREFLIGHT_KEYS, label="preflight")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "scope": "local_static",
        "rollout_contract": "staged_upgrade",
        "phase": "report-app",
        "static_overall": "PASS",
        "overall": "UNVERIFIED",
        "hosted_verified": False,
        "deploy_ready": False,
        "git_sha": source_commit,
        "checked_rollout_input_sha256": fingerprint,
    }
    if any(document.get(key) != value for key, value in expected_scalars.items()):
        raise EvidenceValidationError(
            "preflight source, phase, or static status does not match"
        )
    if document.get("baseline_migrations_are_compatibility_inputs") is not True:
        raise EvidenceValidationError(
            "preflight does not include baseline compatibility inputs"
        )
    if not isinstance(document.get("require_hosted"), bool):
        raise EvidenceValidationError("preflight require_hosted is invalid")
    paths = document.get("checked_rollout_input_paths")
    if not isinstance(paths, list) or not paths:
        raise EvidenceValidationError("preflight checked input paths are invalid")
    normalized_paths: list[str] = []
    for path in paths:
        try:
            normalized_paths.append(
                _safe_repo_path(path, label="preflight checked rollout input")
            )
        except LedgerFormatError as exc:
            raise EvidenceValidationError(
                "preflight checked input paths are invalid"
            ) from exc
    if len(set(normalized_paths)) != len(normalized_paths):
        raise EvidenceValidationError("preflight checked input paths are duplicated")
    checks = document.get("checks")
    if not isinstance(checks, list) or not checks:
        raise EvidenceValidationError("preflight checks are missing")
    statuses: dict[str, str] = {}
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"code", "status", "summary"}:
            raise EvidenceValidationError("preflight check schema is invalid")
        code = check.get("code")
        status_value = check.get("status")
        if (
            not isinstance(code, str)
            or code in statuses
            or not isinstance(status_value, str)
        ):
            raise EvidenceValidationError("preflight check identity is invalid")
        statuses[code] = status_value
    required_pass = {"git.inputs_tracked", "git.inputs_clean", "git.inputs_match_head"}
    if any(statuses.get(code) != "PASS" for code in required_pass):
        raise EvidenceValidationError(
            "preflight Git tracked/clean/match-head checks must PASS"
        )
    if statuses.get("hosted.acceptance") != "UNVERIFIED":
        raise EvidenceValidationError(
            "preflight hosted acceptance must remain UNVERIFIED"
        )
    if any(
        status_value != "PASS"
        for code, status_value in statuses.items()
        if code != "hosted.acceptance"
    ):
        raise EvidenceValidationError(
            "preflight contains a non-hosted check that did not PASS"
        )
    return tuple(normalized_paths)


def _validate_candidate(facts: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    _require_exact_keys(
        facts,
        frozenset({"release_id", "manifest", "registry", "source", "rollout"}),
        label="candidate facts",
    )
    release_id = facts["release_id"]
    if not isinstance(release_id, str) or RELEASE_PATTERN.fullmatch(release_id) is None:
        raise LedgerFormatError("candidate release_id is invalid")
    manifest_ref = _require_object(facts["manifest"], label="candidate.manifest")
    _require_exact_keys(
        manifest_ref,
        frozenset({"path", "sha256", "question_count"}),
        label="candidate.manifest",
    )
    registry_ref = _require_object(facts["registry"], label="candidate.registry")
    _require_exact_keys(
        registry_ref,
        frozenset(
            {
                "path",
                "sha256",
                "registry_id",
                "manifest_sha256",
                "question_count",
                "choice_count",
            }
        ),
        label="candidate.registry",
    )
    source = _require_object(facts["source"], label="candidate.source")
    _require_exact_keys(
        source,
        frozenset({"repo_url", "branch", "commit", "entrypoint"}),
        label="candidate.source",
    )
    repo_url = _https_url(source["repo_url"], label="candidate.source.repo_url")
    branch = source["branch"]
    if not isinstance(branch, str) or SAFE_BRANCH_PATTERN.fullmatch(branch) is None:
        raise LedgerFormatError("candidate source branch is invalid")
    commit = source["commit"]
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise LedgerFormatError("candidate source commit must be a full Git object ID")
    entrypoint = _safe_repo_path(
        source["entrypoint"], label="candidate.source.entrypoint"
    )
    rollout = _require_object(facts["rollout"], label="candidate.rollout")
    _require_exact_keys(
        rollout,
        frozenset({"phase", "fingerprint", "preflight"}),
        label="candidate.rollout",
    )
    if rollout["phase"] != "report-app":
        raise LedgerFormatError("candidate rollout phase must be report-app")
    fingerprint = _sha256(rollout["fingerprint"], label="candidate.rollout.fingerprint")

    manifest_reference = {
        "path": manifest_ref["path"],
        "sha256": manifest_ref["sha256"],
    }
    manifest_path, manifest_hash, manifest_raw, manifest_document = _evidence_document(
        manifest_reference, repo_root=repo_root, label="candidate.manifest"
    )
    manifest = _validate_manifest(manifest_document, manifest_raw)
    registry_reference = {
        "path": registry_ref["path"],
        "sha256": registry_ref["sha256"],
    }
    registry_path, registry_hash, registry_raw, registry_document = _evidence_document(
        registry_reference, repo_root=repo_root, label="candidate.registry"
    )
    registry = _validate_registry(registry_document)
    _preflight_path, _preflight_hash, _preflight_raw, preflight = _evidence_document(
        rollout["preflight"], repo_root=repo_root, label="candidate.rollout.preflight"
    )
    preflight_paths = _validate_preflight(
        preflight, source_commit=commit, fingerprint=fingerprint
    )

    manifest_fact_count = _positive_int(
        manifest_ref["question_count"], label="candidate.manifest.question_count"
    )
    registry_fact_count = _positive_int(
        registry_ref["question_count"], label="candidate.registry.question_count"
    )
    registry_fact_choices = _positive_int(
        registry_ref["choice_count"], label="candidate.registry.choice_count"
    )
    fact_registry_id = registry_ref["registry_id"]
    if (
        not isinstance(fact_registry_id, str)
        or REGISTRY_PATTERN.fullmatch(fact_registry_id) is None
    ):
        raise LedgerFormatError("candidate.registry.registry_id is invalid")
    _sha256(
        registry_ref["manifest_sha256"],
        label="candidate.registry.manifest_sha256",
    )
    if not (
        release_id == manifest["release_id"] == registry["release_id"]
        and manifest_hash
        == registry["manifest_sha256"]
        == registry_ref["manifest_sha256"]
        and manifest_hash == manifest_ref["sha256"]
        and registry_hash == registry_ref["sha256"]
        and registry["registry_id"] == fact_registry_id
        and manifest["question_count"]
        == manifest_fact_count
        == registry["question_count"]
        == registry_fact_count
        and registry["choice_count"] == registry_fact_choices
    ):
        raise EvidenceValidationError(
            "candidate release, manifest, registry, or counts disagree"
        )

    _git(repo_root, ("cat-file", "-e", f"{commit}^{{commit}}"))
    if (
        _rollout_fingerprint_from_commit(repo_root, commit, preflight_paths)
        != fingerprint
    ):
        raise EvidenceValidationError(
            "preflight fingerprint does not match source commit blobs"
        )
    if (
        _git_blob(repo_root, commit, manifest_path, label="candidate.manifest.path")
        != manifest_raw
    ):
        raise EvidenceValidationError("manifest bytes differ from the source commit")
    if (
        _git_blob(repo_root, commit, registry_path, label="candidate.registry.path")
        != registry_raw
    ):
        raise EvidenceValidationError("registry bytes differ from the source commit")
    _git_blob(repo_root, commit, entrypoint, label="candidate.source.entrypoint")
    return {
        "release_id": release_id,
        "manifest_sha256": manifest_hash,
        "question_count": registry["question_count"],
        "choice_count": registry["choice_count"],
        "registry_id": registry["registry_id"],
        "repo_url": repo_url,
        "branch": branch,
        "source_commit": commit,
        "entrypoint": entrypoint,
        "rollout_fingerprint": fingerprint,
    }


def _validate_declaration(facts: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_keys(
        facts,
        frozenset(
            {
                "environment",
                "target_label",
                "provider",
                "project_id",
                "deploy_id",
                "site_url",
                "backend_project_id",
                "ingest_origin_sha256",
                "report_origin_sha256",
            }
        ),
        label="deployment declaration facts",
    )
    return {
        "environment": _safe_text(
            facts["environment"], label="environment", maximum=64
        ),
        "target_label": _safe_text(
            facts["target_label"], label="target_label", maximum=128
        ),
        "provider": _safe_text(facts["provider"], label="provider", maximum=64),
        "project_id": _safe_text(facts["project_id"], label="project_id", maximum=255),
        "deploy_id": _safe_text(facts["deploy_id"], label="deploy_id", maximum=255),
        "site_url": _https_url(facts["site_url"], label="site_url"),
        "backend_project_id": _safe_text(
            facts["backend_project_id"], label="backend_project_id", maximum=255
        ),
        "ingest_origin_sha256": _sha256(
            facts["ingest_origin_sha256"], label="ingest_origin_sha256"
        ),
        "report_origin_sha256": _sha256(
            facts["report_origin_sha256"], label="report_origin_sha256"
        ),
    }


def _deployment_context(state: _DeploymentState) -> tuple[str, dict[str, str]]:
    declaration = state.declaration
    assert declaration is not None
    binding = {
        "deployment_key": state.deployment_key,
        "release_id": state.release_id,
        "manifest_sha256": str(state.candidate["manifest_sha256"]),
        "registry_id": str(state.candidate["registry_id"]),
        "source_commit": str(state.candidate["source_commit"]),
        "environment": str(declaration["environment"]),
        "target_label": str(declaration["target_label"]),
        "provider": str(declaration["provider"]),
        "project_id": str(declaration["project_id"]),
        "deploy_id": str(declaration["deploy_id"]),
        "site_url": str(declaration["site_url"]),
        "backend_project_id": str(declaration["backend_project_id"]),
        "ingest_origin_sha256": str(declaration["ingest_origin_sha256"]),
        "report_origin_sha256": str(declaration["report_origin_sha256"]),
    }
    identifier = f"deployment_context_{hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()}"
    return identifier, binding


def _context_summary(state: _DeploymentState) -> dict[str, str]:
    context_id, binding = _deployment_context(state)
    return {
        "deployment_context_id": context_id,
        "environment": binding["environment"],
        "target_label": binding["target_label"],
        "provider": binding["provider"],
        "project_id": binding["project_id"],
        "deploy_id": binding["deploy_id"],
        "site_url": binding["site_url"],
        "backend_project_id": binding["backend_project_id"],
        "ingest_origin_sha256": binding["ingest_origin_sha256"],
        "report_origin_sha256": binding["report_origin_sha256"],
    }


def _validate_evidence_facts(
    facts: Mapping[str, Any], *, label: str
) -> tuple[Any, Any]:
    _require_exact_keys(facts, frozenset({"evidence", "summary"}), label=label)
    summary = _require_object(facts["summary"], label=f"{label}.summary")
    return facts["evidence"], summary


def _validate_postgres(
    facts: Mapping[str, Any], *, repo_root: Path, state: _DeploymentState
) -> Mapping[str, Any]:
    evidence_ref, summary = _validate_evidence_facts(facts, label="postgres facts")
    _path, _hash, _raw, document = _evidence_document(
        evidence_ref, repo_root=repo_root, label="postgres evidence"
    )
    _require_exact_keys(document, POSTGRES_KEYS, label="postgres evidence")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("evidence_type") != POSTGRES_EVIDENCE_TYPE
        or document.get("accepted") is not True
        or document.get("database_contacted") is not True
        or document.get("transaction_rolled_back") is not True
    ):
        raise EvidenceValidationError(
            "PostgreSQL acceptance status is not a complete PASS"
        )
    checks = document.get("checks")
    if not isinstance(checks, list) or not checks:
        raise EvidenceValidationError("PostgreSQL acceptance checks are missing")
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"code", "status", "summary"}:
            raise EvidenceValidationError(
                "PostgreSQL acceptance check schema is invalid"
            )
        if check.get("status") != "PASS":
            raise EvidenceValidationError(
                "PostgreSQL acceptance contains a non-PASS check"
            )
    evidence_summary = document.get("summary")
    if evidence_summary != {"pass": len(checks), "fail": 0}:
        raise EvidenceValidationError("PostgreSQL acceptance summary is inconsistent")
    server = _require_object(document.get("server"), label="postgres evidence.server")
    _require_exact_keys(
        server,
        frozenset(
            {"observed_at", "database", "role", "server_version_num", "in_recovery"}
        ),
        label="postgres evidence.server",
    )
    observed_at = _utc_timestamp(
        server["observed_at"], label="postgres server observed_at"
    )
    database = _safe_text(server["database"], label="postgres database")
    _safe_text(server["role"], label="postgres role")
    version = _positive_int(
        server["server_version_num"], label="postgres server version"
    )
    if version < 140000 or server["in_recovery"] is not False:
        raise EvidenceValidationError("PostgreSQL server is unsupported or in recovery")
    registry = _require_object(
        document.get("registry"), label="postgres evidence.registry"
    )
    _require_exact_keys(
        registry,
        frozenset(
            {
                "release_id",
                "registry_id",
                "question_count",
                "choice_count",
                "registered_release_count",
                "registered_question_count",
                "registered_choice_count",
                "authority_revision",
                "detail_revision",
            }
        ),
        label="postgres evidence.registry",
    )
    candidate = state.candidate
    if not (
        registry.get("release_id") == state.release_id
        and registry.get("registry_id") == candidate["registry_id"]
        and registry.get("question_count") == candidate["question_count"]
        and registry.get("choice_count") == candidate["choice_count"]
        and registry.get("authority_revision") == "registry_v1"
        and registry.get("detail_revision") == "detail_v1"
        and _positive_int(
            registry.get("registered_release_count"), label="registered release count"
        )
        >= 1
        and _positive_int(
            registry.get("registered_question_count"), label="registered question count"
        )
        >= candidate["question_count"]
        and _positive_int(
            registry.get("registered_choice_count"), label="registered choice count"
        )
        >= candidate["choice_count"]
    ):
        raise EvidenceValidationError(
            "PostgreSQL registry does not match the candidate"
        )
    declaration = state.declaration
    assert declaration is not None
    if document.get("target_label") != declaration["target_label"]:
        raise EvidenceValidationError(
            "PostgreSQL target label does not match the deployment"
        )
    expected_summary = {
        **_context_summary(state),
        "observed_at": observed_at,
        "database": database,
        "server_version_num": version,
        "release_id": state.release_id,
        "registry_id": candidate["registry_id"],
        "question_count": candidate["question_count"],
        "choice_count": candidate["choice_count"],
        "authority_revision": "registry_v1",
        "detail_revision": "detail_v1",
    }
    _require_exact_keys(
        summary, frozenset(expected_summary), label="postgres facts.summary"
    )
    if summary != expected_summary:
        raise EvidenceValidationError(
            "PostgreSQL safe summary does not match its evidence"
        )
    return {
        "evidence_sha256": _hash,
        "observed_at": observed_at,
    }


def _validate_roundtrip(
    facts: Mapping[str, Any], *, repo_root: Path, state: _DeploymentState
) -> Mapping[str, Any]:
    evidence_ref, summary = _validate_evidence_facts(facts, label="roundtrip facts")
    _path, _hash, _raw, document = _evidence_document(
        evidence_ref, repo_root=repo_root, label="roundtrip evidence"
    )
    _require_exact_keys(document, ROUNDTRIP_KEYS, label="roundtrip evidence")
    candidate = state.candidate
    required_true = {
        "ok",
        "conflict_verified",
        "successful_batch_verified",
        "successful_batch_first_write_verified",
        "authority_status_verified",
        "detail_reports_verified",
        "business_snapshot_verified",
        "session_attempt_filters_verified",
    }
    if any(document.get(field_name) is not True for field_name in required_true):
        raise EvidenceValidationError(
            "hosted roundtrip did not verify every core behavior"
        )
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("evidence_type") != ROUNDTRIP_EVIDENCE_TYPE
        or document.get("authority_mode") != "authoritative"
        or document.get("release_id") != state.release_id
        or document.get("manifest_sha256") != candidate["manifest_sha256"]
        or document.get("registry_id") != candidate["registry_id"]
        or document.get("registry_question_count") != candidate["question_count"]
        or document.get("registry_choice_count") != candidate["choice_count"]
    ):
        raise EvidenceValidationError(
            "hosted roundtrip authority does not match the candidate"
        )
    verified_at = _utc_timestamp(document["verified_at"], label="roundtrip verified_at")
    run_id = _safe_text(document["run_id"], label="roundtrip run_id", maximum=48)
    _safe_text(document["question_id"], label="roundtrip question_id", maximum=255)
    event_id = _feedback_identifier(document["event_id"], label="roundtrip event_id")
    if event_id != document["event_id"]:
        raise EvidenceValidationError("roundtrip event_id is not normalized")
    request_fields = (
        "request_id",
        "conflict_request_id",
        "successful_batch_first_request_id",
        "successful_batch_replay_request_id",
    )
    request_ids = [
        _canonical_uuid(document[field_name], label=f"roundtrip {field_name}")
        for field_name in request_fields
    ]
    mixed_request = document.get("mixed_batch_request_id")
    if document.get("mixed_batch_verified") is True:
        request_ids.append(
            _canonical_uuid(mixed_request, label="roundtrip mixed request id")
        )
    elif document.get("mixed_batch_verified") is not False or mixed_request is not None:
        raise EvidenceValidationError("roundtrip mixed-batch fields are inconsistent")
    if len(set(request_ids)) != len(request_ids):
        raise EvidenceValidationError("roundtrip request UUIDs must be distinct")
    _positive_int(document.get("polls"), label="roundtrip polls")
    receipt = _require_object(document.get("receipt"), label="roundtrip receipt")
    _require_exact_keys(
        receipt,
        frozenset({"accepted", "duplicate", "conflict", "rejected", "request_id"}),
        label="roundtrip receipt",
    )
    counters = {
        name: _positive_int(
            receipt[name], label=f"roundtrip receipt {name}", allow_zero=True
        )
        for name in ("accepted", "duplicate", "conflict", "rejected")
    }
    if (
        receipt.get("request_id") != document.get("request_id")
        or counters["conflict"] != 0
        or counters["rejected"] != 0
        or counters["accepted"] + counters["duplicate"] != 1
    ):
        raise EvidenceValidationError(
            "roundtrip receipt is not a complete success receipt"
        )
    expected_summary = {
        **_context_summary(state),
        "verified_at": verified_at,
        "run_id": run_id,
        "release_id": state.release_id,
        "registry_id": candidate["registry_id"],
        "request_id": document["request_id"],
        "conflict_request_id": document["conflict_request_id"],
        "successful_batch_first_request_id": document[
            "successful_batch_first_request_id"
        ],
        "successful_batch_replay_request_id": document[
            "successful_batch_replay_request_id"
        ],
        "authority_mode": "authoritative",
    }
    _require_exact_keys(
        summary, frozenset(expected_summary), label="roundtrip facts.summary"
    )
    if summary != expected_summary:
        raise EvidenceValidationError(
            "roundtrip safe summary does not match its evidence"
        )
    return {
        "evidence_sha256": _hash,
        "verified_at": verified_at,
        "identity_tokens": tuple([run_id, event_id, *request_ids]),
    }


def _validate_mapping(
    facts: Mapping[str, Any], *, repo_root: Path, state: _DeploymentState
) -> Mapping[str, Any]:
    evidence_ref, summary = _validate_evidence_facts(
        facts, label="source mapping facts"
    )
    _path, _hash, _raw, document = _evidence_document(
        evidence_ref, repo_root=repo_root, label="source mapping evidence"
    )
    _require_exact_keys(document, MAPPING_KEYS, label="source mapping evidence")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("evidence_type") != MAPPING_EVIDENCE_TYPE
        or document.get("mapping_authority")
        != "reviewed_provider_control_plane_capture"
    ):
        raise EvidenceValidationError(
            "source mapping must be a reviewed provider-control-plane capture"
        )
    captured_at = _utc_timestamp(
        document["captured_at"], label="source mapping captured_at"
    )
    deployed_at = _utc_timestamp(
        document["deployed_at"], label="source mapping deployed_at"
    )
    if _utc_datetime(deployed_at, label="source mapping deployed_at") > _utc_datetime(
        captured_at, label="source mapping captured_at"
    ):
        raise EvidenceValidationError(
            "provider capture predates the reported deployment"
        )
    if document.get("deployment_status") != "ready":
        raise EvidenceValidationError(
            "provider capture does not report a ready deployment"
        )
    declaration = state.declaration
    assert declaration is not None
    candidate = state.candidate
    context_id, _binding = _deployment_context(state)
    expected = {
        "provider": declaration["provider"],
        "project_id": declaration["project_id"],
        "deploy_id": declaration["deploy_id"],
        "site_url": declaration["site_url"],
        "environment": declaration["environment"],
        "target_label": declaration["target_label"],
        "backend_project_id": declaration["backend_project_id"],
        "ingest_origin_sha256": declaration["ingest_origin_sha256"],
        "report_origin_sha256": declaration["report_origin_sha256"],
        "deployment_context_id": context_id,
        "repo_url": candidate["repo_url"],
        "branch": candidate["branch"],
        "source_commit": candidate["source_commit"],
        "entrypoint": candidate["entrypoint"],
        "release_id": state.release_id,
        "manifest_sha256": candidate["manifest_sha256"],
        "registry_id": candidate["registry_id"],
        "rollout_input_fingerprint": candidate["rollout_fingerprint"],
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise EvidenceValidationError(
            "provider source mapping does not match the deployment"
        )
    _https_url(document["site_url"], label="source mapping site_url")
    _https_url(document["repo_url"], label="source mapping repo_url")
    provider_export = _require_object(
        document["provider_export"], label="source mapping provider_export"
    )
    _require_exact_keys(
        provider_export,
        frozenset({"path", "sha256", "media_type"}),
        label="source mapping provider_export",
    )
    export_reference = {
        "path": provider_export["path"],
        "sha256": provider_export["sha256"],
    }
    export_path, export_sha256, export_raw = _evidence_bytes(
        export_reference,
        repo_root=repo_root,
        label="source mapping provider export",
    )
    if export_path == _path:
        raise EvidenceValidationError(
            "provider export must be separate from its mapping envelope"
        )
    media_type = provider_export["media_type"]
    if media_type == "application/json":
        _parse_json_bytes(
            export_raw,
            label="source mapping provider export JSON",
            limit=MAX_EVIDENCE_BYTES,
        )
    elif media_type == "text/plain":
        try:
            export_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise EvidenceValidationError(
                "provider text export is not valid UTF-8"
            ) from exc
    elif media_type == "image/png":
        if not export_raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise EvidenceValidationError(
                "provider PNG export has an invalid signature"
            )
    elif media_type == "application/pdf":
        if not export_raw.startswith(b"%PDF-"):
            raise EvidenceValidationError(
                "provider PDF export has an invalid signature"
            )
    else:
        raise EvidenceValidationError("provider export media type is unsupported")
    if not export_raw:
        raise EvidenceValidationError("provider export must not be empty")
    expected_summary = {
        **_context_summary(state),
        "captured_at": captured_at,
        "deployed_at": deployed_at,
        "deployment_status": "ready",
        "mapping_authority": "reviewed_provider_control_plane_capture",
        "provider_export_path": export_path,
        "provider_export_sha256": export_sha256,
        "provider_export_media_type": media_type,
        "source_commit": candidate["source_commit"],
        "release_id": state.release_id,
    }
    _require_exact_keys(
        summary, frozenset(expected_summary), label="source mapping facts.summary"
    )
    if summary != expected_summary:
        raise EvidenceValidationError(
            "source mapping safe summary does not match its evidence"
        )
    return {
        "evidence_sha256": _hash,
        "provider_export_sha256": export_sha256,
        "captured_at": captured_at,
        "deployed_at": deployed_at,
    }


def _record_hash(record: Mapping[str, Any]) -> str:
    core = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json_bytes(core)).hexdigest()


def _validate_record_shape(record: Mapping[str, Any], *, draft: bool) -> None:
    _require_exact_keys(
        record, DRAFT_KEYS if draft else RECORD_KEYS, label="ledger event"
    )
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("record_type") != RECORD_TYPE
    ):
        raise LedgerFormatError("ledger event schema or record type is unsupported")
    event_type = record.get("event_type")
    if event_type not in EVENT_TYPES:
        raise LedgerFormatError("ledger event_type is unsupported")
    deployment_key = record.get("deployment_key")
    if (
        not isinstance(deployment_key, str)
        or DEPLOYMENT_KEY_PATTERN.fullmatch(deployment_key) is None
    ):
        raise LedgerFormatError("deployment_key is invalid")
    _utc_timestamp(record.get("recorded_at"), label="recorded_at")
    _safe_text(record.get("recorded_by"), label="recorded_by", maximum=255)
    reviewed_by = _nullable_safe_text(
        record.get("reviewed_by"), label="reviewed_by", maximum=255
    )
    reviewed_events = {"source_mapping_attested", "activated", *TERMINAL_EVENTS}
    if event_type in reviewed_events and reviewed_by is None:
        raise StateTransitionError(
            "source mapping, activation, and terminal events require reviewed_by"
        )
    if event_type in reviewed_events and reviewed_by == record.get("recorded_by"):
        raise StateTransitionError("reviewed decision requires a distinct reviewer")
    _require_object(record.get("facts"), label="ledger event facts")
    if not draft:
        previous = record.get("previous_record_sha256")
        if previous is not None:
            _sha256(previous, label="previous_record_sha256")
        _sha256(record.get("record_sha256"), label="record_sha256")


def _replay(
    records: Sequence[Mapping[str, Any]], *, repo_root: Path
) -> tuple[DeploymentSummary, ...]:
    states: dict[str, _DeploymentState] = {}
    deployment_identities: dict[tuple[str, str, str], str] = {}
    evidence_hash_owners: dict[str, str] = {}
    roundtrip_identity_owners: dict[str, str] = {}
    provider_export_owners: dict[str, str] = {}
    for record in records:
        event_type = str(record["event_type"])
        deployment_key = str(record["deployment_key"])
        facts = _require_object(record["facts"], label="ledger event facts")
        state = states.get(deployment_key)
        if event_type == "candidate_attested":
            if state is not None:
                raise StateTransitionError(
                    "deployment_key already has a candidate event"
                )
            candidate = _validate_candidate(facts, repo_root=repo_root)
            states[deployment_key] = _DeploymentState(
                deployment_key=deployment_key,
                release_id=str(candidate["release_id"]),
                candidate=candidate,
            )
            continue
        if state is None:
            raise StateTransitionError(
                "deployment event appears before candidate_attested"
            )
        if state.terminal_event is not None:
            raise StateTransitionError(
                "no event may follow a terminal deployment event"
            )
        if event_type == "deployment_declared":
            if state.declaration is not None or state.activated or state.evidence:
                raise StateTransitionError(
                    "deployment_declared is out of order or repeated"
                )
            declaration = _validate_declaration(facts)
            identity = (
                declaration["provider"],
                declaration["project_id"],
                declaration["deploy_id"],
            )
            owner = deployment_identities.get(identity)
            if owner is not None and owner != deployment_key:
                raise StateTransitionError(
                    "provider deploy identity is reused by another deployment"
                )
            deployment_identities[identity] = deployment_key
            state.declaration = declaration
            continue
        if state.declaration is None:
            raise StateTransitionError(
                "deployment evidence appears before deployment_declared"
            )
        if event_type in EVIDENCE_EVENTS:
            if state.activated:
                raise StateTransitionError(
                    "deployment evidence cannot be added after activation"
                )
            if event_type in state.evidence:
                raise StateTransitionError(
                    "deployment evidence type may be attached only once"
                )
            if event_type == "postgres_accepted":
                metadata = _validate_postgres(facts, repo_root=repo_root, state=state)
            elif event_type == "roundtrip_accepted":
                metadata = _validate_roundtrip(facts, repo_root=repo_root, state=state)
            else:
                metadata = _validate_mapping(facts, repo_root=repo_root, state=state)
            evidence_sha256 = str(metadata["evidence_sha256"])
            evidence_owner = evidence_hash_owners.get(evidence_sha256)
            if evidence_owner is not None and evidence_owner != deployment_key:
                raise StateTransitionError(
                    "hosted evidence SHA-256 is reused by another deployment"
                )
            evidence_hash_owners[evidence_sha256] = deployment_key
            if event_type == "roundtrip_accepted":
                for token in metadata["identity_tokens"]:
                    identity_owner = roundtrip_identity_owners.get(str(token))
                    if identity_owner is not None and identity_owner != deployment_key:
                        raise StateTransitionError(
                            "roundtrip run, event, or request identity is reused by another deployment"
                        )
                    roundtrip_identity_owners[str(token)] = deployment_key
            if event_type == "source_mapping_attested":
                export_sha256 = str(metadata["provider_export_sha256"])
                export_owner = provider_export_owners.get(export_sha256)
                if export_owner is not None and export_owner != deployment_key:
                    raise StateTransitionError(
                        "provider export SHA-256 is reused by another deployment"
                    )
                provider_export_owners[export_sha256] = deployment_key
            state.evidence.add(event_type)
            state.evidence_metadata[event_type] = metadata
            continue
        if event_type == "activated":
            _require_exact_keys(facts, frozenset(), label="activation facts")
            if state.activated:
                raise StateTransitionError("deployment is already activated")
            if state.evidence != EVIDENCE_EVENTS:
                raise StateTransitionError(
                    "activation requires PostgreSQL, roundtrip, and source-mapping evidence"
                )
            mapping_metadata = state.evidence_metadata["source_mapping_attested"]
            roundtrip_metadata = state.evidence_metadata["roundtrip_accepted"]
            if _utc_datetime(
                roundtrip_metadata["verified_at"], label="roundtrip verified_at"
            ) < _utc_datetime(
                mapping_metadata["deployed_at"], label="provider deployed_at"
            ):
                raise StateTransitionError(
                    "hosted roundtrip evidence predates the provider-reported deployment"
                )
            state.activated = True
            continue
        if event_type in TERMINAL_EVENTS:
            if not state.activated:
                raise StateTransitionError(
                    "only an active deployment can become terminal"
                )
            _require_exact_keys(
                facts,
                frozenset({"reason", "replacement_deployment_key"}),
                label="terminal facts",
            )
            _safe_text(facts["reason"], label="terminal reason", maximum=1000)
            replacement = facts["replacement_deployment_key"]
            if replacement is not None:
                if (
                    not isinstance(replacement, str)
                    or DEPLOYMENT_KEY_PATTERN.fullmatch(replacement) is None
                    or replacement == deployment_key
                ):
                    raise StateTransitionError(
                        "terminal replacement deployment_key is invalid"
                    )
                replacement_state = states.get(replacement)
                if replacement_state is None:
                    raise StateTransitionError(
                        "terminal replacement deployment does not exist"
                    )
                if (
                    not replacement_state.activated
                    or replacement_state.terminal_event is not None
                ):
                    raise StateTransitionError(
                        "replacement deployment must be activated and non-terminal"
                    )
            elif event_type == "superseded":
                raise StateTransitionError(
                    "superseded requires a replacement deployment_key"
                )
            state.terminal_event = event_type
            continue
        raise StateTransitionError("unsupported deployment transition")

    summaries: list[DeploymentSummary] = []
    for key in sorted(states):
        state = states[key]
        declaration = state.declaration
        summaries.append(
            DeploymentSummary(
                deployment_key=key,
                release_id=state.release_id,
                status=state.status,
                evidence=tuple(sorted(state.evidence)),
                provider=declaration["provider"] if declaration else None,
                project_id=declaration["project_id"] if declaration else None,
                deploy_id=declaration["deploy_id"] if declaration else None,
                site_url=declaration["site_url"] if declaration else None,
            )
        )
    return tuple(summaries)


def _verify_ledger_bytes(raw: bytes, *, repo_root: Path) -> LedgerSnapshot:
    if len(raw) > MAX_LEDGER_BYTES:
        raise LedgerFormatError("ledger exceeds the safe byte limit")
    if not raw:
        return LedgerSnapshot(records=(), deployments=(), head_record_sha256=None)
    if not raw.endswith(b"\n"):
        raise LedgerFormatError("non-empty ledger must end with one LF newline")
    lines = raw.splitlines(keepends=True)
    if len(lines) > MAX_RECORDS:
        raise LedgerFormatError("ledger exceeds the safe record limit")
    records: list[Mapping[str, Any]] = []
    previous_hash: str | None = None
    for index, line in enumerate(lines, start=1):
        if line == b"\n" or not line:
            raise LedgerFormatError("ledger must not contain empty lines")
        if len(line) > MAX_LINE_BYTES:
            raise LedgerFormatError("ledger record exceeds the safe line limit")
        if not line.endswith(b"\n") or b"\r" in line:
            raise LedgerFormatError("ledger lines must use canonical LF endings")
        document = _require_object(
            _parse_json_bytes(
                line[:-1], label=f"ledger line {index}", limit=MAX_LINE_BYTES
            ),
            label=f"ledger line {index}",
        )
        _validate_record_shape(document, draft=False)
        if _canonical_json_bytes(document) + b"\n" != line:
            raise LedgerFormatError("ledger line is not exact canonical JSON")
        if document["previous_record_sha256"] != previous_hash:
            raise LedgerFormatError("ledger hash chain is broken")
        expected_hash = _record_hash(document)
        if document["record_sha256"] != expected_hash:
            raise LedgerFormatError("ledger record SHA-256 is invalid")
        previous_hash = expected_hash
        records.append(document)
    deployments = _replay(records, repo_root=repo_root)
    return LedgerSnapshot(
        records=tuple(records),
        deployments=deployments,
        head_record_sha256=previous_hash,
    )


def _safe_ledger_path(repo_root: Path, ledger_path: Path) -> Path:
    candidate = ledger_path.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    candidate = Path(os.path.abspath(candidate))
    _reject_symlink_components(repo_root, candidate)
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise LedgerFormatError(
            "ledger path must remain inside the repository"
        ) from exc
    return candidate


def _read_ledger(path: Path) -> bytes:
    if not path.exists():
        return b""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LedgerFormatError("ledger metadata could not be read") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise LedgerFormatError("ledger must be a regular non-symlink file")
    if metadata.st_size > MAX_LEDGER_BYTES:
        raise LedgerFormatError("ledger exceeds the safe byte limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LedgerFormatError("ledger could not be read") from exc


def verify_ledger(*, repo_root: Path, ledger_path: Path) -> LedgerSnapshot:
    """Strictly verify the complete hash chain, state machine, and evidence."""
    root = _repo_root(repo_root)
    path = _safe_ledger_path(root, ledger_path)
    return _verify_ledger_bytes(_read_ledger(path), repo_root=root)


def _read_draft(path: Path) -> Mapping[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LedgerFormatError("event draft does not exist") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise LedgerFormatError("event draft must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LedgerFormatError("event draft could not be read") from exc
    document = _require_object(
        _parse_json_bytes(raw, label="event draft", limit=MAX_DRAFT_BYTES),
        label="event draft",
    )
    _validate_record_shape(document, draft=True)
    return document


def _atomic_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def append_event(
    *,
    repo_root: Path,
    ledger_path: Path,
    event_json_path: Path,
    confirm: bool,
) -> Mapping[str, Any]:
    """Append exactly one reviewed draft under an exclusive process lock."""
    if not confirm:
        raise ConfirmationRequired("append requires --confirm-append")
    root = _repo_root(repo_root)
    path = _safe_ledger_path(root, ledger_path)
    draft = _read_draft(event_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root, path.parent)
    lock_path = path.with_name(f".{path.name}.lock")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise LedgerFormatError("ledger lock could not be acquired") from exc
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise LedgerFormatError("ledger lock is not a regular file")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        existing = _read_ledger(path)
        snapshot = _verify_ledger_bytes(existing, repo_root=root)
        record = dict(draft)
        record["previous_record_sha256"] = snapshot.head_record_sha256
        record["record_sha256"] = _record_hash(record)
        line = _canonical_json_bytes(record) + b"\n"
        combined = existing + line
        _verify_ledger_bytes(combined, repo_root=root)
        _atomic_replace(path, combined)
        if _read_ledger(path) != combined:
            raise LedgerFormatError("ledger atomic write verification failed")
        return record
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _emit(value: Mapping[str, Any], stream: TextIO) -> None:
    print(_canonical_json_bytes(value).decode("utf-8"), file=stream)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify or append the deployment audit ledger."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "list"):
        command = subparsers.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument(
            "--ledger", type=Path, default=Path("deployments/ledger.jsonl")
        )
    append = subparsers.add_parser("append")
    append.add_argument("--repo", type=Path, default=Path.cwd())
    append.add_argument("--ledger", type=Path, default=Path("deployments/ledger.jsonl"))
    append.add_argument("--event-json", type=Path, required=True)
    append.add_argument("--confirm-append", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "append":
            record = append_event(
                repo_root=arguments.repo,
                ledger_path=arguments.ledger,
                event_json_path=arguments.event_json,
                confirm=arguments.confirm_append,
            )
            _emit(
                {
                    "ok": True,
                    "deployment_key": record["deployment_key"],
                    "event_type": record["event_type"],
                    "record_sha256": record["record_sha256"],
                },
                sys.stdout,
            )
            return 0
        snapshot = verify_ledger(repo_root=arguments.repo, ledger_path=arguments.ledger)
        payload = (
            snapshot.verification_dict()
            if arguments.command == "verify"
            else snapshot.listing_dict()
        )
        _emit(payload, sys.stdout)
        return 0
    except LedgerError as exc:
        _emit({"ok": False, "error": str(exc)}, sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
