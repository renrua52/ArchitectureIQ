"""Export an attested quiz bundle as a canonical feedback question registry.

The exporter is deliberately local and standard-library-only.  It never sends
the answer key over the network: maintainers review the deterministic JSON and
SQL artifacts before applying the latter as a data migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .publisher import MANIFEST_FILENAME, BundlePublishError, build_bundle_manifest
from .versioning import (
    MAX_SAFE_JSON_INTEGER,
    QuestionVersionError,
    compute_question_version,
    normalize_question,
)

try:  # ``tools`` on sys.path (the maintainer CLI and existing tests).
    from question_inspector.release_manifest import (
        QuizManifest,
        ReleaseManifestError,
        load_quiz_manifest,
    )
except ModuleNotFoundError as exc:  # Package import from the repository root.
    if exc.name != "question_inspector":
        raise
    from tools.question_inspector.release_manifest import (
        QuizManifest,
        ReleaseManifestError,
        load_quiz_manifest,
    )


REGISTRY_SCHEMA_VERSION = "1.0"
REGISTRY_ID_PREFIX = "registry_"
RELEASE_ID_PATTERN = re.compile(r"release_[0-9a-f]{64}\Z")
REGISTRY_ID_PATTERN = re.compile(r"registry_[0-9a-f]{64}\Z")
QUESTION_VERSION_PATTERN = re.compile(r"qv1_[0-9a-f]{64}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

_REGISTRY_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "manifest_sha256",
        "registry_id",
        "question_count",
        "choice_count",
        "questions",
    }
)
_QUESTION_KEYS = frozenset(
    {
        "question_id",
        "question_version",
        "family",
        "dataset_id",
        "question_type",
        "correct_letter",
        "correct_candidate_id",
        "choices",
    }
)


class FeedbackRegistryError(ValueError):
    """Raised when a bundle cannot produce a trustworthy registry export."""


def _contains_unicode_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeedbackRegistryError(f"{field} must be a non-empty string")
    if _contains_unicode_surrogate(value):
        raise FeedbackRegistryError(
            f"{field} cannot contain Unicode surrogate code points"
        )
    return value


def _require_identifier(value: Any, *, field: str) -> str:
    resolved = _require_string(value, field=field)
    if (
        resolved != resolved.strip()
        or len(resolved) > 200
        or "\r" in resolved
        or "\n" in resolved
    ):
        raise FeedbackRegistryError(
            f"{field} must be trimmed, at most 200 characters, and contain no newlines"
        )
    return resolved


def _require_count(value: Any, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SAFE_JSON_INTEGER
    ):
        raise FeedbackRegistryError(
            f"{field} must be a non-negative JavaScript-safe integer"
        )
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_strict_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise FeedbackRegistryError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FeedbackRegistryError(f"{label} must be a JSON object: {path}")
    try:
        # Reuse the publisher's cross-runtime safe-number/Unicode contract.
        normalize_question(value)
    except QuestionVersionError as exc:
        raise FeedbackRegistryError(f"invalid {label} at {path}: {exc}") from exc
    return value, raw


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FeedbackRegistryError(f"{label} is not canonical JSON: {exc}") from exc


def _registry_id(core: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json_bytes(core, label="feedback registry core")
    ).hexdigest()
    return f"{REGISTRY_ID_PREFIX}{digest}"


def _registry_identity_core(
    *,
    release_id: str,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return authority content only; manifest bytes are provenance metadata."""
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "release_id": release_id,
        "question_count": len(questions),
        "choice_count": sum(len(question["choices"]) for question in questions),
        "questions": questions,
    }


def _attest_and_rebuild_bundle(root: Path) -> QuizManifest:
    try:
        attested = load_quiz_manifest(root)
    except (ReleaseManifestError, ValueError) as exc:
        raise FeedbackRegistryError(
            f"runtime release attestation failed: {exc}"
        ) from exc
    if attested is None:
        raise FeedbackRegistryError(
            f"bundle has no required {MANIFEST_FILENAME}: {root}"
        )

    disk_manifest, raw_manifest = _read_strict_json_object(
        root / MANIFEST_FILENAME,
        label=MANIFEST_FILENAME,
    )
    observed_manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if observed_manifest_sha256 != attested.manifest_sha256:
        raise FeedbackRegistryError(
            f"{MANIFEST_FILENAME} changed during runtime attestation"
        )

    try:
        rebuilt = build_bundle_manifest(
            root,
            generated_at=attested.generated_at,
        )
    except (BundlePublishError, QuestionVersionError) as exc:
        raise FeedbackRegistryError(
            f"publisher ground-truth validation failed: {exc}"
        ) from exc
    if disk_manifest != rebuilt:
        raise FeedbackRegistryError(
            f"on-disk {MANIFEST_FILENAME} does not exactly match its rebuilt value"
        )
    if rebuilt["release_id"] != attested.release_id:
        raise FeedbackRegistryError(
            "runtime attestation and publisher rebuild disagree on release_id"
        )
    return attested


def _question_registry_entry(
    root: Path,
    manifest_question: Any,
) -> dict[str, Any]:
    question_path = root / PurePosixPath(manifest_question.path) / "question.json"
    question, _ = _read_strict_json_object(
        question_path,
        label=f"question {manifest_question.question_id!r}",
    )

    question_id = _require_identifier(question.get("question_id"), field="question_id")
    family = _require_identifier(question.get("family"), field=f"{question_id}.family")
    dataset_id = _require_identifier(
        question.get("dataset_id"), field=f"{question_id}.dataset_id"
    )
    question_type = _require_identifier(
        question.get("type"), field=f"{question_id}.type"
    )
    try:
        observed_version = compute_question_version(question)
    except QuestionVersionError as exc:
        raise FeedbackRegistryError(
            f"question {question_id!r} violates the canonical JSON contract: {exc}"
        ) from exc
    for field, observed, expected in (
        ("question_id", question_id, manifest_question.question_id),
        ("question_version", observed_version, manifest_question.version),
        ("family", family, manifest_question.family),
        ("dataset_id", dataset_id, manifest_question.dataset_id),
    ):
        if observed != expected:
            raise FeedbackRegistryError(
                f"question {question_id!r} {field} does not match the attested manifest"
            )

    raw_choices = question.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise FeedbackRegistryError(
            f"question {question_id!r} must have at least two choices"
        )
    choices: dict[str, str] = {}
    candidate_ids: set[str] = set()
    for index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, Mapping):
            raise FeedbackRegistryError(
                f"question {question_id!r} choices[{index}] must be a JSON object"
            )
        letter = _require_string(
            raw_choice.get("letter"),
            field=f"question {question_id!r} choices[{index}].letter",
        )
        if len(letter) != 1 or not "A" <= letter <= "Z":
            raise FeedbackRegistryError(
                f"question {question_id!r} has invalid choice letter {letter!r}"
            )
        candidate_id = _require_identifier(
            raw_choice.get("candidate_id"),
            field=f"question {question_id!r} choices[{index}].candidate_id",
        )
        if letter in choices:
            raise FeedbackRegistryError(
                f"question {question_id!r} has duplicate choice letter {letter!r}"
            )
        if candidate_id in candidate_ids:
            raise FeedbackRegistryError(
                f"question {question_id!r} has duplicate candidate_id {candidate_id!r}"
            )
        choices[letter] = candidate_id
        candidate_ids.add(candidate_id)

    correct_letter = _require_string(
        question.get("correct_letter"),
        field=f"question {question_id!r}.correct_letter",
    )
    correct_candidate_id = choices.get(correct_letter)
    if correct_candidate_id is None:
        raise FeedbackRegistryError(
            f"question {question_id!r} correct_letter {correct_letter!r} "
            "does not map to a choice"
        )

    return {
        "question_id": question_id,
        "question_version": observed_version,
        "family": family,
        "dataset_id": dataset_id,
        "question_type": question_type,
        "correct_letter": correct_letter,
        "correct_candidate_id": correct_candidate_id,
        "choices": {letter: choices[letter] for letter in sorted(choices)},
    }


def _registry_core(
    *,
    release_id: str,
    manifest_sha256: str,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
        "question_count": len(questions),
        "choice_count": sum(len(question["choices"]) for question in questions),
        "questions": questions,
    }


def build_feedback_registry(bundle_root: Path | str) -> dict[str, Any]:
    """Return a deterministic registry after both release validation paths pass."""
    root = Path(bundle_root).expanduser().resolve()
    attested = _attest_and_rebuild_bundle(root)
    questions = [
        _question_registry_entry(root, question) for question in attested.questions
    ]
    questions.sort(key=lambda item: (item["question_id"], item["question_version"]))
    if len(questions) != attested.question_count:
        raise FeedbackRegistryError(
            "registry question count does not match the attested manifest"
        )
    core = _registry_core(
        release_id=attested.release_id,
        manifest_sha256=attested.manifest_sha256,
        questions=questions,
    )
    registry = {
        "schema_version": core["schema_version"],
        "release_id": core["release_id"],
        "manifest_sha256": core["manifest_sha256"],
        "registry_id": _registry_id(
            _registry_identity_core(
                release_id=str(core["release_id"]),
                questions=questions,
            )
        ),
        "question_count": core["question_count"],
        "choice_count": core["choice_count"],
        "questions": core["questions"],
    }
    _validate_registry(registry)
    return registry


def _validate_registry(registry: Mapping[str, Any]) -> None:
    if set(registry) != _REGISTRY_KEYS:
        raise FeedbackRegistryError("feedback registry does not use the exact schema")
    if registry["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise FeedbackRegistryError("unsupported feedback registry schema_version")
    release_id = _require_string(registry["release_id"], field="release_id")
    if RELEASE_ID_PATTERN.fullmatch(release_id) is None:
        raise FeedbackRegistryError(
            "release_id must be 'release_' plus 64 lowercase hex digits"
        )
    manifest_sha256 = _require_string(
        registry["manifest_sha256"], field="manifest_sha256"
    )
    if SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise FeedbackRegistryError("manifest_sha256 must be 64 lowercase hex digits")
    registry_id = _require_string(registry["registry_id"], field="registry_id")
    if REGISTRY_ID_PATTERN.fullmatch(registry_id) is None:
        raise FeedbackRegistryError(
            "registry_id must be 'registry_' plus 64 lowercase hex digits"
        )
    question_count = _require_count(registry["question_count"], field="question_count")
    choice_count = _require_count(registry["choice_count"], field="choice_count")
    raw_questions = registry["questions"]
    if not isinstance(raw_questions, list) or not raw_questions:
        raise FeedbackRegistryError("questions must be a non-empty JSON array")

    sort_keys: list[tuple[str, str]] = []
    observed_choices = 0
    for index, raw_question in enumerate(raw_questions):
        field = f"questions[{index}]"
        if not isinstance(raw_question, Mapping) or set(raw_question) != _QUESTION_KEYS:
            raise FeedbackRegistryError(f"{field} does not use the exact schema")
        question_id = _require_identifier(
            raw_question["question_id"], field=f"{field}.question_id"
        )
        question_version = _require_string(
            raw_question["question_version"], field=f"{field}.question_version"
        )
        if QUESTION_VERSION_PATTERN.fullmatch(question_version) is None:
            raise FeedbackRegistryError(
                f"{field}.question_version must be 'qv1_' plus 64 lowercase hex digits"
            )
        for key in ("family", "dataset_id", "question_type", "correct_candidate_id"):
            _require_identifier(raw_question[key], field=f"{field}.{key}")
        _require_string(raw_question["correct_letter"], field=f"{field}.correct_letter")
        raw_choices = raw_question["choices"]
        if not isinstance(raw_choices, Mapping) or len(raw_choices) < 2:
            raise FeedbackRegistryError(
                f"{field}.choices must contain at least two entries"
            )
        choices: dict[str, str] = {}
        candidate_ids: set[str] = set()
        for letter, candidate_id_value in raw_choices.items():
            resolved_letter = _require_string(letter, field=f"{field}.choices letter")
            candidate_id = _require_identifier(
                candidate_id_value,
                field=f"{field}.choices[{resolved_letter!r}]",
            )
            if (
                len(resolved_letter) != 1
                or not "A" <= resolved_letter <= "Z"
                or resolved_letter in choices
            ):
                raise FeedbackRegistryError(
                    f"{field}.choices contains invalid or duplicate letters"
                )
            if candidate_id in candidate_ids:
                raise FeedbackRegistryError(
                    f"{field}.choices contains duplicate candidate IDs"
                )
            choices[resolved_letter] = candidate_id
            candidate_ids.add(candidate_id)
        correct_letter = str(raw_question["correct_letter"])
        if choices.get(correct_letter) != raw_question["correct_candidate_id"]:
            raise FeedbackRegistryError(
                f"{field} correct answer does not match its letter-to-candidate mapping"
            )
        observed_choices += len(choices)
        sort_keys.append((question_id, question_version))

    if len(set(sort_keys)) != len(sort_keys):
        raise FeedbackRegistryError("questions contains duplicate identities")
    if sort_keys != sorted(sort_keys):
        raise FeedbackRegistryError("questions must be sorted by identity")
    if question_count != len(raw_questions) or choice_count != observed_choices:
        raise FeedbackRegistryError("registry question/choice counts do not match rows")

    identity_core = _registry_identity_core(
        release_id=release_id,
        questions=[dict(question) for question in raw_questions],
    )
    if registry_id != _registry_id(identity_core):
        raise FeedbackRegistryError("registry_id does not match registry content")


def serialize_feedback_registry(registry: Mapping[str, Any]) -> str:
    """Serialize a validated registry as stable, human-reviewable JSON."""
    _validate_registry(registry)
    try:
        return (
            json.dumps(
                registry,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError, UnicodeError) as exc:  # Defensive after validation.
        raise FeedbackRegistryError(f"registry cannot be serialized: {exc}") from exc


def _sql_literal(value: str) -> str:
    resolved = _require_string(value, field="SQL text value")
    if "\x00" in resolved:
        raise FeedbackRegistryError("SQL text values cannot contain NUL bytes")
    return "'" + resolved.replace("'", "''") + "'"


def _render_insert(
    table: str,
    columns: tuple[str, ...],
    rows: list[tuple[str | int, ...]],
) -> str:
    if not rows:
        raise FeedbackRegistryError(f"cannot render an empty INSERT for {table}")
    rendered_rows = []
    for row in rows:
        if len(row) != len(columns):
            raise FeedbackRegistryError(f"row shape does not match {table} columns")
        values = [
            str(value) if isinstance(value, int) else _sql_literal(value)
            for value in row
        ]
        rendered_rows.append("    (" + ", ".join(values) + ")")
    return (
        f"insert into {table} (\n"
        + "    "
        + ",\n    ".join(columns)
        + "\n) values\n"
        + ",\n".join(rendered_rows)
        + ";"
    )


def render_feedback_registry_sql(registry: Mapping[str, Any]) -> str:
    """Render append-only PostgreSQL data-migration text for one registry."""
    _validate_registry(registry)
    release_rows = [
        (
            str(registry["release_id"]),
            str(registry["schema_version"]),
            str(registry["manifest_sha256"]),
            str(registry["registry_id"]),
            int(registry["question_count"]),
            int(registry["choice_count"]),
        )
    ]
    question_rows: list[tuple[str | int, ...]] = []
    choice_rows: list[tuple[str | int, ...]] = []
    for question in registry["questions"]:
        choices = question["choices"]
        question_rows.append(
            (
                str(registry["release_id"]),
                str(question["question_id"]),
                str(question["question_version"]),
                str(question["family"]),
                str(question["dataset_id"]),
                str(question["question_type"]),
                str(question["correct_letter"]),
                str(question["correct_candidate_id"]),
                len(choices),
            )
        )
        for letter in sorted(choices):
            choice_rows.append(
                (
                    str(registry["release_id"]),
                    str(question["question_id"]),
                    str(question["question_version"]),
                    str(letter),
                    str(choices[letter]),
                )
            )

    statements = [
        "begin;",
        _render_insert(
            "public.feedback_quiz_releases",
            (
                "release_id",
                "registry_schema_version",
                "manifest_sha256",
                "registry_id",
                "question_count",
                "choice_count",
            ),
            release_rows,
        ),
        _render_insert(
            "public.feedback_quiz_questions",
            (
                "release_id",
                "question_id",
                "question_version",
                "family",
                "dataset_id",
                "question_type",
                "correct_letter",
                "correct_candidate_id",
                "choice_count",
            ),
            question_rows,
        ),
        _render_insert(
            "public.feedback_quiz_choices",
            (
                "release_id",
                "question_id",
                "question_version",
                "letter",
                "candidate_id",
            ),
            choice_rows,
        ),
        "commit;",
    ]
    return "\n\n".join(statements) + "\n"


def _resolved_output_paths(
    bundle_root: Path | str,
    json_output: Path | str,
    sql_output: Path | str,
) -> tuple[Path, Path]:
    root = Path(bundle_root).expanduser().resolve()
    json_path = Path(json_output).expanduser().resolve()
    sql_path = Path(sql_output).expanduser().resolve()
    for label, path in (("JSON", json_path), ("SQL", sql_path)):
        if path == root or root in path.parents:
            raise FeedbackRegistryError(
                f"{label} output must be outside the immutable quiz bundle: {path}"
            )
    if json_path == sql_path:
        raise FeedbackRegistryError("JSON and SQL outputs must use different paths")
    return json_path, sql_path


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _check_exact_output(path: Path, expected: bytes, *, label: str) -> None:
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise FeedbackRegistryError(
            f"cannot read {label} output {path}: {exc}"
        ) from exc
    if observed != expected:
        raise FeedbackRegistryError(
            f"{label} output does not exactly match the attested bundle: {path}"
        )


def export_feedback_registry(
    bundle_root: Path | str,
    *,
    json_output: Path | str,
    sql_output: Path | str,
    check: bool = False,
) -> dict[str, Any]:
    """Build and atomically write, or verify, both registry artifacts."""
    json_path, sql_path = _resolved_output_paths(
        bundle_root,
        json_output,
        sql_output,
    )
    registry = build_feedback_registry(bundle_root)
    json_text = serialize_feedback_registry(registry)
    sql_text = render_feedback_registry_sql(registry)
    if check:
        _check_exact_output(json_path, json_text.encode("utf-8"), label="JSON")
        _check_exact_output(sql_path, sql_text.encode("utf-8"), label="SQL")
        return registry

    _atomic_write_text(json_path, json_text)
    _atomic_write_text(sql_path, sql_text)
    return registry


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "FeedbackRegistryError",
    "build_feedback_registry",
    "export_feedback_registry",
    "render_feedback_registry_sql",
    "serialize_feedback_registry",
]
