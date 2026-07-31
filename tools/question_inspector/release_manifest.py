"""Attest and load a published ArchitectureIQ quiz release.

The inspector remains artifact-only: this module uses only the standard
library plus the inspector's canonical question-version helper.  A present
``quiz_manifest.json`` is a claim about the complete directory, so every claim
is verified before its ``release_id`` can be attached to feedback events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:  # Package import in tests/tools; top-level import in the Streamlit app.
    from .feedback import compute_question_version
except ImportError:  # pragma: no cover - exercised by the app's import style.
    from feedback import compute_question_version


MANIFEST_FILENAME = "quiz_manifest.json"
SUPPORTED_SCHEMA_VERSION = "1.0"
RELEASE_ID_PATTERN = re.compile(r"release_[0-9a-f]{64}\Z")
QUESTION_VERSION_PATTERN = re.compile(r"qv1_[0-9a-f]{64}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

_TOP_LEVEL_REQUIRED = frozenset(
    {
        "schema_version",
        "release_id",
        "source_runs",
        "questions",
        "counts",
        "artifacts",
    }
)
_TOP_LEVEL_OPTIONAL = frozenset({"generated_at"})
_SOURCE_RUN_KEYS = frozenset(
    {
        "run_id",
        "family",
        "dataset_id",
        "path",
        "selected_question_ids",
        "declared_question_count",
        "partial",
    }
)
_QUESTION_KEYS = frozenset(
    {
        "question_id",
        "version",
        "family",
        "dataset_id",
        "path",
        "source_run",
        "source_run_path",
    }
)
_COUNT_KEYS = frozenset(
    {
        "questions",
        "source_runs",
        "datasets",
        "candidate_sets",
        "candidates",
        "artifact_files",
        "artifact_bytes",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "sha256", "size"})
_RELEASE_CORE_KEYS = (
    "schema_version",
    "source_runs",
    "questions",
    "counts",
    "artifacts",
)


class ReleaseManifestError(ValueError):
    """Raised when release metadata or any claimed artifact is invalid."""


@dataclass(frozen=True)
class ManifestQuestion:
    """One immutable question identity declared by a quiz release."""

    question_id: str
    version: str
    path: str
    family: str
    dataset_id: str
    source_run: str
    source_run_path: str


@dataclass(frozen=True)
class _ManifestSourceRun:
    run_id: str
    family: str
    dataset_id: str
    path: str
    selected_question_ids: tuple[str, ...]
    declared_question_count: int
    partial: bool


@dataclass(frozen=True)
class QuizManifest:
    """Fully attested release metadata for one immutable data root."""

    data_root: Path
    release_id: str
    generated_at: str | None
    questions: tuple[ManifestQuestion, ...]
    source_run_count: int
    manifest_sha256: str
    artifact_count: int
    artifact_bytes: int

    @property
    def question_count(self) -> int:
        return len(self.questions)

    def question_dirs(self) -> list[Path]:
        """Return question directories in the manifest's declared order."""
        return [
            (self.data_root / PurePosixPath(item.path)).resolve()
            for item in self.questions
        ]

    def release_id_for(
        self,
        question_id: str,
        question_version: str,
        *,
        question_path: Path | str | None = None,
    ) -> str | None:
        """Return the release only for an exact identity and optional path match."""
        resolved_path = (
            Path(question_path).resolve() if question_path is not None else None
        )
        for item in self.questions:
            expected_path = (self.data_root / PurePosixPath(item.path)).resolve()
            if (
                item.question_id == question_id
                and item.version == question_version
                and (resolved_path is None or resolved_path == expected_path)
            ):
                return self.release_id
        return None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (UnicodeError, ValueError) as exc:
        raise ReleaseManifestError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"{label} must be a JSON object")
    return value


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseManifestError(f"cannot read {label}: {exc}") from exc
    return _strict_json(raw, label=label)


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"{field} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if extra:
        details.append(f"unsupported: {', '.join(extra)}")
    raise ReleaseManifestError(
        f"{field} must use the exact schema ({'; '.join(details)})"
    )


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseManifestError(f"{field} must be a non-empty string")
    return value


def _require_identifier(value: Any, *, field: str) -> str:
    resolved = _require_string(value, field=field)
    if resolved in {".", ".."} or "/" in resolved or "\\" in resolved:
        raise ReleaseManifestError(f"{field} is not a safe identifier")
    return resolved


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ReleaseManifestError(f"{field} must be a {qualifier} integer")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReleaseManifestError(f"{field} must be a boolean")
    return value


def _normalized_path(value: Any, *, field: str) -> str:
    raw = _require_string(value, field=field)
    if "\\" in raw:
        raise ReleaseManifestError(f"{field} must use POSIX path separators")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or raw in {".", ".."}
        or raw != relative.as_posix()
        or ".." in relative.parts
    ):
        raise ReleaseManifestError(f"{field} must be a normalized relative path")
    return relative.as_posix()


def _scan_files(root: Path) -> dict[str, Path]:
    """Return every regular file and reject every symlink/special entry."""
    files: dict[str, Path] = {}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ReleaseManifestError(
                f"cannot inspect bundle directory: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ReleaseManifestError(
                    f"symbolic links are not allowed in a release: {relative}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    files[relative] = path
                else:
                    raise ReleaseManifestError(
                        f"release entries must be regular files or directories: {relative}"
                    )
            except OSError as exc:
                raise ReleaseManifestError(
                    f"cannot inspect release entry {relative}: {exc}"
                ) from exc

    visit(root)
    return files


def _sha256_file(path: Path, *, expected_size: int, field: str) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseManifestError(f"{field} is not a regular file")
        digest = hashlib.sha256()
        observed_size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                observed_size += len(chunk)
    except OSError as exc:
        raise ReleaseManifestError(f"cannot hash {field}: {exc}") from exc
    if observed_size != expected_size:
        raise ReleaseManifestError(
            f"{field} size is {observed_size}, expected {expected_size}"
        )
    return digest.hexdigest()


def _parse_source_runs(document: Mapping[str, Any]) -> tuple[_ManifestSourceRun, ...]:
    raw_runs = document.get("source_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ReleaseManifestError("source_runs must be a non-empty JSON array")
    runs: list[_ManifestSourceRun] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_run in enumerate(raw_runs):
        field = f"source_runs[{index}]"
        item = _require_mapping(raw_run, field=field)
        _require_exact_keys(item, _SOURCE_RUN_KEYS, field=field)
        run_id = _require_identifier(item["run_id"], field=f"{field}.run_id")
        path = _normalized_path(item["path"], field=f"{field}.path")
        if run_id in seen_ids:
            raise ReleaseManifestError(f"duplicate source run id: {run_id!r}")
        if path in seen_paths:
            raise ReleaseManifestError(f"duplicate source run path: {path!r}")
        seen_ids.add(run_id)
        seen_paths.add(path)
        selected = item["selected_question_ids"]
        if not isinstance(selected, list) or not selected:
            raise ReleaseManifestError(
                f"{field}.selected_question_ids must be a non-empty JSON array"
            )
        selected_ids = tuple(
            _require_identifier(value, field=f"{field}.selected_question_ids[{offset}]")
            for offset, value in enumerate(selected)
        )
        if len(set(selected_ids)) != len(selected_ids):
            raise ReleaseManifestError(
                f"{field}.selected_question_ids contains duplicates"
            )
        if list(selected_ids) != sorted(selected_ids):
            raise ReleaseManifestError(f"{field}.selected_question_ids must be sorted")
        declared_count = _require_int(
            item["declared_question_count"],
            field=f"{field}.declared_question_count",
            minimum=1,
        )
        partial = _require_bool(item["partial"], field=f"{field}.partial")
        if len(selected_ids) > declared_count:
            raise ReleaseManifestError(
                f"{field}.selected_question_ids exceeds declared_question_count"
            )
        runs.append(
            _ManifestSourceRun(
                run_id=run_id,
                family=_require_identifier(item["family"], field=f"{field}.family"),
                dataset_id=_require_identifier(
                    item["dataset_id"], field=f"{field}.dataset_id"
                ),
                path=path,
                selected_question_ids=selected_ids,
                declared_question_count=declared_count,
                partial=partial,
            )
        )
    if [item.path for item in runs] != sorted(item.path for item in runs):
        raise ReleaseManifestError("source_runs must be sorted by path")
    return tuple(runs)


def _parse_questions(document: Mapping[str, Any]) -> tuple[ManifestQuestion, ...]:
    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ReleaseManifestError("questions must be a non-empty JSON array")
    questions: list[ManifestQuestion] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_question in enumerate(raw_questions):
        field = f"questions[{index}]"
        item = _require_mapping(raw_question, field=field)
        _require_exact_keys(item, _QUESTION_KEYS, field=field)
        question_id = _require_identifier(
            item["question_id"], field=f"{field}.question_id"
        )
        version = _require_string(item["version"], field=f"{field}.version")
        if QUESTION_VERSION_PATTERN.fullmatch(version) is None:
            raise ReleaseManifestError(
                f"{field}.version must be 'qv1_' plus 64 hex digits"
            )
        path = _normalized_path(item["path"], field=f"{field}.path")
        if question_id in seen_ids:
            raise ReleaseManifestError(f"duplicate question_id: {question_id!r}")
        if path in seen_paths:
            raise ReleaseManifestError(f"duplicate question path: {path!r}")
        seen_ids.add(question_id)
        seen_paths.add(path)
        questions.append(
            ManifestQuestion(
                question_id=question_id,
                version=version,
                family=_require_identifier(item["family"], field=f"{field}.family"),
                dataset_id=_require_identifier(
                    item["dataset_id"], field=f"{field}.dataset_id"
                ),
                path=path,
                source_run=_require_identifier(
                    item["source_run"], field=f"{field}.source_run"
                ),
                source_run_path=_normalized_path(
                    item["source_run_path"], field=f"{field}.source_run_path"
                ),
            )
        )
    sort_keys = [(item.question_id, item.path) for item in questions]
    if sort_keys != sorted(sort_keys):
        raise ReleaseManifestError("questions must be sorted by question_id and path")
    return tuple(questions)


def _parse_artifacts(
    root: Path,
    document: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Path]]:
    raw_artifacts = document.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ReleaseManifestError("artifacts must be a non-empty JSON array")
    artifacts: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        field = f"artifacts[{index}]"
        item = _require_mapping(raw_artifact, field=field)
        _require_exact_keys(item, _ARTIFACT_KEYS, field=field)
        path = _normalized_path(item["path"], field=f"{field}.path")
        if path == MANIFEST_FILENAME:
            raise ReleaseManifestError("quiz_manifest.json cannot inventory itself")
        digest = _require_string(item["sha256"], field=f"{field}.sha256")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseManifestError(
                f"{field}.sha256 must be 64 lowercase hex digits"
            )
        _require_int(item["size"], field=f"{field}.size")
        paths.append(path)
        artifacts.append(item)
    if len(set(paths)) != len(paths):
        raise ReleaseManifestError("artifacts contains duplicate paths")
    if paths != sorted(paths):
        raise ReleaseManifestError("artifacts must be sorted by path")

    physical = _scan_files(root)
    physical.pop(MANIFEST_FILENAME, None)
    declared = set(paths)
    actual = set(physical)
    if declared != actual:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing[:5])}")
        if extra:
            details.append(f"extra: {', '.join(extra[:5])}")
        raise ReleaseManifestError(
            f"artifact inventory does not match physical files ({'; '.join(details)})"
        )

    for index, (path, item) in enumerate(zip(paths, artifacts, strict=True)):
        digest = _sha256_file(
            physical[path],
            expected_size=int(item["size"]),
            field=f"artifacts[{index}] {path}",
        )
        if digest != item["sha256"]:
            raise ReleaseManifestError(
                f"artifacts[{index}] {path} SHA-256 does not match the manifest"
            )
    return tuple(artifacts), physical


def _parse_counts(document: Mapping[str, Any]) -> dict[str, int]:
    raw_counts = _require_mapping(document.get("counts"), field="counts")
    _require_exact_keys(raw_counts, _COUNT_KEYS, field="counts")
    return {
        key: _require_int(raw_counts[key], field=f"counts.{key}") for key in _COUNT_KEYS
    }


def _validate_source_artifacts(
    root: Path,
    runs: tuple[_ManifestSourceRun, ...],
    questions: tuple[ManifestQuestion, ...],
    artifact_paths: set[str],
) -> None:
    run_by_id = {item.run_id: item for item in runs}
    selected_by_run: dict[str, list[str]] = {item.run_id: [] for item in runs}
    for question in questions:
        run = run_by_id.get(question.source_run)
        if run is None:
            raise ReleaseManifestError(
                f"question {question.question_id!r} references an unknown source run"
            )
        if question.source_run_path != run.path:
            raise ReleaseManifestError(
                f"question {question.question_id!r} source_run_path does not match its run"
            )
        if PurePosixPath(question.path).parent.as_posix() != run.path:
            raise ReleaseManifestError(
                f"question {question.question_id!r} path is not inside its source run"
            )
        if question.family != run.family or question.dataset_id != run.dataset_id:
            raise ReleaseManifestError(
                f"question {question.question_id!r} family/dataset does not match its run"
            )
        question_file_relative = f"{question.path}/question.json"
        if question_file_relative not in artifact_paths:
            raise ReleaseManifestError(
                f"question {question.question_id!r} question.json is not inventoried"
            )
        question_document = _read_strict_json(
            root / PurePosixPath(question_file_relative),
            label=f"question {question.question_id}",
        )
        if question_document.get("question_id") != question.question_id:
            raise ReleaseManifestError(
                f"question {question.question_id!r} question_id does not match question.json"
            )
        if (
            question_document.get("family") != question.family
            or question_document.get("dataset_id") != question.dataset_id
        ):
            raise ReleaseManifestError(
                f"question {question.question_id!r} family/dataset does not match question.json"
            )
        if question_document.get("question_run_id") not in {None, question.source_run}:
            raise ReleaseManifestError(
                f"question {question.question_id!r} source run does not match question.json"
            )
        question_run_path = question_document.get("question_run_path")
        if (
            question_run_path is not None
            and _normalized_path(
                question_run_path,
                field=f"question {question.question_id!r} question_run_path",
            )
            != run.path
        ):
            raise ReleaseManifestError(
                f"question {question.question_id!r} question_run_path does not match its run"
            )
        observed_version = compute_question_version(question_document)
        if observed_version != question.version:
            raise ReleaseManifestError(
                f"question {question.question_id!r} version does not match question.json"
            )
        selected_by_run[run.run_id].append(question.question_id)

    for run in runs:
        if PurePosixPath(run.path).name != run.run_id:
            raise ReleaseManifestError(
                f"source run {run.run_id!r} ID does not match its path"
            )
        run_file_relative = f"{run.path}/run.json"
        if run_file_relative not in artifact_paths:
            raise ReleaseManifestError(
                f"source run {run.run_id!r} run.json is not inventoried"
            )
        run_document = _read_strict_json(
            root / PurePosixPath(run_file_relative), label=f"source run {run.run_id}"
        )
        for field, expected in (
            ("run_id", run.run_id),
            ("family", run.family),
            ("dataset_id", run.dataset_id),
        ):
            if run_document.get(field) != expected:
                raise ReleaseManifestError(
                    f"source run {run.run_id!r} {field} does not match run.json"
                )
        declared_ids = run_document.get("question_ids")
        if not isinstance(declared_ids, list) or not declared_ids:
            raise ReleaseManifestError(
                f"source run {run.run_id!r} question_ids must be a string array"
            )
        declared_ids = [
            _require_identifier(
                item, field=f"source run {run.run_id!r} question_ids[{index}]"
            )
            for index, item in enumerate(declared_ids)
        ]
        if len(set(declared_ids)) != len(declared_ids):
            raise ReleaseManifestError(
                f"source run {run.run_id!r} question_ids contains duplicates"
            )
        if "num_questions" in run_document:
            run_question_count = _require_int(
                run_document["num_questions"],
                field=f"source run {run.run_id!r} num_questions",
                minimum=1,
            )
            if run_question_count != len(declared_ids):
                raise ReleaseManifestError(
                    f"source run {run.run_id!r} num_questions does not match question_ids"
                )
        if run.declared_question_count != len(declared_ids):
            raise ReleaseManifestError(
                f"source run {run.run_id!r} declared_question_count does not match run.json"
            )
        selected = tuple(sorted(selected_by_run[run.run_id]))
        if selected != run.selected_question_ids:
            raise ReleaseManifestError(
                f"source run {run.run_id!r} selected_question_ids does not match questions"
            )
        if not set(selected).issubset(declared_ids):
            raise ReleaseManifestError(
                f"source run {run.run_id!r} selects a question absent from run.json"
            )
        expected_partial = set(selected) != set(declared_ids)
        if run.partial is not expected_partial:
            raise ReleaseManifestError(
                f"source run {run.run_id!r} partial does not match selected questions"
            )


def _parse_manifest(
    root: Path, document: Mapping[str, Any], raw: bytes
) -> QuizManifest:
    actual_top_level = set(document)
    allowed_top_level = _TOP_LEVEL_REQUIRED | _TOP_LEVEL_OPTIONAL
    if not _TOP_LEVEL_REQUIRED.issubset(
        actual_top_level
    ) or not actual_top_level.issubset(allowed_top_level):
        missing = sorted(_TOP_LEVEL_REQUIRED - actual_top_level)
        extra = sorted(actual_top_level - allowed_top_level)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unsupported: {', '.join(extra)}")
        raise ReleaseManifestError(
            f"quiz manifest must use the exact top-level schema ({'; '.join(details)})"
        )

    schema_version = _require_string(document["schema_version"], field="schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ReleaseManifestError(
            f"unsupported quiz manifest schema version: {schema_version!r}"
        )
    release_id = _require_string(document["release_id"], field="release_id")
    if RELEASE_ID_PATTERN.fullmatch(release_id) is None:
        raise ReleaseManifestError("release_id must be 'release_' plus 64 hex digits")
    generated_at = None
    if "generated_at" in document:
        generated_at = _require_string(document["generated_at"], field="generated_at")

    runs = _parse_source_runs(document)
    questions = _parse_questions(document)
    artifacts, _ = _parse_artifacts(root, document)
    counts = _parse_counts(document)
    artifact_paths = {str(item["path"]) for item in artifacts}
    _validate_source_artifacts(root, runs, questions, artifact_paths)

    expected_counts = {
        "questions": len(questions),
        "source_runs": len(runs),
        "datasets": len({(item.family, item.dataset_id) for item in questions}),
        "artifact_files": len(artifacts),
        "artifact_bytes": sum(int(item["size"]) for item in artifacts),
    }
    candidate_sets: set[tuple[str, ...]] = set()
    candidates: set[tuple[str, ...]] = set()
    for item in artifacts:
        parts = PurePosixPath(str(item["path"])).parts
        if len(parts) >= 6 and parts[0] == "datasets" and parts[3] == "candidates":
            candidate_sets.add(parts[:5])
            if len(parts) >= 7:
                candidates.add(parts[:6])
    expected_counts["candidate_sets"] = len(candidate_sets)
    expected_counts["candidates"] = len(candidates)
    for field, expected in expected_counts.items():
        if counts[field] != expected:
            raise ReleaseManifestError(
                f"counts.{field} is {counts[field]}, expected {expected}"
            )

    core = {key: document[key] for key in _RELEASE_CORE_KEYS}
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_release = f"release_{hashlib.sha256(canonical).hexdigest()}"
    if release_id != expected_release:
        raise ReleaseManifestError(
            f"release_id does not match the attested manifest core; expected {expected_release}"
        )

    return QuizManifest(
        data_root=root,
        release_id=release_id,
        generated_at=generated_at,
        questions=questions,
        source_run_count=len(runs),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_count=len(artifacts),
        artifact_bytes=counts["artifact_bytes"],
    )


def load_quiz_manifest(data_root: Path | str) -> QuizManifest | None:
    """Load and attest ``quiz_manifest.json`` from ``data_root``.

    A missing manifest denotes an unversioned development root.  A present
    manifest is never advisory: its full content identity and every physical
    file are verified on every call, without relying on mtimes or a cache.
    """
    requested_root = Path(data_root).expanduser()
    if requested_root.is_symlink():
        raise ReleaseManifestError("the release data root cannot be a symbolic link")
    root = requested_root.resolve()
    path = root / MANIFEST_FILENAME
    if path.is_symlink():
        raise ReleaseManifestError(f"{MANIFEST_FILENAME} cannot be a symbolic link")
    if not path.exists():
        return None
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReleaseManifestError(
            f"cannot inspect {MANIFEST_FILENAME}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReleaseManifestError(f"{MANIFEST_FILENAME} must be a regular file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseManifestError(f"cannot read {MANIFEST_FILENAME}: {exc}") from exc
    document = _strict_json(raw, label=MANIFEST_FILENAME)
    return _parse_manifest(root, document, raw)
