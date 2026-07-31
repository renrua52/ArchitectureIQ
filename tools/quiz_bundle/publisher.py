"""Validate and publish canonical ArchitectureIQ questions as a quiz bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from .versioning import compute_question_version


MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "quiz_manifest.json"
RELEASE_ID_PREFIX = "release_"
FALLBACK_DATASET_FILES = ("train.pt", "test.pt", "transition.npz")
REQUIRED_CANDIDATE_FILES = (
    "candidate_spec.json",
    "model.py",
    "loss.py",
    "optimizer.py",
    "train.py",
    "results/summary.json",
    "results/curves.npz",
)
FORBIDDEN_PROMPT_GT_MARKERS = (
    "correct_letter",
    "seed_results",
    "failed_seeds",
    "mean_test_",
    "std_test_",
    "final_test_",
    "win_rate",
    "summary.json",
    "curves.npz",
)
LEGACY_LOWER_IS_BETTER_METRICS = frozenset({"test_mse", "test_ce"})
PROMPT_ANSWER_PATTERNS = (
    re.compile(
        r"\b(?:correct answer|answer key|winner|best choice)\s*"
        r"(?:is|:|=|-)\s*(?:choice\s*)?[A-Z]\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bchoice\s+[A-Z]\s+is\s+(?:the\s+)?(?:correct|winner|best)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^#{1,6}\s*(?:training results|answer key|ground[- ]truth results?)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\b(?:mean|std|final)\s+test[_ ]\w+\s*[:=]\s*[-+]?\d",
        re.IGNORECASE,
    ),
)


class BundlePublishError(ValueError):
    """Raised when source artifacts are unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True)
class ValidatedQuestion:
    """A validated question and the artifacts needed to publish it."""

    question_dir: Path
    run_dir: Path
    dataset_dir: Path
    question: dict[str, Any]
    run: dict[str, Any]
    prompt_file: Path
    candidate_dirs: tuple[Path, ...]
    candidate_set_dirs: tuple[Path, ...]
    dataset_files: tuple[Path, ...]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BundlePublishError(f"missing {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard numeric constant {constant}")
            ),
        )
    except (OSError, ValueError) as exc:
        raise BundlePublishError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundlePublishError(f"{label} must be a JSON object: {path}")
    return value


def _require_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundlePublishError(f"{field} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise BundlePublishError(f"{field} is not a safe identifier: {value!r}")
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BundlePublishError(f"{field} must be a JSON object")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise BundlePublishError(f"{field} must be a {qualifier} integer")
    return value


def _require_number(
    value: Any,
    *,
    field: str,
    finite: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundlePublishError(f"{field} must be a number")
    resolved = float(value)
    if finite and not math.isfinite(resolved):
        raise BundlePublishError(f"{field} must be finite")
    return resolved


def _candidate_budget_total(spec: Mapping[str, Any], *, candidate_id: str) -> int:
    budget = _require_mapping(
        spec.get("budget"), field=f"candidate {candidate_id} budget"
    )
    steps = _require_int(
        budget.get("training_steps"),
        field=f"candidate {candidate_id} budget.training_steps",
        minimum=1,
    )
    batch_size = _require_int(
        budget.get("batch_size"),
        field=f"candidate {candidate_id} budget.batch_size",
        minimum=1,
    )
    total = _require_int(
        budget.get("total_samples_seen"),
        field=f"candidate {candidate_id} budget.total_samples_seen",
        minimum=1,
    )
    if steps * batch_size != total:
        raise BundlePublishError(
            f"candidate {candidate_id} violates training_steps × batch_size = "
            f"total_samples_seen ({steps} × {batch_size} != {total})"
        )
    return total


def _question_evaluation(
    question: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
) -> tuple[str, int, int, bool]:
    evaluation = _require_mapping(
        question.get("evaluation"), field="question.evaluation"
    )
    metric = _require_identifier(
        evaluation.get("selection_metric"),
        field="question.evaluation.selection_metric",
    )
    if dataset_spec.get("selection_metric") != metric:
        raise BundlePublishError(
            "question evaluation metric does not match dataset_spec.selection_metric"
        )
    n_seeds = _require_int(
        evaluation.get("n_seeds"),
        field="question.evaluation.n_seeds",
        minimum=1,
    )
    base_seed = _require_int(
        evaluation.get("base_seed"),
        field="question.evaluation.base_seed",
    )
    dataset_direction = dataset_spec.get("higher_is_better")
    question_direction = evaluation.get("higher_is_better")
    if dataset_direction is not None and not isinstance(dataset_direction, bool):
        raise BundlePublishError("dataset_spec.higher_is_better must be a boolean")
    if question_direction is not None and not isinstance(question_direction, bool):
        raise BundlePublishError(
            "question.evaluation.higher_is_better must be a boolean"
        )
    if (dataset_direction is None) != (question_direction is None):
        raise BundlePublishError(
            "higher_is_better must be declared in both dataset_spec and "
            "question.evaluation"
        )
    if (
        dataset_direction is not None
        and question_direction is not None
        and dataset_direction != question_direction
    ):
        raise BundlePublishError("question and dataset disagree about higher_is_better")
    if question_direction is not None:
        higher_is_better = question_direction
    elif dataset_direction is not None:
        higher_is_better = dataset_direction
    elif metric in LEGACY_LOWER_IS_BETTER_METRICS:
        higher_is_better = False
    else:
        raise BundlePublishError(
            f"selection metric {metric!r} must declare higher_is_better"
        )
    return metric, n_seeds, base_seed, higher_is_better


def _validate_prompt(prompt_path: Path, *, selection_metric: str) -> str:
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BundlePublishError(f"cannot read prompt at {prompt_path}: {exc}") from exc
    if not prompt.strip():
        raise BundlePublishError(f"prompt is empty: {prompt_path}")
    lowered = prompt.lower()
    leaked = [marker for marker in FORBIDDEN_PROMPT_GT_MARKERS if marker in lowered]
    if leaked:
        raise BundlePublishError(
            "prompt contains ground-truth result marker(s): " + ", ".join(leaked)
        )
    normalized_prompt = re.sub(r"[*_`]", "", prompt)
    if any(pattern.search(normalized_prompt) for pattern in PROMPT_ANSWER_PATTERNS):
        raise BundlePublishError("prompt contains an answer or result disclosure")
    if selection_metric != "test_mse" and re.search(
        r"\bbest\s+test\s+mse\b",
        normalized_prompt,
        re.IGNORECASE,
    ):
        raise BundlePublishError(
            "prompt describes best test MSE for a non-test_mse question"
        )
    return prompt


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_reference_text(reference: Any, *, field: str) -> str:
    if not isinstance(reference, str) or not reference:
        raise BundlePublishError(f"{field} must be a non-empty relative path")
    windows = PureWindowsPath(reference)
    normalized = reference.replace("\\", "/")
    parts = normalized.split("/")
    if (
        Path(normalized).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in parts
    ):
        raise BundlePublishError(
            f"{field} contains an absolute path or path traversal: {reference!r}"
        )
    return normalized


def _resolve_reference(
    base: Path,
    reference: Any,
    *,
    data_root: Path,
    field: str,
    kind: str | None = None,
) -> Path:
    normalized = _validate_reference_text(reference, field=field)
    resolved = (base / normalized).resolve()
    if not _is_relative_to(resolved, data_root):
        raise BundlePublishError(
            f"{field} resolves outside data root {data_root}: {reference!r}"
        )
    if kind == "file" and not resolved.is_file():
        raise BundlePublishError(f"missing file referenced by {field}: {resolved}")
    if kind == "dir" and not resolved.is_dir():
        raise BundlePublishError(f"missing directory referenced by {field}: {resolved}")
    return resolved


def _resolve_source(data_root: Path, source: Path | str) -> Path:
    source_path = Path(source)
    candidate = source_path if source_path.is_absolute() else data_root / source_path
    resolved = candidate.resolve()
    if not _is_relative_to(resolved, data_root):
        raise BundlePublishError(f"source is outside data root {data_root}: {source}")
    if resolved.name in {"question.json", "run.json"} and resolved.is_file():
        resolved = resolved.parent
    if not resolved.is_dir():
        raise BundlePublishError(f"source directory does not exist: {resolved}")
    return resolved


def _assert_safe_tree(path: Path, *, root: Path) -> None:
    if path.is_symlink():
        raise BundlePublishError(f"symbolic links are not allowed in bundles: {path}")
    if not _is_relative_to(path.resolve(), root.resolve()):
        raise BundlePublishError(f"artifact resolves outside its root: {path}")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise BundlePublishError(
                    f"symbolic links are not allowed in bundles: {child}"
                )
            if not _is_relative_to(child.resolve(), root.resolve()):
                raise BundlePublishError(f"artifact resolves outside its root: {child}")


def _string_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_leaves(item)


def _dataset_files(dataset_dir: Path, data_root: Path) -> tuple[Path, ...]:
    spec_path = dataset_dir / "dataset_spec.json"
    spec = _read_json(spec_path, label="dataset_spec.json")
    paths = {spec_path}

    canonical_synthesize = dataset_dir / "synthesize.py"
    if not canonical_synthesize.is_file():
        raise BundlePublishError(f"missing synthesize.py: {canonical_synthesize}")
    paths.add(canonical_synthesize.resolve())

    declared = spec.get("files", {})
    if declared is not None and not isinstance(declared, (dict, list, tuple)):
        raise BundlePublishError("dataset_spec.files must be a mapping or list")
    for index, reference in enumerate(_string_leaves(declared)):
        paths.add(
            _resolve_reference(
                dataset_dir,
                reference,
                data_root=data_root,
                field=f"dataset_spec.files[{index}]",
                kind="file",
            )
        )

    # Older specs did not consistently enumerate every materialized artifact.
    for filename in FALLBACK_DATASET_FILES:
        path = dataset_dir / filename
        if path.is_file():
            paths.add(path.resolve())
    return tuple(sorted(paths))


def _validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    candidate_id: str,
    metric: str,
    n_seeds: int,
    base_seed: int,
) -> None:
    if summary.get("candidate_id") != candidate_id:
        raise BundlePublishError(
            f"summary candidate_id does not match candidate {candidate_id}"
        )
    if summary.get("selection_metric") != metric:
        raise BundlePublishError(
            f"candidate {candidate_id} summary selection_metric does not match "
            "the question"
        )
    if summary.get("execution") != "candidate_py_files":
        raise BundlePublishError(
            f"candidate {candidate_id} ground truth was not recorded as "
            "execution='candidate_py_files'"
        )
    summary_n_seeds = _require_int(
        summary.get("n_seeds"),
        field=f"candidate {candidate_id} summary.n_seeds",
        minimum=1,
    )
    summary_base_seed = _require_int(
        summary.get("base_seed"),
        field=f"candidate {candidate_id} summary.base_seed",
    )
    if summary_n_seeds != n_seeds or summary_base_seed != base_seed:
        raise BundlePublishError(
            f"candidate {candidate_id} summary seed configuration does not match "
            "question.evaluation"
        )
    failed_seeds = _require_int(
        summary.get("failed_seeds"),
        field=f"candidate {candidate_id} summary.failed_seeds",
    )
    if failed_seeds > n_seeds:
        raise BundlePublishError(
            f"candidate {candidate_id} summary.failed_seeds exceeds n_seeds"
        )
    if not isinstance(summary.get("excluded"), bool):
        raise BundlePublishError(
            f"candidate {candidate_id} summary.excluded must be a boolean"
        )

    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    final_key = f"final_{metric}"
    all_failed = failed_seeds == n_seeds
    if all_failed:
        if summary.get(mean_key) is not None or summary.get(std_key) is not None:
            raise BundlePublishError(
                f"candidate {candidate_id} fully failed summary metrics must be null"
            )
    else:
        mean = _require_number(
            summary.get(mean_key),
            field=f"candidate {candidate_id} summary.{mean_key}",
        )
        std = _require_number(
            summary.get(std_key),
            field=f"candidate {candidate_id} summary.{std_key}",
        )
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise BundlePublishError(
                f"candidate {candidate_id} summary metrics are invalid"
            )
    seed_results = summary.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != n_seeds:
        raise BundlePublishError(
            f"candidate {candidate_id} summary.seed_results must contain "
            f"exactly {n_seeds} entries"
        )
    observed_failed = 0
    for index, raw_seed in enumerate(seed_results):
        seed_result = _require_mapping(
            raw_seed,
            field=f"candidate {candidate_id} summary.seed_results[{index}]",
        )
        expected_seed = base_seed + index
        result_seed = _require_int(
            seed_result.get("seed"),
            field=f"candidate {candidate_id} summary.seed_results[{index}].seed",
        )
        if result_seed != expected_seed:
            raise BundlePublishError(
                f"candidate {candidate_id} seed_results[{index}].seed must be "
                f"{expected_seed}"
            )
        if not isinstance(seed_result.get("failed"), bool):
            raise BundlePublishError(
                f"candidate {candidate_id} seed_results[{index}].failed must be "
                "a boolean"
            )
        observed_failed += int(seed_result["failed"])
        _require_number(
            seed_result.get(final_key),
            field=(
                f"candidate {candidate_id} summary.seed_results[{index}].{final_key}"
            ),
            finite=not seed_result["failed"],
        )
    if observed_failed != failed_seeds:
        raise BundlePublishError(
            f"candidate {candidate_id} summary.failed_seeds does not match seed_results"
        )


def _validate_candidate_dir(
    candidate_dir: Path,
    *,
    candidate_id: str,
    question: Mapping[str, Any],
    data_root: Path,
    metric: str,
    n_seeds: int,
    base_seed: int,
) -> int:
    if candidate_dir.name != candidate_id:
        raise BundlePublishError(
            f"candidate ID/path mismatch: {candidate_id!r} != {candidate_dir.name!r}"
        )

    spec = _read_json(
        candidate_dir / "candidate_spec.json", label="candidate_spec.json"
    )
    if spec.get("candidate_id") != candidate_id:
        raise BundlePublishError(
            "candidate ID/path mismatch: choice candidate_id does not match "
            f"candidate_spec.json in {candidate_dir}"
        )
    for field in ("family", "dataset_id"):
        if spec.get(field) != question.get(field):
            raise BundlePublishError(
                f"candidate {candidate_id} {field} does not match its question"
            )
    total_samples_seen = _candidate_budget_total(spec, candidate_id=candidate_id)

    for relative in REQUIRED_CANDIDATE_FILES:
        required = candidate_dir / relative
        if not required.is_file():
            raise BundlePublishError(f"missing candidate artifact: {required}")
    declared = spec.get("files", {})
    if declared is not None and not isinstance(declared, (dict, list, tuple)):
        raise BundlePublishError("candidate_spec.files must be a mapping or list")
    for file_index, reference in enumerate(_string_leaves(declared)):
        _resolve_reference(
            candidate_dir,
            reference,
            data_root=data_root,
            field=f"candidate_spec.files[{file_index}]",
            kind="file",
        )

    summary = _read_json(
        candidate_dir / "results" / "summary.json", label="results/summary.json"
    )
    _validate_summary_identity(
        summary,
        candidate_id=candidate_id,
        metric=metric,
        n_seeds=n_seeds,
        base_seed=base_seed,
    )

    _assert_safe_tree(candidate_dir, root=data_root)
    return total_samples_seen


def _validate_candidate_set(
    set_dir: Path,
    *,
    question: Mapping[str, Any],
    data_root: Path,
    metric: str,
    n_seeds: int,
    base_seed: int,
) -> tuple[Path, ...]:
    """Validate a complete candidate set, including every candidate directory."""
    _assert_safe_tree(set_dir, root=data_root)
    set_spec = _read_json(set_dir / "set.json", label="set.json")
    if set_spec.get("set_id") not in {None, set_dir.name}:
        raise BundlePublishError(f"candidate set ID/path mismatch: {set_dir}")
    for field in ("family", "dataset_id"):
        if set_spec.get(field) != question.get(field):
            raise BundlePublishError(
                f"candidate set {set_dir.name} {field} does not match its question"
            )

    candidate_dirs = sorted(
        child.resolve()
        for child in set_dir.iterdir()
        if child.is_dir() and child.name.startswith("c_")
    )
    if not candidate_dirs:
        raise BundlePublishError(f"candidate set contains no candidates: {set_dir}")
    declared_count = _require_int(
        set_spec.get("count"),
        field=f"candidate set {set_dir.name} count",
        minimum=1,
    )
    if declared_count != len(candidate_dirs):
        raise BundlePublishError(
            f"candidate set {set_dir.name} declares {declared_count} candidates "
            f"but contains {len(candidate_dirs)}"
        )
    candidate_totals: set[int] = set()
    for candidate_dir in candidate_dirs:
        candidate_id = _require_identifier(
            candidate_dir.name, field="candidate directory name"
        )
        candidate_totals.add(
            _validate_candidate_dir(
                candidate_dir,
                candidate_id=candidate_id,
                question=question,
                data_root=data_root,
                metric=metric,
                n_seeds=n_seeds,
                base_seed=base_seed,
            )
        )
    set_budget = _require_mapping(
        set_spec.get("budget"), field=f"candidate set {set_dir.name} budget"
    )
    set_total = _require_int(
        set_budget.get("total_samples_seen"),
        field=f"candidate set {set_dir.name} budget.total_samples_seen",
        minimum=1,
    )
    if candidate_totals != {set_total}:
        raise BundlePublishError(
            f"candidate set {set_dir.name} budget does not match its candidates"
        )
    return tuple(candidate_dirs)


def _validate_question_budget(
    question: Mapping[str, Any],
    candidate_totals: Iterable[int],
) -> None:
    budget = _require_mapping(question.get("budget"), field="question.budget")
    mixed = budget.get("mixed", False)
    if not isinstance(mixed, bool):
        raise BundlePublishError("question.budget.mixed must be a boolean")
    totals = sorted(set(candidate_totals))
    if not totals:
        raise BundlePublishError("question has no candidate budgets")
    declared = budget.get("total_samples_seen")
    if len(totals) == 1:
        if mixed:
            raise BundlePublishError(
                "question.budget.mixed cannot be true for one candidate budget"
            )
        declared_total = _require_int(
            declared,
            field="question.budget.total_samples_seen",
            minimum=1,
        )
        if declared_total != totals[0]:
            raise BundlePublishError(
                "question budget does not match its choice candidate specs"
            )
        return

    if mixed is not True:
        raise BundlePublishError(
            "question.budget.mixed must be true for cross-budget choices"
        )
    if not isinstance(declared, list):
        raise BundlePublishError(
            "mixed question.budget.total_samples_seen must be a list"
        )
    declared_totals = [
        _require_int(
            value,
            field=f"question.budget.total_samples_seen[{index}]",
            minimum=1,
        )
        for index, value in enumerate(declared)
    ]
    if declared_totals != totals:
        raise BundlePublishError(
            "mixed question budget must equal sorted unique choice budgets"
        )


def _validate_choice_ground_truth(
    question: Mapping[str, Any],
    choices: list[tuple[Mapping[str, Any], Path]],
    *,
    metric: str,
    n_seeds: int,
    base_seed: int,
    higher_is_better: bool,
) -> None:
    records: list[tuple[str, float]] = []
    candidate_totals: list[int] = []
    for choice, candidate_dir in choices:
        letter = str(choice["letter"])
        candidate_id = str(choice["candidate_id"])
        spec = _read_json(
            candidate_dir / "candidate_spec.json", label="candidate_spec.json"
        )
        summary = _read_json(
            candidate_dir / "results" / "summary.json",
            label="results/summary.json",
        )
        _validate_summary_identity(
            summary,
            candidate_id=candidate_id,
            metric=metric,
            n_seeds=n_seeds,
            base_seed=base_seed,
        )
        if summary["excluded"] is True:
            raise BundlePublishError(
                f"choice candidate {candidate_id} is excluded by ground truth"
            )
        if summary["failed_seeds"] != 0:
            raise BundlePublishError(
                f"choice candidate {candidate_id} has partial seed failures"
            )
        mean = _require_number(
            summary.get(f"mean_{metric}"),
            field=f"choice candidate {candidate_id} mean_{metric}",
        )
        records.append((letter, mean))
        candidate_totals.append(
            _candidate_budget_total(spec, candidate_id=candidate_id)
        )

    ordered = sorted(records, key=lambda item: item[1], reverse=higher_is_better)
    if len(ordered) < 2:
        raise BundlePublishError("question must contain at least two choices")
    winner_letter, winner_mean = ordered[0]
    _, runner_up_mean = ordered[1]
    if winner_mean == runner_up_mean:
        raise BundlePublishError("choice ground truth does not have a unique winner")
    if question.get("correct_letter") != winner_letter:
        raise BundlePublishError(
            "correct_letter does not match the stored ground-truth winner"
        )

    significance = _require_mapping(
        question.get("significance"), field="question.significance"
    )
    if significance.get("passed") is not True:
        raise BundlePublishError("question.significance.passed must be true")
    if significance.get("metric") != metric:
        raise BundlePublishError(
            "question significance metric does not match the selection metric"
        )
    declared_gap = _require_number(
        significance.get("gap"), field="question.significance.gap"
    )
    expected_gap = abs(winner_mean - runner_up_mean)
    if not math.isclose(declared_gap, expected_gap, rel_tol=1e-12, abs_tol=1e-12):
        raise BundlePublishError(
            "question significance gap does not match stored ground truth"
        )
    win_rate = _require_number(
        significance.get("win_rate"), field="question.significance.win_rate"
    )
    if not 0.0 <= win_rate <= 1.0:
        raise BundlePublishError("question.significance.win_rate must be in [0, 1]")

    _validate_question_budget(question, candidate_totals)


def validate_question(
    question_dir: Path | str,
    data_root: Path | str,
    *,
    _candidate_set_cache: dict[Path, tuple[Path, ...]] | None = None,
) -> ValidatedQuestion:
    """Validate one canonical question and every artifact it references."""
    root = Path(data_root).resolve()
    directory = Path(question_dir).resolve()
    if not _is_relative_to(directory, root):
        raise BundlePublishError(f"question is outside data root {root}: {directory}")
    question = _read_json(directory / "question.json", label="question.json")

    question_id = _require_identifier(question.get("question_id"), field="question_id")
    family = _require_identifier(question.get("family"), field="family")
    dataset_id = _require_identifier(question.get("dataset_id"), field="dataset_id")

    run_dir = directory.parent
    run_reference = question.get("question_run_path")
    if run_reference is not None:
        referenced_run = _resolve_reference(
            root,
            run_reference,
            data_root=root,
            field="question_run_path",
            kind="dir",
        )
        if referenced_run != run_dir:
            raise BundlePublishError(
                f"question_run_path does not match question location: {directory}"
            )
    run = _read_json(run_dir / "run.json", label="run.json")
    run_id = _require_identifier(run.get("run_id"), field="run.run_id")
    if run_dir.name != run_id:
        raise BundlePublishError(f"question run ID/path mismatch: {run_dir}")
    if question.get("question_run_id") not in {None, run_id}:
        raise BundlePublishError(f"question_run_id does not match {run_id}")
    for field, expected in (("family", family), ("dataset_id", dataset_id)):
        if run.get(field) != expected:
            raise BundlePublishError(
                f"run {field} does not match question {question_id}"
            )
    declared_ids = run.get("question_ids")
    if not isinstance(declared_ids, list) or not all(
        isinstance(item, str) and item for item in declared_ids
    ):
        raise BundlePublishError("run.question_ids must be a list of strings")
    if len(set(declared_ids)) != len(declared_ids):
        raise BundlePublishError(f"duplicate question_id in run.json: {run_dir}")
    if question_id not in declared_ids:
        raise BundlePublishError(
            f"question {question_id} is not declared by source run {run_id}"
        )

    run_sets = run.get("candidate_sets")
    if not isinstance(run_sets, list) or not run_sets:
        raise BundlePublishError("run.candidate_sets must be a non-empty list")
    run_set_dirs: set[Path] = set()
    for index, reference in enumerate(run_sets):
        run_set_dirs.add(
            _resolve_reference(
                root,
                reference,
                data_root=root,
                field=f"run.candidate_sets[{index}]",
                kind="dir",
            )
        )

    expected_dataset_dir = (root / "datasets" / family / dataset_id).resolve()
    if not expected_dataset_dir.is_dir():
        raise BundlePublishError(f"missing dataset directory: {expected_dataset_dir}")
    dataset_spec = _read_json(
        expected_dataset_dir / "dataset_spec.json", label="dataset_spec.json"
    )
    for field, expected in (("family", family), ("dataset_id", dataset_id)):
        if dataset_spec.get(field) != expected:
            raise BundlePublishError(
                f"dataset_spec {field} does not match question {question_id}"
            )
    metric, n_seeds, base_seed, higher_is_better = _question_evaluation(
        question, dataset_spec
    )

    prompt = question.get("prompt", {})
    if not isinstance(prompt, dict):
        raise BundlePublishError("question.prompt must be a JSON object")
    prompt_path = _resolve_reference(
        directory,
        prompt.get("rendered_path", "prompt.txt"),
        data_root=root,
        field="prompt.rendered_path",
        kind="file",
    )
    _validate_prompt(prompt_path, selection_metric=metric)

    choices = question.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BundlePublishError("question.choices must be a non-empty list")
    if question.get("num_choices") not in {None, len(choices)}:
        raise BundlePublishError("question.num_choices does not match choices")
    letters: list[str] = []
    candidate_ids: set[str] = set()
    choice_candidate_dirs: set[Path] = set()
    choice_candidate_set_dirs: set[Path] = set()
    choice_records: list[tuple[Mapping[str, Any], Path]] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise BundlePublishError(f"choices[{index}] must be a JSON object")
        if choice.get("excluded") is True:
            raise BundlePublishError(
                f"choices[{index}] is marked excluded and cannot be published"
            )
        letter = choice.get("letter")
        if (
            not isinstance(letter, str)
            or len(letter) != 1
            or not ("A" <= letter <= "Z")
        ):
            raise BundlePublishError(f"invalid answer letter in choices[{index}]")
        if letter in letters:
            raise BundlePublishError(f"invalid or duplicate answer letter: {letter!r}")
        letters.append(letter)
        candidate_id = _require_identifier(
            choice.get("candidate_id"), field=f"choices[{index}].candidate_id"
        )
        if candidate_id in candidate_ids:
            raise BundlePublishError(f"duplicate choice candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        candidate_dir = _resolve_reference(
            root,
            choice.get("candidate_path"),
            data_root=root,
            field=f"choices[{index}].candidate_path",
            kind="dir",
        )
        set_dir = _resolve_reference(
            root,
            choice.get("candidate_set_path"),
            data_root=root,
            field=f"choices[{index}].candidate_set_path",
            kind="dir",
        )
        if candidate_dir.name != candidate_id:
            raise BundlePublishError(
                f"candidate ID/path mismatch: {candidate_id!r} != "
                f"{candidate_dir.name!r}"
            )
        if candidate_dir.parent != set_dir:
            raise BundlePublishError(
                "candidate path does not belong to its candidate_set_path: "
                f"{candidate_dir} vs {set_dir}"
            )
        choice_candidate_dirs.add(candidate_dir)
        choice_candidate_set_dirs.add(set_dir)
        choice_records.append((choice, candidate_dir))

    correct_letter = question.get("correct_letter")
    if correct_letter not in letters:
        raise BundlePublishError(
            f"invalid correct_letter {correct_letter!r}; expected one of {letters}"
        )

    declared_sets = question.get("candidate_sets", [])
    if not isinstance(declared_sets, list):
        raise BundlePublishError("question.candidate_sets must be a list")
    declared_set_dirs: set[Path] = set()
    for index, reference in enumerate(declared_sets):
        set_dir = _resolve_reference(
            root,
            reference,
            data_root=root,
            field=f"candidate_sets[{index}]",
            kind="dir",
        )
        _read_json(set_dir / "set.json", label="set.json")
        declared_set_dirs.add(set_dir)
    if not choice_candidate_set_dirs.issubset(declared_set_dirs):
        raise BundlePublishError(
            "choice candidate_set_path is not declared by question.candidate_sets"
        )
    if not declared_set_dirs.issubset(run_set_dirs):
        raise BundlePublishError(
            "question.candidate_sets is not a subset of run.candidate_sets"
        )

    # A set is a canonical artifact. Copying only selected choices while
    # retaining its original set.json would create a misleading, incomplete
    # structure, so every source-run set is validated and published whole.
    cache = _candidate_set_cache if _candidate_set_cache is not None else {}
    all_candidate_dirs: set[Path] = set()
    for set_dir in sorted(run_set_dirs):
        if set_dir not in cache:
            cache[set_dir] = _validate_candidate_set(
                set_dir,
                question=question,
                data_root=root,
                metric=metric,
                n_seeds=n_seeds,
                base_seed=base_seed,
            )
        all_candidate_dirs.update(cache[set_dir])
    if not choice_candidate_dirs.issubset(all_candidate_dirs):
        raise BundlePublishError(
            "a choice references a candidate outside its source run sets"
        )
    _validate_choice_ground_truth(
        question,
        choice_records,
        metric=metric,
        n_seeds=n_seeds,
        base_seed=base_seed,
        higher_is_better=higher_is_better,
    )

    _assert_safe_tree(directory, root=root)
    return ValidatedQuestion(
        question_dir=directory,
        run_dir=run_dir,
        dataset_dir=expected_dataset_dir,
        question=question,
        run=run,
        prompt_file=prompt_path,
        candidate_dirs=tuple(sorted(all_candidate_dirs)),
        candidate_set_dirs=tuple(sorted(run_set_dirs)),
        dataset_files=_dataset_files(expected_dataset_dir, root),
    )


def discover_question_dirs(
    data_root: Path | str, sources: Iterable[Path | str]
) -> list[Path]:
    """Expand question or question-run sources into canonical question dirs."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise BundlePublishError(f"data root does not exist: {root}")
    discovered: list[Path] = []
    for source in sources:
        directory = _resolve_source(root, source)
        if (directory / "question.json").is_file():
            discovered.append(directory)
            continue
        if not (directory / "run.json").is_file():
            raise BundlePublishError(
                f"source is neither a question nor a question run: {directory}"
            )
        run = _read_json(directory / "run.json", label="run.json")
        question_ids = run.get("question_ids")
        if not isinstance(question_ids, list) or not all(
            isinstance(item, str) and item for item in question_ids
        ):
            raise BundlePublishError("run.question_ids must be a list of strings")
        if len(set(question_ids)) != len(question_ids):
            raise BundlePublishError(f"duplicate question_id in run: {directory}")
        by_id: dict[str, Path] = {}
        for question_file in sorted(directory.glob("*/question.json")):
            question = _read_json(question_file, label="question.json")
            question_id = _require_identifier(
                question.get("question_id"), field="question_id"
            )
            if question_id in by_id:
                raise BundlePublishError(
                    f"duplicate question_id {question_id!r} in run {directory}"
                )
            by_id[question_id] = question_file.parent.resolve()
        missing = sorted(set(question_ids) - set(by_id))
        extra = sorted(set(by_id) - set(question_ids))
        if missing:
            raise BundlePublishError(
                f"run {directory.name} is missing question directories: {missing}"
            )
        if extra:
            raise BundlePublishError(
                f"run {directory.name} has undeclared question directories: {extra}"
            )
        discovered.extend(by_id[question_id] for question_id in question_ids)
    if not discovered:
        raise BundlePublishError("at least one question or question run is required")
    return discovered


def _question_json_paths(root: Path) -> list[Path]:
    datasets = root / "datasets"
    if not datasets.is_dir():
        return []
    return sorted(datasets.glob("*/*/questions/*/*/question.json"))


def _validated_questions(
    root: Path, question_dirs: Iterable[Path]
) -> list[ValidatedQuestion]:
    validated: list[ValidatedQuestion] = []
    seen: dict[str, Path] = {}
    candidate_set_cache: dict[Path, tuple[Path, ...]] = {}
    for directory in question_dirs:
        item = validate_question(
            directory, root, _candidate_set_cache=candidate_set_cache
        )
        question_id = item.question["question_id"]
        if question_id in seen:
            raise BundlePublishError(
                f"duplicate question_id {question_id!r}: {seen[question_id]} and "
                f"{item.question_dir}"
            )
        seen[question_id] = item.question_dir
        validated.append(item)
    return sorted(
        validated, key=lambda item: (item.question["question_id"], item.question_dir)
    )


def _copy_map(
    data_root: Path, questions: Iterable[ValidatedQuestion]
) -> dict[Path, Path]:
    files: dict[Path, Path] = {}

    def add(source: Path) -> None:
        resolved = source.resolve()
        if not resolved.is_file():
            raise BundlePublishError(f"missing artifact: {resolved}")
        relative = resolved.relative_to(data_root)
        previous = files.get(relative)
        if previous is not None and previous != resolved:
            raise BundlePublishError(f"artifact destination collision: {relative}")
        files[relative] = resolved

    for item in questions:
        add(item.question_dir / "question.json")
        add(item.prompt_file)
        add(item.run_dir / "run.json")
        for source in item.dataset_files:
            add(source)
        # Candidate sets are published whole, but only through their canonical
        # allowlist. Runtime caches and user-created files must never become
        # release artifacts merely because they share a source directory.
        for set_dir in item.candidate_set_dirs:
            add(set_dir / "set.json")
        for candidate_dir in item.candidate_dirs:
            for relative in REQUIRED_CANDIDATE_FILES:
                add(candidate_dir / relative)
    return dict(sorted(files.items(), key=lambda pair: pair[0].as_posix()))


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    return _sha256_file(first) == _sha256_file(second)


def _validate_target(
    target: Path,
    source_questions: Iterable[ValidatedQuestion],
    copies: Mapping[Path, Path],
) -> None:
    if not target.exists():
        return
    if not target.is_dir():
        raise BundlePublishError(f"target is not a directory: {target}")
    _assert_safe_tree(target, root=target)

    existing_ids: dict[str, Path] = {}
    for question_file in _question_json_paths(target):
        question = _read_json(question_file, label="target question.json")
        question_id = _require_identifier(
            question.get("question_id"), field="target question_id"
        )
        if question_id in existing_ids:
            raise BundlePublishError(
                f"duplicate question_id {question_id!r} already exists in target"
            )
        existing_ids[question_id] = question_file
    for item in source_questions:
        question_id = item.question["question_id"]
        if question_id in existing_ids:
            raise BundlePublishError(
                f"duplicate question_id {question_id!r} already exists in target; "
                "replacement is not supported"
            )

    for relative, source in copies.items():
        destination = target / relative
        if destination.exists():
            if not destination.is_file():
                raise BundlePublishError(f"target artifact collision: {destination}")
            if not _files_equal(source, destination):
                raise BundlePublishError(
                    f"target contains different content at {relative.as_posix()}"
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    root_manifest = (root / MANIFEST_FILENAME).resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BundlePublishError(
                f"symbolic links are not allowed in bundles: {path}"
            )
        if not path.is_file() or path.resolve() == root_manifest:
            continue
        relative = path.relative_to(root).as_posix()
        artifacts.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return artifacts


def _manifest_core(root: Path, questions: list[ValidatedQuestion]) -> dict[str, Any]:
    question_entries: list[dict[str, Any]] = []
    runs: dict[Path, dict[str, Any]] = {}
    candidate_paths: set[Path] = set()
    candidate_set_paths: set[Path] = set()
    dataset_paths: set[Path] = set()

    selected_by_run: dict[Path, list[str]] = defaultdict(list)
    for item in questions:
        question = item.question
        question_id = question["question_id"]
        selected_by_run[item.run_dir].append(question_id)
        candidate_paths.update(item.candidate_dirs)
        candidate_set_paths.update(item.candidate_set_dirs)
        dataset_paths.add(item.dataset_dir)
        question_entries.append(
            {
                "question_id": question_id,
                "version": compute_question_version(question),
                "family": question["family"],
                "dataset_id": question["dataset_id"],
                "path": item.question_dir.relative_to(root).as_posix(),
                "source_run": item.run["run_id"],
                "source_run_path": item.run_dir.relative_to(root).as_posix(),
            }
        )

    for run_dir, selected_ids in selected_by_run.items():
        item = next(question for question in questions if question.run_dir == run_dir)
        declared_ids = item.run["question_ids"]
        selected = sorted(selected_ids)
        runs[run_dir] = {
            "run_id": item.run["run_id"],
            "family": item.run["family"],
            "dataset_id": item.run["dataset_id"],
            "path": run_dir.relative_to(root).as_posix(),
            "selected_question_ids": selected,
            "declared_question_count": len(declared_ids),
            "partial": set(selected) != set(declared_ids),
        }

    canonical_paths = set(_copy_map(root, questions))
    root_manifest = (root / MANIFEST_FILENAME).resolve()
    actual_paths = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != root_manifest
    }
    unexpected = sorted(
        actual_paths - canonical_paths, key=lambda path: path.as_posix()
    )
    if unexpected:
        preview = ", ".join(path.as_posix() for path in unexpected[:5])
        suffix = "" if len(unexpected) <= 5 else f" (+{len(unexpected) - 5} more)"
        raise BundlePublishError(
            f"bundle contains non-canonical artifact(s): {preview}{suffix}"
        )
    missing = sorted(canonical_paths - actual_paths, key=lambda path: path.as_posix())
    if missing:
        preview = ", ".join(path.as_posix() for path in missing[:5])
        raise BundlePublishError(f"bundle is missing canonical artifact(s): {preview}")
    artifacts = _artifact_inventory(root)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_runs": [runs[path] for path in sorted(runs)],
        "questions": sorted(
            question_entries, key=lambda entry: (entry["question_id"], entry["path"])
        ),
        "counts": {
            "questions": len(question_entries),
            "source_runs": len(runs),
            "datasets": len(dataset_paths),
            "candidate_sets": len(candidate_set_paths),
            "candidates": len(candidate_paths),
            "artifact_files": len(artifacts),
            "artifact_bytes": sum(item["size"] for item in artifacts),
        },
        "artifacts": artifacts,
    }


def _release_id(core: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{RELEASE_ID_PREFIX}{hashlib.sha256(canonical).hexdigest()}"


def build_bundle_manifest(
    bundle_root: Path | str, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Scan and validate an existing bundle, returning its stable manifest.

    ``generated_at`` is descriptive only and is deliberately excluded from the
    release content hash.
    """
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise BundlePublishError(f"bundle root does not exist: {root}")
    _assert_safe_tree(root, root=root)
    question_files = _question_json_paths(root)
    if not question_files:
        raise BundlePublishError(f"bundle contains no questions: {root}")
    questions = _validated_questions(root, (path.parent for path in question_files))
    core = _manifest_core(root, questions)
    manifest: dict[str, Any] = {
        "schema_version": core.pop("schema_version"),
        "release_id": _release_id({"schema_version": MANIFEST_SCHEMA_VERSION, **core}),
    }
    if generated_at is not None:
        if not isinstance(generated_at, str) or not generated_at:
            raise BundlePublishError("generated_at must be a non-empty string")
        manifest["generated_at"] = generated_at
    manifest.update(core)
    return manifest


def _write_manifest_value(bundle_root: Path, manifest: Mapping[str, Any]) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)
    destination = bundle_root / MANIFEST_FILENAME
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_FILENAME}.", dir=bundle_root, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def write_bundle_manifest(
    bundle_root: Path | str, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Validate an existing bundle and atomically refresh quiz_manifest.json."""
    root = Path(bundle_root).resolve()
    manifest = build_bundle_manifest(root, generated_at=generated_at)
    _write_manifest_value(root, manifest)
    return manifest


def _apply_copy_map(target: Path, copies: Mapping[Path, Path]) -> None:
    for relative, source in copies.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        shutil.copy2(source, destination)


def _create_directories(directory: Path, created: list[Path]) -> None:
    """Create ``directory`` and remember only paths created by this call chain."""

    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            break
        current = current.parent
    if not current.is_dir():
        raise BundlePublishError(f"target parent is not a directory: {current}")

    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            if not path.is_dir():
                raise BundlePublishError(
                    f"target parent is not a directory: {path}"
                ) from None
        else:
            created.append(path)


def _copy_to_sibling_temporary(source: Path, destination: Path) -> Path:
    """Copy one staged file to a fully written temporary beside ``destination``."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        descriptor = -1
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _install_staged_file(
    source: Path,
    destination: Path,
    *,
    installed: list[Path],
    created_directories: list[Path],
) -> None:
    """Atomically install one complete staged file without replacing a target."""

    _create_directories(destination.parent, created_directories)
    temporary = _copy_to_sibling_temporary(source, destination)
    try:
        try:
            # A same-filesystem hard link is an atomic no-clobber install.  Unlike
            # os.replace(), it cannot overwrite a path created after validation.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise BundlePublishError(
                f"target artifact appeared while publishing: {destination}"
            ) from exc
        installed.append(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_staged_file(source: Path, destination: Path) -> None:
    """Atomically restore a pre-publish file during in-process rollback."""

    temporary = _copy_to_sibling_temporary(source, destination)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rollback_publish(
    *,
    target_root: Path,
    installed: list[Path],
    created_directories: list[Path],
    manifest_write_started: bool,
    manifest_backup: Path | None,
) -> None:
    """Undo files created by one publish attempt after an in-process failure."""

    failures: list[str] = []
    manifest_path = target_root / MANIFEST_FILENAME
    if manifest_write_started:
        try:
            if manifest_backup is None:
                manifest_path.unlink(missing_ok=True)
            else:
                _restore_staged_file(manifest_backup, manifest_path)
        except OSError as exc:
            failures.append(f"restore {manifest_path}: {exc}")

    for path in reversed(installed):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failures.append(f"remove {path}: {exc}")

    for path in reversed(created_directories):
        try:
            path.rmdir()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append(f"remove directory {path}: {exc}")

    if failures:
        raise BundlePublishError(
            "quiz bundle publish rollback was incomplete: " + "; ".join(failures)
        )


def publish_quiz_bundle(
    data_root: Path | str,
    sources: Iterable[Path | str],
    target: Path | str,
    *,
    dry_run: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Publish selected questions, returning the resulting bundle manifest.

    Question JSON and run JSON artifacts are copied byte-for-byte. A question
    directory source therefore creates a partial run when its canonical
    ``run.json`` declares additional questions; the manifest records this via
    ``source_runs[].partial`` and ``selected_question_ids``.
    """
    root = Path(data_root).resolve()
    target_root = Path(target).resolve()
    question_dirs = discover_question_dirs(root, sources)
    questions = _validated_questions(root, question_dirs)
    copies = _copy_map(root, questions)
    _validate_target(target_root, questions, copies)

    with tempfile.TemporaryDirectory(prefix="architecture_iq_quiz_bundle_") as temp:
        staged = Path(temp) / "bundle"
        if target_root.exists():
            shutil.copytree(target_root, staged)
        else:
            staged.mkdir(parents=True)
        _apply_copy_map(staged, copies)
        projected = build_bundle_manifest(staged, generated_at=generated_at)

        if dry_run:
            return projected

        preexisting = {
            relative for relative in copies if (target_root / relative).exists()
        }
        manifest_path = target_root / MANIFEST_FILENAME
        manifest_backup = (
            staged / MANIFEST_FILENAME if manifest_path.is_file() else None
        )
        installed: list[Path] = []
        created_directories: list[Path] = []
        manifest_write_started = False
        try:
            _create_directories(target_root, created_directories)
            for relative in copies:
                if relative in preexisting:
                    continue
                _install_staged_file(
                    staged / relative,
                    target_root / relative,
                    installed=installed,
                    created_directories=created_directories,
                )
            actual = build_bundle_manifest(target_root, generated_at=generated_at)
            if actual != projected:
                raise BundlePublishError(
                    "target changed while publishing; manifest not written"
                )
            manifest_write_started = True
            _write_manifest_value(target_root, actual)
        except BaseException as exc:
            try:
                _rollback_publish(
                    target_root=target_root,
                    installed=installed,
                    created_directories=created_directories,
                    manifest_write_started=manifest_write_started,
                    manifest_backup=manifest_backup,
                )
            except BundlePublishError as rollback_exc:
                raise rollback_exc from exc
            raise
        return actual
