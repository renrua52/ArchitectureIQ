"""Read-only adapter for the wide-v2 setting-to-loss dataset.

The dataset builder intentionally owns candidate generation and ground truth.
This module starts *after* export: it validates the frozen JSONL artifacts,
combines completed environments, and defines target-free grouped evaluation
protocols.  It never imports or executes candidate code and never trains a
model.

Typical preflight usage from the repository root is::

    .venv/bin/python -m tools.meta_model_study.wide validate --require-complete

Omit ``--require-complete`` while ground truth is still being produced.  The
command then reports completed, partial, missing, and invalid environments
without treating expected partial work as a corrupt dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.meta_model_study.features import load_jsonl
from tools.meta_model_study.metrics import (
    load_final_seed_losses,
    ranking_metrics,
    regression_metrics,
)
from tools.meta_model_study.ood import exact_three_choice_accuracy


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/meta_model/setting_to_loss_wide_v2"
DEFAULT_PLAN = REPO_ROOT / "tools/meta_model_dataset/plan_wide_v2.json"

SCHEMA_VERSION = "meta_model_wide_adapter_v1"
SNAPSHOT_SCHEMA_VERSION = "meta_model_wide_snapshot_v1"
ROW_FILES = ("all.jsonl", "train.jsonl", "validation.jsonl")
COMPLETE_FILES = (*ROW_FILES, "manifest.json")
GROUP_LABEL_KEYS = (
    "phase",
    "family",
    "dataset",
    "environment",
    "dataset_cohort",
)
GROUP_AXES = ("environment", "dataset", "family")
SPLITS = ("all", "train", "validation")

_DATASET_DESCRIPTION_EXCLUDED_KEYS = frozenset(
    {"dataset_id", "dataset_path", "files", "significance", "group_labels", "experiment_id"}
)


def _safe_dataset_description(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable, target-free dataset semantics from a frozen dataset spec."""

    def clean(item: Any, key: str = "") -> Any:
        normalized = key.lower()
        if normalized in _DATASET_DESCRIPTION_EXCLUDED_KEYS or "seed" in normalized:
            return None
        if isinstance(item, Mapping):
            result = {
                str(child_key): cleaned
                for child_key in sorted(item, key=str)
                if (cleaned := clean(item[child_key], str(child_key))) is not None
            }
            return result or None
        if isinstance(item, (list, tuple)):
            result = [cleaned for child in item if (cleaned := clean(child, key)) is not None]
            return result or None
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported dataset description value: {type(item).__name__}")

    cleaned = clean(value)
    return {} if cleaned is None else dict(cleaned)


def _attach_dataset_context(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    dataset_id: str,
    dataset_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = {
        "dataset_id": dataset_id,
        "description": {"family": family, **_safe_dataset_description(dataset_spec)},
    }
    return [{**dict(row), "dataset_context": context} for row in rows]


@dataclass(frozen=True)
class WideEnvironment:
    """One validated environment and its three exported row views."""

    path: Path
    experiment_id: str
    family: str
    dataset_id: str
    n_seeds: int
    manifest: dict[str, Any]
    all_rows: tuple[dict[str, Any], ...]
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]

    def rows(self, split: str) -> tuple[dict[str, Any], ...]:
        if split == "all":
            return self.all_rows
        if split == "train":
            return self.train_rows
        if split == "validation":
            return self.validation_rows
        raise ValueError(f"Unknown split {split!r}; expected one of: {', '.join(SPLITS)}")


@dataclass(frozen=True)
class WideCorpus:
    """Deterministically combined rows from a set of validated environments."""

    root: Path
    environments: tuple[WideEnvironment, ...]
    n_seeds: int
    all_rows: tuple[dict[str, Any], ...]
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]

    def rows(self, split: str) -> tuple[dict[str, Any], ...]:
        if split == "all":
            return self.all_rows
        if split == "train":
            return self.train_rows
        if split == "validation":
            return self.validation_rows
        raise ValueError(f"Unknown split {split!r}; expected one of: {', '.join(SPLITS)}")


@dataclass(frozen=True)
class WideSnapshot:
    """One validated frozen selection of environments across dataset roots."""

    path: Path
    sha256: str
    manifest: dict[str, Any]
    corpus: WideCorpus


@dataclass(frozen=True)
class WideGroupFold:
    """One leave-one-group-out fold over a target-free row label."""

    axis: str
    held_out_group: str
    train_indices: np.ndarray
    test_indices: np.ndarray


# Compatibility for callers of the first wide adapter revision.  The distinct
# class name prevents confusion with tools.meta_model_study.ood.GroupFold.
GroupFold = WideGroupFold


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for fingerprint in sorted(
        str(row["example_fingerprint_sha256"]) for row in rows
    ):
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _fingerprint(value: Any, *, context: str) -> str:
    fingerprint = str(value)
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(f"{context} has an invalid SHA-256 fingerprint")
    return fingerprint


def _manifest_n_seeds(manifest: Mapping[str, Any], *, context: str) -> int:
    ground_truth = manifest.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError(f"{context} manifest is missing ground_truth")
    n_seeds = _positive_int(
        ground_truth.get("n_seeds"), name=f"{context}.ground_truth.n_seeds"
    )
    config = manifest.get("config")
    if isinstance(config, Mapping):
        config_gt = config.get("ground_truth")
        if isinstance(config_gt, Mapping) and "n_seeds" in config_gt:
            config_n_seeds = _positive_int(
                config_gt["n_seeds"], name=f"{context}.config.ground_truth.n_seeds"
            )
            if config_n_seeds != n_seeds:
                raise ValueError(
                    f"{context} manifest disagrees about n_seeds: "
                    f"{n_seeds} != {config_n_seeds}"
                )
    return n_seeds


def _validated_group_labels(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    labels: dict[str, str] = {}
    for key in GROUP_LABEL_KEYS:
        label = value.get(key)
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}.{key} must be a non-empty string")
        labels[key] = label
    return labels


def group_value(row: Mapping[str, Any], axis: str) -> str:
    """Return one frozen target-free group label."""

    if axis not in GROUP_AXES:
        raise ValueError(f"Unknown group axis {axis!r}; expected: {', '.join(GROUP_AXES)}")
    return _validated_group_labels(
        row.get("group_labels"), context="row.group_labels"
    )[axis]


def _validate_row(
    row: Mapping[str, Any],
    *,
    experiment_id: str,
    expected_split: str | None,
    expected_n_seeds: int,
    expected_family: str | None,
    expected_dataset: str | None,
    expected_group_labels: Mapping[str, str] | None,
    index: int,
) -> tuple[str, str, str, str]:
    context = f"{experiment_id}/{expected_split or 'all'}[{index}]"
    if row.get("experiment_id") != experiment_id:
        raise ValueError(f"{context} has a mismatched experiment_id")
    split = row.get("split")
    if split not in {"train", "validation"}:
        raise ValueError(f"{context} has invalid split={split!r}")
    if expected_split is not None and split != expected_split:
        raise ValueError(f"{context} has split={split!r}")
    if row.get("usable_for_regression") is not True:
        raise ValueError(f"{context} is not usable_for_regression")

    fingerprint = _fingerprint(
        row.get("example_fingerprint_sha256"), context=context
    )
    family = row.get("family")
    dataset_id = row.get("dataset_id")
    if not isinstance(family, str) or not family:
        raise ValueError(f"{context} has no family")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"{context} has no dataset_id")
    if expected_family is not None and family != expected_family:
        raise ValueError(f"{context} has family={family!r}, expected {expected_family!r}")
    if expected_dataset is not None and dataset_id != expected_dataset:
        raise ValueError(
            f"{context} has dataset_id={dataset_id!r}, expected {expected_dataset!r}"
        )

    row_labels = _validated_group_labels(
        row.get("group_labels"), context=f"{context}.group_labels"
    )
    if expected_group_labels is not None:
        for key in GROUP_LABEL_KEYS:
            if row_labels[key] != expected_group_labels[key]:
                raise ValueError(
                    f"{context} group label {key!r} disagrees with the manifest"
                )
    if row_labels["environment"] != experiment_id:
        raise ValueError(f"{context} environment group does not match experiment_id")
    if row_labels["dataset"] != dataset_id:
        raise ValueError(f"{context} dataset group does not match dataset_id")
    if row_labels["family"] != family:
        raise ValueError(f"{context} family group does not match family")

    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"{context}.target must be a mapping")
    row_n_seeds = _positive_int(target.get("n_seeds"), name=f"{context}.target.n_seeds")
    if row_n_seeds != expected_n_seeds:
        raise ValueError(
            f"{context} target.n_seeds={row_n_seeds}, expected {expected_n_seeds}"
        )
    failed_seeds = target.get("failed_seeds", 0)
    if (
        isinstance(failed_seeds, bool)
        or not isinstance(failed_seeds, int)
        or failed_seeds < 0
        or failed_seeds >= expected_n_seeds
    ):
        raise ValueError(f"{context}.target.failed_seeds is invalid")
    mean_loss = _finite_float(target.get("mean_loss"), name=f"{context}.target.mean_loss")
    log_loss = _finite_float(
        target.get("log_mean_loss"), name=f"{context}.target.log_mean_loss"
    )
    if mean_loss <= 0.0:
        raise ValueError(f"{context}.target.mean_loss must be positive")
    if not math.isclose(math.log(mean_loss), log_loss, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{context} mean_loss/log_mean_loss disagree")
    selection_metric = target.get("selection_metric")
    if not isinstance(selection_metric, str) or not selection_metric:
        raise ValueError(f"{context} has no target.selection_metric")

    derived = row.get("derived")
    if not isinstance(derived, Mapping):
        raise ValueError(f"{context}.derived must be a mapping")
    total_params = _finite_float(
        derived.get("total_params"), name=f"{context}.derived.total_params"
    )
    log_params = _finite_float(
        derived.get("log_total_params"), name=f"{context}.derived.log_total_params"
    )
    if total_params <= 0.0 or not math.isclose(
        math.log(total_params), log_params, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(f"{context} has inconsistent total_params/log_total_params")

    setting = row.get("setting")
    if not isinstance(setting, Mapping):
        raise ValueError(f"{context}.setting must be a mapping")
    for namespace in ("model", "optimizer", "loss", "budget"):
        if not isinstance(setting.get(namespace), Mapping):
            raise ValueError(f"{context}.setting.{namespace} must be a mapping")
    budget = setting["budget"]
    batch_size = _positive_int(budget.get("batch_size"), name=f"{context}.budget.batch_size")
    training_steps = _positive_int(
        budget.get("training_steps"), name=f"{context}.budget.training_steps"
    )
    total_samples = _positive_int(
        budget.get("total_samples_seen"), name=f"{context}.budget.total_samples_seen"
    )
    if batch_size * training_steps != total_samples:
        raise ValueError(f"{context} violates the training budget identity")
    return fingerprint, family, dataset_id, selection_metric


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    expected_split: str | None,
    expected_n_seeds: int,
    expected_family: str | None = None,
    expected_dataset: str | None = None,
    expected_group_labels: Mapping[str, str] | None = None,
) -> tuple[set[str], str, str, str]:
    if not rows:
        raise ValueError(f"{experiment_id}/{expected_split or 'all'} is empty")
    fingerprints: set[str] = set()
    families: set[str] = set()
    datasets: set[str] = set()
    metrics: set[str] = set()
    for index, row in enumerate(rows):
        fingerprint, family, dataset, metric = _validate_row(
            row,
            experiment_id=experiment_id,
            expected_split=expected_split,
            expected_n_seeds=expected_n_seeds,
            expected_family=expected_family,
            expected_dataset=expected_dataset,
            expected_group_labels=expected_group_labels,
            index=index,
        )
        if fingerprint in fingerprints:
            raise ValueError(
                f"{experiment_id}/{expected_split or 'all'} has a duplicate fingerprint"
            )
        fingerprints.add(fingerprint)
        families.add(family)
        datasets.add(dataset)
        metrics.add(metric)
    if len(families) != 1 or len(datasets) != 1 or len(metrics) != 1:
        raise ValueError(
            f"{experiment_id}/{expected_split or 'all'} mixes family, dataset, or metric"
        )
    return fingerprints, families.pop(), datasets.pop(), metrics.pop()


def load_environment(
    environment_dir: str | Path,
    *,
    expected_n_seeds: int | None = None,
) -> WideEnvironment:
    """Load one committed environment and audit its row/split contract."""

    path = Path(environment_dir)
    missing = [name for name in COMPLETE_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete environment {path}: missing {', '.join(missing)}")
    experiment_id = path.name
    manifest = _load_json_object(path / "manifest.json", label="environment manifest")
    if manifest.get("experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id} manifest has a mismatched experiment_id")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{experiment_id} manifest.config must be a mapping")
    if config.get("experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id} manifest.config has a mismatched experiment_id")
    config_labels = _validated_group_labels(
        config.get("group_labels"), context=f"{experiment_id}.manifest.config.group_labels"
    )
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, Mapping):
        raise ValueError(f"{experiment_id} manifest.split_policy must be a mapping")
    split_labels = _validated_group_labels(
        split_policy.get("group_labels"),
        context=f"{experiment_id}.manifest.split_policy.group_labels",
    )
    if any(config_labels[key] != split_labels[key] for key in GROUP_LABEL_KEYS):
        raise ValueError(
            f"{experiment_id} config/split_policy group_labels disagree"
        )
    if config_labels["environment"] != experiment_id:
        raise ValueError(f"{experiment_id} manifest environment group disagrees")
    phase = config.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError(f"{experiment_id} manifest.config.phase must be a string")
    if phase != config_labels["phase"]:
        raise ValueError(f"{experiment_id} config.phase/group_labels.phase disagree")

    n_seeds = _manifest_n_seeds(manifest, context=experiment_id)
    if expected_n_seeds is not None and n_seeds != expected_n_seeds:
        raise ValueError(
            f"{experiment_id} uses {n_seeds} seeds; expected {expected_n_seeds}"
        )

    all_rows = load_jsonl(path / "all.jsonl")
    train_rows = load_jsonl(path / "train.jsonl")
    validation_rows = load_jsonl(path / "validation.jsonl")
    all_ids, family, dataset_id, selection_metric = _validate_rows(
        all_rows,
        experiment_id=experiment_id,
        expected_split=None,
        expected_n_seeds=n_seeds,
        expected_family=config_labels["family"],
        expected_dataset=config_labels["dataset"],
        expected_group_labels=config_labels,
    )
    train_ids, _, _, _ = _validate_rows(
        train_rows,
        experiment_id=experiment_id,
        expected_split="train",
        expected_n_seeds=n_seeds,
        expected_family=family,
        expected_dataset=dataset_id,
        expected_group_labels=config_labels,
    )
    validation_ids, _, _, _ = _validate_rows(
        validation_rows,
        experiment_id=experiment_id,
        expected_split="validation",
        expected_n_seeds=n_seeds,
        expected_family=family,
        expected_dataset=dataset_id,
        expected_group_labels=config_labels,
    )
    if train_ids.intersection(validation_ids):
        raise ValueError(f"{experiment_id} train/validation fingerprints overlap")
    if train_ids.union(validation_ids) != all_ids:
        raise ValueError(f"{experiment_id} all.jsonl is not the exact split union")
    all_by_id = {row["example_fingerprint_sha256"]: row for row in all_rows}
    for row in (*train_rows, *validation_rows):
        fingerprint = row["example_fingerprint_sha256"]
        if row != all_by_id[fingerprint]:
            raise ValueError(f"{experiment_id} split row differs from all.jsonl")

    if split_policy.get("assigned_before_ground_truth") is not True:
        raise ValueError(f"{experiment_id} split was not assigned before ground truth")
    if split_policy.get("group_labels_frozen_before_ground_truth") is not True:
        raise ValueError(f"{experiment_id} group labels were not frozen before ground truth")
    if int(split_policy.get("train", -1)) != len(train_rows):
        raise ValueError(f"{experiment_id} manifest train count disagrees")
    if int(split_policy.get("validation", -1)) != len(validation_rows):
        raise ValueError(f"{experiment_id} manifest validation count disagrees")
    config_budget = _positive_int(
        config.get("budget"), name=f"{experiment_id}.manifest.config.budget"
    )
    config_batch_size = _positive_int(
        config.get("batch_size"), name=f"{experiment_id}.manifest.config.batch_size"
    )
    for index, row in enumerate(all_rows):
        budget = row["setting"]["budget"]
        if (
            budget["total_samples_seen"] != config_budget
            or budget["batch_size"] != config_batch_size
        ):
            raise ValueError(
                f"{experiment_id}/all[{index}] budget disagrees with manifest.config"
            )
    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError(f"{experiment_id} manifest.config.dataset_path must be a string")
    selected = manifest.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError(f"{experiment_id} manifest.selected must be a mapping")
    if int(selected.get("total", -1)) != len(all_rows):
        raise ValueError(f"{experiment_id} manifest selected.total disagrees")
    dataset_spec = config.get("dataset_spec")
    if not isinstance(dataset_spec, Mapping):
        raise ValueError(f"{experiment_id} manifest.config.dataset_spec must be a mapping")
    declared_metric = dataset_spec.get("selection_metric")
    if declared_metric != selection_metric:
        raise ValueError(f"{experiment_id} selection metric disagrees with dataset spec")

    all_rows = _attach_dataset_context(
        all_rows, family=family, dataset_id=dataset_id, dataset_spec=dataset_spec
    )
    train_rows = _attach_dataset_context(
        train_rows, family=family, dataset_id=dataset_id, dataset_spec=dataset_spec
    )
    validation_rows = _attach_dataset_context(
        validation_rows, family=family, dataset_id=dataset_id, dataset_spec=dataset_spec
    )

    return WideEnvironment(
        path=path.resolve(),
        experiment_id=experiment_id,
        family=family,
        dataset_id=dataset_id,
        n_seeds=n_seeds,
        manifest=manifest,
        all_rows=tuple(all_rows),
        train_rows=tuple(train_rows),
        validation_rows=tuple(validation_rows),
    )


def discover_environment_dirs(dataset_root: str | Path) -> tuple[list[Path], list[Path]]:
    """Return deterministic ``(complete, partial)`` direct child directories."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    complete: list[Path] = []
    partial: list[Path] = []
    for path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name):
        files = {name for name in COMPLETE_FILES if (path / name).is_file()}
        looks_like_environment = bool(files) or (path / "sampling_manifest.json").is_file()
        if not looks_like_environment:
            continue
        if len(files) == len(COMPLETE_FILES):
            complete.append(path)
        else:
            partial.append(path)
    return complete, partial


def _combine_environments(
    root: Path,
    environments: Sequence[WideEnvironment],
) -> WideCorpus:
    if not environments:
        raise ValueError("Cannot build a corpus without environments")
    seed_counts = {environment.n_seeds for environment in environments}
    if len(seed_counts) != 1:
        raise ValueError(f"Completed environments mix n_seeds values: {sorted(seed_counts)}")

    frozen_environments = tuple(environments)
    all_rows = tuple(
        row for environment in frozen_environments for row in environment.all_rows
    )
    train_rows = tuple(
        row for environment in frozen_environments for row in environment.train_rows
    )
    validation_rows = tuple(
        row
        for environment in frozen_environments
        for row in environment.validation_rows
    )
    all_fingerprints = [row["example_fingerprint_sha256"] for row in all_rows]
    if len(all_fingerprints) != len(set(all_fingerprints)):
        raise ValueError("Example fingerprints collide across environments")
    if {row["example_fingerprint_sha256"] for row in train_rows}.intersection(
        row["example_fingerprint_sha256"] for row in validation_rows
    ):
        raise ValueError("Combined train and validation fingerprints overlap")
    return WideCorpus(
        root=root,
        environments=frozen_environments,
        n_seeds=next(iter(seed_counts)),
        all_rows=all_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


def _snapshot_environment_path(manifest_path: Path, value: Any, *, index: int) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"snapshot.environments[{index}].path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_snapshot(snapshot_manifest: str | Path) -> WideSnapshot:
    """Load a frozen cross-root environment selection without recomputing GT."""

    path = Path(snapshot_manifest).resolve()
    manifest = _load_json_object(path, label="wide snapshot manifest")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported wide snapshot schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    expected_n_seeds = _positive_int(
        manifest.get("n_seeds"), name="snapshot.n_seeds"
    )
    entries = manifest.get("environments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("snapshot.environments must be a non-empty list")

    environments: list[WideEnvironment] = []
    experiment_ids: set[str] = set()
    environment_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        context = f"snapshot.environments[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} must be a mapping")
        experiment_id = entry.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(f"{context}.experiment_id must be a non-empty string")
        if experiment_id in experiment_ids:
            raise ValueError(f"snapshot repeats experiment_id {experiment_id!r}")
        experiment_ids.add(experiment_id)

        environment_path = _snapshot_environment_path(
            path, entry.get("path"), index=index
        )
        if environment_path in environment_paths:
            raise ValueError(f"snapshot repeats environment path {environment_path}")
        environment_paths.add(environment_path)

        files = entry.get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"{context}.files must be a mapping")
        for filename in COMPLETE_FILES:
            file_contract = files.get(filename)
            if not isinstance(file_contract, Mapping):
                raise ValueError(f"{context}.files.{filename} must be a mapping")
            expected_sha256 = _fingerprint(
                file_contract.get("sha256"),
                context=f"{context}.files.{filename}.sha256",
            )
            actual_sha256 = _sha256_file(environment_path / filename)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{experiment_id}/{filename} SHA-256 mismatch: "
                    f"{actual_sha256} != {expected_sha256}"
                )

        environment = load_environment(
            environment_path, expected_n_seeds=expected_n_seeds
        )
        if environment.experiment_id != experiment_id:
            raise ValueError(
                f"{context}.experiment_id={experiment_id!r} does not match "
                f"loaded environment {environment.experiment_id!r}"
            )
        for filename, rows in (
            ("all.jsonl", environment.all_rows),
            ("train.jsonl", environment.train_rows),
            ("validation.jsonl", environment.validation_rows),
        ):
            expected_rows = _positive_int(
                files[filename].get("rows"),
                name=f"{context}.files.{filename}.rows",
            )
            if len(rows) != expected_rows:
                raise ValueError(
                    f"{experiment_id}/{filename} row count mismatch: "
                    f"{len(rows)} != {expected_rows}"
                )
        environments.append(environment)

    corpus = _combine_environments(path.parent, environments)
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("snapshot.counts must be a mapping")
    actual_counts = {
        "environments": len(corpus.environments),
        "all": len(corpus.all_rows),
        "train": len(corpus.train_rows),
        "validation": len(corpus.validation_rows),
    }
    for name, actual in actual_counts.items():
        expected = _positive_int(counts.get(name), name=f"snapshot.counts.{name}")
        if actual != expected:
            raise ValueError(f"snapshot count {name} mismatch: {actual} != {expected}")
    expected_fingerprints_sha256 = _fingerprint(
        manifest.get("fingerprints_sha256"), context="snapshot.fingerprints_sha256"
    )
    actual_fingerprints_sha256 = _fingerprints_sha256(corpus.all_rows)
    if actual_fingerprints_sha256 != expected_fingerprints_sha256:
        raise ValueError(
            "snapshot fingerprint digest mismatch: "
            f"{actual_fingerprints_sha256} != {expected_fingerprints_sha256}"
        )
    return WideSnapshot(
        path=path,
        sha256=_sha256_file(path),
        manifest=manifest,
        corpus=corpus,
    )


def freeze_snapshot_manifest(
    environment_dirs: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a deterministic manifest for already-exported completed GT."""

    if not environment_dirs:
        raise ValueError("At least one environment directory is required")
    output = Path(output_path).resolve()
    environments = [load_environment(path) for path in environment_dirs]
    corpus = _combine_environments(output.parent, environments)
    entries: list[dict[str, Any]] = []
    for environment in environments:
        relative_path = os.path.relpath(environment.path, output.parent)
        files: dict[str, dict[str, Any]] = {}
        for filename in COMPLETE_FILES:
            contract: dict[str, Any] = {
                "sha256": _sha256_file(environment.path / filename)
            }
            if filename == "all.jsonl":
                contract["rows"] = len(environment.all_rows)
            elif filename == "train.jsonl":
                contract["rows"] = len(environment.train_rows)
            elif filename == "validation.jsonl":
                contract["rows"] = len(environment.validation_rows)
            files[filename] = contract
        entries.append(
            {
                "experiment_id": environment.experiment_id,
                "path": Path(relative_path).as_posix(),
                "files": files,
            }
        )
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "n_seeds": corpus.n_seeds,
        "counts": {
            "environments": len(corpus.environments),
            "all": len(corpus.all_rows),
            "train": len(corpus.train_rows),
            "validation": len(corpus.validation_rows),
        },
        "fingerprints_sha256": _fingerprints_sha256(corpus.all_rows),
        "environments": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_corpus(
    dataset_root: str | Path,
    *,
    expected_environment_count: int | None = None,
    expected_n_seeds: int | None = None,
    require_no_partial: bool = False,
) -> WideCorpus:
    """Load and combine completed environments with cross-environment audits."""

    root = Path(dataset_root).resolve()
    complete, partial = discover_environment_dirs(root)
    if require_no_partial and partial:
        raise ValueError(f"Found {len(partial)} partial environments under {root}")
    if expected_environment_count is not None and len(complete) != expected_environment_count:
        raise ValueError(
            f"Found {len(complete)} completed environments; "
            f"expected {expected_environment_count}"
        )
    if not complete:
        raise ValueError(f"No completed environments found under {root}")
    environments = tuple(
        load_environment(path, expected_n_seeds=expected_n_seeds) for path in complete
    )
    return _combine_environments(root, environments)


def load_seed_losses(environment: WideEnvironment, *, split: str = "validation") -> np.ndarray:
    """Load stored final GT values using the environment's declared seed count."""

    rows = environment.rows(split)
    failed = [
        str(row["example_fingerprint_sha256"])
        for row in rows
        if int(row["target"].get("failed_seeds", 0)) > 0
    ]
    if failed:
        raise ValueError(
            "Per-seed curve loading requires failed_seeds == 0 for complete "
            "seed coverage; found partial coverage for "
            f"{len(failed)} {split} rows in {environment.experiment_id}"
        )
    return load_final_seed_losses(
        rows,
        environment.path.parent,
        expected_n_seeds=environment.n_seeds,
    )


def make_group_folds(
    rows: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    declared_groups: Sequence[str] | None = None,
) -> list[WideGroupFold]:
    """Build exhaustive leave-one-environment/dataset/family-out folds."""

    if not rows:
        raise ValueError("Cannot make group folds from empty rows")
    keys = np.asarray([group_value(row, axis) for row in rows], dtype=object)
    observed = set(str(value) for value in keys)
    groups = tuple(sorted(observed)) if declared_groups is None else tuple(declared_groups)
    if len(groups) != len(set(groups)):
        raise ValueError("declared_groups contains duplicates")
    if set(groups) != observed:
        raise ValueError(
            f"{axis} group mismatch; missing={sorted(set(groups) - observed)}, "
            f"unexpected={sorted(observed - set(groups))}"
        )
    folds: list[WideGroupFold] = []
    test_partitions: list[np.ndarray] = []
    for group in groups:
        test = np.flatnonzero(keys == group).astype(np.int64, copy=False)
        train = np.flatnonzero(keys != group).astype(np.int64, copy=False)
        if test.size == 0 or train.size == 0:
            raise ValueError(f"{axis}={group!r} needs non-empty train and test rows")
        if set(keys[train]).intersection(set(keys[test])):
            raise AssertionError(f"Group leakage in {axis}={group!r}")
        folds.append(WideGroupFold(axis, group, train, test))
        test_partitions.append(test)
    if not np.array_equal(np.sort(np.concatenate(test_partitions)), np.arange(len(rows))):
        raise AssertionError(f"{axis} test folds do not partition the rows")
    return folds


def group_protocol_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe all three group protocols without fitting an estimator."""

    result: dict[str, Any] = {}
    for axis in GROUP_AXES:
        folds = make_group_folds(rows, axis)
        result[axis] = {
            "protocol": f"leave_one_{axis}_out",
            "n_folds": len(folds),
            "groups": [
                {
                    "group": fold.held_out_group,
                    "n_train": int(fold.train_indices.size),
                    "n_test": int(fold.test_indices.size),
                }
                for fold in folds
            ],
        }
    return result


def _prediction_vector(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float] | np.ndarray | Mapping[str, float],
) -> np.ndarray:
    if isinstance(predictions, Mapping):
        missing = [
            str(row["example_fingerprint_sha256"])
            for row in rows
            if str(row["example_fingerprint_sha256"]) not in predictions
        ]
        if missing:
            raise KeyError(f"Missing predictions for {len(missing)} rows")
        values: Any = [predictions[str(row["example_fingerprint_sha256"])] for row in rows]
    else:
        values = predictions
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(rows),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected {len(rows)} finite row-aligned predictions")
    return vector


def _within_environment_scope(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(
        [_finite_float(row["target"]["log_mean_loss"], name="target.log_mean_loss") for row in rows],
        dtype=np.float64,
    )
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(group_value(row, "environment"), []).append(index)

    environments: dict[str, Any] = {}
    n_three_choice = 0
    n_three_choice_correct = 0
    comparable_pairs = 0
    concordance_numerator = 0.0
    spearman_values: list[float] = []
    for environment in sorted(grouped):
        indices = np.asarray(grouped[environment], dtype=np.int64)
        true_group = truth[indices]
        predicted_group = predicted[indices]
        ranking = ranking_metrics(true_group, predicted_group)
        choices = exact_three_choice_accuracy(true_group, predicted_group)
        environments[environment] = {
            "n": int(indices.size),
            "log": regression_metrics(true_group, predicted_group),
            "ranking": ranking,
            "three_choice": choices,
        }
        pairs = int(ranking["n_comparable_pairs"])
        if ranking["pair_concordance"] is not None:
            comparable_pairs += pairs
            concordance_numerator += float(ranking["pair_concordance"]) * pairs
        if ranking["spearman"] is not None:
            spearman_values.append(float(ranking["spearman"]))
        n_three_choice += int(choices["n_groups"])
        n_three_choice_correct += int(choices["n_correct"])

    return {
        "scope": "within_environment_only",
        "n_rows": len(rows),
        "n_environments": len(environments),
        "micro_log_regression": regression_metrics(truth, predicted),
        "within_environment_ranking": {
            "pair_concordance": (
                concordance_numerator / comparable_pairs if comparable_pairs else None
            ),
            "n_comparable_pairs": comparable_pairs,
            "macro_spearman": (
                float(np.mean(spearman_values)) if spearman_values else None
            ),
        },
        "three_choice": {
            "n_groups": n_three_choice,
            "n_correct": n_three_choice_correct,
            "accuracy": (
                n_three_choice_correct / n_three_choice if n_three_choice else None
            ),
            "scope": "all_triples_within_each_environment",
        },
        "environments": environments,
    }


def within_environment_metrics(
    rows: Sequence[Mapping[str, Any]],
    predicted_log_loss: Sequence[float] | np.ndarray | Mapping[str, float],
) -> dict[str, Any]:
    """Score calibration and choice/ranking only among comparable environments.

    ``all`` and ``benchmark_eligible`` use identical same-environment rules.
    The original top-level fields remain aliases of ``all`` for compatibility
    with the first adapter revision.
    """

    if not rows:
        raise ValueError("At least one row is required")
    predicted = _prediction_vector(rows, predicted_log_loss)
    all_metrics = _within_environment_scope(rows, predicted)
    eligible_mask = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    eligible_metrics = None
    if np.any(eligible_mask):
        eligible_metrics = _within_environment_scope(
            [row for row, keep in zip(rows, eligible_mask) if keep],
            predicted[eligible_mask],
        )
    return {
        **all_metrics,
        "all": all_metrics,
        "benchmark_eligible": eligible_metrics,
    }


def _plan_expectations(plan_path: Path) -> dict[str, Any]:
    plan = _load_json_object(plan_path, label="wide plan")
    defaults = plan.get("defaults")
    experiments = plan.get("experiments")
    if not isinstance(defaults, Mapping) or not isinstance(experiments, list):
        raise ValueError("Wide plan must contain defaults and experiments")
    default_rows = _positive_int(defaults.get("num_rows"), name="plan.defaults.num_rows")
    default_train = _positive_int(
        defaults.get("train_rows"), name="plan.defaults.train_rows"
    )
    default_gt = defaults.get("ground_truth")
    if not isinstance(default_gt, Mapping):
        raise ValueError("plan.defaults.ground_truth must be a mapping")
    n_seeds = _positive_int(
        default_gt.get("n_seeds"), name="plan.defaults.ground_truth.n_seeds"
    )
    default_phase = defaults.get("phase")
    if not isinstance(default_phase, str) or not default_phase:
        raise ValueError("plan.defaults.phase must be a non-empty string")
    ids: list[str] = []
    environment_expectations: dict[str, dict[str, Any]] = {}
    total_rows = 0
    train_rows = 0
    for index, item in enumerate(experiments):
        if not isinstance(item, Mapping):
            raise ValueError(f"plan.experiments[{index}] must be an object")
        experiment_id = item.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(f"plan.experiments[{index}] has no experiment_id")
        ids.append(experiment_id)
        item_rows = _positive_int(item.get("num_rows", default_rows), name="num_rows")
        item_train = _positive_int(
            item.get("train_rows", default_train), name="train_rows"
        )
        if item_train >= item_rows:
            raise ValueError(
                f"plan experiment {experiment_id!r} needs train_rows < num_rows"
            )
        budget = _positive_int(
            item.get("budget", defaults.get("budget")),
            name=f"plan.experiments[{index}].budget",
        )
        batch_size = _positive_int(
            item.get("batch_size", defaults.get("batch_size")),
            name=f"plan.experiments[{index}].batch_size",
        )
        phase = item.get("phase", default_phase)
        dataset_path = item.get("dataset_path", defaults.get("dataset_path"))
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"plan.experiments[{index}].phase must be a string")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise ValueError(
                f"plan.experiments[{index}].dataset_path must be a string"
            )
        environment_expectations[experiment_id] = {
            "num_rows": item_rows,
            "train_rows": item_train,
            "validation_rows": item_rows - item_train,
            "budget": budget,
            "batch_size": batch_size,
            "phase": phase,
            "dataset_path": dataset_path,
        }
        total_rows += item_rows
        train_rows += item_train
    if len(ids) != len(set(ids)):
        raise ValueError("Wide plan contains duplicate experiment ids")
    return {
        "experiment_ids": ids,
        "environments": environment_expectations,
        "n_environments": len(ids),
        "n_seeds": n_seeds,
        "all_rows": total_rows,
        "train_rows": train_rows,
        "validation_rows": total_rows - train_rows,
    }


def _environment_plan_mismatches(
    environment: WideEnvironment, expected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    config = environment.manifest["config"]
    actual = {
        "num_rows": len(environment.all_rows),
        "train_rows": len(environment.train_rows),
        "validation_rows": len(environment.validation_rows),
        "budget": config["budget"],
        "batch_size": config["batch_size"],
        "phase": config["phase"],
        "dataset_path": config["dataset_path"],
    }
    return {
        key: {"actual": actual[key], "expected": expected[key]}
        for key in actual
        if actual[key] != expected[key]
    }


def validate_root(
    dataset_root: str | Path,
    *,
    plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a strict but partial-run-safe preflight report."""

    root = Path(dataset_root).resolve()
    complete, partial = discover_environment_dirs(root)
    expectations = (
        _plan_expectations(Path(plan_path).resolve()) if plan_path is not None else None
    )
    expected_n_seeds = expectations["n_seeds"] if expectations else None
    valid: list[WideEnvironment] = []
    errors: dict[str, str] = {}
    environment_mismatches: dict[str, dict[str, dict[str, Any]]] = {}
    for path in complete:
        try:
            environment = load_environment(path, expected_n_seeds=expected_n_seeds)
            if expectations and path.name in expectations["environments"]:
                mismatches = _environment_plan_mismatches(
                    environment, expectations["environments"][path.name]
                )
                if mismatches:
                    environment_mismatches[path.name] = mismatches
                    fields = ", ".join(sorted(mismatches))
                    raise ValueError(f"{path.name} disagrees with plan fields: {fields}")
            valid.append(environment)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors[path.name] = str(exc)

    observed_dirs = {path.name for path in (*complete, *partial)}
    if expectations:
        expected_ids = set(expectations["experiment_ids"])
        missing = sorted(expected_ids - observed_dirs)
        unexpected = sorted(observed_dirs - expected_ids)
    else:
        missing = []
        unexpected = []
    counts = {
        "environments": len(valid),
        "all": sum(len(environment.all_rows) for environment in valid),
        "train": sum(len(environment.train_rows) for environment in valid),
        "validation": sum(len(environment.validation_rows) for environment in valid),
    }
    group_counts = {
        axis: dict(
            sorted(
                Counter(
                    group_value(row, axis)
                    for environment in valid
                    for row in environment.all_rows
                ).items()
            )
        )
        for axis in GROUP_AXES
    }
    count_mismatches: dict[str, dict[str, int]] = {}
    if expectations and not partial and not missing:
        for actual_key, expected_key in (
            ("environments", "n_environments"),
            ("all", "all_rows"),
            ("train", "train_rows"),
            ("validation", "validation_rows"),
        ):
            if counts[actual_key] != expectations[expected_key]:
                count_mismatches[actual_key] = {
                    "actual": counts[actual_key],
                    "expected": expectations[expected_key],
                }
    invalid = bool(errors or unexpected or count_mismatches)
    incomplete = bool(partial or missing)
    status = "invalid" if invalid else "partial" if incomplete else "complete"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_root": str(root),
        "plan_path": str(Path(plan_path).resolve()) if plan_path is not None else None,
        "plan_sha256": (
            _sha256_file(Path(plan_path).resolve()) if plan_path is not None else None
        ),
        "expected": expectations,
        "counts": counts,
        "completed_environments": [environment.experiment_id for environment in valid],
        "partial_environments": [path.name for path in partial],
        "missing_environments": missing,
        "unexpected_environments": unexpected,
        "validation_errors": errors,
        "environment_mismatches": environment_mismatches,
        "count_mismatches": count_mismatches,
        "n_seeds_observed": sorted({environment.n_seeds for environment in valid}),
        "group_counts": group_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate exported manifests/rows")
    validate.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    plan_options = validate.add_mutually_exclusive_group()
    plan_options.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="frozen wide plan used for expected per-environment metadata",
    )
    plan_options.add_argument(
        "--no-plan",
        action="store_true",
        help="validate self-consistency only, without plan completeness checks",
    )
    validate.add_argument(
        "--require-complete",
        action="store_true",
        help="return non-zero while planned environments are partial or missing",
    )
    validate.add_argument(
        "--include-group-protocols",
        action="store_true",
        help="also enumerate leave-one-group-out fold sizes over completed train rows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    plan_path = None if args.no_plan else args.plan
    expected_errors = (FileNotFoundError, KeyError, OSError, TypeError, ValueError)
    group_errors = expected_errors + (AssertionError,)
    try:
        report = validate_root(args.dataset_root, plan_path=plan_path)
    except expected_errors as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "invalid",
            "dataset_root": str(args.dataset_root.resolve()),
            "plan_path": str(plan_path.resolve()) if plan_path is not None else None,
            "validation_errors": {
                "preflight": f"{type(exc).__name__}: {exc}",
            },
        }
    if (
        args.include_group_protocols
        and report["status"] != "invalid"
        and report["counts"]["environments"]
    ):
        try:
            corpus = load_corpus(args.dataset_root)
            report["train_group_protocols"] = group_protocol_manifest(corpus.train_rows)
        except group_errors as exc:
            report["status"] = "invalid"
            error = f"{type(exc).__name__}: {exc}"
            report["group_protocol_error"] = error
            report["validation_errors"]["group_protocols"] = error
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if report["status"] == "invalid":
        return 1
    if args.require_complete and report["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
