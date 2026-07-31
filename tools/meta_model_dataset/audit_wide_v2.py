"""Audit completed (or in-progress) wide-v2 meta-model ground truth.

This is deliberately a read-only, standard-library-only audit of artifacts
already produced by :mod:`tools.meta_model_dataset.build`.  It never imports
generated candidate code, recomputes ground truth, or loads ``curves.npz``.

The report has two layers:

* hard-validity gates establish that a requested phase is terminal, exported,
  and internally consistent;
* capacity gates bound unusable/failed-seed rates and require reserve headroom
  before the next phase is started.  A capacity warning does not invalidate an
  otherwise valid exported dataset.

Both a machine-readable JSON report and a short Markdown report are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MAX_ATTEMPT_UNUSABLE_RATE = 0.05
DEFAULT_MAX_FAILED_SEED_RATE = 0.05
DEFAULT_MAX_RESERVE_CONSUMPTION_RATE = 0.80


@dataclass(frozen=True)
class Thresholds:
    max_attempt_unusable_rate: float
    max_failed_seed_rate: float
    max_reserve_consumption_rate: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, json.JSONDecodeError) as error:
                    errors.append(f"{path}:{line_number}: {error}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"{path}:{line_number}: row is not an object")
                    continue
                rows.append(value)
    except OSError as error:
        errors.append(f"{path}: {error}")
    return rows, errors


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolved(experiment: dict[str, Any], defaults: dict[str, Any], key: str) -> Any:
    value = experiment.get(key)
    return defaults.get(key) if value is None else value


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _target_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    log_values: list[float] = []
    missing_or_nonfinite = 0
    nonpositive = 0
    inconsistent_log = 0
    benchmark_eligible = 0
    total = 0
    for row in rows:
        total += 1
        target = row.get("target")
        if not isinstance(target, dict):
            missing_or_nonfinite += 1
            continue
        raw = target.get("mean_loss")
        if not _is_finite_number(raw):
            missing_or_nonfinite += 1
            continue
        number = float(raw)
        values.append(number)
        if number <= 0:
            nonpositive += 1
        log_value = target.get("log_mean_loss")
        if log_value is not None and _is_finite_number(log_value):
            log_number = float(log_value)
            log_values.append(log_number)
            if number <= 0 or not math.isclose(
                log_number, math.log(number), rel_tol=1e-9, abs_tol=1e-9
            ):
                inconsistent_log += 1
        elif number > 0:
            inconsistent_log += 1
        if bool(target.get("benchmark_eligible")):
            benchmark_eligible += 1

    values.sort()
    log_values.sort()
    return {
        "total": total,
        "finite": len(values),
        "missing_or_nonfinite": missing_or_nonfinite,
        "nonpositive": nonpositive,
        "inconsistent_log": inconsistent_log,
        "benchmark_eligible": benchmark_eligible,
        "benchmark_eligible_rate": _ratio(benchmark_eligible, total),
        "range": [values[0], values[-1]] if values else None,
        "quantiles": {
            "p01": _quantile(values, 0.01),
            "p50": _quantile(values, 0.50),
            "p99": _quantile(values, 0.99),
        },
        "log_range": [log_values[0], log_values[-1]] if log_values else None,
    }


def _attempt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row.get("selection_role") == "primary"]
    reserve = [row for row in rows if row.get("selection_role") == "reserve"]
    unusable = [row for row in rows if not bool(row.get("usable_for_regression"))]
    primary_unusable = [
        row for row in primary if not bool(row.get("usable_for_regression"))
    ]
    reserve_unusable = [
        row for row in reserve if not bool(row.get("usable_for_regression"))
    ]
    failed_attempts = 0
    failed_seeds = 0
    seed_slots = 0
    missing_seed_audit = 0
    marker_or_execution_failures = 0
    for row in rows:
        target = row.get("target")
        provenance = row.get("provenance")
        if not isinstance(target, dict):
            missing_seed_audit += 1
            marker_or_execution_failures += 1
            continue
        n_seeds = target.get("n_seeds")
        seed_failures = target.get("failed_seeds")
        if isinstance(n_seeds, int) and n_seeds >= 0:
            seed_slots += n_seeds
        else:
            missing_seed_audit += 1
        if isinstance(seed_failures, int) and seed_failures >= 0:
            failed_seeds += seed_failures
            failed_attempts += seed_failures > 0
        else:
            missing_seed_audit += 1
        if not isinstance(provenance, dict) or provenance.get("execution") != (
            "candidate_py_files"
        ):
            marker_or_execution_failures += 1
    return {
        "total": len(rows),
        "primary": len(primary),
        "reserve": len(reserve),
        "usable": len(rows) - len(unusable),
        "unusable": len(unusable),
        "unusable_rate": _ratio(len(unusable), len(rows)),
        "primary_unusable": len(primary_unusable),
        "primary_unusable_rate": _ratio(len(primary_unusable), len(primary)),
        "reserve_unusable": len(reserve_unusable),
        "usable_reserve": len(reserve) - len(reserve_unusable),
        "failed_attempts": failed_attempts,
        "failed_attempt_rate": _ratio(failed_attempts, len(rows)),
        "failed_seeds": failed_seeds,
        "seed_slots": seed_slots,
        "failed_seed_rate": _ratio(failed_seeds, seed_slots),
        "missing_seed_audit_fields": missing_seed_audit,
        "marker_or_execution_failures": marker_or_execution_failures,
        "targets": _target_summary(rows),
    }


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _marker_summary(
    experiment_dir: Path,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    statuses: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    gt_configs: Counter[str] = Counter()
    missing = 0
    invalid = 0
    for record in records:
        artifact_dir = record.get("artifact_dir")
        if not isinstance(artifact_dir, str):
            invalid += 1
            errors.append("sampling record has no artifact_dir")
            continue
        marker_path = experiment_dir / artifact_dir / "results" / "meta_model_gt.json"
        if not marker_path.is_file():
            missing += 1
            continue
        try:
            marker = _read_json(marker_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid += 1
            errors.append(f"invalid marker {marker_path}: {error}")
            continue
        status = str(marker.get("status", "missing_status"))
        statuses[status] += 1
        context = marker.get("execution_context_sha256")
        gt_config = marker.get("gt_config_sha256")
        if isinstance(context, str):
            contexts[context] += 1
        else:
            errors.append(f"marker has no execution context: {marker_path}")
        if isinstance(gt_config, str):
            gt_configs[gt_config] += 1
        else:
            errors.append(f"marker has no GT config hash: {marker_path}")
    terminal = statuses.get("ok", 0) + statuses.get("error", 0)
    return (
        {
            "expected": len(records),
            "present": len(records) - missing,
            "missing": missing,
            "invalid": invalid,
            "terminal": terminal,
            "statuses": dict(sorted(statuses.items())),
            "execution_context_sha256": dict(sorted(contexts.items())),
            "gt_config_sha256": dict(sorted(gt_configs.items())),
        },
        errors,
    )


def _validate_selected_rows(
    rows: list[dict[str, Any]],
    expected_group_labels: dict[str, Any],
) -> dict[str, Any]:
    fingerprints = [row.get("example_fingerprint_sha256") for row in rows]
    invalid_execution = 0
    unusable = 0
    group_mismatches = 0
    invalid_parameter_counts = 0
    for row in rows:
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("execution") != (
            "candidate_py_files"
        ):
            invalid_execution += 1
        unusable += not bool(row.get("usable_for_regression"))
        group_mismatches += row.get("group_labels") != expected_group_labels
        derived = row.get("derived")
        if (
            not isinstance(derived, dict)
            or not isinstance(derived.get("total_params"), int)
            or int(derived["total_params"]) <= 0
        ):
            invalid_parameter_counts += 1
    return {
        "unique_fingerprints": len(set(fingerprints)),
        "duplicate_fingerprints": len(fingerprints) - len(set(fingerprints)),
        "invalid_execution": invalid_execution,
        "unusable": unusable,
        "group_label_mismatches": group_mismatches,
        "invalid_parameter_counts": invalid_parameter_counts,
        "targets": _target_summary(rows),
    }


def _expected_experiments(
    plan: dict[str, Any], phase: str
) -> list[dict[str, Any]]:
    defaults = plan.get("defaults", {})
    selected = []
    for experiment in plan.get("experiments", []):
        if _resolved(experiment, defaults, "phase") != phase:
            continue
        num_rows = int(_resolved(experiment, defaults, "num_rows"))
        train_rows = int(_resolved(experiment, defaults, "train_rows"))
        reserve_rows = int(_resolved(experiment, defaults, "reserve_rows"))
        selected.append(
            {
                "experiment_id": str(experiment["experiment_id"]),
                "phase": phase,
                "num_rows": num_rows,
                "train_rows": train_rows,
                "validation_rows": num_rows - train_rows,
                "reserve_rows": reserve_rows,
            }
        )
    return selected


def _audit_experiment(
    expected: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_id = expected["experiment_id"]
    experiment_dir = output_root / experiment_id
    errors: list[str] = []
    warnings: list[str] = []
    sampling_path = experiment_dir / "sampling_manifest.json"
    sampling: dict[str, Any] | None = None
    config: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    if not sampling_path.is_file():
        errors.append("missing sampling_manifest.json")
    else:
        try:
            sampling = _read_json(sampling_path)
            config = sampling.get("config", {})
            raw_records = sampling.get("records", [])
            if isinstance(raw_records, list):
                records = [row for row in raw_records if isinstance(row, dict)]
            else:
                errors.append("sampling records are not a list")
            if sampling.get("config_sha256") != _sha256_json(config):
                errors.append("sampling config SHA-256 mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid sampling manifest: {error}")

    expected_attempts = int(expected["num_rows"]) + int(expected["reserve_rows"])
    if records and len(records) != expected_attempts:
        errors.append(
            f"sampling count {len(records)} != expected {expected_attempts}"
        )
    if config:
        for field in ("phase", "num_rows", "train_rows", "reserve_rows"):
            if config.get(field) != expected[field]:
                errors.append(
                    f"sampling config {field}={config.get(field)!r} "
                    f"!= plan {expected[field]!r}"
                )

    marker_stats, marker_errors = _marker_summary(experiment_dir, records)
    errors.extend(marker_errors)

    file_names = (
        "manifest.json",
        "attempts.jsonl",
        "all.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "feature_schema.json",
    )
    export_files = {
        name: (experiment_dir / name).is_file() for name in file_names
    }
    exported = all(export_files.values())
    manifest: dict[str, Any] | None = None
    if export_files["manifest.json"]:
        try:
            manifest = _read_json(experiment_dir / "manifest.json")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid export manifest: {error}")

    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for file_name in ("attempts.jsonl", "all.jsonl", "train.jsonl", "validation.jsonl"):
        path = experiment_dir / file_name
        if path.is_file():
            rows, row_errors = _read_jsonl(path)
            rows_by_file[file_name] = rows
            errors.extend(row_errors)
        else:
            rows_by_file[file_name] = []

    attempts = rows_by_file["attempts.jsonl"]
    selected = rows_by_file["all.jsonl"]
    train = rows_by_file["train.jsonl"]
    validation = rows_by_file["validation.jsonl"]
    attempt_stats = _attempt_summary(attempts)
    expected_groups = config.get("group_labels", {})
    selected_checks = _validate_selected_rows(selected, expected_groups)

    replacement_count = 0
    if manifest is not None:
        replacement_count = len(
            manifest.get("attempts", {}).get("replacements", [])
        )
    selected_replacements = sum(
        row.get("dataset_role") == "reserve_replacement" for row in selected
    )
    usable_reserve = int(attempt_stats["usable_reserve"])
    reserve_headroom = usable_reserve - replacement_count
    reserve_consumption_rate = _ratio(replacement_count, int(expected["reserve_rows"]))
    usable_reserve_consumption_rate = _ratio(replacement_count, usable_reserve)

    export_counts_exact = (
        len(attempts) == expected_attempts
        and len(selected) == int(expected["num_rows"])
        and len(train) == int(expected["train_rows"])
        and len(validation) == int(expected["validation_rows"])
        and replacement_count == selected_replacements
        and replacement_count == int(attempt_stats["primary_unusable"])
    )
    markers_terminal = (
        marker_stats["terminal"] == expected_attempts
        and marker_stats["missing"] == 0
        and marker_stats["invalid"] == 0
    )

    marker_contexts = marker_stats["execution_context_sha256"]
    marker_gt_configs = marker_stats["gt_config_sha256"]
    manifest_context = None
    manifest_gt_config = None
    context_matches = False
    gt_config_matches = False
    if manifest is not None:
        ground_truth = manifest.get("ground_truth", {})
        manifest_context = ground_truth.get("execution_context_sha256")
        manifest_gt_config = ground_truth.get("config_sha256")
        context_matches = (
            isinstance(manifest_context, str)
            and marker_contexts == {manifest_context: expected_attempts}
        )
        gt_config_matches = (
            isinstance(manifest_gt_config, str)
            and marker_gt_configs == {manifest_gt_config: expected_attempts}
        )
        if sampling is not None and manifest.get("config_sha256") != sampling.get(
            "config_sha256"
        ):
            errors.append("export/sampling config SHA-256 mismatch")

    context_consistent = context_matches and gt_config_matches
    if exported and reserve_headroom == 0:
        warnings.append("all usable reserve rows were consumed")
    if exported and replacement_count > int(expected["reserve_rows"]):
        errors.append("replacement count exceeds predeclared reserve")
    target_range = selected_checks["targets"]["range"]
    if target_range and target_range[0] > 0 and target_range[1] / target_range[0] > 1e12:
        warnings.append("selected loss dynamic range exceeds 1e12; fit log-loss too")

    family = config.get("dataset_spec", {}).get("family")
    dataset_id = config.get("dataset_spec", {}).get("dataset_id")
    result = {
        "experiment_id": experiment_id,
        "phase": expected["phase"],
        "family": family,
        "dataset_id": dataset_id,
        "dataset_cohort": config.get("group_labels", {}).get("dataset_cohort"),
        "budget": config.get("budget"),
        "batch_size": config.get("batch_size"),
        "expected": {
            **expected,
            "attempts": expected_attempts,
        },
        "sampling": {
            "present": sampling is not None,
            "records": len(records),
            "config_sha256": sampling.get("config_sha256") if sampling else None,
        },
        "markers": marker_stats,
        "export": {
            "complete": exported,
            "files": export_files,
            "counts_exact": export_counts_exact,
            "attempts": attempt_stats,
            "selected": {
                "total": len(selected),
                "train": len(train),
                "validation": len(validation),
                **selected_checks,
            },
        },
        "reserve": {
            "planned": int(expected["reserve_rows"]),
            "usable": usable_reserve,
            "replacements": replacement_count,
            "selected_replacements": selected_replacements,
            "headroom_usable_rows": reserve_headroom,
            "consumption_rate_of_planned": reserve_consumption_rate,
            "consumption_rate_of_usable": usable_reserve_consumption_rate,
        },
        "execution_context": {
            "consistent": context_consistent,
            "manifest_execution_context_sha256": manifest_context,
            "manifest_gt_config_sha256": manifest_gt_config,
            "marker_contexts": marker_contexts,
            "marker_gt_configs": marker_gt_configs,
        },
        "terminal_and_exported": markers_terminal and exported,
        "errors": errors,
        "warnings": warnings,
    }
    return result, attempts, selected


def _get_setting_number(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row.get("setting", row.get("spec", {}))
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if _is_finite_number(value) else None


def _planned_setting_number(
    record: dict[str, Any], path: tuple[str, ...]
) -> float | None:
    value: Any = record.get("spec", {})
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if _is_finite_number(value) else None


def _cell_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempt = _attempt_summary(rows)
    target = attempt.pop("targets")
    return {**attempt, "targets": target}


def _extreme_cells(
    experiments: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    axes = {
        "optimizer.lr": ("optimizer", "lr"),
        "optimizer.weight_decay": ("optimizer", "weight_decay"),
        "loss.lambda": ("loss", "lambda"),
    }
    planned_records: list[dict[str, Any]] = []
    planned_family: dict[str, str] = {}
    for experiment in experiments:
        sampling_path = output_root / experiment["experiment_id"] / "sampling_manifest.json"
        if not sampling_path.is_file():
            continue
        try:
            sampling = _read_json(sampling_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        family = sampling.get("config", {}).get("dataset_spec", {}).get("family")
        for record in sampling.get("records", []):
            if isinstance(record, dict):
                planned_records.append(record)
                fingerprint = str(record.get("fingerprint", ""))
                planned_family[fingerprint] = str(family)

    families = sorted(
        {
            str(experiment.get("family"))
            for experiment in experiments
            if experiment.get("family")
        }
    )
    axis_values: dict[str, list[float]] = {}
    for axis, path in axes.items():
        axis_values[axis] = sorted(
            {
                value
                for record in planned_records
                if (value := _planned_setting_number(record, path)) is not None
            }
        )

    cells: list[dict[str, Any]] = []
    cell_definitions: list[tuple[str, list[tuple[str, float]]]] = []
    for axis, values in axis_values.items():
        if not values:
            continue
        cell_definitions.append((f"{axis}=min", [(axis, values[0])]))
        if values[-1] != values[0]:
            cell_definitions.append((f"{axis}=max", [(axis, values[-1])]))
    for left, right in (
        ("optimizer.lr", "optimizer.weight_decay"),
        ("optimizer.lr", "loss.lambda"),
        ("optimizer.weight_decay", "loss.lambda"),
    ):
        if axis_values.get(left) and axis_values.get(right):
            cell_definitions.append(
                (
                    f"{left}=max & {right}=max",
                    [(left, axis_values[left][-1]), (right, axis_values[right][-1])],
                )
            )

    row_by_fingerprint = {
        str(row.get("example_fingerprint_sha256")): row for row in attempts
    }
    for cell_id, predicates in cell_definitions:
        planned_matches: list[dict[str, Any]] = []
        for record in planned_records:
            if all(
                _planned_setting_number(record, axes[axis]) == expected_value
                for axis, expected_value in predicates
            ):
                planned_matches.append(record)
        outcome_rows = [
            row_by_fingerprint[str(record.get("fingerprint"))]
            for record in planned_matches
            if str(record.get("fingerprint")) in row_by_fingerprint
        ]
        family_planned = Counter(
            planned_family.get(str(record.get("fingerprint")), "unknown")
            for record in planned_matches
        )
        family_outcomes: dict[str, Any] = {}
        for family in families:
            family_rows = [row for row in outcome_rows if row.get("family") == family]
            family_outcomes[family] = {
                "planned": family_planned.get(family, 0),
                "observed": len(family_rows),
                "unusable": sum(
                    not bool(row.get("usable_for_regression")) for row in family_rows
                ),
                "unusable_rate": _ratio(
                    sum(
                        not bool(row.get("usable_for_regression"))
                        for row in family_rows
                    ),
                    len(family_rows),
                ),
            }
        cells.append(
            {
                "cell": cell_id,
                "predicates": {axis: value for axis, value in predicates},
                "planned": len(planned_matches),
                "observed": len(outcome_rows),
                "outcomes": _cell_outcomes(outcome_rows),
                "by_family": family_outcomes,
            }
        )
    one_dimensional = [cell for cell in cells if len(cell["predicates"]) == 1]
    coverage_complete = bool(one_dimensional) and all(
        cell["planned"] > 0
        and all(cell["by_family"][family]["planned"] > 0 for family in families)
        for cell in one_dimensional
    )
    return {
        "axis_values": axis_values,
        "families": families,
        "one_dimensional_extremes_covered_in_every_family": coverage_complete,
        "cells": cells,
    }


def _gate(
    name: str,
    passed: bool,
    requirement: str,
    observed: Any,
    *,
    category: str,
    details: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "passed": bool(passed),
        "requirement": requirement,
        "observed": observed,
        "details": details or [],
    }


def audit_wide_v2(
    plan_path: Path,
    *,
    phase: str,
    thresholds: Thresholds,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    output_root = _repo_path(str(plan["output_root"]))
    expected = _expected_experiments(plan, phase)
    if not expected:
        raise ValueError(f"No experiments in phase {phase!r}")

    experiments: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    for expected_experiment in expected:
        experiment, attempts, selected = _audit_experiment(
            expected_experiment, output_root
        )
        experiments.append(experiment)
        all_attempts.extend(attempts)
        all_selected.extend(selected)

    expected_attempts = sum(item["num_rows"] + item["reserve_rows"] for item in expected)
    expected_selected = sum(item["num_rows"] for item in expected)
    expected_train = sum(item["train_rows"] for item in expected)
    expected_validation = sum(item["validation_rows"] for item in expected)
    phase_design = plan.get("design", {}).get("phases", {}).get(phase, {})
    design_selected = phase_design.get("selected_rows")

    attempt_stats = _attempt_summary(all_attempts)
    selected_stats = _target_summary(all_selected)
    extreme = _extreme_cells(experiments, all_attempts, output_root)

    attempt_fingerprints = [
        row.get("example_fingerprint_sha256") for row in all_attempts
    ]
    duplicate_attempt_fingerprints = len(attempt_fingerprints) - len(
        set(attempt_fingerprints)
    )
    environment_coverage = {
        "expected": len(expected),
        "observed_sampling": sum(
            bool(experiment["sampling"]["present"]) for experiment in experiments
        ),
        "terminal_and_exported": sum(
            bool(experiment["terminal_and_exported"]) for experiment in experiments
        ),
        "families": _counter_dict(
            experiment["family"] for experiment in experiments if experiment["family"]
        ),
        "datasets": _counter_dict(
            experiment["dataset_id"]
            for experiment in experiments
            if experiment["dataset_id"]
        ),
        "dataset_cohorts": _counter_dict(
            experiment["dataset_cohort"]
            for experiment in experiments
            if experiment["dataset_cohort"]
        ),
        "budgets": _counter_dict(
            experiment["budget"] for experiment in experiments
            if experiment["budget"] is not None
        ),
        "batch_sizes": _counter_dict(
            experiment["batch_size"]
            for experiment in experiments
            if experiment["batch_size"] is not None
        ),
    }

    phase_complete = all(
        experiment["terminal_and_exported"] for experiment in experiments
    )
    sampling_exact = all(
        experiment["sampling"]["present"]
        and experiment["sampling"]["records"]
        == experiment["expected"]["attempts"]
        for experiment in experiments
    )
    exports_exact = all(
        experiment["export"]["complete"]
        and experiment["export"]["counts_exact"]
        for experiment in experiments
    )
    selected_integrity = all(
        experiment["export"]["selected"]["targets"]["missing_or_nonfinite"] == 0
        and experiment["export"]["selected"]["targets"]["inconsistent_log"] == 0
        and experiment["export"]["selected"]["invalid_execution"] == 0
        and experiment["export"]["selected"]["unusable"] == 0
        and experiment["export"]["selected"]["group_label_mismatches"] == 0
        and experiment["export"]["selected"]["invalid_parameter_counts"] == 0
        for experiment in experiments
    )
    contexts_consistent = all(
        experiment["execution_context"]["consistent"] for experiment in experiments
    )
    no_artifact_errors = not any(experiment["errors"] for experiment in experiments)
    reserve_capacity_valid = all(
        experiment["reserve"]["headroom_usable_rows"] >= 0
        for experiment in experiments
    )

    unusable_rates = {
        experiment["experiment_id"]: experiment["export"]["attempts"][
            "unusable_rate"
        ]
        for experiment in experiments
    }
    failed_seed_rates = {
        experiment["experiment_id"]: experiment["export"]["attempts"][
            "failed_seed_rate"
        ]
        for experiment in experiments
    }
    reserve_rates = {
        experiment["experiment_id"]: experiment["reserve"][
            "consumption_rate_of_planned"
        ]
        for experiment in experiments
    }
    max_unusable = max(
        (value for value in unusable_rates.values() if value is not None),
        default=None,
    )
    max_failed_seed = max(
        (value for value in failed_seed_rates.values() if value is not None),
        default=None,
    )
    max_reserve = max(
        (value for value in reserve_rates.values() if value is not None),
        default=None,
    )
    unusable_rate_ok = (
        phase_complete
        and max_unusable is not None
        and max_unusable <= thresholds.max_attempt_unusable_rate
    )
    failed_seed_rate_ok = (
        phase_complete
        and max_failed_seed is not None
        and max_failed_seed <= thresholds.max_failed_seed_rate
    )
    reserve_headroom_ok = (
        phase_complete
        and max_reserve is not None
        and max_reserve <= thresholds.max_reserve_consumption_rate
    )

    gates = [
        _gate(
            "phase_terminal_and_exported",
            phase_complete,
            "every planned environment has terminal GT markers and all export files",
            environment_coverage,
            category="hard_validity",
            details=[
                experiment["experiment_id"]
                for experiment in experiments
                if not experiment["terminal_and_exported"]
            ],
        ),
        _gate(
            "sampling_and_environment_counts",
            sampling_exact
            and expected_attempts == sum(item["sampling"]["records"] for item in experiments),
            f"exactly {len(expected)} environments and {expected_attempts} frozen attempts",
            {
                "environments": len(experiments),
                "sampling_records": sum(
                    item["sampling"]["records"] for item in experiments
                ),
            },
            category="hard_validity",
        ),
        _gate(
            "export_counts_and_splits",
            exports_exact
            and len(all_attempts) == expected_attempts
            and len(all_selected) == expected_selected,
            (
                f"attempts={expected_attempts}, selected={expected_selected}, "
                f"train={expected_train}, validation={expected_validation}"
            ),
            {
                "attempts": len(all_attempts),
                "selected": len(all_selected),
                "train": sum(
                    item["export"]["selected"]["train"] for item in experiments
                ),
                "validation": sum(
                    item["export"]["selected"]["validation"]
                    for item in experiments
                ),
                "design_selected_rows": design_selected,
            },
            category="hard_validity",
        ),
        _gate(
            "selected_rows_valid",
            phase_complete and selected_integrity,
            "all selected losses/log-losses finite and consistent; rows usable and executed from candidate files",
            selected_stats,
            category="hard_validity",
        ),
        _gate(
            "execution_context_consistent",
            phase_complete and contexts_consistent,
            "one context and GT-config hash per environment, matching its export manifest",
            {
                experiment["experiment_id"]: experiment["execution_context"]
                for experiment in experiments
            },
            category="hard_validity",
        ),
        _gate(
            "fingerprints_unique_and_artifacts_clean",
            duplicate_attempt_fingerprints == 0 and no_artifact_errors,
            "no duplicate attempt fingerprints or artifact/schema errors",
            {
                "duplicate_attempt_fingerprints": duplicate_attempt_fingerprints,
                "artifact_errors": sum(len(item["errors"]) for item in experiments),
            },
            category="hard_validity",
        ),
        _gate(
            "reserve_capacity_sufficient",
            phase_complete and reserve_capacity_valid,
            "every unusable primary has a predeclared usable reserve replacement",
            {
                item["experiment_id"]: item["reserve"] for item in experiments
            },
            category="hard_validity",
        ),
        _gate(
            "attempt_unusable_rate",
            unusable_rate_ok,
            (
                "per-environment unusable attempts <= "
                f"{thresholds.max_attempt_unusable_rate:.1%}"
            ),
            {"maximum": max_unusable, "by_environment": unusable_rates},
            category="capacity",
        ),
        _gate(
            "failed_seed_rate",
            failed_seed_rate_ok,
            (
                "per-environment failed seed slots <= "
                f"{thresholds.max_failed_seed_rate:.1%}"
            ),
            {"maximum": max_failed_seed, "by_environment": failed_seed_rates},
            category="capacity",
        ),
        _gate(
            "reserve_headroom",
            reserve_headroom_ok,
            (
                "per-environment replacements consume <= "
                f"{thresholds.max_reserve_consumption_rate:.1%} of planned reserve"
            ),
            {"maximum": max_reserve, "by_environment": reserve_rates},
            category="capacity",
        ),
        _gate(
            "extreme_cell_coverage",
            extreme["one_dimensional_extremes_covered_in_every_family"],
            "min/max LR, weight decay, and loss lambda are sampled in every family",
            {
                "axis_values": extreme["axis_values"],
                "covered": extreme[
                    "one_dimensional_extremes_covered_in_every_family"
                ],
            },
            category="capacity",
        ),
    ]
    hard_gates = [gate for gate in gates if gate["category"] == "hard_validity"]
    capacity_gates = [gate for gate in gates if gate["category"] == "capacity"]
    hard_pass = all(gate["passed"] for gate in hard_gates)
    capacity_pass = phase_complete and all(gate["passed"] for gate in capacity_gates)
    if not phase_complete:
        overall = "INCOMPLETE"
    elif not hard_pass:
        overall = "HARD_FAIL"
    elif capacity_pass:
        overall = "PASS"
    else:
        overall = "VALID_WITH_CAPACITY_WARNING"

    warnings = [
        {
            "experiment_id": experiment["experiment_id"],
            "message": warning,
        }
        for experiment in experiments
        for warning in experiment["warnings"]
    ]
    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "audit_scope": "stored artifacts only; no GT or generated code execution",
        "plan": _portable(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "output_root": _portable(output_root),
        "phase": phase,
        "thresholds": {
            "max_attempt_unusable_rate": thresholds.max_attempt_unusable_rate,
            "max_failed_seed_rate": thresholds.max_failed_seed_rate,
            "max_reserve_consumption_rate": thresholds.max_reserve_consumption_rate,
        },
        "overall": overall,
        "hard_pass": hard_pass,
        "capacity_pass": capacity_pass,
        # Backward-readable aliases for early local reports.
        "decision": overall,
        "integrity_passed": hard_pass,
        "scale_readiness_passed": hard_pass and capacity_pass,
        "expected": {
            "environments": len(expected),
            "attempts": expected_attempts,
            "selected": expected_selected,
            "train": expected_train,
            "validation": expected_validation,
            "design_selected_rows": design_selected,
        },
        "observed": {
            "environment_coverage": environment_coverage,
            "attempts": attempt_stats,
            "selected": selected_stats,
            "duplicate_attempt_fingerprints": duplicate_attempt_fingerprints,
        },
        "gates": gates,
        "extreme_cells": extreme,
        "warnings": warnings,
        "experiments": experiments,
    }


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def _number(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) >= 1e5 or abs(number) < 1e-3:
        return f"{number:.3e}"
    return f"{number:.5g}"


def _markdown(report: dict[str, Any]) -> str:
    expected = report["expected"]
    observed = report["observed"]
    lines = [
        f"# Wide-v2 {report['phase']} audit",
        "",
        f"Overall: **{report['overall']}**. Hard validity: "
        f"**{'PASS' if report['hard_pass'] else 'FAIL'}**. "
        f"Capacity: **{'PASS' if report['capacity_pass'] else 'WARNING'}**.",
        "",
        f"Plan SHA-256: `{report['plan_sha256']}`",
        "",
        "This audit reads frozen manifests, GT markers, and JSONL exports only. "
        "It does not execute candidate code or recompute ground truth.",
        "",
        "## Gate",
        "",
        "| check | class | result | requirement |",
        "|---|---|---:|---|",
    ]
    for gate in report["gates"]:
        result = "PASS" if gate["passed"] else "FAIL"
        lines.append(
            f"| `{gate['name']}` | {gate['category']} | **{result}** | "
            f"{gate['requirement']} |"
        )
    lines.extend(
        [
            "",
            "Hard-validity failures mean the exported data must not be used. "
            "Capacity failures are risk warnings for the B2 decision and do not "
            "invalidate a hard-valid B1 export. Thresholds are explicit CLI inputs; "
            "the defaults are 5% unusable attempts, 5% failed seed slots, and at "
            "most 80% reserve consumption in every environment.",
            "",
            "## Coverage and totals",
            "",
            "| item | expected | observed |",
            "|---|---:|---:|",
            f"| environments | {expected['environments']} | "
            f"{observed['environment_coverage']['terminal_and_exported']} terminal/exported |",
            f"| attempts | {expected['attempts']} | {observed['attempts']['total']} |",
            f"| selected | {expected['selected']} | {observed['selected']['total']} |",
            f"| train | {expected['train']} | "
            f"{sum(item['export']['selected']['train'] for item in report['experiments'])} |",
            f"| validation | {expected['validation']} | "
            f"{sum(item['export']['selected']['validation'] for item in report['experiments'])} |",
            "",
            "## Environment diagnostics",
            "",
            "| environment | family | attempts | unusable | failed seeds | replacements / reserve | benchmark eligible | selected loss range |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for experiment in report["experiments"]:
        attempts = experiment["export"]["attempts"]
        selected = experiment["export"]["selected"]["targets"]
        reserve = experiment["reserve"]
        target_range = selected["range"]
        rendered_range = (
            "—"
            if target_range is None
            else f"{_number(target_range[0])} .. {_number(target_range[1])}"
        )
        lines.append(
            f"| `{experiment['experiment_id']}` | {experiment['family'] or '—'} | "
            f"{attempts['total']} | {attempts['unusable']} "
            f"({_percent(attempts['unusable_rate'])}) | "
            f"{attempts['failed_seeds']}/{attempts['seed_slots']} "
            f"({_percent(attempts['failed_seed_rate'])}) | "
            f"{reserve['replacements']}/{reserve['planned']} "
            f"({_percent(reserve['consumption_rate_of_planned'])}) | "
            f"{selected['benchmark_eligible']}/{selected['total']} "
            f"({_percent(selected['benchmark_eligible_rate'])}) | {rendered_range} |"
        )

    lines.extend(
        [
            "",
            "## Extreme cells",
            "",
            "| cell | planned | observed | unusable | failed seeds | target range |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for cell in report["extreme_cells"]["cells"]:
        outcome = cell["outcomes"]
        target_range = outcome["targets"]["range"]
        rendered_range = (
            "—"
            if target_range is None
            else f"{_number(target_range[0])} .. {_number(target_range[1])}"
        )
        lines.append(
            f"| `{cell['cell']}` | {cell['planned']} | {cell['observed']} | "
            f"{outcome['unusable']} ({_percent(outcome['unusable_rate'])}) | "
            f"{outcome['failed_seeds']}/{outcome['seed_slots']} "
            f"({_percent(outcome['failed_seed_rate'])}) | {rendered_range} |"
        )

    failed_hard = [
        gate
        for gate in report["gates"]
        if not gate["passed"] and gate["category"] == "hard_validity"
    ]
    failed_capacity = [
        gate
        for gate in report["gates"]
        if not gate["passed"] and gate["category"] == "capacity"
    ]
    if failed_hard:
        lines.extend(["", "## Hard-validity failures", ""])
        for gate in failed_hard:
            lines.append(f"- `{gate['name']}`: {gate['requirement']}")
            for detail in gate["details"]:
                lines.append(f"  - `{detail}`")
    if failed_capacity:
        lines.extend(["", "## Capacity warnings", ""])
        for gate in failed_capacity:
            lines.append(f"- `{gate['name']}`: {gate['requirement']}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(
                f"- `{warning['experiment_id']}`: {warning['message']}"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("tools/meta_model_dataset/plan_wide_v2.json")
    )
    parser.add_argument("--phase", default="b1_pilot")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--max-attempt-unusable-rate",
        type=float,
        default=DEFAULT_MAX_ATTEMPT_UNUSABLE_RATE,
    )
    parser.add_argument(
        "--max-failed-seed-rate",
        type=float,
        default=DEFAULT_MAX_FAILED_SEED_RATE,
    )
    parser.add_argument(
        "--max-reserve-consumption-rate",
        type=float,
        default=DEFAULT_MAX_RESERVE_CONSUMPTION_RATE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rates = (
        args.max_attempt_unusable_rate,
        args.max_failed_seed_rate,
        args.max_reserve_consumption_rate,
    )
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("rate thresholds must be between 0 and 1")
    plan_path = _repo_path(args.plan).resolve()
    report = audit_wide_v2(
        plan_path,
        phase=str(args.phase),
        thresholds=Thresholds(
            max_attempt_unusable_rate=float(args.max_attempt_unusable_rate),
            max_failed_seed_rate=float(args.max_failed_seed_rate),
            max_reserve_consumption_rate=float(args.max_reserve_consumption_rate),
        ),
    )
    plan = _read_json(plan_path)
    output_root = _repo_path(str(plan["output_root"]))
    json_out = args.json_out or output_root / f"audit_{args.phase}.json"
    markdown_out = args.markdown_out or output_root / f"AUDIT_{str(args.phase).upper()}.md"
    _write_json(json_out, report)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "hard_pass": report["hard_pass"],
                "capacity_pass": report["capacity_pass"],
                "phase": report["phase"],
                "expected": report["expected"],
                "observed_attempts": report["observed"]["attempts"]["total"],
                "observed_selected": report["observed"]["selected"]["total"],
                "json": str(json_out),
                "markdown": str(markdown_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["overall"] in {"PASS", "VALID_WITH_CAPACITY_WARNING"}:
        return 0
    if report["overall"] == "INCOMPLETE":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
