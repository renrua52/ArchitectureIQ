"""Pure data helpers for the setting-to-loss meta-model experiment.

The expensive builder lives in :mod:`tools.meta_model_dataset.build`.  This
module keeps fingerprinting, pre-GT splitting, feature extraction, and export
validation small enough to test independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch

from architecture_iq.registry import get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from architecture_iq.significance.validator import final_metric_key, mean_metric_key
from architecture_iq.util import read_json


META_DATASET_SCHEMA_VERSION = "1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_candidate_fingerprint(spec: dict[str, Any]) -> str:
    """Return a collision-resistant identity for the complete candidate setting.

    ``candidate_id`` is deliberately excluded because it is a six-hex display
    ID derived from the rest of the spec.  It is too short for deduplicating a
    1k-scale research dataset.
    """

    body = {key: value for key, value in spec.items() if key != "candidate_id"}
    return sha256_json(body)


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def generated_parameter_counts(
    candidate_dir: Path,
    spec: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Count parameters in the generated ``model.py`` actually used by GT.

    The registry-built module is checked as a second implementation of the same
    model spec.  A mismatch means spec, renderer, and executable code have
    diverged, so exporting a supposedly exact parameter-count feature would be
    unsafe.
    """

    spec = spec or read_json(candidate_dir / "candidate_spec.json")
    with torch.random.fork_rng():
        train_module = load_candidate_train(candidate_dir)
        model_class = getattr(train_module, "Model", None)
        if model_class is None:
            raise RuntimeError(f"Generated train module has no Model: {candidate_dir}")
        generated = model_class()
        if not isinstance(generated, torch.nn.Module):
            raise TypeError(f"Generated Model is not torch.nn.Module: {candidate_dir}")
        generated_counts = _parameter_counts(generated)

        registry_model = get_model_type(spec["model"]["type"]).build_module(spec["model"])
        registry_counts = _parameter_counts(registry_model)

    if generated_counts != registry_counts:
        raise ValueError(
            "Generated/registry parameter-count mismatch for "
            f"{candidate_dir}: generated={generated_counts}, "
            f"registry={registry_counts}"
        )
    return generated_counts


def _flatten_json(value: Any, prefix: str, out: dict[str, int | float | str]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_json(value[key], child, out)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_json(item, f"{prefix}[{index}]", out)
        return
    if isinstance(value, bool):
        out[prefix] = int(value)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value) if isinstance(value, float) else int(value)
        if not math.isfinite(float(number)):
            raise ValueError(f"Non-finite setting feature {prefix}={value!r}")
        out[prefix] = number
        return
    if isinstance(value, str):
        out[prefix] = value
        return
    if value is None:
        return
    raise TypeError(f"Unsupported setting feature {prefix}: {type(value).__name__}")


def setting_features(
    setting: dict[str, Any],
    parameter_counts: dict[str, int],
) -> dict[str, int | float | str]:
    features: dict[str, int | float | str] = {}
    _flatten_json(setting, "", features)
    total_params = int(parameter_counts["total_params"])
    if total_params <= 0:
        raise ValueError(f"Model has no parameters: {total_params}")
    features["derived.total_params"] = total_params
    features["derived.trainable_params"] = int(parameter_counts["trainable_params"])
    features["derived.log_total_params"] = math.log(total_params)
    return dict(sorted(features.items()))


def _optional_summary(candidate_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    summary_path = candidate_dir / "results" / "summary.json"
    if not summary_path.is_file():
        return None, summary_path
    return read_json(summary_path), summary_path


def build_attempt_row(
    *,
    experiment_id: str,
    profile_name: str,
    dataset_spec: dict[str, Any],
    candidate_dir: Path,
    split: str,
    selection_role: str,
    stratum: str,
    relative_to: Path | None = None,
    include_summary: bool = True,
) -> dict[str, Any]:
    """Build one auditable row from a candidate spec and its stored GT."""

    spec_path = candidate_dir / "candidate_spec.json"
    spec = read_json(spec_path)
    fingerprint = full_candidate_fingerprint(spec)
    if spec["dataset_id"] != dataset_spec["dataset_id"]:
        raise ValueError(
            f"Candidate {candidate_dir} belongs to {spec['dataset_id']}, "
            f"expected {dataset_spec['dataset_id']}"
        )
    if spec["family"] != dataset_spec["family"]:
        raise ValueError(
            f"Candidate {candidate_dir} belongs to {spec['family']}, "
            f"expected {dataset_spec['family']}"
        )
    budget = spec["budget"]
    if int(budget["training_steps"]) * int(budget["batch_size"]) != int(
        budget["total_samples_seen"]
    ):
        raise ValueError(f"Invalid training budget in {spec_path}")

    counts = generated_parameter_counts(candidate_dir, spec)
    setting = {
        "model": deepcopy(spec["model"]),
        "optimizer": deepcopy(spec["optimizer"]),
        "loss": deepcopy(spec["loss"]),
        "budget": deepcopy(spec["budget"]),
    }
    features = setting_features(setting, counts)

    summary_path = candidate_dir / "results" / "summary.json"
    summary = None
    if include_summary:
        summary, summary_path = _optional_summary(candidate_dir)
    metric = str(dataset_spec["selection_metric"])
    mean_loss: float | None = None
    std_loss: float | None = None
    log_mean_loss: float | None = None
    failed_seeds: int | None = None
    excluded: bool | None = None
    n_seeds: int | None = None
    execution: str | None = None
    benchmark_eligible: bool | None = None
    summary_sha256: str | None = None

    if summary is not None:
        if summary.get("candidate_id") != spec.get("candidate_id"):
            raise ValueError(f"Candidate/summary ID mismatch in {candidate_dir}")
        if summary.get("selection_metric") != metric:
            raise ValueError(f"Candidate/summary metric mismatch in {candidate_dir}")
        raw_mean = summary.get(mean_metric_key(metric))
        raw_std = summary.get(f"std_{metric}")
        mean_loss = float(raw_mean) if raw_mean is not None else None
        std_loss = float(raw_std) if raw_std is not None else None
        if mean_loss is not None and math.isfinite(mean_loss) and mean_loss > 0:
            log_mean_loss = math.log(mean_loss)
        failed_seeds = int(summary.get("failed_seeds", 0))
        excluded = bool(summary.get("excluded", False))
        n_seeds = int(summary["n_seeds"])
        execution = str(summary.get("execution"))
        summary_sha256 = sha256_file(summary_path)

        fail_threshold = float(
            dataset_spec.get("significance", {}).get("fail_threshold", math.inf)
        )
        final_key = final_metric_key(metric)
        seed_results = summary.get("seed_results", [])
        benchmark_eligible = bool(seed_results) and all(
            not bool(result.get("failed"))
            and result.get(final_key) is not None
            and math.isfinite(float(result[final_key]))
            and float(result[final_key]) <= fail_threshold
            for result in seed_results
        )

    usable = (
        summary is not None
        and execution == "candidate_py_files"
        and failed_seeds == 0
        and mean_loss is not None
        and math.isfinite(mean_loss)
    )

    def portable(path: Path) -> str:
        resolved = path.resolve()
        if relative_to is None:
            return str(resolved)
        try:
            return str(resolved.relative_to(relative_to.resolve()))
        except ValueError:
            return str(resolved)

    return {
        "schema_version": META_DATASET_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "example_fingerprint_sha256": fingerprint,
        "candidate_id_short": spec["candidate_id"],
        "split": split,
        "selection_role": selection_role,
        "stratum": stratum,
        "profile": profile_name,
        "family": spec["family"],
        "dataset_id": spec["dataset_id"],
        "setting": setting,
        "derived": {
            **counts,
            "log_total_params": features["derived.log_total_params"],
        },
        "features": features,
        "target": {
            "selection_metric": metric,
            "mean_loss": mean_loss,
            "log_mean_loss": log_mean_loss,
            "std_loss": std_loss,
            "n_seeds": n_seeds,
            "failed_seeds": failed_seeds,
            "excluded": excluded,
            "benchmark_eligible": benchmark_eligible,
        },
        "usable_for_regression": usable,
        "provenance": {
            "candidate_path": portable(candidate_dir),
            "candidate_spec_path": portable(spec_path),
            "candidate_spec_sha256": sha256_file(spec_path),
            "summary_path": portable(summary_path),
            "summary_sha256": summary_sha256,
            "execution": execution,
            "parameter_count_method": "generated_train_module.Model.parameters",
            "parameter_count_cross_check": "registry_model_family.build_module",
        },
    }


def _nested_get(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing stratification field {dotted_path!r}")
        current = current[part]
    if isinstance(current, (dict, list)):
        return _canonical_json(current)
    return current


def split_stratum(spec: dict[str, Any], stratify_by: Iterable[str]) -> str:
    values = [f"{path}={_nested_get(spec, path)}" for path in stratify_by]
    return "|".join(values) if values else "all"


def _stable_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['fingerprint']}".encode("utf-8")
        ).hexdigest(),
    )


def _validation_quotas(
    groups: dict[str, list[dict[str, Any]]],
    validation_count: int,
) -> dict[str, int]:
    total = sum(len(group) for group in groups.values())
    if not 0 <= validation_count <= total:
        raise ValueError(
            f"validation_count must be in [0, {total}], got {validation_count}"
        )
    if total == 0:
        return {}

    ideals = {
        key: validation_count * len(group) / total for key, group in groups.items()
    }
    quotas = {key: int(math.floor(value)) for key, value in ideals.items()}
    remaining = validation_count - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (ideals[key] - quotas[key], len(groups[key]), key),
        reverse=True,
    )
    for key in priority:
        if remaining <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Could not apportion {validation_count} validation rows")
    return quotas


def _assign_grouped_split(
    records: list[dict[str, Any]],
    *,
    validation_count: int,
    seed: int,
    selection_role: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record)
    quotas = _validation_quotas(groups, validation_count)

    assigned: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        ordered = _stable_order(groups[stratum], seed)
        n_validation = quotas[stratum]
        validation_fingerprints = {
            record["fingerprint"] for record in ordered[:n_validation]
        }
        for record in groups[stratum]:
            row = deepcopy(record)
            row["split"] = (
                "validation"
                if row["fingerprint"] in validation_fingerprints
                else "train"
            )
            row["selection_role"] = selection_role
            assigned.append(row)
    return assigned


def assign_pre_execution_splits(
    records: list[dict[str, Any]],
    *,
    num_rows: int,
    train_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Assign train/validation and reserve roles before any GT is observed."""

    if not 0 < train_rows < num_rows:
        raise ValueError("train_rows must be between zero and num_rows")
    if len(records) < num_rows:
        raise ValueError(f"Need at least {num_rows} sampled records, got {len(records)}")
    primary = records[:num_rows]
    reserve = records[num_rows:]
    validation_rows = num_rows - train_rows
    assigned = _assign_grouped_split(
        primary,
        validation_count=validation_rows,
        seed=seed,
        selection_role="primary",
    )
    if reserve:
        reserve_validation = round(len(reserve) * validation_rows / num_rows)
        assigned.extend(
            _assign_grouped_split(
                reserve,
                validation_count=reserve_validation,
                seed=seed + 1,
                selection_role="reserve",
            )
        )
    return assigned


def select_usable_rows(
    attempt_rows: list[dict[str, Any]],
    *,
    train_rows: int,
    validation_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Select exact split sizes, replacing unusable primaries transparently."""

    selected: list[dict[str, Any]] = []
    replacements: list[dict[str, str]] = []
    used: set[str] = set()
    reserves = [
        row
        for row in attempt_rows
        if row["selection_role"] == "reserve" and row["usable_for_regression"]
    ]

    for primary in [row for row in attempt_rows if row["selection_role"] == "primary"]:
        if primary["usable_for_regression"]:
            chosen = deepcopy(primary)
            chosen["dataset_role"] = "primary"
            selected.append(chosen)
            used.add(chosen["example_fingerprint_sha256"])
            continue

        matching = [
            row
            for row in reserves
            if row["split"] == primary["split"]
            and row["stratum"] == primary["stratum"]
            and row["example_fingerprint_sha256"] not in used
        ]
        if not matching:
            matching = [
                row
                for row in reserves
                if row["split"] == primary["split"]
                and row["example_fingerprint_sha256"] not in used
            ]
        if not matching:
            raise RuntimeError(
                "Not enough usable reserve settings to replace "
                f"{primary['example_fingerprint_sha256']} in {primary['split']}"
            )
        replacement = deepcopy(matching[0])
        replacement["dataset_role"] = "reserve_replacement"
        replacement["replaces_fingerprint_sha256"] = primary[
            "example_fingerprint_sha256"
        ]
        selected.append(replacement)
        used.add(replacement["example_fingerprint_sha256"])
        replacements.append(
            {
                "split": primary["split"],
                "replaced": primary["example_fingerprint_sha256"],
                "replacement": replacement["example_fingerprint_sha256"],
            }
        )

    actual = {
        "train": sum(row["split"] == "train" for row in selected),
        "validation": sum(row["split"] == "validation" for row in selected),
    }
    expected = {"train": train_rows, "validation": validation_rows}
    if actual != expected:
        raise RuntimeError(f"Selected split counts {actual} do not match {expected}")
    fingerprints = [row["example_fingerprint_sha256"] for row in selected]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("Duplicate full fingerprints in selected dataset")
    return selected, replacements


def build_feature_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe raw model inputs without inspecting targets."""

    observed: dict[str, list[int | float | str]] = defaultdict(list)
    for row in rows:
        for name, value in row["features"].items():
            observed[name].append(value)

    fields = []
    for name in sorted(observed):
        values = observed[name]
        if all(isinstance(value, (int, float)) for value in values):
            kind = "numeric"
            field: dict[str, Any] = {
                "name": name,
                "kind": kind,
                "observed_count": len(values),
                "missing_fill": 0.0,
                "add_presence_indicator": len(values) != len(rows),
            }
        elif all(isinstance(value, str) for value in values):
            field = {
                "name": name,
                "kind": "categorical",
                "categories": sorted(set(values)),
                "observed_count": len(values),
                "missing_category": "__MISSING__",
                "unknown_category": "__UNKNOWN__",
            }
        else:
            raise TypeError(f"Mixed feature types for {name}: {values[:3]}")
        fields.append(field)

    return {
        "schema_version": META_DATASET_SCHEMA_VERSION,
        "feature_source": "setting_plus_generated_parameter_count",
        "target_columns_are_features": False,
        "fields": fields,
        "target": {
            "raw": "target.mean_loss",
            "log": "target.log_mean_loss",
            "lower_is_better": True,
        },
    }
