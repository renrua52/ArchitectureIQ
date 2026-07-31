"""Deterministically merge a frozen supplemental reserve into one failed B2 export.

This module never trains a model.  It reconstructs both attempt tables using
the unchanged builder's execution-context and GT-marker checks, requires the
base-only canonical selector to fail with the pre-registered exact capacity
error, replays the sidecar's ordinary 50-row export, and then invokes the same
unchanged selector on base attempts followed by tier-2 reserve rows.

Only ``usable_for_regression`` is used as a label-dependent missingness gate.
No loss value, benchmark eligibility flag, or target rank participates in
activation or replacement choice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from architecture_iq.registry import ensure_registries
from tools.meta_model_dataset import build as standard_builder
from tools.meta_model_dataset.core import (
    build_attempt_row,
    build_feature_schema,
    select_usable_rows,
)
from tools.meta_model_dataset.supplemental_reserve_common import (
    BASE_PLAN_PATH,
    POLICY_PATH,
    SUPPLEMENTAL_PLAN_PATH,
    atomic_write_json,
    atomic_write_jsonl,
    portable,
    read_json,
    read_jsonl,
    repo_path,
    sha256_file,
    sha256_json,
    utc_now,
)


class RescueNotRequired(RuntimeError):
    """Raised when the unchanged base exporter has sufficient capacity."""


def _find_experiment(plan: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [
        experiment
        for experiment in plan.get("experiments", [])
        if str(experiment.get("experiment_id")) == experiment_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one experiment {experiment_id!r}, found {len(matches)}"
        )
    return matches[0]


def _load_context(
    plan_path: Path,
    experiment_id: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    plan, profile, output_root = standard_builder.load_plan(plan_path)
    experiment = _find_experiment(plan, experiment_id)
    dataset_path = repo_path(experiment["dataset_path"])
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    config = standard_builder._experiment_config(  # noqa: SLF001
        plan, experiment, profile, dataset_spec
    )
    standard_builder._validate_config(config, profile)  # noqa: SLF001
    sampling_path = output_root / experiment_id / "sampling_manifest.json"
    if not sampling_path.is_file():
        raise FileNotFoundError(f"Missing sampling manifest: {sampling_path}")
    sampling_manifest = read_json(sampling_path)
    if sampling_manifest.get("config_sha256") != sha256_json(config):
        raise ValueError(f"Config hash mismatch for {sampling_path}")
    if sampling_manifest.get("config") != config:
        raise ValueError(f"Stored config differs from resolved plan: {sampling_path}")
    return config, output_root / experiment_id, sampling_manifest, plan


def _context_projection(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "profile",
            "profile_config",
            "dataset_path",
            "dataset_spec",
            "budget",
            "batch_size",
            "vary",
            "fixed",
            "stratify_by",
            "ground_truth",
            "group_labels",
        )
    }


def _rebuild_attempts(
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    sampling_manifest: dict[str, Any],
    provenance_relative_to: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    gt_hash = standard_builder._gt_config_hash(config)  # noqa: SLF001
    dataset_path = repo_path(config["dataset_path"])
    execution_context, context_hash = standard_builder._execution_context(  # noqa: SLF001
        config, dataset_path
    )
    attempts: list[dict[str, Any]] = []
    marker_states: list[str] = []
    for record in sampling_manifest["records"]:
        candidate_dir = experiment_dir / record["artifact_dir"]
        inputs_hash = standard_builder._candidate_execution_inputs_sha256(  # noqa: SLF001
            candidate_dir, context_hash
        )
        marker_state = standard_builder._gt_marker_state(  # noqa: SLF001
            candidate_dir,
            config=config,
            gt_config_sha256=gt_hash,
            execution_context_sha256=context_hash,
            execution_inputs_sha256=inputs_hash,
        )
        if marker_state is None:
            raise RuntimeError(
                f"Unverified or stale GT for {candidate_dir}; finish standard GT first"
            )
        marker_states.append(marker_state)
        row = build_attempt_row(
            experiment_id=config["experiment_id"],
            profile_name=config["profile"],
            dataset_spec=config["dataset_spec"],
            candidate_dir=candidate_dir,
            split=record["split"],
            selection_role=record["selection_role"],
            stratum=record["stratum"],
            relative_to=provenance_relative_to,
            include_summary=marker_state == "ok",
        )
        row["sampling_index"] = int(record["sampling_index"])
        if "group_labels" in record:
            row["group_labels"] = deepcopy(record["group_labels"])
        attempts.append(row)
    marker_state_sha256 = sha256_json(marker_states)
    return attempts, execution_context, context_hash, marker_state_sha256


def _selection_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fingerprint": row["example_fingerprint_sha256"],
            "split": row["split"],
            "stratum": row["stratum"],
            "dataset_role": row.get("dataset_role"),
            "replaces": row.get("replaces_fingerprint_sha256"),
        }
        for row in rows
    ]


def _require_standard_sidecar_export(
    *,
    attempts: list[dict[str, Any]],
    config: dict[str, Any],
    experiment_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    manifest_path = experiment_dir / "manifest.json"
    all_path = experiment_dir / "all.jsonl"
    if not manifest_path.is_file() or not all_path.is_file():
        raise FileNotFoundError(
            f"Sidecar standard export is incomplete for {experiment_dir}"
        )
    manifest = read_json(manifest_path)
    selected, replacements = select_usable_rows(
        attempts,
        train_rows=int(config["train_rows"]),
        validation_rows=int(config["num_rows"] - config["train_rows"]),
    )
    selected.sort(key=lambda row: int(row["sampling_index"]))
    stored = read_jsonl(all_path)
    if _selection_signature(stored) != _selection_signature(selected):
        raise ValueError(
            f"Sidecar standard all.jsonl does not match deterministic replay: {all_path}"
        )
    if len(selected) != 50:
        raise ValueError(f"Sidecar selected {len(selected)} rows instead of 50")
    if sum(row["split"] == "train" for row in selected) != 45:
        raise ValueError("Sidecar selected train split is not 45")
    if sum(row["split"] == "validation" for row in selected) != 5:
        raise ValueError("Sidecar selected validation split is not 5")
    if not all(row["usable_for_regression"] for row in selected):
        raise ValueError("Sidecar standard selection contains unusable rows")
    return selected, replacements, manifest


def _activation_failure(
    attempts: list[dict[str, Any]],
    *,
    train_rows: int,
    validation_rows: int,
    message_prefix: str,
) -> str:
    try:
        select_usable_rows(
            attempts,
            train_rows=train_rows,
            validation_rows=validation_rows,
        )
    except RuntimeError as error:
        message = str(error)
        if not message.startswith(message_prefix):
            raise RuntimeError(
                f"Base canonical selection failed for a non-registered reason: {message}"
            ) from error
        return message
    raise RescueNotRequired(
        "Base-only canonical selection succeeds; supplemental reserve activation is forbidden"
    )


def _tier2_reserves(
    sidecar_selected: list[dict[str, Any]],
    *,
    base_attempt_count: int,
) -> list[dict[str, Any]]:
    reserves: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sidecar_selected):
        row = deepcopy(source)
        local_origin = {
            "selection_role": source.get("selection_role"),
            "dataset_role": source.get("dataset_role"),
            "replaces_fingerprint_sha256": source.get(
                "replaces_fingerprint_sha256"
            ),
            "sampling_index": int(source["sampling_index"]),
        }
        row.pop("dataset_role", None)
        row.pop("replaces_fingerprint_sha256", None)
        row["selection_role"] = "reserve"
        row["reserve_tier"] = "supplemental_v1"
        row["supplemental_origin"] = local_origin
        row["source_sampling_index"] = int(source["sampling_index"])
        row["sampling_index"] = base_attempt_count + ordinal
        reserves.append(row)
    return reserves


def _enrich_replacements(
    replacements: list[dict[str, str]],
    combined_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_fingerprint = {
        row["example_fingerprint_sha256"]: row for row in combined_attempts
    }
    enriched: list[dict[str, Any]] = []
    for replacement in replacements:
        primary = by_fingerprint[replacement["replaced"]]
        donor = by_fingerprint[replacement["replacement"]]
        if primary["split"] != donor["split"]:
            raise RuntimeError("Replacement crossed the frozen train/validation split")
        source_tier = (
            "supplemental_v1"
            if donor.get("reserve_tier") == "supplemental_v1"
            else "base"
        )
        enriched.append(
            {
                **replacement,
                "source_tier": source_tier,
                "stratum_match": primary["stratum"] == donor["stratum"],
                "replacement_route": (
                    "same_split_and_stratum"
                    if primary["stratum"] == donor["stratum"]
                    else "same_split"
                ),
            }
        )
    return enriched


def _validate_final_selection(
    *,
    base_attempts: list[dict[str, Any]],
    combined_attempts: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
    train_rows: int,
    validation_rows: int,
) -> None:
    if len(selected) != train_rows + validation_rows:
        raise RuntimeError("Merged selected total is incorrect")
    if sum(row["split"] == "train" for row in selected) != train_rows:
        raise RuntimeError("Merged train count is incorrect")
    if sum(row["split"] == "validation" for row in selected) != validation_rows:
        raise RuntimeError("Merged validation count is incorrect")
    fingerprints = [row["example_fingerprint_sha256"] for row in selected]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("Merged selection has duplicate fingerprints")
    if not all(row["usable_for_regression"] for row in selected):
        raise RuntimeError("Merged selection contains unusable rows")

    usable_base_primary = {
        row["example_fingerprint_sha256"]
        for row in base_attempts
        if row["selection_role"] == "primary" and row["usable_for_regression"]
    }
    selected_fingerprints = set(fingerprints)
    if not usable_base_primary <= selected_fingerprints:
        raise RuntimeError("A usable base primary was not preserved")
    unusable_base_primary = [
        row
        for row in base_attempts
        if row["selection_role"] == "primary" and not row["usable_for_regression"]
    ]
    if len(replacements) != len(unusable_base_primary):
        raise RuntimeError("Each unusable base primary must have exactly one replacement")
    replaced = {replacement["replaced"] for replacement in replacements}
    if replaced != {
        row["example_fingerprint_sha256"] for row in unusable_base_primary
    }:
        raise RuntimeError("Replacement mapping does not cover unusable base primaries")

    by_fingerprint = {
        row["example_fingerprint_sha256"]: row for row in combined_attempts
    }
    for replacement in replacements:
        primary = by_fingerprint[replacement["replaced"]]
        donor = by_fingerprint[replacement["replacement"]]
        if primary["split"] != donor["split"]:
            raise RuntimeError("Replacement crossed split")


def _validate_replacement_priority(
    *,
    combined_attempts: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> None:
    """Replay donor availability and prove no same-stratum option was skipped."""

    replacement_by_primary = {
        replacement["replaced"]: replacement for replacement in replacements
    }
    reserves = [
        row
        for row in combined_attempts
        if row["selection_role"] == "reserve" and row["usable_for_regression"]
    ]
    used: set[str] = set()
    primaries = [
        row for row in combined_attempts if row["selection_role"] == "primary"
    ]
    for primary in primaries:
        primary_fingerprint = primary["example_fingerprint_sha256"]
        if primary["usable_for_regression"]:
            used.add(primary_fingerprint)
            continue
        replacement = replacement_by_primary.get(primary_fingerprint)
        if replacement is None:
            raise RuntimeError("Missing replacement during priority replay")
        exact = [
            row
            for row in reserves
            if row["split"] == primary["split"]
            and row["stratum"] == primary["stratum"]
            and row["example_fingerprint_sha256"] not in used
        ]
        fallback = [
            row
            for row in reserves
            if row["split"] == primary["split"]
            and row["example_fingerprint_sha256"] not in used
        ]
        expected = exact[0] if exact else (fallback[0] if fallback else None)
        if expected is None:
            raise RuntimeError("No donor available during priority replay")
        if replacement["replacement"] != expected["example_fingerprint_sha256"]:
            raise RuntimeError(
                "Replacement skipped the first available same-stratum/same-split donor"
            )
        if exact and replacement["replacement_route"] != "same_split_and_stratum":
            raise RuntimeError("Same-stratum donor existed but fallback was recorded")
        used.add(replacement["replacement"])


def build_merge_payload(
    *,
    experiment_id: str,
    contract_path: Path,
) -> dict[str, Any]:
    ensure_registries()
    policy = read_json(POLICY_PATH)
    contract = read_json(contract_path)
    if contract.get("contract_id") != policy.get("contract_id"):
        raise ValueError("Freeze contract ID mismatch")
    if contract.get("policy_sha256") != sha256_json(policy):
        raise ValueError("Freeze policy hash mismatch")
    source_hashes = (
        ("base_plan", BASE_PLAN_PATH),
        ("supplemental_plan", SUPPLEMENTAL_PLAN_PATH),
        ("policy_source", POLICY_PATH),
    )
    for field, path in source_hashes:
        if contract.get(field, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Frozen source changed: {field}")

    frozen_environment = (
        contract.get("timing_snapshot", {})
        .get("before_prepare", {})
        .get("environments", {})
        .get(experiment_id)
    )
    if not isinstance(frozen_environment, dict):
        raise KeyError(f"Experiment is not frozen: {experiment_id}")
    if frozen_environment.get("export_exists"):
        raise RescueNotRequired(
            f"{experiment_id} was already successfully exported at freeze time"
        )

    base_config, base_dir, base_sampling, _base_plan = _load_context(
        BASE_PLAN_PATH, experiment_id
    )
    side_config, side_dir, side_sampling, _side_plan = _load_context(
        SUPPLEMENTAL_PLAN_PATH, experiment_id
    )
    if (base_dir / "manifest.json").is_file():
        raise RescueNotRequired(
            f"{experiment_id} now has a successful ordinary base export"
        )
    if _context_projection(base_config) != _context_projection(side_config):
        raise ValueError("Base and sidecar execution contexts differ")

    frozen = contract.get("experiments", {}).get(experiment_id, {})
    if frozen.get("base_sampling_manifest", {}).get("sha256") != sha256_file(
        base_dir / "sampling_manifest.json"
    ):
        raise ValueError("Base sampling manifest changed after freeze")
    if frozen.get("supplemental_sampling_manifest", {}).get(
        "sha256"
    ) != sha256_file(side_dir / "sampling_manifest.json"):
        raise ValueError("Supplemental sampling manifest changed after freeze")

    merged_root = repo_path(policy["merged_output_root"])
    base_attempts, base_execution_context, base_context_hash, base_states_hash = (
        _rebuild_attempts(
            config=base_config,
            experiment_dir=base_dir,
            sampling_manifest=base_sampling,
            provenance_relative_to=merged_root,
        )
    )
    side_attempts, side_execution_context, side_context_hash, side_states_hash = (
        _rebuild_attempts(
            config=side_config,
            experiment_dir=side_dir,
            sampling_manifest=side_sampling,
            provenance_relative_to=merged_root,
        )
    )
    if base_context_hash != side_context_hash:
        raise ValueError("Base and sidecar execution-context hashes differ")
    if standard_builder._gt_config_hash(base_config) != (  # noqa: SLF001
        standard_builder._gt_config_hash(side_config)  # noqa: SLF001
    ):
        raise ValueError("Base and sidecar GT-config hashes differ")

    train_rows = int(base_config["train_rows"])
    validation_rows = int(base_config["num_rows"] - train_rows)
    activation_error = _activation_failure(
        base_attempts,
        train_rows=train_rows,
        validation_rows=validation_rows,
        message_prefix=policy["activation"]["message_prefix"],
    )
    side_selected, side_replacements, side_manifest = (
        _require_standard_sidecar_export(
            attempts=side_attempts,
            config=side_config,
            experiment_dir=side_dir,
        )
    )

    tier2 = _tier2_reserves(
        side_selected,
        base_attempt_count=len(base_attempts),
    )
    combined_attempts = [*base_attempts, *tier2]
    selected, raw_replacements = select_usable_rows(
        combined_attempts,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    enriched_replacements = _enrich_replacements(
        raw_replacements, combined_attempts
    )
    _validate_final_selection(
        base_attempts=base_attempts,
        combined_attempts=combined_attempts,
        selected=selected,
        replacements=enriched_replacements,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    _validate_replacement_priority(
        combined_attempts=combined_attempts,
        replacements=enriched_replacements,
    )
    selected.sort(key=lambda row: int(row["sampling_index"]))
    train = [row for row in selected if row["split"] == "train"]
    validation = [row for row in selected if row["split"] == "validation"]
    feature_schema = build_feature_schema(train)
    feature_schema["fit_split"] = "train"

    raw_targets = [float(row["target"]["mean_loss"]) for row in selected]
    log_targets = [
        float(row["target"]["log_mean_loss"])
        for row in selected
        if row["target"]["log_mean_loss"] is not None
    ]
    if not raw_targets or not all(math.isfinite(value) for value in raw_targets):
        raise RuntimeError("Merged selection contains a non-finite target")
    short_id_groups: dict[str, set[str]] = defaultdict(set)
    for row in combined_attempts:
        short_id_groups[row["candidate_id_short"]].add(
            row["example_fingerprint_sha256"]
        )
    short_collisions = {
        key: sorted(values)
        for key, values in short_id_groups.items()
        if len(values) > 1
    }
    supplemental_used = sum(
        replacement["source_tier"] == "supplemental_v1"
        for replacement in enriched_replacements
    )
    manifest = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "config_sha256": base_sampling["config_sha256"],
        "config": base_config,
        "created_at": base_sampling["created_at"],
        "exported_at": utc_now(),
        "unit": "one independently sampled and executed candidate setting",
        "split_policy": {
            "assigned_before_ground_truth": True,
            "train": len(train),
            "validation": len(validation),
            "validation_role": "locked_holdout; tune or cross-validate only within train",
            "stratify_by": base_config["stratify_by"],
            "full_sha256_identity": True,
            "group_labels_frozen_before_ground_truth": True,
            "group_labels": base_config["group_labels"],
            "cross_split_replacement_allowed": False,
        },
        "external_evaluation": {
            **base_config["external_evaluation"],
            "overlap_with_selected": sum(
                row["example_fingerprint_sha256"]
                in set(
                    base_config["external_evaluation"][
                        "excluded_fingerprints_sha256"
                    ]
                )
                for row in selected
            ),
        },
        "ground_truth": {
            **base_config["ground_truth"],
            "config_sha256": standard_builder._gt_config_hash(  # noqa: SLF001
                base_config
            ),
            "verified_successful_attempts": sum(
                row["provenance"]["execution"] == "candidate_py_files"
                for row in combined_attempts
            ),
            "total_attempts_in_merged_pool": len(combined_attempts),
            "base_execution_context": base_execution_context,
            "supplemental_execution_context": side_execution_context,
            "execution_context_sha256": base_context_hash,
            "canonical_path": "write_candidate -> run_ground_truth -> results/summary.json",
        },
        "attempts": {
            "total_in_merged_pool": len(combined_attempts),
            "usable_in_merged_pool": sum(
                row["usable_for_regression"] for row in combined_attempts
            ),
            "unusable_in_merged_pool": sum(
                not row["usable_for_regression"] for row in combined_attempts
            ),
            "base_total": len(base_attempts),
            "base_primary": sum(
                row["selection_role"] == "primary" for row in base_attempts
            ),
            "base_reserve": sum(
                row["selection_role"] == "reserve" for row in base_attempts
            ),
            "supplemental_pool": len(tier2),
            "supplemental_generation_attempts": len(side_attempts),
            "supplemental_internal_reserve": len(side_attempts) - len(tier2),
            "supplemental_used": supplemental_used,
            "replacements": enriched_replacements,
        },
        "selected": {
            "total": len(selected),
            "benchmark_eligible": sum(
                bool(row["target"]["benchmark_eligible"]) for row in selected
            ),
            "target_range": [min(raw_targets), max(raw_targets)],
            "log_target_range": (
                [min(log_targets), max(log_targets)] if log_targets else None
            ),
        },
        "parameter_count": {
            "feature": "derived.total_params",
            "recommended_scaled_feature": "derived.log_total_params",
            "method": "instantiate Model imported by generated train.py and sum parameters",
            "cross_check": "registry ModelFamily.build_module must match exactly",
            "range": [
                min(int(row["derived"]["total_params"]) for row in selected),
                max(int(row["derived"]["total_params"]) for row in selected),
            ],
        },
        "short_candidate_id_collisions": short_collisions,
        "supplemental_reserve": {
            "contract_id": policy["contract_id"],
            "freeze_contract_sha256": sha256_file(contract_path),
            "base_plan_sha256": sha256_file(BASE_PLAN_PATH),
            "supplemental_plan_sha256": sha256_file(SUPPLEMENTAL_PLAN_PATH),
            "base_sampling_manifest_sha256": sha256_file(
                base_dir / "sampling_manifest.json"
            ),
            "supplemental_sampling_manifest_sha256": sha256_file(
                side_dir / "sampling_manifest.json"
            ),
            "supplemental_standard_manifest_sha256": sha256_file(
                side_dir / "manifest.json"
            ),
            "base_marker_state_vector_sha256": base_states_hash,
            "supplemental_marker_state_vector_sha256": side_states_hash,
            "generation_attempts": len(side_attempts),
            "usable_pool": len(side_selected),
            "train_pool": sum(row["split"] == "train" for row in side_selected),
            "validation_pool": sum(
                row["split"] == "validation" for row in side_selected
            ),
            "internal_replacements": side_replacements,
            "activation_error": activation_error,
            "activation": "base-only canonical select_usable_rows exact capacity failure",
            "permitted_label_gate": "usable_for_regression only",
            "replacement_policy": policy["selection"]["replacement_order"],
            "cross_split_allowed": False,
            "loss_value_or_rank_used": False,
        },
        "files": {
            "attempts": "attempts.jsonl",
            "all": "all.jsonl",
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "feature_schema": "feature_schema.json",
            "base_sampling_manifest": portable(
                base_dir / "sampling_manifest.json"
            ),
            "supplemental_sampling_manifest": portable(
                side_dir / "sampling_manifest.json"
            ),
            "supplemental_full_attempts": portable(side_dir / "attempts.jsonl"),
            "supplemental_standard_manifest": portable(
                side_dir / "manifest.json"
            ),
        },
    }
    return {
        "experiment_id": experiment_id,
        "output_dir": merged_root / experiment_id,
        "attempts": combined_attempts,
        "all": selected,
        "train": train,
        "validation": validation,
        "feature_schema": feature_schema,
        "manifest": manifest,
    }


def merge_experiment(
    *,
    experiment_id: str,
    contract_path: Path,
) -> Path:
    payload = build_merge_payload(
        experiment_id=experiment_id,
        contract_path=contract_path,
    )
    output_dir: Path = payload["output_dir"]
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Merged output already exists: {output_dir}; audit it instead of overwriting"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_dir / "attempts.jsonl", payload["attempts"])
    atomic_write_jsonl(output_dir / "all.jsonl", payload["all"])
    atomic_write_jsonl(output_dir / "train.jsonl", payload["train"])
    atomic_write_jsonl(output_dir / "validation.jsonl", payload["validation"])
    atomic_write_json(output_dir / "feature_schema.json", payload["feature_schema"])
    # Manifest is the commit marker and is intentionally written last.
    atomic_write_json(output_dir / "manifest.json", payload["manifest"])
    return output_dir


def audit_merged_experiment(
    *,
    experiment_id: str,
    contract_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        expected = build_merge_payload(
            experiment_id=experiment_id,
            contract_path=contract_path,
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        return {
            "schema_version": "1.0",
            "audit": "rescue",
            "status": "error",
            "experiment_id": experiment_id,
            "errors": [f"deterministic replay failed: {error}"],
        }
    output_dir: Path = expected["output_dir"]
    file_names = (
        "attempts.jsonl",
        "all.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "feature_schema.json",
        "manifest.json",
    )
    for name in file_names:
        if not (output_dir / name).is_file():
            errors.append(f"missing merged file: {output_dir / name}")
    if errors:
        return {
            "schema_version": "1.0",
            "audit": "rescue",
            "status": "error",
            "experiment_id": experiment_id,
            "errors": errors,
        }

    for key, name in (
        ("attempts", "attempts.jsonl"),
        ("all", "all.jsonl"),
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
    ):
        if read_jsonl(output_dir / name) != expected[key]:
            errors.append(f"deterministic replay mismatch: {name}")
    if read_json(output_dir / "feature_schema.json") != expected["feature_schema"]:
        errors.append("feature_schema was not rebuilt from final train rows")
    stored_manifest = read_json(output_dir / "manifest.json")
    expected_manifest = expected["manifest"]
    # exported_at is the sole intentionally time-varying field during replay.
    stored_without_time = dict(stored_manifest)
    expected_without_time = dict(expected_manifest)
    stored_without_time.pop("exported_at", None)
    expected_without_time.pop("exported_at", None)
    if stored_without_time != expected_without_time:
        errors.append("merged manifest deterministic replay mismatch")

    selected = expected["all"]
    replacements = expected_manifest["attempts"]["replacements"]
    cross_split = 0
    by_fingerprint = {
        row["example_fingerprint_sha256"]: row for row in expected["attempts"]
    }
    for replacement in replacements:
        cross_split += (
            by_fingerprint[replacement["replaced"]]["split"]
            != by_fingerprint[replacement["replacement"]]["split"]
        )
    checks = {
        "activation_error_is_exact_capacity_failure": expected_manifest[
            "supplemental_reserve"
        ]["activation_error"].startswith(
            read_json(POLICY_PATH)["activation"]["message_prefix"]
        ),
        "loss_value_or_rank_used": False,
        "selected_unique": len(selected)
        == len({row["example_fingerprint_sha256"] for row in selected}),
        "selected_all_usable": all(
            row["usable_for_regression"] for row in selected
        ),
        "cross_split_replacements": cross_split,
        "feature_schema_rebuilt_from_train": not any(
            error.startswith("feature_schema") for error in errors
        ),
        "deterministic_replay": not errors,
    }
    if not checks["activation_error_is_exact_capacity_failure"]:
        errors.append("activation was not the registered exact capacity failure")
    if cross_split:
        errors.append("a replacement crossed split")
    if not checks["selected_unique"] or not checks["selected_all_usable"]:
        errors.append("merged selected rows are invalid")
    return {
        "schema_version": "1.0",
        "audit": "rescue",
        "status": "ok" if not errors else "error",
        "experiment_id": experiment_id,
        "totals": {
            "selected": len(selected),
            "train": len(expected["train"]),
            "validation": len(expected["validation"]),
            "base_attempts": expected_manifest["attempts"]["base_total"],
            "supplemental_generation_attempts": expected_manifest["attempts"][
                "supplemental_generation_attempts"
            ],
            "supplemental_pool": expected_manifest["attempts"][
                "supplemental_pool"
            ],
            "replacements": len(replacements),
            "supplemental_used": expected_manifest["attempts"][
                "supplemental_used"
            ],
        },
        "checks": checks,
        "errors": errors,
        "files_sha256": {
            name: sha256_file(output_dir / name) for name in file_names
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and replay selection without writing merged files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = read_json(POLICY_PATH)
    contract_path = (
        args.contract.resolve()
        if args.contract
        else repo_path(policy["freeze_contract"])
    )
    try:
        if args.dry_run:
            payload = build_merge_payload(
                experiment_id=args.experiment,
                contract_path=contract_path,
            )
            output_dir = payload["output_dir"]
            status = "validated"
        else:
            output_dir = merge_experiment(
                experiment_id=args.experiment,
                contract_path=contract_path,
            )
            status = "merged"
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": status,
                "experiment_id": args.experiment,
                "output_dir": portable(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
