"""Shared, side-effect-free helpers for the frozen wide-v2 reserve sidecar.

This module deliberately does not inspect ground-truth targets.  The prepare
path is allowed to hash base export files as opaque bytes and count marker
pathnames, but it never parses summaries, curves, attempts, or selected rows.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "tools/meta_model_dataset/supplemental_reserve_policy_v1.json"
BASE_PLAN_PATH = ROOT / "tools/meta_model_dataset/plan_wide_v2.json"
SUPPLEMENTAL_PLAN_PATH = (
    ROOT / "tools/meta_model_dataset/plan_wide_v2_supplemental_reserve_v1.json"
)
PROFILE_PATH = ROOT / "profiles/meta_wide_v2.yaml"
CONTRACT_ID = "wide_v2_supplemental_reserve_v1"
PHASE = "b2_scale"
SEED_MODULUS = 2_147_483_647


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
    temporary.replace(path)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolved(plan: dict[str, Any], experiment: dict[str, Any], key: str) -> Any:
    if key in experiment:
        return experiment[key]
    return plan.get("defaults", {}).get(key)


def phase_experiments(
    plan: dict[str, Any], phase: str = PHASE
) -> list[dict[str, Any]]:
    return [
        experiment
        for experiment in plan.get("experiments", [])
        if str(resolved(plan, experiment, "phase")) == phase
    ]


def deterministic_seed(
    base_plan_sha256: str,
    experiment_id: str,
    purpose: str,
    contract_id: str = CONTRACT_ID,
) -> int:
    material = f"{base_plan_sha256}:{contract_id}:{experiment_id}:{purpose}"
    value = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    return value % SEED_MODULUS or 1


def base_sampling_path(base_output_root: Path, experiment_id: str) -> Path:
    return base_output_root / experiment_id / "sampling_manifest.json"


def marker_count(experiment_dir: Path) -> int:
    return sum(
        1
        for _ in experiment_dir.glob(
            "candidates/*/results/meta_model_gt.json"
        )
    )


def summary_count(experiment_dir: Path) -> int:
    return sum(
        1 for _ in experiment_dir.glob("candidates/*/results/summary.json")
    )


def opaque_export_hashes(experiment_dir: Path) -> dict[str, str]:
    """Hash a completed export without parsing any label-bearing file."""

    names = (
        "sampling_manifest.json",
        "attempts.jsonl",
        "all.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "feature_schema.json",
        "manifest.json",
    )
    return {
        name: sha256_file(experiment_dir / name)
        for name in names
        if (experiment_dir / name).is_file()
    }


def timing_snapshot(
    base_output_root: Path,
    experiment_ids: Iterable[str],
) -> dict[str, Any]:
    environments: dict[str, Any] = {}
    for experiment_id in experiment_ids:
        experiment_dir = base_output_root / experiment_id
        export_exists = (experiment_dir / "manifest.json").is_file()
        markers = marker_count(experiment_dir)
        if export_exists:
            timing_class = "post_success_immutable"
        elif markers:
            timing_class = "pre_export_label_blind_contingency"
        else:
            timing_class = "pre_gt"
        environments[experiment_id] = {
            "marker_count": markers,
            "export_exists": export_exists,
            "timing_class": timing_class,
            "opaque_export_file_sha256": (
                opaque_export_hashes(experiment_dir) if export_exists else {}
            ),
        }
    snapshot = {"environments": environments}
    snapshot["sha256"] = sha256_json(environments)
    return snapshot


def fingerprint_set_sha256(records: list[dict[str, Any]]) -> str:
    return sha256_json(sorted(str(record["fingerprint"]) for record in records))


def contextual_experiment_fields(
    plan: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    defaults = plan.get("defaults", {})
    ground_truth = {
        **defaults.get("ground_truth", {}),
        **experiment.get("ground_truth", {}),
    }
    return {
        "experiment_id": str(experiment["experiment_id"]),
        "phase": str(resolved(plan, experiment, "phase")),
        "dataset_path": str(experiment["dataset_path"]),
        "budget": int(experiment["budget"]),
        "batch_size": int(experiment["batch_size"]),
        "vary": sorted(resolved(plan, experiment, "vary") or []),
        "fixed": experiment.get("fixed", defaults.get("fixed", {})),
        "stratify_by": list(resolved(plan, experiment, "stratify_by") or []),
        "ground_truth": ground_truth,
        "group_labels": experiment.get(
            "group_labels", defaults.get("group_labels", {})
        ),
    }


def validate_static_plans(
    base_plan_path: Path = BASE_PLAN_PATH,
    supplemental_plan_path: Path = SUPPLEMENTAL_PLAN_PATH,
    policy_path: Path = POLICY_PATH,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate that the source-controlled plan is a deterministic B2 copy."""

    base_plan = read_json(base_plan_path)
    supplemental_plan = read_json(supplemental_plan_path)
    policy = read_json(policy_path)
    base_sha = sha256_file(base_plan_path)
    errors: list[str] = []

    if policy.get("contract_id") != CONTRACT_ID:
        errors.append("policy contract_id mismatch")
    if supplemental_plan.get("contract_id") != CONTRACT_ID:
        errors.append("supplemental plan contract_id mismatch")
    if supplemental_plan.get("base_plan_sha256") != base_sha:
        errors.append("supplemental plan base_plan_sha256 mismatch")
    if portable(repo_path(policy["base_plan"])) != portable(base_plan_path):
        errors.append("policy base plan path mismatch")
    if portable(repo_path(policy["supplemental_plan"])) != portable(
        supplemental_plan_path
    ):
        errors.append("policy supplemental plan path mismatch")
    if supplemental_plan.get("output_root") != policy.get(
        "supplemental_output_root"
    ):
        errors.append("supplemental output_root mismatch")

    base_experiments = {
        str(experiment["experiment_id"]): experiment
        for experiment in phase_experiments(base_plan)
    }
    supplemental_experiments = {
        str(experiment["experiment_id"]): experiment
        for experiment in phase_experiments(supplemental_plan)
    }
    if set(base_experiments) != set(supplemental_experiments):
        errors.append("B2 experiment IDs differ between base and supplemental plans")
    if len(supplemental_experiments) != 21:
        errors.append("supplemental plan must contain exactly 21 B2 experiments")

    ignored_gt_difference = {"n_seeds", "base_seed", "fail_threshold_mode"}
    for experiment_id in sorted(set(base_experiments) & set(supplemental_experiments)):
        base_fields = contextual_experiment_fields(
            base_plan, base_experiments[experiment_id]
        )
        supplemental_fields = contextual_experiment_fields(
            supplemental_plan, supplemental_experiments[experiment_id]
        )
        # Ground-truth defaults are required to resolve identically.  The local
        # variable makes any future exception explicit instead of silently
        # dropping the entire field.
        _ = ignored_gt_difference
        if base_fields != supplemental_fields:
            errors.append(f"context fields differ for {experiment_id}")

        supplemental_experiment = supplemental_experiments[experiment_id]
        for purpose, key in (("sampling", "sampling_seed"), ("split", "split_seed")):
            expected_seed = deterministic_seed(base_sha, experiment_id, purpose)
            if int(supplemental_experiment.get(key, -1)) != expected_seed:
                errors.append(f"{key} mismatch for {experiment_id}")
        expected_exclusion = (
            f"data/meta_model/setting_to_loss_wide_v2/{experiment_id}/"
            "sampling_manifest.json"
        )
        if supplemental_experiment.get("exclude_sampling_manifests") != [
            expected_exclusion
        ]:
            errors.append(f"base-manifest exclusion mismatch for {experiment_id}")

    defaults = supplemental_plan.get("defaults", {})
    expected_counts = {"num_rows": 50, "train_rows": 45, "reserve_rows": 17}
    for key, expected in expected_counts.items():
        if int(defaults.get(key, -1)) != expected:
            errors.append(f"supplemental default {key} must be {expected}")

    if errors:
        raise ValueError("; ".join(errors))
    return base_plan, supplemental_plan, policy

