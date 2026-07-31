"""Audit the frozen wide-v2 supplemental reserve and any completed rescue.

The default ``freeze`` audit is label-blind: it reads plans, sampling
manifests, candidate specs, and the freeze contract only.  It may count current
marker pathnames, but never opens markers, summaries, curves, attempts, or
selected rows.  ``rescue`` mode delegates to the deterministic replay audit in
``merge_supplemental_reserve`` and is intended only after GT/export/merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tools.meta_model_dataset.core import (
    full_candidate_fingerprint,
    split_stratum,
)
from tools.meta_model_dataset.supplemental_reserve_common import (
    BASE_PLAN_PATH,
    POLICY_PATH,
    PROFILE_PATH,
    SUPPLEMENTAL_PLAN_PATH,
    atomic_write_json,
    base_sampling_path,
    fingerprint_set_sha256,
    marker_count,
    phase_experiments,
    portable,
    read_json,
    repo_path,
    sha256_file,
    sha256_json,
    summary_count,
    validate_static_plans,
)


FORBIDDEN_PREPARE_KEYS = {
    "target",
    "usable_for_regression",
    "dataset_role",
    "replaces_fingerprint_sha256",
    "mean_loss",
    "log_mean_loss",
    "std_loss",
    "benchmark_eligible",
}


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_PREPARE_KEYS:
                found.append(child_path)
            found.extend(_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _contract_content_hash(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    recorded = payload.pop("content_sha256", None)
    if not isinstance(recorded, str):
        return ""
    return sha256_json(payload)


def _verify_snapshot(snapshot: dict[str, Any], errors: list[str], name: str) -> None:
    environments = snapshot.get("environments")
    _add(errors, isinstance(environments, dict), f"{name} environments missing")
    if isinstance(environments, dict):
        _add(
            errors,
            snapshot.get("sha256") == sha256_json(environments),
            f"{name} digest mismatch",
        )


def _record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record.get("selection_role"))] += 1
        counts[f"{record.get('selection_role')}:{record.get('split')}"] += 1
    return dict(sorted(counts.items()))


def audit_freeze_contract(
    *,
    contract_path: Path,
    base_plan_path: Path = BASE_PLAN_PATH,
    supplemental_plan_path: Path = SUPPLEMENTAL_PLAN_PATH,
    policy_path: Path = POLICY_PATH,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        base_plan, supplemental_plan, policy = validate_static_plans(
            base_plan_path, supplemental_plan_path, policy_path
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        return {
            "schema_version": "1.0",
            "audit": "freeze",
            "status": "error",
            "errors": [f"static plan validation failed: {error}"],
        }

    if not contract_path.is_file():
        return {
            "schema_version": "1.0",
            "audit": "freeze",
            "status": "error",
            "errors": [f"missing freeze contract: {contract_path}"],
        }
    contract = read_json(contract_path)
    _add(errors, contract.get("contract_id") == policy["contract_id"], "contract_id mismatch")
    _add(
        errors,
        contract.get("content_sha256") == _contract_content_hash(contract),
        "freeze contract content hash mismatch",
    )

    source_checks = (
        ("base_plan", base_plan_path),
        ("supplemental_plan", supplemental_plan_path),
        ("policy_source", policy_path),
        ("profile", PROFILE_PATH),
    )
    for field, path in source_checks:
        entry = contract.get(field)
        if not isinstance(entry, dict):
            errors.append(f"missing contract source {field}")
            continue
        _add(errors, entry.get("path") == portable(path), f"{field} path mismatch")
        _add(
            errors,
            entry.get("sha256") == sha256_file(path),
            f"{field} sha256 mismatch",
        )
    _add(errors, contract.get("policy") == policy, "embedded policy mismatch")
    _add(
        errors,
        contract.get("policy_sha256") == sha256_json(policy),
        "embedded policy hash mismatch",
    )

    timing = contract.get("timing_snapshot", {})
    before = timing.get("before_prepare", {})
    after = timing.get("after_prepare", {})
    _verify_snapshot(before, errors, "before_prepare")
    _verify_snapshot(after, errors, "after_prepare")
    _add(errors, before == after, "base timing snapshot changed during prepare")
    _add(
        errors,
        timing.get("unchanged_during_prepare") is True,
        "unchanged_during_prepare is not true",
    )
    _add(
        errors,
        timing.get("label_files_parsed") is False,
        "freeze contract does not attest label-blind preparation",
    )
    initial_markers = timing.get("supplemental_marker_counts", {})
    initial_summaries = timing.get("supplemental_summary_counts", {})
    _add(
        errors,
        isinstance(initial_markers, dict) and not any(initial_markers.values()),
        "supplemental markers existed at freeze",
    )
    _add(
        errors,
        isinstance(initial_summaries, dict) and not any(initial_summaries.values()),
        "supplemental summaries existed at freeze",
    )

    # Successful base exports were frozen as opaque hashes.  They must remain
    # byte-identical; no contents are parsed here.
    snapshot_environments = before.get("environments", {})
    for experiment_id, snapshot in snapshot_environments.items():
        if not snapshot.get("export_exists"):
            continue
        experiment_dir = repo_path(policy["base_output_root"]) / experiment_id
        for name, expected_hash in snapshot.get(
            "opaque_export_file_sha256", {}
        ).items():
            path = experiment_dir / name
            _add(errors, path.is_file(), f"immutable base export file missing: {path}")
            if path.is_file():
                _add(
                    errors,
                    sha256_file(path) == expected_hash,
                    f"successful base export changed: {path}",
                )

    base_output_root = repo_path(policy["base_output_root"])
    supplemental_output_root = repo_path(policy["supplemental_output_root"])
    contract_experiments = contract.get("experiments", {})
    supplemental_experiments = {
        str(experiment["experiment_id"]): experiment
        for experiment in phase_experiments(supplemental_plan)
    }
    _add(
        errors,
        set(contract_experiments) == set(supplemental_experiments),
        "freeze contract experiment set mismatch",
    )

    global_base: set[str] = set()
    global_supplemental: set[str] = set()
    total_base_records = 0
    current_markers: dict[str, int] = {}
    current_summaries: dict[str, int] = {}
    experiment_reports: dict[str, Any] = {}
    for experiment_id in sorted(supplemental_experiments):
        frozen = contract_experiments.get(experiment_id, {})
        base_path = base_sampling_path(base_output_root, experiment_id)
        supplemental_path = (
            supplemental_output_root / experiment_id / "sampling_manifest.json"
        )
        if not base_path.is_file() or not supplemental_path.is_file():
            errors.append(f"sampling manifest missing for {experiment_id}")
            continue
        base_manifest = read_json(base_path)
        supplemental_manifest = read_json(supplemental_path)
        base_records = base_manifest.get("records", [])
        supplemental_records = supplemental_manifest.get("records", [])
        if not isinstance(base_records, list) or not isinstance(
            supplemental_records, list
        ):
            errors.append(f"invalid records for {experiment_id}")
            continue

        _add(
            errors,
            base_manifest.get("config_sha256")
            == sha256_json(base_manifest.get("config")),
            f"base config hash mismatch: {experiment_id}",
        )
        _add(
            errors,
            supplemental_manifest.get("config_sha256")
            == sha256_json(supplemental_manifest.get("config")),
            f"supplemental config hash mismatch: {experiment_id}",
        )
        _add(
            errors,
            frozen.get("base_sampling_manifest", {}).get("sha256")
            == sha256_file(base_path),
            f"base sampling hash changed: {experiment_id}",
        )
        _add(
            errors,
            frozen.get("supplemental_sampling_manifest", {}).get("sha256")
            == sha256_file(supplemental_path),
            f"supplemental sampling hash changed: {experiment_id}",
        )
        _add(errors, len(supplemental_records) == 67, f"record count != 67: {experiment_id}")
        counts = _record_counts(supplemental_records)
        expected_counts = {
            "primary": 50,
            "primary:train": 45,
            "primary:validation": 5,
            "reserve": 17,
        }
        for name, expected in expected_counts.items():
            _add(
                errors,
                counts.get(name, 0) == expected,
                f"{name} count mismatch in {experiment_id}",
            )
        config = supplemental_manifest.get("config", {})
        stratify_by = config.get("stratify_by", [])
        local: set[str] = set()
        missing_candidate_files = 0
        for record in supplemental_records:
            fingerprint = str(record.get("fingerprint"))
            spec = record.get("spec")
            if not isinstance(spec, dict):
                errors.append(f"record spec missing: {experiment_id}")
                continue
            _add(
                errors,
                full_candidate_fingerprint(spec) == fingerprint,
                f"fingerprint mismatch: {experiment_id}",
            )
            _add(
                errors,
                split_stratum(spec, stratify_by) == record.get("stratum"),
                f"stratum mismatch: {experiment_id}/{fingerprint}",
            )
            _add(
                errors,
                record.get("group_labels") == config.get("group_labels"),
                f"group label mismatch: {experiment_id}/{fingerprint}",
            )
            _add(errors, fingerprint not in local, f"local duplicate: {experiment_id}")
            local.add(fingerprint)
            candidate_dir = supplemental_output_root / experiment_id / str(
                record.get("artifact_dir")
            )
            expected_files = (
                "candidate_spec.json",
                "model.py",
                "loss.py",
                "optimizer.py",
                "train.py",
            )
            missing_candidate_files += sum(
                not (candidate_dir / name).is_file() for name in expected_files
            )
            spec_path = candidate_dir / "candidate_spec.json"
            if spec_path.is_file():
                _add(
                    errors,
                    read_json(spec_path) == spec,
                    f"candidate spec differs from sampling record: {candidate_dir}",
                )
        base_fingerprints = {str(record["fingerprint"]) for record in base_records}
        overlap = base_fingerprints & local
        _add(errors, not overlap, f"base/supplemental overlap: {experiment_id}")
        _add(
            errors,
            fingerprint_set_sha256(base_records)
            == frozen.get("base_fingerprint_set_sha256"),
            f"base fingerprint-set hash mismatch: {experiment_id}",
        )
        _add(
            errors,
            fingerprint_set_sha256(supplemental_records)
            == frozen.get("supplemental_fingerprint_set_sha256"),
            f"supplemental fingerprint-set hash mismatch: {experiment_id}",
        )
        excluded = set(
            config.get("external_evaluation", {}).get(
                "excluded_fingerprints_sha256", []
            )
        )
        _add(
            errors,
            base_fingerprints <= excluded,
            f"not all base attempts are excluded: {experiment_id}",
        )
        forbidden = _forbidden_keys(supplemental_manifest)
        _add(
            errors,
            not forbidden,
            f"label/selection fields in prepared manifest {experiment_id}: {forbidden[:3]}",
        )
        global_base.update(base_fingerprints)
        total_base_records += len(base_records)
        global_supplemental.update(local)
        current_markers[experiment_id] = marker_count(
            supplemental_output_root / experiment_id
        )
        current_summaries[experiment_id] = summary_count(
            supplemental_output_root / experiment_id
        )
        experiment_reports[experiment_id] = {
            "base_records": len(base_records),
            "supplemental_records": len(supplemental_records),
            "role_split_counts": counts,
            "missing_candidate_files": missing_candidate_files,
            "overlap": len(overlap),
            "current_marker_count": current_markers[experiment_id],
            "current_summary_count": current_summaries[experiment_id],
        }
        _add(
            errors,
            missing_candidate_files == 0,
            f"missing candidate inputs: {experiment_id}",
        )

    _add(
        errors,
        not (global_base & global_supplemental),
        "global base/supplemental overlap",
    )
    _add(
        errors,
        len(global_base) == total_base_records,
        "global base fingerprints are not unique",
    )
    expected_supplemental = 21 * 67
    _add(
        errors,
        len(global_supplemental) == expected_supplemental,
        "global supplemental fingerprints are not unique",
    )
    _add(
        errors,
        not _forbidden_keys(supplemental_plan),
        "label/selection fields found in supplemental plan",
    )

    return {
        "schema_version": "1.0",
        "audit": "freeze",
        "status": "ok" if not errors else "error",
        "contract": portable(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "supplemental_plan_sha256": sha256_file(supplemental_plan_path),
        "label_files_parsed": False,
        "ground_truth_started_after_freeze": any(current_markers.values()),
        "totals": {
            "experiments": len(experiment_reports),
            "base_unique_fingerprints": len(global_base),
            "supplemental_attempts": len(global_supplemental),
            "supplemental_primary": sum(
                report["role_split_counts"].get("primary", 0)
                for report in experiment_reports.values()
            ),
            "supplemental_internal_reserve": sum(
                report["role_split_counts"].get("reserve", 0)
                for report in experiment_reports.values()
            ),
            "current_markers": sum(current_markers.values()),
            "current_summaries": sum(current_summaries.values()),
        },
        "checks": {
            "preparation_was_label_blind": timing.get("label_files_parsed") is False,
            "base_snapshot_unchanged_during_prepare": before == after,
            "supplemental_gt_absent_at_freeze": not any(initial_markers.values()),
            "global_overlap": len(global_base & global_supplemental),
            "successful_base_exports_opaque_hash_verified": True,
        },
        "errors": errors,
        "experiments": experiment_reports,
    }


def _markdown(report: dict[str, Any]) -> str:
    totals = report.get("totals", {})
    lines = [
        "# Supplemental reserve freeze audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"- Experiments: {totals.get('experiments', 0)}",
        f"- Supplemental attempts: {totals.get('supplemental_attempts', 0)}",
        f"- Frozen primary pool: {totals.get('supplemental_primary', 0)}",
        f"- Internal reserve: {totals.get('supplemental_internal_reserve', 0)}",
        f"- Current GT markers (not parsed): {totals.get('current_markers', 0)}",
        "- Freeze audit parsed no target-bearing files.",
        "- Activation remains limited to the exact base-only reserve-capacity error.",
    ]
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "rescue"), default="freeze")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = read_json(POLICY_PATH)
    contract_path = (
        args.contract.resolve()
        if args.contract
        else repo_path(policy["freeze_contract"])
    )
    if args.mode == "freeze":
        report = audit_freeze_contract(contract_path=contract_path)
        default_root = repo_path(policy["supplemental_output_root"])
        default_name = "freeze_audit"
    else:
        if not args.experiment:
            print("error: --experiment is required for rescue audit", file=sys.stderr)
            return 1
        from tools.meta_model_dataset.merge_supplemental_reserve import (
            audit_merged_experiment,
        )

        report = audit_merged_experiment(
            experiment_id=args.experiment,
            contract_path=contract_path,
        )
        default_root = repo_path(policy["merged_output_root"]) / args.experiment
        default_name = "rescue_audit"
    json_out = args.json_out or default_root / f"{default_name}.json"
    markdown_out = args.markdown_out or default_root / f"{default_name}.md"
    atomic_write_json(json_out, report)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_out.with_name(f".{markdown_out.name}.tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown_out)
    print(json.dumps(report.get("totals", {}), indent=2, sort_keys=True))
    print(f"status={report['status']} report={portable(json_out)}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
