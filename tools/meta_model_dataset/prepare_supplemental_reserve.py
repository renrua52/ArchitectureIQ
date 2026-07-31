"""Freeze and prepare the label-blind wide-v2 supplemental reserve sidecar.

The script reads only plans, profiles, dataset inputs, and sampling manifests.
It counts marker *pathnames* and hashes completed base exports as opaque bytes;
it never parses GT markers, summaries, curves, attempts, selected rows, or
targets.  Candidate specs are materialized through the unchanged standard
builder with ``stage='prepare'``.  No ground truth is run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.meta_model_dataset.build import build_from_plan
from tools.meta_model_dataset.supplemental_reserve_common import (
    BASE_PLAN_PATH,
    CONTRACT_ID,
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
    timing_snapshot,
    utc_now,
    validate_static_plans,
)


def _contract_path(policy: dict[str, Any]) -> Path:
    return repo_path(policy["freeze_contract"])


def _freeze_experiments(
    *,
    base_plan: dict[str, Any],
    supplemental_plan: dict[str, Any],
    base_output_root: Path,
    supplemental_output_root: Path,
) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for experiment in phase_experiments(supplemental_plan):
        experiment_id = str(experiment["experiment_id"])
        base_path = base_sampling_path(base_output_root, experiment_id)
        supplemental_path = (
            supplemental_output_root / experiment_id / "sampling_manifest.json"
        )
        if not base_path.is_file():
            raise FileNotFoundError(f"Missing base sampling manifest: {base_path}")
        if not supplemental_path.is_file():
            raise FileNotFoundError(
                f"Missing supplemental sampling manifest: {supplemental_path}"
            )
        base_manifest = read_json(base_path)
        supplemental_manifest = read_json(supplemental_path)
        base_records = base_manifest.get("records")
        supplemental_records = supplemental_manifest.get("records")
        if not isinstance(base_records, list) or not isinstance(
            supplemental_records, list
        ):
            raise TypeError(f"Sampling records are not lists for {experiment_id}")
        base_fingerprints = {str(record["fingerprint"]) for record in base_records}
        supplemental_fingerprints = {
            str(record["fingerprint"]) for record in supplemental_records
        }
        overlap = base_fingerprints & supplemental_fingerprints
        if overlap:
            raise ValueError(f"Base/supplemental overlap in {experiment_id}")
        experiments[experiment_id] = {
            "base_sampling_manifest": {
                "path": portable(base_path),
                "sha256": sha256_file(base_path),
            },
            "supplemental_sampling_manifest": {
                "path": portable(supplemental_path),
                "sha256": sha256_file(supplemental_path),
            },
            "base_records": len(base_records),
            "supplemental_records": len(supplemental_records),
            "supplemental_selected_target": 50,
            "supplemental_train_target": 45,
            "supplemental_validation_target": 5,
            "supplemental_internal_reserve": 17,
            "base_fingerprint_set_sha256": fingerprint_set_sha256(base_records),
            "supplemental_fingerprint_set_sha256": fingerprint_set_sha256(
                supplemental_records
            ),
            "overlap": 0,
        }
    return experiments


def prepare_and_freeze(
    *,
    base_plan_path: Path = BASE_PLAN_PATH,
    supplemental_plan_path: Path = SUPPLEMENTAL_PLAN_PATH,
    policy_path: Path = POLICY_PATH,
) -> Path:
    base_plan, supplemental_plan, policy = validate_static_plans(
        base_plan_path, supplemental_plan_path, policy_path
    )
    contract_path = _contract_path(policy)
    if contract_path.exists():
        raise FileExistsError(
            f"Freeze contract already exists: {contract_path}; audit it instead of "
            "sampling again"
        )

    base_output_root = repo_path(policy["base_output_root"])
    supplemental_output_root = repo_path(policy["supplemental_output_root"])
    experiment_ids = [
        str(experiment["experiment_id"])
        for experiment in phase_experiments(base_plan)
    ]
    before = timing_snapshot(base_output_root, experiment_ids)

    build_from_plan(
        plan_path=supplemental_plan_path.resolve(),
        stage="prepare",
        workers=1,
        requested_phases={"b2_scale"},
    )

    after = timing_snapshot(base_output_root, experiment_ids)
    if before != after:
        raise RuntimeError(
            "Base marker/export snapshot changed while supplemental reserve was "
            "being prepared; pause B2 and retry in a fresh output root"
        )
    supplemental_marker_counts = {
        experiment_id: marker_count(supplemental_output_root / experiment_id)
        for experiment_id in experiment_ids
    }
    supplemental_summary_counts = {
        experiment_id: summary_count(supplemental_output_root / experiment_id)
        for experiment_id in experiment_ids
    }
    if any(supplemental_marker_counts.values()) or any(
        supplemental_summary_counts.values()
    ):
        raise RuntimeError(
            "Supplemental GT already exists before freeze; refuse post-label freeze"
        )

    experiments = _freeze_experiments(
        base_plan=base_plan,
        supplemental_plan=supplemental_plan,
        base_output_root=base_output_root,
        supplemental_output_root=supplemental_output_root,
    )
    contract = {
        "schema_version": "1.0",
        "contract_id": CONTRACT_ID,
        "created_at": utc_now(),
        "base_plan": {
            "path": portable(base_plan_path),
            "sha256": sha256_file(base_plan_path),
        },
        "supplemental_plan": {
            "path": portable(supplemental_plan_path),
            "sha256": sha256_file(supplemental_plan_path),
        },
        "policy_source": {
            "path": portable(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "policy": policy,
        "policy_sha256": sha256_json(policy),
        "profile": {
            "path": portable(PROFILE_PATH),
            "sha256": sha256_file(PROFILE_PATH),
        },
        "timing_snapshot": {
            "before_prepare": before,
            "after_prepare": after,
            "unchanged_during_prepare": True,
            "label_files_parsed": False,
            "supplemental_marker_counts": supplemental_marker_counts,
            "supplemental_summary_counts": supplemental_summary_counts,
            "note": (
                "Only marker path counts, export existence, and opaque file hashes "
                "were observed; no GT or target-bearing file was parsed."
            ),
        },
        "experiments": experiments,
    }
    contract["content_sha256"] = sha256_json(contract)
    atomic_write_json(contract_path, contract)
    return contract_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, default=BASE_PLAN_PATH)
    parser.add_argument(
        "--supplemental-plan", type=Path, default=SUPPLEMENTAL_PLAN_PATH
    )
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract_path = prepare_and_freeze(
            base_plan_path=args.base_plan.resolve(),
            supplemental_plan_path=args.supplemental_plan.resolve(),
            policy_path=args.policy.resolve(),
        )
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
            {"status": "frozen", "contract": portable(contract_path)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
