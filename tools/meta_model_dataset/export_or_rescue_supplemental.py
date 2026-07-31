"""Resume-safe wide-v2 B2 standard export with frozen sidecar rescue.

Run this only after base ``--stage gt --phase b2_scale`` has completed.  For
each B2 environment the driver first invokes the unchanged standard builder in
``export`` mode.  A sidecar is activated only when that call raises the exact
pre-registered reserve-capacity ``RuntimeError``.  The rescue path is:

``standard sidecar build(stage='all') -> deterministic merge -> rescue audit``

Every other exception stops the run immediately.  Existing successful base
exports and already-audited merged exports are never rewritten.  Progress is
written atomically after every state transition, so rerunning the same command
resumes from on-disk artifacts.  Ground truth remains exclusively in the
unchanged standard builder/run_ground_truth path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from tools.meta_model_dataset import build as standard_builder
from tools.meta_model_dataset.merge_supplemental_reserve import (
    audit_merged_experiment,
    merge_experiment,
)
from tools.meta_model_dataset.supplemental_reserve_common import (
    BASE_PLAN_PATH,
    POLICY_PATH,
    SUPPLEMENTAL_PLAN_PATH,
    atomic_write_json,
    canonical_json,
    phase_experiments,
    portable,
    read_json,
    repo_path,
    sha256_file,
    sha256_json,
    utc_now,
)


DRIVER_ID = "wide_v2_export_or_rescue_v1"
MIN_WORKERS = 1
MAX_WORKERS = 4


EventSink = Callable[[dict[str, Any]], None]


def _stdout_event(event: dict[str, Any]) -> None:
    print(canonical_json(event), flush=True)


def _contract_payload_sha256(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    recorded = payload.pop("content_sha256", None)
    if not isinstance(recorded, str):
        return ""
    return sha256_json(payload)


def _validate_inputs(
    *,
    base_plan_path: Path,
    supplemental_plan_path: Path,
    policy_path: Path,
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
]:
    base_plan = read_json(base_plan_path)
    supplemental_plan = read_json(supplemental_plan_path)
    policy = read_json(policy_path)
    contract = read_json(contract_path)
    errors: list[str] = []

    if contract.get("contract_id") != policy.get("contract_id"):
        errors.append("contract_id mismatch")
    if contract.get("policy_sha256") != sha256_json(policy):
        errors.append("policy hash mismatch")
    if contract.get("content_sha256") != _contract_payload_sha256(contract):
        errors.append("freeze contract content hash mismatch")
    for field, path in (
        ("base_plan", base_plan_path),
        ("supplemental_plan", supplemental_plan_path),
        ("policy_source", policy_path),
    ):
        source = contract.get(field)
        if not isinstance(source, dict) or source.get("sha256") != sha256_file(path):
            errors.append(f"frozen source changed: {field}")

    base_ids = [
        str(experiment["experiment_id"])
        for experiment in phase_experiments(base_plan, "b2_scale")
    ]
    supplemental_ids = [
        str(experiment["experiment_id"])
        for experiment in phase_experiments(supplemental_plan, "b2_scale")
    ]
    if not base_ids or len(base_ids) != len(set(base_ids)):
        errors.append("base B2 experiment IDs are missing or duplicated")
    if base_ids != supplemental_ids:
        errors.append("base and supplemental B2 experiment order differs")
    frozen_ids = set(contract.get("experiments", {}))
    if set(base_ids) != frozen_ids:
        errors.append("freeze contract experiment set differs from B2 plan")
    if repo_path(base_plan["output_root"]).resolve() != repo_path(
        policy["base_output_root"]
    ).resolve():
        errors.append("base output root differs from policy")
    if repo_path(supplemental_plan["output_root"]).resolve() != repo_path(
        policy["supplemental_output_root"]
    ).resolve():
        errors.append("supplemental output root differs from policy")
    activation = policy.get("activation", {})
    if activation.get("exception_type") != "RuntimeError" or not isinstance(
        activation.get("message_prefix"), str
    ):
        errors.append("activation exception policy is invalid")
    if activation.get("successful_base_exports_immutable") is not True:
        errors.append("policy does not freeze successful base exports")

    frozen_environments = (
        contract.get("timing_snapshot", {})
        .get("before_prepare", {})
        .get("environments")
    )
    if not isinstance(frozen_environments, dict) or set(frozen_environments) != set(
        base_ids
    ):
        errors.append("freeze timing snapshot experiment set differs from B2 plan")
    else:
        base_root = repo_path(policy["base_output_root"])
        for experiment_id, snapshot in frozen_environments.items():
            if not snapshot.get("export_exists"):
                continue
            for name, expected_hash in snapshot.get(
                "opaque_export_file_sha256", {}
            ).items():
                path = base_root / experiment_id / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    errors.append(
                        f"successful frozen base export changed: {experiment_id}/{name}"
                    )
    if errors:
        raise ValueError("; ".join(errors))
    return base_plan, supplemental_plan, policy, contract, base_ids


def _source_identity(
    *,
    base_plan_path: Path,
    supplemental_plan_path: Path,
    policy_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    return {
        "base_plan": {
            "path": portable(base_plan_path),
            "sha256": sha256_file(base_plan_path),
        },
        "supplemental_plan": {
            "path": portable(supplemental_plan_path),
            "sha256": sha256_file(supplemental_plan_path),
        },
        "policy": {
            "path": portable(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "freeze_contract": {
            "path": portable(contract_path),
            "sha256": sha256_file(contract_path),
        },
    }


def _new_progress(
    *,
    experiment_ids: list[str],
    sources: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": "1.0",
        "driver_id": DRIVER_ID,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "workers": workers,
        "worker_limit": MAX_WORKERS,
        "sources": sources,
        "experiment_order": experiment_ids,
        "experiments": {
            experiment_id: {
                "status": "pending",
                "attempts": 0,
                "history": [],
            }
            for experiment_id in experiment_ids
        },
        "summary": {},
    }


def _load_or_create_progress(
    *,
    progress_path: Path,
    experiment_ids: list[str],
    sources: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    if not progress_path.is_file():
        return _new_progress(
            experiment_ids=experiment_ids,
            sources=sources,
            workers=workers,
        )
    progress = read_json(progress_path)
    if progress.get("driver_id") != DRIVER_ID:
        raise ValueError(f"Progress belongs to another driver: {progress_path}")
    if progress.get("sources") != sources:
        raise ValueError(f"Frozen sources changed since progress began: {progress_path}")
    if progress.get("experiment_order") != experiment_ids:
        raise ValueError(f"Experiment order changed since progress began: {progress_path}")
    if set(progress.get("experiments", {})) != set(experiment_ids):
        raise ValueError(f"Progress experiment set is invalid: {progress_path}")
    progress["workers"] = workers
    progress["worker_limit"] = MAX_WORKERS
    progress["status"] = "running"
    return progress


def _summary(progress: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in progress["experiments"].values():
        status = str(state.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _record_transition(
    *,
    progress: dict[str, Any],
    progress_path: Path,
    experiment_id: str,
    status: str,
    event_sink: EventSink,
    details: dict[str, Any] | None = None,
) -> None:
    timestamp = utc_now()
    state = progress["experiments"][experiment_id]
    state["status"] = status
    state["updated_at"] = timestamp
    event = {
        "schema_version": "1.0",
        "driver_id": DRIVER_ID,
        "time": timestamp,
        "experiment_id": experiment_id,
        "status": status,
        **(details or {}),
    }
    state.setdefault("history", []).append(event)
    progress["updated_at"] = timestamp
    progress["summary"] = _summary(progress)
    atomic_write_json(progress_path, progress)
    event_sink(event)


def _complete_progress(
    *,
    progress: dict[str, Any],
    progress_path: Path,
    event_sink: EventSink,
) -> None:
    progress["status"] = "complete"
    progress["updated_at"] = utc_now()
    progress["summary"] = _summary(progress)
    atomic_write_json(progress_path, progress)
    event_sink(
        {
            "schema_version": "1.0",
            "driver_id": DRIVER_ID,
            "time": progress["updated_at"],
            "status": "complete",
            "summary": progress["summary"],
            "progress_manifest": portable(progress_path),
        }
    )


def _dry_run_report(
    *,
    experiment_ids: list[str],
    policy: dict[str, Any],
    workers: int,
    sources: dict[str, Any],
) -> dict[str, Any]:
    base_root = repo_path(policy["base_output_root"])
    supplemental_root = repo_path(policy["supplemental_output_root"])
    merged_root = repo_path(policy["merged_output_root"])
    actions: list[dict[str, Any]] = []
    for experiment_id in experiment_ids:
        base_manifest = base_root / experiment_id / "manifest.json"
        sidecar_manifest = supplemental_root / experiment_id / "manifest.json"
        merged_manifest = merged_root / experiment_id / "manifest.json"
        if base_manifest.is_file() and merged_manifest.is_file():
            action = "error_conflicting_base_and_merged_exports"
        elif base_manifest.is_file():
            action = "skip_existing_base_export"
        elif merged_manifest.is_file():
            action = "audit_existing_rescue"
        elif sidecar_manifest.is_file():
            action = "attempt_base_export_then_resume_merge_if_exact_capacity_failure"
        else:
            action = "attempt_base_export_then_conditionally_run_sidecar_rescue"
        actions.append(
            {
                "experiment_id": experiment_id,
                "action": action,
                "base_manifest_exists": base_manifest.is_file(),
                "sidecar_manifest_exists": sidecar_manifest.is_file(),
                "merged_manifest_exists": merged_manifest.is_file(),
            }
        )
    return {
        "schema_version": "1.0",
        "driver_id": DRIVER_ID,
        "status": "dry_run",
        "workers": workers,
        "worker_limit": MAX_WORKERS,
        "sources": sources,
        "writes_performed": False,
        "builder_calls_performed": False,
        "actions": actions,
    }


def run_driver(
    *,
    workers: int,
    dry_run: bool = False,
    base_plan_path: Path = BASE_PLAN_PATH,
    supplemental_plan_path: Path = SUPPLEMENTAL_PLAN_PATH,
    policy_path: Path = POLICY_PATH,
    contract_path: Path | None = None,
    progress_path: Path | None = None,
    event_sink: EventSink = _stdout_event,
) -> dict[str, Any]:
    if not MIN_WORKERS <= workers <= MAX_WORKERS:
        raise ValueError(
            f"workers must be between {MIN_WORKERS} and {MAX_WORKERS}, got {workers}"
        )
    base_plan_path = base_plan_path.resolve()
    supplemental_plan_path = supplemental_plan_path.resolve()
    policy_path = policy_path.resolve()
    raw_policy = read_json(policy_path)
    contract_path = (
        contract_path.resolve()
        if contract_path is not None
        else repo_path(raw_policy["freeze_contract"]).resolve()
    )
    _base_plan, _supplemental_plan, policy, _contract, experiment_ids = (
        _validate_inputs(
            base_plan_path=base_plan_path,
            supplemental_plan_path=supplemental_plan_path,
            policy_path=policy_path,
            contract_path=contract_path,
        )
    )
    sources = _source_identity(
        base_plan_path=base_plan_path,
        supplemental_plan_path=supplemental_plan_path,
        policy_path=policy_path,
        contract_path=contract_path,
    )
    if dry_run:
        report = _dry_run_report(
            experiment_ids=experiment_ids,
            policy=policy,
            workers=workers,
            sources=sources,
        )
        event_sink(report)
        return report

    progress_path = (
        progress_path.resolve()
        if progress_path is not None
        else (
            repo_path(policy["base_output_root"])
            / "supplemental_reserve_driver_v1.json"
        ).resolve()
    )
    progress = _load_or_create_progress(
        progress_path=progress_path,
        experiment_ids=experiment_ids,
        sources=sources,
        workers=workers,
    )
    base_root = repo_path(policy["base_output_root"])
    supplemental_root = repo_path(policy["supplemental_output_root"])
    merged_root = repo_path(policy["merged_output_root"])
    exact_prefix = str(policy["activation"]["message_prefix"])

    for experiment_id in experiment_ids:
        state = progress["experiments"][experiment_id]
        state["attempts"] = int(state.get("attempts", 0)) + 1
        base_manifest = base_root / experiment_id / "manifest.json"
        sidecar_manifest = supplemental_root / experiment_id / "manifest.json"
        merged_manifest = merged_root / experiment_id / "manifest.json"
        try:
            if base_manifest.is_file() and merged_manifest.is_file():
                raise RuntimeError(
                    "Both base and merged manifests exist; refuse ambiguous resume"
                )
            if base_manifest.is_file():
                _record_transition(
                    progress=progress,
                    progress_path=progress_path,
                    experiment_id=experiment_id,
                    status="base_export_already_complete",
                    event_sink=event_sink,
                    details={"base_manifest_sha256": sha256_file(base_manifest)},
                )
                continue
            if merged_manifest.is_file():
                audit = audit_merged_experiment(
                    experiment_id=experiment_id,
                    contract_path=contract_path,
                )
                atomic_write_json(
                    merged_manifest.parent / "rescue_audit.json", audit
                )
                if audit.get("status") != "ok":
                    raise RuntimeError(
                        f"Existing rescue audit failed: {audit.get('errors')}"
                    )
                _record_transition(
                    progress=progress,
                    progress_path=progress_path,
                    experiment_id=experiment_id,
                    status="rescue_already_complete",
                    event_sink=event_sink,
                    details={
                        "merged_manifest_sha256": sha256_file(merged_manifest),
                        "audit_status": "ok",
                    },
                )
                continue

            _record_transition(
                progress=progress,
                progress_path=progress_path,
                experiment_id=experiment_id,
                status="attempting_base_export",
                event_sink=event_sink,
            )
            try:
                standard_builder.build_from_plan(
                    plan_path=base_plan_path,
                    stage="export",
                    workers=workers,
                    requested_experiments={experiment_id},
                    requested_phases={"b2_scale"},
                )
            except RuntimeError as error:
                activation_error = str(error)
                if not activation_error.startswith(exact_prefix):
                    raise
            else:
                if not base_manifest.is_file():
                    raise RuntimeError(
                        "Standard export returned without writing its manifest"
                    )
                _record_transition(
                    progress=progress,
                    progress_path=progress_path,
                    experiment_id=experiment_id,
                    status="base_export_complete",
                    event_sink=event_sink,
                    details={"base_manifest_sha256": sha256_file(base_manifest)},
                )
                continue

            _record_transition(
                progress=progress,
                progress_path=progress_path,
                experiment_id=experiment_id,
                status="exact_capacity_failure",
                event_sink=event_sink,
                details={"activation_error": activation_error},
            )
            if not sidecar_manifest.is_file():
                _record_transition(
                    progress=progress,
                    progress_path=progress_path,
                    experiment_id=experiment_id,
                    status="running_sidecar_standard_build",
                    event_sink=event_sink,
                    details={"workers": workers, "stage": "all"},
                )
                standard_builder.build_from_plan(
                    plan_path=supplemental_plan_path,
                    stage="all",
                    workers=workers,
                    requested_experiments={experiment_id},
                    requested_phases={"b2_scale"},
                )
            if not sidecar_manifest.is_file():
                raise RuntimeError(
                    "Sidecar standard build returned without a manifest"
                )
            _record_transition(
                progress=progress,
                progress_path=progress_path,
                experiment_id=experiment_id,
                status="sidecar_standard_export_complete",
                event_sink=event_sink,
                details={
                    "sidecar_manifest_sha256": sha256_file(sidecar_manifest)
                },
            )

            if not merged_manifest.is_file():
                _record_transition(
                    progress=progress,
                    progress_path=progress_path,
                    experiment_id=experiment_id,
                    status="merging_frozen_reserve",
                    event_sink=event_sink,
                )
                merge_experiment(
                    experiment_id=experiment_id,
                    contract_path=contract_path,
                )
            if not merged_manifest.is_file():
                raise RuntimeError("Merge returned without a manifest")
            audit = audit_merged_experiment(
                experiment_id=experiment_id,
                contract_path=contract_path,
            )
            atomic_write_json(merged_manifest.parent / "rescue_audit.json", audit)
            if audit.get("status") != "ok":
                raise RuntimeError(f"Rescue audit failed: {audit.get('errors')}")
            _record_transition(
                progress=progress,
                progress_path=progress_path,
                experiment_id=experiment_id,
                status="rescue_complete",
                event_sink=event_sink,
                details={
                    "merged_manifest_sha256": sha256_file(merged_manifest),
                    "audit_status": "ok",
                },
            )
        except Exception as error:  # noqa: BLE001 - fail-closed progress boundary
            progress["status"] = "error"
            _record_transition(
                progress=progress,
                progress_path=progress_path,
                experiment_id=experiment_id,
                status="error",
                event_sink=event_sink,
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            progress["status"] = "error"
            progress["updated_at"] = utc_now()
            progress["summary"] = _summary(progress)
            atomic_write_json(progress_path, progress)
            return progress

    _complete_progress(
        progress=progress,
        progress_path=progress_path,
        event_sink=event_sink,
    )
    return progress


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Sidecar GT workers; hard-limited to 1..4.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-plan", type=Path, default=BASE_PLAN_PATH)
    parser.add_argument(
        "--supplemental-plan", type=Path, default=SUPPLEMENTAL_PLAN_PATH
    )
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--progress", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_driver(
            workers=args.workers,
            dry_run=args.dry_run,
            base_plan_path=args.base_plan,
            supplemental_plan_path=args.supplemental_plan,
            policy_path=args.policy,
            contract_path=args.contract,
            progress_path=args.progress,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        _stdout_event(
            {
                "schema_version": "1.0",
                "driver_id": DRIVER_ID,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return 1
    if result.get("status") not in {"complete", "dry_run"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
