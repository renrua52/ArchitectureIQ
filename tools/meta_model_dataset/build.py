"""Build reproducible setting-to-loss datasets through the canonical GT path.

Run from the repository root:

    .venv/bin/python -m tools.meta_model_dataset.build \
        --plan tools/meta_model_dataset/plan_60q_id_v1.json --stage all

The builder samples candidate specs, renders the normal candidate Python files,
executes them with ``run_ground_truth``, and exports one row per setting.  It
never constructs labels with a parallel training implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pin_thread_environment() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


_pin_thread_environment()

from architecture_iq.candidates.generator import (  # noqa: E402
    sample_candidate,
    sample_loss,
    sample_model,
    sample_optimizer,
    write_candidate,
)
from architecture_iq.candidates.sets import parse_varying_axes  # noqa: E402
from architecture_iq.paths import ROOT  # noqa: E402
from architecture_iq.profile import Profile, load_profile  # noqa: E402
from architecture_iq.registry import (  # noqa: E402
    ensure_registries,
    get_dataset_family,
    get_model_type,
)
from architecture_iq.util import read_json, write_json  # noqa: E402
from tools.meta_model_dataset.core import (  # noqa: E402
    assign_pre_execution_splits,
    build_attempt_row,
    build_feature_schema,
    full_candidate_fingerprint,
    select_usable_rows,
    sha256_file,
    sha256_json,
    split_stratum,
)


STAGES = ("prepare", "gt", "export", "all")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _portable_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def _experiment_config(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    profile: Profile,
    dataset_spec: dict[str, Any],
) -> dict[str, Any]:
    defaults = plan.get("defaults", {})
    wide_contract = str(plan.get("schema_version", "1.0")).split(".", 1)[0] == "2"
    gt_defaults = defaults.get("ground_truth", {})
    gt_experiment = experiment.get("ground_truth", {})
    plan_exclusions = plan.get("exclusions", {})
    excluded_sets = [
        _repo_path(path)
        for path in [
            *plan_exclusions.get("candidate_sets", []),
            *experiment.get("exclude_candidate_sets", []),
        ]
    ]
    excluded_sampling_manifests = [
        _repo_path(path)
        for path in [
            *plan_exclusions.get("sampling_manifests", []),
            *experiment.get("exclude_sampling_manifests", []),
        ]
    ]
    excluded_fingerprints: set[str] = set()
    exclusion_sources: list[dict[str, Any]] = []
    for set_path in excluded_sets:
        spec_paths = sorted(set_path.glob("*/candidate_spec.json"))
        if not spec_paths:
            raise FileNotFoundError(f"No candidate specs under excluded set: {set_path}")
        relevant = 0
        for spec_path in spec_paths:
            spec = read_json(spec_path)
            if (
                spec.get("dataset_id") != dataset_spec["dataset_id"]
                or spec.get("family") != dataset_spec["family"]
            ):
                continue
            excluded_fingerprints.add(full_candidate_fingerprint(spec))
            relevant += 1
        exclusion_sources.append(
            {
                "kind": "candidate_set",
                "path": _portable_repo_path(set_path),
                "source_sha256": sha256_json(
                    [sha256_file(path) for path in spec_paths]
                ),
                "candidate_specs": len(spec_paths),
                "relevant_candidate_specs": relevant,
            }
        )
    for manifest_path in excluded_sampling_manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing excluded sampling manifest: {manifest_path}"
            )
        manifest = read_json(manifest_path)
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"Excluded sampling manifest has no records: {manifest_path}"
            )
        relevant = 0
        for record in records:
            spec = record.get("spec")
            if not isinstance(spec, dict):
                raise TypeError(f"Excluded record has no spec: {manifest_path}")
            if (
                spec.get("dataset_id") != dataset_spec["dataset_id"]
                or spec.get("family") != dataset_spec["family"]
            ):
                continue
            excluded_fingerprints.add(full_candidate_fingerprint(spec))
            relevant += 1
        exclusion_sources.append(
            {
                "kind": "sampling_manifest",
                "path": _portable_repo_path(manifest_path),
                "source_sha256": sha256_file(manifest_path),
                "candidate_specs": len(records),
                "relevant_candidate_specs": relevant,
            }
        )

    phase = str(experiment.get("phase", defaults.get("phase", "default")))
    group_labels = {
        "phase": phase,
        "family": str(dataset_spec["family"]),
        "dataset": str(dataset_spec["dataset_id"]),
        "environment": str(experiment["experiment_id"]),
    }
    custom_group_labels = deepcopy(experiment.get("group_labels", {}))
    overlap = set(group_labels) & set(custom_group_labels)
    if overlap:
        raise ValueError(
            "group_labels cannot override automatic labels: "
            f"{sorted(overlap)}"
        )
    group_labels.update(custom_group_labels)
    config = {
        "schema_version": str(plan.get("schema_version", "1.0")),
        "experiment_id": str(experiment["experiment_id"]),
        "profile": profile.name,
        "profile_config": profile.raw,
        "dataset_path": _portable_repo_path(_repo_path(experiment["dataset_path"])),
        "dataset_spec": dataset_spec,
        "budget": int(experiment["budget"]),
        "batch_size": int(experiment["batch_size"]),
        "vary": sorted(
            experiment.get(
                "vary",
                defaults.get("vary", ["model", "optimizer", "loss"]),
            )
        ),
        "fixed": deepcopy(experiment.get("fixed", defaults.get("fixed", {}))),
        "external_evaluation": {
            "excluded_candidate_sets": [
                _portable_repo_path(path) for path in excluded_sets
            ],
            "excluded_fingerprints_sha256": sorted(excluded_fingerprints),
            **({"exclusion_sources": exclusion_sources} if wide_contract else {}),
        },
        "stratify_by": list(
            experiment.get(
                "stratify_by",
                defaults.get("stratify_by", ["optimizer.type"]),
            )
        ),
        "num_rows": int(experiment.get("num_rows", defaults.get("num_rows", 1000))),
        "train_rows": int(
            experiment.get("train_rows", defaults.get("train_rows", 900))
        ),
        "reserve_rows": int(
            experiment.get("reserve_rows", defaults.get("reserve_rows", 50))
        ),
        "sampling_seed": int(
            experiment.get("sampling_seed", defaults.get("sampling_seed", 0))
        ),
        "split_seed": int(
            experiment.get("split_seed", defaults.get("split_seed", 1))
        ),
        "ground_truth": {
            "n_seeds": int(gt_experiment.get("n_seeds", gt_defaults.get("n_seeds", 10))),
            "base_seed": int(
                gt_experiment.get("base_seed", gt_defaults.get("base_seed", 0))
            ),
            "fail_threshold_mode": str(
                gt_experiment.get(
                    "fail_threshold_mode",
                    gt_defaults.get("fail_threshold_mode", "finite_only"),
                )
            ),
        },
    }
    if wide_contract:
        config["phase"] = phase
        config["group_labels"] = group_labels
    return config


def _validate_config(config: dict[str, Any], profile: Profile) -> None:
    num_rows = config["num_rows"]
    train_rows = config["train_rows"]
    reserve_rows = config["reserve_rows"]
    if num_rows <= 1:
        raise ValueError("num_rows must be greater than one")
    if not 0 < train_rows < num_rows:
        raise ValueError("train_rows must be strictly between zero and num_rows")
    if reserve_rows < 0:
        raise ValueError("reserve_rows must be non-negative")
    if config["ground_truth"]["n_seeds"] < 1:
        raise ValueError("ground_truth.n_seeds must be positive")
    if config["ground_truth"]["fail_threshold_mode"] not in {
        "finite_only",
        "profile",
    }:
        raise ValueError("fail_threshold_mode must be 'finite_only' or 'profile'")
    if "phase" in config:
        if not config["phase"]:
            raise ValueError("phase must be non-empty")
        if not config["group_labels"]:
            raise ValueError("group_labels must be non-empty")
        for name, value in config["group_labels"].items():
            if not isinstance(name, str) or not name:
                raise TypeError("group label names must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise TypeError(f"group label {name!r} must be a finite scalar")

    dataset_spec = config["dataset_spec"]
    family = dataset_spec["family"]
    if family not in profile.pools["dataset_families"]:
        raise ValueError(f"Dataset family {family!r} is not in profile {profile.name}")
    if config["budget"] not in profile.budget_values:
        raise ValueError(
            f"Budget {config['budget']} is not in profile {profile.name}: "
            f"{profile.budget_values}"
        )
    if config["budget"] % config["batch_size"] != 0:
        raise ValueError("budget must be divisible by batch_size")
    if config["batch_size"] not in profile.optimizer_grids["batch_size"]:
        raise ValueError("batch_size is not in the profile optimizer grid")

    varying_axes = parse_varying_axes(config["vary"])
    fixed = config["fixed"]
    overlap = varying_axes & fixed.keys()
    if overlap:
        raise ValueError(f"Axes cannot be both varying and fixed: {sorted(overlap)}")
    for axis in {"model", "optimizer", "loss"} - varying_axes:
        if axis not in fixed:
            continue
        if not isinstance(fixed[axis], dict):
            raise TypeError(f"fixed.{axis} must be an object")
    if "loss" in fixed:
        loss_id = fixed["loss"].get("loss_id")
        if loss_id not in profile.pools["losses"][family]:
            raise ValueError(f"Loss {loss_id!r} is not allowed for {family}")


def load_plan(path: Path) -> tuple[dict[str, Any], Profile, Path]:
    plan = read_json(path)
    profile = load_profile(str(plan.get("profile", "v1")))
    output_root = _repo_path(plan.get("output_root", "data/meta_model/setting_to_loss"))
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Plan must contain a non-empty experiments list")
    names = [str(experiment.get("experiment_id", "")) for experiment in experiments]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("experiment_id values must be present and unique")
    return plan, profile, output_root


def _shared_fixed_values(
    *,
    config: dict[str, Any],
    profile: Profile,
    rng: random.Random,
) -> dict[str, Any]:
    family = config["dataset_spec"]["family"]
    dataset_params = config["dataset_spec"]["params"]
    varying_axes = parse_varying_axes(config["vary"])
    fixed = deepcopy(config["fixed"])
    fixed["batch_size"] = config["batch_size"]
    fixed["_dataset_params"] = dataset_params

    if "model" not in varying_axes and "model" not in fixed:
        fixed["model"] = sample_model(
            profile,
            rng,
            family=family,
            dataset_params=dataset_params,
        )
    if "optimizer" not in varying_axes and "optimizer" not in fixed:
        fixed["optimizer"] = sample_optimizer(profile, rng)
    if "loss" not in varying_axes and "loss" not in fixed:
        fixed["loss"] = sample_loss(profile, family, rng)
    return fixed


def _sample_records(config: dict[str, Any], profile: Profile) -> list[dict[str, Any]]:
    rng = random.Random(config["sampling_seed"])
    fixed = _shared_fixed_values(config=config, profile=profile, rng=rng)
    dataset_spec = config["dataset_spec"]
    desired = config["num_rows"] + config["reserve_rows"]
    seen: set[str] = set(
        config["external_evaluation"]["excluded_fingerprints_sha256"]
    )
    records: list[dict[str, Any]] = []
    max_attempts = max(desired * 100, 1000)

    for _ in range(max_attempts):
        if len(records) >= desired:
            break
        spec = sample_candidate(
            profile,
            dataset_id=dataset_spec["dataset_id"],
            family=dataset_spec["family"],
            budget=config["budget"],
            rng=rng,
            fixed=deepcopy(fixed),
        )
        fingerprint = full_candidate_fingerprint(spec)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        record = {
            "sampling_index": len(records),
            "fingerprint": fingerprint,
            "candidate_id_short": spec["candidate_id"],
            "artifact_dir": f"candidates/{spec['candidate_id']}__{fingerprint[:16]}",
            "stratum": split_stratum(spec, config["stratify_by"]),
            "spec": spec,
        }
        if "group_labels" in config:
            record["group_labels"] = deepcopy(config["group_labels"])
        records.append(record)
    if len(records) != desired:
        raise RuntimeError(
            f"Could only sample {len(records)} of {desired} unique full settings"
        )
    return assign_pre_execution_splits(
        records,
        num_rows=config["num_rows"],
        train_rows=config["train_rows"],
        seed=config["split_seed"],
    )


def prepare_experiment(
    config: dict[str, Any],
    profile: Profile,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    experiment_dir = output_root / config["experiment_id"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    sampling_path = experiment_dir / "sampling_manifest.json"
    config_hash = sha256_json(config)

    if sampling_path.is_file():
        sampling_manifest = read_json(sampling_path)
        if sampling_manifest.get("config_sha256") != config_hash:
            raise ValueError(
                f"Existing {sampling_path} was built from a different config; "
                "choose another output_root or remove that tool-owned experiment directory"
            )
        records = sampling_manifest["records"]
    else:
        records = _sample_records(config, profile)
        sampling_manifest = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config_sha256": config_hash,
            "config": config,
            "split_assigned_before_ground_truth": True,
            "full_fingerprint_used_for_identity": True,
            "created_at": _utc_now(),
            "records": records,
        }
        write_json(sampling_path, sampling_manifest)

    for record in records:
        spec = record["spec"]
        if full_candidate_fingerprint(spec) != record["fingerprint"]:
            raise ValueError(f"Fingerprint mismatch in {sampling_path}")
        candidate_dir = experiment_dir / record["artifact_dir"]
        model_family = get_model_type(spec["model"]["type"])
        write_candidate(spec, candidate_dir, model_family)

    short_counts = Counter(record["candidate_id_short"] for record in records)
    collision_count = sum(count - 1 for count in short_counts.values() if count > 1)
    print(
        f"[prepare] {config['experiment_id']}: {len(records)} unique settings "
        f"({config['num_rows']} primary + {config['reserve_rows']} reserve), "
        f"short-id collisions={collision_count}, "
        "external candidates excluded="
        f"{len(config['external_evaluation']['excluded_fingerprints_sha256'])}",
        flush=True,
    )
    return experiment_dir, sampling_manifest


def _clone_profile_for_gt(
    profile_name: str,
    *,
    n_seeds: int,
    base_seed: int,
) -> Profile:
    profile = deepcopy(load_profile(profile_name))
    profile.ground_truth = deepcopy(profile.ground_truth)
    profile.ground_truth["n_seeds"] = n_seeds
    profile.ground_truth["base_seed"] = base_seed
    return profile


def _gt_worker(
    *,
    candidate_dir_string: str,
    dataset_path_string: str,
    profile_name: str,
    n_seeds: int,
    base_seed: int,
    fail_threshold_mode: str,
    expected_profile_config: dict[str, Any],
    gt_config_sha256: str,
    execution_context_sha256: str,
    execution_inputs_sha256: str,
) -> dict[str, Any]:
    _pin_thread_environment()
    import torch

    torch.set_num_threads(1)
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    candidate_dir = Path(candidate_dir_string)
    dataset_path = Path(dataset_path_string)
    profile = _clone_profile_for_gt(
        profile_name,
        n_seeds=n_seeds,
        base_seed=base_seed,
    )
    override = math.inf if fail_threshold_mode == "finite_only" else None
    marker_path = candidate_dir / "results" / "meta_model_gt.json"
    try:
        if profile.raw != expected_profile_config:
            raise RuntimeError(
                "profile changed after the experiment config was frozen; restart GT"
            )
        observed_inputs = _candidate_execution_inputs_sha256(
            candidate_dir,
            execution_context_sha256,
        )
        if observed_inputs != execution_inputs_sha256:
            raise RuntimeError(
                "candidate execution inputs changed after scheduling: "
                f"expected {execution_inputs_sha256}, got {observed_inputs}"
            )
        summary = run_ground_truth(
            candidate_dir,
            profile,
            dataset_path,
            fail_threshold_override=override,
        )
        observed_inputs = _candidate_execution_inputs_sha256(
            candidate_dir,
            execution_context_sha256,
        )
        if observed_inputs != execution_inputs_sha256:
            raise RuntimeError(
                "candidate execution inputs changed during GT: "
                f"expected {execution_inputs_sha256}, got {observed_inputs}"
            )
        summary_path = candidate_dir / "results" / "summary.json"
        curves_path = candidate_dir / "results" / "curves.npz"
        marker = {
            "schema_version": "1.0",
            "status": "ok",
            "gt_config_sha256": gt_config_sha256,
            "execution_context_sha256": execution_context_sha256,
            "execution_inputs_sha256": execution_inputs_sha256,
            "completed_at": _utc_now(),
            "summary_sha256": sha256_file(summary_path),
            "curves_sha256": sha256_file(curves_path),
        }
        write_json(marker_path, marker)
        return {
            "artifact": candidate_dir.name,
            "status": "ok",
            "failed_seeds": int(summary["failed_seeds"]),
            "excluded": bool(summary["excluded"]),
        }
    except Exception as error:  # noqa: BLE001
        marker = {
            "schema_version": "1.0",
            "status": "error",
            "gt_config_sha256": gt_config_sha256,
            "execution_context_sha256": execution_context_sha256,
            "execution_inputs_sha256": execution_inputs_sha256,
            "completed_at": _utc_now(),
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(marker_path, marker)
        return {
            "artifact": candidate_dir.name,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _gt_config_hash(config: dict[str, Any]) -> str:
    return sha256_json(
        {
            "profile": config["profile"],
            "profile_config": config["profile_config"],
            "dataset_spec": config["dataset_spec"],
            **config["ground_truth"],
        }
    )


def _pipeline_source_manifest() -> dict[str, str]:
    source_root = ROOT / "src" / "architecture_iq"
    source_paths = [
        *sorted(source_root.rglob("*.py")),
        Path(__file__).resolve(),
        Path(__file__).with_name("core.py").resolve(),
    ]
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
    }


def _execution_context(
    config: dict[str, Any],
    dataset_path: Path,
) -> tuple[dict[str, Any], str]:
    dataset_files = {}
    for logical_name, relative_path in sorted(config["dataset_spec"]["files"].items()):
        artifact_path = dataset_path / str(relative_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing dataset artifact: {artifact_path}")
        dataset_files[logical_name] = {
            "path": str(relative_path),
            "sha256": sha256_file(artifact_path),
        }
    import numpy as np
    import torch

    context = {
        "gt_config_sha256": _gt_config_hash(config),
        "dataset_artifacts": dataset_files,
        "pipeline_source_files": _pipeline_source_manifest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    return context, sha256_json(context)


def _candidate_execution_inputs_sha256(
    candidate_dir: Path,
    execution_context_sha256: str,
) -> str:
    spec = read_json(candidate_dir / "candidate_spec.json")
    paths = {"candidate_spec": candidate_dir / "candidate_spec.json"}
    for logical_name, relative_path in sorted(spec["files"].items()):
        paths[logical_name] = candidate_dir / str(relative_path)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing candidate execution inputs: {missing}")
    return sha256_json(
        {
            "execution_context_sha256": execution_context_sha256,
            "candidate_files": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in sorted(paths.items())
            },
        }
    )


def _gt_marker_state(
    candidate_dir: Path,
    *,
    config: dict[str, Any],
    gt_config_sha256: str,
    execution_context_sha256: str,
    execution_inputs_sha256: str,
) -> str | None:
    summary_path = candidate_dir / "results" / "summary.json"
    curves_path = candidate_dir / "results" / "curves.npz"
    marker_path = candidate_dir / "results" / "meta_model_gt.json"
    if not marker_path.is_file():
        return None
    try:
        marker = read_json(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        marker.get("gt_config_sha256") != gt_config_sha256
        or marker.get("execution_context_sha256") != execution_context_sha256
        or marker.get("execution_inputs_sha256") != execution_inputs_sha256
    ):
        return None
    status = marker.get("status")
    if status == "error":
        return "error"
    if status != "ok" or not summary_path.is_file() or not curves_path.is_file():
        return None
    try:
        summary = read_json(summary_path)
        spec = read_json(candidate_dir / "candidate_spec.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_summary = {
        "candidate_id": spec["candidate_id"],
        "selection_metric": config["dataset_spec"]["selection_metric"],
        "execution": "candidate_py_files",
        "n_seeds": config["ground_truth"]["n_seeds"],
        "base_seed": config["ground_truth"]["base_seed"],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        return None
    if marker.get("summary_sha256") != sha256_file(summary_path):
        return None
    if marker.get("curves_sha256") != sha256_file(curves_path):
        return None
    return "ok"


def _gt_complete(
    candidate_dir: Path,
    *,
    config: dict[str, Any],
    gt_config_sha256: str,
    execution_context_sha256: str,
    execution_inputs_sha256: str,
) -> bool:
    return (
        _gt_marker_state(
            candidate_dir,
            config=config,
            gt_config_sha256=gt_config_sha256,
            execution_context_sha256=execution_context_sha256,
            execution_inputs_sha256=execution_inputs_sha256,
        )
        == "ok"
    )


def run_ground_truth_for_experiment(
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    sampling_manifest: dict[str, Any],
    workers: int,
    limit: int | None,
) -> dict[str, int]:
    gt_config = config["ground_truth"]
    gt_hash = _gt_config_hash(config)
    dataset_path = _repo_path(config["dataset_path"])
    execution_context, context_hash = _execution_context(config, dataset_path)
    work = []
    for record in sampling_manifest["records"]:
        candidate_dir = experiment_dir / record["artifact_dir"]
        inputs_hash = _candidate_execution_inputs_sha256(candidate_dir, context_hash)
        work.append((candidate_dir, inputs_hash))
    pending = [
        (candidate_dir, inputs_hash)
        for candidate_dir, inputs_hash in work
        if not _gt_complete(
            candidate_dir,
            config=config,
            gt_config_sha256=gt_hash,
            execution_context_sha256=context_hash,
            execution_inputs_sha256=inputs_hash,
        )
    ]
    if limit is not None:
        pending = pending[:limit]

    total_records = len(sampling_manifest["records"])
    already_done = sum(
        _gt_complete(
            candidate_dir,
            config=config,
            gt_config_sha256=gt_hash,
            execution_context_sha256=context_hash,
            execution_inputs_sha256=inputs_hash,
        )
        for candidate_dir, inputs_hash in work
    )
    print(
        f"[gt] {config['experiment_id']}: done={already_done}/{total_records}, "
        f"launching={len(pending)}, workers={workers}",
        flush=True,
    )
    counts: Counter[str] = Counter()
    if not pending:
        return dict(counts)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _gt_worker,
                candidate_dir_string=str(candidate_dir),
                dataset_path_string=str(dataset_path),
                profile_name=config["profile"],
                n_seeds=gt_config["n_seeds"],
                base_seed=gt_config["base_seed"],
                fail_threshold_mode=gt_config["fail_threshold_mode"],
                expected_profile_config=config["profile_config"],
                gt_config_sha256=gt_hash,
                execution_context_sha256=context_hash,
                execution_inputs_sha256=inputs_hash,
            )
            for candidate_dir, inputs_hash in pending
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "error":
                print(
                    f"[gt {completed}/{len(pending)}] {result['artifact']}: "
                    f"ERROR {result['error']}",
                    flush=True,
                )
            elif completed % 10 == 0 or completed == len(pending):
                print(
                    f"[gt {completed}/{len(pending)}] {result['artifact']}: "
                    f"failed_seeds={result['failed_seeds']}",
                    flush=True,
                )
    return dict(counts)


def export_experiment(
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    sampling_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    attempts = []
    gt_hash = _gt_config_hash(config)
    dataset_path = _repo_path(config["dataset_path"])
    execution_context, context_hash = _execution_context(config, dataset_path)
    for record in sampling_manifest["records"]:
        candidate_dir = experiment_dir / record["artifact_dir"]
        inputs_hash = _candidate_execution_inputs_sha256(candidate_dir, context_hash)
        marker_state = _gt_marker_state(
            candidate_dir,
            config=config,
            gt_config_sha256=gt_hash,
            execution_context_sha256=context_hash,
            execution_inputs_sha256=inputs_hash,
        )
        if marker_state is None:
            raise RuntimeError(
                f"Unverified or stale GT for {candidate_dir}; run --stage gt first"
            )
        row = build_attempt_row(
            experiment_id=config["experiment_id"],
            profile_name=config["profile"],
            dataset_spec=config["dataset_spec"],
            candidate_dir=candidate_dir,
            split=record["split"],
            selection_role=record["selection_role"],
            stratum=record["stratum"],
            relative_to=output_root,
            include_summary=marker_state == "ok",
        )
        row["sampling_index"] = int(record["sampling_index"])
        if "group_labels" in record:
            row["group_labels"] = deepcopy(record["group_labels"])
        attempts.append(row)

    validation_rows = config["num_rows"] - config["train_rows"]
    selected, replacements = select_usable_rows(
        attempts,
        train_rows=config["train_rows"],
        validation_rows=validation_rows,
    )
    selected.sort(key=lambda row: int(row["sampling_index"]))
    train = [row for row in selected if row["split"] == "train"]
    validation = [row for row in selected if row["split"] == "validation"]

    _write_jsonl(experiment_dir / "attempts.jsonl", attempts)
    _write_jsonl(experiment_dir / "all.jsonl", selected)
    _write_jsonl(experiment_dir / "train.jsonl", train)
    _write_jsonl(experiment_dir / "validation.jsonl", validation)
    feature_schema = build_feature_schema(train)
    feature_schema["fit_split"] = "train"
    write_json(experiment_dir / "feature_schema.json", feature_schema)

    short_id_groups: dict[str, set[str]] = defaultdict(set)
    for row in attempts:
        short_id_groups[row["candidate_id_short"]].add(
            row["example_fingerprint_sha256"]
        )
    short_collisions = {
        key: sorted(values) for key, values in short_id_groups.items() if len(values) > 1
    }
    raw_targets = [float(row["target"]["mean_loss"]) for row in selected]
    log_targets = [
        float(row["target"]["log_mean_loss"])
        for row in selected
        if row["target"]["log_mean_loss"] is not None
    ]
    manifest = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "config_sha256": sampling_manifest["config_sha256"],
        "config": config,
        "created_at": sampling_manifest["created_at"],
        "exported_at": _utc_now(),
        "unit": "one independently sampled and executed candidate setting",
        "split_policy": {
            "assigned_before_ground_truth": True,
            "train": len(train),
            "validation": len(validation),
            "validation_role": "locked_holdout; tune or cross-validate only within train",
            "stratify_by": config["stratify_by"],
            "full_sha256_identity": True,
            **(
                {
                    "group_labels_frozen_before_ground_truth": True,
                    "group_labels": config["group_labels"],
                }
                if "group_labels" in config
                else {}
            ),
        },
        "external_evaluation": {
            **config["external_evaluation"],
            "overlap_with_selected": sum(
                row["example_fingerprint_sha256"]
                in set(config["external_evaluation"]["excluded_fingerprints_sha256"])
                for row in selected
            ),
        },
        "ground_truth": {
            **config["ground_truth"],
            "config_sha256": gt_hash,
            "verified_successful_attempts": sum(
                row["provenance"]["execution"] == "candidate_py_files"
                for row in attempts
            ),
            "total_attempts": len(attempts),
            "execution_context": execution_context,
            "execution_context_sha256": context_hash,
            "canonical_path": "write_candidate -> run_ground_truth -> results/summary.json",
        },
        "attempts": {
            "total": len(attempts),
            "usable": sum(row["usable_for_regression"] for row in attempts),
            "unusable": sum(not row["usable_for_regression"] for row in attempts),
            "primary": sum(row["selection_role"] == "primary" for row in attempts),
            "reserve": sum(row["selection_role"] == "reserve" for row in attempts),
            "replacements": replacements,
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
        "files": {
            "attempts": "attempts.jsonl",
            "all": "all.jsonl",
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "feature_schema": "feature_schema.json",
            "sampling_manifest": "sampling_manifest.json",
        },
    }
    write_json(experiment_dir / "manifest.json", manifest)
    print(
        f"[export] {config['experiment_id']}: train={len(train)}, "
        f"validation={len(validation)}, replacements={len(replacements)}, "
        f"benchmark_eligible={manifest['selected']['benchmark_eligible']}",
        flush=True,
    )
    return manifest


def _selected_experiments(
    plan: dict[str, Any],
    requested: set[str] | None,
    requested_phases: set[str] | None = None,
) -> list[dict[str, Any]]:
    experiments = plan["experiments"]
    defaults = plan.get("defaults", {})
    known_phases = {
        str(experiment.get("phase", defaults.get("phase", "default")))
        for experiment in experiments
    }
    if requested_phases:
        missing_phases = requested_phases - known_phases
        if missing_phases:
            raise ValueError(f"Unknown phase values: {sorted(missing_phases)}")
        experiments = [
            experiment
            for experiment in experiments
            if str(experiment.get("phase", defaults.get("phase", "default")))
            in requested_phases
        ]
    if requested is None:
        return experiments
    selected = [
        experiment
        for experiment in experiments
        if str(experiment["experiment_id"]) in requested
    ]
    missing = requested - {str(experiment["experiment_id"]) for experiment in selected}
    if missing:
        raise ValueError(f"Unknown experiment_id values: {sorted(missing)}")
    return selected


def build_from_plan(
    *,
    plan_path: Path,
    stage: str,
    workers: int,
    limit_per_experiment: int | None = None,
    requested_experiments: set[str] | None = None,
    requested_phases: set[str] | None = None,
) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    if workers < 1:
        raise ValueError("workers must be positive")
    ensure_registries()
    plan, profile, output_root = load_plan(plan_path)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    for experiment in _selected_experiments(
        plan,
        requested_experiments,
        requested_phases,
    ):
        dataset_path = _repo_path(experiment["dataset_path"])
        dataset_spec_path = dataset_path / "dataset_spec.json"
        if not dataset_spec_path.is_file():
            raise FileNotFoundError(f"Missing dataset spec: {dataset_spec_path}")
        dataset_spec = read_json(dataset_spec_path)
        config = _experiment_config(plan, experiment, profile, dataset_spec)
        _validate_config(config, profile)
        get_dataset_family(dataset_spec["family"]).load_tensors(dataset_path)

        experiment_dir, sampling_manifest = prepare_experiment(
            config,
            profile,
            output_root,
        )
        gt_counts: dict[str, int] | None = None
        manifest: dict[str, Any] | None = None
        if stage in {"gt", "all"}:
            gt_counts = run_ground_truth_for_experiment(
                config=config,
                experiment_dir=experiment_dir,
                sampling_manifest=sampling_manifest,
                workers=workers,
                limit=limit_per_experiment,
            )
        if stage in {"export", "all"} and limit_per_experiment is None:
            manifest = export_experiment(
                config=config,
                experiment_dir=experiment_dir,
                sampling_manifest=sampling_manifest,
                output_root=output_root,
            )
        results.append(
            {
                "experiment_id": config["experiment_id"],
                "path": _portable_repo_path(experiment_dir),
                "gt_counts_this_invocation": gt_counts,
                "exported": manifest is not None,
            }
        )

    run_manifest = {
        "schema_version": "1.0",
        "plan": _portable_repo_path(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "stage": stage,
        "requested_phases": sorted(requested_phases) if requested_phases else None,
        "updated_at": _utc_now(),
        "experiments": results,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel GT worker processes.",
    )
    parser.add_argument(
        "--limit-per-experiment",
        type=int,
        help="Run only this many pending GT candidates per experiment; disables export.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Only run this experiment_id (repeatable).",
    )
    parser.add_argument(
        "--phase",
        action="append",
        help="Only run experiments in this phase (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = build_from_plan(
            plan_path=args.plan.resolve(),
            stage=args.stage,
            workers=args.workers,
            limit_per_experiment=args.limit_per_experiment,
            requested_experiments=set(args.experiment) if args.experiment else None,
            requested_phases=set(args.phase) if args.phase else None,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
