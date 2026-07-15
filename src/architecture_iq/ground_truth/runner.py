from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from architecture_iq.candidates.generator import write_candidate
from architecture_iq.profile import Profile, validate_execution_device
from architecture_iq.registry import get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from architecture_iq.significance.validator import final_metric_key, mean_metric_key
from architecture_iq.util import git_commit_hash, write_json
from architecture_iq.paths import ROOT


def _sync_candidate_files(candidate_path: Path, spec: dict[str, Any]) -> None:
    """Rewrite on-disk .py files from spec so execution matches candidate_spec.json."""
    model_family = get_model_type(spec["model"]["type"])
    write_candidate(spec, candidate_path, model_family)


def _resolve_execution_device(candidate_spec: dict[str, Any], profile: Profile) -> torch.device:
    requested = validate_execution_device(
        str(candidate_spec.get("execution", {}).get("device", profile.execution_device))
    )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for this candidate but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(requested)


def run_single_seed(
    candidate_path: Path,
    candidate_spec: dict[str, Any],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    fail_threshold: float,
    *,
    selection_metric: str,
    device: torch.device,
) -> dict[str, Any]:
    train_mod = load_candidate_train(candidate_path)
    if not hasattr(train_mod, "train_and_eval"):
        raise RuntimeError(
            f"{candidate_path}/train.py must define train_and_eval(); regenerate the candidate"
        )

    result = train_mod.train_and_eval(
        train_x,
        train_y,
        test_x,
        test_y,
        steps=int(candidate_spec["budget"]["training_steps"]),
        batch_size=int(candidate_spec["budget"]["batch_size"]),
        seed=seed,
        fail_threshold=fail_threshold,
        device=str(device),
    )
    final_key = final_metric_key(selection_metric)
    if final_key not in result:
        raise KeyError(f"train_and_eval missing {final_key!r}")
    return {
        "seed": seed,
        "failed": bool(result["failed"]),
        final_key: float(result[final_key]),
        "eval_samples": list(result["eval_samples"]),
        "step_metrics": list(result["step_metrics"]),
    }


def run_ground_truth(
    candidate_path: Path,
    profile: Profile,
    dataset_path: Path | None = None,
    *,
    sync_files: bool = True,
    fail_threshold_override: float | None = None,
) -> dict[str, Any]:
    from architecture_iq.util import read_json

    candidate_path = candidate_path.resolve()
    spec = read_json(candidate_path / "candidate_spec.json")
    device = _resolve_execution_device(spec, profile)
    if sync_files:
        _sync_candidate_files(candidate_path, spec)

    family_name = spec["family"]
    family = get_dataset_family(family_name)
    if dataset_path is None:
        dataset_path = candidate_path.parents[2]
    dataset_path = dataset_path.resolve()
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    selection_metric = dataset_spec["selection_metric"]
    final_key = final_metric_key(selection_metric)
    sig_cfg = dataset_spec.get("significance", {})
    gt_cfg = profile.ground_truth
    fail_threshold = (
        float(fail_threshold_override)
        if fail_threshold_override is not None
        else float(sig_cfg.get("fail_threshold", gt_cfg["fail_threshold"]))
    )
    batch_size = int(spec["budget"]["batch_size"])

    n_seeds = profile.n_seeds
    base_seed = profile.base_seed
    seed_results: list[dict[str, Any]] = []

    for i in range(n_seeds):
        seed_results.append(
            run_single_seed(
                candidate_path,
                spec,
                train_x,
                train_y,
                test_x,
                test_y,
                base_seed + i,
                fail_threshold,
                selection_metric=selection_metric,
                device=device,
            )
        )

    ok = [r for r in seed_results if not r["failed"]]
    failed_count = len(seed_results) - len(ok)
    finals = [r[final_key] for r in ok] or [float("inf")]

    max_len = max((len(r["step_metrics"]) for r in ok), default=0)
    curves = np.full((n_seeds, max_len), np.nan, dtype=np.float64)
    sample_axis: list[int] | None = None
    for i, r in enumerate(seed_results):
        if r["failed"]:
            continue
        curves[i, : len(r["step_metrics"])] = r["step_metrics"]
        if sample_axis is None:
            sample_axis = r["eval_samples"]

    mean_key = mean_metric_key(selection_metric)
    std_key = f"std_{selection_metric}"
    summary = {
        "schema_version": profile.schema_version,
        "candidate_id": spec["candidate_id"],
        "selection_metric": selection_metric,
        "execution": "candidate_py_files",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "failed_seeds": failed_count,
        "excluded": failed_count >= int(profile.ground_truth["max_failed_seeds"]),
        mean_key: float(np.mean(finals)) if ok else float("inf"),
        std_key: float(np.std(finals)) if ok else float("inf"),
        **(
            {
                "mean_test_mse": float(np.mean(finals)) if ok else float("inf"),
                "std_test_mse": float(np.std(finals)) if ok else float("inf"),
            }
            if selection_metric == "test_mse"
            else {}
        ),
        "seed_results": [
            {
                "seed": r["seed"],
                "failed": r["failed"],
                final_key: r[final_key],
                **(
                    {"final_test_mse": r[final_key], "mean_test_mse": r[final_key]}
                    if selection_metric == "test_mse"
                    else {}
                ),
            }
            for r in seed_results
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": spec.get("execution", {}).get("device", profile.execution_device),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "git_commit": git_commit_hash(ROOT),
        },
    }
    summary = {k: v for k, v in summary.items() if v is not None}

    results_dir = candidate_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "summary.json", summary)
    np.savez(
        results_dir / "curves.npz",
        curves=curves,
        samples=np.asarray(sample_axis or [], dtype=np.int64),
        batch_size=batch_size,
    )
    return summary
