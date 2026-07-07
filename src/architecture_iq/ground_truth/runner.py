from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from architecture_iq.candidates.generator import write_candidate
from architecture_iq.profile import Profile
from architecture_iq.registry import get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from architecture_iq.util import git_commit_hash, write_json
from architecture_iq.paths import ROOT


def _sync_candidate_files(candidate_path: Path, spec: dict[str, Any]) -> None:
    """Rewrite on-disk .py files from spec so execution matches candidate_spec.json."""
    model_family = get_model_type(spec["model"]["type"])
    write_candidate(spec, candidate_path, model_family)


def run_single_seed(
    candidate_path: Path,
    candidate_spec: dict[str, Any],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    fail_threshold: float,
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
    )
    return {
        "seed": seed,
        "failed": bool(result["failed"]),
        "final_test_mse": float(result["final_test_mse"]),
        "eval_samples": list(result["eval_samples"]),
        "step_metrics": list(result["step_metrics"]),
    }


def run_ground_truth(
    candidate_path: Path,
    profile: Profile,
    dataset_path: Path | None = None,
    *,
    sync_files: bool = True,
) -> dict[str, Any]:
    from architecture_iq.util import read_json

    candidate_path = candidate_path.resolve()
    spec = read_json(candidate_path / "candidate_spec.json")
    if sync_files:
        _sync_candidate_files(candidate_path, spec)

    family_name = spec["family"]
    family = get_dataset_family(family_name)
    if dataset_path is None:
        dataset_path = candidate_path.parents[2]
    dataset_path = dataset_path.resolve()
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    sig_cfg = dataset_spec.get("significance", {})
    gt_cfg = profile.ground_truth
    fail_threshold = float(sig_cfg.get("fail_threshold", gt_cfg["fail_threshold"]))
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
            )
        )

    ok = [r for r in seed_results if not r["failed"]]
    failed_count = len(seed_results) - len(ok)
    finals = [r["final_test_mse"] for r in ok] or [float("inf")]

    max_len = max((len(r["step_metrics"]) for r in ok), default=0)
    curves = np.full((n_seeds, max_len), np.nan, dtype=np.float64)
    sample_axis: list[int] | None = None
    for i, r in enumerate(seed_results):
        if r["failed"]:
            continue
        curves[i, : len(r["step_metrics"])] = r["step_metrics"]
        if sample_axis is None:
            sample_axis = r["eval_samples"]

    summary = {
        "schema_version": profile.schema_version,
        "candidate_id": spec["candidate_id"],
        "selection_metric": dataset_spec["selection_metric"],
        "execution": "candidate_py_files",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "failed_seeds": failed_count,
        "excluded": failed_count >= int(profile.ground_truth["max_failed_seeds"]),
        "mean_test_mse": float(np.mean(finals)) if ok else float("inf"),
        "std_test_mse": float(np.std(finals)) if ok else float("inf"),
        "seed_results": [
            {"seed": r["seed"], "failed": r["failed"], "final_test_mse": r["final_test_mse"]}
            for r in seed_results
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(torch.device("cpu")),
            "git_commit": git_commit_hash(ROOT),
        },
    }

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
