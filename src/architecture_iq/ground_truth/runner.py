from __future__ import annotations

import inspect
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from architecture_iq.candidates.generator import write_candidate
from architecture_iq.profile import Profile, validate_execution_device
from architecture_iq.registry import get_dataset_family, get_model_type
from architecture_iq.runtime.loader import load_candidate_train
from architecture_iq.significance.validator import final_metric_key, mean_metric_key
from architecture_iq.util import git_commit_hash, write_json
from architecture_iq.paths import ROOT


ProgressCallback = Callable[[dict[str, Any]], None]

# Each generate-candidates worker process runs its own interpreter; without a
# clamp every process defaults torch to all cores and N parallel workers
# oversubscribe OpenMP into a spin deadlock (observed with 8 workers).
# Override with ARCHITECTURE_IQ_TORCH_THREADS if the host needs a different cap.
DEFAULT_TORCH_THREADS = 8


def _clamp_process_torch_threads() -> int:
    raw = os.environ.get("ARCHITECTURE_IQ_TORCH_THREADS")
    try:
        threads = int(raw) if raw is not None else DEFAULT_TORCH_THREADS
    except ValueError as exc:
        raise ValueError(
            "ARCHITECTURE_IQ_TORCH_THREADS must be a positive integer"
        ) from exc
    if threads <= 0:
        raise ValueError("ARCHITECTURE_IQ_TORCH_THREADS must be a positive integer")
    torch.set_num_threads(threads)
    return threads


def _emit_progress(callback: ProgressCallback | None, event: dict[str, Any]) -> None:
    """Report optional UI progress without affecting ground-truth execution."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        # Progress is observational; a UI refresh failure must not invalidate GT.
        return


def _sync_candidate_files(candidate_path: Path, spec: dict[str, Any]) -> None:
    """Rewrite on-disk .py files from spec so execution matches candidate_spec.json."""
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    model_family = get_model_type(spec["model"]["type"])
    write_candidate(spec, candidate_path, model_family)


def _resolve_execution_device(candidate_spec: dict[str, Any], profile: Profile) -> torch.device:
    requested = validate_execution_device(
        str(candidate_spec.get("execution", {}).get("device", "cpu"))
    )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for this candidate but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(requested)
def _seed_parallelism_config(
    device: torch.device,
    n_seeds: int,
    progress_callback: ProgressCallback | None,
) -> tuple[int, int | None]:
    """Return opt-in CPU seed workers and their per-worker Torch threads."""
    if device.type != "cpu" or progress_callback is not None:
        return 1, None
    raw_workers = os.environ.get("ARCHITECTURE_IQ_SEED_WORKERS")
    if raw_workers is None:
        return 1, None
    try:
        requested_workers = int(raw_workers)
    except ValueError as exc:
        raise ValueError("ARCHITECTURE_IQ_SEED_WORKERS must be a positive integer") from exc
    if requested_workers <= 0:
        raise ValueError("ARCHITECTURE_IQ_SEED_WORKERS must be a positive integer")
    if requested_workers == 1:
        return 1, None
    raw_threads = os.environ.get("ARCHITECTURE_IQ_SEED_TORCH_THREADS", "1")
    try:
        torch_threads = int(raw_threads)
    except ValueError as exc:
        raise ValueError(
            "ARCHITECTURE_IQ_SEED_TORCH_THREADS must be a positive integer"
        ) from exc
    if torch_threads <= 0:
        raise ValueError("ARCHITECTURE_IQ_SEED_TORCH_THREADS must be a positive integer")
    return min(requested_workers, n_seeds), torch_threads


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
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_mod = load_candidate_train(candidate_path)
    if not hasattr(train_mod, "train_and_eval"):
        raise RuntimeError(
            f"{candidate_path}/train.py must define train_and_eval(); regenerate the candidate"
        )

    kwargs = {
        "steps": int(candidate_spec["budget"]["training_steps"]),
        "batch_size": int(candidate_spec["budget"]["batch_size"]),
        "seed": seed,
        "fail_threshold": fail_threshold,
    }
    parameters = inspect.signature(train_mod.train_and_eval).parameters
    supports_device = "device" in parameters
    if supports_device:
        kwargs["device"] = str(device)
    elif device.type != "cpu":
        raise RuntimeError(
            f"{candidate_path}/train.py predates device-aware execution; "
            "regenerate this CUDA candidate from candidate_spec.json"
        )
    if progress_callback is not None and "progress_callback" in parameters:
        kwargs["progress_callback"] = progress_callback
    result = train_mod.train_and_eval(
        train_x,
        train_y,
        test_x,
        test_y,
        **kwargs,
    )
    final_key = final_metric_key(selection_metric)
    if final_key not in result:
        raise KeyError(f"train_and_eval missing {final_key!r}")
    seed_result: dict[str, Any] = {
        "seed": seed,
        "failed": bool(result["failed"]),
        final_key: float(result[final_key]),
        "eval_samples": list(result["eval_samples"]),
        "step_metrics": list(result["step_metrics"]),
    }
    if "final_test_accuracy" in result:
        seed_result["final_test_accuracy"] = float(result["final_test_accuracy"])
    return seed_result
def _run_single_seed_cpu_worker(
    payload: tuple[str, str, dict[str, Any], int, float, str, int],
) -> dict[str, Any]:
    """Run one CPU seed in an isolated process without writing artifacts."""
    candidate_text, dataset_text, spec, seed, fail_threshold, selection_metric, torch_threads = payload
    torch.set_num_threads(torch_threads)
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    candidate_path = Path(candidate_text)
    dataset_path = Path(dataset_text)
    family = get_dataset_family(spec["family"])
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    return run_single_seed(
        candidate_path,
        spec,
        train_x,
        train_y,
        test_x,
        test_y,
        seed,
        fail_threshold,
        selection_metric=selection_metric,
        device=torch.device("cpu"),
    )


def run_ground_truth(
    candidate_path: Path,
    profile: Profile,
    dataset_path: Path | None = None,
    *,
    sync_files: bool = True,
    fail_threshold_override: float | None = None,
    progress_callback: ProgressCallback | None = None,
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
    seed_workers, seed_torch_threads = _seed_parallelism_config(
        device, n_seeds, progress_callback
    )
    if seed_workers == 1:
        # Seed-parallel workers set their own per-worker thread counts; the
        # plain path clamps so N parallel candidate processes cannot each
        # grab every core (see DEFAULT_TORCH_THREADS above).
        _clamp_process_torch_threads()

    training_steps = int(spec["budget"]["training_steps"])
    total_samples_seen = int(spec["budget"]["total_samples_seen"])
    if seed_workers > 1:
        assert seed_torch_threads is not None
        payloads = [
            (
                str(candidate_path),
                str(dataset_path),
                spec,
                base_seed + i,
                fail_threshold,
                selection_metric,
                seed_torch_threads,
            )
            for i in range(n_seeds)
        ]
        with ProcessPoolExecutor(max_workers=seed_workers) as executor:
            # executor.map preserves input order for paired seed significance.
            seed_results = list(executor.map(_run_single_seed_cpu_worker, payloads))
    else:
        for i in range(n_seeds):
            seed_index = i + 1
            seed = base_seed + i
            base_event = {
                "seed_index": seed_index,
                "n_seeds": n_seeds,
                "seed": seed,
                "training_steps": training_steps,
                "total_samples_seen": total_samples_seen,
                "selection_metric": selection_metric,
            }
            _emit_progress(progress_callback, {"phase": "seed_started", **base_event})

            def report_evaluation(
                event: dict[str, Any],
                *,
                context: dict[str, Any] = base_event,
            ) -> None:
                _emit_progress(progress_callback, {"phase": "evaluation", **context, **event})

            seed_result = run_single_seed(
                candidate_path,
                spec,
                train_x,
                train_y,
                test_x,
                test_y,
                seed,
                fail_threshold,
                selection_metric=selection_metric,
                device=device,
                progress_callback=report_evaluation,
            )
            seed_results.append(seed_result)
            _emit_progress(
                progress_callback,
                {
                    "phase": "seed_finished",
                    **base_event,
                    "failed": bool(seed_result["failed"]),
                    "metric": float(seed_result[final_key]),
                },
            )
    ok = [r for r in seed_results if not r["failed"]]
    failed_count = len(seed_results) - len(ok)
    finals = [r[final_key] for r in ok] or [float("inf")]
    accuracies = [r["final_test_accuracy"] for r in ok if "final_test_accuracy" in r]

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
        **(
            {
                "mean_test_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
                "std_test_accuracy": float(np.std(accuracies)) if accuracies else float("nan"),
            }
            if any("final_test_accuracy" in r for r in seed_results)
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
                **({"final_test_accuracy": r["final_test_accuracy"]} if "final_test_accuracy" in r else {}),
            }
            for r in seed_results
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": spec.get("execution", {}).get("device", "cpu"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "seed_workers": seed_workers,
            "torch_threads_per_seed": seed_torch_threads or torch.get_num_threads(),
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
