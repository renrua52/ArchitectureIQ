"""V1-final (v1.1) parallel ground-truth driver.

Scans all candidate sets under ``data/datasets/`` and runs ``run_ground_truth``
for every candidate missing ``results/summary.json``. Reuses the canonical GT
path only; the only addition is process-level parallelism, tuned by the
feasibility probe (config C3: ~15 candidate workers x 2 seed workers, each
single torch thread -> ~2765 candidates/hour on this 32-core box).

Resume-safe: candidates with an existing ``results/summary.json`` are skipped,
so re-running after an interruption continues where it stopped.

Usage:
    python tools/benchmark_v1_gt.py --profile v1.1 --workers 15 --seed-workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architecture_iq.paths import DATA_DIR

STATUS_PATH = DATA_DIR / "benchmark_v1_gt_status.json"


def _pin_single_thread() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _gt_worker(
    candidate_dir_str: str,
    dataset_path_str: str,
    profile_name: str,
    seed_workers: int,
) -> tuple[str, str]:
    _pin_single_thread()
    os.environ["ARCHITECTURE_IQ_SEED_WORKERS"] = str(seed_workers)
    os.environ["ARCHITECTURE_IQ_SEED_TORCH_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    profile = load_profile(profile_name)
    candidate_dir = Path(candidate_dir_str)
    dataset_path = Path(dataset_path_str)
    try:
        summary = run_ground_truth(candidate_dir, profile, dataset_path)
        status = "excluded" if summary.get("excluded") else (
            f"failed_seeds={summary['failed_seeds']}"
            if summary.get("failed_seeds")
            else "ok"
        )
        return str(candidate_dir), status
    except Exception as exc:  # noqa: BLE001
        return str(candidate_dir), f"ERROR: {type(exc).__name__}: {exc}"


def _pending_candidates() -> list[tuple[str, str]]:
    """(candidate_dir, dataset_path) for candidates lacking GT results."""
    work: list[tuple[str, str]] = []
    datasets_root = DATA_DIR / "datasets"
    for set_manifest in sorted(datasets_root.glob("*/*/candidates/*/set.json")):
        set_path = set_manifest.parent
        dataset_path = set_path.parent.parent
        for cand in sorted(set_path.glob("c_*")):
            if (cand / "candidate_spec.json").is_file() and not (
                cand / "results" / "summary.json"
            ).is_file():
                work.append((str(cand), str(dataset_path)))
    return work


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v1.1")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--seed-workers", type=int, default=2)
    args = ap.parse_args()

    work = _pending_candidates()
    total = len(work)
    print(f"[gt] {total} candidates pending, workers={args.workers} "
          f"seed_workers={args.seed_workers}", flush=True)
    if total == 0:
        print("[done] nothing to do", flush=True)
        return

    t0 = time.monotonic()
    done = 0
    results: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_gt_worker, cd, dp, args.profile, args.seed_workers): cd
            for cd, dp in work
        }
        for fut in as_completed(futs):
            cd, status = fut.result()
            results[cd] = status
            done += 1
            if status != "ok" or done % 25 == 0 or done == total:
                rate = done / max(time.monotonic() - t0, 1e-9) * 3600
                print(
                    f"[gt {done}/{total} ~{rate:.0f}/h] {Path(cd).name}: {status}",
                    flush=True,
                )

    n_ok = sum(1 for v in results.values() if v == "ok")
    n_err = sum(1 for v in results.values() if v.startswith("ERROR"))
    n_excl = sum(1 for v in results.values() if v == "excluded")
    n_fail = sum(1 for v in results.values() if v.startswith("failed_seeds"))
    elapsed = time.monotonic() - t0
    STATUS_PATH.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "workers": args.workers,
                "seed_workers": args.seed_workers,
                "elapsed_seconds": round(elapsed, 1),
                "candidates": total,
                "ok": n_ok,
                "excluded": n_excl,
                "failed_seeds": n_fail,
                "error": n_err,
                "status": results,
            },
            indent=2,
        )
    )
    print(
        f"[done] ok={n_ok} excluded={n_excl} failed_seeds={n_fail} error={n_err} "
        f"in {elapsed / 3600:.2f}h; status -> {STATUS_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
