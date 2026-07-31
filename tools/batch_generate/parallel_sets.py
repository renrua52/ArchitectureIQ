"""Parallel candidate-set generator for large-budget question batches.

Reuses the canonical pipeline (`sample_candidate_set_pool`, `write_candidate`,
`run_ground_truth`) but fans the ground-truth loop out across a process pool so
all CPU cores are used. Each worker pins torch to a single thread — these models
are tiny, so many single-threaded processes beat few multi-threaded ones.

This does NOT reimplement any generation logic; it only parallelizes the GT loop
that `generate_candidate_set` runs sequentially. Question assembly is left to the
CLI / `generate_questions` afterwards.

Usage: driven by a JSON plan on stdin or --plan file. See batch_plan.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any


def _pin_single_thread() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _gt_worker(candidate_dir_str: str, dataset_path_str: str, profile_name: str) -> tuple[str, str]:
    """Run ground truth for one already-written candidate. Returns (cid, status)."""
    _pin_single_thread()
    import torch

    torch.set_num_threads(1)
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries
    from architecture_iq.ground_truth.runner import run_ground_truth

    ensure_registries()
    profile = load_profile(profile_name)
    candidate_dir = Path(candidate_dir_str)
    dataset_path = Path(dataset_path_str)
    try:
        summary = run_ground_truth(candidate_dir, profile, dataset_path)
        status = "excluded" if summary.get("excluded") else (
            f"failed_seeds={summary['failed_seeds']}" if summary.get("failed_seeds") else "ok"
        )
        return candidate_dir.name, status
    except Exception as exc:  # noqa: BLE001
        return candidate_dir.name, f"ERROR: {type(exc).__name__}: {exc}"


def write_set_skeleton(plan_item: dict[str, Any], profile: Any) -> tuple[Path, Path, list[Path]]:
    """Sample specs and write all candidate .py files + set manifest (no GT yet).

    Returns (dataset_path, set_path, [candidate_dirs]).
    """
    from architecture_iq.candidates.sets import (
        make_set_name,
        sample_candidate_set_pool,
        write_set_manifest,
    )
    from architecture_iq.candidates.generator import write_candidate
    from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir

    dataset_path = Path(plan_item["dataset_path"]).resolve()
    dataset_spec = json.loads((dataset_path / "dataset_spec.json").read_text())
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]
    dataset_params = dataset_spec["params"]

    budget = int(plan_item["budget"])
    count = int(plan_item["count"])
    varying_axes = frozenset(plan_item["vary"])
    seed = int(plan_item["seed"])
    rng = random.Random(seed)

    specs = sample_candidate_set_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=None,
        dataset_params=dataset_params,
    )

    set_name = make_set_name(budget, varying_axes, salt=rng.randint(0, 2**31 - 1))
    set_path = candidate_set_dir(dataset_path, set_name)
    set_path.mkdir(parents=True, exist_ok=False)

    shared_record: dict[str, Any] = {}
    if specs:
        shared_record["batch_size"] = specs[0]["budget"]["batch_size"]
        if "model" not in varying_axes:
            shared_record["model"] = specs[0]["model"]
        if "optimizer" not in varying_axes:
            shared_record["optimizer"] = specs[0]["optimizer"]
        if "loss" not in varying_axes:
            shared_record["loss"] = specs[0]["loss"]

    write_set_manifest(
        set_path,
        set_name=set_name,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        fixed_shared=shared_record,
        seed=seed,
        profile=profile,
        dataset_id=dataset_id,
        family=family,
    )

    from architecture_iq.registry import get_model_type

    candidate_dirs: list[Path] = []
    for spec in specs:
        out = candidate_in_set_dir(set_path, spec["candidate_id"])
        model_family = get_model_type(spec["model"]["type"])
        write_candidate(spec, out, model_family)
        candidate_dirs.append(out)
    return dataset_path, set_path, candidate_dirs


def main() -> None:
    _pin_single_thread()
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON plan file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="v1")
    args = ap.parse_args()

    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    profile = load_profile(args.profile)

    plan = json.loads(Path(args.plan).read_text())
    items = plan["sets"]

    # Phase 1: write all skeletons (fast, sequential — sampling only).
    all_work: list[tuple[Path, Path]] = []  # (candidate_dir, dataset_path)
    set_index: list[dict[str, Any]] = []
    for item in items:
        try:
            dataset_path, set_path, cand_dirs = write_set_skeleton(item, profile)
        except Exception as exc:  # noqa: BLE001
            print(f"[skeleton FAILED] {item.get('label','')}: {exc}", flush=True)
            continue
        for cd in cand_dirs:
            all_work.append((cd, dataset_path))
        set_index.append(
            {
                "label": item.get("label", set_path.name),
                "dataset_path": str(dataset_path),
                "set_path": str(set_path),
                "n_candidates": len(cand_dirs),
                "vary": item["vary"],
                "budget": item["budget"],
            }
        )
        print(f"[skeleton] {item.get('label','')}: {len(cand_dirs)} candidates at {set_path}", flush=True)

    print(f"[gt] launching {len(all_work)} GT runs across {args.workers} workers", flush=True)

    # Phase 2: parallel GT across the full candidate pool.
    done = 0
    total = len(all_work)
    results: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_gt_worker, str(cd), str(dp), args.profile): cd.name
            for cd, dp in all_work
        }
        for fut in as_completed(futs):
            cid, status = fut.result()
            results[cid] = status
            done += 1
            if status != "ok" or done % 10 == 0 or done == total:
                print(f"[gt {done}/{total}] {cid}: {status}", flush=True)

    n_ok = sum(1 for v in results.values() if v == "ok")
    n_err = sum(1 for v in results.values() if v.startswith("ERROR"))
    n_excl = sum(1 for v in results.values() if v == "excluded")
    n_fail = sum(1 for v in results.values() if v.startswith("failed_seeds"))
    print(
        f"[done] ok={n_ok} excluded={n_excl} failed_seeds={n_fail} error={n_err}",
        flush=True,
    )

    out = {"sets": set_index, "gt_status": results}
    Path(plan.get("index_out", "tools/batch_generate/_last_run_index.json")).write_text(
        json.dumps(out, indent=2)
    )
    print("[index] written", flush=True)


if __name__ == "__main__":
    main()
