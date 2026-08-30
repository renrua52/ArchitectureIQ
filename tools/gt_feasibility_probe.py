"""GT feasibility probe for the v1-final 1000-question rebuild.

Measures per-candidate ground-truth wall time on this (CPU-only) host and
tunes candidate/seed parallelism for maximum throughput. Uses the standard
pipeline only: ``write_candidate`` + ``run_ground_truth``.

Modes:
  prepare   - ensure one dataset instance per family; write probe candidates
  worker    - run GT for a single candidate (used as a subprocess unit)
  phase-a   - sequential per-cell latency baseline (representative matrix)
  phase-b   - parallel throughput comparison across worker configs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROBE_ROOT = REPO / "data" / "gt_probe"
RESULTS_DIR = PROBE_ROOT / "results"

MLP_FAMILIES = ["univariate_regression", "multivariate_regression", "synthetic_tabular_classification"]
# Question share of the pre-v1.4 build, when synthetic_tabular_classification
# alone covered the xor / spiral / general_tabular buckets -- hence its weight 3.
# Under v1.4 those are three families with one bucket each; this probe still
# loads the default (v1) profile, which has no config for the two new families,
# so the weights are left as the record of what was actually timed.
FAMILY_WEIGHTS = {
    "univariate_regression": 1,
    "multivariate_regression": 1,
    "bigram_lm": 1,
    "synthetic_tabular_classification": 3,
}
BUDGET_TIERS = [2048, 4096, 8192, 16384]
FIXED_OPTIMIZER = {"type": "Adam", "lr": 1.0e-3, "weight_decay": 0, "betas": [0.9, 0.999]}
FAMILY_LOSS = {
    "univariate_regression": {"loss_id": "mse"},
    "multivariate_regression": {"loss_id": "mse"},
    "bigram_lm": {"loss_id": "cross_entropy"},
    "synthetic_tabular_classification": {"loss_id": "cross_entropy"},
}


def _imports():
    from architecture_iq.profile import Profile
    from architecture_iq.registry import ensure_registries, get_model_type
    from architecture_iq.util import read_json, write_json

    ensure_registries()
    return Profile, get_model_type, read_json, write_json


def _load_profile(n_seeds: int | None = None):
    Profile, _, _, _ = _imports()
    profile = Profile.load()
    if n_seeds is not None:
        profile = dataclasses.replace(
            profile, ground_truth={**profile.ground_truth, "n_seeds": n_seeds}
        )
    return profile


def _dataset_params(dataset_path: Path) -> dict:
    _, _, read_json, _ = _imports()
    return dict(read_json(dataset_path / "dataset_spec.json")["params"])


def ensure_datasets(profile) -> dict[str, Path]:
    """One dataset instance per family; create stabcls if missing."""
    from architecture_iq.datasets import create_dataset, list_dataset_instances

    datasets: dict[str, Path] = {}
    for family in FAMILY_WEIGHTS:
        instances = list_dataset_instances(family=family)
        if instances:
            datasets[family] = instances[0].path
            continue
        _, path = create_dataset(profile, seed=21, family_name=family)
        datasets[family] = path
        print(f"[prepare] created dataset for {family}: {path}")
    return datasets


def _mlp_spec(size: str, params: dict) -> dict:
    depth, width, residual, ln = (
        (1, 16, False, False) if size == "small" else (6, 256, True, True)
    )
    return {
        "type": "mlp",
        "depth": depth,
        "width": width,
        "residual": residual,
        "layer_norm": [ln] * depth,
        "activation": "relu",
        "input_dim": int(params.get("input_dim", 1)),
        "output_dim": int(params.get("num_classes", 1)),
    }


def _bigram_model_spec(kind: str, params: dict) -> dict:
    base = {
        "vocab_size": int(params["vocab_size"]),
        "context_length": int(params["context_length"]),
    }
    if kind == "transformer_small":
        return {"type": "transformer_lm", **base, "d_model": 32, "num_layers": 1, "num_heads": 2, "d_ff": 64}
    if kind == "transformer_large":
        return {"type": "transformer_lm", **base, "d_model": 128, "num_layers": 3, "num_heads": 4, "d_ff": 256}
    if kind == "gru_large":
        return {"type": "gru_lm", **base, "d_model": 64, "num_layers": 2, "layer_residual": False}
    raise ValueError(kind)


def phase_a_cells() -> list[dict]:
    cells = []
    for family in MLP_FAMILIES:
        for size in ("small", "large"):
            for budget in (2048, 16384):
                cells.append({"family": family, "model": f"mlp_{size}", "budget": budget})
    for kind in ("transformer_small", "transformer_large", "gru_large"):
        for budget in (2048, 16384):
            cells.append({"family": "bigram_lm", "model": kind, "budget": budget})
    return cells


def build_candidate(profile, datasets: dict[str, Path], cell: dict, out_root: Path) -> Path:
    """Materialize one probe candidate through the standard write path."""
    from architecture_iq.candidates.generator import build_candidate_spec, write_candidate

    _, get_model_type, _, _ = _imports()
    family = cell["family"]
    dataset_path = datasets[family]
    params = _dataset_params(dataset_path)
    if cell["model"].startswith("mlp"):
        model = _mlp_spec(cell["model"].split("_")[1], params)
    else:
        model = _bigram_model_spec(cell["model"], params)
    spec = build_candidate_spec(
        profile,
        dataset_id=dataset_path.name,
        family=family,
        budget=cell["budget"],
        batch_size=cell.get("batch_size", 32),
        model=model,
        optimizer=dict(FIXED_OPTIMIZER),
        loss=dict(FAMILY_LOSS[family]),
    )
    out_dir = out_root / f"{family}__{cell['model']}__b{cell['budget']}__{spec['candidate_id']}"
    if not (out_dir / "candidate_spec.json").exists():
        write_candidate(spec, out_dir, get_model_type(model["type"]))
    meta = {"family": family, "model": cell["model"], "budget": cell["budget"],
            "candidate_dir": out_dir.name}
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def build_mixed_batch(profile, datasets: dict[str, Path], n: int, out_root: Path, seed: int = 0) -> list[Path]:
    """Random candidates mirroring the planned production mix (post-KAN)."""
    from architecture_iq.candidates.generator import (
        build_candidate_spec, sample_loss, sample_model, sample_optimizer, write_candidate,
        valid_batch_sizes,
    )

    _, get_model_type, _, _ = _imports()
    rng = random.Random(seed)
    families = list(FAMILY_WEIGHTS)
    weights = [FAMILY_WEIGHTS[f] for f in families]
    paths = []
    for i in range(n):
        family = rng.choices(families, weights=weights)[0]
        dataset_path = datasets[family]
        params = _dataset_params(dataset_path)
        model_type = rng.choice(["transformer_lm", "gru_lm"]) if family == "bigram_lm" else "mlp"
        model = sample_model(profile, rng, family=family, dataset_params=params, model_type=model_type)
        budget = rng.choice(BUDGET_TIERS)
        batch_size = rng.choice(valid_batch_sizes(profile, budget))
        spec = build_candidate_spec(
            profile,
            dataset_id=dataset_path.name,
            family=family,
            budget=budget,
            batch_size=batch_size,
            model=model,
            optimizer=sample_optimizer(profile, rng),
            loss=sample_loss(profile, family, rng),
        )
        out_dir = out_root / f"mix{i:03d}__{family}__{spec['candidate_id']}"
        if not (out_dir / "candidate_spec.json").exists():
            write_candidate(spec, out_dir, get_model_type(model["type"]))
        paths.append(out_dir)
    return paths


def run_worker(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    from architecture_iq.ground_truth.runner import run_ground_truth

    profile = _load_profile(n_seeds=args.n_seeds)
    started = time.monotonic()
    summary = run_ground_truth(Path(args.candidate_path), profile, Path(args.dataset_path))
    wall = time.monotonic() - started
    out = {
        "candidate_path": args.candidate_path,
        "wall_s": round(wall, 3),
        "n_seeds": args.n_seeds,
        "threads": args.threads,
        "seed_workers": int(os.environ.get("ARCHITECTURE_IQ_SEED_WORKERS", "1")),
        "status": "ok",
    }
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_path).write_text(json.dumps(out))


def _spawn_worker(candidate: Path, dataset: Path, result: Path, *, n_seeds: int,
                  threads: int, seed_workers: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    if seed_workers > 1:
        env["ARCHITECTURE_IQ_SEED_WORKERS"] = str(seed_workers)
        env["ARCHITECTURE_IQ_SEED_TORCH_THREADS"] = str(threads)
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "worker",
        "--candidate-path", str(candidate),
        "--dataset-path", str(dataset),
        "--result-path", str(result),
        "--n-seeds", str(n_seeds),
        "--threads", str(threads),
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, cwd=REPO
    )


def _run_queue(jobs: list[dict], concurrency: int, timeout_s: float) -> list[dict]:
    """Run worker subprocesses with bounded concurrency; collect result dicts."""
    results: list[dict] = []
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, dict, float]] = []
    while pending or running:
        while pending and len(running) < concurrency:
            job = pending.pop(0)
            proc = _spawn_worker(
                job["candidate"], job["dataset"], job["result"],
                n_seeds=job["n_seeds"], threads=job["threads"],
                seed_workers=job["seed_workers"],
            )
            running.append((proc, job, time.monotonic()))
        time.sleep(2)
        still = []
        for proc, job, started in running:
            rc = proc.poll()
            if rc is None:
                if time.monotonic() - started > timeout_s:
                    proc.kill()
                    results.append({**job["meta"], "status": "timeout", "wall_s": timeout_s})
                else:
                    still.append((proc, job, started))
                continue
            if rc == 0 and job["result"].exists():
                results.append({**job["meta"], **json.loads(job["result"].read_text())})
            else:
                results.append({**job["meta"], "status": f"crash rc={rc}", "wall_s": None})
        running = still
    return results


def cmd_prepare(_args) -> None:
    profile = _load_profile()
    datasets = ensure_datasets(profile)
    cells_dir = PROBE_ROOT / "phase_a"
    for cell in phase_a_cells():
        path = build_candidate(profile, datasets, cell, cells_dir)
        print(f"[prepare] {path.name}")
    batch = build_mixed_batch(profile, datasets, 40, PROBE_ROOT / "phase_b")
    print(f"[prepare] mixed batch: {len(batch)} candidates")
    print("[prepare] datasets:", {k: str(v) for k, v in datasets.items()})


def cmd_phase_a(args) -> None:
    profile = _load_profile()
    datasets = ensure_datasets(profile)
    jobs = []
    for cell in phase_a_cells():
        cand = build_candidate(profile, datasets, cell, PROBE_ROOT / "phase_a")
        jobs.append({
            "candidate": cand, "dataset": datasets[cell["family"]],
            "result": RESULTS_DIR / f"phase_a__{cand.name}.json",
            "n_seeds": args.n_seeds, "threads": 1, "seed_workers": 1,
            "meta": {"family": cell["family"], "model": cell["model"], "budget": cell["budget"]},
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _run_queue(jobs, concurrency=1, timeout_s=args.timeout)
    out = RESULTS_DIR / "phase_a_summary.json"
    out.write_text(json.dumps(results, indent=1))
    for r in results:
        per_seed = round(r["wall_s"] / args.n_seeds, 2) if r.get("wall_s") else None
        print(f"{r['family']:38s} {r['model']:18s} b={r['budget']:6d} "
              f"status={r['status']:8s} wall={r.get('wall_s')}s per_seed={per_seed}s")
    print(f"summary -> {out}")


def cmd_phase_b(args) -> None:
    profile = _load_profile()
    datasets = ensure_datasets(profile)
    batch = sorted((PROBE_ROOT / "phase_b").glob("mix*"))
    if not batch:
        batch = build_mixed_batch(profile, datasets, 40, PROBE_ROOT / "phase_b")
    configs = {
        "C1_30x1x1": {"concurrency": 30, "seed_workers": 1},
        "C2_8x4x1": {"concurrency": 8, "seed_workers": 4},
        "C3_15x2x1": {"concurrency": 15, "seed_workers": 2},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        cfg = configs[name]
        jobs = [{
            "candidate": cand,
            "dataset": datasets[json.loads((cand / "candidate_spec.json").read_text())["family"]],
            "result": RESULTS_DIR / f"phase_b_{name}__{cand.name}.json",
            "n_seeds": args.n_seeds, "threads": 1,
            "seed_workers": cfg["seed_workers"],
            "meta": {"config": name, "candidate": cand.name},
        } for cand in batch]
        started = time.monotonic()
        results = _run_queue(jobs, concurrency=cfg["concurrency"], timeout_s=args.timeout)
        wall = time.monotonic() - started
        done = [r for r in results if r["status"] == "ok"]
        summary = {
            "config": name, **cfg, "n_candidates": len(jobs), "completed": len(done),
            "wall_s": round(wall, 1),
            "candidates_per_hour": round(len(done) / wall * 3600, 1) if wall else None,
            "failures": [r for r in results if r["status"] != "ok"],
        }
        (RESULTS_DIR / f"phase_b_{name}_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("prepare")
    worker = sub.add_parser("worker")
    worker.add_argument("--candidate-path", required=True)
    worker.add_argument("--dataset-path", required=True)
    worker.add_argument("--result-path", required=True)
    worker.add_argument("--n-seeds", type=int, default=10)
    worker.add_argument("--threads", type=int, default=1)
    pa = sub.add_parser("phase-a")
    pa.add_argument("--n-seeds", type=int, default=2)
    pa.add_argument("--timeout", type=float, default=1200)
    pb = sub.add_parser("phase-b")
    pb.add_argument("--n-seeds", type=int, default=10)
    pb.add_argument("--timeout", type=float, default=2400)
    pb.add_argument("--configs", nargs="+", default=["C1_30x1x1", "C2_8x4x1"],
                    choices=["C1_30x1x1", "C2_8x4x1", "C3_15x2x1"])
    args = parser.parse_args()
    {"prepare": cmd_prepare, "worker": run_worker,
     "phase-a": cmd_phase_a, "phase-b": cmd_phase_b}[args.mode](args)


if __name__ == "__main__":
    main()
