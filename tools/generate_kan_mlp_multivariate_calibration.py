"""Formal V2 KAN--MLP calibration for multivariate regression.

This is intentionally independent from ``generate_kan_mlp_demo.py``.  It
reuses the normal candidate renderer, ground-truth runner, and significance
validator, while writing calibration summaries under the multivariate phase-3
output directory.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.candidates.sets import make_set_name, write_set_manifest
from architecture_iq.ground_truth.runner import run_ground_truth, run_single_seed
from architecture_iq.models.kan import KanModelFamily  # noqa: F401 - registry side effect
from architecture_iq.models.mlp import MlpModelFamily  # noqa: F401 - registry side effect
from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir
from architecture_iq.profile import load_profile
from architecture_iq.prompts.renderer import write_prompt
from architecture_iq.questions.generator import generate_questions
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "kan_mlp_calibration_multivariate"


def _optimizer() -> dict[str, Any]:
    return {"type": "Adam", "lr": 0.003, "weight_decay": 0.0, "betas": [0.9, 0.999]}


def _loss() -> dict[str, Any]:
    return {"loss_id": "mse"}


def kan_spec(*, input_dim: int, depth: int, width: int, grid_size: int) -> dict[str, Any]:
    return {"type": "kan", "variant": "efficient_spline_v1", "input_dim": input_dim,
            "output_dim": 1, "depth": depth, "width": width, "grid_size": grid_size,
            "spline_order": 3, "grid_range": [-1.0, 1.0], "base_activation": "silu"}


def mlp_spec(*, input_dim: int, depth: int, width: int) -> dict[str, Any]:
    return {"type": "mlp", "input_dim": input_dim, "output_dim": 1, "depth": depth,
            "width": width, "residual": False, "layer_norm": [False] * depth,
            "activations": ["silu"] * depth, "leaky_relu_slope": 0.01}


def pair_specs(input_dim: int) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    return [
        ("kan_d1_w4_g3__mlp_d1_w11", kan_spec(input_dim=input_dim, depth=1, width=4, grid_size=3), mlp_spec(input_dim=input_dim, depth=1, width=11)),
        ("kan_d1_w8_g5__mlp_d1_w18", kan_spec(input_dim=input_dim, depth=1, width=8, grid_size=5), mlp_spec(input_dim=input_dim, depth=1, width=18)),
        ("kan_d2_w8_g5__mlp_d2_w24", kan_spec(input_dim=input_dim, depth=2, width=8, grid_size=5), mlp_spec(input_dim=input_dim, depth=2, width=24)),
    ]


def parameter_count(model_spec: dict[str, Any]) -> int:
    module = get_model_type(model_spec["type"]).build_module(model_spec)
    return sum(p.numel() for p in module.parameters())


def _candidate(profile: Any, dataset_id: str, family: str, budget: int, model: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    spec = build_candidate_spec(profile, dataset_id=dataset_id, family=family, budget=budget,
                                batch_size=32, model=model, optimizer=_optimizer(), loss=_loss())
    write_candidate(spec, out_dir, get_model_type(model["type"]))
    return spec


def calibrate_dataset(profile_name: str, dataset_path: Path, budgets: list[int], n_seeds: int) -> dict[str, Any]:
    profile = load_profile(profile_name)
    dataset_path = dataset_path.resolve()
    ds = read_json(dataset_path / "dataset_spec.json")
    family_name = ds["family"]
    input_dim = int(ds["params"]["input_dim"])
    family = get_dataset_family(family_name)
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        if budget % 32:
            raise ValueError(f"budget {budget} must be divisible by batch size 32")
        for pair_name, kan, mlp in pair_specs(input_dim):
            for label, model in (("kan", kan), ("mlp", mlp)):
                out_dir = OUTPUT_ROOT / "candidates" / ds["dataset_id"] / str(budget) / pair_name / label
                out_dir.mkdir(parents=True, exist_ok=True)
                spec = _candidate(profile, ds["dataset_id"], family_name, budget, model, out_dir)
                started = time.perf_counter()
                seed_results = [run_single_seed(out_dir, spec, train_x, train_y, test_x, test_y,
                                                profile.base_seed + i, float("inf"),
                                                selection_metric=ds["selection_metric"],
                                                device=torch.device("cpu")) for i in range(n_seeds)]
                elapsed = time.perf_counter() - started
                finals = [r["final_test_mse"] for r in seed_results if not r["failed"]]
                rows.append({"dataset_id": ds["dataset_id"], "input_dim": input_dim,
                             "expression": ds["params"].get("expression"), "budget": budget,
                             "batch_size": 32, "pair": pair_name, "label": label, "model": model,
                             "parameters": parameter_count(model), "seeds": n_seeds,
                             "failed_seeds": sum(bool(r["failed"]) for r in seed_results),
                             "failure_rate": sum(bool(r["failed"]) for r in seed_results) / n_seeds,
                             "mean_test_mse": sum(finals) / len(finals) if finals else float("inf"),
                             "seed_results": seed_results, "elapsed_seconds": elapsed})
                print(f"[{input_dim}d {budget} {pair_name}] {label}: mse={rows[-1]['mean_test_mse']:.6g} params={rows[-1]['parameters']} failed={rows[-1]['failed_seeds']}/{n_seeds} time={elapsed:.2f}s", flush=True)
    out = OUTPUT_ROOT / f"calibration_{ds['dataset_id']}.json"
    write_json(out, {"profile": profile.name, "profile_hash": profile.profile_hash,
                     "dataset_id": ds["dataset_id"], "input_dim": input_dim,
                     "n_seeds": n_seeds, "rows": rows})
    return read_json(out)


def generate_questions_for_dataset(profile_name: str, dataset_path: Path, budget: int, seed: int) -> dict[str, Any]:
    profile = load_profile(profile_name)
    dataset_path = dataset_path.resolve()
    ds = read_json(dataset_path / "dataset_spec.json")
    input_dim = int(ds["params"]["input_dim"])
    runs: list[dict[str, Any]] = []
    for index, (pair_name, kan, mlp) in enumerate(pair_specs(input_dim)):
        set_name = make_set_name(budget, frozenset({"model"}), salt=f"mvar-kan-mlp-{input_dim}-{seed}-{index}")
        set_path = candidate_set_dir(dataset_path, set_name)
        set_path.mkdir(parents=True, exist_ok=False)
        write_set_manifest(set_path, set_name=set_name, budget=budget, count=2,
                           varying_axes=frozenset({"model"}), fixed_shared={"batch_size": 32,
                           "optimizer": _optimizer(), "loss": _loss()}, seed=seed + index,
                           profile=profile, dataset_id=ds["dataset_id"], family=ds["family"])
        paths: list[Path] = []
        for model in (kan, mlp):
            spec = build_candidate_spec(profile, dataset_id=ds["dataset_id"], family=ds["family"],
                                        budget=budget, batch_size=32, model=model,
                                        optimizer=_optimizer(), loss=_loss())
            out_dir = candidate_in_set_dir(set_path, spec["candidate_id"])
            write_candidate(spec, out_dir, get_model_type(model["type"]))
            run_ground_truth(out_dir, profile, dataset_path)
            paths.append(out_dir)
        summaries = [load_summary(path) for path in paths]
        sig = validate_significance(summaries, profile, metric=ds["selection_metric"])
        run_path: Path | None = None
        question_error: str | None = None
        if sig.passed:
            try:
                run_path, _ = generate_questions(profile, dataset_path=dataset_path,
                                                 candidate_set_paths=[set_path], rng=random.Random(seed + index),
                                                 num_questions=1, num_choices=2, seed=seed + index)
            except RuntimeError as exc:
                question_error = str(exc)
        if run_path is not None:
            for qdir in sorted(run_path.glob("q_*")):
                write_prompt(qdir)
        runs.append({"pair": pair_name, "pair_name": pair_name, "set_path": str(set_path),
                     "question_run": str(run_path) if run_path is not None else None,
                     "question_error": question_error,
                     "significance": {"passed": sig.passed, "gap": sig.gap,
                                       "win_rate": sig.win_rate, "winner_index": sig.winner_index,
                                       "reason": sig.reason},
                     "summaries": [{"candidate_id": s["candidate_id"], "mean_test_mse": s["mean_test_mse"],
                                    "std_test_mse": s["std_test_mse"], "failed_seeds": s["failed_seeds"],
                                    "excluded": s["excluded"]} for s in summaries]})
        print(f"[{input_dim}d {pair_name}] question={run_path} significance={sig.passed} reason={sig.reason or question_error or 'not attempted'}", flush=True)
    out = OUTPUT_ROOT / f"questions_{ds['dataset_id']}.json"
    write_json(out, {"profile": profile.name, "profile_hash": profile.profile_hash,
                     "dataset_id": ds["dataset_id"], "input_dim": input_dim,
                     "budget": budget, "runs": runs})
    return read_json(out)


def main() -> None:
    ensure_registries()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "generate"))
    parser.add_argument("--profile", default="v2.1")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1024])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7300)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate":
        calibrate_dataset(args.profile, args.dataset_path, args.budgets, args.seeds)
    else:
        generate_questions_for_dataset(args.profile, args.dataset_path, args.budget, args.seed)


if __name__ == "__main__":
    main()
