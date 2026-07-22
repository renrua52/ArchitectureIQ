"""Formal KAN--MLP calibration for synthetic tabular classification.

This tool uses the normal ArchitectureIQ candidate renderer and training path.
It keeps dataset, budget, batch size, optimizer, and loss fixed within each
controlled pair, then validates the primary ``test_ce`` metric over multiple
training seeds.  Calibration output is separate from question artifacts until
an explicit ``generate`` command is run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.candidates.sets import make_set_name, write_set_manifest
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir
from architecture_iq.profile import load_profile
from architecture_iq.prompts.renderer import write_prompt
from architecture_iq.questions.generator import generate_questions
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "kan_mlp_calibration_classification"
FAMILY = "synthetic_tabular_classification"
BATCH_SIZE = 32
DEFAULT_BUDGET = 8192
DEFAULT_SEEDS = 10


def _optimizer() -> dict[str, Any]:
    return {
        "type": "AdamW",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-5,
        "betas": [0.9, 0.999],
    }


def _loss() -> dict[str, Any]:
    return {"loss_id": "cross_entropy"}


def kan_spec(*, input_dim: int, depth: int, width: int = 8) -> dict[str, Any]:
    return {
        "type": "kan",
        "variant": "efficient_spline_v1",
        "input_dim": input_dim,
        "output_dim": 2,
        "depth": depth,
        "width": width,
        "grid_size": 5,
        "spline_order": 3,
        "grid_range": [-1.0, 1.0],
        "base_activation": "silu",
    }


def mlp_spec(*, input_dim: int, depth: int, width: int) -> dict[str, Any]:
    return {
        "type": "mlp",
        "input_dim": input_dim,
        "output_dim": 2,
        "depth": depth,
        "width": width,
        "residual": False,
        "layer_norm": [False] * depth,
        "activations": ["silu"] * depth,
        "leaky_relu_slope": 0.01,
    }


# Widths are selected per input dimension because output_dim=2 changes the
# parameter-count matching relative to regression.
_MLP_WIDTHS = {
    4: {1: 28, 2: 26},
    8: {1: 30, 2: 27},
    16: {1: 34, 2: 30},
}


def pair_specs(input_dim: int) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if input_dim not in _MLP_WIDTHS:
        raise ValueError(f"classification calibration expects input_dim in {sorted(_MLP_WIDTHS)}, got {input_dim}")
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for depth in (1, 2):
        mlp_width = _MLP_WIDTHS[input_dim][depth]
        name = f"kan_cls_d{depth}_w8_g5__mlp_d{depth}_w{mlp_width}_i{input_dim}"
        pairs.append((name, kan_spec(input_dim=input_dim, depth=depth), mlp_spec(input_dim=input_dim, depth=depth, width=mlp_width)))
    return pairs


def parameter_count(model_spec: dict[str, Any]) -> int:
    ensure_registries()
    model = get_model_type(model_spec["type"]).build_module(model_spec)
    return sum(parameter.numel() for parameter in model.parameters())


def _assert_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    if dataset_spec.get("family") != FAMILY:
        raise ValueError(f"Expected {FAMILY!r}, got {dataset_spec.get('family')!r}")
    return dataset_spec


def _write_candidate(
    profile: Any,
    dataset_spec: dict[str, Any],
    budget: int,
    model: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    spec = build_candidate_spec(
        profile,
        dataset_id=dataset_spec["dataset_id"],
        family=FAMILY,
        budget=budget,
        batch_size=BATCH_SIZE,
        model=model,
        optimizer=_optimizer(),
        loss=_loss(),
    )
    write_candidate(spec, out_dir, get_model_type(model["type"]))
    return spec


def calibrate_dataset(
    profile_name: str,
    dataset_path: Path,
    *,
    budget: int = DEFAULT_BUDGET,
    n_seeds: int = DEFAULT_SEEDS,
) -> dict[str, Any]:
    ensure_registries()
    profile = load_profile(profile_name)
    profile.ground_truth["n_seeds"] = int(n_seeds)
    dataset_path = dataset_path.resolve()
    dataset_spec = _assert_dataset(dataset_path)
    input_dim = int(dataset_spec["params"]["input_dim"])
    if budget % BATCH_SIZE:
        raise ValueError(f"budget {budget} must be divisible by batch size {BATCH_SIZE}")
    family = get_dataset_family(FAMILY)
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    dataset_output = OUTPUT_ROOT / "candidates" / dataset_spec["dataset_id"] / str(budget)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for pair_name, kan, mlp in pair_specs(input_dim):
        pair_rows: dict[str, dict[str, Any]] = {}
        for label, model in (("kan", kan), ("mlp", mlp)):
            out_dir = dataset_output / pair_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            spec = _write_candidate(profile, dataset_spec, budget, model, out_dir)
            started = time.perf_counter()
            summary = run_ground_truth(out_dir, profile, dataset_path)
            elapsed = time.perf_counter() - started
            row = {
                "dataset_id": dataset_spec["dataset_id"],
                "input_dim": input_dim,
                "rule_family": dataset_spec["params"]["rule_family"],
                "budget": budget,
                "batch_size": BATCH_SIZE,
                "optimizer": _optimizer(),
                "loss": _loss(),
                "pair": pair_name,
                "label": label,
                "model": model,
                "candidate_id": spec["candidate_id"],
                "candidate_path": str(out_dir.relative_to(ROOT)),
                "profile_hash": profile.profile_hash,
                "parameters": parameter_count(model),
                "n_seeds": n_seeds,
                "failed_seeds": summary["failed_seeds"],
                "failure_rate": summary["failed_seeds"] / n_seeds,
                "mean_test_ce": summary["mean_test_ce"],
                "std_test_ce": summary["std_test_ce"],
                "mean_test_accuracy": summary.get("mean_test_accuracy"),
                "std_test_accuracy": summary.get("std_test_accuracy"),
                "seed_results": summary["seed_results"],
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            pair_rows[label] = row
            print(
                f"[{dataset_spec['dataset_id']} {pair_name}] {label}: "
                f"ce={row['mean_test_ce']:.6g} acc={row['mean_test_accuracy']:.4f} "
                f"params={row['parameters']} failed={row['failed_seeds']}/{n_seeds} "
                f"time={elapsed:.2f}s",
                flush=True,
            )
        summaries = [
            {"candidate_id": pair_rows["kan"]["candidate_id"], "mean_test_ce": pair_rows["kan"]["mean_test_ce"], "std_test_ce": pair_rows["kan"]["std_test_ce"], "failed_seeds": pair_rows["kan"]["failed_seeds"], "excluded": pair_rows["kan"]["failed_seeds"] >= int(profile.ground_truth["max_failed_seeds"]), "seed_results": pair_rows["kan"]["seed_results"]},
            {"candidate_id": pair_rows["mlp"]["candidate_id"], "mean_test_ce": pair_rows["mlp"]["mean_test_ce"], "std_test_ce": pair_rows["mlp"]["std_test_ce"], "failed_seeds": pair_rows["mlp"]["failed_seeds"], "excluded": pair_rows["mlp"]["failed_seeds"] >= int(profile.ground_truth["max_failed_seeds"]), "seed_results": pair_rows["mlp"]["seed_results"]},
        ]
        observed_wins = sum(1 for kan_seed, mlp_seed in zip(pair_rows["kan"]["seed_results"], pair_rows["mlp"]["seed_results"]) if kan_seed["final_test_ce"] < mlp_seed["final_test_ce"])
        observed_win_rate = observed_wins / n_seeds
        sig = validate_significance(summaries, profile, metric="test_ce")
        pairs.append({
            "pair": pair_name,
            "kan_candidate_id": pair_rows["kan"]["candidate_id"],
            "mlp_candidate_id": pair_rows["mlp"]["candidate_id"],
            "kan_parameters": pair_rows["kan"]["parameters"],
            "mlp_parameters": pair_rows["mlp"]["parameters"],
            "parameter_ratio": pair_rows["kan"]["parameters"] / pair_rows["mlp"]["parameters"],
            "observed_kan_seed_wins": observed_wins,
            "observed_kan_seed_win_rate": observed_win_rate,
            "significance": {
                "passed": sig.passed,
                "gap": sig.gap,
                "win_rate": sig.win_rate,
                "winner_index": sig.winner_index,
                "metric": sig.metric,
                "reason": sig.reason,
            },
        })
        print(f"[{dataset_spec['dataset_id']} {pair_name}] passed={sig.passed} gap={sig.gap:.6g} win_rate={sig.win_rate:.3f} reason={sig.reason or 'ok'}", flush=True)

    out = OUTPUT_ROOT / f"calibration_{dataset_spec['dataset_id']}.json"
    write_json(out, {
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "dataset_id": dataset_spec["dataset_id"],
        "family": FAMILY,
        "input_dim": input_dim,
        "rule_family": dataset_spec["params"]["rule_family"],
        "budget": budget,
        "batch_size": BATCH_SIZE,
        "n_seeds": n_seeds,
        "metric": "test_ce",
        "rows": rows,
        "pairs": pairs,
        "calibration_protocol": {
            "optimizer": _optimizer(),
            "loss": _loss(),
            "gap_min": profile.significance["gap_min"],
            "win_rate_min": profile.significance["win_rate_min"],
            "use_non_overlap": profile.significance.get("use_non_overlap", True),
        },
    })
    return read_json(out)


def generate_questions_for_dataset(
    profile_name: str,
    dataset_path: Path,
    *,
    calibration_path: Path | None = None,
    budget: int = DEFAULT_BUDGET,
    seed: int = 9100,
) -> dict[str, Any]:
    ensure_registries()
    from architecture_iq.profile import load_profile

    profile = load_profile(profile_name)
    dataset_path = dataset_path.resolve()
    dataset_spec = _assert_dataset(dataset_path)
    calibration_path = calibration_path or OUTPUT_ROOT / f"calibration_{dataset_spec['dataset_id']}.json"
    calibration = read_json(calibration_path)
    if calibration.get("profile_hash") != profile.profile_hash:
        raise ValueError("Calibration profile_hash does not match the active profile; recalibrate before generating questions")
    generated: list[dict[str, Any]] = []
    for index, pair in enumerate(calibration.get("pairs", [])):
        if not pair["significance"]["passed"]:
            continue
        pair_name = pair["pair"]
        row_by_id = {row["candidate_id"]: row for row in calibration["rows"] if row["pair"] == pair_name}
        kan = row_by_id[pair["kan_candidate_id"]]["model"]
        mlp = row_by_id[pair["mlp_candidate_id"]]["model"]
        set_name = make_set_name(budget, frozenset({"model"}), salt=f"classification-kan-{dataset_spec['dataset_id']}-{profile.profile_hash}-{seed}-{index}")
        set_path = candidate_set_dir(dataset_path, set_name)
        if set_path.exists():
            raise FileExistsError(f"Question candidate set already exists: {set_path}")
        set_path.mkdir(parents=True, exist_ok=False)
        write_set_manifest(set_path, set_name=set_name, budget=budget, count=2, varying_axes=frozenset({"model"}), fixed_shared={"batch_size": BATCH_SIZE, "optimizer": _optimizer(), "loss": _loss()}, seed=seed + index, profile=profile, dataset_id=dataset_spec["dataset_id"], family=FAMILY)
        paths: list[Path] = []
        for model in (kan, mlp):
            spec = build_candidate_spec(profile, dataset_id=dataset_spec["dataset_id"], family=FAMILY, budget=budget, batch_size=BATCH_SIZE, model=model, optimizer=_optimizer(), loss=_loss())
            out_dir = candidate_in_set_dir(set_path, spec["candidate_id"])
            write_candidate(spec, out_dir, get_model_type(model["type"]))
            run_ground_truth(out_dir, profile, dataset_path)
            paths.append(out_dir)
        run_path, _ = generate_questions(profile, dataset_path=dataset_path, candidate_set_paths=[set_path], rng=random.Random(seed + index), num_questions=1, num_choices=2, seed=seed + index)
        for question_dir in sorted(run_path.glob("q_*")):
            write_prompt(question_dir)
        generated.append({"pair": pair_name, "set_path": str(set_path.relative_to(ROOT)), "question_run": str(run_path.relative_to(ROOT))})
    out = OUTPUT_ROOT / f"questions_{dataset_spec['dataset_id']}.json"
    write_json(out, {"profile": profile.name, "profile_hash": profile.profile_hash, "dataset_id": dataset_spec["dataset_id"], "family": FAMILY, "generated": generated})
    return read_json(out)


def main() -> None:
    ensure_registries()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--profile", default="v2.1")
    cal.add_argument("--dataset-path", type=Path, required=True)
    cal.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    cal.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    gen = sub.add_parser("generate")
    gen.add_argument("--profile", default="v2.1")
    gen.add_argument("--dataset-path", type=Path, required=True)
    gen.add_argument("--calibration", type=Path)
    gen.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    gen.add_argument("--seed", type=int, default=9100)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate":
        calibrate_dataset(args.profile, args.dataset_path, budget=args.budget, n_seeds=args.seeds)
    else:
        generate_questions_for_dataset(args.profile, args.dataset_path, calibration_path=args.calibration, budget=args.budget, seed=args.seed)


if __name__ == "__main__":
    main()
