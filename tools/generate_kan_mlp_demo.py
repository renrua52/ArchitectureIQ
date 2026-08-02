"""Calibrate and generate controlled KAN-vs-MLP regression questions.

This tool deliberately keeps the non-model part of a pair identical: dataset,
sample budget, batch size, optimizer, and loss.  It is a small phase-3 tool,
not a replacement for the generic candidate/question samplers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.candidates.sets import make_set_name, write_set_manifest
from architecture_iq.ground_truth.runner import run_ground_truth, run_single_seed
from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir
from architecture_iq.profile import load_profile
from architecture_iq.questions.generator import generate_questions
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.util import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = ROOT / "outputs" / "kan_mlp_calibration"
DEFAULT_PAIR_CONFIG = ROOT / "configs" / "kan_mlp_pairs_v2.1.yaml"


def _pair_config_hash(config_path: Path) -> str:
    """Hash normalized config semantics, independent of YAML formatting."""
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "kan_mlp_pairs" in payload:
        payload = payload["kan_mlp_pairs"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _model_parameter_count(model_spec: dict[str, Any]) -> int:
    family = get_model_type(model_spec["type"])
    module = family.build_module(model_spec)
    return sum(p.numel() for p in module.parameters())


def _load_pair_config(profile: Any, config_path: Path | None = None) -> dict[str, Any]:
    """Load the external controlled-pair contract.

    Pair settings intentionally live outside the profile so changing a
    diagnostic pair cannot silently change a formal profile hash.
    """
    if config_path is None:
        config_path = DEFAULT_PAIR_CONFIG
    source = str(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"KAN/MLP pair config not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(config, dict) and "kan_mlp_pairs" in config:
        config = config["kan_mlp_pairs"]
    if not isinstance(config, dict):
        raise ValueError(f"{source} does not define a kan_mlp_pairs mapping")
    required = ("pairs", "batch_size", "optimizer", "loss")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{source} kan_mlp_pairs is missing: {', '.join(missing)}")
    if not isinstance(config["pairs"], list) or not config["pairs"]:
        raise ValueError(f"{source} kan_mlp_pairs.pairs must be a non-empty list")
    return copy.deepcopy(config)


def _pair_specs(config: dict[str, Any], input_dim: int) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Materialize configured pair specs, injecting the dataset's input dimension."""
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for pair in config["pairs"]:
        if not isinstance(pair, dict) or not isinstance(pair.get("name"), str):
            raise ValueError("Each kan_mlp_pairs entry needs a string name")
        if not isinstance(pair.get("kan"), dict) or not isinstance(pair.get("mlp"), dict):
            raise ValueError(f"Pair {pair.get('name')!r} must define kan and mlp mappings")
        kan = copy.deepcopy(pair["kan"])
        mlp = copy.deepcopy(pair["mlp"])
        kan["type"] = "kan"
        mlp["type"] = "mlp"
        kan["input_dim"] = input_dim
        mlp["input_dim"] = input_dim
        kan.setdefault("output_dim", 1)
        mlp.setdefault("output_dim", 1)
        pairs.append((pair["name"], kan, mlp))
    return pairs


def _fixed_training(config: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    batch_size = int(config["batch_size"])
    optimizer = copy.deepcopy(config["optimizer"])
    loss = copy.deepcopy(config["loss"])
    if not isinstance(optimizer, dict) or not isinstance(loss, dict):
        raise ValueError("kan_mlp_pairs optimizer and loss must be mappings")
    return batch_size, optimizer, loss


def _write_pair_candidate(
    *,
    profile: Any,
    dataset_id: str,
    family: str,
    budget: int,
    batch_size: int,
    model_spec: dict[str, Any],
    out_dir: Path,
    optimizer: dict[str, Any],
    loss: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    spec = build_candidate_spec(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        batch_size=batch_size,
        model=model_spec,
        optimizer=optimizer,
        loss=loss,
    )
    write_candidate(spec, out_dir, get_model_type(model_spec["type"]))
    return spec, out_dir


def calibrate(*, profile_name: str, dataset_path: Path, budgets: list[int] | None, seeds: int, pair_config_path: Path | None = None) -> Path:
    profile = load_profile(profile_name)
    pair_config_path = (pair_config_path or DEFAULT_PAIR_CONFIG).resolve()
    pair_config = _load_pair_config(profile, pair_config_path)
    pair_config_hash = _pair_config_hash(pair_config_path)
    batch_size, optimizer, loss = _fixed_training(pair_config)
    device = torch.device(profile.execution_device)
    budgets = list(pair_config.get("budgets", [2048, 5120])) if budgets is None else budgets
    dataset_path = dataset_path.resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    family_name = dataset_spec["family"]
    input_dim = int(dataset_spec["params"].get("input_dim", 1))
    family = get_dataset_family(family_name)
    train_x, train_y, test_x, test_y = family.load_tensors(dataset_path)
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        if budget % batch_size:
            raise ValueError(f"budget {budget} must be divisible by batch size {batch_size}")
        for pair_name, kan, mlp in _pair_specs(pair_config, input_dim):
            for label, model in (("kan", kan), ("mlp", mlp)):
                out_dir = CALIBRATION_ROOT / dataset_spec["dataset_id"] / str(budget) / pair_name / label
                out_dir.mkdir(parents=True, exist_ok=True)
                spec, out_dir = _write_pair_candidate(
                    profile=profile,
                    dataset_id=dataset_spec["dataset_id"],
                    family=family_name,
                    budget=budget,
                    batch_size=batch_size,
                    model_spec=model,
                    out_dir=out_dir,
                    optimizer=optimizer,
                    loss=loss,
                )
                start = time.perf_counter()
                seed_results = [
                    run_single_seed(
                        out_dir,
                        spec,
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        seed,
                        float("inf"),
                        selection_metric=dataset_spec["selection_metric"],
                        device=device,
                    )
                    for seed in range(seeds)
                ]
                elapsed = time.perf_counter() - start
                finals = [r["final_test_mse"] for r in seed_results if not r["failed"]]
                rows.append(
                    {
                        "dataset_id": dataset_spec["dataset_id"],
                        "pair_config_hash": pair_config_hash,
                        "expression": dataset_spec["params"].get("expression"),
                        "budget": budget,
                        "batch_size": batch_size,
                        "optimizer": optimizer,
                        "loss": loss,
                        "pair": pair_name,
                        "label": label,
                        "model": model,
                        "parameters": _model_parameter_count(model),
                        "seeds": seeds,
                        "failed_seeds": sum(r["failed"] for r in seed_results),
                        "mean_test_mse": sum(finals) / len(finals) if finals else float("inf"),
                        "seed_results": seed_results,
                        "elapsed_seconds": elapsed,
                    }
                )
                print(f"[{budget}/{pair_name}] {label}: mse={rows[-1]['mean_test_mse']:.6g} params={rows[-1]['parameters']} time={elapsed:.2f}s", flush=True)

    out = CALIBRATION_ROOT / f"calibration_{dataset_spec['dataset_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"profile": profile.name, "profile_hash": profile.profile_hash, "pair_config_path": str(pair_config_path.relative_to(ROOT)), "pair_config_hash": pair_config_hash, "rows": rows})
    print(f"Calibration report: {out}")
    return out


def generate(*, profile_name: str, dataset_path: Path, budget: int, pair_names: list[str], seed: int, pair_config_path: Path | None = None) -> list[Path]:
    profile = load_profile(profile_name)
    pair_config_path = (pair_config_path or DEFAULT_PAIR_CONFIG).resolve()
    pair_config = _load_pair_config(profile, pair_config_path)
    pair_config_hash = _pair_config_hash(pair_config_path)
    batch_size, optimizer, loss = _fixed_training(pair_config)
    dataset_path = dataset_path.resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    family_name = dataset_spec["family"]
    input_dim = int(dataset_spec["params"].get("input_dim", 1))
    pair_map = {name: (kan, mlp) for name, kan, mlp in _pair_specs(pair_config, input_dim)}
    unknown = [name for name in pair_names if name not in pair_map]
    if unknown:
        raise ValueError(f"Unknown pair(s): {unknown}; choose from {sorted(pair_map)}")

    out_runs: list[Path] = []
    if budget % batch_size:
        raise ValueError(f"budget {budget} must be divisible by batch size {batch_size}")
    for index, pair_name in enumerate(pair_names):
        kan, mlp = pair_map[pair_name]
        salt = f"kan-mlp-demo-{seed}-{index}-{pair_name}"
        set_name = make_set_name(budget, frozenset({"model"}), salt=salt)
        set_path = candidate_set_dir(dataset_path, set_name)
        set_path.mkdir(parents=True, exist_ok=False)
        fixed_shared = {"batch_size": batch_size, "optimizer": optimizer, "loss": loss, "pair_config_hash": pair_config_hash}
        write_set_manifest(
            set_path,
            set_name=set_name,
            budget=budget,
            count=2,
            varying_axes=frozenset({"model"}),
            fixed_shared=fixed_shared,
            seed=seed + index,
            profile=profile,
            dataset_id=dataset_spec["dataset_id"],
            family=family_name,
        )
        paths: list[Path] = []
        for model in (kan, mlp):
            spec = build_candidate_spec(
                profile,
                dataset_id=dataset_spec["dataset_id"],
                family=family_name,
                budget=budget,
                batch_size=batch_size,
                model=model,
                optimizer=optimizer,
                loss=loss,
            )
            out_dir = candidate_in_set_dir(set_path, spec["candidate_id"])
            write_candidate(spec, out_dir, get_model_type(model["type"]))
            summary = run_ground_truth(out_dir, profile, dataset_path)
            print(f"[{pair_name}] {model['type']} mean_test_mse={summary.get('mean_test_mse'):.6g} excluded={summary['excluded']}", flush=True)
            paths.append(out_dir)
        if any(read_json(path / "results" / "summary.json").get("excluded") for path in paths):
            print(f"[{pair_name}] skipped question because a candidate was excluded", flush=True)
            continue
        run_path, _ = generate_questions(
            profile,
            dataset_path=dataset_path,
            candidate_set_paths=[set_path],
            rng=random.Random(seed + index),
            num_questions=1,
            num_choices=2,
            seed=seed + index,
        )
        from architecture_iq.prompts.renderer import write_prompt

        for question_dir in sorted(run_path.glob("q_*")):
            write_prompt(question_dir)
        out_runs.append(run_path)
        print(f"[{pair_name}] question run: {run_path}", flush=True)
    return out_runs


def main() -> None:
    ensure_registries()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--profile", default="v2.1")
    cal.add_argument("--dataset-path", type=Path, required=True)
    cal.add_argument("--budgets", type=int, nargs="+")
    cal.add_argument("--seeds", type=int, default=2)
    cal.add_argument("--pair-config", type=Path)
    gen = sub.add_parser("generate")
    gen.add_argument("--profile", default="v2.1")
    gen.add_argument("--dataset-path", type=Path, required=True)
    gen.add_argument("--budget", type=int, required=True)
    gen.add_argument("--pair", dest="pairs", action="append")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--pair-config", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(profile_name=args.profile, dataset_path=args.dataset_path, budgets=args.budgets, seeds=args.seeds, pair_config_path=args.pair_config)
    else:
        profile = load_profile(args.profile)
        input_dim = int(read_json(args.dataset_path.resolve() / "dataset_spec.json")["params"].get("input_dim", 1))
        pair_config = _load_pair_config(profile, args.pair_config)
        names = args.pairs or [name for name, _, _ in _pair_specs(pair_config, input_dim)]
        generate(profile_name=args.profile, dataset_path=args.dataset_path, budget=args.budget, pair_names=names, seed=args.seed, pair_config_path=args.pair_config)


if __name__ == "__main__":
    main()
