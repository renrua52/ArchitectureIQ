#!/usr/bin/env python3
"""Reproduce the ArchitectureIQ ground truth for one question from this folder alone.

This script is self-contained: it does not import the ArchitectureIQ package. It
mirrors the semantics of `architecture_iq.ground_truth.runner.run_ground_truth`
and `architecture_iq.runtime.loader`:

    dataset/synthesize.py  ->  synthesize()      ->  train/test tensors
    choices/<L>/train.py   ->  train_and_eval()  ->  final_{selection_metric}
    mean/std over seeds base_seed .. base_seed + n_seeds - 1

Requires only `torch` (CPU build is enough) and `numpy`.

    python reproduce.py                      # every choice, every seed
    python reproduce.py --seeds 1            # quick smoke run
    python reproduce.py --letters A,C        # only some choices
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

BUNDLE = Path(__file__).resolve().parent
SIBLING_MODULES = ("model", "optimizer", "loss", "train")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_module_from_file(path: Path, module_name: str) -> ModuleType:
    """Same mechanism as architecture_iq.runtime.loader.load_module_from_file."""
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_candidate_train(candidate_dir: Path) -> ModuleType:
    """Load choices/<L>/train.py so its sibling `model`/`loss`/`optimizer` imports resolve.

    Clearing the cached sibling modules is what keeps choice B from silently
    training choice A's model — the loader in the pipeline does exactly this.
    """
    train_file = candidate_dir / "train.py"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train.py in {candidate_dir}")
    path_str = str(candidate_dir)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        for name in SIBLING_MODULES:
            sys.modules.pop(name, None)
        return _load_module_from_file(train_file, f"candidate_train_{candidate_dir.name}")
    finally:
        if inserted:
            sys.path.remove(path_str)


def _relative_error(recomputed: float, recorded: float) -> float:
    denominator = max(abs(recorded), 1e-12)
    return abs(recomputed - recorded) / denominator


def main() -> int:
    question = _read_json(BUNDLE / "question.json")
    dataset_spec = _read_json(BUNDLE / "dataset" / "dataset_spec.json")
    letters = [choice["letter"] for choice in question["choices"]]

    parser = argparse.ArgumentParser(description=f"Reproduce ground truth for {question['question_id']}")
    parser.add_argument(
        "--seeds",
        type=int,
        default=int(question["n_seeds"]),
        help=f"how many seeds to run per choice (default {question['n_seeds']}, the ground-truth count)",
    )
    parser.add_argument(
        "--letters",
        default=",".join(letters),
        help="comma-separated subset of choices to run (default: all)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="relative tolerance when comparing against the recorded reference (default 1e-4)",
    )
    parser.add_argument("--device", default=str(question.get("device", "cpu")))
    parser.add_argument(
        "--threads",
        type=int,
        default=question.get("torch_threads_per_seed"),
        help="torch intra-op threads; defaults to the thread count the ground truth was recorded with",
    )
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="also write the synthesized splits to dataset/train.pt and dataset/test.pt",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    if args.threads:
        torch.set_num_threads(int(args.threads))

    requested = [letter.strip().upper() for letter in args.letters.split(",") if letter.strip()]
    unknown = [letter for letter in requested if letter not in letters]
    if unknown:
        parser.error(f"unknown choice letter(s) {unknown}; this question has {letters}")

    metric = str(question["selection_metric"])
    final_key = f"final_{metric}"
    base_seed = int(question["base_seed"])
    fail_threshold = float(
        question.get("fail_threshold", dataset_spec.get("significance", {}).get("fail_threshold", float("inf")))
    )

    print(f"question:   {question['question_id']}  ({question['type']}, {question['family']})")
    print(f"dataset:    {question['dataset_id']}")
    print(f"metric:     {metric}  (lower is better)")
    print(f"seeds:      {args.seeds} of {question['n_seeds']}, starting at {base_seed}")
    print(f"device:     {args.device}   torch threads: {torch.get_num_threads()}")
    print(f"torch:      {torch.__version__}")
    print()

    synth = _load_module_from_file(BUNDLE / "dataset" / "synthesize.py", "bundle_synthesize")
    train_x, train_y, test_x, test_y = synth.synthesize()
    print(f"synthesized train={tuple(train_x.shape)} test={tuple(test_x.shape)}")
    if args.save_tensors:
        torch.save({"x": train_x, "y": train_y}, BUNDLE / "dataset" / "train.pt")
        torch.save({"x": test_x, "y": test_y}, BUNDLE / "dataset" / "test.pt")
        print("wrote dataset/train.pt and dataset/test.pt")
    print()

    mismatches: list[str] = []
    comparisons = 0
    results: dict[str, dict] = {}

    for letter in requested:
        choice_dir = BUNDLE / "choices" / letter
        spec = _read_json(choice_dir / "candidate_spec.json")
        reference_path = choice_dir / "reference" / "summary.json"
        reference = _read_json(reference_path) if reference_path.exists() else None
        reference_seeds = {
            int(entry["seed"]): float(entry[final_key])
            for entry in (reference or {}).get("seed_results", [])
            if final_key in entry
        }

        steps = int(spec["budget"]["training_steps"])
        batch_size = int(spec["budget"]["batch_size"])
        print(f"=== choice {letter}  ({spec['candidate_id']})")
        print(f"    {spec['model']['type']} / {spec['optimizer']['type']} / {spec['loss']['loss_id']}")
        print(f"    steps={steps} batch_size={batch_size} total_samples_seen={spec['budget']['total_samples_seen']}")

        train_mod = _load_candidate_train(choice_dir)
        if not hasattr(train_mod, "train_and_eval"):
            raise RuntimeError(f"{choice_dir}/train.py does not define train_and_eval()")

        finals: list[float] = []
        failed = 0
        for index in range(int(args.seeds)):
            seed = base_seed + index
            started = time.perf_counter()
            result = train_mod.train_and_eval(
                train_x,
                train_y,
                test_x,
                test_y,
                steps=steps,
                batch_size=batch_size,
                seed=seed,
                fail_threshold=fail_threshold,
                device=str(args.device),
            )
            if final_key not in result:
                raise KeyError(f"train_and_eval did not return {final_key!r}")
            value = float(result[final_key])
            elapsed = time.perf_counter() - started
            if bool(result["failed"]):
                failed += 1
            else:
                finals.append(value)

            line = f"    seed {seed:>3}  {metric}={value:.10g}  ({elapsed:.1f}s)"
            if seed in reference_seeds:
                recorded = reference_seeds[seed]
                error = _relative_error(value, recorded)
                status = "ok" if error <= args.tol else "MISMATCH"
                comparisons += 1
                line += f"  recorded={recorded:.10g}  rel_err={error:.3g}  {status}"
                if error > args.tol:
                    mismatches.append(f"{letter} seed {seed}: {value!r} vs recorded {recorded!r} (rel_err={error:.3g})")
            print(line)

        mean = float(np.mean(finals)) if finals else float("inf")
        std = float(np.std(finals)) if finals else float("inf")
        results[letter] = {"mean": mean, "std": std, "failed_seeds": failed}
        summary_line = f"    mean {metric} = {mean:.10g} ± {std:.10g}   (failed seeds: {failed})"

        if reference is not None and int(args.seeds) == int(question["n_seeds"]):
            recorded_mean = float(reference[f"mean_{metric}"])
            error = _relative_error(mean, recorded_mean)
            status = "ok" if error <= args.tol else "MISMATCH"
            comparisons += 1
            summary_line += f"\n    recorded mean = {recorded_mean:.10g}   rel_err={error:.3g}  {status}"
            if error > args.tol:
                mismatches.append(f"{letter} mean: {mean!r} vs recorded {recorded_mean!r} (rel_err={error:.3g})")
        print(summary_line)
        print()

    winner = min(results, key=lambda letter: results[letter]["mean"])
    partial = len(requested) < len(letters) or int(args.seeds) != int(question["n_seeds"])
    if len(requested) > 1:
        print(f"lowest mean {metric}: choice {winner}")
    correct = question.get("correct_letter")
    if correct is None:
        print("This bundle was downloaded before the question was answered, so it carries no")
        print("reference results. Answer the question on architecture-iq.com and download it")
        print("again to check these numbers against the published ground truth.")
    elif partial:
        print(f"recorded correct answer: {correct} (partial run — not a like-for-like comparison)")
    elif winner == correct:
        print(f"recorded correct answer: {correct} — reproduced ✓")
    else:
        mismatches.append(f"winner {winner} != recorded correct answer {correct}")
        print(f"recorded correct answer: {correct} — MISMATCH")

    if mismatches:
        print()
        print(f"{len(mismatches)} mismatch(es) beyond rel. tolerance {args.tol:g}:")
        for entry in mismatches:
            print(f"  - {entry}")
        print()
        print("Small deviations are expected on a different torch build or thread count")
        print("(float32 reductions are not associative). See README.md.")
        return 1

    print()
    if comparisons == 0:
        print(f"ran {len(requested)} choice(s); nothing to compare against in this bundle.")
    else:
        print(f"reproduced {comparisons} recorded value(s) within relative tolerance {args.tol:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
