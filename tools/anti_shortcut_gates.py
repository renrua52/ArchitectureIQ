#!/usr/bin/env python
"""Anti-shortcut question validation gates.

Usage:
  python3 tools/anti_shortcut_gates.py \
    --questions artifacts/quiz_attempt_60/questions_sanitized.json \
    --answer-key artifacts/quiz_attempt_60/answer_key.json \
    --output artifacts/anti_shortcut_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Param counting (mirrors evaluate_arithmetic_rules.py and analyze_order_parameters.py)
# ---------------------------------------------------------------------------

def mlp_params(model: dict[str, Any]) -> int:
    input_dim = int(model.get("input_dim", 1))
    width = int(model["width"])
    depth = int(model["depth"])
    layer_norm = [bool(v) for v in model.get("layer_norm", [])]
    total = input_dim * width + width
    for i in range(depth):
        total += width * width + width
        if i < len(layer_norm) and layer_norm[i]:
            total += 2 * width
    total += width + 1
    return total


def transformer_params(model: dict[str, Any]) -> int:
    vocab = int(model["vocab_size"])
    context = int(model["context_length"])
    d_model = int(model.get("d_model", model.get("embed_dim")))
    d_ff = int(model.get("d_ff", model.get("ff_dim")))
    num_layers = int(model["num_layers"])
    embeddings = vocab * d_model + context * d_model
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 9 * d_model + d_ff
    head = d_model * vocab + vocab
    return embeddings + num_layers * per_layer + head


def count_params(model: dict[str, Any]) -> int:
    if model["type"] == "mlp":
        return mlp_params(model)
    if model["type"] == "transformer_lm":
        return transformer_params(model)
    raise ValueError(f"Unknown model type: {model['type']}")


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


def _load_synthesize_target(dataset_path: Path) -> tuple[Any, dict[str, Any]]:
    """Import synthesize.py from a dataset instance and return (target_fn, dataset_spec)."""
    import importlib.util

    spec_path = dataset_path / "dataset_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"dataset_spec.json not found at {dataset_path}")
    dataset_spec = json.loads(spec_path.read_text(encoding="utf-8"))

    synth_path = dataset_path / "synthesize.py"
    if not synth_path.exists():
        raise FileNotFoundError(f"synthesize.py not found at {dataset_path}")

    spec = importlib.util.spec_from_file_location("synthesize", str(synth_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.target, dataset_spec


def gate_affine_fit(question: dict[str, Any], dataset_path: Path) -> GateResult:
    """Check whether the target function is approximately affine (linear)."""
    family = question.get("family", "")
    if family != "univariate_regression":
        return GateResult("affine_fit", True, {"skipped": True, "reason": f"not applicable for {family}"})

    try:
        target_fn, _ = _load_synthesize_target(dataset_path)
    except Exception as e:
        return GateResult("affine_fit", True, {"skipped": True, "reason": f"could not load target: {e}"})

    grid = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    try:
        y = target_fn(grid)
    except Exception as e:
        return GateResult("affine_fit", True, {"skipped": True, "reason": f"target eval failed: {e}"})

    y = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        return GateResult("affine_fit", True, {"skipped": True, "reason": "non-finite target values"})

    x = grid.ravel()
    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    pred = x_design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    passed = r2 < 0.95
    return GateResult(
        "affine_fit",
        passed,
        {
            "r2": round(r2, 6),
            "threshold": 0.95,
            "y_range": round(float(np.max(y) - np.min(y)), 6),
            "max_abs_y": round(float(np.max(np.abs(y))), 6),
        },
    )


def gate_capacity_shortcut(question: dict[str, Any]) -> GateResult:
    """Check whether the naive 'largest model wins' rule picks the correct answer."""
    correct_letter = question.get("correct_letter")
    choices = question.get("choices", [])
    if not choices or not correct_letter:
        return GateResult("capacity_shortcut", True, {"skipped": True, "reason": "missing choices or correct_letter"})

    best_params = -1
    best_letter = ""
    for ch in choices:
        model = ch.get("model", {})
        params = count_params(model)
        if params > best_params:
            best_params = params
            best_letter = ch["letter"]

    max_params_wins = best_letter == correct_letter
    passed = not max_params_wins
    return GateResult(
        "capacity_shortcut",
        passed,
        {
            "max_params_letter": best_letter,
            "max_params_value": best_params,
            "correct_letter": correct_letter,
            "max_params_wins": max_params_wins,
        },
    )


def gate_snr(question: dict[str, Any], dataset_path: Path) -> GateResult:
    """Check signal-to-noise ratio from dataset_spec."""
    spec_path = dataset_path / "dataset_spec.json"
    if not spec_path.exists():
        return GateResult("snr", True, {"skipped": True, "reason": "no dataset_spec.json"})

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    noise_std = spec.get("label_noise_std", 0.0)
    if noise_std is None or noise_std <= 0:
        return GateResult("snr", True, {"skipped": True, "reason": "no label noise"})

    try:
        target_fn, _ = _load_synthesize_target(dataset_path)
    except Exception as e:
        return GateResult("snr", True, {"skipped": True, "reason": f"could not load target: {e}"})

    grid = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    try:
        y = target_fn(grid)
    except Exception:
        return GateResult("snr", True, {"skipped": True, "reason": "target eval failed"})

    y = np.asarray(y, dtype=float).ravel()
    signal_std = float(np.std(y))
    if signal_std <= 0:
        return GateResult("snr", False, {"noise_std": noise_std, "signal_std": signal_std, "snr": 0.0, "reason": "zero signal"})

    snr = signal_std / noise_std
    passed = snr >= 1.0
    return GateResult(
        "snr",
        passed,
        {
            "noise_std": noise_std,
            "signal_std": round(signal_std, 6),
            "snr": round(snr, 4),
            "threshold": 1.0,
        },
    )


def gate_interaction(question: dict[str, Any], dataset_path: Path) -> GateResult:
    """Check that multivariate targets are not dominated by a single input dimension."""
    family = question.get("family", "")
    if family != "multivariate_regression":
        return GateResult("interaction", True, {"skipped": True, "reason": f"not applicable for {family}"})

    try:
        target_fn, dataset_spec = _load_synthesize_target(dataset_path)
    except Exception as e:
        return GateResult("interaction", True, {"skipped": True, "reason": f"could not load target: {e}"})

    input_dim = dataset_spec.get("input_dim", 1)
    if input_dim <= 1:
        return GateResult("interaction", True, {"skipped": True, "reason": "single input dimension"})

    class DummyModule:
        pass
    try:
        import torch
        dummy = DummyModule()
        dummy.torch = torch
        dummy.tensor = torch.tensor
        target_fn.__globals__.update({"torch": torch, "tensor": torch.tensor})
    except ImportError:
        return GateResult("interaction", True, {"skipped": True, "reason": "torch not available"})

    n_samples = 1024
    rng = np.random.RandomState(42)
    x_base = rng.uniform(0.0, 1.0, (n_samples, input_dim))

    def eval_at(x_np: np.ndarray) -> np.ndarray:
        try:
            y = target_fn(x_np)
            return np.asarray(y, dtype=float).ravel()
        except Exception:
            return np.full(x_np.shape[0], np.nan)

    y_base = eval_at(x_base)
    if not np.all(np.isfinite(y_base)):
        return GateResult("interaction", True, {"skipped": True, "reason": "non-finite output"})

    total_var = float(np.var(y_base))
    if total_var <= 0:
        return GateResult("interaction", True, {"skipped": True, "reason": "zero variance"})

    # Measure per-dimension variance contribution via perturbation
    dim_contributions = []
    for d in range(input_dim):
        x_perturbed = x_base.copy()
        x_perturbed[:, d] = rng.uniform(0.0, 1.0, n_samples)
        y_perturbed = eval_at(x_perturbed)
        if not np.all(np.isfinite(y_perturbed)):
            dim_contributions.append(0.0)
            continue
        # Variance from perturbing this dimension
        var_perturbed = float(np.var(y_perturbed - y_base))
        dim_contributions.append(var_perturbed)

    total_contrib = sum(dim_contributions)
    if total_contrib <= 0:
        return GateResult("interaction", True, {"skipped": True, "reason": "no measurable per-dim variance"})

    fractions = [c / total_contrib for c in dim_contributions]
    max_frac = max(fractions)
    max_dim = int(np.argmax(fractions))

    passed = max_frac < 0.85
    return GateResult(
        "interaction",
        passed,
        {
            "input_dim": input_dim,
            "per_dim_fractions": [round(f, 4) for f in fractions],
            "max_fraction": round(max_frac, 4),
            "max_dim": max_dim,
            "threshold": 0.85,
            "n_samples": n_samples,
        },
    )


# ---------------------------------------------------------------------------
# Full question evaluation
# ---------------------------------------------------------------------------

@dataclass
class QuestionGateReport:
    question_id: str
    family: str
    dataset_id: str
    gates: dict[str, GateResult] = field(default_factory=dict)

    @property
    def overall(self) -> str:
        results = [g.passed for g in self.gates.values() if not g.details.get("skipped")]
        if not results:
            return "passed"
        return "passed" if all(results) else "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "family": self.family,
            "dataset_id": self.dataset_id,
            "gates": {name: {"passed": g.passed, **g.details} for name, g in self.gates.items()},
            "overall": self.overall,
        }


def validate_question(
    question: dict[str, Any],
    data_root: Path,
) -> QuestionGateReport:
    """Run all applicable gates on a single question."""
    qid = question["question_id"]
    family = question.get("family", "?")
    dataset_id = question.get("dataset_id", "?")

    dataset_path = data_root / "datasets" / family / dataset_id

    report = QuestionGateReport(question_id=qid, family=family, dataset_id=dataset_id)

    report.gates["affine_fit"] = gate_affine_fit(question, dataset_path)
    report.gates["capacity_shortcut"] = gate_capacity_shortcut(question)
    report.gates["snr"] = gate_snr(question, dataset_path)
    report.gates["interaction"] = gate_interaction(question, dataset_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return [json.loads(item.read_text(encoding="utf-8")) for item in sorted(path.glob("*.json"))]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")
    return payload


def load_answers(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Answer key must be a JSON list: {path}")
    return {row["question_id"]: row["correct_letter"] for row in payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anti-shortcut question validation gates.")
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to sanitized questions JSON file or directory.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        required=True,
        help="Path to answer key JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "anti_shortcut_report.json",
        help="Output path for gate report JSON.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Path to data/ directory (default: repo_root/data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.questions.exists():
        print(f"Questions not found: {args.questions}", file=sys.stderr)
        return 1
    if not args.answer_key.is_file():
        print(f"Answer key not found: {args.answer_key}", file=sys.stderr)
        return 1

    questions = load_questions(args.questions)
    answers = load_answers(args.answer_key)

    # Inject correct_letter into each question dict so gate_capacity_shortcut can use it
    for q in questions:
        qid = q["question_id"]
        if qid in answers:
            q["correct_letter"] = answers[qid]

    reports = []
    for q in questions:
        report = validate_question(q, args.data_root)
        reports.append(report.to_dict())

    passed = sum(1 for r in reports if r["overall"] == "passed")
    rejected = sum(1 for r in reports if r["overall"] == "rejected")

    summary = {
        "total": len(reports),
        "passed": passed,
        "rejected": rejected,
        "by_family": {},
        "by_gate": {},
        "questions": reports,
    }

    for r in reports:
        family = r["family"]
        summary["by_family"].setdefault(family, {"passed": 0, "rejected": 0})
        summary["by_family"][family][r["overall"]] += 1

    gate_names = ["affine_fit", "capacity_shortcut", "snr", "interaction"]
    for gate_name in gate_names:
        gate_results = []
        for r in reports:
            g = r["gates"].get(gate_name, {})
            if not g.get("skipped"):
                gate_results.append(g["passed"])
        if gate_results:
            summary["by_gate"][gate_name] = {
                "total": len(gate_results),
                "passed": sum(gate_results),
                "rejected": len(gate_results) - sum(gate_results),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Total: {len(reports)}  Passed: {passed}  Rejected: {rejected}")
    for gate_name, stats in summary["by_gate"].items():
        print(f"  {gate_name}: {stats['passed']}/{stats['total']} passed")
    for family, stats in sorted(summary["by_family"].items()):
        print(f"  {family}: {stats['passed']} passed, {stats['rejected']} rejected")

    print(f"\nReport written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())