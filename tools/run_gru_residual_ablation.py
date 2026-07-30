#!/usr/bin/env python3
"""Paired one-seed GRU residual ablation on existing ArchitectureIQ candidates.

The experiment is deliberately outside the candidate/profile pipeline. It keeps
all candidate settings fixed and changes only the GRU stack from PyTorch's
multi-layer GRU to a stack of single-layer GRUs with ``h <- h + GRU(h)`` after
every layer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture_iq.models.gru_lm import CausalGruLM

DEFAULT_CANDIDATES = (
    "c_4184cf",
    "c_a8f5c3",
    "c_ad8bf4",
    "c_4c058b",
    "c_999e9b",
)


class LayerResidualGruLM(nn.Module):
    """Causal GRU LM with an identity residual after every recurrent layer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        for layer in self.layers:
            residual = h
            h, _ = layer(h)
            h = h + residual
        return self.head(h)


@dataclass(frozen=True)
class RunResult:
    curve: list[float]
    initial_parameters_match: bool
    parameters: int


def _candidate_spec(candidate_dir: Path) -> dict[str, Any]:
    return json.loads((candidate_dir / "candidate_spec.json").read_text(encoding="utf-8"))


def _load_data(dataset_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train = torch.load(dataset_dir / "train.pt", weights_only=True)
    test = torch.load(dataset_dir / "test.pt", weights_only=True)
    return train["x"], train["y"], test["x"], test["y"]


def _residual_model(spec: dict[str, Any]) -> LayerResidualGruLM:
    model = spec["model"]
    return LayerResidualGruLM(
        vocab_size=int(model["vocab_size"]),
        context_length=int(model["context_length"]),
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
    )


def _baseline_model(spec: dict[str, Any]) -> CausalGruLM:
    model = spec["model"]
    return CausalGruLM(
        vocab_size=int(model["vocab_size"]),
        context_length=int(model["context_length"]),
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
    )


def _initial_parameters_match(baseline: nn.Module, residual: nn.Module) -> bool:
    baseline_state = baseline.state_dict()
    residual_state = residual.state_dict()
    expected: dict[str, torch.Tensor] = {
        "token_embed.weight": baseline_state["token_embed.weight"],
        "head.weight": baseline_state["head.weight"],
        "head.bias": baseline_state["head.bias"],
    }
    num_layers = len(residual.layers)  # type: ignore[arg-type]
    for layer in range(num_layers):
        for parameter in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            expected[f"layers.{layer}.{parameter}"] = baseline_state[f"gru.{parameter[:-1]}{layer}"]
    return set(expected) == set(residual_state) and all(
        torch.equal(expected[name], residual_state[name]) for name in expected
    )


def _test_ce(model: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        logits = model(test_x)
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), test_y.reshape(-1)).item())


def _train_one(
    model: nn.Module,
    *,
    spec: dict[str, Any],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
) -> list[float]:
    budget = spec["budget"]
    optimizer_spec = spec["optimizer"]
    torch.manual_seed(seed)
    # Recreate the candidate model after resetting the global generator, exactly
    # as the generated train.py does. The caller has already constructed an
    # aligned model for validation; create the train instance here for the paired
    # RNG protocol.
    if isinstance(model, LayerResidualGruLM):
        model = _residual_model(spec)
    else:
        model = _baseline_model(spec)
    model = model.cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_spec["lr"]),
        betas=tuple(float(value) for value in optimizer_spec["betas"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    steps = int(budget["training_steps"])
    batch_size = int(budget["batch_size"])
    curve: list[float] = []
    for _step in range(steps):
        model.train()
        idx = torch.randint(0, train_x.shape[0], (batch_size,))
        logits = model(train_x[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), train_y[idx].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append(_test_ce(model, test_x, test_y))
    return curve


def _run_pair(
    spec: dict[str, Any],
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
) -> tuple[RunResult, RunResult]:
    torch.manual_seed(seed)
    baseline_for_check = _baseline_model(spec)
    torch.manual_seed(seed)
    residual_for_check = _residual_model(spec)
    aligned = _initial_parameters_match(baseline_for_check, residual_for_check)
    baseline_curve = _train_one(
        baseline_for_check,
        spec=spec,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        seed=seed,
    )
    residual_curve = _train_one(
        residual_for_check,
        spec=spec,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        seed=seed,
    )
    return (
        RunResult(baseline_curve, aligned, sum(parameter.numel() for parameter in baseline_for_check.parameters())),
        RunResult(residual_curve, aligned, sum(parameter.numel() for parameter in residual_for_check.parameters())),
    )


def _stored_seed_zero(candidate_dir: Path) -> np.ndarray:
    with np.load(candidate_dir / "results" / "curves.npz", allow_pickle=False) as data:
        return np.asarray(data["curves"], dtype=float)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidate-set", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    candidate_set = args.candidate_set.resolve()
    output_dir = args.output_dir.resolve()
    candidates = tuple(args.candidate) or DEFAULT_CANDIDATES
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    train_x, train_y, test_x, test_y = _load_data(dataset_dir)
    rows: list[dict[str, Any]] = []
    baseline_curves: list[np.ndarray] = []
    residual_curves: list[np.ndarray] = []
    for candidate_id in candidates:
        candidate_dir = candidate_set / candidate_id
        spec = _candidate_spec(candidate_dir)
        if spec["model"]["type"] != "gru_lm":
            raise ValueError(f"{candidate_id} is not a GRU candidate")
        baseline, residual = _run_pair(
            spec,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            seed=args.seed,
        )
        stored = _stored_seed_zero(candidate_dir)
        baseline_curve = np.asarray(baseline.curve, dtype=float)
        residual_curve = np.asarray(residual.curve, dtype=float)
        baseline_curves.append(baseline_curve)
        residual_curves.append(residual_curve)
        baseline_delta = float(baseline_curve[7] - baseline_curve[-1])
        residual_delta = float(residual_curve[7] - residual_curve[-1])
        rows.append(
            {
                "candidate_id": candidate_id,
                "num_layers": int(spec["model"]["num_layers"]),
                "d_model": int(spec["model"]["d_model"]),
                "seed": args.seed,
                "initial_parameters_match": baseline.initial_parameters_match,
                "baseline_parameters": baseline.parameters,
                "residual_parameters": residual.parameters,
                "baseline_step_8_test_ce": float(baseline_curve[7]),
                "baseline_step_160_test_ce": float(baseline_curve[-1]),
                "baseline_drop_8_to_160": baseline_delta,
                "residual_step_8_test_ce": float(residual_curve[7]),
                "residual_step_160_test_ce": float(residual_curve[-1]),
                "residual_drop_8_to_160": residual_delta,
                "residual_minus_baseline_final_test_ce": float(residual_curve[-1] - baseline_curve[-1]),
                "residual_minus_baseline_drop": float(residual_delta - baseline_delta),
                "baseline_seed0_reproduction_max_abs_error": float(np.max(np.abs(baseline_curve - stored))),
            }
        )

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "curves.npz",
        candidate_ids=np.asarray(candidates),
        baseline=np.asarray(baseline_curves),
        layer_residual=np.asarray(residual_curves),
        samples=np.arange(1, 161, dtype=np.int64) * 32,
    )
    payload = {
        "schema_version": "gru_layer_residual_ablation_v1",
        "dataset": str(dataset_dir),
        "candidate_set": str(candidate_set),
        "seed": args.seed,
        "residual_definition": "After every one-layer GRU: h = h + GRU_layer(h). No normalization or extra parameters.",
        "rows": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GRU layer-residual one-seed ablation",
        "",
        "- Seed: `0`",
        "- Residual: after every one-layer GRU, `h = h + GRU_layer(h)`.",
        "- All other settings are copied from each selected candidate.",
        "",
        "| candidate | layers | width | baseline 8→160 | residual 8→160 | final residual-baseline | drop gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {num_layers} | {d_model} | {baseline_step_8_test_ce:.4f} → {baseline_step_160_test_ce:.4f} "
            "({baseline_drop_8_to_160:+.4f}) | {residual_step_8_test_ce:.4f} → {residual_step_160_test_ce:.4f} "
            "({residual_drop_8_to_160:+.4f}) | {residual_minus_baseline_final_test_ce:+.4f} | {residual_minus_baseline_drop:+.4f} |".format(**row)
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
