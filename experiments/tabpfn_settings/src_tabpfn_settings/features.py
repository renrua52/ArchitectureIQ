"""Encode ArchitectureIQ candidate_spec (+ summary) into tabular rows for TabPFN."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FEATURE_COLUMNS: list[str] = [
    "model_type",
    "trainable_parameter_count",
    "log_params",
    "batch_size",
    "training_steps",
    "total_samples_seen",
    "optimizer_type",
    "lr",
    "log_lr",
    "weight_decay",
    "momentum",
    "loss_id",
    "loss_lambda",
    # MLP / shared depth-width style
    "depth",
    "width",
    "residual",
    "layer_norm_frac",
    "activation_primary",
    # KAN
    "grid_size",
    "spline_order",
    "base_activation",
    # LM-ish (optional later)
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "layer_residual",
]

TARGET_CANDIDATES = (
    "mean_test_ce",
    "mean_test_mse",
    "mean_test_accuracy",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_log(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return float(math.log(v))


def encode_spec(spec: dict[str, Any], summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten one candidate into a feature dict + optional targets."""
    model = spec.get("model") or {}
    optimizer = spec.get("optimizer") or {}
    loss = spec.get("loss") or {}
    budget = spec.get("budget") or {}

    params = spec.get("trainable_parameter_count")
    if params is None and summary is not None:
        params = None

    layer_norm = model.get("layer_norm") or []
    if isinstance(layer_norm, list) and layer_norm:
        ln_frac = sum(1 for x in layer_norm if x) / len(layer_norm)
    else:
        ln_frac = None

    activations = model.get("activations") or []
    if isinstance(activations, list) and activations:
        act_primary = str(activations[0])
    else:
        act_primary = None

    depth = model.get("depth")
    width = model.get("width")
    num_layers = model.get("num_layers")
    if depth is None and num_layers is not None:
        depth = num_layers
    d_model = model.get("d_model")
    if width is None and d_model is not None:
        width = d_model

    row: dict[str, Any] = {
        "candidate_id": spec.get("candidate_id"),
        "family": spec.get("family"),
        "dataset_id": spec.get("dataset_id"),
        "profile": spec.get("profile"),
        "model_type": model.get("type"),
        "trainable_parameter_count": params,
        "log_params": _safe_log(params),
        "batch_size": budget.get("batch_size"),
        "training_steps": budget.get("training_steps"),
        "total_samples_seen": budget.get("total_samples_seen"),
        "optimizer_type": optimizer.get("type"),
        "lr": optimizer.get("lr"),
        "log_lr": _safe_log(optimizer.get("lr")),
        "weight_decay": optimizer.get("weight_decay"),
        "momentum": optimizer.get("momentum"),
        "loss_id": loss.get("loss_id"),
        "loss_lambda": loss.get("lambda"),
        "depth": depth,
        "width": width,
        "residual": model.get("residual"),
        "layer_norm_frac": ln_frac,
        "activation_primary": act_primary,
        "grid_size": model.get("grid_size"),
        "spline_order": model.get("spline_order"),
        "base_activation": model.get("base_activation"),
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": model.get("num_heads"),
        "d_ff": model.get("d_ff"),
        "layer_residual": model.get("layer_residual"),
        "selection_metric": (summary or {}).get("selection_metric"),
        "excluded": bool((summary or {}).get("excluded", False)),
    }

    if summary:
        for key in TARGET_CANDIDATES:
            if key in summary:
                row[key] = summary[key]
        # also copy mean_* dynamically if present
        for key, value in summary.items():
            if key.startswith("mean_") and key not in row:
                row[key] = value

    return row


def iter_candidate_dirs(pack_root: Path) -> list[Path]:
    """Find candidate dirs that have both candidate_spec.json and results/summary.json."""
    out: list[Path] = []
    for spec_path in pack_root.rglob("candidate_spec.json"):
        cand_dir = spec_path.parent
        summary = cand_dir / "results" / "summary.json"
        if summary.is_file():
            out.append(cand_dir)
    return sorted(out)


def rows_from_pack(pack_root: Path, *, skip_excluded: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand_dir in iter_candidate_dirs(pack_root):
        spec = load_json(cand_dir / "candidate_spec.json")
        summary = load_json(cand_dir / "results" / "summary.json")
        row = encode_spec(spec, summary)
        row["source_dir"] = str(cand_dir.relative_to(pack_root))
        cid = str(row.get("candidate_id") or cand_dir.name)
        if cid in seen:
            continue
        seen.add(cid)
        if skip_excluded and row.get("excluded"):
            continue
        rows.append(row)
    return rows
