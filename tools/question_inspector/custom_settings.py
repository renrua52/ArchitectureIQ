"""Build and run user-defined settings from the question inspector."""

from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.profile import Profile
from architecture_iq.registry import (
    ensure_registries,
    get_dataset_family,
    get_model_type,
)
from architecture_iq.util import short_hash, write_json

CUSTOM_SETTINGS_DIR = "custom_settings"
SETTING_MANIFEST = "setting.json"
SETTING_INDEX = "index.json"
MAX_RETAINED_SETTINGS = 2


def compatible_model_types(profile: Profile, family: str) -> list[str]:
    """Return profile model types that can train on ``family``."""
    ensure_registries()
    compatible = set(get_dataset_family(family).compatible_model_types())
    return [name for name in profile.pools["model_types"] if name in compatible]


def build_model_spec(
    model_type: str,
    params: dict[str, Any],
    dataset_params: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate a model spec from inspector form values."""
    ensure_registries()
    if model_type == "mlp":
        depth = int(params["depth"])
        width = int(params["width"])
        if depth <= 0:
            raise ValueError("Depth must be greater than zero.")
        if width <= 0:
            raise ValueError("Width must be greater than zero.")
        activations = [str(value) for value in params["activations"]]
        layer_norm = [bool(value) for value in params["layer_norm"]]
        spec = {
            "type": "mlp",
            "depth": depth,
            "width": width,
            "residual": bool(params.get("residual", False)),
            "layer_norm": layer_norm,
            "activations": activations,
            "input_dim": int(dataset_params.get("input_dim", 1)),
        }
    elif model_type == "transformer_lm":
        d_model = int(params["d_model"])
        num_layers = int(params["num_layers"])
        num_heads = int(params["num_heads"])
        d_ff = int(params["d_ff"])
        if min(d_model, num_layers, num_heads, d_ff) <= 0:
            raise ValueError("Transformer dimensions must be greater than zero.")
        if "vocab_size" not in dataset_params or "context_length" not in dataset_params:
            raise ValueError("The selected dataset is missing language-model dimensions.")
        spec = {
            "type": "transformer_lm",
            "vocab_size": int(dataset_params["vocab_size"]),
            "context_length": int(dataset_params["context_length"]),
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
        }
    else:
        raise ValueError(f"Unsupported architecture: {model_type}")

    get_model_type(model_type).validate(spec)
    return spec


def build_optimizer_spec(
    optimizer_type: str,
    *,
    lr: float,
    weight_decay: float,
    momentum: float | None = None,
    betas: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build a validated optimizer spec."""
    if lr <= 0:
        raise ValueError("Learning rate must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("Weight decay cannot be negative.")
    if optimizer_type not in {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad"}:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    spec: dict[str, Any] = {
        "type": optimizer_type,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }
    if optimizer_type == "SGD":
        resolved_momentum = float(momentum or 0.0)
        if not 0 <= resolved_momentum < 1:
            raise ValueError("SGD momentum must be in [0, 1).")
        spec["momentum"] = resolved_momentum
    elif optimizer_type in {"Adam", "AdamW"}:
        beta1, beta2 = betas or (0.9, 0.999)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1).")
        spec["betas"] = [float(beta1), float(beta2)]
    return spec


def build_loss_spec(loss_id: str, *, lambda_value: float | None = None) -> dict[str, Any]:
    """Build a loss spec, including an optional regularization coefficient."""
    spec: dict[str, Any] = {"loss_id": loss_id}
    if loss_id.endswith("_l1") or loss_id.endswith("_l2"):
        resolved_lambda = 0.0 if lambda_value is None else float(lambda_value)
        if resolved_lambda < 0:
            raise ValueError("Loss lambda cannot be negative.")
        spec["lambda"] = resolved_lambda
    return spec


def form_values_from_candidate_spec(
    spec: dict[str, Any],
    *,
    source_letter: str,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map every editable candidate field to its inspector widget value."""
    budget = spec["budget"]
    model = spec["model"]
    optimizer = spec["optimizer"]
    loss = spec["loss"]
    values: dict[str, Any] = {
        "label": f"From {source_letter}",
        "budget": int(budget["total_samples_seen"]),
        "batch_size": int(budget["batch_size"]),
        "model_type": model["type"],
        "optimizer_type": optimizer["type"],
        "learning_rate": float(optimizer["lr"]),
        "weight_decay": float(optimizer.get("weight_decay", 0.0)),
        "loss": loss["loss_id"],
    }
    if evaluation:
        if "n_seeds" in evaluation:
            values["n_seeds"] = int(evaluation["n_seeds"])
        if "base_seed" in evaluation:
            values["base_seed"] = int(evaluation["base_seed"])

    if model["type"] == "mlp":
        values.update(
            {
                "mlp_depth": int(model["depth"]),
                "mlp_width": int(model["width"]),
                "mlp_residual": bool(model.get("residual", False)),
            }
        )
        for index, activation in enumerate(model["activations"]):
            values[f"mlp_activation_{index}"] = activation
        for index, use_norm in enumerate(model["layer_norm"]):
            values[f"mlp_norm_{index}"] = bool(use_norm)
    elif model["type"] == "transformer_lm":
        d_model = model["d_model"] if "d_model" in model else model["embed_dim"]
        d_ff = model["d_ff"] if "d_ff" in model else model["ff_dim"]
        values.update(
            {
                "transformer_d_model": int(d_model),
                "transformer_layers": int(model["num_layers"]),
                "transformer_heads": int(model["num_heads"]),
                "transformer_d_ff": int(d_ff),
            }
        )

    if optimizer["type"] == "SGD":
        values["momentum"] = float(optimizer.get("momentum", 0.0))
    elif optimizer["type"] in {"Adam", "AdamW"}:
        betas = optimizer.get("betas", [0.9, 0.999])
        values["beta1"] = float(betas[0])
        values["beta2"] = float(betas[1])
    if "lambda" in loss:
        values["loss_lambda"] = float(loss["lambda"])
    return values


def build_custom_setting_spec(
    profile: Profile,
    dataset_spec: dict[str, Any],
    *,
    budget: int,
    batch_size: int,
    model: dict[str, Any],
    optimizer: dict[str, Any],
    loss: dict[str, Any],
) -> dict[str, Any]:
    """Build a standard candidate spec for an inspector experiment."""
    family = str(dataset_spec["family"])
    dataset_id = str(dataset_spec["dataset_id"])
    if budget <= 0:
        raise ValueError("Total samples must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    if budget % batch_size:
        raise ValueError("Total samples must be divisible by batch size.")
    if model["type"] not in compatible_model_types(profile, family):
        raise ValueError(
            f"Architecture {model['type']!r} is not compatible with {family!r}."
        )
    allowed_losses = set(profile.pools["losses"][family])
    if loss["loss_id"] not in allowed_losses:
        raise ValueError(f"Loss {loss['loss_id']!r} is not compatible with {family!r}.")
    if optimizer["type"] not in profile.pools["optimizers"]:
        raise ValueError(f"Optimizer {optimizer['type']!r} is not enabled by the profile.")

    return build_candidate_spec(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=int(budget),
        batch_size=int(batch_size),
        model=model,
        optimizer=optimizer,
        loss=loss,
    )


def custom_setting_run_id(
    spec: dict[str, Any],
    *,
    n_seeds: int,
    base_seed: int,
    sequence: int,
) -> str:
    payload = {
        "candidate_id": spec["candidate_id"],
        "n_seeds": int(n_seeds),
        "base_seed": int(base_seed),
        "sequence": int(sequence),
    }
    return f"setting_{sequence:04d}_{short_hash(payload)}"


def _reserve_setting_identity(
    question_root: Path,
    spec: dict[str, Any],
    *,
    label_prefix: str,
    n_seeds: int,
    base_seed: int,
) -> tuple[str, str, int]:
    """Reserve a monotonically increasing id and display name."""
    root = question_root / CUSTOM_SETTINGS_DIR
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / SETTING_INDEX
    index: dict[str, Any] = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            index = {}
    existing_sequences = [
        int(run.get("sequence", 0))
        for run in list_custom_setting_runs(question_root)
    ]
    sequence = max(
        1,
        int(index.get("next_sequence", 1)),
        max(existing_sequences, default=0) + 1,
    )
    while True:
        run_id = custom_setting_run_id(
            spec,
            n_seeds=n_seeds,
            base_seed=base_seed,
            sequence=sequence,
        )
        if not (root / run_id).exists():
            break
        sequence += 1
    prefix = label_prefix.strip() or "Setting"
    label = f"{prefix} #{sequence:04d}"
    write_json(index_path, {"next_sequence": sequence + 1})
    return run_id, label, sequence


def _finite_metric(value: Any) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return metric if math.isfinite(metric) else float("inf")


def _run_final_metric(candidate_dir: Path, manifest: dict[str, Any]) -> float:
    if "final_metric" in manifest:
        return _finite_metric(manifest["final_metric"])
    summary_path = candidate_dir / "results" / "summary.json"
    if not summary_path.is_file():
        return float("inf")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return float("inf")
    metric = str(manifest.get("selection_metric", summary.get("selection_metric", "")))
    return _finite_metric(summary.get(f"mean_{metric}"))


def prune_custom_setting_runs(
    question_root: Path,
    *,
    newest_id: str,
) -> list[dict[str, Any]]:
    """Keep only the newest run and the lowest-loss historical run."""
    runs = list_custom_setting_runs(question_root)
    newest = next(
        (run for run in runs if run["custom_setting_id"] == newest_id),
        None,
    )
    keep_ids = {newest_id}
    historical = [run for run in runs if run["custom_setting_id"] != newest_id]
    if historical:
        best_historical = min(historical, key=lambda run: run["final_metric"])
        keep_ids.add(best_historical["custom_setting_id"])

    root = (question_root / CUSTOM_SETTINGS_DIR).resolve()
    for run in runs:
        if run["custom_setting_id"] in keep_ids:
            continue
        candidate_dir = Path(run["candidate_dir"]).resolve()
        if candidate_dir.parent == root:
            shutil.rmtree(candidate_dir)

    retained = list_custom_setting_runs(question_root)
    if newest is None:
        return retained[:MAX_RETAINED_SETTINGS]
    return retained


def enforce_custom_setting_retention(question_root: Path) -> list[dict[str, Any]]:
    """Apply the two-run retention rule to settings created by older versions."""
    runs = list_custom_setting_runs(question_root)
    if len(runs) <= MAX_RETAINED_SETTINGS:
        return runs
    return prune_custom_setting_runs(
        question_root,
        newest_id=runs[0]["custom_setting_id"],
    )


def run_custom_setting(
    question_root: Path,
    dataset_path: Path,
    profile: Profile,
    spec: dict[str, Any],
    *,
    label: str,
    n_seeds: int,
    base_seed: int,
    inherited_from: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize and train one setting without modifying benchmark candidates."""
    if n_seeds <= 0:
        raise ValueError("Number of seeds must be greater than zero.")
    run_id, unique_label, sequence = _reserve_setting_identity(
        question_root,
        spec,
        label_prefix=label,
        n_seeds=n_seeds,
        base_seed=base_seed,
    )
    output_dir = question_root / CUSTOM_SETTINGS_DIR / run_id

    ensure_registries()
    write_candidate(spec, output_dir, get_model_type(spec["model"]["type"]))
    run_profile = deepcopy(profile)
    run_profile.ground_truth["n_seeds"] = int(n_seeds)
    run_profile.ground_truth["base_seed"] = int(base_seed)
    try:
        summary = run_ground_truth(
            output_dir,
            run_profile,
            dataset_path=dataset_path,
            fail_threshold_override=float("inf"),
        )
    except Exception:
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        raise

    manifest = {
        "schema_version": profile.schema_version,
        "custom_setting_id": run_id,
        "sequence": sequence,
        "label": unique_label,
        "candidate_id": spec["candidate_id"],
        "n_seeds": int(n_seeds),
        "base_seed": int(base_seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": summary["selection_metric"],
        "final_metric": _finite_metric(
            summary.get(f"mean_{summary['selection_metric']}")
        ),
        **({"inherited_from": inherited_from} if inherited_from else {}),
    }
    write_json(output_dir / SETTING_MANIFEST, manifest)
    retained = prune_custom_setting_runs(question_root, newest_id=run_id)
    return {
        **manifest,
        "candidate_dir": output_dir,
        "summary": summary,
        "retained_settings": retained,
    }


def list_custom_setting_runs(question_root: Path) -> list[dict[str, Any]]:
    """Load completed custom settings for a question, newest first."""
    root = question_root / CUSTOM_SETTINGS_DIR
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for manifest_path in root.glob(f"*/{SETTING_MANIFEST}"):
        candidate_dir = manifest_path.parent
        curves_path = candidate_dir / "results" / "curves.npz"
        spec_path = candidate_dir / "candidate_spec.json"
        if not curves_path.is_file() or not spec_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        runs.append(
            {
                **manifest,
                "candidate_dir": candidate_dir,
                "final_metric": _run_final_metric(candidate_dir, manifest),
            }
        )
    return sorted(
        runs,
        key=lambda run: (
            int(run.get("sequence", 0)),
            str(run.get("created_at", "")),
        ),
        reverse=True,
    )
