#!/usr/bin/env python3
"""Bake curated ArchitectureIQ questions into static JSON for frontend/quiz."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_number(value: Any) -> Any:
    """Coerce floats to JSON-safe values (NaN/Inf → None)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_number(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_number(item) for key, item in value.items()}
    return value

ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "tools" / "question_inspector"
if str(INSPECTOR) not in sys.path:
    sys.path.insert(0, str(INSPECTOR))

from artifact_loader import (  # noqa: E402
    candidate_file_paths,
    list_question_dirs,
    load_question_bundle,
    read_json_file,
    read_text_file,
)
from candidate_curves import load_candidate_curves  # noqa: E402
from prompt_format import (  # noqa: E402
    format_loss_nl,
    format_model_spec_lines,
    format_optimizer_nl,
)

DEFAULT_DATA = ROOT / "examples" / "quiz_demo" / "bundle"
OUT = ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"
MAX_POINTS = 180
CHOICE_COLORS = ["#7c6cff", "#f26e4f", "#20a87e", "#2f7de1", "#e0b144"]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "-"
    return str(value)


def _flatten_spec(spec: dict[str, Any]) -> dict[str, Any]:
    model = spec.get("model", {})
    return {
        "training steps": spec.get("budget", {}).get("training_steps"),
        "batch size": spec.get("budget", {}).get("batch_size"),
        "total samples seen": spec.get("budget", {}).get("total_samples_seen"),
        "model type": model.get("type"),
        "layers": model.get("depth") or model.get("num_layers"),
        "width": model.get("width"),
        "d_model": model.get("d_model") or model.get("embed_dim"),
        "trainable parameter count": spec.get("trainable_parameter_count", "—"),
        "num_heads": model.get("num_heads"),
        "residual": model.get("residual"),
        "layer norm": model.get("layer_norm"),
        "activations": model.get("activations"),
        "optimizer": spec.get("optimizer", {}).get("type"),
        "learning rate": spec.get("optimizer", {}).get("lr"),
        "weight decay": spec.get("optimizer", {}).get("weight_decay"),
        "betas": spec.get("optimizer", {}).get("betas"),
        "momentum": spec.get("optimizer", {}).get("momentum"),
        "loss": spec.get("loss", {}).get("loss_id"),
        "lambda": spec.get("loss", {}).get("lambda"),
    }


def _shared_and_variant(
    specs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    flat = {letter: _flatten_spec(spec) for letter, spec in specs.items()}
    letters = list(flat)
    shared: list[dict[str, str]] = []
    variant: dict[str, list[dict[str, str]]] = {letter: [] for letter in letters}
    if not letters:
        return shared, variant
    for key in flat[letters[0]]:
        values = [flat[letter].get(key) for letter in letters]
        if all(value == values[0] for value in values):
            if values[0] is not None:
                shared.append({"label": key, "value": _fmt(values[0])})
        else:
            for letter, value in zip(letters, values, strict=True):
                if value is not None:
                    variant[letter].append({"label": key, "value": _fmt(value)})
    return shared, variant


def _downsample(values: list[float]) -> list[float]:
    if len(values) <= MAX_POINTS:
        return values
    step = max(1, len(values) // MAX_POINTS)
    return values[::step][:MAX_POINTS]


def _tensor_1d(tensor: torch.Tensor) -> list[float]:
    arr = tensor.detach().cpu().reshape(-1).float().tolist()
    return _downsample([float(x) for x in arr])


def _example_pair(family: str, train: dict[str, Any]) -> dict[str, Any] | None:
    if "x" not in train or "y" not in train:
        return None
    x = train["x"][0].detach().cpu()
    y = train["y"][0].detach().cpu()
    if family == "univariate_regression":
        return {
            "input": float(x.reshape(-1)[0]),
            "output": float(y.reshape(-1)[0]),
        }
    if family == "multivariate_regression":
        return {
            "input": [float(v) for v in x.reshape(-1).tolist()],
            "output": float(y.reshape(-1)[0]),
        }
    if family == "bigram_lm":
        return {
            "input": [int(v) for v in x.reshape(-1).tolist()],
            "output": [int(v) for v in y.reshape(-1).tolist()],
        }
    return {
        "input": x.tolist(),
        "output": y.tolist(),
    }


def _classification_feature_pair(params: dict[str, Any], input_dim: int) -> tuple[int, int, str]:
    active = [int(v) for v in params.get("active_features", []) if isinstance(v, int) and 0 <= int(v) < input_dim]
    rule_family = str(params.get("rule_family", ""))
    if rule_family == "sparse_interaction":
        pairs = params.get("interaction_pairs", [])
        weights = params.get("rule_weights", [])
        valid = []
        for i, pair in enumerate(pairs):
            if isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, int) for v in pair):
                weight = float(weights[i]) if i < len(weights) else 0.0
                valid.append((abs(weight), int(pair[0]), int(pair[1])))
        if valid:
            _, first, second = max(valid)
            return first, second, "largest-magnitude interaction"
    if rule_family == "smooth_additive" and len(active) >= 2:
        weights = params.get("rule_weights", [])
        ranked = sorted(((abs(float(weights[i])) if i < len(weights) else 0.0, v) for i, v in enumerate(active)), reverse=True)
        return ranked[0][1], ranked[1][1], "largest-magnitude additive effects"
    if len(active) >= 2:
        return active[0], active[1], "rule-aware active features"
    return 0, 1 if input_dim > 1 else 0, "available feature coordinates"


def _classification_points(x: np.ndarray, y: np.ndarray, first: int, second: int, maximum: int) -> list[dict[str, Any]]:
    labels = np.asarray(y).reshape(-1)
    points: list[dict[str, Any]] = []
    unique_labels = sorted({int(label) for label in labels.tolist() if np.isfinite(label)})
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        if len(indices) > maximum:
            indices = np.linspace(0, len(indices) - 1, maximum, dtype=int)
        for index in indices.tolist():
            points.append({"x": float(x[index, first]), "y": float(x[index, second]), "label": label})
    return points


def _classification_plot(train: dict[str, Any], test: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    x_train = train["x"].detach().cpu().numpy()
    y_train = train["y"].detach().cpu().reshape(-1).numpy().astype(int)
    x_test = test["x"].detach().cpu().numpy()
    y_test = test["y"].detach().cpu().reshape(-1).numpy().astype(int)
    if x_train.ndim != 2 or x_train.shape[1] < 2:
        return {"kind": "none", "reason": "classification dataset has fewer than two features"}
    first, second, note = _classification_feature_pair(params, x_train.shape[1])
    first_values = x_train[:, first]
    second_values = x_train[:, second]
    finite = np.isfinite(first_values) & np.isfinite(second_values) & np.isfinite(y_train)
    first_values = first_values[finite]
    second_values = second_values[finite]
    labels = y_train[finite]
    bins = 24
    def edges(values: np.ndarray) -> np.ndarray:
        low, high = float(np.min(values)), float(np.max(values))
        if low == high:
            low -= 0.5
            high += 0.5
        margin = 0.05 * (high - low)
        return np.linspace(low - margin, high + margin, bins + 1)
    x_edges, y_edges = edges(first_values), edges(second_values)
    counts, _, _ = np.histogram2d(first_values, second_values, bins=(x_edges, y_edges))
    positive, _, _ = np.histogram2d(first_values[labels == 1], second_values[labels == 1], bins=(x_edges, y_edges))
    probability = np.full(counts.shape, np.nan, dtype=float)
    np.divide(positive, counts, out=probability, where=counts > 0)
    return {
        "kind": "classification",
        "train": _classification_points(x_train, y_train, first, second, 220),
        "test": _classification_points(x_test, y_test, first, second, 90),
        "xEdges": x_edges.tolist(),
        "yEdges": y_edges.tolist(),
        "probability": _json_number(probability.tolist()),
        "featurePair": [first, second],
        "selectionNote": note,
        "xLabel": f"feature {first}",
        "yLabel": f"feature {second}",
        "legend": "empirical P(class 1)",
        "min": 0.0,
        "max": 1.0,
    }


def _dataset_payload(dataset_dir: Path, family: str) -> dict[str, Any]:
    spec = read_json_file(dataset_dir / "dataset_spec.json")
    params = spec.get("params", {})
    out: dict[str, Any] = {
        "family": family,
        "datasetId": spec.get("dataset_id") or dataset_dir.name,
        "selectionMetric": spec.get("selection_metric"),
        "params": params,
        "files": {
            name: read_text_file(dataset_dir / name)
            if name.endswith(".py")
            else read_json_file(dataset_dir / name)
            for name in ("dataset_spec.json", "synthesize.py")
            if (dataset_dir / name).is_file()
        },
    }

    train_path = dataset_dir / "train.pt"
    test_path = dataset_dir / "test.pt"
    train = None
    if train_path.is_file():
        train = torch.load(train_path, weights_only=True)
        example = _example_pair(family, train)
        if example is not None:
            out["example"] = example

    if family == "univariate_regression" and train is not None and test_path.is_file():
        test = torch.load(test_path, weights_only=True)
        tx, ty = _tensor_1d(train["x"]), _tensor_1d(train["y"])
        vx, vy = _tensor_1d(test["x"]), _tensor_1d(test["y"])
        out["plot"] = {
            "kind": "scatter",
            "train": [{"x": x, "y": y} for x, y in zip(tx, ty, strict=False)],
            "test": [{"x": x, "y": y} for x, y in zip(vx, vy, strict=False)],
        }
    elif family == "synthetic_tabular_classification" and train is not None and test_path.is_file():
        test = torch.load(test_path, weights_only=True)
        out["plot"] = _classification_plot(train, test, params)
    elif family == "bigram_lm":
        transition = dataset_dir / "transition.npz"
        if transition.is_file():
            import numpy as np

            data = np.load(transition)
            matrix = np.asarray(data["P"] if "P" in data else data[data.files[0]], dtype=float)
            out["plot"] = {
                "kind": "heatmap",
                "matrix": matrix.tolist(),
                "xLabel": "next token",
                "yLabel": "current token",
                "legend": "P(next | current)",
                "min": float(np.nanmin(matrix)),
                "max": float(np.nanmax(matrix)),
            }
        out["plot"] = out.get("plot") or {"kind": "none"}
    else:
        out["plot"] = {"kind": "none"}
        if train is not None:
            x = train.get("x")
            if x is not None:
                out["tensorShapes"] = {
                    "trainX": list(x.shape),
                    "trainY": list(train["y"].shape) if "y" in train else None,
                }

    return out


def _curve_series(choice: dict[str, Any]) -> dict[str, Any] | None:
    candidate_dir = choice["candidate_dir"]
    spec = read_json_file(candidate_dir / "candidate_spec.json")
    budget = spec.get("budget", {})
    curves_path = candidate_dir / "results" / "curves.npz"
    loaded = load_candidate_curves(
        curves_path,
        total_samples_seen=budget.get("total_samples_seen"),
        batch_size=budget.get("batch_size"),
    )
    if "error" in loaded:
        return None
    curves = loaded["curves"]
    samples = loaded["eval_samples"]
    mean = _json_number(curves.mean(axis=0).tolist())
    std = (
        _json_number(curves.std(axis=0).tolist())
        if curves.shape[0] > 1
        else [0.0] * len(mean)
    )
    return {
        "letter": choice["letter"],
        "samples": samples,
        "mean": mean,
        "std": std,
    }


def _summary_row(choice: dict[str, Any]) -> dict[str, Any]:
    summary_path = choice["candidate_dir"] / "results" / "summary.json"
    summary = read_json_file(summary_path) if summary_path.is_file() else {}
    metric = summary.get("selection_metric", "test_mse")
    mean = summary.get(f"mean_{metric}")
    std = summary.get(f"std_{metric}")
    label = "Metrics unavailable"
    if mean is not None and "error" not in summary:
        if std is None:
            label = f"{metric}: {mean:.6f}"
        else:
            label = f"{metric}: {mean:.6f} ± {std:.6f}"
    return {
        "letter": choice["letter"],
        "candidateId": choice["candidate_id"],
        "metric": metric,
        "mean": _json_number(mean) if isinstance(mean, float) else mean,
        "std": _json_number(std) if isinstance(std, float) else std,
        "label": label,
    }


def _title(question: dict[str, Any]) -> str:
    metric = (
        question.get("significance", {}).get("metric")
        or question.get("evaluation", {}).get("selection_metric")
        or "test metric"
    )
    return f"Which training setup will reach the lowest {str(metric).replace('_', ' ')}?"


def bake_question(question_dir: Path, data_root: Path) -> dict[str, Any]:
    bundle = load_question_bundle(question_dir, data_root)
    q = bundle.question
    specs = {
        choice["letter"]: read_json_file(choice["candidate_dir"] / "candidate_spec.json")
        for choice in bundle.choices
    }
    shared, variant = _shared_and_variant(specs)

    choices_public = []
    for index, choice in enumerate(bundle.choices):
        letter = choice["letter"]
        spec = specs[letter]
        files = {}
        for name, path in candidate_file_paths(choice["candidate_dir"], include_summary=False).items():
            if name.endswith(".json"):
                files[name] = read_json_file(path)
            else:
                files[name] = read_text_file(path)
        choices_public.append(
            {
                "letter": letter,
                "candidateId": choice["candidate_id"],
                "color": CHOICE_COLORS[index % len(CHOICE_COLORS)],
                "variant": variant.get(letter, []),
                "modelLines": format_model_spec_lines(spec.get("model", {})),
                "optimizerLines": [
                    line[2:] if line.startswith("- ") else line
                    for line in format_optimizer_nl(spec.get("optimizer", {})).splitlines()
                ],
                "lossLines": [
                    line[2:] if line.startswith("- ") else line
                    for line in format_loss_nl(spec.get("loss", {})).splitlines()
                ],
                "files": files,
            }
        )

    ranked = [_summary_row(choice) for choice in bundle.choices]
    ranked.sort(key=lambda row: (row["mean"] is None, row["mean"] if row["mean"] is not None else float("inf")))

    reveal_files = {}
    for choice in bundle.choices:
        summary_path = choice["candidate_dir"] / "results" / "summary.json"
        if summary_path.is_file():
            reveal_files[choice["letter"]] = {"summary.json": read_json_file(summary_path)}

    budget = q.get("budget", {})
    samples = budget.get("total_samples_seen") if isinstance(budget, dict) else budget

    return {
        "id": q["question_id"],
        "title": _title(q),
        "family": q.get("family"),
        "datasetId": q.get("dataset_id"),
        "type": q.get("type"),
        "profile": q.get("profile"),
        "budget": budget,
        "metric": q.get("significance", {}).get("metric")
        or q.get("evaluation", {}).get("selection_metric"),
        "evaluation": q.get("evaluation", {}),
        "invariantAxes": q.get("invariant_axes", []),
        "varyingAxes": q.get("varying_axes", []),
        "numChoices": q.get("num_choices", len(bundle.choices)),
        "detail": {
            "prompt": bundle.prompt_text,
            "shared": shared,
            "dataset": _dataset_payload(bundle.dataset_dir, str(q.get("family"))),
            "choices": choices_public,
        },
        "reveal": {
            "correctLetter": q["correct_letter"],
            "ranked": ranked,
            "curves": [
                series
                for series in (_curve_series(choice) for choice in bundle.choices)
                if series is not None
            ],
            "files": reveal_files,
        },
        "summary": {
            "id": q["question_id"],
            "type": q.get("type"),
            "datasetId": q.get("dataset_id"),
            "family": q.get("family"),
            "budget": samples,
            "choices": len(bundle.choices),
            "metric": q.get("significance", {}).get("metric")
            or q.get("evaluation", {}).get("selection_metric"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA,
        help="Directory containing datasets/ (default: examples/quiz_demo/bundle)",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional question-run directory name filter (repeatable), e.g. run_20q_3c_b09206",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    dirs = list_question_dirs(data_root)
    if args.run:
        allowed = set(args.run)
        dirs = [path for path in dirs if path.parent.name in allowed]
    if not dirs:
        raise SystemExit(f"No questions found under {data_root}")

    baked = [bake_question(path, data_root) for path in dirs]
    # Prefer univariate first for nicer default ordering, then others.
    baked.sort(key=lambda item: (0 if item["family"] == "univariate_regression" else 1, item["id"]))
    payload = {
        "schema_version": 1,
        "questions": [item["summary"] for item in baked],
        "byId": {item["id"]: item for item in baked},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_number(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"exported {len(baked)} questions → {args.out}")


if __name__ == "__main__":
    main()
