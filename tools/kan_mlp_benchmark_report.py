"""Summarise controlled KAN/MLP calibration runs and simple baselines.

Calibration files produced by :mod:`generate_kan_mlp_demo` contain one row per
model and pair.  This tool deliberately consumes those recorded rows instead
of re-running candidates, so the report remains a pure analysis of the frozen
benchmark artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "kan_mlp_calibration"
DEFAULT_OUTPUT = ROOT / "outputs" / "kan_mlp_benchmark_report"
MODEL_TYPES = ("kan", "mlp")


def _number(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(default)


def _model_type(row: dict[str, Any]) -> str:
    model = row.get("model")
    value = row.get("label") or row.get("model_type")
    if value is None and isinstance(model, dict):
        value = model.get("type")
    return str(value or "").strip().lower()


def _parameter_count(model: dict[str, Any]) -> int:
    """Count trainable parameters from a model spec when a row lacks a count."""
    kind = str(model.get("type", "")).lower()
    if kind == "mlp":
        input_dim = int(model.get("input_dim", 1))
        output_dim = int(model.get("output_dim", 1))
        width = int(model["width"])
        depth = int(model["depth"])
        total = input_dim * width + width
        for i in range(depth):
            total += width * width + width
            norms = model.get("layer_norm", [])
            if i < len(norms) and bool(norms[i]):
                total += 2 * width
        return total + width * output_dim + output_dim
    if kind == "kan":
        input_dim = int(model.get("input_dim", 1))
        output_dim = int(model.get("output_dim", 1))
        width = int(model["width"])
        depth = int(model["depth"])
        basis_factor = 1 + int(model["grid_size"]) + int(model["spline_order"])
        dims = [input_dim] + [width] * (depth + 1) + [output_dim]
        return sum(dims[i] * dims[i + 1] * basis_factor for i in range(len(dims) - 1))
    raise ValueError(f"Cannot infer parameter count for model type {kind!r}")


def normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row with stable fields used by the report."""
    out = dict(row)
    out["model_type"] = _model_type(row)
    if out["model_type"] not in MODEL_TYPES:
        raise ValueError(f"Expected KAN or MLP row, got {out['model_type']!r}")
    if "parameters" not in out and "num_params" in out:
        out["parameters"] = out["num_params"]
    if "parameters" not in out:
        out["parameters"] = _parameter_count(row["model"])
    out["parameters"] = int(float(out["parameters"]))
    out["elapsed_seconds"] = _number(
        row, "elapsed_seconds", "runtime_seconds", "elapsed", "time_seconds"
    )
    seeds = row.get("seed_results")
    if isinstance(seeds, list):
        total = len(seeds)
        failed = sum(1 for seed in seeds if bool(seed.get("failed")))
    else:
        total = int(_number(row, "seeds", "n_seeds", default=0))
        failed = int(_number(row, "failed_seeds", "failures", default=0))
    out["seeds"] = total
    out["failed_seeds"] = failed
    out["failure_rate"] = failed / total if total else 0.0
    out["metric"] = _number(
        row, "mean_test_mse", "mean_metric", "final_metric", "metric", default=float("nan")
    )
    out["dataset_id"] = str(row.get("dataset_id", ""))
    out["budget"] = int(_number(row, "budget", default=0))
    out["pair"] = str(row.get("pair") or row.get("pair_id") or "")
    if not out["pair"]:
        out["pair"] = f"{out['dataset_id']}::{out['budget']}"
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load calibration rows from one JSON file or a directory."""
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    rows: list[dict[str, Any]] = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        values = payload.get("rows", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON list or {{rows: [...]}}: {file}")
        config_hash = payload.get("pair_config_hash") if isinstance(payload, dict) else None
        for value in values:
            row = dict(value)
            if config_hash and "pair_config_hash" not in row:
                row["pair_config_hash"] = config_hash
            rows.append(normalise_row(row))
    return rows


def pair_groups(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("dataset_id", "")), str(row["pair"]), int(row.get("budget", 0)))
        grouped[key][row["model_type"]] = row
    result = []
    for (dataset_id, pair, budget), models in sorted(grouped.items()):
        if not all(kind in models for kind in MODEL_TYPES):
            continue
        kan, mlp = models["kan"], models["mlp"]
        km, mm = float(kan["metric"]), float(mlp["metric"])
        if km < mm:
            winner = "kan"
        elif mm < km:
            winner = "mlp"
        else:
            winner = "tie"
        result.append(
            {
                "dataset_id": dataset_id,
                "pair": pair,
                "budget": budget,
                "kan": kan,
                "mlp": mlp,
                "winner": winner,
            }
        )
    return result


def _accuracy(predictions: list[tuple[str, str]]) -> dict[str, Any]:
    correct = sum(pred == truth for pred, truth in predictions)
    return {
        "n": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions) if predictions else None,
        "predictions": [{"predicted": p, "actual": a, "ok": p == a} for p, a in predictions],
    }


def evaluate_baselines(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate fixed, parameter, runtime, and lookup baselines.

    ``lookup`` is keyed by dataset and budget (the protocol's natural
    instance key), and is intentionally reported as an in-sample diagnostic;
    it is not a claim of blind generalisation.
    """
    fixed = {
        "always_kan": lambda g: "kan",
        "always_mlp": lambda g: "mlp",
        "more_params": lambda g: "kan" if g["kan"]["parameters"] > g["mlp"]["parameters"] else ("mlp" if g["mlp"]["parameters"] > g["kan"]["parameters"] else "tie"),
        "fewer_params": lambda g: "kan" if g["kan"]["parameters"] < g["mlp"]["parameters"] else ("mlp" if g["mlp"]["parameters"] < g["kan"]["parameters"] else "tie"),
        "faster_runtime": lambda g: "kan" if g["kan"]["elapsed_seconds"] < g["mlp"]["elapsed_seconds"] else ("mlp" if g["mlp"]["elapsed_seconds"] < g["kan"]["elapsed_seconds"] else "tie"),
    }
    result = {}
    for name, predictor in fixed.items():
        result[name] = _accuracy([(predictor(g), g["winner"]) for g in groups])

    # Dataset+budget lookup: use the majority observed winner for each key.
    by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for group in groups:
        by_key[(group["dataset_id"], group["budget"])].append(group["winner"])
    lookup: dict[tuple[str, int], str] = {}
    for key, winners in by_key.items():
        counts = Counter(winners)
        best = counts.most_common()
        lookup[key] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else "tie"
    result["lookup"] = _accuracy(
        [(lookup[(g["dataset_id"], g["budget"])], g["winner"]) for g in groups]
    )
    return result


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = pair_groups(rows)
    model_summary: dict[str, Any] = {}
    for kind in MODEL_TYPES:
        values = [row for row in rows if row["model_type"] == kind]
        params = [int(row["parameters"]) for row in values]
        runtimes = [float(row["elapsed_seconds"]) for row in values]
        failed = sum(int(row["failed_seeds"]) for row in values)
        seeds = sum(int(row["seeds"]) for row in values)
        wins = sum(group["winner"] == kind for group in groups)
        model_summary[kind] = {
            "rows": len(values),
            "pairs": wins,
            "wins": wins,
            "win_rate": wins / len(groups) if groups else None,
            "parameters": {
                "mean": statistics.mean(params) if params else None,
                "median": statistics.median(params) if params else None,
                "min": min(params) if params else None,
                "max": max(params) if params else None,
            },
            "runtime_seconds": {
                "mean": statistics.mean(runtimes) if runtimes else None,
                "median": statistics.median(runtimes) if runtimes else None,
                "total": sum(runtimes),
            },
            "failed_seeds": failed,
            "seeds": seeds,
            "failure_rate": failed / seeds if seeds else 0.0,
        }
    pair_config_hashes = sorted(
        {str(row["pair_config_hash"]) for row in rows if row.get("pair_config_hash")}
    )
    return {
        "n_rows": len(rows),
        "n_pairs": len(groups),
        "pair_config_hashes": pair_config_hashes,
        "models": model_summary,
        "baselines": evaluate_baselines(groups),
        "pairs": [
            {
                "dataset_id": g["dataset_id"],
                "pair": g["pair"],
                "budget": g["budget"],
                "winner": g["winner"],
                "kan_parameters": g["kan"]["parameters"],
                "mlp_parameters": g["mlp"]["parameters"],
                "kan_runtime_seconds": g["kan"]["elapsed_seconds"],
                "mlp_runtime_seconds": g["mlp"]["elapsed_seconds"],
            }
            for g in groups
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# KAN/MLP Benchmark Report", "", f"Pairs: {report['n_pairs']}  Rows: {report['n_rows']}", "", "## Model summary", "", "| model | pairs won | win rate | mean params | mean runtime (s) | failure rate |", "|---|---:|---:|---:|---:|---:|"]
    for kind in MODEL_TYPES:
        model = report["models"][kind]
        lines.append(f"| {kind} | {model['wins']} | {model['win_rate']:.3f} | {model['parameters']['mean']:.1f} | {model['runtime_seconds']['mean']:.3f} | {model['failure_rate']:.3f} |" )
    lines.extend(["", "## Baselines", "", "| baseline | correct | accuracy |", "|---|---:|---:|"])
    for name, result in report["baselines"].items():
        accuracy = result["accuracy"]
        lines.append(f"| {name} | {result['correct']}/{result['n']} | {'n/a' if accuracy is None else f'{accuracy:.3f}'} |" )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Calibration JSON file(s) or directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = args.inputs or [DEFAULT_INPUT]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        print(f"Input not found: {missing[0]}", file=sys.stderr)
        return 2
    rows = []
    for path in inputs:
        rows.extend(load_rows(path))
    if not rows:
        print("No calibration rows found.", file=sys.stderr)
        return 1
    report = summarise(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"n_rows": report["n_rows"], "n_pairs": report["n_pairs"], "baselines": {k: v["accuracy"] for k, v in report["baselines"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
