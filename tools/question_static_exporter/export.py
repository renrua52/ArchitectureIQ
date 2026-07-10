"""Export ArchitectureIQ questions as a static offline quiz.

The generated folder is intentionally browser-native: users can unzip it and
open ``index.html`` directly on Windows or macOS without installing Python,
Streamlit, PyTorch, or project dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

# Some Windows conda environments load more than one OpenMP runtime when torch,
# numpy, and matplotlib are imported together. The exporter only reads tensors
# and renders static plots, so prefer a successful offline package build.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "question_inspector"))
from artifact_loader import (  # noqa: E402
    candidate_file_paths,
    dataset_file_paths,
    format_metrics,
    list_question_dirs,
    load_question_bundle,
    read_json_file,
    read_text_file,
)
from candidate_curves import load_candidate_curves  # noqa: E402
from expression_latex import expression_to_latex  # noqa: E402
from prompt_format import format_model_spec_lines  # noqa: E402


EXPORT_SCHEMA_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class ExportOptions:
    data_root: Path
    out_dir: Path
    title: str
    limit: int | None
    zip_path: Path | None
    include_code: bool
    overwrite: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_js(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(f"window.ARCHITECTURE_IQ_DATA = {text};\n", encoding="utf-8")


def _copy_templates(out_dir: Path) -> None:
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy2(TEMPLATES / name, out_dir / name)


def _write_package_readme(out_dir: Path) -> None:
    readme = """ArchitectureIQ Quiz - Offline Static Version

How to use
1. Unzip ArchitectureIQ-quiz.zip.
2. Open the ArchitectureIQ-quiz folder.
3. Double-click index.html.
4. Use Next/Random or the question dropdown to navigate.
5. Select an answer to reveal the correct answer, ranked metrics, and learning curves.

Notes
- Works offline after unzipping.
- No Python, Streamlit, PyTorch, or repository checkout is required for quiz users.
- Supported target users: Windows and macOS users with a modern browser.
- Answers are embedded in the local static files, so this package is for practice, demos, or teaching, not for hidden-answer exams.

Maintainer rebuild command
python tools/question_static_exporter/export.py --data-root data --out outputs/ArchitectureIQ-quiz --zip outputs/ArchitectureIQ-quiz.zip --overwrite
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")


def _question_budget(question: dict[str, Any]) -> int | None:
    budget = question.get("budget")
    if isinstance(budget, dict):
        total = budget.get("total_samples_seen")
    else:
        total = budget
    return int(total) if total is not None else None


def _metric_display_name(metric: str) -> str:
    if metric == "test_ce":
        return "test cross-entropy"
    if metric == "test_mse":
        return "test MSE"
    return metric


def _choice_color(index: int) -> str:
    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    return palette[index % len(palette)]


def _safe_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "item"


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _format_optimizer_lines(opt: dict[str, Any]) -> list[str]:
    lines = [
        f"Type: {opt.get('type', '?')}",
        f"Learning rate: {opt.get('lr')}",
        f"Weight decay: {opt.get('weight_decay', 0)}",
    ]
    if opt.get("type") == "SGD" and "momentum" in opt:
        lines.append(f"Momentum: {opt['momentum']}")
    if opt.get("type") in {"Adam", "AdamW"} and "betas" in opt:
        lines.append(f"Betas: {opt['betas']}")
    return lines


def _format_loss_lines(loss: dict[str, Any]) -> list[str]:
    lines = [f"Loss: {loss.get('loss_id', '?')}"]
    if "lambda" in loss:
        lines.append(f"Lambda: {loss['lambda']}")
    return lines


def _format_training_lines(budget: dict[str, Any]) -> list[str]:
    return [
        f"Training steps: {budget.get('training_steps')}",
        f"Batch size: {budget.get('batch_size')}",
        f"Total samples seen: {budget.get('total_samples_seen')}",
    ]


def _candidate_spec_blocks(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"label": "Training", "lines": _format_training_lines(spec.get("budget", {}))},
        {"label": "Model", "lines": format_model_spec_lines(spec.get("model", {}))},
        {
            "label": "Optimizer",
            "lines": _format_optimizer_lines(spec.get("optimizer", {})),
        },
        {"label": "Loss", "lines": _format_loss_lines(spec.get("loss", {}))},
    ]


def _summarize_metric(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary or "error" in summary:
        return {"display": "Metrics unavailable"}
    metric = summary.get("selection_metric", "test_mse")
    mean = _number_or_none(summary.get(f"mean_{metric}"))
    std = _number_or_none(summary.get(f"std_{metric}"))
    return {
        "metric": metric,
        "mean": mean,
        "std": std,
        "display": format_metrics(summary),
        "n_seeds": summary.get("n_seeds"),
        "failed_seeds": summary.get("failed_seeds"),
        "excluded": summary.get("excluded", False),
    }


def _read_text_assets(paths: dict[str, Path], *, include_code: bool) -> dict[str, str]:
    assets: dict[str, str] = {}
    for name, path in paths.items():
        if not include_code and name.endswith(".py"):
            continue
        if path.suffix == ".json":
            try:
                assets[name] = json.dumps(
                    read_json_file(path), ensure_ascii=False, indent=2
                )
            except Exception:
                assets[name] = read_text_file(path)
        else:
            assets[name] = read_text_file(path)
    return assets


def _load_dataset_arrays(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import torch

        train = torch.load(
            dataset_dir / "train.pt", weights_only=True, map_location="cpu"
        )
        test = torch.load(
            dataset_dir / "test.pt", weights_only=True, map_location="cpu"
        )
        return (
            np.asarray(train["x"].detach().cpu()),
            np.asarray(train["y"].detach().cpu()),
            np.asarray(test["x"].detach().cpu()),
            np.asarray(test["y"].detach().cpu()),
        )
    except Exception as exc:
        print(f"[warn] failed to load dataset tensors from {dataset_dir}: {exc}")
        return None


def _plot_dataset(bundle: Any, out_path: Path) -> bool:
    spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    family = spec.get("family", "univariate_regression")
    params = spec.get("params", {})
    arrays = _load_dataset_arrays(bundle.dataset_dir)
    if arrays is None:
        return False
    train_x, train_y, test_x, test_y = arrays

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=140)
    try:
        if family == "bigram_lm":
            transition_path = bundle.dataset_dir / "transition.npz"
            if transition_path.is_file():
                probs = np.load(transition_path)["probs"]
                im = ax.imshow(probs, aspect="auto", cmap="viridis", origin="lower")
                ax.set_xlabel("next token y")
                ax.set_ylabel("current token x")
                ax.set_title("Bigram transition P(y | x)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                sample = train_x[: min(6, len(train_x))]
                ax.imshow(sample, aspect="auto", cmap="viridis")
                ax.set_title("Sample train windows")
                ax.set_xlabel("position")
                ax.set_ylabel("sample")
        elif family == "multivariate_regression":
            input_dim = int(
                params.get("input_dim", train_x.shape[1] if train_x.ndim > 1 else 1)
            )
            y_train = np.squeeze(train_y)
            y_test = np.squeeze(test_y)
            if input_dim >= 2:
                train_sc = ax.scatter(
                    train_x[:, 0],
                    train_x[:, 1],
                    c=y_train,
                    s=12,
                    alpha=0.65,
                    cmap="viridis",
                    label="train",
                )
                ax.scatter(
                    test_x[:, 0],
                    test_x[:, 1],
                    c=y_test,
                    s=12,
                    alpha=0.65,
                    cmap="viridis",
                    marker="x",
                    label="test",
                )
                ax.set_xlabel("x0")
                ax.set_ylabel("x1")
                ax.set_title("Dataset projection (color = target y)")
                fig.colorbar(train_sc, ax=ax, label="y")
            else:
                ax.scatter(
                    train_x[:, 0], y_train, s=10, alpha=0.55, label="train", c="#2563eb"
                )
                ax.scatter(
                    test_x[:, 0], y_test, s=10, alpha=0.55, label="test", c="#dc2626"
                )
                ax.set_xlabel("x0")
                ax.set_ylabel("y")
                ax.set_title("Dataset points")
                ax.legend(loc="best")
        else:
            ax.scatter(
                np.squeeze(train_x),
                np.squeeze(train_y),
                s=10,
                alpha=0.55,
                label="train",
                c="#2563eb",
            )
            ax.scatter(
                np.squeeze(test_x),
                np.squeeze(test_y),
                s=10,
                alpha=0.55,
                label="test",
                c="#dc2626",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Dataset points")
            ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path)
        return True
    finally:
        plt.close(fig)


def _plot_curves(bundle: Any, q: dict[str, Any], metric: str, out_path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    any_plotted = False
    try:
        for index, choice in enumerate(bundle.choices):
            letter = choice["letter"]
            color = _choice_color(index)
            candidate_dir = choice["candidate_dir"]
            spec = read_json_file(candidate_dir / "candidate_spec.json")
            budget = spec.get("budget", {})
            total_samples_seen = budget.get("total_samples_seen")
            batch_size = budget.get("batch_size")
            if total_samples_seen is None or batch_size is None:
                continue
            loaded = load_candidate_curves(
                candidate_dir / "results" / "curves.npz",
                total_samples_seen=int(total_samples_seen),
                batch_size=int(batch_size),
            )
            if "error" in loaded:
                continue

            curves = np.asarray(loaded["curves"], dtype=np.float64)
            x = np.asarray(loaded["eval_samples"], dtype=np.int64)
            if curves.size == 0 or not np.isfinite(curves).any():
                continue
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0)
            valid = np.isfinite(mean)
            if not valid.any():
                continue
            x_valid = x[valid]
            mean_valid = mean[valid]
            std_valid = std[valid]
            ax.fill_between(
                x_valid,
                mean_valid - std_valid,
                mean_valid + std_valid,
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                x_valid,
                mean_valid,
                color=color,
                linewidth=2.2,
                label=f"{letter} · {choice['candidate_id']}",
            )
            any_plotted = True

        if not any_plotted:
            return False
        ax.set_xlabel("Samples seen")
        ax.set_ylabel(_metric_display_name(metric))
        ax.set_title("Learning curves (mean ± std across seeds)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path)
        return True
    finally:
        plt.close(fig)


def _dataset_info(spec: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    family = spec.get("family", "univariate_regression")
    params = spec.get("params", {})
    info = {
        "dataset_id": dataset_id,
        "family": family,
        "selection_metric": spec.get("selection_metric"),
        "params": params,
        "latex_expression": None,
        "summary_lines": [],
    }
    if family == "bigram_lm":
        info["summary_lines"] = [
            f"Vocab size: {params.get('vocab_size', '-')}",
            f"Context length: {params.get('context_length', '-')}",
            "Fixed bigram law P(y|x) shared by train and test.",
        ]
        return info

    expression = params.get("expression", "-")
    domain = params.get("domain", [0, 1])
    info["latex_expression"] = expression_to_latex(expression)
    if family == "multivariate_regression":
        info["summary_lines"] = [
            f"Expression: {expression}",
            f"Input dimension: {params.get('input_dim', '-')}",
            f"Domain: [{domain[0]}, {domain[1]}] per coordinate",
        ]
    else:
        info["summary_lines"] = [
            f"Expression: {expression}",
            f"Domain: [{domain[0]}, {domain[1]}]",
        ]
    return info


def _selection_metric(bundle: Any, q: dict[str, Any]) -> str:
    if "evaluation" in q:
        return str(q["evaluation"].get("selection_metric", "test_mse"))
    spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    return str(spec.get("selection_metric", "test_mse"))


def _ranked_results(
    choices: list[dict[str, Any]], correct_letter: str
) -> list[dict[str, Any]]:
    rows = []
    for choice in choices:
        metric = choice["metric"].get("metric", "test_mse")
        mean = choice["metric"].get("mean")
        rows.append(
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidate_id"],
                "mean": mean,
                "std": choice["metric"].get("std"),
                "metric": metric,
                "correct": choice["letter"] == correct_letter,
            }
        )
    rows.sort(
        key=lambda row: (
            row["mean"] is None,
            row["mean"] if row["mean"] is not None else float("inf"),
        )
    )
    return rows


def _export_question(
    question_dir: Path,
    *,
    data_root: Path,
    out_assets: Path,
    include_code: bool,
) -> dict[str, Any]:
    bundle = load_question_bundle(question_dir, data_root)
    q = bundle.question
    metric = _selection_metric(bundle, q)
    qid = q.get("question_id", question_dir.name)
    qslug = _safe_slug(str(qid))

    dataset_spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    dataset_plot = out_assets / f"{qslug}_dataset.png"
    curves_plot = out_assets / f"{qslug}_curves.png"
    has_dataset_plot = _plot_dataset(bundle, dataset_plot)
    has_curves_plot = _plot_curves(bundle, q, metric, curves_plot)

    choices: list[dict[str, Any]] = []
    for choice in bundle.choices:
        candidate_dir = choice["candidate_dir"]
        spec = read_json_file(candidate_dir / "candidate_spec.json")
        summary_path = candidate_dir / "results" / "summary.json"
        summary = read_json_file(summary_path) if summary_path.is_file() else {}
        files = _read_text_assets(
            candidate_file_paths(candidate_dir, include_summary=True),
            include_code=include_code,
        )
        choices.append(
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidate_id"],
                "candidate_path": _relative_to_root(candidate_dir, data_root),
                "spec": spec,
                "spec_blocks": _candidate_spec_blocks(spec),
                "metric": _summarize_metric(summary),
                "files": files,
            }
        )

    dataset_files = _read_text_assets(
        dataset_file_paths(bundle.dataset_dir), include_code=include_code
    )
    exported = {
        "id": qid,
        "label": f"{qid} · {q.get('type', '?')} · {q.get('dataset_id', '?')} · n={_question_budget(q) or '?'}",
        "family": q.get("family"),
        "dataset_id": q.get("dataset_id"),
        "type": q.get("type"),
        "budget": q.get("budget"),
        "budget_total_samples": _question_budget(q),
        "num_choices": q.get("num_choices", len(choices)),
        "correct_letter": q.get("correct_letter"),
        "metric": metric,
        "metric_display": _metric_display_name(metric),
        "question": q,
        "question_path": _relative_to_root(question_dir, data_root),
        "prompt_text": bundle.prompt_text,
        "dataset": {
            "path": _relative_to_root(bundle.dataset_dir, data_root),
            "spec": dataset_spec,
            "info": _dataset_info(dataset_spec, bundle.dataset_dir.name),
            "files": dataset_files,
            "plot": f"assets/{dataset_plot.name}" if has_dataset_plot else None,
        },
        "choices": choices,
        "ranked_results": _ranked_results(choices, str(q.get("correct_letter"))),
        "curves_plot": f"assets/{curves_plot.name}" if has_curves_plot else None,
        "significance": q.get("significance", {}),
        "evaluation": q.get("evaluation", {}),
    }
    return exported


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True)


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))


def export_static_quiz(options: ExportOptions) -> dict[str, Any]:
    data_root = options.data_root.resolve()
    question_dirs = list_question_dirs(data_root)
    if options.limit is not None:
        question_dirs = question_dirs[: options.limit]
    if not question_dirs:
        raise RuntimeError(f"No questions found under {data_root}")

    _prepare_out_dir(options.out_dir, overwrite=options.overwrite)
    _copy_templates(options.out_dir)
    _write_package_readme(options.out_dir)
    assets = options.out_dir / "assets"

    questions = []
    failures: list[dict[str, str]] = []
    for index, question_dir in enumerate(question_dirs, start=1):
        try:
            questions.append(
                _export_question(
                    question_dir,
                    data_root=data_root,
                    out_assets=assets,
                    include_code=options.include_code,
                )
            )
            print(f"[{index}/{len(question_dirs)}] exported {question_dir.name}")
        except Exception as exc:
            failures.append({"path": str(question_dir), "error": str(exc)})
            print(f"[warn] failed {question_dir}: {exc}")

    if not questions:
        raise RuntimeError("Export failed: no questions were exported")

    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "title": options.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_root": str(data_root),
            "question_count_found": len(question_dirs),
            "question_count_exported": len(questions),
            "failures": failures,
        },
        "questions": questions,
    }
    _write_json_js(options.out_dir / "data.js", payload)

    manifest = {
        "title": options.title,
        "generated_at": payload["generated_at"],
        "output_dir": str(options.out_dir.resolve()),
        "zip_path": str(options.zip_path.resolve()) if options.zip_path else None,
        "questions": len(questions),
        "failures": failures,
    }
    (options.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if options.zip_path:
        _zip_dir(options.out_dir, options.zip_path)
    return manifest


def _parse_args() -> ExportOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data", type=Path, help="Directory containing datasets/"
    )
    parser.add_argument(
        "--out",
        default=Path("outputs/ArchitectureIQ-quiz"),
        type=Path,
        help="Output folder",
    )
    parser.add_argument(
        "--title", default="ArchitectureIQ Quiz", help="Title shown in the static app"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Export only the first N questions"
    )
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("outputs/ArchitectureIQ-quiz.zip"),
    )
    parser.add_argument("--no-zip", action="store_true", help="Only write the folder")
    parser.add_argument(
        "--exclude-code", action="store_true", help="Do not embed .py source files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder if it exists",
    )
    args = parser.parse_args()
    return ExportOptions(
        data_root=args.data_root,
        out_dir=args.out,
        title=args.title,
        limit=args.limit,
        zip_path=None if args.no_zip else args.zip_path,
        include_code=not args.exclude_code,
        overwrite=args.overwrite,
    )


def main() -> None:
    manifest = export_static_quiz(_parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
