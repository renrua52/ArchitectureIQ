from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CACHE = ROOT / "artifacts" / ".cache"
ARTIFACT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ARTIFACT_CACHE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT = ROOT / "artifacts" / "order_parameter_analysis"


OPT_COLORS = {
    "SGD": "#4C78A8",
    "Adam": "#F58518",
    "AdamW": "#54A24B",
    "RMSprop": "#B279A2",
    "Adagrad": "#E45756",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    raise ValueError(f"unknown model type {model['type']!r}")


def describe_curve(curves_path: Path) -> dict[str, Any] | None:
    data = np.load(curves_path)
    curves = np.asarray(data["curves"], dtype=float)
    samples = np.asarray(data["samples"], dtype=float)
    if curves.size == 0 or samples.size == 0:
        return None

    mean_curve = np.nanmean(curves, axis=0)
    finite = np.isfinite(mean_curve) & (mean_curve > 0)
    finite &= np.isfinite(samples)
    if finite.sum() < 3:
        return None
    mean_curve = mean_curve[finite]
    samples = samples[finite]
    std_curve = np.nanstd(curves[:, finite], axis=0)

    start = float(mean_curve[0])
    final = float(mean_curve[-1])
    min_metric = float(np.nanmin(mean_curve))
    log_improvement = float(math.log(start / final)) if start > 0 and final > 0 else float("nan")

    n_early = max(3, int(math.ceil(len(mean_curve) * 0.25)))
    x = samples[:n_early] / samples[-1]
    y = np.log(mean_curve[:n_early])
    if np.ptp(x) > 0:
        early_slope = float(-np.polyfit(x, y, 1)[0])
    else:
        early_slope = float("nan")

    log_auc = float(np.trapezoid(np.log(mean_curve), samples / samples[-1]))
    return {
        "samples": samples,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "start_metric": start,
        "final_metric": final,
        "min_metric": min_metric,
        "log_improvement": log_improvement,
        "early_slope": early_slope,
        "log_auc": log_auc,
    }


def candidate_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec_path in sorted((ROOT / "data").glob("datasets/*/*/candidates/*/c_*/candidate_spec.json")):
        curves_path = spec_path.parent / "results" / "curves.npz"
        if not curves_path.exists():
            continue
        spec = load_json(spec_path)
        curve = describe_curve(curves_path)
        if curve is None:
            continue
        opt = spec["optimizer"]
        model = spec["model"]
        set_name = spec_path.parents[1].name
        dataset_id = spec["dataset_id"]
        family = spec["family"]
        budget = spec["budget"]
        params = count_params(model)
        records.append(
            {
                "family": family,
                "dataset_id": dataset_id,
                "set_name": set_name,
                "candidate_id": spec["candidate_id"],
                "candidate_dir": str(spec_path.parent.relative_to(ROOT)),
                "model_type": model["type"],
                "depth": int(model.get("depth", model.get("num_layers", 0))),
                "width": int(model.get("width", model.get("d_model", 0))),
                "residual": bool(model.get("residual", False)),
                "num_params": params,
                "log_num_params": math.log(params),
                "optimizer": opt["type"],
                "lr": float(opt["lr"]),
                "log_lr": math.log(float(opt["lr"])),
                "weight_decay": float(opt.get("weight_decay", 0.0)),
                "momentum": float(opt.get("momentum", 0.0)),
                "batch_size": int(budget["batch_size"]),
                "training_steps": int(budget["training_steps"]),
                "total_samples_seen": int(budget["total_samples_seen"]),
                **{k: v for k, v in curve.items() if k not in {"samples", "mean_curve", "std_curve"}},
                "_samples": curve["samples"],
                "_mean_curve": curve["mean_curve"],
                "_std_curve": curve["std_curve"],
            }
        )
    return records


def one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    cats = sorted(set(values))
    if len(cats) <= 1:
        return np.zeros((len(values), 0)), []
    cols = []
    names = []
    base = cats[0]
    for cat in cats[1:]:
        cols.append([1.0 if v == cat else 0.0 for v in values])
        names.append(f"{cat}_vs_{base}")
    return np.asarray(cols, dtype=float).T, names


def ols(y: np.ndarray, cols: list[np.ndarray], names: list[str]) -> dict[str, Any]:
    mask = np.isfinite(y)
    for col in cols:
        mask &= np.isfinite(col)
    y = y[mask]
    x_cols = [col[mask] for col in cols]
    if len(y) < 4:
        return {"n": int(len(y)), "r2": float("nan"), "coef": {}}
    x = np.column_stack([np.ones(len(y)), *x_cols])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(y)),
        "r2": r2,
        "coef": {"intercept": float(coef[0]), **{name: float(c) for name, c in zip(names, coef[1:])}},
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks

    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def eta_proxy_fit(records: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    y = np.asarray([r[target_key] for r in records], dtype=float)
    log_lr = np.asarray([r["log_lr"] for r in records], dtype=float)
    log_params = np.asarray([r["log_num_params"] for r in records], dtype=float)
    opt_matrix, opt_names = one_hot([r["optimizer"] for r in records])
    opt_cols = [opt_matrix[:, i] for i in range(opt_matrix.shape[1])]

    models = {
        "log_lr": ols(y, [log_lr], ["log_lr"]),
        "log_params": ols(y, [log_params], ["log_num_params"]),
        "log_lr_plus_optimizer": ols(y, [log_lr, *opt_cols], ["log_lr", *opt_names]),
        "log_lr_optimizer_params": ols(
            y,
            [log_lr, log_params, *opt_cols],
            ["log_lr", "log_num_params", *opt_names],
        ),
    }
    best = models["log_lr_plus_optimizer"]
    opt_coef = best["coef"]
    cats = sorted(set(r["optimizer"] for r in records))
    base = cats[0] if cats else ""
    offsets = {base: 0.0}
    for name, value in opt_coef.items():
        if "_vs_" in name:
            offsets[name.split("_vs_")[0]] = value
    return {"models": models, "optimizer_offsets": offsets}


def plot_curves(records: list[dict[str, Any]], set_key: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for r in records:
        samples = r["_samples"]
        curve = r["_mean_curve"]
        opt = r["optimizer"]
        label = opt if opt not in [line.get_label() for line in ax.lines] else "_nolegend_"
        ax.plot(
            samples,
            curve,
            color=OPT_COLORS.get(opt, "#777777"),
            alpha=0.42,
            linewidth=1.2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("samples seen")
    ax.set_ylabel("mean test metric, log scale")
    ax.set_title(f"Learning curves: {set_key} ({len(records)} candidates)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(title="optimizer", ncols=3, fontsize=9)
    path = OUT / f"{set_key.replace('/', '__')}_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter(records: list[dict[str, Any]], set_key: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, y_key, y_label in [
        (axes[0], "early_slope", "early progress slope"),
        (axes[1], "final_metric", "final metric"),
    ]:
        for opt in sorted(set(r["optimizer"] for r in records)):
            group = [r for r in records if r["optimizer"] == opt]
            size = 18 + 11 * (np.asarray([r["log_num_params"] for r in group]) - min(r["log_num_params"] for r in records))
            axes_y = np.asarray([r[y_key] for r in group], dtype=float)
            ax.scatter(
                [r["lr"] for r in group],
                axes_y,
                s=size,
                alpha=0.72,
                color=OPT_COLORS.get(opt, "#777777"),
                edgecolors="white",
                linewidths=0.4,
                label=opt,
            )
        ax.set_xscale("log")
        if y_key == "final_metric":
            ax.set_yscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(title="optimizer", fontsize=8)
    fig.suptitle(f"Order-parameter probes: {set_key}")
    path = OUT / f"{set_key.replace('/', '__')}_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_csv(records: list[dict[str, Any]]) -> Path:
    path = OUT / "candidate_metrics.csv"
    fields = [
        "family",
        "dataset_id",
        "set_name",
        "candidate_id",
        "model_type",
        "depth",
        "width",
        "residual",
        "num_params",
        "optimizer",
        "lr",
        "weight_decay",
        "momentum",
        "batch_size",
        "training_steps",
        "total_samples_seen",
        "start_metric",
        "final_metric",
        "min_metric",
        "log_improvement",
        "early_slope",
        "log_auc",
        "candidate_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in records:
            writer.writerow({field: r.get(field, "") for field in fields})
    return path


def anchored_progress(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_final = min(r["final_metric"] for r in group)
    rows: list[dict[str, Any]] = []
    for r in group:
        samples = np.asarray(r["_samples"], dtype=float)
        curve = np.asarray(r["_mean_curve"], dtype=float)
        if len(samples) == 0 or len(curve) == 0 or best_final <= 0 or r["start_metric"] <= best_final:
            continue
        denom = math.log(r["start_metric"]) - math.log(best_final)
        if denom <= 0:
            continue
        progress = (math.log(r["start_metric"]) - np.log(curve)) / denom

        def crossing(level: float) -> float:
            hit = np.where(progress >= level)[0]
            if len(hit) == 0:
                return float("nan")
            i = int(hit[0])
            if i == 0:
                return float(samples[0])
            p0, p1 = float(progress[i - 1]), float(progress[i])
            s0, s1 = float(samples[i - 1]), float(samples[i])
            if p1 == p0:
                return s1
            alpha = (level - p0) / (p1 - p0)
            return s0 + alpha * (s1 - s0)

        p_final = float(progress[-1])
        p_auc = float(np.trapezoid(np.clip(progress, -1.0, 1.5), samples / samples[-1]))
        rows.append(
            {
                **{k: r[k] for k in r if not k.startswith("_")},
                "anchor_best_final": best_final,
                "anchored_final_progress": p_final,
                "anchored_progress_auc": p_auc,
                "samples_to_p25": crossing(0.25),
                "samples_to_p50": crossing(0.50),
                "samples_to_p75": crossing(0.75),
                "_progress": progress,
            }
        )
    return rows


def write_anchor_csv(rows: list[dict[str, Any]]) -> Path:
    path = OUT / "anchored_progress_metrics.csv"
    fields = [
        "family",
        "dataset_id",
        "set_name",
        "candidate_id",
        "optimizer",
        "lr",
        "num_params",
        "depth",
        "width",
        "final_metric",
        "anchor_best_final",
        "anchored_final_progress",
        "anchored_progress_auc",
        "samples_to_p25",
        "samples_to_p50",
        "samples_to_p75",
        "candidate_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def plot_anchored_progress(records: list[dict[str, Any]], set_key: str) -> Path:
    best_final = min(r["final_metric"] for r in records)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for r in records:
        if r["start_metric"] <= best_final or best_final <= 0:
            continue
        samples = r["_samples"]
        curve = r["_mean_curve"]
        denom = math.log(r["start_metric"]) - math.log(best_final)
        if denom <= 0:
            continue
        progress = (math.log(r["start_metric"]) - np.log(curve)) / denom
        opt = r["optimizer"]
        label = opt if opt not in [line.get_label() for line in ax.lines] else "_nolegend_"
        ax.plot(
            samples,
            progress,
            color=OPT_COLORS.get(opt, "#777777"),
            alpha=0.42,
            linewidth=1.2,
            label=label,
        )
    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color="#999999", linewidth=0.8, alpha=0.35)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("samples seen")
    ax.set_ylabel("anchored progress to best final metric")
    ax.set_title(f"Anchored progress curves: {set_key}")
    ax.grid(True, alpha=0.22)
    ax.legend(title="optimizer", ncols=3, fontsize=9)
    path = OUT / f"{set_key.replace('/', '__')}_anchored_progress.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_time_to_progress(rows: list[dict[str, Any]], set_key: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, key, title in [
        (axes[0], "samples_to_p50", "samples to 50% anchored progress"),
        (axes[1], "samples_to_p75", "samples to 75% anchored progress"),
    ]:
        for opt in sorted(set(r["optimizer"] for r in rows)):
            group = [r for r in rows if r["optimizer"] == opt and np.isfinite(r[key])]
            if not group:
                continue
            sizes = 20 + 10 * (
                np.asarray([r["log_num_params"] for r in group])
                - min(r["log_num_params"] for r in rows)
            )
            ax.scatter(
                [r["lr"] for r in group],
                [r[key] for r in group],
                s=sizes,
                alpha=0.76,
                color=OPT_COLORS.get(opt, "#777777"),
                edgecolors="white",
                linewidths=0.4,
                label=opt,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("samples")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(title="optimizer", fontsize=8)
    fig.suptitle(f"Anchored time-to-progress: {set_key}")
    path = OUT / f"{set_key.replace('/', '__')}_time_to_progress.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = candidate_records()
    csv_path = write_csv(records)

    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_set[f"{r['family']}/{r['dataset_id']}/{r['set_name']}"].append(r)

    set_summaries: dict[str, Any] = {}
    plots: dict[str, list[str]] = {}
    all_anchor_rows: list[dict[str, Any]] = []
    for set_key, group in sorted(by_set.items()):
        if len(group) < 8:
            continue
        anchor_rows = anchored_progress(group)
        all_anchor_rows.extend(anchor_rows)
        fit_early = eta_proxy_fit(group, "early_slope")
        fit_final = eta_proxy_fit(group, "log_improvement")
        fit_anchor = eta_proxy_fit(anchor_rows, "anchored_progress_auc") if anchor_rows else {}
        rho_params_final = spearman(
            np.asarray([r["log_num_params"] for r in group], dtype=float),
            -np.asarray([r["final_metric"] for r in group], dtype=float),
        )
        rho_params_progress = spearman(
            np.asarray([r["log_num_params"] for r in group], dtype=float),
            np.asarray([r["log_improvement"] for r in group], dtype=float),
        )
        rho_lr_early = spearman(
            np.asarray([r["log_lr"] for r in group], dtype=float),
            np.asarray([r["early_slope"] for r in group], dtype=float),
        )
        rho_lr_anchor_auc = spearman(
            np.asarray([r["log_lr"] for r in anchor_rows], dtype=float),
            np.asarray([r["anchored_progress_auc"] for r in anchor_rows], dtype=float),
        )
        rho_params_anchor_auc = spearman(
            np.asarray([r["log_num_params"] for r in anchor_rows], dtype=float),
            np.asarray([r["anchored_progress_auc"] for r in anchor_rows], dtype=float),
        )
        p50_reached = [r for r in anchor_rows if np.isfinite(r["samples_to_p50"])]
        p75_reached = [r for r in anchor_rows if np.isfinite(r["samples_to_p75"])]
        set_summaries[set_key] = {
            "n_candidates": len(group),
            "optimizers": sorted(set(r["optimizer"] for r in group)),
            "num_params_range": [int(min(r["num_params"] for r in group)), int(max(r["num_params"] for r in group))],
            "lr_range": [float(min(r["lr"] for r in group)), float(max(r["lr"] for r in group))],
            "spearman_log_params_vs_better_final": rho_params_final,
            "spearman_log_params_vs_log_improvement": rho_params_progress,
            "spearman_log_lr_vs_early_slope": rho_lr_early,
            "spearman_log_lr_vs_anchored_auc": rho_lr_anchor_auc,
            "spearman_log_params_vs_anchored_auc": rho_params_anchor_auc,
            "p50_reached": len(p50_reached),
            "p75_reached": len(p75_reached),
            "early_slope_regressions": fit_early,
            "log_improvement_regressions": fit_final,
            "anchored_auc_regressions": fit_anchor,
        }
        curve_path = plot_curves(group, set_key)
        scatter_path = plot_scatter(group, set_key)
        anchored_path = plot_anchored_progress(group, set_key)
        time_path = plot_time_to_progress(anchor_rows, set_key)
        plots[set_key] = [
            str(curve_path.relative_to(ROOT)),
            str(scatter_path.relative_to(ROOT)),
            str(anchored_path.relative_to(ROOT)),
            str(time_path.relative_to(ROOT)),
        ]

    anchor_csv_path = write_anchor_csv(all_anchor_rows)

    summary = {
        "n_candidates": len(records),
        "n_sets": len(by_set),
        "csv": str(csv_path.relative_to(ROOT)),
        "anchored_csv": str(anchor_csv_path.relative_to(ROOT)),
        "plots": plots,
        "sets": set_summaries,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Order Parameter Analysis",
        "",
        f"Loaded {len(records)} candidates with learning curves across {len(by_set)} candidate sets.",
        f"Per-candidate metrics: `{csv_path.relative_to(ROOT)}`.",
        f"Anchored progress metrics: `{anchor_csv_path.relative_to(ROOT)}`.",
        "",
    ]
    for set_key, s in set_summaries.items():
        early = s["early_slope_regressions"]["models"]
        improve = s["log_improvement_regressions"]["models"]
        anchored = s["anchored_auc_regressions"].get("models", {})
        lines.extend(
            [
                f"## {set_key}",
                "",
                f"- candidates: {s['n_candidates']}",
                f"- optimizers: {', '.join(s['optimizers'])}",
                f"- params: {s['num_params_range'][0]} to {s['num_params_range'][1]}",
                f"- lr: {s['lr_range'][0]} to {s['lr_range'][1]}",
                f"- Spearman log(params) vs better final metric: {s['spearman_log_params_vs_better_final']:.3f}",
                f"- Spearman log(params) vs log improvement: {s['spearman_log_params_vs_log_improvement']:.3f}",
                f"- Spearman log(lr) vs early slope: {s['spearman_log_lr_vs_early_slope']:.3f}",
                f"- Spearman log(lr) vs anchored progress AUC: {s['spearman_log_lr_vs_anchored_auc']:.3f}",
                f"- Spearman log(params) vs anchored progress AUC: {s['spearman_log_params_vs_anchored_auc']:.3f}",
                f"- reached p50 / p75 anchored progress: {s['p50_reached']} / {s['p75_reached']}",
                f"- R2 early slope, log_lr only: {early['log_lr']['r2']:.3f}",
                f"- R2 early slope, log_lr + optimizer: {early['log_lr_plus_optimizer']['r2']:.3f}",
                f"- R2 early slope, log_lr + optimizer + log(params): {early['log_lr_optimizer_params']['r2']:.3f}",
                f"- R2 log improvement, log_lr + optimizer + log(params): {improve['log_lr_optimizer_params']['r2']:.3f}",
                *(
                    [
                        f"- R2 anchored progress AUC, log_lr + optimizer + log(params): {anchored['log_lr_optimizer_params']['r2']:.3f}",
                    ]
                    if anchored
                    else []
                ),
                f"- plots: {', '.join(plots[set_key])}",
                "",
            ]
        )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
