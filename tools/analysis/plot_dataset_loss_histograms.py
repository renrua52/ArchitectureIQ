"""One loss histogram per problem (dataset), from the columnar backend storage.

Reads ``backend/data`` via the storage repository API, plots candidate mean
loss histograms (log x-axis when the range is wide), and writes one PNG per
problem plus a grid overview.

Usage:
    .venv/bin/python tools/analysis/plot_dataset_loss_histograms.py \
        [--out artifacts/loss_distributions] [--grid]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from architecture_iq.storage import repository as repo
from architecture_iq.storage.schema import PROBLEM_SPEC_JSON

GRID_COLS = 5


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def plot_one(problem_id: str, out: Path) -> dict | None:
    spec = repo.read_problem_spec(problem_id)
    metric = spec.get("selection_metric", "test_mse")
    family = spec.get("family", "?")
    mean_key = _mean_key(metric)

    means, seed_means = [], []
    for cid in repo.list_candidate_ids(problem_id):
        try:
            s = repo.read_summary(problem_id, cid)
        except FileNotFoundError:
            continue
        if s.get("excluded") or mean_key not in s:
            continue
        means.append(float(s[mean_key]))
        if s.get("seed_results"):
            final_key = f"final_{metric}"
            seed_means.extend(
                float(seed[final_key]) for seed in s["seed_results"] if final_key in seed
            )

    if not means:
        return None
    means = np.array(means)
    use_log = means.max() / max(means.min(), 1e-12) > 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if use_log:
        bins = np.logspace(np.log10(max(means.min(), 1e-12)), np.log10(means.max()), 30)
        ax.hist(means, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_xscale("log")
    else:
        ax.hist(means, bins=25, color="#4C72B0", edgecolor="white")
    ax.axvline(np.median(means), color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"median={np.median(means):.4g}")
    ax.set_xlabel(metric)
    ax.set_ylabel("candidates")
    ax.set_title(f"{problem_id}  [{family}]  n={len(means)} candidates\n"
                 f"min={means.min():.4g}  max={means.max():.4g}  med={np.median(means):.4g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {
        "problem_id": problem_id,
        "family": family,
        "metric": metric,
        "n": int(len(means)),
        "median": float(np.median(means)),
        "min": float(means.min()),
        "max": float(means.max()),
        "n_seed_values": int(len(seed_means)),
    }


def plot_grid(rows: list[dict], out: Path) -> None:
    n = len(rows)
    cols = GRID_COLS
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.6 * rows_n))
    axes = np.atleast_2d(axes)
    for i, info in enumerate(rows):
        ax = axes[i // cols][i % cols]
        png = out.parent / "by_problem" / f"{info['problem_id']}.png"
        img = plt.imread(png)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{info['problem_id']}  n={info['n']}", fontsize=8)
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Per-problem candidate loss distributions", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved grid: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/loss_distributions"))
    ap.add_argument("--grid", action="store_true", help="also render a grid overview")
    args = ap.parse_args()

    by_dir = args.out / "by_problem"
    by_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for problem_id in repo.list_problems():
        info = plot_one(problem_id, by_dir / f"{problem_id}.png")
        if info is None:
            skipped.append(problem_id)
            continue
        rows.append(info)

    rows.sort(key=lambda r: r["problem_id"])
    print(f"problems plotted: {len(rows)}  (skipped, no candidates: {skipped})")
    print(f"{'problem_id':14s} {'family':24s} {'metric':9s} {'n':>4s} {'min':>10s} {'median':>10s} {'max':>10s}")
    for r in rows:
        print(f"{r['problem_id']:14s} {r['family']:24s} {r['metric']:9s} {r['n']:4d} "
              f"{r['min']:10.4g} {r['median']:10.4g} {r['max']:10.4g}")

    (args.out / "summary.csv").write_text(
        "problem_id,family,metric,n,min,median,max\n"
        + "\n".join(
            f"{r['problem_id']},{r['family']},{r['metric']},{r['n']},{r['min']},{r['median']},{r['max']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    if args.grid:
        plot_grid(rows, args.out / "grid_overview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
