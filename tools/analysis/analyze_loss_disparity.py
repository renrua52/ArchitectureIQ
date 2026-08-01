"""Per-question choice loss disparity analysis.

For every question (unique choice set), load per-candidate GT summaries and
express each choice's mean loss relative to the median choice (median = 1):

  - histogram of non-median ratios (all questions, and the 3-choice subset)
  - histogram of per-question spread (max / min of choice means)
  - per-seed loss distributions for a few random 3-choice questions

Usage:
    .venv/bin/python tools/analysis/analyze_loss_disparity.py \
        [--glob 'data/datasets/*/*/questions/run_*/q_*/question.json'] \
        [--out artifacts/loss_disparity] [--random-n 3]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_GLOB = "data/datasets/*/*/questions/run_*/q_*/question.json"


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def _final_key(metric: str) -> str:
    return f"final_{metric}"


def load_summary(candidate_path: str) -> dict | None:
    p = DATA / candidate_path / "results" / "summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def collect(questions: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    skipped = {"missing_summary": 0, "excluded": 0, "no_metric": 0}

    for qpath in questions:
        q = json.loads(qpath.read_text(encoding="utf-8"))
        key = (q["dataset_id"], tuple(sorted(c["candidate_id"] for c in q["choices"])))
        if key in seen:
            continue
        seen.add(key)

        summaries = []
        for choice in q["choices"]:
            s = load_summary(choice["candidate_path"])
            if s is None:
                skipped["missing_summary"] += 1
                summaries = []
                break
            if s.get("excluded"):
                skipped["excluded"] += 1
                summaries = []
                break
            summaries.append(s)
        if not summaries:
            continue

        metric = summaries[0].get("selection_metric", "test_mse")
        mean_key = _mean_key(metric)
        final_key = _final_key(metric)
        if any(mean_key not in s for s in summaries):
            skipped["no_metric"] += 1
            continue

        means = np.array([s[mean_key] for s in summaries], dtype=np.float64)
        if np.any(means <= 0) or np.any(~np.isfinite(means)):
            skipped["no_metric"] += 1
            continue

        med = float(np.median(means))
        ratios = means / med
        med_idx = int(np.argmin(np.abs(means - med)))
        non_median_ratios = [float(r) for i, r in enumerate(ratios) if i != med_idx]

        per_seed = [
            [float(s["seed_results"][j][final_key]) for j in range(len(s["seed_results"]))]
            for s in summaries
        ]

        rows.append({
            "dataset_id": q["dataset_id"],
            "question_id": q["question_id"],
            "type": q.get("type", "?"),
            "num_choices": len(q["choices"]),
            "letters": [c["letter"] for c in q["choices"]],
            "means": [float(m) for m in means],
            "ratios": [float(r) for r in ratios],
            "non_median_ratios": non_median_ratios,
            "spread": float(means.max() / means.min()),
            "metric": metric,
            "per_seed": per_seed,
        })
    return rows, skipped


def plot_ratio_hist(rows: list[dict], out: Path, subset_3: bool = False) -> None:
    if subset_3:
        vals = [r for r in rows if r["num_choices"] == 3]
        title = f"3-choice questions: non-median ratio (n={len(vals)} questions)"
    else:
        vals = rows
        title = f"All questions: non-median ratio (n={len(vals)} questions)"
    ratios = [x for r in vals for x in r["non_median_ratios"]]
    log2 = np.log2(ratios)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log2, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="median (=1)")
    ticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(np.log2(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("ratio to median (log2 scale)")
    ax.set_ylabel("choices")
    ax.set_title(title)
    pcts = np.percentile(log2, [50, 90, 99])
    for p, lab in zip(pcts, ["50%", "90%", "99%"]):
        ax.axvline(p, color="#C44E52", linestyle=":", linewidth=1)
        ax.text(p, ax.get_ylim()[1] * 0.95, f" {lab} {2**p:.2f}x", fontsize=8, color="#C44E52")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.name}: median={np.median(ratios):.3f}x "
          f"p90={np.percentile(ratios, 90):.2f}x max={max(ratios):.1f}x")


def plot_spread_hist(rows: list[dict], out: Path) -> None:
    spreads = np.array([r["spread"] for r in rows])
    log2 = np.log2(spreads)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(log2, bins=40, color="#55A868", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="max = min (=1)")
    ticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(np.log2(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("question spread max/min of choice means (log2)")
    ax.set_ylabel("questions")
    ax.set_title(f"Per-question spread of choice losses (n={len(rows)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    med = float(np.median(spreads))
    gt2 = float(np.mean(spreads > 2) * 100)
    gt5 = float(np.mean(spreads > 5) * 100)
    print(f"saved {out.name}: median spread={med:.2f}x, >2x: {gt2:.1f}%, >5x: {gt5:.1f}%")


def plot_random_3choice(rows: list[dict], out: Path, n: int, rng: random.Random) -> None:
    pool = [r for r in rows if r["num_choices"] == 3]
    picked = rng.sample(pool, min(n, len(pool)))
    fig, axes = plt.subplots(1, len(picked), figsize=(5.2 * len(picked), 4.2), squeeze=False)
    for ax, r in zip(axes[0], picked):
        means = np.array(r["means"])
        med = float(np.median(means))
        order = np.argsort(means)
        labels = [f"{r['letters'][i]}\n{r['means'][i]/med:.2f}x" for i in order]
        data = [np.array(r["per_seed"][i]) for i in order]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#55A868", "#C44E52"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for i, d in zip(order, data):
            ax.scatter(np.full_like(d, list(order).index(i) + 1 + 0.08), d, s=6, alpha=0.5, color="black")
        ax.set_title(f"{r['dataset_id']} / {r['question_id']}\n{r['type']} · {r['metric']}")
        ax.set_ylabel("per-seed loss")
    fig.suptitle(f"Random 3-setting loss distributions (median = 1x)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}: {len(picked)} random 3-choice questions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "loss_disparity")
    ap.add_argument("--random-n", type=int, default=3, help="random 3-setting questions to plot")
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    question_files = sorted(globmod.glob(args.glob))
    if not question_files:
        print(f"no questions matched: {args.glob}")
        return 1

    rows, skipped = collect([Path(f) for f in question_files])
    print(f"question files: {len(question_files)}, unique choice sets: {len(rows)}, skipped: {skipped}")
    by_n = {}
    for r in rows:
        by_n.setdefault(r["num_choices"], 0)
        by_n[r["num_choices"]] += 1
    print("by num_choices:", dict(sorted(by_n.items())))

    args.out.mkdir(parents=True, exist_ok=True)
    plot_ratio_hist(rows, args.out / "ratios_all.png")
    plot_ratio_hist(rows, args.out / "ratios_3choice.png", subset_3=True)
    plot_spread_hist(rows, args.out / "spread_maxmin.png")
    plot_random_3choice(rows, args.out / "random_3setting_distributions.png", args.random_n,
                        random.Random(args.seed))

    # compact stats table
    ratios = [x for r in rows for x in r["non_median_ratios"]]
    ratios3 = [x for r in rows if r["num_choices"] == 3 for x in r["non_median_ratios"]]
    spreads = [r["spread"] for r in rows]
    print("\n--- non-median ratios (median=1) ---")
    for name, arr in (("all", ratios), ("3-choice", ratios3)):
        a = np.array(arr)
        print(f"{name:8s} n={len(a):4d} med={np.median(a):.3f}x p90={np.percentile(a,90):.2f}x "
              f"max={a.max():.1f}x | <=1.1x {np.mean(a<=1.1)*100:.1f}% <=1.5x {np.mean(a<=1.5)*100:.1f}% "
              f">2x {np.mean(a>2)*100:.1f}% >5x {np.mean(a>5)*100:.1f}% >10x {np.mean(a>10)*100:.1f}%")
    s = np.array(spreads)
    print(f"spread   n={len(s):4d} med={np.median(s):.2f}x p90={np.percentile(s,90):.2f}x "
          f"| <=1.5x {np.mean(s<=1.5)*100:.1f}% >2x {np.mean(s>2)*100:.1f}% >5x {np.mean(s>5)*100:.1f}%")

    # examples: most/least disparate 3-choice questions
    ex = sorted((r for r in rows if r["num_choices"] == 3),
                key=lambda r: max(r["non_median_ratios"]))
    print("\n--- least disparate 3-choice ---")
    for r in ex[:3]:
        print(f"  {r['dataset_id']}/{r['question_id']} {r['type']} ratios={[round(x,2) for x in r['ratios']]}")
    print("--- most disparate 3-choice ---")
    for r in ex[-3:]:
        print(f"  {r['dataset_id']}/{r['question_id']} {r['type']} ratios={[round(x,2) for x in r['ratios']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
