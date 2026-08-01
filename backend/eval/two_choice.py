"""Two-choice loss comparison items with few-shot reference settings.

Design (eval-side prototype):
  * Every item = one problem (dataset) + K reference settings with their
    measured losses (few-shot demos) + a target pair (A/B) to compare.
  * The model must answer "which setting has the HIGHER/LOWER loss", based on
    the references — not on its own priors.
  * Demos are sampled from the SAME problem (same metric scale) and never
    overlap the target pair (no leakage). Target pairs are filtered to be
    answerable but not trivial (ratio in [1.2, 5], win_rate >= 0.8).

Usage:
    .venv/bin/python -m backend.eval.two_choice --items-per-problem 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from architecture_iq.candidates.axes import choices_have_contrast
from architecture_iq.prompts.formatters import (
    format_loss_nl,
    format_model_nl,
    format_optimizer_nl,
    format_training_schedule,
)
from architecture_iq.storage import repository as repo

SCHEMA_VERSION = "0.1"
OUT_DIR = Path("artifacts/eval_probe")
MIN_RATIO = 1.2
MAX_RATIO = 5.0
MIN_WIN_RATE = 0.8
N_DEMOS = 3


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def _final_key(metric: str) -> str:
    return f"final_{metric}"


def setting_summary(config: dict) -> str:
    model = config.get("model", {})
    try:
        model_nl = format_model_nl(model)
    except KeyError:
        model_nl = json.dumps(model, sort_keys=True)
    parts = [model_nl]
    if "optimizer" in config:
        parts.append(format_optimizer_nl(config["optimizer"]))
    if "loss" in config:
        parts.append(format_loss_nl(config["loss"]))
    parts.append(format_training_schedule(config.get("budget", {})))
    return " | ".join(parts)


def pair_stats(summary_a: dict, summary_b: dict, metric: str) -> dict:
    """Ordered-pair comparison: is A's loss higher than B's?"""
    final_key = _final_key(metric)
    mean_key = _mean_key(metric)
    ma, mb = float(summary_a[mean_key]), float(summary_b[mean_key])
    n = len(summary_a["seed_results"])
    wins = 0
    for i in range(n):
        fa = float(summary_a["seed_results"][i][final_key])
        fb = float(summary_b["seed_results"][i][final_key])
        if fa > fb:
            wins += 1
    return {
        "mean_a": ma,
        "mean_b": mb,
        "ratio": float(max(ma, mb) / max(min(ma, mb), 1e-12)),
        "win_rate_a_higher": wins / n,
        "n_seeds": n,
    }


def _load_candidates(problem_id: str) -> list[tuple[str, dict, dict]]:
    rows = []
    for cid in repo.list_candidate_ids(problem_id):
        try:
            config = repo.read_candidate_config(problem_id, cid)
            summary = repo.read_summary(problem_id, cid)
        except FileNotFoundError:
            continue
        if summary.get("excluded"):
            continue
        rows.append((cid, config, summary))
    return rows


def build_item(problem_id: str, rng: random.Random, metric: str) -> dict | None:
    rows = _load_candidates(problem_id)
    if len(rows) < N_DEMOS + 2:
        return None
    mean_key = _mean_key(metric)
    rows.sort(key=lambda r: float(r[2][mean_key]))

    # demos: spread across the loss range (near min / median / max)
    demo_idx = sorted({0, len(rows) // 2, len(rows) - 1})
    demos = [rows[i] for i in demo_idx]
    pool = [r for i, r in enumerate(rows) if i not in demo_idx]

    rng.shuffle(pool)
    for idx_a in range(len(pool) - 1):
        for idx_b in range(idx_a + 1, len(pool)):
            cid_a, config_a, sum_a = pool[idx_a]
            cid_b, config_b, sum_b = pool[idx_b]
            if not choices_have_contrast([config_a, config_b]):
                continue
            st = pair_stats(sum_a, sum_b, metric)
            if st["ratio"] < MIN_RATIO or st["ratio"] > MAX_RATIO:
                continue
            win = max(st["win_rate_a_higher"], 1 - st["win_rate_a_higher"])
            if win < MIN_WIN_RATE:
                continue
            higher = st["mean_a"] > st["mean_b"]
            ask_higher = bool(rng.getrandbits(1))
            if ask_higher:
                answer = "A" if higher else "B"
            else:
                answer = "A" if not higher else "B"
            item = {
                "schema_version": SCHEMA_VERSION,
                "task": "two_choice_loss_compare",
                "problem_id": problem_id,
                "metric": metric,
                "ask": "higher" if ask_higher else "lower",
                "demos": [
                    {"candidate_id": cid, "setting": config, "loss": float(summary[mean_key])}
                    for cid, config, summary in demos
                ],
                "target": {
                    "A": {"candidate_id": cid_a, "setting": config_a},
                    "B": {"candidate_id": cid_b, "setting": config_b},
                },
                "answer": answer,
                "statistics": {"ratio": round(st["ratio"], 3),
                               "win_rate": round(win, 3),
                               "n_seeds": st["n_seeds"]},
            }
            return item
    return None


def render_prompt(item: dict) -> str:
    metric = item["metric"]
    lines = [
        "You are given a dataset, plus several reference settings with their measured losses.",
        "Use the references to calibrate your judgment — do not rely on general priors.",
        "",
        f"[Dataset]",
        f"problem: {item['problem_id']}",
        f"metric: {metric} (lower is better)",
        "",
        f"[Reference settings ({len(item['demos'])})]",
    ]
    for i, demo in enumerate(item["demos"], start=1):
        lines.append(f"{i}. {setting_summary(demo['setting'])}  =>  loss = {demo['loss']:.4g}")
    lines.append("")
    lines.append(f"[Question] Which setting has the {item['ask'].upper()} loss?")
    for letter in ("A", "B"):
        t = item["target"][letter]
        lines.append(f"{letter}. {setting_summary(t['setting'])}")
    lines.append("")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items-per-problem", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    (args.out / "items").mkdir(parents=True, exist_ok=True)
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)

    ratios, built, skipped = [], 0, 0
    for problem_id in repo.list_problems():
        spec = repo.read_problem_spec(problem_id)
        metric = spec.get("selection_metric", "test_mse")
        n_items = 0
        for _ in range(args.items_per_problem):
            item = build_item(problem_id, rng, metric)
            if item is None:
                continue
            n_items += 1
            ratios.append(item["statistics"]["ratio"])
            tag = f"{problem_id}_{n_items}"
            (args.out / "items" / f"{tag}.json").write_text(
                json.dumps(item, indent=2), encoding="utf-8")
            (args.out / "prompts" / f"{tag}.txt").write_text(
                render_prompt(item), encoding="utf-8")
        if n_items == 0:
            skipped += 1
        else:
            built += n_items
            print(f"{problem_id}: {n_items} items")

    import statistics
    print(f"\nbuilt {built} items across {len(repo.list_problems()) - skipped} problems "
          f"({skipped} problems skipped: too few candidates)")
    if ratios:
        print(f"target ratio: median={statistics.median(ratios):.2f}x "
              f"min={min(ratios):.2f}x max={max(ratios):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
