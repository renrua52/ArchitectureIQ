#!/usr/bin/env python
"""Re-evaluate saved predictions using only triples that pass the full
benchmark significance validator (gap + win_rate + non_overlap).

Vectorized version: precompute per-candidate arrays, enumerate triples with
numpy indexing, and batch the significance checks.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np


def load_gt_rows(base: Path) -> list[dict]:
    rows: list[dict] = []
    for env in sorted(os.listdir(base)):
        for split in ("validation", "train"):
            p = base / env / f"{split}.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
    return rows


def load_candidate_summary(base: Path, row: dict) -> dict | None:
    sp = base / row["provenance"]["summary_path"]
    if not sp.exists():
        return None
    return json.loads(sp.read_text())


def evaluate_env_vectorized(
    env_rows: list[dict],
    pred_by_fp: dict[str, float],
    summaries_by_fp: dict[str, dict],
    metric: str,
    *,
    gap_min: float,
    win_rate_min: float,
    use_non_overlap: bool,
    higher_is_better: bool = False,
) -> dict:
    n = len(env_rows)
    if n < 3:
        return None

    fps = [r["example_fingerprint_sha256"] for r in env_rows]
    # skip if any prediction missing
    if any(fp not in pred_by_fp for fp in fps):
        return None
    if any(fp not in summaries_by_fp for fp in fps):
        return None

    raw_losses = np.array(
        [r["target"]["mean_loss"] for r in env_rows], dtype=np.float64
    )
    pred_log = np.array([pred_by_fp[fp] for fp in fps], dtype=np.float64)
    pred_raw = np.exp(pred_log)

    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    final_key = f"final_{metric}"

    means = np.array(
        [summaries_by_fp[fp][mean_key] for fp in fps], dtype=np.float64
    )
    stds = np.array(
        [summaries_by_fp[fp][std_key] for fp in fps], dtype=np.float64
    )
    # seed finals matrix: (n_cands, n_seeds)
    seed_finals = []
    for fp in fps:
        sr = summaries_by_fp[fp]["seed_results"]
        seed_finals.append(
            [float("inf") if s["failed"] else s[final_key] for s in sr]
        )
    seed_finals = np.array(seed_finals, dtype=np.float64)  # (n, n_seeds)
    n_seeds = seed_finals.shape[1]

    # enumerate all triples
    idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    nt = idx.shape[0]

    # gather per-triple arrays
    t_means = means[idx]  # (nt, 3)
    t_stds = stds[idx]
    t_raw = raw_losses[idx]
    t_pred = pred_raw[idx]
    t_seed = seed_finals[idx]  # (nt, 3, n_seeds)

    # winner = argmin (minimize metric). if higher_is_better, negate.
    if higher_is_better:
        order_means = -t_means
        order_seed = -t_seed
    else:
        order_means = t_means
        order_seed = t_seed

    np.argmin(order_means, axis=1)  # (nt,)
    # sort to get runner-up: second smallest
    sorted_idx = np.argsort(order_means, axis=1)
    w = sorted_idx[:, 0]
    ru = sorted_idx[:, 1]
    gaps = np.abs(t_means[np.arange(nt), ru] - t_means[np.arange(nt), w])

    # prediction selection
    sel = np.argmin(t_pred, axis=1)  # (nt,)
    best = np.argmin(t_raw, axis=1)
    correct = sel == best

    # all-triples accuracy
    all_correct = int(correct.sum())
    all_total = nt

    # gap >= 0.05 filter
    gap_mask = gaps >= gap_min
    gap_total = int(gap_mask.sum())
    gap_correct = int(correct[gap_mask].sum()) if gap_total else 0

    # full significance: gap, win_rate, non_overlap
    sig_pass = gaps >= gap_min  # (nt,)

    # win_rate: for each seed, argmin of seed finals; count matches winner
    seed_winners = np.argmin(order_seed, axis=1)  # (nt, n_seeds)
    wins = (seed_winners == w[:, None]).sum(axis=1)  # (nt,)
    win_rate = wins / n_seeds
    sig_pass &= win_rate >= win_rate_min

    if use_non_overlap:
        non_overlap_ok = (
            t_means[np.arange(nt), w] + t_stds[np.arange(nt), w]
            < t_means[np.arange(nt), ru] - t_stds[np.arange(nt), ru]
        )
        sig_pass &= non_overlap_ok

    sig_total = int(sig_pass.sum())
    sig_correct = int(correct[sig_pass].sum()) if sig_total else 0

    return {
        "environment": env_rows[0]["experiment_id"],
        "family": env_rows[0]["family"],
        "n_rows": n,
        "n_triples": all_total,
        "n_significant": sig_total,
        "all_accuracy": all_correct / all_total if all_total else None,
        "n_gap_ge_0_05": gap_total,
        "gap_ge_0_05_accuracy": (
            gap_correct / gap_total if gap_total else None
        ),
        "significant_accuracy": (
            sig_correct / sig_total if sig_total else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-base", default="data/meta_model/setting_to_loss_wide_v2")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gap-min", type=float, default=0.05)
    ap.add_argument("--win-rate-min", type=float, default=0.7)
    ap.add_argument("--no-non-overlap", action="store_true")
    ap.add_argument("--split", default="validation", choices=["validation", "train", "both"])
    args = ap.parse_args()

    base = Path(args.gt_base)
    print("loading GT rows...")
    rows = load_gt_rows(base)
    if args.split != "both":
        rows = [r for r in rows if r["split"] == args.split]
    print(f"  {len(rows)} rows (split={args.split})")

    metric_by_fam: dict[str, str] = {}
    for r in rows:
        metric_by_fam[r["family"]] = r["target"]["selection_metric"]
    print("metric by family:", metric_by_fam)

    print("loading candidate summaries...")
    summaries_by_fp: dict[str, dict] = {}
    for r in rows:
        fp = r["example_fingerprint_sha256"]
        if fp in summaries_by_fp:
            continue
        sj = load_candidate_summary(base, r)
        if sj is not None:
            summaries_by_fp[fp] = sj
    print(f"  {len(summaries_by_fp)} summaries loaded")

    print("loading predictions...")
    pred_doc = json.loads(Path(args.predictions).read_text())
    methods = pred_doc["methods"]

    # group rows by environment
    by_env: dict[str, list[dict]] = {}
    for r in rows:
        by_env.setdefault(r["experiment_id"], []).append(r)
    print(f"  {len(by_env)} environments")

    results: dict[str, dict] = {}
    for m in methods:
        name = m["method"]
        print(f"method: {name}", flush=True)
        pred_by_fp = {
            p["example_fingerprint_sha256"]: p["predicted_log_loss"]
            for p in m["predictions"]
        }
        env_stats = []
        tot_triples = 0
        tot_sig = 0
        tot_all_correct = 0
        tot_gap_correct = 0
        tot_gap_total = 0
        tot_sig_correct = 0
        for env, env_rows in by_env.items():
            fam = env_rows[0]["family"]
            metric = metric_by_fam.get(fam, "test_mse")
            res = evaluate_env_vectorized(
                env_rows,
                pred_by_fp,
                summaries_by_fp,
                metric,
                gap_min=args.gap_min,
                win_rate_min=args.win_rate_min,
                use_non_overlap=not args.no_non_overlap,
                higher_is_better=False,
            )
            if res is None:
                continue
            env_stats.append(res)
            tot_triples += res["n_triples"]
            tot_sig += res["n_significant"]
            tot_all_correct += int(res["all_accuracy"] * res["n_triples"])
            tot_gap_total += res["n_gap_ge_0_05"]
            tot_gap_correct += int(
                (res["gap_ge_0_05_accuracy"] or 0) * res["n_gap_ge_0_05"]
            )
            tot_sig_correct += int(
                (res["significant_accuracy"] or 0) * res["n_significant"]
            )

        def macro(key):
            vals = [s[key] for s in env_stats if s.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        results[name] = {
            "n_environments": len(env_stats),
            "total_triples": tot_triples,
            "n_significant_triples": tot_sig,
            "macro": {
                "all_three_choice_accuracy": macro("all_accuracy"),
                "gap_ge_0_05_three_choice_accuracy": macro(
                    "gap_ge_0_05_accuracy"
                ),
                "significant_three_choice_accuracy": macro(
                    "significant_accuracy"
                ),
            },
            "micro": {
                "all_three_choice_accuracy": (
                    tot_all_correct / tot_triples if tot_triples else None
                ),
                "gap_ge_0_05_three_choice_accuracy": (
                    tot_gap_correct / tot_gap_total if tot_gap_total else None
                ),
                "significant_three_choice_accuracy": (
                    tot_sig_correct / tot_sig if tot_sig else None
                ),
            },
            "per_environment": env_stats,
        }
        mac = results[name]["macro"]
        print(
            f"  macro all={mac['all_three_choice_accuracy']}, "
            f"gap_ge_0_05={mac['gap_ge_0_05_three_choice_accuracy']}, "
            f"significant={mac['significant_three_choice_accuracy']}"
        )
        print(f"  sig triples: {tot_sig}/{tot_triples}")

    out = {
        "predictions_source": args.predictions,
        "gt_base": str(base),
        "split": args.split,
        "significance": {
            "gap_min": args.gap_min,
            "win_rate_min": args.win_rate_min,
            "use_non_overlap": not args.no_non_overlap,
            "higher_is_better": False,
        },
        "metric_by_family": metric_by_fam,
        "methods": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
