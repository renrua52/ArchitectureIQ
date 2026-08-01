"""Rank-based scoring for eval responses (partial credit instead of all-or-nothing).

Ranking contract (2026-08-01 decision):
  * Each option's ground-truth rank is its position by the dataset's selection
    metric (lower is better) among the question's options, computed from the
    on-disk ``results/summary.json`` (the same GT authority the question
    builder and scorer use).
  * The model's choice scores ``n_options - rank`` points: 1st place = 5,
    ... 6th place = 0 for a 6-option select_best; for two_choice this
    degenerates to 1 / 0 (i.e. the old binary accuracy).

Because rank is derived from option candidate ids + current summaries, it is
robust to stale/incorrect embedded answer keys (see docs/eval-sets.md §8).

Usage:
    .venv/bin/python -m backend.eval.ranking --responses <responses.jsonl> \\
        --set select_best_v2          # select_best responses
    .venv/bin/python -m backend.eval.ranking --responses <responses.jsonl> \\
        --two-choice-dir artifacts/eval_probe_local/items
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from architecture_iq.storage import repository as repo

SETS_ROOT = Path("backend/eval/sets")


def option_means(item: dict) -> dict[str, float] | None:
    """letter -> GT mean metric (lower is better), from on-disk summary.json."""
    metric = item.get("metric")
    if not metric:
        return None
    mean_key = f"mean_{metric}"
    if "options" in item:
        entries = [(o["letter"], o["candidate_id"]) for o in item["options"]]
    elif "target" in item:
        entries = [(k, v["candidate_id"]) for k, v in item["target"].items()]
    else:
        return None
    means: dict[str, float] = {}
    for letter, cid in entries:
        try:
            s = repo.read_summary(item["problem_id"], cid)
        except FileNotFoundError:
            return None
        if mean_key not in s:
            return None
        means[letter] = float(s[mean_key])
    return means


def rank_result(item: dict, answer: str | None) -> dict:
    """GT rank (1 = best) and rank score (n - rank) for the model's answer."""
    means = option_means(item)
    n = len(means) if means else 0
    if not means or not answer or answer not in means:
        return {"rank": None, "rank_score": None, "n_options": n, "top1": None,
                "top2": None, "top3": None}
    # select_best is always lower-is-better; two_choice may ask for higher.
    reverse = item.get("ask") == "higher"
    order = sorted(means, key=means.get, reverse=reverse)
    rank = order.index(answer) + 1
    return {
        "rank": rank,
        "rank_score": n - rank,
        "n_options": n,
        "top1": rank == 1,
        "top2": rank <= 2,
        "top3": rank <= 3,
    }


def score_rows(rows: list[dict]) -> list[dict]:
    """Fill rank fields into saved response rows (in place) and return them."""
    for r in rows:
        if r.get("rank") is not None:
            continue
        item = {"problem_id": r.get("problem_id"), "metric": r.get("metric"),
                "ask": r.get("ask")}
        options = r.get("options")
        target = r.get("target")
        if options:
            item["options"] = options
        elif target:
            item["target"] = target
        info = rank_result(item, r.get("answer"))
        r.update(info)
        # top1 == is_correct when the embedded key agrees with current GT
        if info["top1"] is not None:
            r["is_correct"] = bool(info["top1"])
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    scored = [r for r in rows if r.get("rank") is not None]
    n = len(scored)
    if not n:
        return {"label": label, "scored": 0}
    n_choices = max((r.get("n_options") or 0) for r in scored)
    ranks = [r["rank"] for r in scored]
    scores = [r["rank_score"] for r in scored]
    out = {
        "label": label,
        "scored": n,
        "n_options": n_choices,
        "mean_rank": round(statistics.mean(ranks), 3),
        "median_rank": statistics.median(ranks),
        "mean_rank_score": round(statistics.mean(scores), 3),
        "max_rank_score": n_choices - 1,
        "rank_score_norm": round(statistics.mean(scores) / max(n_choices - 1, 1), 4),
        "top1": round(sum(1 for r in scored if r["top1"]) / n, 4),
        "top2": round(sum(1 for r in scored if r["top2"]) / n, 4),
        "top3": round(sum(1 for r in scored if r["top3"]) / n, 4),
        "rank_dist": dict(Counter(ranks)),
    }
    # stratum breakdown by winner-vs-runner-up ratio (embedded stats)
    stratum_tot: Counter = Counter()
    stratum_rank: dict[str, list[int]] = {}
    for r in scored:
        ratio = r.get("ratio") or 0
        if ratio < 1.15:
            name = "tight(<1.15)"
        elif ratio < 2.0:
            name = "medium(1.15-2)"
        else:
            name = "loose(>=2)"
        stratum_tot[name] += 1
        stratum_rank.setdefault(name, []).append(r["rank"])
    out["by_stratum"] = {
        k: {"n": stratum_tot[k], "mean_rank": round(statistics.mean(stratum_rank[k]), 3),
            "top1": round(sum(1 for rank in stratum_rank[k] if rank == 1) / stratum_tot[k], 4)}
        for k in stratum_tot
    }
    return out


def load_items(set_name: str, two_choice_dir: str | None) -> dict[str, dict]:
    items: dict[str, dict] = {}
    if set_name:
        path = SETS_ROOT / set_name / "questions.jsonl"
        for line in path.open(encoding="utf-8"):
            q = json.loads(line)
            items[q["question_id"]] = q
    if two_choice_dir:
        for f in Path(two_choice_dir).glob("*.json"):
            q = json.load(open(f))
            items[f.stem] = q
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", required=True, help="responses jsonl from batch_eval")
    ap.add_argument("--set", default=None, help="select_best set name (e.g. select_best_v2)")
    ap.add_argument("--two-choice-dir", default=None, help="two-choice item dir")
    args = ap.parse_args()
    items = load_items(args.set, args.two_choice_dir)
    rows = [json.loads(l) for l in open(args.responses, encoding="utf-8")]
    # attach the item (options/target/refs) to each row for rank computation
    for r in rows:
        it = items.get(r.get("question_id")) or items.get(str(r.get("question_id")))
        if it:
            r["problem_id"] = it.get("problem_id", r.get("problem_id"))
            r["metric"] = it.get("metric", r.get("metric"))
            r["ask"] = it.get("ask", r.get("ask"))
            if "options" in it:
                r["options"] = it["options"]
            if "target" in it:
                r["target"] = it["target"]
            r["ratio"] = r.get("ratio") or it.get("statistics", {}).get("ratio")
    score_rows(rows)
    print(json.dumps(summarize(rows, Path(args.responses).stem), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
