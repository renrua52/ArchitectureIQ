"""Convert the legacy 60-question 3-choice bundle into eval-set format.

Design (2026-08-01 discussion, user):
  * The old question's three choices become calibration hints (references
    with measured losses) — "这三选一的几个题目都应该在 hint 里面".
  * New options are drawn from the same candidate pool a few config edits
    away from the old winner, so each question tests whether a model can
    transfer "similar experiments" (the hints) to unseen nearby settings.
    Option losses are never shown; the model picks the lowest-loss option.
  * This is the legacy-bundle analogue of select_best (see questions.py);
    the harness, ranking scoring and report tooling are shared.

Usage:
    .venv/bin/python -m backend.eval.old60 --set-name select_best_old60
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from architecture_iq.storage import repository as repo
from architecture_iq.util import short_hash

from backend.eval.questions import (
    N_REFS,
    _mean_key,
    _pick_references,
    _win_rate,
    config_edit_distance,
    ratio_floor,
    render_select_best,
    salient_distance,
)

SCHEMA_VERSION = "0.1"
SETS_ROOT = Path("backend/eval/sets")
OLD_QUESTIONS_ROOT = Path("examples/quiz_demo/bundle/datasets")
N_OPTIONS = 3          # keep the legacy 3-choice structure for comparability
MAX_EDIT_DIST = 8      # options must be a few config edits from the old winner
PAIRWISE_SALIENT_MIN = 2


def _floors(metric: str) -> tuple[float, float]:
    """(ratio_floor, win_rate_min). CE (bigram) is compressed, so it is relaxed."""
    if metric == "test_ce":
        return 1.02, 0.6
    return 1.15, 0.7


def load_old_questions() -> list[dict]:
    qs = []
    for qf in sorted(OLD_QUESTIONS_ROOT.glob("*/*/questions/*/q_*/question.json")):
        qs.append(json.load(open(qf)))
    return qs


def load_rows(problem_id: str) -> tuple[str, list[tuple[str, dict, dict]]]:
    spec = repo.read_problem_spec(problem_id)
    metric = spec.get("selection_metric", "test_mse")
    mean_key = _mean_key(metric)
    rows = []
    for cid in repo.list_candidate_ids(problem_id):
        try:
            cfg = repo.read_candidate_config(problem_id, cid)
            s = repo.read_summary(problem_id, cid)
        except FileNotFoundError:
            continue
        if s.get("excluded") or mean_key not in s:
            continue
        rows.append((cid, cfg, s))
    rows.sort(key=lambda r: float(r[2][mean_key]))
    return metric, rows


def _pick_options(rows: list[tuple], base_id: str, base_cfg: dict, metric: str,
                  exclude: set[str], rng: random.Random) -> list[tuple[str, dict, dict, int]]:
    """Nearby (few-edit) options near the old winner, pairwise distinct."""
    scored = []
    for cid, cfg, s in rows:
        if cid in exclude or cid == base_id:
            continue
        d = config_edit_distance(base_cfg, cfg)
        sd = salient_distance(base_cfg, cfg)
        if d < 1 or d > MAX_EDIT_DIST or not (1 <= sd <= 3):
            continue
        scored.append((d, sd, cid, cfg, s))
    rng.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1], rng.random()))
    picked, seen = [], set()
    for d, sd, cid, cfg, s in scored:
        if cid in seen:
            continue
        if all(salient_distance(cfg, pc) >= PAIRWISE_SALIENT_MIN for _, _, _, pc, _ in picked):
            picked.append((d, sd, cid, cfg, s))
            seen.add(cid)
        if len(picked) >= N_OPTIONS:
            break
    return [(cid, cfg, s, d) for d, sd, cid, cfg, s in picked]


def build_one(old_q: dict, rng: random.Random, max_tries: int = 20) -> dict | None:
    pid = old_q["dataset_id"]
    metric, rows = load_rows(pid)
    mean_key = _mean_key(metric)
    hint_ids = [ch["candidate_id"] for ch in old_q["choices"]]
    by_id = {cid: (cfg, s) for cid, cfg, s in rows}
    hints = [(cid, *by_id[cid]) for cid in hint_ids if cid in by_id]
    if len(hints) < 3:
        return None
    base_id = min(hints, key=lambda h: float(h[2][mean_key]))[0]
    base_cfg = by_id[base_id][0]

    floor, wr_min = _floors(metric)
    for _ in range(max_tries):
        options = _pick_options(rows, base_id, base_cfg, metric, set(hint_ids), rng)
        if len(options) < N_OPTIONS:
            return None
        pool = [(cid, cfg, s) for cid, cfg, s, _ in options]
        means = [float(s[mean_key]) for _, _, s in pool]
        winner_i = min(range(len(pool)), key=lambda i: means[i])
        runner_i = min((i for i in range(len(pool)) if i != winner_i), key=lambda i: means[i])
        wr = _win_rate(pool[winner_i][2], pool[runner_i][2], metric)
        ratio = max(means[winner_i], means[runner_i]) / max(min(means[winner_i], means[runner_i]), 1e-12)
        if wr < wr_min or ratio < floor:
            continue

        # hints = the 3 old choices + 2 nearest others (bracketing the hint range)
        hint_means = [float(s[mean_key]) for _, _, s in hints]
        lo, hi = min(hint_means), max(hint_means)
        refs = [{"candidate_id": cid, "setting": cfg, "loss": float(s[mean_key])}
                for cid, cfg, s in hints]
        extra = _pick_references(rows, set(hint_ids) | {p[0] for p in pool}, metric, rng,
                                 anchor_range=(lo, hi))
        if len(extra) < N_REFS - 3:
            continue
        refs = refs + extra[: N_REFS - 3]
        rng.shuffle(refs)

        winner_id = pool[winner_i][0]
        rng.shuffle(pool)
        letters = [chr(ord("A") + i) for i in range(len(pool))]
        options_out, correct = [], None
        for letter, (cid, cfg, _) in zip(letters, pool):
            options_out.append({"letter": letter, "candidate_id": cid,
                                "is_base": cid == base_id, "setting": cfg})
            if cid == winner_id:
                correct = letter

        body = {
            "schema_version": SCHEMA_VERSION,
            "type": "select_best_old60",
            "problem_id": pid,
            "metric": metric,
            "references": refs,
            "options": options_out,
            "correct_letter": correct,
            "statistics": {
                "winner_candidate": winner_id,
                "runner_up_candidate": pool[runner_i][0],
                "ratio": round(ratio, 3),
                "win_rate": round(wr, 3),
                "n_seeds": len(pool[0][2]["seed_results"]),
            },
            "provenance": {
                "old_question_id": old_q["question_id"],
                "old_correct_letter": old_q["correct_letter"],
                "hint_candidate_ids": hint_ids,
                "base_candidate_id": base_id,
            },
        }
        body["question_id"] = f"sb60_{short_hash(body)}"
        body["prompt"] = render_select_best(body)
        return body
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-name", default="select_best_old60")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--limit", type=int, default=0, help="0 = all old questions")
    args = ap.parse_args()

    old_qs = load_old_questions()
    if args.limit:
        old_qs = old_qs[: args.limit]
    rng = random.Random(args.seed)
    items, skipped = [], []
    for q in old_qs:
        item = build_one(q, rng)
        if item is None:
            skipped.append(q["question_id"])
            continue
        items.append(item)
    # dedupe identical question bodies
    seen, uniq = set(), []
    for it in items:
        h = it["question_id"]
        if h in seen:
            continue
        seen.add(h)
        uniq.append(it)

    out_dir = SETS_ROOT / args.set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for item in uniq:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "set_name": args.set_name,
        "type": "select_best_old60",
        "seed": args.seed,
        "items": len(uniq),
        "old_questions_loaded": len(old_qs),
        "skipped_old_questions": skipped,
        "filters": {"win_rate_min": "per-metric (CE 0.6 / MSE 0.7)", "ratio_min": "per-metric (CE 1.02 / MSE 1.15)",
                    "n_references": N_REFS, "n_options": N_OPTIONS,
                    "max_edit_dist_from_winner": MAX_EDIT_DIST,
                    "salient_min": 1, "salient_max": 2, "pairwise_salient_min": 2},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(f"wrote {len(uniq)} items to {out_dir}/questions.jsonl")
    print(f"skipped {len(skipped)} old questions: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
