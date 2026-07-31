"""Difficulty / hardness filter for ArchitectureIQ questions.

Scores each question on three ORTHOGONAL axes (a good hard question is high on all):

1. VALIDITY  — is the answer decided by the data, not noise?
   Signal: GT win_rate, gap, non-overlap of winner vs runner-up std bands.
   (A question can only be "hard" if it is first "valid".)

2. ANTI-HEURISTIC — does the winner DEFY the lazy heuristics?
   We check whether the correct choice is the max-param / deepest / widest model,
   or the lone/most-aggressive Adam. Each heuristic the winner defies = harder.

3. BLIND DIFFICULTY — would a simple heuristic ENSEMBLE get it wrong?
   We simulate several one-line "expert shortcuts" and see how many pick the wrong
   answer. High shortcut-miss-rate = the question can't be solved by pattern matching.

Reads artifacts only (question.json + candidate_spec.json + results/summary.json).
Does NOT import training code or recompute GT. Usage:
    python tools/difficulty/score_questions.py [--glob 'data/datasets/**/question.json'] [--top 20]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

DATA_ROOT = Path("data")  # candidate_path in question.json is relative to data/


def _read(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def mlp_param_count(m: dict, in_dim: int) -> int:
    """Exact-ish MLP parameter count from an mlp model spec."""
    width = int(m.get("width", 0))
    depth = int(m.get("depth", 0))
    if width == 0 or depth == 0:
        return 0
    p = 0
    prev = in_dim
    for _ in range(depth):
        p += prev * width + width  # linear + bias
        prev = width
    p += prev * 1 + 1  # output head (scalar regression)
    # layer norm params (2 per feature per normed layer)
    ln = m.get("layer_norm", [])
    if isinstance(ln, list):
        p += 2 * width * sum(1 for f in ln if f)
    return p


def transformer_param_count(m: dict, vocab: int) -> int:
    d = int(m.get("d_model", 0))
    L = int(m.get("num_layers", 0))
    dff = int(m.get("d_ff", 4 * d if d else 0))
    if d == 0 or L == 0:
        return 0
    per_layer = 4 * d * d + 2 * d * dff  # attn qkvo + 2 ffn matrices
    return vocab * d + L * per_layer + d * vocab  # embed + layers + unembed


def param_count(spec: dict) -> int:
    m = spec["model"]
    if m["type"] == "mlp":
        in_dim = int(m.get("input_dim", 1))
        return mlp_param_count(m, in_dim)
    return transformer_param_count(m, int(m.get("vocab_size", 32)))


def model_depth(m: dict) -> int:
    return int(m.get("depth", m.get("num_layers", 0)))


def model_width(m: dict) -> int:
    return int(m.get("width", m.get("d_model", 0)))


def opt_aggressiveness(o: dict) -> float:
    """Higher = more aggressive/adaptive optimizer, the naive 'safe' pick."""
    lr = float(o.get("lr", 0))
    adaptive = 1.0 if o.get("type") in {"Adam", "AdamW", "RMSprop", "Adagrad"} else 0.0
    return adaptive * 10 + lr  # adaptivity dominates, lr breaks ties


def _winner_letter(q: dict) -> str:
    return q["correct_letter"]


def score_question(qpath: Path) -> dict[str, Any] | None:
    q = _read(qpath)
    choices = q["choices"]
    if len(choices) < 2:
        return None
    specs = {}
    sums = {}
    for c in choices:
        cp = DATA_ROOT / c["candidate_path"]
        specs[c["letter"]] = _read(cp / "candidate_spec.json")
        sums[c["letter"]] = _read(cp / "results" / "summary.json")
    correct = _winner_letter(q)
    metric = q["evaluation"]["selection_metric"]
    mean_key, std_key = f"mean_{metric}", f"std_{metric}"

    # --- Axis 1: validity (from stored GT) ---
    sig = q.get("significance", {})
    win_rate = float(sig.get("win_rate", 0.0))
    gap = float(sig.get("gap", 0.0))
    letters = list(specs)
    means = {L: sums[L].get(mean_key, math.inf) for L in letters}
    stds = {L: sums[L].get(std_key, math.inf) for L in letters}
    ordered = sorted(letters, key=lambda L: means[L])
    winner, runner = ordered[0], ordered[1]
    # normalized gap: gap relative to winner mean (scale-free)
    norm_gap = gap / (abs(means[winner]) + 1e-9)
    non_overlap = means[winner] + stds[winner] < means[runner] - stds[runner]
    valid = win_rate >= 0.7 and gap > 0 and (correct == winner)

    # --- Axis 2: anti-heuristic (winner defies lazy picks) ---
    pc = {L: param_count(specs[L]) for L in letters}
    dpt = {L: model_depth(specs[L]["model"]) for L in letters}
    wid = {L: model_width(specs[L]["model"]) for L in letters}
    agg = {L: opt_aggressiveness(specs[L]["optimizer"]) for L in letters}
    def argmax(d):
        return max(d, key=d.get)
    def argmin(d):
        return min(d, key=d.get)
    heuristics = {
        "pick_max_param": argmax(pc),
        "pick_min_param": argmin(pc),
        "pick_deepest": argmax(dpt),
        "pick_widest": argmax(wid),
        "pick_most_adaptive_opt": argmax(agg),
    }
    # only count heuristics that are actually discriminative (not all-equal)
    discriminative = {}
    for name, pick in heuristics.items():
        base = {"pick_max_param": pc, "pick_min_param": pc, "pick_deepest": dpt,
                "pick_widest": wid, "pick_most_adaptive_opt": agg}[name]
        if len(set(base.values())) > 1:
            discriminative[name] = (pick == correct)
    n_disc = len(discriminative)
    n_heur_correct = sum(1 for v in discriminative.values() if v)
    # anti-heuristic score: fraction of discriminative heuristics that MISS
    anti_heuristic = (n_disc - n_heur_correct) / n_disc if n_disc else 0.0

    # --- Axis 3: blind difficulty proxy ---
    # A naive solver would follow the majority vote of discriminative heuristics.
    if discriminative:
        votes = {}
        for name in discriminative:
            base = {"pick_max_param": pc, "pick_min_param": pc, "pick_deepest": dpt,
                    "pick_widest": wid, "pick_most_adaptive_opt": agg}[name]
            # each heuristic's pick
            pick = argmax(base) if name != "pick_min_param" else argmin(base)
            votes[pick] = votes.get(pick, 0) + 1
        ensemble_pick = max(votes, key=votes.get)
        ensemble_wrong = ensemble_pick != correct
    else:
        ensemble_wrong = False

    # --- composite hardness (only meaningful if valid) ---
    # weight: must be valid; reward anti-heuristic and ensemble-wrong; reward tight-but-clean gap
    hardness = 0.0
    if valid:
        hardness = (
            0.5 * anti_heuristic
            + 0.3 * (1.0 if ensemble_wrong else 0.0)
            + 0.2 * min(win_rate, 1.0)  # stable GT is part of a GOOD hard question
        )

    return {
        "question": str(qpath.parent.relative_to(DATA_ROOT / "datasets")),
        "qid": q["question_id"],
        "type": q["type"],
        "family": q["family"],
        "num_choices": len(choices),
        "correct": correct,
        "valid": valid,
        "win_rate": round(win_rate, 3),
        "gap": round(gap, 5),
        "norm_gap": round(norm_gap, 4),
        "non_overlap": non_overlap,
        "n_discriminative_heuristics": n_disc,
        "heuristics_correct": n_heur_correct,
        "anti_heuristic": round(anti_heuristic, 3),
        "ensemble_heuristic_wrong": ensemble_wrong,
        "hardness": round(hardness, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/datasets/**/questions/run_*/q_*/question.json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="tools/difficulty/_scores.json")
    args = ap.parse_args()

    rows = []
    for f in glob.glob(args.glob, recursive=True):
        try:
            r = score_question(Path(f))
            if r:
                rows.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f}: {exc}")

    rows.sort(key=lambda r: (r["hardness"], r["anti_heuristic"], r["norm_gap"]), reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    valid = [r for r in rows if r["valid"]]
    print(f"scored {len(rows)} questions ({len(valid)} valid)")
    ens = sum(1 for r in valid if r["ensemble_heuristic_wrong"])
    print(f"valid questions where naive heuristic-ensemble is WRONG: {ens}/{len(valid)}")
    print(f"\n=== TOP {args.top} HARDEST (valid) ===")
    print(f"{'hardness':>8} {'anti':>5} {'ens✗':>4} {'wr':>4} {'ngap':>6}  {'type':16} {'fam':20} {'qid'}")
    for r in valid[: args.top]:
        print(f"{r['hardness']:>8.3f} {r['anti_heuristic']:>5.2f} "
              f"{'Y' if r['ensemble_heuristic_wrong'] else '.':>4} {r['win_rate']:>4.2f} "
              f"{r['norm_gap']:>6.2f}  {r['type']:16} {r['family']:20} {r['qid']}")


if __name__ == "__main__":
    main()
