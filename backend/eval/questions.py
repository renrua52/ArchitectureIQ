"""Eval-side question-set (题集) generator.

Design (per 2026-08-01 discussion):
  * A 题集 is a JSONL file under ``backend/eval/sets/{set_name}/questions.jsonl``;
    one JSON per line, each line = one question instance (which problem, which
    candidate ids, references, options, answer, prompt text).
  * Type ``select_best``: 5 random reference settings (with losses) + options =
    base setting + 5 modified settings (base included, losses NOT shown).
    Model picks the lowest-loss setting. Cross-seed gap must not be too small.
  * Type ``propose_improvement``: 5 random references + base + 5 modified
    settings WITH their losses revealed; model proposes a NEW setting (JSON
    config); scored later by running GT on the proposal.
  * Options are built from existing candidates in ``backend/data`` that are
    close to the base (few config edits), so all GT is already on disk.

Usage:
    .venv/bin/python -m backend.eval.questions --type select_best --items-per-problem 5
    .venv/bin/python -m backend.eval.questions --type propose_improvement --items-per-problem 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from architecture_iq.prompts.formatters import (
    format_loss_nl,
    format_model_nl,
    format_optimizer_nl,
    format_training_schedule,
)
from architecture_iq.storage import repository as repo
from architecture_iq.util import short_hash

SCHEMA_VERSION = "0.1"
SETS_ROOT = Path("backend/eval/sets")
N_REFS = 5          # random reference settings with losses
N_MODS = 5          # modified settings per base
WIN_RATE_MIN = 0.7  # winner vs runner-up, cross-seed
RATIO_MIN = 1.05    # mild gap requirement (user: cross-seed gap not too small)


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def _final_key(metric: str) -> str:
    return f"final_{metric}"


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v, sort_keys=True)
        else:
            out[key] = v
    return out


def config_edit_distance(a: dict, b: dict) -> int:
    fa, fb = _flatten(a), _flatten(b)
    keys = set(fa) | set(fb)
    return sum(1 for k in keys if fa.get(k) != fb.get(k))


# Salient keys shown prominently to the model: two options that differ only in
# non-salient fields (e.g. per-layer activations) read as duplicates and are
# ill-formed choices. select_best v1.2 requires pairwise salient distance >= 2.
SALIENT_KEYS = (
    "model.type",
    "model.depth",
    "model.num_layers",
    "model.width",
    "model.d_model",
    "model.residual",
    "optimizer.type",
    "optimizer.lr",
    "optimizer.weight_decay",
    "loss.loss_id",
    "budget.batch_size",
)


def salient_distance(a: dict, b: dict) -> int:
    fa, fb = _flatten(a), _flatten(b)
    return sum(1 for k in SALIENT_KEYS if fa.get(k) != fb.get(k))


def ratio_floor(metric: str) -> float:
    """Per-metric winner-vs-runner-up ratio floor (cross-seed gap not too small).

    CE is compressed (bigram max/min ~1.14x) so its floor is lower than MSE's.
    """
    return 1.15 if metric == "test_mse" else 1.03


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


def _load(problem_id: str) -> tuple[str, list[tuple[str, dict, dict]]]:
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


def _win_rate(sum_a: dict, sum_b: dict, metric: str) -> float:
    final_key = _final_key(metric)
    n = len(sum_a["seed_results"])
    wins = sum(1 for i in range(n)
               if float(sum_a["seed_results"][i][final_key]) < float(sum_b["seed_results"][i][final_key]))
    return wins / n


def _pick_references(rows: list[tuple], exclude: set[str], metric: str,
                       rng: random.Random, anchor_range: tuple[float, float] | None = None) -> list[dict]:
    """Pick N_REFs calibration references, anchored around the option loss range.

    When ``anchor_range`` (lo, hi) is given, references bracket the option
    losses (two below lo, one mid, two above hi) so the model never has to
    extrapolate beyond the reference scale.
    """
    mean_key = _mean_key(metric)
    pool = [r for r in rows if r[0] not in exclude]
    if len(pool) <= N_REFS:
        return []

    def nearest(target: float, seen: set[int]) -> int | None:
        best, best_i = None, None
        for i, (_, _, s) in enumerate(pool):
            if i in seen:
                continue
            d = abs(float(s[mean_key]) - target)
            if best is None or d < best:
                best, best_i = d, i
        return best_i

    if anchor_range is not None:
        lo, hi = anchor_range
        if lo > 0 and hi > 0 and hi >= lo:
            targets = [lo * 0.9, lo * 1.05, (lo + hi) / 2, hi * 0.95, hi * 1.1]
        else:
            targets = [lo, (lo + hi) / 2, hi, hi, hi]
    else:
        vals = sorted(float(s[mean_key]) for _, _, s in pool)
        lo, hi = vals[0], vals[-1]
        targets = [lo, lo * 0.75 + hi * 0.25, (lo + hi) / 2, lo * 0.25 + hi * 0.75, hi]

    picks, seen = [], set()
    for target in targets:
        i = nearest(target, seen)
        if i is None:
            continue
        seen.add(i)
        picks.append(pool[i])
    rng.shuffle(picks)
    return [{"candidate_id": cid, "setting": cfg, "loss": float(s[mean_key])}
            for cid, cfg, s in picks]


def _pick_mods(rows: list[tuple], base_id: str, base_cfg: dict, metric: str,
               rng: random.Random, *, salient_min: int = 1, salient_max: int = 2,
               pairwise_salient_min: int = 0) -> list[tuple[str, dict, dict, int]]:
    """Modified settings near the base (1-2 salient edits by default).

    ``pairwise_salient_min >= 2`` guarantees no two options read as look-alikes
    (used by select_best v1.2); propose_improvement keeps 0 (demos are shown
    with losses, near-identical pairs are fine there).
    """
    scored = []
    for cid, cfg, s in rows:
        if cid == base_id:
            continue
        d = config_edit_distance(base_cfg, cfg)
        sd = salient_distance(base_cfg, cfg)
        if d < 1 or not (salient_min <= sd <= salient_max):
            continue
        scored.append((d, sd, cid, cfg, s))
    rng.shuffle(scored)
    scored.sort(key=lambda x: (x[0], x[1], rng.random()))
    picked, seen = [], set()
    for d, sd, cid, cfg, s in scored:
        if cid in seen:
            continue
        if all(salient_distance(cfg, pc) >= pairwise_salient_min for _, _, _, pc, _ in picked):
            picked.append((d, sd, cid, cfg, s))
            seen.add(cid)
        if len(picked) >= N_MODS:
            break
    return [(cid, cfg, s, d) for d, sd, cid, cfg, s in picked]


def render_select_best(item: dict) -> str:
    lines = [
        "You are given a dataset and several reference settings with their measured losses.",
        "Use the references to calibrate your judgment; do not rely on general priors.",
        "",
        f"[Dataset]  problem: {item['problem_id']}  metric: {item['metric']} (lower is better)",
        "",
        f"[Reference settings ({len(item['references'])})]",
    ]
    for i, ref in enumerate(item["references"], start=1):
        lines.append(f"{i}. {setting_summary(ref['setting'])}  =>  loss = {ref['loss']:.4g}")
    lines.append("")
    lines.append(f"[Question] Which of the following {len(item['options'])} settings has the LOWEST loss?")
    for opt in item["options"]:
        lines.append(f"{opt['letter']}. {setting_summary(opt['setting'])}")
    lines.append("")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def render_propose_improvement(item: dict) -> str:
    lines = [
        "You are given a dataset, a base setting, and several settings (with losses).",
        "Propose a NEW setting (JSON config) that you expect to beat the base setting.",
        "",
        f"[Dataset]  problem: {item['problem_id']}  metric: {item['metric']} (lower is better)",
        "",
        f"[Reference settings ({len(item['references'])})]",
    ]
    for i, ref in enumerate(item["references"], start=1):
        lines.append(f"{i}. {setting_summary(ref['setting'])}  =>  loss = {ref['loss']:.4g}")
    lines.append("")
    lines.append("[Base setting]")
    lines.append(f"loss = {item['base']['loss']:.4g}")
    lines.append(setting_summary(item["base"]["setting"]))
    lines.append("")
    lines.append("[Other settings with losses (for calibration)]")
    for i, demo in enumerate(item["improved_demos"], start=1):
        lines.append(f"{i}. {setting_summary(demo['setting'])}  =>  loss = {demo['loss']:.4g}")
    lines.append("")
    lines.append("Output your proposed config as a JSON object with keys "
                 "model / optimizer / loss / budget (batch_size).")
    return "\n".join(lines)


def build_select_best(problem_id: str, rng: random.Random, max_tries: int = 20) -> dict | None:
    metric, rows = _load(problem_id)
    mean_key = _mean_key(metric)
    if len(rows) < N_REFS + N_MODS + 2:
        return None

    floor = ratio_floor(metric)
    for _ in range(max_tries):
        base = rows[rng.randrange(len(rows))]
        base_id, base_cfg, base_sum = base
        mods = _pick_mods(rows, base_id, base_cfg, metric, rng, pairwise_salient_min=2)
        if len(mods) < N_MODS:
            continue
        pool = [(base_id, base_cfg, base_sum, True)] + [(cid, cfg, s, False) for cid, cfg, s, _ in mods]
        means = [float(s[mean_key]) for _, _, s, _ in pool]
        winner_i = min(range(len(pool)), key=lambda i: means[i])
        runner_i = min((i for i in range(len(pool)) if i != winner_i), key=lambda i: means[i])
        wr = _win_rate(pool[winner_i][2], pool[runner_i][2], metric)
        ratio = max(means[winner_i], means[runner_i]) / max(min(means[winner_i], means[runner_i]), 1e-12)
        if wr < WIN_RATE_MIN or ratio < floor:
            continue
        winner_id = pool[winner_i][0]
        runner_id = pool[runner_i][0]

        exclude = {p[0] for p in pool}
        opt_lo = min(means)
        opt_hi = max(means)
        refs = _pick_references(rows, exclude, metric, rng, anchor_range=(opt_lo, opt_hi))
        if len(refs) < N_REFS:
            continue

        rng.shuffle(pool)
        letters = [chr(ord("A") + i) for i in range(len(pool))]
        options = []
        correct = None
        for letter, (cid, cfg, _, is_base) in zip(letters, pool):
            options.append({"letter": letter, "candidate_id": cid,
                            "is_base": is_base, "setting": cfg})
            if cid == winner_id:
                correct = letter

        body = {
            "schema_version": SCHEMA_VERSION,
            "type": "select_best",
            "problem_id": problem_id,
            "metric": metric,
            "references": refs,
            "options": options,
            "correct_letter": correct,
            "statistics": {
                "winner_candidate": winner_id,
                "runner_up_candidate": runner_id,
                "ratio": round(ratio, 3),
                "win_rate": round(wr, 3),
                "n_seeds": len(base_sum["seed_results"]),
            },
        }
        body["question_id"] = f"sb_{short_hash(body)}"
        body["prompt"] = render_select_best(body)
        return body
    return None


def build_propose_improvement(problem_id: str, rng: random.Random, max_tries: int = 20) -> dict | None:
    metric, rows = _load(problem_id)
    mean_key = _mean_key(metric)
    if len(rows) < N_REFS + N_MODS + 2:
        return None

    base = rows[rng.randrange(len(rows))]
    base_id, base_cfg, base_sum = base
    mods = _pick_mods(rows, base_id, base_cfg, metric, rng,
                      salient_min=1, salient_max=99)
    if len(mods) < N_MODS:
        return None
    exclude = {base_id} | {m[0] for m in mods}
    refs = _pick_references(rows, exclude, metric, rng)
    if len(refs) < N_REFS:
        return None

    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "propose_improvement",
        "problem_id": problem_id,
        "metric": metric,
        "references": refs,
        "base": {"candidate_id": base_id, "setting": base_cfg, "loss": float(base_sum[mean_key])},
        "improved_demos": [
            {"candidate_id": cid, "setting": cfg, "loss": float(s[mean_key])} for cid, cfg, s, _ in mods
        ],
    }
    body["question_id"] = f"pi_{short_hash(body)}"
    body["prompt"] = render_propose_improvement(body)
    return body


BUILDERS = {"select_best": build_select_best, "propose_improvement": build_propose_improvement}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", choices=sorted(BUILDERS), default="select_best")
    ap.add_argument("--items-per-problem", type=int, default=5)
    ap.add_argument("--problems", help="comma-separated problem ids; default all")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--set-name", default=None, help="override set folder name")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    builder = BUILDERS[args.type]
    problem_ids = args.problems.split(",") if args.problems else repo.list_problems()
    set_name = args.set_name or f"{args.type}_v1"
    out_dir = SETS_ROOT / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    skipped = []
    for pid in problem_ids:
        for _ in range(args.items_per_problem):
            item = builder(pid, rng)
            if item is None:
                continue
            items.append(item)
        if not any(i["problem_id"] == pid for i in items[-args.items_per_problem:]):
            skipped.append(pid)

    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "set_name": set_name,
        "type": args.type,
        "seed": args.seed,
        "items": len(items),
        "problems": len(problem_ids),
        "skipped_problems": skipped,
        "filters": {"win_rate_min": WIN_RATE_MIN, "ratio_min": "per-metric",
                    "ratio_min_mse": 1.15, "ratio_min_ce": 1.03,
                    "n_references": N_REFS, "n_mods": N_MODS,
                    "salient_min": 1, "salient_max": 2,
                    "pairwise_salient_min": 2 if args.type == "select_best" else 0},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(f"wrote {len(items)} items to {out_dir}/questions.jsonl")
    print(f"skipped problems (too few candidates / no valid question): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
