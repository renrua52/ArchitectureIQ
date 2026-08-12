"""Work-tree generation for the AutoResearch benchmark.

Design: docs/plan-autoresearch-eval.md
A work tree = a **good base config** + children (1-2 salient edits), drawn from
existing candidates with stored GT, plus lit/unlit status. Trees are eval-side
views over the columnar backend (problems/candidates/results); they live under
``backend/eval/trees/{problem_id}/{tree_id}.json`` and are gitignored with the
rest of ``backend/eval/trees/`` (generated artifacts).

The tree is the single source of truth for the AutoResearch loop: lit = result
revealed to the model, unlit = hidden. The oracle is the best stored loss
among all tree nodes (all nodes come with GT, so lighting is free; only
genuinely new proposed configs cost compute).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from architecture_iq.storage import repository as repo
from architecture_iq.util import short_hash

from backend.eval.questions import (
    config_edit_distance,
    salient_distance,
    setting_summary,
)

TREES_ROOT = Path("backend/eval/trees")
BASE_QUALITY_QUANTILE = (0.30, 0.65)  # base is a good-but-not-best setting
N_CHILDREN = 8


def _mean_key(metric: str) -> str:
    return f"mean_{metric}"


def load_rows(problem_id: str) -> tuple[str, list[tuple[str, dict, dict]]]:
    """(metric, rows) with rows=[(candidate_id, config, summary)] sorted by loss asc.

    Only candidates with a valid stored result and no exclusion flag count.
    """
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
        if s.get("excluded") or mean_key not in s or float(s[mean_key]) <= 0:
            continue
        rows.append((cid, cfg, s))
    rows.sort(key=lambda r: float(r[2][mean_key]))
    return metric, rows


def pick_base(rows: list[tuple], rng: random.Random,
              quantile: tuple[float, float] = BASE_QUALITY_QUANTILE) -> tuple | None:
    """Pick a base in the ``quantile`` band of sorted losses (good but improvable).

    Returns a row tuple or None if the band is empty.
    """
    lo, hi = quantile
    n = len(rows)
    i0 = max(0, min(n - 1, int(lo * n)))
    i1 = max(i0, min(n - 1, int(hi * n)))
    if i1 < i0 or i1 >= n:
        return None
    return rows[rng.randrange(i0, i1 + 1)]


def edit_labels(base_cfg: dict, child_cfg: dict) -> list[str]:
    """Human-readable list of changed fields, e.g. ``model.depth 3->4``."""
    out: list[str] = []

    def walk(a, b, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in ("candidate_id", "problem_id", "schema_version", "files"):
                    continue
                walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        else:
            if a != b:
                out.append(f"{prefix} {a!r}->{b!r}")

    walk(base_cfg, child_cfg, "")
    return out


def budget_ok(base_cfg: dict, cfg: dict) -> bool:
    """Children must respect the AutoResearch budget rule relative to the base:
    same total_samples_seen (clean comparisons) and params <= 1.1x base params.
    """
    from backend.eval.score_proposal import estimate_params, MAX_PARAM_RATIO
    try:
        base_budget = int(base_cfg["budget"]["total_samples_seen"])
        child_budget = int(cfg["budget"]["total_samples_seen"])
        base_params = estimate_params(base_cfg["model"])
        child_params = estimate_params(cfg["model"])
    except (KeyError, ValueError):
        return False
    return child_budget == base_budget and child_params <= MAX_PARAM_RATIO * base_params


def pick_children(base_cfg: dict, rows: list[tuple], rng: random.Random, *,
                  n: int = N_CHILDREN, salient_min: int = 1, salient_max: int = 2,
                  exclude: set[str] | None = None) -> list[tuple]:
    """Nearest children: 1-2 salient edits, fewest total edits first.

    Children must respect the budget rule (same training budget, params
    <= 1.1x base) so every tree node is a legal AutoResearch move.
    """
    exclude = exclude or set()
    scored = []
    for cid, cfg, s in rows:
        if cid in exclude:
            continue
        if not budget_ok(base_cfg, cfg):
            continue
        d = config_edit_distance(base_cfg, cfg)
        sd = salient_distance(base_cfg, cfg)
        if d < 1 or not (salient_min <= sd <= salient_max):
            continue
        scored.append((d, sd, cid, cfg, s))
    scored.sort(key=lambda x: (x[0], x[1]))
    picked: list[tuple] = []
    for d, sd, cid, cfg, s in scored:
        if all(config_edit_distance(cfg, pc) >= 1 for _, _, _, pc, _ in picked):
            picked.append((d, sd, cid, cfg, s))
        if len(picked) >= n:
            break
    return picked


def build_tree(problem_id: str, rng: random.Random, *,
               n_children: int = N_CHILDREN,
               base_quantile: tuple[float, float] = BASE_QUALITY_QUANTILE,
               max_tries: int = 12) -> dict | None:
    metric, rows = load_rows(problem_id)
    if len(rows) < n_children + 4:
        return None
    mean_key = _mean_key(metric)
    children: list[tuple] = []
    base = None
    for _ in range(max_tries):
        base = pick_base(rows, rng, base_quantile)
        if base is None:
            return None
        base_id, base_cfg, _ = base
        # prefer 1-2 salient edits (comparable); relax when the problem is sparse
        for sd_max in (2, 3, 4):
            children = pick_children(base_cfg, rows, rng, n=n_children,
                                     salient_max=sd_max, exclude={base_id})
            if len(children) >= 3:
                break
        if len(children) >= 3:
            break
    if base is None or len(children) < 3:
        return None
    base_id, base_cfg, base_sum = base
    nodes = [{
        "candidate_id": base_id,
        "role": "base",
        "edits": [],
        "loss": float(base_sum[mean_key]),
        "params": _params_of(base_cfg),
    }]
    for d, sd, cid, cfg, s in children:
        nodes.append({
            "candidate_id": cid,
            "role": "child",
            "edits": edit_labels(base_cfg, cfg),
            "loss": float(s[mean_key]),
            "params": _params_of(cfg),
        })
    tree = {
        "schema_version": "0.1",
        "problem_id": problem_id,
        "tree_id": f"tree_{short_hash(json.dumps({problem_id: nodes}, sort_keys=True))[:8]}",
        "metric": metric,
        "base": base_id,
        "base_loss": float(base_sum[mean_key]),
        "nodes": nodes,
        "budget_rule": {
            "params_ratio_max": 1.1,
            "total_samples_seen": int(base_cfg["budget"]["total_samples_seen"]),
        },
    }
    tree["oracle"] = min(float(s["loss"]) for s in nodes)
    return tree


def _params_of(cfg: dict) -> float:
    from backend.eval.score_proposal import estimate_params
    try:
        return estimate_params(cfg["model"])
    except (KeyError, ValueError):
        return 0.0


def tree_path(problem_id: str, tree_id: str) -> Path:
    return TREES_ROOT / problem_id / f"{tree_id}.json"


def save_tree(tree: dict) -> Path:
    p = tree_path(tree["problem_id"], tree["tree_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_tree(problem_id: str, tree_id: str) -> dict:
    return json.loads(tree_path(problem_id, tree_id).read_text(encoding="utf-8"))


def list_trees(problem_id: str | None = None) -> list[str]:
    root = TREES_ROOT / problem_id if problem_id else TREES_ROOT
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(TREES_ROOT))[:-5] for p in root.glob("*/*.json"))


def node_loss(tree: dict, candidate_id: str) -> float | None:
    for n in tree["nodes"]:
        if n["candidate_id"] == candidate_id:
            return float(n["loss"])
    return None


def _model_line_wrap(tree: dict, n: dict, with_loss: bool) -> str:
    cfg = repo.read_candidate_config(tree["problem_id"], n["candidate_id"])
    label = "BASE" if n.get("role") == "base" else f"node {n['candidate_id']}"
    line = f"- {label}: {setting_line(tree, n)}"
    if with_loss and n.get("loss") is not None:
        line += f"  => loss = {n['loss']:.4g}"
    return line


def tree_view(tree: dict, lit_ids: set[str]) -> dict:
    """Split tree nodes into lit (loss shown) and unlit (hidden) for a prompt."""
    lit, unlit = [], []
    for n in tree["nodes"]:
        entry = {"candidate_id": n["candidate_id"], "role": n.get("role", "child"),
                 "edits": n.get("edits", [])}
        if n["candidate_id"] in lit_ids:
            entry["loss"] = n["loss"]
            lit.append(entry)
        else:
            unlit.append(entry)
    return {"lit": lit, "unlit": unlit}


def setting_line(tree: dict, n: dict) -> str:
    cfg = repo.read_candidate_config(tree["problem_id"], n["candidate_id"])
    label = n.get("role", "child")
    parts = [f"[{label} {n['candidate_id']}]", setting_summary(cfg)]
    if n.get("edits"):
        parts.append("edits: " + "; ".join(n["edits"]))
    return " | ".join(parts)


def pick_few_shot(tree: dict, lit_ids: set[str], rng: random.Random,
                  k: int = 3) -> list[str]:
    """Choose 2-3 lit children spread across the loss range (decision-tree few-shot)."""
    lit_children = [n for n in tree["nodes"]
                    if n["candidate_id"] in lit_ids and n.get("role") != "base"]
    if not lit_children:
        return []
    lit_children.sort(key=lambda n: n["loss"])
    n = len(lit_children)
    idxs = sorted(set(int(i) for i in (0, (n - 1) / 2, n - 1)[:k]))
    picks = [lit_children[i]["candidate_id"] for i in idxs]
    # pad randomly if fewer distinct picks than k
    rng.shuffle(lit_children)
    for n in lit_children:
        if len(picks) >= k:
            break
        if n["candidate_id"] not in picks:
            picks.append(n["candidate_id"])
    return picks


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build work trees for problems")
    ap.add_argument("--problems", help="comma-separated problem ids; default all")
    ap.add_argument("--trees-per-problem", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260802)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pids = args.problems.split(",") if args.problems else repo.list_problems()
    total = 0
    for pid in pids:
        made = 0
        for _ in range(args.trees_per_problem):
            t = build_tree(pid, rng)
            if t is None:
                continue
            save_tree(t)
            made += 1
        total += made
        print(f"{pid}: {made} trees")
    print(f"total trees: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
