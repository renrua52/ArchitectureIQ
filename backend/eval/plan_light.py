"""L1 ``plan_light``: plan which unlit configs to light first (tree-framed).

Input  : a work tree with base + few lit children (losses) + N unlit configs.
Task   : (A) rank the unlit configs by predicted loss (ascending, best first);
         (B) name the ONE you would light first.
Scoring: (A) Spearman rho vs the tree's true ordering; (B) top-1 = node whose
         true loss is lowest (hit), else the improvement its true loss gives.

Usage:
    .venv/bin/python -m backend.eval.plan_light --build --items-per-tree 1 --seed 20260802
    .venv/bin/python -m backend.eval.plan_light --set plan_light_v1 --limit 5 --model gpt-5.6-luna
    .venv/bin/python -m backend.eval.plan_light --set plan_light_v1 --score
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx

from backend.eval import worktree
from backend.eval.autoresearch import build_initial_state, N_UNLIT_SHOWN
from backend.eval.batch_eval import call_llm, parse_answer, relay_config, resolve_api_key

SETS_ROOT = Path("backend/eval/sets")
N_UNLIT = 6          # unlit nodes per item (cap for prompt size)


def build_item(tree: dict, rng: random.Random) -> dict | None:
    lit, few = build_initial_state(tree, rng)
    unlit = [n for n in tree["nodes"] if n["candidate_id"] not in lit][:N_UNLIT]
    # keep at least 3 unlit: drop extra few-shot children from the lit set
    while len(unlit) < 3 and few:
        dropped = few.pop()
        lit.discard(dropped)
        unlit = [n for n in tree["nodes"] if n["candidate_id"] not in lit][:N_UNLIT]
    if len(unlit) < 3:
        return None
    rng.shuffle(unlit)
    letters = [chr(ord("A") + i) for i in range(len(unlit))]
    by_letter = {l: n["candidate_id"] for l, n in zip(letters, unlit)}
    item = {
        "schema_version": "0.1",
        "type": "plan_light",
        "problem_id": tree["problem_id"],
        "tree_id": tree["tree_id"],
        "metric": tree["metric"],
        "lit": [n for n in lit if n != tree["base"]],
        "options": by_letter,
        "true_loss": {n["candidate_id"]: n["loss"] for n in unlit},
    }
    # answer keys
    true_order = sorted(unlit, key=lambda n: n["loss"])
    item["correct_letter"] = next(
        l for l, cid in by_letter.items() if cid == true_order[0]["candidate_id"])
    item["prompt"] = render_plan_light(tree, lit, unlit, letters)
    return item


def render_plan_light(tree: dict, lit: set[str], unlit: list[dict], letters: list[str]) -> str:
    lines = [
        "You are an AutoResearch agent planning the next experiments for a toy ML task.",
        "Some configs in the work tree are lit (tested, loss known); some are unlit.",
        "Use the lit losses to calibrate your architecture intuition; plan from the tree.",
        "",
        f"[Dataset]  problem: {tree['problem_id']}  metric: {tree['metric']} (lower is better)",
        "",
        "[Lit configs (loss known)]",
    ]
    for n in tree["nodes"]:
        if n["candidate_id"] in lit:
            lines.append(worktree._model_line_wrap(tree, n, True))
    lines.append("")
    lines.append("[Unlit configs (NOT tested; ranked by your prediction)]")
    for letter, n in zip(letters, unlit):
        lines.append(f"{letter}. {worktree.setting_line(tree, n)}")
    lines.append("")
    lines.append("Output: (1) the letter of the config you would LIGHT FIRST; "
                 "(2) the full ranked order of all unlit letters by expected loss "
                 "(best/lowest loss first). Format: 'Light: X\\nRank: X > Y > Z'")
    return "\n".join(lines)


def parse_rank(text: str) -> tuple[str | None, list[str] | None]:
    """Parse 'Light: B' and 'Rank: C > A > B' (letters may include whitespace)."""
    light = None
    rank = None
    for line in (text or "").splitlines():
        low = line.lower()
        if "light" in low and ":" in line:
            m = __import__("re").search(r"light\s*[:=]\s*([A-Fa-f])", low)
            if m:
                light = m.group(1).upper()
        if "rank" in low and ":" in line:
            letters = __import__("re").findall(r"[A-Fa-f]", line.split(":", 1)[1])
            if letters:
                rank = [x.upper() for x in letters]
    if rank is None and text:
        import re
        letters = re.findall(r"\b([A-Fa-f])\b", text)
        if letters:
            rank = [x.upper() for x in letters]
    return light, rank


def _spearman(a: list[str], b: list[str]) -> float | None:
    import statistics
    if len(a) < 2 or len(b) < 2:
        return None
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    pa = {x: i for i, x in enumerate(a)}
    pb = {x: i for i, x in enumerate(b)}
    xs, ys = [], []
    for x in common:
        xs.append(pa[x])
        ys.append(pb[x])
    n = len(xs)
    dx = [x - statistics.mean(xs) for x in xs]
    dy = [y - statistics.mean(ys) for y in ys]
    denom = (sum(d * d for d in dx) * sum(d * d for d in dy)) ** 0.5
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def score_item(item: dict, light: str | None, rank: list[str] | None) -> dict:
    true_order = sorted(item["true_loss"], key=lambda cid: item["true_loss"][cid])
    letters = list(item["options"].keys())
    true_letters = [l for l in letters if item["options"][l] in true_order]
    res = {"correct_light": item["correct_letter"], "light": light,
           "light_hit": light == item["correct_letter"]}
    if rank:
        res["rank"] = rank
        res["spearman"] = _spearman(rank, true_letters)
    else:
        res["spearman"] = None
    return res


async def _run_set(set_name: str, model: str, limit: int, concurrency: int) -> int:
    items = [json.loads(l) for l in (SETS_ROOT / set_name / "questions.jsonl").open(encoding="utf-8")][:limit]
    key = resolve_api_key()
    headers = {"Authorization": f"Bearer {key}"}
    base_url = relay_config()["eval"]["base_url"].rstrip("/") + "/v1"
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        responses = await asyncio.gather(*[call_llm(client, sem, it["prompt"], model) for it in items])
    out = SETS_ROOT / set_name / f"answers_{model}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for it, (text, reasoning) in zip(items, responses):
            light, rank = parse_rank(text)
            sc = score_item(it, light, rank)
            f.write(json.dumps({"question_id": it["tree_id"], "problem_id": it["problem_id"],
                                **sc, "raw": text, "reasoning": reasoning}, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="build items into a set")
    ap.add_argument("--set-name", default="plan_light_v1")
    ap.add_argument("--items-per-tree", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.build:
        rng = random.Random(args.seed)
        out = SETS_ROOT / args.set_name
        out.mkdir(parents=True, exist_ok=True)
        items = []
        for pid in sorted({p.parent.name for p in (worktree.TREES_ROOT).glob("*/*.json")}):
            for tpath in sorted((worktree.TREES_ROOT / pid).glob("*.json")):
                tree = json.loads(tpath.read_text(encoding="utf-8"))
                for _ in range(args.items_per_tree):
                    it = build_item(tree, rng)
                    if it:
                        items.append(it)
        with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        (out / "manifest.json").write_text(json.dumps(
            {"type": "plan_light", "items": len(items), "seed": args.seed,
             "unlit_per_item": N_UNLIT}, indent=2), encoding="utf-8")
        print(f"wrote {len(items)} plan_light items to {out}")
        return 0

    if args.score:
        # aggregate answers files
        from collections import Counter
        hits, sps = [], []
        for f in (SETS_ROOT / args.set_name).glob("answers_*.jsonl"):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                if "light_hit" in r:
                    hits.append(r["light_hit"])
                if r.get("spearman") is not None:
                    sps.append(r["spearman"])
        print(f"light_hit: {sum(hits)}/{len(hits)} = {sum(hits)/max(1,len(hits)):.3f}")
        print(f"spearman: n={len(sps)} mean={sum(sps)/max(1,len(sps)):.3f}")
        return 0

    return asyncio.run(_run_set(args.set_name, args.model, args.limit, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
