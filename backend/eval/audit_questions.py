"""Post-hoc question audit with a strong LLM (claude-opus-5 via the eval relay).

For each sampled question we send the full prompt + the ground-truth losses
(winner/loser with seed std) and ask the model to (1) answer, (2) rate how
fairly decidable the question is, (3) name the likely failure mode, and
(4) suggest a fix. Used to characterise which questions are "easily bad".

Usage:
    .venv/bin/python -m backend.eval.audit_questions \
        --pack-xor /tmp/packs/benchmark_releases/question_packs/xor-v2.5-100q-37b9da \
        --pack-gru /tmp/packs/benchmark_releases/question_packs/gru-v2.5-100q-a48abc \
        --set select_best_v2 --set select_best_old60 \
        --per-source 3 --out artifacts/eval_runs/audit_opus_20260801.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import httpx

from architecture_iq.storage import repository as repo


def relay_creds() -> tuple[str, str]:
    conf = json.load(open(Path.home() / ".agents" / "relay.json"))
    eval_cfg = conf["eval"]
    return eval_cfg["base_url"].rstrip("/") + "/v1", eval_cfg["api_key"]


def load_pack_questions(pack_root: str, n: int, rng: random.Random,
                        prefer: str = "tight") -> list[dict]:
    import glob
    qs = []
    for qf in sorted(glob.glob(os.path.join(pack_root, "data/**/questions/*/q_*/question.json"),
                               recursive=True)):
        q = json.load(open(qf))
        prompt = Path(str(qf).replace("question.json", "prompt.txt"))
        q["_prompt"] = prompt.read_text(encoding="utf-8") if prompt.exists() else ""
        # GT losses
        means = {}
        for ch in q["choices"]:
            sp = os.path.join(pack_root, "data", ch["candidate_path"].replace("\\", "/"),
                              "results", "summary.json")
            s = json.load(open(sp))
            metric = q["significance"]["metric"]
            means[ch["letter"]] = (float(s[f"mean_{metric}"]), float(s.get(f"std_{metric}", 0.0)))
        q["_gt"] = means
        qs.append(q)
    def _ratio(q):
        vals = [q["_gt"][c["letter"]][0] for c in q["choices"]]
        return max(vals) / max(min(vals), 1e-12)
    qs.sort(key=lambda q: _ratio(q) if prefer == "tight" else -_ratio(q))
    return rng.sample(qs[: max(n * 2, 6)], min(n, len(qs)))


def load_set_questions(set_name: str, n: int, rng: random.Random) -> list[dict]:
    items = [json.loads(l) for l in open(f"backend/eval/sets/{set_name}/questions.jsonl")]
    rng.shuffle(items)
    out = []
    for it in items[:n]:
        it["_prompt"] = it.get("prompt", "")
        means = {}
        for op in it["options"]:
            s = repo.read_summary(it["problem_id"], op["candidate_id"])
            means[op["letter"]] = (float(s[f"mean_{it['metric']}"]), float(s.get(f"std_{it['metric']}", 0.0)))
        it["_gt"] = means
        it["_set"] = set_name
        out.append(it)
    return out


def audit_prompt(question: dict) -> str:
    gt = question["_gt"]
    gt_lines = "\n".join(f"  {l}: mean loss = {v[0]:.4f} (std {v[1]:.4f})" for l, v in sorted(gt.items()))
    correct = question.get("correct_letter", question.get("answer"))
    return f"""{question["_prompt"]}

===== GROUND TRUTH (for your post-hoc audit only; do not treat as part of the question) =====
GT losses (lower is better):
{gt_lines}
Stored correct answer: {correct}

===== AUDIT TASKS =====
1) Which choice would you pick from the question alone (letter)?
2) Rate how fairly decidable this question is by reasoning alone, 1-5
   (5 = a strong reasoner can decide it reliably; 1 = pure noise / prior-only).
3) Name the dominant failure mode, choosing from:
   [param_prior] bigger model / more params reveals the answer
   [gap_too_small] winner-vs-runner gap is within seed noise
   [metric_compressed] metric scale (e.g. CE ~3.2) makes gaps unreadable
   [references_misleading] the reference losses point away from the winner
   [type_prior] a model family (KAN/GRU/etc.) wins systematically
   [ok] no obvious flaw
4) One-sentence concrete fix.

Output as JSON with keys: answer, decidability, failure_mode, fix.
"""


async def call_once(client: httpx.AsyncClient, prompt: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "high",
    }
    r = await client.post("/chat/completions", json=payload, timeout=180)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    msg = r.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {"content": content, "reasoning": reasoning}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-xor", default=None)
    ap.add_argument("--pack-gru", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--prefer", default="tight", choices=["tight", "loose"])
    ap.add_argument("--out", default="artifacts/eval_runs/audit_opus_20260801.jsonl")
    ap.add_argument("--model", default="claude-opus-5")
    args = ap.parse_args()

    rng = random.Random(20260801)
    questions = []
    if args.pack_xor:
        for q in load_pack_questions(args.pack_xor, args.per_source, rng, args.prefer):
            q["_source"] = "pack_xor"; questions.append(q)
    if args.pack_gru:
        for q in load_pack_questions(args.pack_gru, args.per_source, rng, args.prefer):
            q["_source"] = "pack_gru"; questions.append(q)
    for sn in args.set:
        for q in load_set_questions(sn, args.per_source, rng):
            q["_source"] = f"set:{sn}"; questions.append(q)
    print(f"auditing {len(questions)} questions", file=sys.stderr)

    base_url, key = relay_creds()
    headers = {"Authorization": f"Bearer {key}"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=200) as client:
        for q in questions:
            resp = await call_once(client, audit_prompt(q), args.model)
            rec = {
                "source": q.get("_source"),
                "question_id": q.get("question_id"),
                "model": args.model,
                "correct_letter": q.get("correct_letter", q.get("answer")),
                "gt": q["_gt"],
                "audit": resp,
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            print(f"  [{n_done}/{len(questions)}] {q.get('_source')} {q.get('question_id')} "
                  f"{'OK' if 'content' in resp else resp.get('error', '')}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
