"""Batch concurrent propose_improvement: LLM proposes configs, then GT scores them.

Phase 1 (propose): send each ``propose_improvement`` question prompt to the
relay API concurrently, parse the JSON config, normalize/validate against the
closed set (reuses ``score_proposal.normalize_proposal``), and save proposals.

Phase 2 (score): run GT for every saved proposal (``write_candidate`` +
``run_ground_truth``) across a process pool and compare per-seed with the base
candidate (reuses ``score_proposal.score_question``).

Usage:
    .venv/bin/python -m backend.eval.batch_propose --set propose_improvement_v1.1 \
        --limit 50 --concurrency 16 --propose-only
    .venv/bin/python -m backend.eval.batch_propose --set propose_improvement_v1.1 \
        --limit 50 --score-only --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import httpx

from backend.eval import score_proposal
from backend.eval.batch_eval import call_llm, parse_answer, resolve_api_key, DEFAULT_MODEL  # reuse plumbing

SETS_ROOT = Path("backend/eval/sets")
OUT_ROOT = Path("artifacts/eval_runs")
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"  # deepseek-v4-flash (local key)


def load_questions(set_name: str) -> list[dict]:
    path = SETS_ROOT / set_name / "questions.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from a model response (handles markdown fences)."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        start, end = text.index("{"), text.rindex("}")
        return json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


async def run_proposals(items: list[dict], model: str, concurrency: int,
                        base_url: str) -> list[dict]:
    key = resolve_api_key()
    headers = {"Authorization": f"Bearer {key}"}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        tasks = [call_llm(client, sem, it["prompt"], model) for it in items]
        texts = await asyncio.gather(*tasks)

    out = []
    for it, text in zip(items, texts):
        prop = extract_json(text)
        if prop is None:
            out.append({"ok": False, "question_id": it["question_id"],
                        "problem_id": it["problem_id"], "raw": text,
                        "errors": ["json parse failed"]})
            continue
        prop, disp_notes = score_proposal.normalize_proposal_display(prop)
        spec, notes, errors = score_proposal.normalize_proposal(it["base"]["setting"], prop)
        notes = disp_notes + notes
        errors += score_proposal._validate_loss(it, spec)
        if errors:
            out.append({"ok": False, "question_id": it["question_id"],
                        "problem_id": it["problem_id"], "errors": errors,
                        "notes": notes, "raw": text})
            continue
        out.append({"ok": True, "question_id": it["question_id"],
                    "problem_id": it["problem_id"], "proposal": prop,
                    "snapped": spec, "notes": notes, "raw": text})
    return out


def _score_one(q: dict) -> dict:
    """Score one proposal with GT (runs in a worker process)."""
    try:
        prop, disp_notes = score_proposal.normalize_proposal_display(q["_proposal_raw"])
        return score_proposal.score_question(q["_question"], prop)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {
            "ok": False,
            "question_id": q["_question"]["question_id"],
            "problem_id": q["_question"]["problem_id"],
            "errors": [f"gt crashed: {type(e).__name__}: {e}"],
            "traceback": traceback.format_exc()[-1500:],
        }


def score_with_gt(proposals: list[dict], workers: int,
                  score_path: Path | None = None) -> list[dict]:
    """proposals: list of {'ok': True, 'question_id', ..., 'snapped': spec}.

    Writes results incrementally to ``score_path`` so partial progress survives
    worker crashes.
    """
    scored = []
    jobs = []
    for p in proposals:
        if not p.get("ok"):
            scored.append({"question_id": p["question_id"], "ok": False,
                           "errors": p.get("errors", ["invalid proposal"])})
            continue
        q = _load_question(p["question_id"])
        # re-normalize from the raw proposal so normalization fixes apply
        # to already-stored proposal files
        jobs.append({"_question": q, "_proposal_raw": p["proposal"]})
    if not jobs:
        return scored
    f = None
    if score_path is not None:
        score_path.parent.mkdir(parents=True, exist_ok=True)
        f = score_path.open("a", encoding="utf-8")  # append: resume-safe
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_score_one, j) for j in jobs]
            for fut in as_completed(futs):
                res = fut.result()
                scored.append(res)
                if f is not None:
                    slim = {k: res.get(k) for k in (
                        "ok", "question_id", "problem_id", "metric", "base_loss",
                        "proposal_loss", "ratio_vs_base", "win_rate_vs_base",
                        "errors", "notes", "param_constraint_ok", "params",
                        "traceback")}
                    f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    f.flush()
    finally:
        if f is not None:
            f.close()
    return scored


def _load_question(question_id: str) -> dict:
    for set_dir in SETS_ROOT.iterdir():
        path = set_dir / "questions.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            q = json.loads(line)
            if q["question_id"] == question_id:
                return q
    raise KeyError(question_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="propose_improvement_v1.1")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--propose-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT_ROOT / f"{args.set.replace('/', '_')}_proposals.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        proposals = [json.loads(l) for l in out_path.open(encoding="utf-8")]
    else:
        items = load_questions(args.set)[: args.limit]
        print(f"proposing {len(items)} configs with {args.model} @ {args.concurrency} ...")
        proposals = asyncio.run(run_proposals(items, args.model, args.concurrency, args.base_url))
        n_ok = sum(1 for p in proposals if p.get("ok"))
        print(f"proposals ok: {n_ok}/{len(proposals)}")
        with out_path.open("w", encoding="utf-8") as f:
            for p in proposals:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote proposals to {out_path}")
        if args.propose_only:
            return 0

    score_path = out_path.with_name(out_path.stem + "_scored.jsonl")
    if score_path.exists():
        done = {json.loads(l)["question_id"] for l in score_path.open(encoding="utf-8")
                if json.loads(l).get("ok")}
        proposals = [p for p in proposals if p["question_id"] not in done]
        print(f"resume: {len(done)} already ok, {len(proposals)} remain (failed re-run)")
    print(f"scoring {sum(1 for p in proposals if p.get('ok'))} proposals with GT "
          f"({args.workers} workers)...")
    scored = score_with_gt(proposals, args.workers, score_path)
    n_ok = sum(1 for s in scored if s.get("ok"))
    beats = [s for s in scored if s.get("ok") and s["proposal_loss"] < s["base_loss"]]
    import statistics
    ratios = [s["ratio_vs_base"] for s in scored if s.get("ok")]
    print(f"GT scored: {n_ok}/{len(scored)} ok; {len(beats)} beat base "
          f"({len(beats)/max(1,n_ok):.0%}); median ratio_vs_base="
          f"{statistics.median(ratios) if ratios else '-'}")

    summary = {
        "set": args.set, "model": args.model, "total": len(scored),
        "gt_ok": n_ok, "beat_base": len(beats),
        "win_rate_vs_base_ge_0.7": sum(1 for s in beats if s.get("win_rate_vs_base", 0) >= 0.7),
        "median_ratio": round(statistics.median(ratios), 4) if ratios else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
