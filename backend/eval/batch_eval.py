"""Batch concurrent LLM evaluation of eval-set questions via the relay API.

Reads a 题集 (``backend/eval/sets/{set}/questions.jsonl``) or a directory of
two-choice items, sends each prompt to the configured LLM relay concurrently
(``OPENAI_API_KEY`` + ``OPENAI_BASE_URL`` or ``--base-url``/``--key``), parses
the letter answer, scores against ground truth, and writes per-question results
plus a summary with family / ratio-stratum breakdown.

Usage:
    .venv/bin/python -m backend.eval.batch_eval --set select_best_v1.2 --limit 50
    .venv/bin/python -m backend.eval.batch_eval --two-choice-dir artifacts/eval_probe/items --limit 50
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openai.phybench.cn/v1 \\
        .venv/bin/python -m backend.eval.batch_eval --set select_best_v1.2 --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from architecture_iq.storage import repository as repo

# Default provider: local DeepSeek key (deepseek-v4-flash). The phybench relay
# (openai.phybench.cn, gpt-5.6-terra) is used only when explicitly requested via
# --base-url / --model or OPENAI_BASE_URL.
RELAY_FILE = Path.home() / ".agents" / "relay.json"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # deepseek-v4-flash
DEEPSEEK_KEY_FILES = (
    Path.home() / ".codex-deepseek/.deepseek_api_key",
    Path.home() / ".codex/.deepseek_api_key",
)
SETS_ROOT = Path("backend/eval/sets")
OUT_ROOT = Path("artifacts/eval_runs")
LETTER_RE = re.compile(r"\b([A-F])\b")


def relay_config() -> dict | None:
    """Read the relay key file (eval credentials + model names); None if missing."""
    try:
        return json.loads(RELAY_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def default_base_url() -> str:
    d = relay_config()
    if d and d.get("eval", {}).get("base_url"):
        return d["eval"]["base_url"].rstrip("/") + "/v1"
    return DEEPSEEK_BASE_URL


def default_model() -> str:
    d = relay_config()
    if d and d.get("models", {}).get("debug"):
        return d["models"]["debug"][0]
    return DEEPSEEK_MODEL


def resolve_api_key() -> str:
    d = relay_config()
    if d and d.get("eval", {}).get("api_key"):
        return d["eval"]["api_key"]
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    for p in DEEPSEEK_KEY_FILES:
        if p.exists():
            v = p.read_text().strip()
            if v:
                return v
    raise SystemExit("no API key: set DEEPSEEK_API_KEY/OPENAI_API_KEY or a key file")


def parse_answer(text: str) -> str | None:
    """Extract the answer letter from a model response (normalized to upper).

    Handles bare letters, bolded final answers (``Answer: **B**``), trailing
    standalone letters, and final-line answers. The LAST conclusive letter wins
    (final answer), never the first letter mentioned in reasoning.
    """
    if not text:
        return None
    t = text.strip()
    # exact letter (possibly lowercase)
    if re.fullmatch(r"[A-Fa-f]", t):
        return t.upper()
    # standalone letter alone on the final line: "B", "B.", "B)", "**B**"
    for line in reversed(t.splitlines()):
        m = re.fullmatch(r"\s*[\*# ]*([A-Fa-f])[\*#.):\s]*\s*", line)
        if m:
            return m.group(1).upper()
    # "Answer: B" / "letter = b" / "Answer: **B**" / "answer is B" /
    # "pick B" — prefer the LAST occurrence (the final answer), not a mention
    # inside the reasoning.
    for pat in (
        r"(?i)(?:answer|letter|option|choice)\s*[:=\-]?\s*[\*# ]*\(?\s*([A-Fa-f])\s*\)?\s*[\*#.]*",
        r"(?i)\banswer\s+is\s+[\*# ]*([A-Fa-f])[\*#.]*",
        r"(?i)\b(?:pick|choose|select)\s+[\*# ]*([A-Fa-f])[\*#.]*",
    ):
        matches = list(re.finditer(pat, t))
        if matches:
            return matches[-1].group(1).upper()
    # fallback: last standalone letter anywhere in the text
    matches = list(re.finditer(r"\b([A-Fa-f])\b", t))
    return matches[-1].group(1).upper() if matches else None

def load_items(set_name: str | None, two_choice_dir: str | None) -> list[dict]:
    if set_name:
        path = SETS_ROOT / set_name / "questions.jsonl"
        items = [json.loads(line) for line in path.open(encoding="utf-8")]
        return items
    if two_choice_dir:
        from backend.eval.two_choice import render_prompt as render_tc_prompt
        items = []
        for f in sorted(Path(two_choice_dir).glob("*.json")):
            item = json.loads(f.read_text(encoding="utf-8"))
            item["prompt"] = render_tc_prompt(item)
            item["question_id"] = f.stem
            items.append(item)
        return items
    raise SystemExit("need --set or --two-choice-dir")


async def call_llm(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    prompt: str,
    model: str,
    max_retries: int = 4,
    timeout: float = 180.0,
) -> str | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "high",  # AGENTS.md §10: default high, no token caps
        "temperature": 0.0,
    }
    backoff = [1, 2, 4, 8, 16, 30]
    for attempt in range(max_retries):
        try:
            async with sem:
                r = await client.post("/chat/completions", json=payload, timeout=timeout)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                # Some relay models (Kimi-K3, qwen3.7-max) put the answer in
                # content or reasoning_content depending on reasoning mode.
                content = msg.get("content") or ""
                if not content.strip():
                    content = msg.get("reasoning_content") or ""
                if content.strip():
                    return content
                # empty 200: relay hiccup — retry (loop continues)
            if r.status_code in (429, 500, 502, 503, 529):
                wait = backoff[min(attempt, len(backoff) - 1)]
                await asyncio.sleep(wait)
                continue
            print(f"  !! HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return None
        except httpx.TimeoutException:
            await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
        except Exception as e:  # noqa: BLE001
            print(f"  !! error: {e}", file=sys.stderr)
            await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
    return None


def family_of(problem_id: str) -> str:
    try:
        return repo.read_problem_spec(problem_id).get("family", "?")
    except FileNotFoundError:
        return "?"


async def run_batch(items: list[dict], model: str, concurrency: int, base_url: str) -> list[dict]:
    key = resolve_api_key()
    headers = {"Authorization": f"Bearer {key}"}
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        tasks = [call_llm(client, sem, it["prompt"], model) for it in items]
        responses = await asyncio.gather(*tasks)
    for it, text in zip(items, responses):
        answer = parse_answer(text)
        correct_key = "correct_letter" if "correct_letter" in it else "answer"
        correct = it.get(correct_key)
        results.append({
            "_set": it.get("_set"),
            "question_id": it.get("question_id", it.get("task")),
            "model": model,
            "problem_id": it.get("problem_id"),
            "family": family_of(it.get("problem_id", "")),
            "type": it.get("type", it.get("task", "two_choice")),
            "metric": it.get("metric"),
            "ratio": it.get("statistics", {}).get("ratio"),
            "correct": correct,
            "answer": answer,
            "is_correct": answer is not None and answer == correct,
            "raw_response": text,
        })
    return results


def summarize(results: list[dict], label: str) -> dict:
    n = len(results)
    ok = [r for r in results if r.get("correct")]
    scored = [r for r in ok if r["answer"] is not None]
    acc = sum(1 for r in scored if r["is_correct"]) / len(scored) if scored else 0.0
    parsed = len(scored) / n if n else 0.0
    stratum_tot: Counter = Counter()
    stratum_ok: Counter = Counter()
    family_tot: Counter = Counter()
    family_ok: Counter = Counter()
    for r in scored:
        ratio = r.get("ratio") or 0
        if ratio < 1.15:
            name = "tight(<1.15)"
        elif ratio < 2.0:
            name = "medium(1.15-2)"
        else:
            name = "loose(>=2)"
        stratum_tot[name] += 1
        stratum_ok[name] += r["is_correct"]
        family_tot[r["family"]] += 1
        family_ok[r["family"]] += r["is_correct"]
    first_type = results[0].get("type") if results else ""
    n_choices = 2 if "two_choice" in first_type else 6
    out = {
        "label": label,
        "total": n,
        "scored": len(scored),
        "unparsed": n - len(scored),
        "accuracy": round(acc, 4),
        "random_baseline": round(1.0 / n_choices, 4),
        "by_stratum": {
            k: {"acc": round(stratum_ok[k] / stratum_tot[k], 4), "n": stratum_tot[k]}
            for k in stratum_tot
        },
        "by_family": {
            k: {"acc": round(family_ok[k] / family_tot[k], 4), "n": family_tot[k]}
            for k in family_tot
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default=None)
    ap.add_argument("--sets", default=None, help="comma-separated set names: mixed "
                    "concurrency with per-set results (e.g. select_best_v1.2,select_best_v1.1)")
    ap.add_argument("--two-choice-dir", default=None)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", default_base_url()),
                    help="default DeepSeek api.deepseek.com; phybench relay only via --base-url")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reason-suffix", action="store_true",
                    help="append a step-by-step reasoning instruction before the answer "
                         "request (avoids first-option decode collapse on short-answer formats)")
    args = ap.parse_args()

    if args.sets:
        set_names = [s.strip() for s in args.sets.split(",")]
        items = []
        for sn in set_names:
            its = load_items(sn, None)
            if len(its) > args.limit:
                its = its[: args.limit]
            for it in its:
                it["_set"] = sn
            items.extend(its)
        label = args.label or "+".join(set_names)
    else:
        items = load_items(args.set, args.two_choice_dir)
        if len(items) > args.limit:
            items = items[: args.limit]
        for it in items:
            it["_set"] = args.set or Path(args.two_choice_dir).name
        label = args.label or args.set or Path(args.two_choice_dir).name
    if args.reason_suffix:
        suffix = ("\n\nFirst reason step by step about how the reference losses "
                  "calibrate each option, then answer with the letter only.")
        for it in items:
            it["prompt"] = (it.get("prompt") or "") + suffix

    out = Path(args.out) if args.out else OUT_ROOT / f"{label.replace('/', '_')}.jsonl"
    done_ids = set()
    if out.exists() and args.resume:
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("answer") is not None:
                done_ids.add(r.get("question_id") or r.get("q"))
        items = [it for it in items
                 if (it.get("question_id") or it.get("q")) not in done_ids]
        print(f"resume: skipping {len(done_ids)} done, {len(items)} remain")
    print(f"evaluating {len(items)} items ({label}) with {args.model} "
          f"@ {args.concurrency} concurrency ...")

    t0 = time.time()
    results = asyncio.run(run_batch(items, args.model, args.concurrency, args.base_url))
    elapsed = time.time() - t0

    summary = summarize(results, label)
    summary["model"] = args.model
    if out.exists() and args.resume:
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append(r)
        summary = summarize(results, label)
        summary["model"] = args.model
    summary["concurrency"] = args.concurrency
    summary["elapsed_s"] = round(elapsed, 1)
    by_set: dict[str, list[dict]] = {}
    for r in results:
        by_set.setdefault(r.get("_set", "?"), []).append(r)
    if len(by_set) > 1:
        summary["per_set"] = {sn: summarize(rows, sn) for sn, rows in sorted(by_set.items())}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
