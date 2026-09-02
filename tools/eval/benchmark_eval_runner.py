#!/usr/bin/env python3
"""Async multi-model evaluation runner for ArchitectureIQ question sets.

Design goals:
- resumable: per-(model, question) result files + ledger; re-running only
  fills gaps;
- per-model concurrency limits, models run in parallel as asyncio tasks;
- pluggable backends via plain OpenAI-compatible HTTP; any endpoint
  (openrouter, vapi-style relays, direct APIs) works as long as it speaks
  POST {base}/chat/completions.

Usage (from the machine hosting the question set):

    python3 tools/eval/benchmark_eval_runner.py \
        --questions-root data/datasets \
        --out-dir data/evals/v15 \
        --backends-config data/evals/v15_backends.json \
        --models gpt-5.6-sol,claude-opus-5,kimi-k3 \
        --concurrency 8 --max-tokens 16384

The backends config maps provider names to {base_url, api_key, models}. Keys
live inline in this file; it sits under gitignored `data/` so it is never
committed. Designated location: `data/evals/eval_keys.json`.

    {
      "vapi": {
        "base_url": "https://api.gpt.ge/v1",
        "api_key": "sk-...",
        "models": ["gpt-5.6-sol", "claude-opus-5", "kimi-k3"]
      },
      "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-v1-...",
        "models": []
      }
    }

`api_key_env` is still honored as a fallback (key from the environment) for
CI or machines that prefer not to store keys on disk.
`--models` accepts names or `provider:name` to disambiguate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request


# --------------------------------------------------------------------------
# Question loading (self-contained; no main-package imports so this file can
# be rsync'd to the server and run with any python>=3.10)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionItem:
    question_dir: Path
    question_id: str
    question: dict[str, Any]
    prompt_text: str
    prompt_hash: str
    valid_letters: frozenset[str]

    @property
    def correct_letter(self) -> str:
        return str(self.question["correct_letter"]).upper()


def load_question(question_dir: Path) -> QuestionItem:
    q = json.loads((question_dir / "question.json").read_text(encoding="utf-8"))
    prompt_rel = q.get("prompt", {}).get("rendered_path", "prompt.txt")
    prompt_path = question_dir / prompt_rel
    prompt_text = prompt_path.read_text(encoding="utf-8")
    letters = frozenset(str(c["letter"]).upper() for c in q["choices"])
    return QuestionItem(
        question_dir=question_dir,
        question_id=q.get("question_id", question_dir.name),
        question=q,
        prompt_text=prompt_text,
        prompt_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        valid_letters=letters,
    )


def list_questions(questions_root: Path, ledger_path: Path | None) -> list[QuestionItem]:
    """Prefer the build ledger (exact 500-item order); fall back to rglob."""
    items: list[QuestionItem] = []
    if ledger_path and ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            qdir = Path(rec["question_path"])
            if (qdir / "question.json").is_file():
                item = load_question(qdir)
                item.question.setdefault("item_id", rec.get("item_id"))
                items.append(item)
    else:
        for qfile in sorted(questions_root.rglob("questions/*/*/question.json")):
            items.append(load_question(qfile.parent))
    return items


# --------------------------------------------------------------------------
# Response parsing (mirrors tools/llm_eval/response_parser.py semantics)
# --------------------------------------------------------------------------

import re

_ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
_EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.IGNORECASE | re.DOTALL)


def parse_choice_letter(response: str, valid_letters: frozenset[str]) -> str | None:
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return None
    letter = matches[-1].upper()
    return letter if letter in valid_letters else None


def split_chain_of_thought(response: str, parsed: str | None) -> str | None:
    if parsed is None:
        return response
    matches = list(_ANSWER_RE.finditer(response))
    if not matches:
        return None
    last = matches[-1]
    return (response[: last.start()] + response[last.end() :]).strip() or None


# --------------------------------------------------------------------------
# HTTP client (stdlib only; async via asyncio.to_thread)
# --------------------------------------------------------------------------

class BackendError(RuntimeError):
    pass


def http_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    extra_headers: dict[str, str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if params:
        payload.update(params)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **extra_headers,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt >= max_retries:
                raise BackendError(f"HTTP {exc.code}: {detail}") from exc
            if exc.code == 524:
                # relay upstream congestion: each attempt already burned
                # minutes server-side; wait minutes, not seconds.
                time.sleep(min(300.0, 60.0 * (attempt + 1)))
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise BackendError(f"request failed: {exc}") from exc
        time.sleep(min(30.0, 1.5 * (2**attempt) + random.random()))
    raise BackendError("unreachable")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str
    base_url: str
    api_key: str
    concurrency: int
    extra_headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.name)


def result_path(out_dir: Path, spec: ModelSpec, question_id: str) -> Path:
    return out_dir / "results" / spec.slug / f"{question_id}.json"


def ledger_path(out_dir: Path, spec: ModelSpec) -> Path:
    return out_dir / "ledger" / f"{spec.slug}.jsonl"


def load_done(out_dir: Path, spec: ModelSpec) -> set[str]:
    """Question ids already completed successfully for this model.

    Authoritative source = per-question result files (one record each); the
    ledger is an append-only log for auditing, not the resume key.
    """
    done: set[str] = set()
    rdir = out_dir / "results" / spec.slug
    if rdir.is_dir():
        for path in rdir.glob("*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if rec.get("parsed_letter") is not None and rec.get("ok"):
                done.add(rec["question_id"])
    return done


async def eval_one(
    item: QuestionItem,
    spec: ModelSpec,
    out_dir: Path,
    sem: asyncio.Semaphore,
    *,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    max_exchanges: int,
) -> None:
    async with sem:
        t0 = time.time()
        ok = False
        parsed = None
        response_text = ""
        error = None
        usage = None
        try:
            raw = await asyncio.to_thread(
                http_chat_completion,
                base_url=spec.base_url,
                api_key=spec.api_key,
                model=spec.name,
                prompt=item.prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
                extra_headers=spec.extra_headers,
                params=spec.params,
            )
            msg = raw["choices"][0]["message"]
            response_text = msg.get("content") or ""
            usage = raw.get("usage")
            parsed = parse_choice_letter(response_text, item.valid_letters)
            ok = parsed is not None
            if parsed is None:
                # One clarification exchange, matching completion.py behavior.
                # Full token budget: reasoning models must think before the tag.
                clar = await asyncio.to_thread(
                    http_chat_completion,
                    base_url=spec.base_url,
                    api_key=spec.api_key,
                    model=spec.name,
                    prompt=(
                        item.prompt_text
                        + "\n\nYour previous reply contained no <answer> tag. "
                        + "Reply with ONLY the tag, e.g. <answer>A</answer>."
                    ),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    extra_headers=spec.extra_headers,
                    params=spec.params,
                )
                retry_msg = clar["choices"][0]["message"].get("content") or ""
                parsed = parse_choice_letter(retry_msg, item.valid_letters)
                if parsed is not None:
                    response_text = retry_msg
                    ok = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        explanation = None
        if response_text:
            m = _EXPL_RE.search(response_text)
            explanation = m.group(1).strip() if m else None
        record = {
            "question_id": item.question_id,
            "family": item.question.get("family"),
            "question_type": item.question.get("type"),
            "budget": item.question.get("budget", {}).get("total_samples_seen"),
            "model": spec.name,
            "provider": spec.provider,
            "request_params": spec.params or None,
            "prompt_hash": item.prompt_hash,
            "correct_letter": item.correct_letter,
            "parsed_letter": parsed,
            "correct": bool(parsed is not None and parsed == item.correct_letter),
            "ok": ok,
            "error": error,
            "seconds": round(time.time() - t0, 1),
            "usage": usage,
            "explanation": explanation,
            "response": response_text,
        }
        out = result_path(out_dir, spec, item.question_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(out)
        lp = ledger_path(out_dir, spec)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_model(
    spec: ModelSpec,
    items: list[QuestionItem],
    out_dir: Path,
    *,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    limit: int | None,
) -> dict[str, Any]:
    done = load_done(out_dir, spec)
    if limit is not None:
        # limit applies to the global question list, so a resumed run with a
        # larger limit continues from where the smaller one stopped
        # (pilot-first, then full).
        items = items[:limit]
    pending = [it for it in items if it.question_id not in done]
    sem = asyncio.Semaphore(spec.concurrency)
    started = time.time()
    await asyncio.gather(
        *[
            eval_one(
                it,
                spec,
                out_dir,
                sem,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
                max_exchanges=1,
            )
            for it in pending
        ]
    )
    # re-read per-question results for the summary (includes resumed ones)
    results: list[dict[str, Any]] = []
    rdir = out_dir / "results" / spec.slug
    if rdir.is_dir():
        for path in rdir.glob("*.json"):
            try:
                results.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    scored = [r for r in results if r.get("parsed_letter") is not None and r.get("ok")]
    correct = sum(1 for r in scored if r["correct"])
    return {
        "model": spec.name,
        "provider": spec.provider,
        "total": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "seconds": round(time.time() - started, 1),
    }


def load_backends(config_path: Path, *, default_concurrency: int) -> dict[str, ModelSpec]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    specs: dict[str, ModelSpec] = {}
    for provider, block in cfg.items():
        api_key = str(block.get("api_key") or "").strip() or os.environ.get(
            str(block.get("api_key_env") or ""), ""
        ).strip()
        if not api_key:
            raise SystemExit(
                f"missing api_key (or env {block.get('api_key_env')!r}) for provider {provider}"
            )
        block_params = dict(block.get("params", {}))
        for entry in block.get("models", []):
            if isinstance(entry, str):
                name, model_params, model_conc = entry, {}, None
            else:
                name = entry["name"]
                model_params = dict(entry.get("params", {}))
                model_conc = entry.get("concurrency")
            params = {**block_params, **model_params}
            specs[f"{provider}:{name}"] = ModelSpec(
                provider=provider,
                name=name,
                base_url=block["base_url"],
                api_key=api_key,
                concurrency=int(model_conc or block.get("concurrency", default_concurrency)),
                extra_headers=block.get("extra_headers", {}),
                params=params,
            )
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions-root", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=None, help="benchmark ledger jsonl (preferred ordering)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--backends-config", type=Path, required=True)
    ap.add_argument("--models", required=True, help="comma list; entries may be name or provider:name")
    ap.add_argument("--concurrency", type=int, default=8, help="default per-model concurrency")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="completion token cap; 0 omits the field (provider default)")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap questions per model (pilot)")
    ap.add_argument("--only", type=Path, default=None,
                    help="json list of question_id or ledger item_id; eval exactly this subset")
    args = ap.parse_args()

    specs = load_backends(args.backends_config, default_concurrency=args.concurrency)
    wanted: list[ModelSpec] = []
    for token in args.models.split(","):
        token = token.strip()
        if not token:
            continue
        if token in specs:
            wanted.append(specs[token])
            continue
        matches = [s for s in specs.values() if s.name == token]
        if not matches:
            raise SystemExit(f"model {token!r} not found in backends config")
        wanted.extend(matches)

    items = list_questions(args.questions_root, args.ledger)
    if args.only:
        raw = json.loads(args.only.read_text(encoding="utf-8"))
        # Entries may be plain ids or rich dicts carrying question_id/item_id.
        wanted_ids = {
            str(e.get("question_id") or e.get("item_id")) if isinstance(e, dict) else str(e)
            for e in raw
        }
        items = [
            it for it in items
            if it.question_id in wanted_ids or it.question.get("item_id") in wanted_ids
        ]
    if not items:
        raise SystemExit("no questions found")
    print(f"[{datetime.now(timezone.utc).isoformat()}] {len(items)} questions, "
          f"models: {[s.name for s in wanted]}", flush=True)

    async def run_all() -> list[dict[str, Any]]:
        return await asyncio.gather(
            *[
                run_model(
                    spec,
                    items,
                    args.out_dir,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                    max_retries=args.max_retries,
                    limit=args.limit,
                )
                for spec in wanted
            ]
        )

    summaries = asyncio.run(run_all())
    summary_path = args.out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "questions": len(items),
        "runs": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    for s in summaries:
        acc = s["accuracy"]
        print(f"{s['model']:32s} {s['correct']}/{s['total']}"
              + (f"  acc={acc:.3f}" if acc is not None else "  acc=n/a"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
