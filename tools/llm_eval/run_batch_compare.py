#!/usr/bin/env python3
"""Compare three question batches (demo / XOR / GRU) with one relay model.

Reads batch artifacts directly (no architecture_iq imports):
- demo:  the 46-question demo bake backup under /tmp/aiq_demo_bake_backup.json
         (prompt = byId[q].detail.prompt, GT = byId[q].reveal.correctLetter)
- xor:   benchmark_releases/question_packs/xor-v2.5-100q-37b9da
- gru:   benchmark_releases/question_packs/gru-v2.5-100q-a48abc

Credentials come from ~/.agents/relay.json (eval key only). No max_tokens is
sent; reasoning_effort defaults to high.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from prompt_wrapper import format_eval_prompt  # noqa: E402
from question_loader import load_question_item  # noqa: E402
from response_parser import parse_choice_letter  # noqa: E402

DEMO_BAKE = Path("/tmp/aiq_demo_bake_backup.json")
XOR_PACK = ROOT / "benchmark_releases/question_packs/xor-v2.5-100q-37b9da"
GRU_PACK = ROOT / "benchmark_releases/question_packs/gru-v2.5-100q-a48abc"


def _load_demo_batch() -> list[dict[str, Any]]:
    payload = json.loads(DEMO_BAKE.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for qid, item in payload["byId"].items():
        prompt = item["detail"]["prompt"]
        correct = item["reveal"]["correctLetter"].upper()
        letters = frozenset(c["letter"].upper() for c in item["detail"]["choices"])
        out.append(
            {
                "question_id": qid,
                "batch": "demo",
                "prompt": prompt,
                "correct_letter": correct,
                "valid_letters": letters,
                "family": item.get("family"),
                "type": item.get("type"),
                "significance": item.get("significance"),
                "source": str(DEMO_BAKE),
            }
        )
    return out


def _load_pack_batch(batch: str, pack_root: Path) -> list[dict[str, Any]]:
    collection = json.loads(
        (pack_root / "collection.json").read_text(encoding="utf-8")
    )
    data_root = pack_root / "data"
    out: list[dict[str, Any]] = []
    for rel in collection["question_paths"]:
        item = load_question_item(data_root / rel)
        q = item.question
        out.append(
            {
                "question_id": item.question_id,
                "batch": batch,
                "prompt": item.prompt_text,
                "correct_letter": item.correct_letter,
                "valid_letters": item.valid_letters,
                "family": q.get("family"),
                "type": q.get("type"),
                "significance": q.get("significance"),
                "source": str(data_root / rel),
            }
        )
    return out


def load_batches() -> dict[str, list[dict[str, Any]]]:
    return {
        "demo": _load_demo_batch(),
        "xor": _load_pack_batch("xor", XOR_PACK),
        "gru": _load_pack_batch("gru", GRU_PACK),
    }


def _relay_eval_config() -> dict[str, str]:
    path = Path(os.path.expanduser("~/.agents/relay.json"))
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    relay = json.loads(path.read_text(encoding="utf-8"))
    eval_cfg = relay.get("eval")
    if not eval_cfg or not eval_cfg.get("api_key") or not eval_cfg.get("base_url"):
        raise SystemExit("relay.json eval section is missing key/base_url")
    return eval_cfg


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def complete_one(
    config: dict[str, str],
    model: str,
    prompt: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "reasoning_effort": "high",
    }
    url = config["base_url"].rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    for attempt in range(2):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600, context=_ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            if attempt == 0 and "reasoning_effort" in err_text:
                body.pop("reasoning_effort", None)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {err_text[:400]}") from exc
    raise RuntimeError("unreachable")


def _message_text(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking", "content"):
        piece = message.get(key)
        if piece:
            chunks.append(str(piece))
    return "\n\n".join(chunks)


def evaluate_row(
    config: dict[str, str],
    model: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    letters = row["valid_letters"]
    eval_prompt = format_eval_prompt(row["prompt"], letters)
    raw = complete_one(config, model, eval_prompt)
    message = raw["choices"][0]["message"]
    full_text = _message_text(message)
    parsed = parse_choice_letter(full_text, letters)
    finish = raw["choices"][0].get("finish_reason")
    usage = raw.get("usage")
    return {
        "question_id": row["question_id"],
        "batch": row["batch"],
        "family": row["family"],
        "type": row["type"],
        "significance": row["significance"],
        "ground_truth_letter": row["correct_letter"],
        "parsed_letter": parsed,
        "correct": parsed == row["correct_letter"],
        "model_response": full_text,
        "eval_prompt": eval_prompt,
        "finish_reason": finish,
        "usage": usage,
    }


def summarize(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_batch: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_batch.setdefault(r.get("batch", "unknown"), []).append(r)
    summary: dict[str, Any] = {}
    for batch, rows in sorted(by_batch.items()):
        total = len(rows)
        parsed = [r for r in rows if r.get("parsed_letter") is not None]
        correct = sum(1 for r in rows if r.get("correct"))
        a_answers = sum(1 for r in rows if r.get("parsed_letter") == "A")
        gt_a = sum(1 for r in rows if r.get("ground_truth_letter") == "A")
        gaps = [
            float(r["significance"].get("gap"))
            for r in rows
            if r.get("significance") and r["significance"].get("gap") is not None
        ]
        wins = [
            float(r["significance"].get("win_rate"))
            for r in rows
            if r.get("significance") and r["significance"].get("win_rate") is not None
        ]
        summary[batch] = {
            "total": total,
            "parsed": len(parsed),
            "correct": correct,
            "accuracy": correct / total if total else None,
            "answers_A": a_answers,
            "answers_A_fraction": a_answers / total if total else None,
            "gt_A": gt_a,
            "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "mean_win_rate": (sum(wins) / len(wins)) if wins else None,
        }
    return {
        "model": model,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "summary": summary,
        "overall_accuracy": (
            sum(1 for r in results if r["correct"]) / len(results) if results else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--per-batch", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", nargs="+", default=["demo", "xor", "gru"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    batches = load_batches()
    for name in args.batches:
        if name not in batches:
            raise SystemExit(f"unknown batch {name!r}; have {sorted(batches)}")

    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    for name in args.batches:
        pool = batches[name]
        picked = rng.sample(pool, min(args.per_batch, len(pool)))
        sampled.extend(picked)
        print(f"{name}: {len(pool)} questions, sampling {len(picked)}", flush=True)

    config = _relay_eval_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else ROOT / "llm_runs" / f"batch3_{stamp}_{args.model}"
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # skip question_ids already on disk (resume after a crash)
    done_ids = {p.stem.split("_", 1)[1] for p in results_dir.glob("*.json")}
    todo = [row for row in sampled if str(row["question_id"]) not in done_ids]
    if todo:
        print(f"resume: {len(done_ids)} already on disk, running {len(todo)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_row, config, args.model, row): row for row in todo}
            done = 0
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - record and continue
                    result = {
                        "question_id": row["question_id"],
                        "batch": row["batch"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "correct": False,
                    }
                qid = str(result["question_id"])
                (results_dir / f"{result['batch']}_{qid}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                done += 1
                if done % 5 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} done", flush=True)

    results = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    (out_dir / "run.json").write_text(
        json.dumps(summarize(results, args.model), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir}", flush=True)
    print(json.dumps(summarize(results, args.model)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
