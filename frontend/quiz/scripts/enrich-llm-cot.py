#!/usr/bin/env python3
"""Inject llmCot into a BakeFile from benchmarks/v1_llm/llm_runs.

Usage:
  python scripts/enrich-llm-cot.py \\
    --bake frontend/quiz/public/data/questions.json \\
    --runs /path/to/benchmarks/v1_llm/llm_runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXCLUDE = {"Kimi-K3", "grok-4.2"}
PREFERRED = [
    "claude-opus-5",
    "claude-sonnet-4-6",
    "gpt-5.2",
    "gpt-5.6-sol",
    "gpt-5.4-mini",
    "gpt-4o",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "GLM-5.1",
    "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash",
]
MAX_CHARS = 12_000
MIN_CHARS = 80


def pick_text(rec: dict) -> tuple[str | None, str | None]:
    parts = rec.get("message_parts") or {}
    candidates = [
        ("reasoning_content", (parts.get("reasoning_content") or parts.get("reasoning") or "").strip()),
        ("chain_of_thought", (rec.get("chain_of_thought") or "").strip()),
        ("content", (parts.get("content") or "").strip()),
        ("model_response", (rec.get("model_response") or "").strip()),
    ]
    for source, text in candidates:
        if len(text) >= MIN_CHARS:
            return text, source
    for source, text in candidates:
        if text:
            return text, source
    return None, None


def model_rank(name: str) -> tuple[int, str]:
    try:
        return PREFERRED.index(name), name
    except ValueError:
        return len(PREFERRED), name


def enrich(bake: dict, runs: Path) -> dict:
    models = sorted(
        [p.name for p in runs.iterdir() if p.is_dir() and p.name not in EXCLUDE],
        key=model_rank,
    )
    stats = {"with_cot": 0, "no_correct": 0, "correct_no_cot": 0}
    for qid, item in bake["byId"].items():
        candidates = []
        saw_correct = False
        for model in models:
            path = runs / model / "results" / f"{qid}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("error") or rec.get("parsed_letter") is None:
                continue
            if not rec.get("correct"):
                continue
            saw_correct = True
            text, source = pick_text(rec)
            if not text or len(text) < MIN_CHARS:
                continue
            candidates.append((model_rank(model), len(text), model, text, source, rec.get("parsed_letter")))
        if candidates:
            candidates.sort(key=lambda row: (row[0][0], row[1]))
            _, _, model, text, source, letter = candidates[0]
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS].rstrip() + "\n\n…[truncated for quiz display]"
            item["llmCot"] = {
                "available": True,
                "model": model,
                "parsedLetter": letter,
                "source": source,
                "text": text,
            }
            stats["with_cot"] += 1
        elif saw_correct:
            item["llmCot"] = {"available": False, "reason": "no_cot"}
            stats["correct_no_cot"] += 1
        else:
            item["llmCot"] = {"available": False, "reason": "no_correct"}
            stats["no_correct"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    bake = json.loads(args.bake.read_text(encoding="utf-8"))
    stats = enrich(bake, args.runs)
    args.bake.write_text(json.dumps(bake, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
