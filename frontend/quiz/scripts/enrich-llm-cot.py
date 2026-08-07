#!/usr/bin/env python3
"""Inject multi-model llmCot into a BakeFile from benchmarks/v1_llm/llm_runs.

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


def truncate(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS].rstrip() + "\n\n…[truncated for quiz display]"


def enrich(bake: dict, runs: Path) -> dict:
    models = sorted(
        [p.name for p in runs.iterdir() if p.is_dir() and p.name not in EXCLUDE],
        key=model_rank,
    )
    stats = {
        "questions": 0,
        "with_any_cot": 0,
        "with_correct_cot": 0,
        "no_correct": 0,
        "entries": 0,
    }
    for qid, item in bake["byId"].items():
        stats["questions"] += 1
        entries: list[dict] = []
        saw_correct = False
        for model in models:
            path = runs / model / "results" / f"{qid}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("error"):
                continue
            parsed = rec.get("parsed_letter")
            correct = bool(rec.get("correct")) and parsed is not None
            if correct:
                saw_correct = True
            text, source = pick_text(rec)
            if not text or len(text) < MIN_CHARS:
                continue
            entries.append(
                {
                    "model": model,
                    "correct": correct,
                    "parsedLetter": parsed,
                    "source": source,
                    "text": truncate(text),
                }
            )
        # Prefer a correct model as default; else first entry.
        default_model = None
        for entry in entries:
            if entry["correct"]:
                default_model = entry["model"]
                break
        if default_model is None and entries:
            default_model = entries[0]["model"]

        if entries:
            item["llmCot"] = {
                "available": True,
                "defaultModel": default_model,
                "entries": entries,
            }
            stats["with_any_cot"] += 1
            stats["entries"] += len(entries)
            if any(e["correct"] for e in entries):
                stats["with_correct_cot"] += 1
            elif not saw_correct:
                stats["no_correct"] += 1
        else:
            item["llmCot"] = {
                "available": False,
                "reason": "no_correct" if not saw_correct else "no_cot",
                "defaultModel": None,
                "entries": [],
            }
            if not saw_correct:
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
    print(f"size_mb={args.bake.stat().st_size / 1e6:.2f}")


if __name__ == "__main__":
    main()
