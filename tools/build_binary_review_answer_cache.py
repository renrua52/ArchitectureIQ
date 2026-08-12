#!/usr/bin/env python3
"""Build full model-response assets for the binary review viewer.

The compact review JSON intentionally keeps only a short response excerpt. The
viewer loads this cache only after a question is opened, so the complete model
analysis remains available without making the table payload much heavier.

Example:
    .venv/bin/python tools/build_binary_review_answer_cache.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path("/tmp/v1bundle"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/v1_review/binary_questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/v1_review/answers"),
    )
    return parser.parse_args()


def response_text(result: dict) -> str:
    response = result.get("model_response")
    if response:
        return str(response)
    parts = result.get("message_parts") or {}
    reasoning = parts.get("reasoning_content") or ""
    content = parts.get("content") or ""
    if reasoning and content:
        return f"{reasoning}\n\n{content}"
    return str(reasoning or content)


def main() -> None:
    args = parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))
    models_root = args.bundle_root / "benchmarks" / "v1_llm" / "llm_runs"
    model_dirs = sorted(path for path in models_root.iterdir() if path.is_dir())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    response_count = 0
    for question in data["questions"]:
        answers = []
        qid = question["question_id"]
        for model_dir in model_dirs:
            result_path = model_dir / "results" / f"{qid}.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            response = response_text(result)
            answers.append(
                {
                    "model": model_dir.name,
                    "letter": result.get("parsed_letter"),
                    "correct": result.get("correct"),
                    "response": response,
                    "truncated": bool(result.get("truncated", False)),
                    "finish_reason": result.get("finish_reason"),
                    "continuation_count": result.get("continuation_count", 0),
                }
            )
            response_count += 1
        (args.output_dir / f"{qid}.json").write_text(
            json.dumps(
                {"question_id": qid, "answer_count": len(answers), "answers": answers},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        written += 1
    manifest = {
        "schema": "v1_binary_review_answers_v1",
        "question_count": written,
        "response_count": response_count,
        "models": [path.name for path in model_dirs],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {written} question answer assets ({response_count} responses) to {args.output_dir}")


if __name__ == "__main__":
    main()
