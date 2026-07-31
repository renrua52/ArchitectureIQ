#!/usr/bin/env python
"""LLM blind baseline evaluation for ArchitectureIQ 60-question set.

Uses the same prompt and protocol as the original GPT-5.6-SOL blind evaluation.
Tests multiple LLMs via the phybench API provider.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openai import OpenAI
from llm_eval.provider_config import provider_api_key, provider_base_url

BASE_URL = "https://api.gpt.ge/v1"

PROMPT_PATH = Path("artifacts/quiz_attempt_60/llm_baseline_eval/prompt.txt")
CORRECT_PATH = Path("artifacts/quiz_attempt_60/llm_baseline_eval/correct_answers.json")
OUTPUT_DIR = Path("artifacts/quiz_attempt_60/llm_baseline_eval")


def load_prompt() -> str:
    return PROMPT_PATH.read_text()


def load_correct() -> dict[str, str]:
    return json.loads(CORRECT_PATH.read_text())


def call_llm(client: OpenAI, model: str, prompt: str) -> str:
    """Call the LLM with the blind prompt, return raw text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=16384,
        temperature=0.0,
    )
    return response.choices[0].message.content


def extract_predictions(raw: str) -> list[dict] | None:
    """Extract JSON predictions from LLM response."""
    # Try to find JSON in the response
    # First try direct parse
    try:
        data = json.loads(raw)
        if "predictions" in data:
            return data["predictions"]
    except Exception:
        pass

    # Try to find JSON block
    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "predictions" in data:
                return data["predictions"]
        except Exception:
            pass

    # Try to find predictions array directly
    pred_match = re.search(r'"predictions"\s*:\s*\[', raw)
    if pred_match:
        # Find the matching closing bracket
        start = pred_match.start()
        # Try to parse from there
        brace = 0
        bracket = 0
        for i in range(start, len(raw)):
            if raw[i] == '{':
                brace += 1
            elif raw[i] == '}':
                brace -= 1
            elif raw[i] == '[':
                bracket += 1
            elif raw[i] == ']':
                bracket -= 1
                if bracket == 0:
                    try:
                        data = json.loads(raw[start:i+1])
                        if isinstance(data, dict) and "predictions" in data:
                            return data["predictions"]
                        elif isinstance(data, list):
                            return data
                    except Exception:
                        pass
                    break

    return None


def score_predictions(predictions: list[dict], correct: dict[str, str]) -> dict:
    """Score predictions against correct answers."""
    total = 0
    correct_count = 0
    by_family = {}
    
    # Load question metadata for family lookup
    import glob
    q_meta = {}
    for qf in sorted(glob.glob("artifacts/quiz_attempt_60/sanitized_questions/*.json")):
        q = json.load(open(qf))
        q_meta[q["question_id"]] = q["family"]

    for pred in predictions:
        qid = pred.get("question_id", "")
        letter = pred.get("predicted_letter", "")
        if qid in correct:
            total += 1
            fam = q_meta.get(qid, "unknown")
            if fam not in by_family:
                by_family[fam] = {"correct": 0, "total": 0}
            by_family[fam]["total"] += 1
            if letter == correct[qid]:
                correct_count += 1
                by_family[fam]["correct"] += 1

    return {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "by_family": {f: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"]/v["total"]}
                      for f, v in by_family.items()},
    }


def run_eval(model_name: str, model_display: str) -> dict:
    """Run blind evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_display} ({model_name})")
    print(f"{'='*60}")

    client = OpenAI(api_key=provider_api_key(), base_url=provider_base_url(default=BASE_URL))
    prompt = load_prompt()
    correct = load_correct()

    # Call the LLM
    t0 = time.time()
    try:
        raw = call_llm(client, model_name, prompt)
        elapsed = time.time() - t0
        print(f"  Response received in {elapsed:.1f}s, length: {len(raw)} chars")
    except Exception as e:
        print(f"  API call failed: {e}")
        return {"model": model_display, "api_name": model_name, "error": str(e)}

    # Save raw response
    raw_path = OUTPUT_DIR / f"{model_display}_raw.txt"
    raw_path.write_text(raw)

    # Extract predictions
    predictions = extract_predictions(raw)
    if predictions is None:
        print("  Failed to extract predictions from response")
        # Save for debugging
        return {"model": model_display, "api_name": model_name, "error": "Failed to extract predictions",
                "raw_length": len(raw), "raw_preview": raw[:500]}

    print(f"  Extracted {len(predictions)} predictions")

    # Save predictions
    pred_data = {
        "agent": "ArchitectureIQ blind-answer LLM baseline",
        "model": model_display,
        "reasoning_effort": "high",
        "source_used": "visible_sanitized_questions_only",
        "forbidden_files_viewed": False,
        "predictions": predictions,
    }
    pred_path = OUTPUT_DIR / f"{model_display}_answers.json"
    pred_path.write_text(json.dumps(pred_data, indent=2))

    # Score
    scores = score_predictions(predictions, correct)
    print(f"  Score: {scores['correct']}/{scores['total']} = {scores['accuracy']:.4f}")
    for fam, s in sorted(scores["by_family"].items()):
        print(f"    {fam}: {s['correct']}/{s['total']} = {s['accuracy']:.4f}")

    result = {
        "model": model_display,
        "api_name": model_name,
        "elapsed_seconds": elapsed,
        "raw_length": len(raw),
        "n_predictions": len(predictions),
        **scores,
    }

    # Save scored result
    scored_path = OUTPUT_DIR / f"{model_display}_scored.json"
    scored_path.write_text(json.dumps(result, indent=2))

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models_to_test = [
        ("gpt-5.5", "gpt-5.5-high"),
        ("claude-opus-4-8-high", "claude-opus-4.8-high"),
    ]

    results = []
    for api_name, display_name in models_to_test:
        result = run_eval(api_name, display_name)
        results.append(result)
        # Brief pause between models
        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30s} {'Score':>10s} {'Accuracy':>10s}")
    print("-" * 55)
    for r in results:
        if "accuracy" in r:
            print(f"{r['model']:<30s} {r['correct']}/{r['total']:<8} {r['accuracy']:>10.4f}")
        else:
            print(f"{r['model']:<30s} ERROR: {r.get('error', '?')}")

    # Also show reference scores
    print("\nReference scores:")
    print(f"{'cv_champion (meta-model)':<30s} 54/60       0.9000")
    print(f"{'GPT-5.6-SOL blind':<30s} 25/60       0.4167")
    print(f"{'random':<30s} 20/60       0.3333")

    # Save summary
    summary = {"results": results, "reference": {
        "cv_champion": {"correct": 54, "total": 60, "accuracy": 0.9},
        "gpt56_sol_blind": {"correct": 25, "total": 60, "accuracy": 0.4167},
        "random": {"correct": 20, "total": 60, "accuracy": 0.3333},
    }}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
