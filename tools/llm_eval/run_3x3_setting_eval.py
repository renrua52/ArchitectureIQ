#!/usr/bin/env python
"""LLM blind baseline evaluation for the 3-family x 3-setting x 10-question bundle.

Mirrors tools/llm_baseline_eval.py's protocol but scores by (family, setting)
instead of just family, and points at the setting3x3_eval bundle.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openai import OpenAI
from provider_config import provider_api_key, provider_base_url

BASE_URL = "https://api.gpt.ge/v1"

BUNDLE_DIR = Path("artifacts/setting3x3_eval")
PROMPT_PATH = BUNDLE_DIR / "prompt.txt"
CORRECT_PATH = BUNDLE_DIR / "answer_key.json"
META_PATH = BUNDLE_DIR / "question_meta.json"
OUTPUT_DIR = BUNDLE_DIR


def load_prompt() -> str:
    return PROMPT_PATH.read_text()


def load_correct() -> dict[str, str]:
    return json.loads(CORRECT_PATH.read_text())


def load_meta() -> dict[str, dict]:
    return json.loads(META_PATH.read_text())


def load_sanitized() -> list[dict]:
    return json.loads((BUNDLE_DIR / "questions_sanitized.json").read_text())


HEADER_TEMPLATE = """# 3-family x 3-setting x 10-question blind baseline prompt (per-family batch: {family})

You are an independent blind-answer agent for ArchitectureIQ.

STRICT PROTOCOL:

- You will receive {n}-question sanitized subset (family: {family}; 3 dataset instances/"settings" within this family; 10 questions per setting).
- Answer only from the visible sanitized questions and qualitative reasoning.
- Do not read answer keys, feedback files, scoring files, result summaries, curves, previous attempts, repository files, or any hidden ground-truth artifacts.
- Do not run shell commands, Python, Node, jq, scripts, training, local simulations, approximate experiments, or data reconstruction.
- You may compare across the visible questions, repeated candidates, model families, optimizers, learning rates, budgets, and architecture patterns, including across settings within this family.
- Return strict JSON only with keys: `agent`, `model`, `reasoning_effort`, `source_used`, `forbidden_files_viewed`, and `predictions`.
- `predictions` must be an array of exactly {n} records.
- Each prediction must contain: `n`, `question_id`, `predicted_letter`, `predicted_candidate_id`, `confidence`, and `reason` (reason must be at most one short sentence).
- `predicted_candidate_id` must match the selected letter in that question.

Sanitized {n}-question JSON:

"""


def build_family_prompt(family: str, questions: list[dict]) -> str:
    header = HEADER_TEMPLATE.format(family=family, n=len(questions))
    return header + json.dumps(questions, indent=2) + "\n"


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 16384) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


def extract_predictions(raw: str) -> list[dict] | None:
    try:
        data = json.loads(raw)
        if "predictions" in data:
            return data["predictions"]
    except Exception:
        pass

    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "predictions" in data:
                return data["predictions"]
        except Exception:
            pass

    pred_match = re.search(r'"predictions"\s*:\s*\[', raw)
    if pred_match:
        start = pred_match.start()
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
                        data = json.loads(raw[start:i + 1])
                        if isinstance(data, dict) and "predictions" in data:
                            return data["predictions"]
                        elif isinstance(data, list):
                            return data
                    except Exception:
                        pass
                    break

    return None


def score_predictions(predictions: list[dict], correct: dict[str, str], meta: dict[str, dict]) -> dict:
    total = 0
    correct_count = 0
    by_family = {}
    by_setting = {}

    for pred in predictions:
        qid = pred.get("question_id", "")
        letter = pred.get("predicted_letter", "")
        if qid in correct:
            total += 1
            m = meta.get(qid, {})
            fam = m.get("family", "unknown")
            setting = f"{fam}/{m.get('dataset_id', 'unknown')}"

            by_family.setdefault(fam, {"correct": 0, "total": 0})
            by_setting.setdefault(setting, {"correct": 0, "total": 0})
            by_family[fam]["total"] += 1
            by_setting[setting]["total"] += 1

            if letter == correct[qid]:
                correct_count += 1
                by_family[fam]["correct"] += 1
                by_setting[setting]["correct"] += 1

    return {
        "total": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
        "by_family": {f: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"]}
                      for f, v in by_family.items()},
        "by_setting": {s: {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"]}
                       for s, v in by_setting.items()},
    }


def run_eval(model_name: str, model_display: str, max_tokens: int = 16384, per_family: bool = False) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Testing: {model_display} ({model_name}) per_family={per_family}")
    print(f"{'=' * 60}")

    client = OpenAI(api_key=provider_api_key(), base_url=provider_base_url(default=BASE_URL))
    correct = load_correct()
    meta = load_meta()

    all_predictions: list[dict] = []
    raw_chunks: list[str] = []
    total_elapsed = 0.0

    if per_family:
        sanitized = load_sanitized()
        families = sorted({q["family"] for q in sanitized})
        for family in families:
            fam_questions = [q for q in sanitized if q["family"] == family]
            prompt = build_family_prompt(family, fam_questions)
            print(f"  [{family}] calling with {len(fam_questions)} questions...")
            t0 = time.time()
            try:
                raw = call_llm(client, model_name, prompt, max_tokens=max_tokens)
                elapsed = time.time() - t0
                total_elapsed += elapsed
                print(f"    Response received in {elapsed:.1f}s, length: {len(raw)} chars")
            except Exception as e:
                print(f"    API call failed: {e}")
                return {"model": model_display, "api_name": model_name, "error": str(e), "family": family}

            (OUTPUT_DIR / f"{model_display}_{family}_raw.txt").write_text(raw)
            raw_chunks.append(raw)

            preds = extract_predictions(raw)
            if preds is None:
                print(f"    Failed to extract predictions for {family}")
                return {"model": model_display, "api_name": model_name,
                        "error": f"Failed to extract predictions for {family}",
                        "raw_length": len(raw), "raw_preview": raw[:500]}
            print(f"    Extracted {len(preds)} predictions")
            all_predictions.extend(preds)
            time.sleep(1)
    else:
        prompt = load_prompt()
        t0 = time.time()
        try:
            raw = call_llm(client, model_name, prompt, max_tokens=max_tokens)
            total_elapsed = time.time() - t0
            print(f"  Response received in {total_elapsed:.1f}s, length: {len(raw)} chars")
        except Exception as e:
            print(f"  API call failed: {e}")
            return {"model": model_display, "api_name": model_name, "error": str(e)}

        raw_path = OUTPUT_DIR / f"{model_display}_raw.txt"
        raw_path.write_text(raw)
        raw_chunks.append(raw)

        preds = extract_predictions(raw)
        if preds is None:
            print("  Failed to extract predictions from response")
            return {"model": model_display, "api_name": model_name, "error": "Failed to extract predictions",
                     "raw_length": len(raw), "raw_preview": raw[:500]}
        all_predictions = preds

    print(f"  Extracted {len(all_predictions)} total predictions")

    pred_data = {
        "agent": "ArchitectureIQ 3x3-setting blind-answer LLM baseline",
        "model": model_display,
        "reasoning_effort": "high",
        "source_used": "visible_sanitized_questions_only",
        "forbidden_files_viewed": False,
        "predictions": all_predictions,
    }
    pred_path = OUTPUT_DIR / f"{model_display}_answers.json"
    pred_path.write_text(json.dumps(pred_data, indent=2))

    scores = score_predictions(all_predictions, correct, meta)
    print(f"  Score: {scores['correct']}/{scores['total']} = {scores['accuracy']:.4f}")
    for fam, s in sorted(scores["by_family"].items()):
        print(f"    {fam}: {s['correct']}/{s['total']} = {s['accuracy']:.4f}")

    result = {
        "model": model_display,
        "api_name": model_name,
        "elapsed_seconds": total_elapsed,
        "raw_length": sum(len(r) for r in raw_chunks),
        "n_predictions": len(all_predictions),
        **scores,
    }

    scored_path = OUTPUT_DIR / f"{model_display}_scored.json"
    scored_path.write_text(json.dumps(result, indent=2))

    return result


def main():
    import sys

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    only = sys.argv[1] if len(sys.argv) > 1 else None

    models_to_test = [
        ("gpt-5.5", "gpt-5.5-high", 16384, False),
        ("claude-opus-4-8-high", "claude-opus-4.8-high", 16384, True),
    ]
    if only:
        models_to_test = [m for m in models_to_test if only in m[1]]

    for api_name, display_name, max_tokens, per_family in models_to_test:
        run_eval(api_name, display_name, max_tokens=max_tokens, per_family=per_family)
        time.sleep(2)

    # Merge with any previously-scored models (so partial reruns still
    # produce a combined summary).
    all_display_names = ["gpt-5.5-high", "claude-opus-4.8-high"]
    results = []
    for display_name in all_display_names:
        scored_path = OUTPUT_DIR / f"{display_name}_scored.json"
        if scored_path.exists():
            results.append(json.loads(scored_path.read_text()))

    print(f"\n{'=' * 60}")
    print("SUMMARY (3-family x 3-setting x 10-question bundle)")
    print(f"{'=' * 60}")
    print(f"{'Model':<30s} {'Score':>10s} {'Accuracy':>10s}")
    print("-" * 55)
    for r in results:
        if "accuracy" in r:
            print(f"{r['model']:<30s} {r['correct']}/{r['total']:<8} {r['accuracy']:>10.4f}")
        else:
            print(f"{r['model']:<30s} ERROR: {r.get('error', '?')}")

    summary = {"results": results, "reference": {
        "random": {"accuracy": 1 / 3},
    }}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
