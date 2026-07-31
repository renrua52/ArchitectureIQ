#!/usr/bin/env python
"""Per-setting isolated batch eval: for each of the 9 settings, send its 10
questions as ONE standalone API call (no cross-setting context at all),
for both gpt-5.5-high and claude-opus-4.8-high via the api.gpt.ge endpoint.
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
OUT_DIR = BUNDLE_DIR / "isolated_batch"

HEADER_TEMPLATE = """# Isolated single-setting batch blind baseline ({family} / {dataset_id})

You are an independent blind-answer agent for ArchitectureIQ.

STRICT PROTOCOL:

- You will receive exactly 10 sanitized questions, all from ONE dataset instance/"setting" ({family} / {dataset_id}). No other settings or questions are visible to you in this call.
- Answer only from the visible sanitized questions and qualitative reasoning.
- Do not read answer keys, feedback files, scoring files, result summaries, curves, previous attempts, repository files, or any hidden ground-truth artifacts.
- Do not run shell commands, Python, Node, jq, scripts, training, local simulations, approximate experiments, or data reconstruction.
- You may compare across the 10 visible questions in this setting (repeated candidates, model families, optimizers, learning rates, budgets, architecture patterns).
- Return strict JSON only with keys: `agent`, `model`, `reasoning_effort`, `source_used`, `forbidden_files_viewed`, and `predictions`.
- `predictions` must be an array of exactly 10 records.
- Each prediction must contain: `n`, `question_id`, `predicted_letter`, `predicted_candidate_id`, `confidence`, and `reason` (at most one short sentence).
- `predicted_candidate_id` must match the selected letter in that question.

Sanitized 10-question JSON:

"""


def load_sanitized() -> list[dict]:
    return json.loads((BUNDLE_DIR / "questions_sanitized.json").read_text())


def load_correct() -> dict[str, str]:
    return json.loads((BUNDLE_DIR / "answer_key.json").read_text())


def group_by_setting(sanitized: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for q in sanitized:
        key = f"{q['family']}/{q['dataset_id']}"
        groups.setdefault(key, []).append(q)
    return groups


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
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
    m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if "predictions" in data:
                return data["predictions"]
        except Exception:
            pass
    m = re.search(r'"predictions"\s*:\s*\[', raw)
    if m:
        start = m.start()
        bracket = 0
        for i in range(start, len(raw)):
            if raw[i] == '[':
                bracket += 1
            elif raw[i] == ']':
                bracket -= 1
                if bracket == 0:
                    try:
                        data = json.loads(raw[start:i + 1])
                        if isinstance(data, dict) and "predictions" in data:
                            return data["predictions"]
                        if isinstance(data, list):
                            return data
                    except Exception:
                        pass
                    break
    return None


def run_one(client: OpenAI, model_name: str, model_display: str, setting_key: str,
            questions: list[dict], correct: dict[str, str], max_tokens: int) -> dict:
    family, dataset_id = setting_key.split("/")
    prompt = HEADER_TEMPLATE.format(family=family, dataset_id=dataset_id) + json.dumps(questions, indent=2) + "\n"
    safe_key = setting_key.replace("/", "__")

    t0 = time.time()
    try:
        raw = call_llm(client, model_name, prompt, max_tokens)
        elapsed = time.time() - t0
    except Exception as e:
        return {"model": model_display, "setting": setting_key, "error": str(e)}

    (OUT_DIR / f"{model_display}__{safe_key}_raw.txt").write_text(raw)
    preds = extract_predictions(raw)
    if preds is None:
        return {"model": model_display, "setting": setting_key, "error": "extract_failed",
                "raw_preview": raw[:300]}

    correct_n = 0
    for p in preds:
        qid = p.get("question_id")
        if qid in correct and p.get("predicted_letter") == correct[qid]:
            correct_n += 1
    total = len(questions)
    result = {
        "model": model_display, "setting": setting_key, "family": family, "dataset_id": dataset_id,
        "correct": correct_n, "total": total, "accuracy": correct_n / total if total else 0,
        "elapsed_seconds": elapsed, "n_predictions": len(preds),
    }
    print(f"  [{model_display}] {setting_key}: {correct_n}/{total} ({elapsed:.1f}s)")
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sanitized = load_sanitized()
    correct = load_correct()
    groups = group_by_setting(sanitized)
    client = OpenAI(api_key=provider_api_key(), base_url=provider_base_url(default=BASE_URL))

    models = [("gpt-5.5", "gpt-5.5-high", 16384), ("claude-opus-4-8-high", "claude-opus-4.8-high", 16384)]

    all_results = []
    for api_name, display_name, max_tokens in models:
        print(f"\n=== {display_name} ===")
        for setting_key, questions in sorted(groups.items()):
            result = run_one(client, api_name, display_name, setting_key, questions, correct, max_tokens)
            all_results.append(result)
            time.sleep(1)

    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))

    print(f"\n{'=' * 60}\nSUMMARY (isolated per-setting batch, 10Q each)\n{'=' * 60}")
    for display_name in ["gpt-5.5-high", "claude-opus-4.8-high"]:
        rs = [r for r in all_results if r.get("model") == display_name and "accuracy" in r]
        total_c = sum(r["correct"] for r in rs)
        total_t = sum(r["total"] for r in rs)
        print(f"{display_name}: {total_c}/{total_t} = {total_c/total_t:.4f}" if total_t else f"{display_name}: no results")
        for r in rs:
            print(f"    {r['setting']:55s} {r['correct']}/{r['total']}")


if __name__ == "__main__":
    main()
