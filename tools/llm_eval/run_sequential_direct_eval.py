#!/usr/bin/env python
"""Sequential feedback loop for Claude-Opus-4.8-high via api.gpt.ge.

This mirrors tools/llm_eval/run_sequential_codex_eval.py but uses the
OpenAI-compatible api.gpt.ge endpoint directly instead of codex exec,
because the phybench provider doesn't expose claude models.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI
from provider_config import provider_api_key, provider_base_url

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = REPO_ROOT / "artifacts" / "setting3x3_eval" / "sequential"

MODEL_NAME = "claude-opus-4-8-medium"
MAX_TOKENS = 32768


def client():
    return OpenAI(
        base_url=provider_base_url(default="https://api.gpt.ge/v1"),
        api_key=provider_api_key(),
    )


SYS_PROMPT = """You are a blind-answer benchmark agent for ArchitectureIQ. You are participating in a sequential-feedback quiz session.

Rules:
- In each turn I will show you the current question (JSON) and optionally up to 8 prior lessons you recorded.
- The question includes choices labeled with letters A, B, C, ... Each choice has model architecture, optimizer, loss, and budget information.
- You must choose ONE letter and give a one-sentence reason.
- After every answer I will reveal the correct letter and the ground-truth mean metrics for each choice. Use this feedback to update your mental model and write a one-sentence lesson that may help future questions.
- Do NOT ask for clarification. Do NOT output code. Do NOT run simulations. Only respond with valid JSON.

Output strictly the following JSON format (no markdown, no extra text):
{
  "letter": "A",
  "reason": "one short sentence explaining the choice",
  "lesson": "one short sentence summarizing what the feedback taught you"
}
"""


def run_cli(*args):
    return subprocess.run(
        ["python3", "tools/sequential_feedback_session.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def parse_json_response(content: str) -> dict | None:
    raw = content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].strip()
    # find outermost JSON object by first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_one(setting_dir: Path) -> dict:
    setting_key = setting_dir.name
    session_path = setting_dir / f"session_{MODEL_NAME}.json"
    summary_path = setting_dir / f"summary_{MODEL_NAME}.json"
    questions_path = setting_dir / "questions.json"
    feedback_path = setting_dir / "feedback.json"

    if session_path.exists():
        session_path.unlink()

    run_cli(
        "init",
        "--session", str(session_path),
        "--questions", str(questions_path),
        "--feedback", str(feedback_path),
        "--experiment", f"setting3x3_sequential_{MODEL_NAME}_{setting_key}",
        "--force",
    )

    c = client()
    messages = [{"role": "system", "content": SYS_PROMPT}]

    while True:
        cur = json.loads(run_cli("current", "--session", str(session_path)).stdout)
        if cur.get("done"):
            break

        question_text = json.dumps(cur, ensure_ascii=False, indent=2)
        messages.append({"role": "user", "content": question_text})

        parsed = None
        attempt = 0
        last_content = ""
        while parsed is None and attempt < 3:
            attempt += 1
            resp = c.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            last_content = content
            messages.append({"role": "assistant", "content": content})
            parsed = parse_json_response(content)
            if parsed is None:
                fix_msg = "Your previous response was not valid JSON. Please output ONLY the required JSON object with keys letter, reason, lesson."
                messages.append({"role": "user", "content": fix_msg})
        if parsed is None:
            raise ValueError(f"Failed to get valid JSON after {attempt} attempts. Last content:\n{last_content}")

        letter = parsed["letter"].strip().upper()
        reason = parsed["reason"]

        ans = json.loads(
            run_cli(
                "answer",
                "--session", str(session_path),
                "--letter", letter,
                "--reason", reason,
            ).stdout
        )

        lesson = parsed.get("lesson", "")
        if lesson:
            run_cli("lesson", "--session", str(session_path), "--lesson", lesson)

        # feed the feedback back as a user message so the model sees it in next turn
        feedback_text = json.dumps({
            "feedback": ans["feedback"],
            "lesson_recorded": lesson,
        }, ensure_ascii=False, indent=2)
        messages.append({"role": "user", "content": feedback_text})

    summary = json.loads(
        run_cli(
            "summary",
            "--session", str(session_path),
            "--output", str(summary_path),
        ).stdout
    )
    print(f"{setting_key}: {summary['correct_count']}/{summary['answered_questions']} = {summary['overall_accuracy']:.2f}")
    return {"setting": setting_key, "summary": summary}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    setting_dirs = sorted(SEQ_DIR.iterdir())
    if only:
        setting_dirs = [d for d in setting_dirs if only in d.name]

    all_results = []
    for d in setting_dirs:
        if not d.is_dir():
            continue
        all_results.append(run_one(d))

    out_path = SEQ_DIR / f"all_results_{MODEL_NAME}.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    total_c = sum(r["summary"]["correct_count"] for r in all_results)
    total_t = sum(r["summary"]["answered_questions"] for r in all_results)
    print(f"\n=== TOTAL {MODEL_NAME}: {total_c}/{total_t} = {total_c/total_t:.4f} ===")


if __name__ == "__main__":
    main()
