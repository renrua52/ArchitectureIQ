#!/usr/bin/env python
"""Drive 9 sequential-feedback codex sessions (one per setting), each working
through 10 questions one-by-one with real feedback in between, using the
`codex` CLI (phybench provider) as the acting agent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = REPO_ROOT / "artifacts" / "setting3x3_eval" / "sequential"

MODEL = "claude-opus-4-8-high"
MODEL_DISPLAY = "claude-opus-4.8-high"
REASONING_EFFORT = "high"

PROMPT_TEMPLATE = """You are a blind-answer benchmark agent for ArchitectureIQ. Work through a
10-question sequential-feedback session using the CLI tool
`tools/sequential_feedback_session.py`. Run all commands with
`python3 tools/sequential_feedback_session.py ...` from the repo root
({repo_root}).

Session file: {session_path}
Questions file: {questions_path}
Feedback file: {feedback_path}
Experiment name: {experiment_name}

Steps:
1. Run: python3 tools/sequential_feedback_session.py init --session {session_path} --questions {questions_path} --feedback {feedback_path} --experiment "{experiment_name}"
2. Loop until done:
   a. Run: python3 tools/sequential_feedback_session.py current --session {session_path}
      This prints the current question (choices with model/optimizer/loss/budget, no metrics)
      and any prior lessons you recorded. If it says "done": true, stop looping.
   b. Decide your predicted letter using ONLY the visible question JSON and your own
      qualitative reasoning about architectures/optimizers/losses/budgets. Do NOT read any
      other files in the repo (no question.json, no results/, no summary.json, no curves,
      no git history). Do NOT run training or simulations.
   c. Submit: python3 tools/sequential_feedback_session.py answer --session {session_path} --letter <LETTER> --reason "<one short sentence>"
      This reveals the correct answer and metric feedback for that question.
   d. Record what you learned: python3 tools/sequential_feedback_session.py lesson --session {session_path} --lesson "<one short sentence lesson from the feedback>"
   e. Go back to (a) for the next question.
3. When current says done, run:
   python3 tools/sequential_feedback_session.py summary --session {session_path} --output {summary_path}
4. Print the final summary JSON as your last message.

Do not skip steps, do not answer multiple questions at once, and do not fabricate results.
"""


def run_one(setting_dir: Path) -> dict:
    setting_key = setting_dir.name
    session_path = setting_dir / f"session_{MODEL_DISPLAY}.json"
    summary_path = setting_dir / f"summary_{MODEL_DISPLAY}.json"
    questions_path = setting_dir / "questions.json"
    feedback_path = setting_dir / "feedback.json"

    if session_path.exists():
        session_path.unlink()

    prompt = PROMPT_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        session_path=session_path.relative_to(REPO_ROOT),
        questions_path=questions_path.relative_to(REPO_ROOT),
        feedback_path=feedback_path.relative_to(REPO_ROOT),
        experiment_name=f"setting3x3_sequential_{MODEL_DISPLAY}_{setting_key}",
        summary_path=summary_path.relative_to(REPO_ROOT),
    )

    log_path = setting_dir / f"codex_log_{MODEL_DISPLAY}.txt"
    cmd = [
        "codex", "exec",
        "-c", f"model_provider=\"phybench\"",
        "-c", f"model=\"{MODEL}\"",
        "-c", f"model_reasoning_effort=\"{REASONING_EFFORT}\"",
        "--ephemeral",
        "-s", "workspace-write",
        "-C", str(REPO_ROOT),
        prompt,
    ]
    print(f"=== Running codex session for {setting_key} ===", flush=True)
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
        )

    ok = summary_path.exists()
    summary = json.loads(summary_path.read_text()) if ok else None
    print(f"  exit_code={result.returncode} summary_exists={ok}", flush=True)
    if summary:
        print(f"  {summary['correct_count']}/{summary['answered_questions']} = {summary['overall_accuracy']:.2f}", flush=True)
    return {"setting": setting_key, "exit_code": result.returncode, "summary": summary}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    setting_dirs = sorted(SEQ_DIR.iterdir())
    if only:
        setting_dirs = [d for d in setting_dirs if only in d.name]

    all_results = []
    for d in setting_dirs:
        if not d.is_dir():
            continue
        r = run_one(d)
        all_results.append(r)

    out_path = SEQ_DIR / f"all_results_{MODEL_DISPLAY}.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    total_c = sum(r["summary"]["correct_count"] for r in all_results if r["summary"])
    total_t = sum(r["summary"]["answered_questions"] for r in all_results if r["summary"])
    print(f"\n=== TOTAL {MODEL_DISPLAY}: {total_c}/{total_t} = {total_c/total_t:.4f} ===" if total_t else "no results")


if __name__ == "__main__":
    main()
