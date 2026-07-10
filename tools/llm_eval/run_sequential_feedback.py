#!/usr/bin/env python3
"""Sequential ArchitectureIQ eval with revealed-answer history in the prompt."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from completion import fetch_model_response  # noqa: E402
from llm_client import LLMClient, ModelConfig  # noqa: E402
from prompt_wrapper import format_eval_prompt  # noqa: E402
from question_loader import QuestionItem, list_questions  # noqa: E402
from runner import (  # noqa: E402
    QuestionResult,
    default_run_dir,
    evaluate_question,
    summarize_results,
)


@dataclass(frozen=True)
class HistoryEntry:
    question_index: int
    question_id: str
    family: str
    question_type: str
    picked_letter: str | None
    correct_letter: str
    correct: bool

    @classmethod
    def from_result(cls, question_index: int, result: QuestionResult) -> "HistoryEntry":
        return cls(
            question_index=question_index,
            question_id=result.question_id,
            family=result.family,
            question_type=result.question_type,
            picked_letter=result.parsed_letter,
            correct_letter=result.ground_truth_letter,
            correct=result.correct,
        )


def _default_questions_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "llm_runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _history_intro() -> str:
    return (
        "## Revealed feedback from earlier questions in this same run\n"
        "The items below are already finished. For each earlier question you can see:\n"
        "- the question id and coarse metadata\n"
        "- your earlier choice\n"
        "- the revealed correct answer\n"
        "- whether your earlier choice was correct\n\n"
        "Use this as feedback on your previous reasoning patterns. The current question "
        "still needs to be answered on its own merits."
    )


def _format_history_block(history: list[HistoryEntry]) -> str:
    if not history:
        return ""

    lines = [_history_intro(), ""]
    for entry in history:
        picked = entry.picked_letter if entry.picked_letter is not None else "UNPARSED"
        outcome = "correct" if entry.correct else "incorrect"
        lines.append(
            f"- Q{entry.question_index}: {entry.question_id} | family={entry.family} | "
            f"type={entry.question_type} | your_choice={picked} | "
            f"correct_answer={entry.correct_letter} | outcome={outcome}"
        )
    return "\n".join(lines)


def _build_prompt(
    item: QuestionItem,
    history: list[HistoryEntry],
    question_index: int,
    total_questions: int,
) -> tuple[str, str]:
    base_prompt = format_eval_prompt(item.prompt_text, item.valid_letters)
    history_block = _format_history_block(history)
    if not history_block:
        return (
            f"Question {question_index} of {total_questions}\n\n"
            f"{base_prompt}",
            "",
        )

    prompt = (
        f"Question {question_index} of {total_questions}\n\n"
        f"{history_block}\n\n"
        "## Current question\n\n"
        f"{base_prompt}"
    )
    return prompt, history_block


def _result_payload(
    result: QuestionResult,
    *,
    question_index: int,
    total_questions: int,
    history_context: str,
) -> dict[str, Any]:
    payload = result.to_dict()
    payload["question_index"] = question_index
    payload["total_questions"] = total_questions
    payload["history_entry_count"] = question_index - 1
    payload["history_context"] = history_context
    return payload


def _write_question_result(
    run_dir: Path,
    result: QuestionResult,
    *,
    question_index: int,
    total_questions: int,
    history_context: str,
) -> Path:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{result.question_id}.json"
    out.write_text(
        json.dumps(
            _result_payload(
                result,
                question_index=question_index,
                total_questions=total_questions,
                history_context=history_context,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _load_cached_result(path: Path) -> QuestionResult:
    cached = json.loads(path.read_text(encoding="utf-8"))
    return QuestionResult(
        question_id=cached["question_id"],
        prompt_hash=cached["prompt_hash"],
        ground_truth_letter=cached["ground_truth_letter"],
        parsed_letter=cached.get("parsed_letter"),
        correct=bool(cached["correct"]),
        model_response=cached["model_response"],
        chain_of_thought=cached.get("chain_of_thought"),
        question_type=cached.get("question_type", "?"),
        family=cached.get("family", "?"),
        eval_prompt=cached.get("eval_prompt", ""),
        finish_reason=cached.get("finish_reason"),
        truncated=bool(cached.get("truncated", False)),
        continuation_count=int(cached.get("continuation_count", 0)),
        usage=cached.get("usage"),
        message_parts=cached.get("message_parts"),
    )


def _write_progress_event(
    progress_path: Path,
    *,
    question_index: int,
    result: QuestionResult,
    summary: dict[str, Any],
) -> None:
    event = {
        "timestamp": _utc_now_iso(),
        "question_index": question_index,
        "question_id": result.question_id,
        "parsed_letter": result.parsed_letter,
        "ground_truth_letter": result.ground_truth_letter,
        "correct": result.correct,
        "summary": summary,
    }
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def _write_manifest(
    run_dir: Path,
    *,
    questions_root: Path,
    model_config: ModelConfig,
    total_questions: int,
    results: list[QuestionResult],
    created_at: str,
    progress_every: int,
) -> dict[str, Any]:
    summary = summarize_results(results)
    manifest = {
        "run_id": run_dir.name,
        "created_at": created_at,
        "updated_at": _utc_now_iso(),
        "questions_root": str(questions_root.resolve()),
        "mode": "sequential_feedback_summary",
        "history_policy": "include all earlier answers as compact revealed-feedback lines",
        "model": model_config.to_dict(),
        "workers": 1,
        "progress_every": progress_every,
        "progress": {
            "completed": len(results),
            "remaining": total_questions - len(results),
            "total_questions": total_questions,
        },
        "summary": summary,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _progress_line(question_index: int, total_questions: int, result: QuestionResult, summary: dict[str, Any]) -> str:
    accuracy = summary["accuracy"]
    acc_text = f"{accuracy:.1%}" if accuracy is not None else "n/a"
    picked = result.parsed_letter or "UNPARSED"
    verdict = "correct" if result.correct else "wrong"
    return (
        f"[{question_index}/{total_questions}] {result.question_id} "
        f"picked={picked} gt={result.ground_truth_letter} {verdict}; "
        f"cumulative={summary['correct']}/{summary['parsed']} ({acc_text}), "
        f"unparsed={summary['unparsed']}"
    )


def run_sequential_feedback(
    *,
    questions_root: Path,
    run_dir: Path,
    model_config: ModelConfig,
    client: LLMClient,
    limit: int | None = None,
    skip_existing: bool = False,
    progress_every: int = 5,
) -> dict[str, Any]:
    items = list_questions(questions_root)
    if limit is not None:
        items = items[:limit]

    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    created_at = _utc_now_iso()
    results: list[QuestionResult] = []
    history: list[HistoryEntry] = []
    total_questions = len(items)

    _write_manifest(
        run_dir,
        questions_root=questions_root,
        model_config=model_config,
        total_questions=total_questions,
        results=results,
        created_at=created_at,
        progress_every=progress_every,
    )

    for question_index, item in enumerate(items, start=1):
        result_path = run_dir / "results" / f"{item.question_id}.json"
        prompt, history_context = _build_prompt(item, history, question_index, total_questions)
        used_cache = skip_existing and result_path.is_file()

        if used_cache:
            result = _load_cached_result(result_path)
        else:
            exchange = fetch_model_response(client, prompt, model_config, item.valid_letters)
            result = evaluate_question(item, exchange, prompt)
            _write_question_result(
                run_dir,
                result,
                question_index=question_index,
                total_questions=total_questions,
                history_context=history_context,
            )

        results.append(result)
        history.append(HistoryEntry.from_result(question_index, result))
        summary = summarize_results(results)

        if not used_cache:
            _write_progress_event(
                progress_path,
                question_index=question_index,
                result=result,
                summary=summary,
            )

        print(_progress_line(question_index, total_questions, result, summary), flush=True)

        if question_index % progress_every == 0 or question_index == total_questions:
            checkpoint_acc = summary["accuracy"]
            checkpoint_acc_text = f"{checkpoint_acc:.1%}" if checkpoint_acc is not None else "n/a"
            print(
                f"Checkpoint {question_index}/{total_questions}: "
                f"{summary['correct']}/{summary['parsed']} correct "
                f"({checkpoint_acc_text})",
                flush=True,
            )

        _write_manifest(
            run_dir,
            questions_root=questions_root,
            model_config=model_config,
            total_questions=total_questions,
            results=results,
            created_at=created_at,
            progress_every=progress_every,
        )

    return _write_manifest(
        run_dir,
        questions_root=questions_root,
        model_config=model_config,
        total_questions=total_questions,
        results=results,
        created_at=created_at,
        progress_every=progress_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ArchitectureIQ questions sequentially, with each new prompt seeing "
            "the revealed answers from earlier questions."
        )
    )
    parser.add_argument(
        "questions_root",
        nargs="?",
        default=str(_default_questions_root()),
        help="Data root, question run dir, or legacy data/questions (default: data)",
    )
    parser.add_argument("--model", required=True, help="Model name passed to the chat API")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--runs-root",
        default=str(_default_runs_root()),
        help="Parent directory for evaluation runs (default: llm_runs)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory for this run (default: auto under --runs-root)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Suffix for auto-generated run directory name",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N questions")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse cached per-question results already present in the run dir",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print an explicit checkpoint every N questions (default: 5)",
    )
    args = parser.parse_args()

    questions_root = Path(args.questions_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = default_run_dir(runs_root, args.model, args.run_id)

    config = ModelConfig(
        name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    client = LLMClient()

    manifest = run_sequential_feedback(
        questions_root=questions_root,
        run_dir=run_dir,
        model_config=config,
        client=client,
        limit=args.limit,
        skip_existing=args.skip_existing,
        progress_every=max(1, args.progress_every),
    )

    summary = manifest["summary"]
    accuracy = summary["accuracy"]
    acc_text = f"{accuracy:.1%}" if accuracy is not None else "n/a"
    print(f"Run directory: {run_dir}")
    print(
        f"Final accuracy: {summary['correct']}/{summary['parsed']} parsed "
        f"({acc_text}); {summary['unparsed']} unparsed"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
