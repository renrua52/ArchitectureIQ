#!/usr/bin/env python3
"""CLI entrypoint for ArchitectureIQ LLM evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient, ModelConfig  # noqa: E402
from runner import default_run_dir, run_evaluation  # noqa: E402


def _default_questions_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "llm_runs"


def _load_manifest_selection(
    path: Path, questions_root: Path
) -> tuple[list[str] | None, list[Path] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        raise ValueError(f"Question manifest {path} must contain a 'questions' list")

    question_ids: list[str] = []
    for question in questions:
        question_id = question.get("question_id") if isinstance(question, dict) else question
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Question manifest {path} contains an invalid question_id")
        question_ids.append(question_id)
    if not question_ids:
        raise ValueError(f"Question manifest {path} contains no questions")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Question manifest {path} contains duplicate question IDs")

    has_paths = [
        isinstance(question, dict) and "question_path" in question
        for question in questions
    ]
    if any(has_paths) and not all(has_paths):
        raise ValueError(
            f"Question manifest {path} must provide question_path for every question or none"
        )
    if not any(has_paths):
        return question_ids, None

    question_paths: list[Path] = []
    for question in questions:
        assert isinstance(question, dict)
        rel_path = question["question_path"]
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(f"Question manifest {path} contains an invalid question_path")
        resolved = (questions_root / rel_path).resolve()
        try:
            resolved.relative_to(questions_root)
        except ValueError as exc:
            raise ValueError(
                f"Question manifest {path} path escapes questions root: {rel_path}"
            ) from exc
        question_paths.append(resolved)
    return None, question_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM on ArchitectureIQ question prompts and score answers."
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
        "--question-manifest",
        default=None,
        help="Public JSON manifest whose ordered questions[].question_id list selects a subset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent API requests (default: 4; use 1 for sequential)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse cached per-question results already present in the run dir",
    )
    args = parser.parse_args()

    questions_root = Path(args.questions_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    question_ids, question_paths = (
        _load_manifest_selection(
            Path(args.question_manifest).expanduser().resolve(), questions_root
        )
        if args.question_manifest
        else (None, None)
    )
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

    manifest = run_evaluation(
        questions_root=questions_root,
        run_dir=run_dir,
        model_config=config,
        client=client,
        limit=args.limit,
        question_ids=question_ids,
        question_paths=question_paths,
        skip_existing=args.skip_existing,
        workers=args.workers,
    )

    summary = manifest["summary"]
    accuracy = summary["accuracy"]
    acc_text = f"{accuracy:.1%}" if accuracy is not None else "n/a"
    print(f"Run directory: {run_dir}")
    print(
        f"Accuracy: {summary['correct']}/{summary['parsed']} parsed "
        f"({acc_text}); {summary['unparsed']} unparsed"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
