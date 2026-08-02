#!/usr/bin/env python3
"""Build an ordered Inspector review collection from question-run manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_run_questions(run_dir: Path, data_root: Path) -> list[Path]:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise ValueError(f"Question run is missing run.json: {run_dir}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    question_ids = manifest.get("question_ids")
    if not isinstance(question_ids, list) or not question_ids:
        raise ValueError(f"Question run has no question_ids: {run_dir}")

    questions: list[Path] = []
    for question_id in question_ids:
        if not isinstance(question_id, str):
            raise ValueError(f"Non-string question id in {manifest_path}: {question_id!r}")
        question_dir = (run_dir / question_id).resolve()
        if not question_dir.is_relative_to(data_root) or not (question_dir / "question.json").is_file():
            raise ValueError(f"Question artifact is missing or outside data/: {question_dir}")
        questions.append(question_dir)
    return questions


def build_collection(root: Path, run_values: list[str], *, title: str) -> dict[str, Any]:
    data_root = (root / "data").resolve()
    question_paths: list[str] = []
    source_runs: list[str] = []
    seen: set[Path] = set()
    for value in run_values:
        run_dir = _resolve_from_root(root, value)
        for question_dir in _read_run_questions(run_dir, data_root):
            if question_dir in seen:
                raise ValueError(f"Question appears in more than one supplied run: {question_dir}")
            seen.add(question_dir)
            question_paths.append(question_dir.relative_to(data_root).as_posix())
        source_runs.append(run_dir.relative_to(data_root).as_posix())
    return {
        "schema_version": "question_review_collection_v1",
        "title": title,
        "question_paths": question_paths,
        "source_runs": source_runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question-run",
        action="append",
        required=True,
        help="Question-run directory; repeat in the desired review order.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path, relative to repo root.")
    parser.add_argument(
        "--title",
        default="ArchitectureIQ question review",
        help="Human-readable collection title.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    try:
        collection = build_collection(root, args.question_run, title=args.title)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot build review collection: {exc}")
        return 1

    output = _resolve_from_root(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(collection['question_paths'])} questions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())