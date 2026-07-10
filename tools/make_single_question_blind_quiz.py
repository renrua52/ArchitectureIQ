#!/usr/bin/env python3
"""Make one-question blind bundles from sanitized ArchitectureIQ quiz artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _format_dict(data: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}- {key}:")
            lines.extend(_format_dict(value, indent + 2))
        else:
            lines.append(f"{pad}- {key}: {value}")
    return lines


def render_prompt(question_number: int, q: dict[str, Any]) -> str:
    letters = [choice["letter"] for choice in q["choices"]]
    lines = [
        "# ArchitectureIQ Single-Question Blind Quiz",
        "",
        "Choose the setting expected to achieve the best final held-out metric. "
        "Do not run experiments, inspect files outside this single-question bundle, "
        "or look for answer keys.",
        "",
        f"Question: q{question_number:02d}",
        f"Family: {q['family']}",
        f"Question type: {q.get('question_type', q.get('type', 'unknown'))}",
        f"Metric: {q['selection_metric']} (lower is better)",
        "",
        "## Dataset params",
    ]
    lines.extend(_format_dict(q.get("dataset_params", {})))
    lines.extend(["", "## Choices"])
    for choice in q["choices"]:
        public = {
            key: value
            for key, value in choice.items()
            if key not in {"candidate_id", "candidate_path", "candidate_set_path"}
        }
        lines.extend(["", f"### Choice {choice['letter']}"])
        lines.extend(_format_dict(public))
    lines.extend(
        [
            "",
            "## Answer format",
            f"Return only JSON: {{\"q{question_number:02d}\": \"{letters[0]}\"}}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def make_bundle(
    source_dir: Path,
    output_dir: Path,
    key_output: Path,
    *,
    force: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    key_output = key_output.resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("Output directory must be outside the source directory.")
    if source_dir.is_relative_to(output_dir):
        raise ValueError("Output directory cannot contain the source directory.")
    if key_output == output_dir or key_output.is_relative_to(output_dir):
        raise ValueError("Private answer key must be outside the public blind bundle.")

    questions = read_json(source_dir / "questions_sanitized.json")
    answer_key_rows = read_json(source_dir / "answer_key.json")
    correct_by_qid = {
        row["question_id"]: row["correct_letter"]
        for row in answer_key_rows
    }
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    private_key: dict[str, str] = {}
    public_manifest: list[dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        blind_id = f"q{idx:02d}"
        qid = q["question_id"]
        qdir = output_dir / blind_id
        qdir.mkdir()
        (qdir / "prompt.md").write_text(render_prompt(idx, q), encoding="utf-8")
        public = {
            "blind_id": blind_id,
            "family": q["family"],
            "question_type": q.get("question_type", q.get("type", "unknown")),
            "selection_metric": q["selection_metric"],
            "num_choices": len(q["choices"]),
            "prompt": f"{blind_id}/prompt.md",
        }
        write_json(qdir / "manifest.json", public)
        public_manifest.append(public)
        private_key[blind_id] = correct_by_qid[qid]

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "single_question_blind_quiz_v1",
            "source": source_dir.name,
            "num_questions": len(public_manifest),
            "questions": public_manifest,
        },
    )
    write_json(key_output, {"answers": private_key})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory after path-safety checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        make_bundle(
            args.source_dir,
            args.output_dir,
            args.key_output,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(args.output_dir.resolve())
    print(args.key_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
