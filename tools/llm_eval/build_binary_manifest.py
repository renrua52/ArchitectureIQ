#!/usr/bin/env python3
"""Build a public, candidate-disjoint binary-evaluation manifest from question runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(data_root: Path, run_paths: list[Path]) -> dict:
    data_root = data_root.resolve()
    seen_candidates: dict[str, str] = {}
    seen_questions: set[str] = set()
    questions: list[dict] = []
    sources: list[str] = []

    for run_path in run_paths:
        run_path = run_path.resolve()
        run = _json(run_path / "run.json")
        if int(run.get("num_choices", 0)) != 2:
            raise ValueError(f"{run_path} is not a binary question run")
        if run.get("non_repeating_candidates") is not True:
            raise ValueError(f"{run_path} does not attest candidate-disjoint generation")
        try:
            sources.append(str(run_path.relative_to(data_root)))
        except ValueError as exc:
            raise ValueError(f"{run_path} is outside data root {data_root}") from exc

        for question_id in run["question_ids"]:
            question_dir = run_path / question_id
            question = _json(question_dir / "question.json")
            prompt_path = question_dir / question.get("prompt", {}).get("rendered_path", "prompt.txt")
            if question.get("question_id") != question_id:
                raise ValueError(f"Question ID mismatch in {question_dir}")
            if len(question.get("choices", [])) != 2 or not prompt_path.is_file():
                raise ValueError(f"Invalid binary question artifact {question_dir}")
            if question_id in seen_questions:
                raise ValueError(f"Duplicate question ID {question_id}")
            seen_questions.add(question_id)
            for choice in question["choices"]:
                candidate_path = str(choice["candidate_path"])
                if candidate_path in seen_candidates:
                    raise ValueError(
                        f"Candidate {candidate_path} is reused by {seen_candidates[candidate_path]} "
                        f"and {question_id}"
                    )
                seen_candidates[candidate_path] = question_id
            questions.append(
                {
                    "question_id": question_id,
                    "question_path": str(question_dir.relative_to(data_root)),
                    "prompt_sha256": _prompt_sha256(prompt_path),
                    "family": question["family"],
                    "dataset_id": question["dataset_id"],
                    "type": question["type"],
                }
            )

    return {
        "schema_version": "architecture_iq.binary_eval.v1",
        "question_count": len(questions),
        "choice_count": 2,
        "random_baseline": 0.5,
        "candidate_reuse": "forbidden",
        "sources": sources,
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_paths", nargs="+", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.data_root, args.run_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(manifest['questions'])} binary, candidate-disjoint questions to {args.output}"
    )


if __name__ == "__main__":
    main()
