#!/usr/bin/env python3
"""Validate candidate-disjoint leakage-safe collection artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "correct_letter",
    "correct_candidate_id",
    "choice_mean_metrics",
    "ground_truth",
    "significance",
    "evaluation",
    "candidate_path",
    "candidate_set_path",
    "source_question_dir",
}
FORBIDDEN_PROMPT_MARKERS = (
    "correct_letter",
    "correct_candidate_id",
    "choice_mean_metrics",
    "results/summary.json",
    "results\\summary.json",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_forbidden_keys(payload: Any, location: str = "$<root>") -> list[str]:
    violations: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{location}.{key}"
            if key in FORBIDDEN_PUBLIC_KEYS:
                violations.append(child)
            violations.extend(_find_forbidden_keys(value, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            violations.extend(_find_forbidden_keys(value, f"{location}[{index}]"))
    return violations


def validate_collection(collection_dir: Path) -> dict[str, Any]:
    collection_dir = collection_dir.resolve()
    manifest_path = collection_dir / "collection.json"
    support_path = collection_dir / "support.json"
    holdout_path = collection_dir / "holdout.json"
    for path in (manifest_path, support_path, holdout_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    support_public = read_json(support_path)
    holdout_public = read_json(holdout_path)
    if manifest.get("schema_version") != "leakage_safe_collection_v1":
        raise ValueError("Unsupported or missing collection schema_version")
    if manifest.get("candidate_reuse_policy") != "globally_disjoint":
        raise ValueError("Collection must declare candidate_reuse_policy=globally_disjoint")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Collection manifest must contain non-empty records")

    seen_questions: dict[str, str] = {}
    seen_candidates: dict[str, str] = {}
    support_ids: list[str] = []
    holdout_ids: list[str] = []
    support_datasets: set[str] = set()
    holdout_datasets: set[str] = set()
    source_public_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        question_id = str(record["question_id"])
        split = str(record["split"])
        dataset_id = str(record["dataset_id"])
        candidate_ids = [str(item) for item in record["candidate_ids"]]
        if split not in {"support", "holdout"}:
            raise ValueError(f"Invalid split for {question_id}: {split!r}")
        if question_id in seen_questions:
            raise ValueError(
                f"Duplicate question_id {question_id!r} in {seen_questions[question_id]} "
                f"and {split}"
            )
        seen_questions[question_id] = split
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Question {question_id} has missing or repeated candidate_ids")
        for candidate_id in candidate_ids:
            previous = seen_candidates.get(candidate_id)
            if previous is not None:
                raise ValueError(
                    f"candidate_id {candidate_id!r} is reused by {previous} and {question_id}"
                )
            seen_candidates[candidate_id] = question_id

        source_dir = Path(record["source_question_dir"])
        if not (source_dir / "question.json").is_file():
            raise FileNotFoundError(f"Missing source question for {question_id}: {source_dir}")
        source_question = read_json(source_dir / "question.json")
        if str(source_question.get("question_id", source_dir.name)) != question_id:
            raise ValueError(f"Manifest question_id does not match source for {question_id}")
        if str(source_question.get("dataset_id")) != dataset_id:
            raise ValueError(
                f"Manifest dataset_id does not match source for {question_id}"
            )
        source_ids = [str(choice["candidate_id"]) for choice in source_question["choices"]]
        if source_ids != candidate_ids:
            raise ValueError(f"Manifest candidate_ids do not match source for {question_id}")
        prompt_rel = source_question.get("prompt", {}).get("rendered_path", "prompt.txt")
        if not (source_dir / prompt_rel).is_file():
            raise FileNotFoundError(f"Missing source prompt for {question_id}")

        source_public_by_id[question_id] = {
            "dataset_id": dataset_id,
            "candidate_ids": source_ids,
            "prompt": (source_dir / prompt_rel).read_text(encoding="utf-8"),
        }

        if split == "support":
            support_ids.append(question_id)
            support_datasets.add(dataset_id)
        else:
            holdout_ids.append(question_id)
            holdout_datasets.add(dataset_id)

    if support_ids != manifest.get("support_question_ids"):
        raise ValueError("support_question_ids do not match manifest records")
    if holdout_ids != manifest.get("holdout_question_ids"):
        raise ValueError("holdout_question_ids do not match manifest records")

    public_by_split = {"support": support_public, "holdout": holdout_public}
    for split, payload in public_by_split.items():
        if not isinstance(payload, list):
            raise ValueError(f"{split}.json must contain a list")
        expected_ids = support_ids if split == "support" else holdout_ids
        actual_ids = [str(item.get("question_id")) for item in payload]
        if actual_ids != expected_ids:
            raise ValueError(f"{split}.json question order does not match collection.json")
        forbidden = _find_forbidden_keys(payload)
        if forbidden:
            raise ValueError(
                f"{split}.json exposes private fields: {', '.join(forbidden[:5])}"
            )
        for item in payload:
            question_id = str(item.get("question_id"))
            expected = source_public_by_id[question_id]
            public_candidate_ids = [
                str(choice.get("candidate_id")) for choice in item.get("choices", [])
            ]
            if str(item.get("dataset_id")) != expected["dataset_id"]:
                raise ValueError(
                    f"Public dataset_id does not match source for {question_id}"
                )
            if public_candidate_ids != expected["candidate_ids"]:
                raise ValueError(
                    f"Public candidate_ids do not match source for {question_id}"
                )
            prompt = str(item.get("prompt", ""))
            if not prompt.strip():
                raise ValueError(f"Public question {question_id} has no prompt")
            lowered = prompt.lower()
            marker = next(
                (value for value in FORBIDDEN_PROMPT_MARKERS if value in lowered),
                None,
            )
            if marker is not None:
                raise ValueError(
                    f"Public prompt {question_id} contains private marker {marker!r}"
                )
            if prompt != expected["prompt"]:
                raise ValueError(f"Public prompt does not match source for {question_id}")

    split_mode = manifest.get("split_mode")
    if split_mode == "id":
        unseen = holdout_datasets - support_datasets
        if unseen:
            raise ValueError(
                "ID holdout contains datasets absent from support: " + ", ".join(sorted(unseen))
            )
    elif split_mode == "ood":
        overlap = support_datasets & holdout_datasets
        if overlap:
            raise ValueError(
                "OOD support/holdout dataset overlap: " + ", ".join(sorted(overlap))
            )
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    return {
        "valid": True,
        "collection_id": manifest.get("collection_id"),
        "split_mode": split_mode,
        "support_questions": len(support_ids),
        "holdout_questions": len(holdout_ids),
        "unique_candidates": len(seen_candidates),
        "support_dataset_ids": sorted(support_datasets),
        "holdout_dataset_ids": sorted(holdout_datasets),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_collection(args.collection_dir)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
