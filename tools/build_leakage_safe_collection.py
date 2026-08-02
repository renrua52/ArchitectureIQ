#!/usr/bin/env python3
"""Build candidate-disjoint support/holdout collections from question runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "leakage_safe_collection_v1"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    family: str
    dataset_id: str
    candidate_ids: tuple[str, ...]
    question_dir: Path
    prompt_text: str
    choices: tuple[dict[str, str], ...]

    def audit_payload(self, split: str) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "family": self.family,
            "dataset_id": self.dataset_id,
            "candidate_ids": list(self.candidate_ids),
            "source_question_dir": str(self.question_dir),
            "split": split,
        }

    def public_payload(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "family": self.family,
            "dataset_id": self.dataset_id,
            "prompt": self.prompt_text,
            "choices": [dict(choice) for choice in self.choices],
        }


def _load_question(question_dir: Path) -> QuestionRecord:
    question_dir = question_dir.resolve()
    question = read_json(question_dir / "question.json")
    prompt_rel = question.get("prompt", {}).get("rendered_path", "prompt.txt")
    prompt_path = question_dir / prompt_rel
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prompt for {question_dir}: {prompt_path}")

    choices = tuple(
        {
            "letter": str(choice["letter"]).upper(),
            "candidate_id": str(choice["candidate_id"]),
        }
        for choice in question["choices"]
    )
    candidate_ids = tuple(choice["candidate_id"] for choice in choices)
    if not candidate_ids:
        raise ValueError(f"Question has no candidates: {question_dir}")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"Question repeats a candidate_id internally: {question_dir}")

    return QuestionRecord(
        question_id=str(question.get("question_id", question_dir.name)),
        family=str(question["family"]),
        dataset_id=str(question["dataset_id"]),
        candidate_ids=candidate_ids,
        question_dir=question_dir,
        prompt_text=prompt_path.read_text(encoding="utf-8"),
        choices=choices,
    )


def load_question_runs(question_runs: Iterable[Path]) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    for run_path in question_runs:
        run_path = run_path.resolve()
        if not (run_path / "run.json").is_file():
            raise FileNotFoundError(f"Missing run.json: {run_path}")
        question_dirs = sorted(
            path
            for path in run_path.iterdir()
            if path.is_dir() and (path / "question.json").is_file()
        )
        if not question_dirs:
            raise ValueError(f"Question run contains no questions: {run_path}")
        records.extend(_load_question(path) for path in question_dirs)
    return records


def select_candidate_disjoint(
    records: Iterable[QuestionRecord],
    *,
    limit: int | None = None,
) -> list[QuestionRecord]:
    selected: list[QuestionRecord] = []
    used_candidate_ids: set[str] = set()
    for record in records:
        ids = set(record.candidate_ids)
        if ids.isdisjoint(used_candidate_ids):
            selected.append(record)
            used_candidate_ids.update(ids)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def _records_by_dataset(
    records: Iterable[QuestionRecord], rng: random.Random
) -> dict[str, list[QuestionRecord]]:
    grouped: dict[str, list[QuestionRecord]] = {}
    for record in records:
        grouped.setdefault(record.dataset_id, []).append(record)
    for items in grouped.values():
        rng.shuffle(items)
        items[:] = select_candidate_disjoint(items)
    return grouped


def _split_id(
    records: list[QuestionRecord],
    *,
    support_count: int,
    holdout_count: int,
    rng: random.Random,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    grouped = _records_by_dataset(records, rng)
    eligible = [dataset_id for dataset_id, items in grouped.items() if len(items) >= 2]
    rng.shuffle(eligible)

    max_datasets = min(support_count, holdout_count, len(eligible))
    for dataset_count in range(1, max_datasets + 1):
        for dataset_ids in combinations(eligible, dataset_count):
            reserved_support = [grouped[dataset_id][0] for dataset_id in dataset_ids]
            holdout_pool = [
                record
                for dataset_id in dataset_ids
                for record in grouped[dataset_id][1:]
            ]
            holdout = select_candidate_disjoint(holdout_pool, limit=holdout_count)
            if len(holdout) < holdout_count:
                continue

            used_ids = {
                candidate_id
                for record in (*reserved_support, *holdout)
                for candidate_id in record.candidate_ids
            }
            remaining = [
                record
                for record in records
                if set(record.candidate_ids).isdisjoint(used_ids)
                and record not in reserved_support
                and record not in holdout
            ]
            rng.shuffle(remaining)
            support_extra = select_candidate_disjoint(
                remaining, limit=support_count - len(reserved_support)
            )
            support = reserved_support + support_extra
            if len(support) == support_count:
                rng.shuffle(support)
                rng.shuffle(holdout)
                return support, holdout

    raise ValueError(
        "Cannot build an ID split with candidate-disjoint questions and dataset "
        "coverage from support to holdout. Generate more questions per dataset."
    )


def _split_ood(
    records: list[QuestionRecord],
    *,
    support_count: int,
    holdout_count: int,
    rng: random.Random,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    grouped = _records_by_dataset(records, rng)
    dataset_ids = list(grouped)
    rng.shuffle(dataset_ids)
    if len(dataset_ids) < 2:
        raise ValueError("OOD split requires at least two dataset_ids")

    for support_dataset_count in range(1, len(dataset_ids)):
        for support_dataset_ids in combinations(dataset_ids, support_dataset_count):
            support_set = set(support_dataset_ids)
            support_pool = [
                record
                for dataset_id in support_dataset_ids
                for record in grouped[dataset_id]
            ]
            holdout_pool = [
                record
                for dataset_id in dataset_ids
                if dataset_id not in support_set
                for record in grouped[dataset_id]
            ]
            support = select_candidate_disjoint(support_pool, limit=support_count)
            holdout = select_candidate_disjoint(holdout_pool, limit=holdout_count)
            if len(support) == support_count and len(holdout) == holdout_count:
                rng.shuffle(support)
                rng.shuffle(holdout)
                return support, holdout

    raise ValueError(
        "Cannot build an OOD split with disjoint dataset_ids and the requested counts."
    )


def split_collection(
    records: list[QuestionRecord],
    *,
    support_count: int,
    holdout_count: int,
    split_mode: str,
    seed: int,
) -> tuple[list[QuestionRecord], list[QuestionRecord]]:
    if support_count < 1 or holdout_count < 1:
        raise ValueError("support_count and holdout_count must both be positive")
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    if split_mode == "id":
        return _split_id(
            records,
            support_count=support_count,
            holdout_count=holdout_count,
            rng=rng,
        )
    if split_mode == "ood":
        return _split_ood(
            records,
            support_count=support_count,
            holdout_count=holdout_count,
            rng=rng,
        )
    raise ValueError(f"Unknown split_mode: {split_mode!r}")


def build_collection(
    question_runs: list[Path],
    output_dir: Path,
    *,
    support_count: int,
    holdout_count: int,
    split_mode: str,
    seed: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    records = load_question_runs(question_runs)
    support, holdout = split_collection(
        records,
        support_count=support_count,
        holdout_count=holdout_count,
        split_mode=split_mode,
        seed=seed,
    )
    collection_records = [
        *(record.audit_payload("support") for record in support),
        *(record.audit_payload("holdout") for record in holdout),
    ]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "split_mode": split_mode,
        "questions": [
            {
                "question_id": item["question_id"],
                "dataset_id": item["dataset_id"],
                "candidate_ids": item["candidate_ids"],
                "split": item["split"],
            }
            for item in collection_records
        ],
    }
    collection_id = "lc_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    manifest = {
        **identity,
        "collection_id": collection_id,
        "candidate_reuse_policy": "globally_disjoint",
        "dataset_policy": (
            "holdout_datasets_seen_in_support"
            if split_mode == "id"
            else "support_holdout_dataset_disjoint"
        ),
        "source_runs": [str(path.resolve()) for path in question_runs],
        "support_question_ids": [record.question_id for record in support],
        "holdout_question_ids": [record.question_id for record in holdout],
        "records": collection_records,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "collection.json", manifest)
    write_json(output_dir / "support.json", [record.public_payload() for record in support])
    write_json(output_dir / "holdout.json", [record.public_payload() for record in holdout])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-count", type=int, required=True)
    parser.add_argument("--holdout-count", type=int, required=True)
    parser.add_argument("--split-mode", choices=("id", "ood"), default="id")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_collection(
            list(args.question_run),
            args.output,
            support_count=args.support_count,
            holdout_count=args.holdout_count,
            split_mode=args.split_mode,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
