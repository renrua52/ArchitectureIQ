"""Named question runs under a dataset instance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile
from architecture_iq.util import read_json, short_hash, write_json

RUN_MANIFEST = "run.json"


def questions_base_dir(dataset_path: Path) -> Path:
    return dataset_path / "questions"


def question_run_dir(dataset_path: Path, run_name: str) -> Path:
    return questions_base_dir(dataset_path) / run_name


def question_in_run_dir(run_path: Path, question_id: str) -> Path:
    return run_path / question_id


def make_run_name(
    *,
    num_questions: int,
    num_choices: int,
    candidate_set_names: list[str],
    salt: Any,
) -> str:
    suffix = short_hash(
        {
            "num_questions": num_questions,
            "num_choices": num_choices,
            "candidate_sets": sorted(candidate_set_names),
            "salt": salt,
        }
    )
    return f"run_{num_questions}q_{num_choices}c_{suffix}"


def write_run_manifest(
    run_path: Path,
    *,
    run_name: str,
    profile: Profile,
    dataset_id: str,
    family: str,
    candidate_set_paths: list[Path],
    num_questions: int,
    num_choices: int,
    seed: int,
    question_ids: list[str],
) -> None:
    data_root = DATA_DIR.resolve()
    manifest = {
        "schema_version": profile.schema_version,
        "run_id": run_name,
        "dataset_id": dataset_id,
        "family": family,
        "candidate_sets": [
            str(p.resolve().relative_to(data_root)) for p in candidate_set_paths
        ],
        "num_questions": num_questions,
        "num_choices": num_choices,
        "question_ids": question_ids,
        "seed": seed,
        "profile": profile.name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    write_json(run_path / RUN_MANIFEST, manifest)


def list_question_runs(dataset_path: Path) -> list[Path]:
    base = questions_base_dir(dataset_path)
    if not base.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in base.iterdir()
        if p.is_dir() and (p / RUN_MANIFEST).is_file()
    )


def list_questions_in_run(run_path: Path) -> list[Path]:
    if not run_path.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in run_path.iterdir()
        if p.is_dir() and (p / "question.json").is_file()
    )


def load_run_manifest(run_path: Path) -> dict[str, Any]:
    return read_json(run_path / RUN_MANIFEST)
