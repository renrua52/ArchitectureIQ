"""Named question runs under a dataset instance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile
from architecture_iq.util import read_json, short_hash, write_json

RUN_MANIFEST = "run.json"
DEFAULT_CANDIDATE_REUSE_POLICY = "globally_disjoint_within_run"
CANDIDATE_REUSE_POLICIES = frozenset(
    {
        DEFAULT_CANDIDATE_REUSE_POLICY,
        "blind_pair_unique",
        "sequential_bounded_reuse",
    }
)


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
    candidate_reuse_policy: str = DEFAULT_CANDIDATE_REUSE_POLICY,
    run_purpose: str | None = None,
    canonical_blind_evaluation: bool | None = None,
    max_candidate_uses: int | None = None,
    pair_reuse_policy: str | None = None,
    required_model_types: list[str] | None = None,
    max_winner_model_type_fraction: float | None = None,
    artifact_root: Path | None = None,
) -> None:
    if candidate_reuse_policy not in CANDIDATE_REUSE_POLICIES:
        raise ValueError(f"Unknown candidate reuse policy: {candidate_reuse_policy}")
    data_root = (artifact_root or DATA_DIR).resolve()
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
        "candidate_reuse_policy": candidate_reuse_policy,
        "question_ids": question_ids,
        "seed": seed,
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if candidate_reuse_policy != DEFAULT_CANDIDATE_REUSE_POLICY:
        if run_purpose not in {"review_blind_pool", "review_practice_pool"}:
            raise ValueError("Non-canonical runs require a recognized run_purpose")
        if canonical_blind_evaluation is not False:
            raise ValueError("Non-canonical reuse runs must declare canonical_blind_evaluation=false")
        if pair_reuse_policy != "unique":
            raise ValueError("Non-canonical reuse runs must declare pair_reuse_policy=unique")
        if not required_model_types or len(set(required_model_types)) != num_choices:
            raise ValueError("Non-canonical reuse runs require one distinct model type per choice")
        if candidate_reuse_policy == "sequential_bounded_reuse":
            if not isinstance(max_candidate_uses, int) or isinstance(max_candidate_uses, bool) or max_candidate_uses < 1:
                raise ValueError("sequential_bounded_reuse requires max_candidate_uses >= 1")
        elif max_candidate_uses is not None:
            raise ValueError("blind_pair_unique must not declare max_candidate_uses")
        if max_winner_model_type_fraction is not None:
            if candidate_reuse_policy != "blind_pair_unique":
                raise ValueError("winner-family caps require blind_pair_unique")
            if not 0.5 <= float(max_winner_model_type_fraction) <= 1.0:
                raise ValueError("winner-family cap must be in [0.5, 1.0]")
        manifest.update(
            {
                "run_purpose": run_purpose,
                "canonical_blind_evaluation": False,
                "candidate_reuse_allowed": True,
                "pair_reuse_policy": "unique",
                "required_model_types": sorted(required_model_types),
            }
        )
        if max_candidate_uses is not None:
            manifest["max_candidate_uses"] = max_candidate_uses
        if max_winner_model_type_fraction is not None:
            manifest["max_winner_model_type_fraction"] = float(max_winner_model_type_fraction)
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
