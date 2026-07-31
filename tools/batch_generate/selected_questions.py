"""Build a canonical question run from explicitly selected candidates.

This tool assembles questions only.  It never trains candidates or recomputes
ground truth: the selected candidates must already have ``summary.json`` files.
Compatibility and significance are delegated to
``questions.generator.build_question_record`` and prompts are rendered through
the same ``write_prompt`` path used by normal question generation.

Example plan (paths are resolved relative to the current working directory)::

    {
      "profile": "v1",
      "dataset_path": "data/datasets/univariate_regression/sym_123456",
      "candidate_set_paths": [
        "data/datasets/univariate_regression/sym_123456/candidates/set_..."
      ],
      "seed": 31001,
      "questions": [
        {"candidate_ids": ["c_111111", "c_222222", "c_333333", "c_444444"]},
        {"candidate_paths": ["data/datasets/.../c_555555", "data/datasets/.../c_666666"]}
      ]
    }

Every question in one plan must have the same number of choices.  A question
may also be written as a bare list or use ``{"candidates": [...]}``; in those
forms each reference is treated as a candidate ID when it exactly matches an
ID in the selected sets, and otherwise as a path.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from architecture_iq.candidates.sets import (
    list_candidates_in_set,
    load_set_manifest,
)
from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile, load_profile
from architecture_iq.prompts.renderer import write_prompt
from architecture_iq.questions.generator import build_question_record
from architecture_iq.questions.runs import (
    make_run_name,
    question_in_run_dir,
    question_run_dir,
    write_run_manifest,
)
from architecture_iq.registry import ensure_registries
from architecture_iq.significance.validator import load_summary
from architecture_iq.util import read_json, write_json


def _resolve_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _load_dataset(plan: dict[str, Any], *, base_dir: Path) -> tuple[Path, dict[str, Any]]:
    if "dataset_path" not in plan:
        raise ValueError("Plan is missing dataset_path")
    dataset_path = _resolve_path(
        plan["dataset_path"], base_dir=base_dir, field="dataset_path"
    )
    spec_path = dataset_path / "dataset_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Dataset is missing dataset_spec.json: {dataset_path}")
    dataset_spec = read_json(spec_path)

    expected_path = (
        DATA_DIR
        / "datasets"
        / str(dataset_spec["family"])
        / str(dataset_spec["dataset_id"])
    ).resolve()
    if dataset_path != expected_path:
        raise ValueError(
            f"dataset_path must be the canonical path {expected_path}, got {dataset_path}"
        )
    return dataset_path, dataset_spec


def _load_candidate_sets(
    plan: dict[str, Any],
    *,
    base_dir: Path,
    dataset_path: Path,
    dataset_spec: dict[str, Any],
    profile: Profile,
) -> tuple[list[Path], dict[Path, dict[str, Any]], dict[str, list[Path]]]:
    raw_paths = plan.get("candidate_set_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("candidate_set_paths must be a non-empty list")

    set_paths = [
        _resolve_path(value, base_dir=base_dir, field="candidate_set_paths[]")
        for value in raw_paths
    ]
    if len(set(set_paths)) != len(set_paths):
        raise ValueError("candidate_set_paths contains duplicate paths")

    candidates_by_path: dict[Path, dict[str, Any]] = {}
    candidates_by_id: dict[str, list[Path]] = {}
    expected_parent = (dataset_path / "candidates").resolve()

    for set_path in set_paths:
        if set_path.parent != expected_parent:
            raise ValueError(
                f"Candidate set {set_path} is not inside {expected_parent}"
            )
        manifest_path = set_path / "set.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Candidate set is missing set.json: {set_path}")
        manifest = load_set_manifest(set_path)
        if (
            manifest.get("dataset_id") != dataset_spec["dataset_id"]
            or manifest.get("family") != dataset_spec["family"]
        ):
            raise ValueError(
                f"Candidate set {set_path} belongs to "
                f"{manifest.get('family')}/{manifest.get('dataset_id')}, expected "
                f"{dataset_spec['family']}/{dataset_spec['dataset_id']}"
            )
        if manifest.get("profile") != profile.name:
            raise ValueError(
                f"Candidate set {set_path} uses profile {manifest.get('profile')!r}, "
                f"expected {profile.name!r}"
            )

        completed = list_candidates_in_set(set_path)
        if not completed:
            raise ValueError(f"Candidate set has no completed candidates: {set_path}")
        for candidate_path in completed:
            spec = read_json(candidate_path / "candidate_spec.json")
            if (
                spec.get("dataset_id") != dataset_spec["dataset_id"]
                or spec.get("family") != dataset_spec["family"]
            ):
                raise ValueError(
                    f"Candidate {candidate_path} belongs to "
                    f"{spec.get('family')}/{spec.get('dataset_id')}, expected "
                    f"{dataset_spec['family']}/{dataset_spec['dataset_id']}"
                )
            candidates_by_path[candidate_path] = spec
            candidate_id = str(spec["candidate_id"])
            candidates_by_id.setdefault(candidate_id, []).append(candidate_path)

    return set_paths, candidates_by_path, candidates_by_id


def _question_references(question: Any, *, index: int) -> tuple[str, list[Any]]:
    if isinstance(question, list):
        return "auto", question
    if not isinstance(question, dict):
        raise TypeError(f"questions[{index}] must be a list or JSON object")

    keys = [
        key
        for key in ("candidate_ids", "candidate_paths", "candidates")
        if key in question
    ]
    if len(keys) != 1:
        raise ValueError(
            f"questions[{index}] must contain exactly one of candidate_ids, "
            "candidate_paths, or candidates"
        )
    key = keys[0]
    references = question[key]
    if not isinstance(references, list):
        raise TypeError(f"questions[{index}].{key} must be a list")
    mode = {
        "candidate_ids": "id",
        "candidate_paths": "path",
        "candidates": "auto",
    }[key]
    return mode, references


def _candidate_from_id(
    candidate_id: Any,
    *,
    candidates_by_id: dict[str, list[Path]],
    field: str,
) -> Path:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise TypeError(f"{field} must be a non-empty candidate ID string")
    matches = candidates_by_id.get(candidate_id, [])
    if not matches:
        raise ValueError(
            f"{field}={candidate_id!r} is not in the specified candidate sets"
        )
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"{field}={candidate_id!r} is ambiguous across candidate sets: {rendered}"
        )
    return matches[0]


def _candidate_from_path(
    value: Any,
    *,
    base_dir: Path,
    candidates_by_path: dict[Path, dict[str, Any]],
    field: str,
) -> Path:
    path = _resolve_path(value, base_dir=base_dir, field=field)
    if path not in candidates_by_path:
        raise ValueError(f"{field}={path} is not in the specified candidate sets")
    return path


def _resolve_subsets(
    plan: dict[str, Any],
    *,
    base_dir: Path,
    candidates_by_path: dict[Path, dict[str, Any]],
    candidates_by_id: dict[str, list[Path]],
) -> list[list[Path]]:
    raw_questions = plan.get("questions", plan.get("candidate_subsets"))
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("questions must be a non-empty list of candidate subsets")

    subsets: list[list[Path]] = []
    seen_subsets: set[frozenset[Path]] = set()
    expected_choices: int | None = None
    for question_index, question in enumerate(raw_questions):
        mode, references = _question_references(question, index=question_index)
        if len(references) < 2:
            raise ValueError(f"questions[{question_index}] must have at least two choices")

        selected: list[Path] = []
        for reference_index, reference in enumerate(references):
            field = f"questions[{question_index}][{reference_index}]"
            if mode == "id" or (
                mode == "auto"
                and isinstance(reference, str)
                and reference in candidates_by_id
            ):
                candidate_path = _candidate_from_id(
                    reference,
                    candidates_by_id=candidates_by_id,
                    field=field,
                )
            else:
                candidate_path = _candidate_from_path(
                    reference,
                    base_dir=base_dir,
                    candidates_by_path=candidates_by_path,
                    field=field,
                )

            summary = load_summary(candidate_path)
            failed_seeds = int(summary.get("failed_seeds", 0))
            if summary.get("excluded") or failed_seeds != 0:
                raise ValueError(
                    f"Candidate {candidate_path} is not eligible "
                    f"(excluded={bool(summary.get('excluded'))}, "
                    f"failed_seeds={failed_seeds})"
                )
            selected.append(candidate_path)

        if len(set(selected)) != len(selected):
            raise ValueError(f"questions[{question_index}] contains duplicate candidates")
        if expected_choices is None:
            expected_choices = len(selected)
        elif len(selected) != expected_choices:
            raise ValueError(
                "All questions in one run must have the same number of choices "
                f"({expected_choices} expected, got {len(selected)} at index "
                f"{question_index})"
            )

        subset_key = frozenset(selected)
        if subset_key in seen_subsets:
            raise ValueError(
                f"questions[{question_index}] duplicates an earlier candidate subset"
            )
        seen_subsets.add(subset_key)
        subsets.append(selected)
    return subsets


def build_selected_question_run(
    plan: dict[str, Any],
    profile: Profile,
    *,
    base_dir: Path | None = None,
) -> tuple[Path, list[tuple[dict[str, Any], Path]]]:
    """Validate *plan* and build one canonical, non-overwriting question run."""

    plan = _require_mapping(plan, field="plan")
    base_dir = (base_dir or Path.cwd()).resolve()
    raw_seed = plan.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise TypeError("seed must be an integer")
    seed = raw_seed

    dataset_path, dataset_spec = _load_dataset(plan, base_dir=base_dir)
    set_paths, candidates_by_path, candidates_by_id = _load_candidate_sets(
        plan,
        base_dir=base_dir,
        dataset_path=dataset_path,
        dataset_spec=dataset_spec,
        profile=profile,
    )
    subsets = _resolve_subsets(
        plan,
        base_dir=base_dir,
        candidates_by_path=candidates_by_path,
        candidates_by_id=candidates_by_id,
    )

    rng = random.Random(seed)
    num_questions = len(subsets)
    num_choices = len(subsets[0])
    subset_ids = [
        [candidates_by_path[path]["candidate_id"] for path in subset]
        for subset in subsets
    ]
    run_name = make_run_name(
        num_questions=num_questions,
        num_choices=num_choices,
        candidate_set_names=[path.name for path in set_paths],
        salt={"seed": seed, "candidate_subsets": subset_ids},
    )
    run_path = question_run_dir(dataset_path, run_name)
    data_root = DATA_DIR.resolve()

    # Build every record before creating the run.  This leaves no partial run
    # when canonical compatibility/significance validation rejects a subset.
    records: list[dict[str, Any]] = []
    for subset in subsets:
        record = build_question_record(
            profile,
            dataset_spec=dataset_spec,
            dataset_path=dataset_path,
            candidate_paths=subset,
            candidate_set_paths=set_paths,
            rng=rng,
        )
        record["question_run_id"] = run_name
        record["question_run_path"] = str(run_path.relative_to(data_root))
        records.append(record)

    question_ids = [record["question_id"] for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Selected subsets produced duplicate question IDs")

    # mkdir(..., exist_ok=False) is the no-overwrite boundary.  If writing a
    # newly-created run fails, remove only that incomplete directory so the
    # same validated plan can be retried safely.
    run_path.mkdir(parents=True, exist_ok=False)
    results: list[tuple[dict[str, Any], Path]] = []
    try:
        for record in records:
            question_path = question_in_run_dir(run_path, record["question_id"])
            question_path.mkdir(parents=True, exist_ok=False)
            write_json(question_path / "question.json", record)
            write_prompt(question_path)
            results.append((record, question_path))

        write_run_manifest(
            run_path,
            run_name=run_name,
            profile=profile,
            dataset_id=dataset_spec["dataset_id"],
            family=dataset_spec["family"],
            candidate_set_paths=set_paths,
            num_questions=num_questions,
            num_choices=num_choices,
            seed=seed,
            question_ids=question_ids,
        )
    except BaseException:
        shutil.rmtree(run_path)
        raise
    return run_path, results


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(payload, field="plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a question run from explicitly selected candidates"
    )
    parser.add_argument("--plan", required=True, type=Path, help="JSON selection plan")
    parser.add_argument(
        "--profile",
        help="Profile name (defaults to plan.profile, then v1)",
    )
    args = parser.parse_args(argv)

    plan = load_plan(args.plan.resolve())
    profile_name = args.profile or plan.get("profile", "v1")
    if not isinstance(profile_name, str) or not profile_name:
        raise TypeError("profile must be a non-empty string")

    ensure_registries()
    profile = load_profile(profile_name)
    run_path, results = build_selected_question_run(
        plan,
        profile,
        base_dir=Path.cwd(),
    )
    print(
        json.dumps(
            {
                "run_path": str(run_path),
                "question_ids": [record["question_id"] for record, _ in results],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
