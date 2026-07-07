from __future__ import annotations

import math
import random
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import (
    choices_compatible,
    infer_axes,
    infer_question_type,
)
from architecture_iq.candidates.sets import list_candidates_in_set
from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile
from architecture_iq.questions.runs import (
    make_run_name,
    question_in_run_dir,
    question_run_dir,
    write_run_manifest,
)
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, short_hash, write_json

CandidateProgress = Callable[[int, int, str], None]


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def eligible_candidate_paths(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not load_summary(p).get("excluded")]


def load_candidate_pool_from_sets(set_paths: list[Path]) -> list[Path]:
    pool: list[Path] = []
    for set_path in set_paths:
        pool.extend(list_candidates_in_set(set_path))
    return eligible_candidate_paths(pool)


def _validate_pool_dataset(pool: list[Path], dataset_path: Path) -> None:
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]
    for path in pool:
        spec = read_json(path / "candidate_spec.json")
        if spec["dataset_id"] != dataset_id or spec["family"] != family:
            raise ValueError(
                f"Candidate {path} belongs to {spec['family']}/{spec['dataset_id']}, "
                f"expected {family}/{dataset_id}"
            )


def find_significant_subsets(
    pool: list[Path],
    profile: Profile,
    rng: random.Random,
    *,
    num_choices: int | None = None,
    limit: int | None = None,
    max_attempts: int | None = None,
    question_type: str | None = None,
    selection_metric: str = "test_mse",
) -> list[list[Path]]:
    """Return significant candidate subsets; exhaustive when feasible."""
    num_choices = num_choices if num_choices is not None else profile.num_choices
    if len(pool) < num_choices:
        return []

    summary_map = {p: load_summary(p) for p in pool}
    n_combos = math.comb(len(pool), num_choices)
    max_exhaustive = int(
        profile.question_generation.get("max_exhaustive_combinations", 500_000)
    )

    passing: list[list[Path]] = []

    def _subset_ok(combo: tuple[Path, ...]) -> bool:
        specs = [read_json(p / "candidate_spec.json") for p in combo]
        if not choices_compatible(specs, question_type):
            return False
        sig = validate_significance(
            [summary_map[p] for p in combo],
            profile,
            metric=selection_metric,
        )
        return sig.passed

    if n_combos <= max_exhaustive:
        for combo in combinations(pool, num_choices):
            if _subset_ok(combo):
                passing.append(list(combo))
    else:
        attempts = max_attempts if max_attempts is not None else int(
            profile.question_generation["max_attempts"]
        )
        seen: set[frozenset[str]] = set()
        for _ in range(attempts):
            paths = rng.sample(pool, num_choices)
            key = frozenset(p.name for p in paths)
            if key in seen:
                continue
            seen.add(key)
            if _subset_ok(tuple(paths)):
                passing.append(paths)

    if not passing:
        return []

    rng.shuffle(passing)
    if limit is not None:
        return passing[:limit]
    return passing


def select_significant_candidates(
    pool: list[Path],
    profile: Profile,
    rng: random.Random,
    *,
    num_choices: int | None = None,
    max_attempts: int | None = None,
) -> list[Path] | None:
    subsets = find_significant_subsets(
        pool,
        profile,
        rng,
        num_choices=num_choices,
        limit=1,
        max_attempts=max_attempts,
    )
    return subsets[0] if subsets else None


def _candidate_set_key(paths: list[Path]) -> frozenset[str]:
    return frozenset(p.name for p in paths)


def _pick_distinct_subsets(
    subsets: list[list[Path]],
    num_questions: int,
) -> list[list[Path]]:
    seen: set[frozenset[str]] = set()
    picked: list[list[Path]] = []
    for subset in subsets:
        key = _candidate_set_key(subset)
        if key in seen:
            continue
        seen.add(key)
        picked.append(subset)
        if len(picked) >= num_questions:
            break
    return picked


def _budget_field(specs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {spec["budget"]["total_samples_seen"] for spec in specs}
    if len(totals) == 1:
        return {"total_samples_seen": next(iter(totals))}
    return {"total_samples_seen": sorted(totals), "mixed": True}


def build_question_record(
    profile: Profile,
    *,
    dataset_spec: dict[str, Any],
    dataset_path: Path,
    candidate_paths: list[Path],
    candidate_set_paths: list[Path],
    rng: random.Random,
) -> dict[str, Any]:
    summaries = [load_summary(p) for p in candidate_paths]
    specs = [read_json(p / "candidate_spec.json") for p in candidate_paths]
    if not choices_compatible(specs):
        invariant, varying = infer_axes(specs)
        raise ValueError(
            "Candidates are not compatible for a question "
            f"(invariant={invariant}, varying={varying})"
        )

    question_type = infer_question_type(specs)
    invariant_axes, varying_axes = infer_axes(specs)

    sig = validate_significance(summaries, profile, metric=dataset_spec["selection_metric"])
    if not sig.passed:
        raise ValueError(f"Significance failed: {sig.reason}")

    for path in candidate_paths:
        cand_budget = read_json(path / "candidate_spec.json")["budget"]
        steps = cand_budget["training_steps"]
        bs = cand_budget["batch_size"]
        total = cand_budget["total_samples_seen"]
        if steps * bs != total:
            raise ValueError(
                f"Candidate {path.name} violates batch_size × training_steps = "
                f"total_samples_seen ({bs} × {steps} != {total})"
            )

    winner_path = candidate_paths[sig.winner_index]
    others = [p for i, p in enumerate(candidate_paths) if i != sig.winner_index]
    rng.shuffle(others)
    ordered_paths = [winner_path] + others
    rng.shuffle(ordered_paths)

    letters = _letters(len(ordered_paths))
    choices = []
    correct_letter = "A"
    data_root = DATA_DIR.resolve()
    for letter, path in zip(letters, ordered_paths):
        path = path.resolve()
        spec = read_json(path / "candidate_spec.json")
        choices.append(
            {
                "letter": letter,
                "candidate_id": spec["candidate_id"],
                "candidate_path": str(path.relative_to(data_root)),
                "candidate_set_path": str(path.parent.relative_to(data_root)),
            }
        )
        if path == winner_path:
            correct_letter = letter

    body = {
        "schema_version": profile.schema_version,
        "family": dataset_spec["family"],
        "dataset_id": dataset_spec["dataset_id"],
        "budget": _budget_field(specs),
        "type": question_type,
        "invariant_axes": invariant_axes,
        "varying_axes": varying_axes,
        "candidate_sets": [
            str(p.resolve().relative_to(data_root)) for p in candidate_set_paths
        ],
        "num_choices": len(choices),
        "choices": choices,
        "correct_letter": correct_letter,
        "significance": {
            "passed": sig.passed,
            "gap": sig.gap,
            "win_rate": sig.win_rate,
            "metric": sig.metric,
        },
        "evaluation": {
            "selection_metric": dataset_spec["selection_metric"],
            "n_seeds": profile.n_seeds,
            "base_seed": profile.base_seed,
        },
        "prompt": {
            "template_version": profile.prompts["template_version"],
            "rendered_path": "prompt.txt",
        },
    }
    qid = f"q_{short_hash(body)}"
    body["question_id"] = qid
    body["profile"] = profile.name
    return body


def _write_question(
    profile: Profile,
    *,
    dataset_spec: dict[str, Any],
    dataset_path: Path,
    candidate_paths: list[Path],
    candidate_set_paths: list[Path],
    run_path: Path,
    run_name: str,
    rng: random.Random,
) -> tuple[dict[str, Any], Path]:
    record = build_question_record(
        profile,
        dataset_spec=dataset_spec,
        dataset_path=dataset_path,
        candidate_paths=candidate_paths,
        candidate_set_paths=candidate_set_paths,
        rng=rng,
    )
    data_root = DATA_DIR.resolve()
    record["question_run_id"] = run_name
    record["question_run_path"] = str(run_path.resolve().relative_to(data_root))

    out = question_in_run_dir(run_path, record["question_id"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "question.json", record)
    return record, out


def generate_questions(
    profile: Profile,
    *,
    dataset_path: Path,
    candidate_set_paths: list[Path],
    rng: random.Random,
    num_questions: int = 1,
    num_choices: int | None = None,
    seed: int = 0,
) -> tuple[Path, list[tuple[dict[str, Any], Path]]]:
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    if not candidate_set_paths:
        raise ValueError("At least one candidate set path is required")

    resolved_sets = [p.resolve() for p in candidate_set_paths]
    pool = load_candidate_pool_from_sets(resolved_sets)
    _validate_pool_dataset(pool, dataset_path)

    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    n_choices = num_choices if num_choices is not None else profile.num_choices
    if len(pool) < n_choices:
        raise RuntimeError(
            f"Not enough eligible candidates ({len(pool)}) for {n_choices} choices"
        )

    subsets = find_significant_subsets(
        pool,
        profile,
        rng,
        num_choices=n_choices,
        selection_metric=dataset_spec["selection_metric"],
    )
    if not subsets:
        raise RuntimeError(
            f"Failed to find significant {n_choices}-candidate subsets in pool of {len(pool)}"
        )

    selected_sets = _pick_distinct_subsets(subsets, num_questions)
    if len(selected_sets) < num_questions:
        raise RuntimeError(
            f"Requested {num_questions} distinct questions but only "
            f"{len(selected_sets)} significant subsets exist "
            f"({len(subsets)} total passing subsets)."
        )

    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]

    run_name = make_run_name(
        num_questions=num_questions,
        num_choices=n_choices,
        candidate_set_names=[p.name for p in resolved_sets],
        salt=rng.randint(0, 2**31 - 1),
    )
    run_path = question_run_dir(dataset_path.resolve(), run_name)
    run_path.mkdir(parents=True, exist_ok=False)

    results: list[tuple[dict[str, Any], Path]] = []
    for selected in selected_sets:
        results.append(
            _write_question(
                profile,
                dataset_spec=dataset_spec,
                dataset_path=dataset_path,
                candidate_paths=selected,
                candidate_set_paths=resolved_sets,
                run_path=run_path,
                run_name=run_name,
                rng=rng,
            )
        )

    write_run_manifest(
        run_path,
        run_name=run_name,
        profile=profile,
        dataset_id=dataset_id,
        family=family,
        candidate_set_paths=resolved_sets,
        num_questions=num_questions,
        num_choices=n_choices,
        seed=seed,
        question_ids=[record["question_id"] for record, _ in results],
    )
    return run_path, results
