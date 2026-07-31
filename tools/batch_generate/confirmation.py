"""Re-run question candidates on fresh seeds without touching original GT.

The confirmation path deliberately follows the repository invariant for every
unique candidate::

    candidate_spec.json -> write_candidate(temp) -> run_ground_truth(temp)

Only the returned ``summary.json`` payload is copied into the confirmation
index.  The temporary candidate directory (including its ``results`` folder)
is removed before the worker returns, so the source candidate's results are
never rewritten.

Examples::

    python tools/batch_generate/confirmation.py \
      data/datasets/.../questions/run_8q_4c_abcdef \
      --base-seed 10000 --n-seeds 20 \
      --output artifacts/confirmation.json

    python tools/batch_generate/confirmation.py question_paths.json \
      --base-seed 20000 --n-seeds 20 --workers 12 \
      --output artifacts/confirmation.json

Sources may be question-run directories, question directories,
``question.json`` files, or JSON/text files containing paths to any of those.
Relative paths inside a list are resolved relative to the list file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4


Worker = Callable[[str, str, str, int, int, str | None], dict[str, Any]]


def _pin_single_thread() -> None:
    """Pin numerical libraries before torch/numpy are imported in a worker."""

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _clone_profile(profile_name: str, *, base_seed: int, n_seeds: int) -> Any:
    """Load and clone a profile, changing only confirmation GT seed settings."""

    from architecture_iq.profile import load_profile

    profile = deepcopy(load_profile(profile_name))
    ground_truth = deepcopy(profile.ground_truth)
    ground_truth["base_seed"] = int(base_seed)
    ground_truth["n_seeds"] = int(n_seeds)
    profile.ground_truth = ground_truth

    # Keep ``raw`` coherent for callers that inspect it instead of the dataclass
    # field.  No other profile settings (especially significance thresholds or
    # max_failed_seeds) are changed.
    profile.raw = deepcopy(profile.raw)
    profile.raw["ground_truth"] = deepcopy(ground_truth)
    return profile


def _confirm_candidate_worker(
    candidate_path_str: str,
    dataset_path_str: str,
    profile_name: str,
    base_seed: int,
    n_seeds: int,
    temp_root_str: str | None,
) -> dict[str, Any]:
    """Confirm one candidate through the canonical generated-code GT path."""

    _pin_single_thread()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before parallel work has started.
        # The process-level environment and intra-op setting are still pinned.
        pass

    from architecture_iq.candidates.generator import write_candidate
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.registry import ensure_registries, get_model_type
    from architecture_iq.util import read_json

    ensure_registries()
    candidate_path = Path(candidate_path_str).resolve()
    dataset_path = Path(dataset_path_str).resolve()
    candidate_spec = read_json(candidate_path / "candidate_spec.json")
    profile = _clone_profile(
        profile_name,
        base_seed=base_seed,
        n_seeds=n_seeds,
    )
    model_family = get_model_type(candidate_spec["model"]["type"])

    started = time.perf_counter()
    temp_root = Path(temp_root_str).resolve() if temp_root_str else None
    prefix = f"architecture_iq_confirm_{candidate_spec['candidate_id']}_"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temp_root) as temporary:
        temp_candidate = Path(temporary) / candidate_spec["candidate_id"]
        write_candidate(candidate_spec, temp_candidate, model_family)
        summary = run_ground_truth(temp_candidate, profile, dataset_path)

    return {
        "candidate_path": str(candidate_path),
        "candidate_id": candidate_spec["candidate_id"],
        "dataset_path": str(dataset_path),
        "profile": profile_name,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    }


def _question_paths_from_run(run_path: Path) -> list[Path]:
    manifest_path = run_path / "run.json"
    if not manifest_path.is_file():
        raise ValueError(f"Not a question run (missing run.json): {run_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    question_ids = manifest.get("question_ids")
    if question_ids:
        paths = [run_path / str(question_id) / "question.json" for question_id in question_ids]
    else:
        paths = sorted(run_path.glob("*/question.json"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Question run references missing file: {missing[0]}")
    return paths


def collect_question_paths(sources: list[str | Path]) -> list[Path]:
    """Expand runs/questions/list files into de-duplicated question paths."""

    questions: list[Path] = []
    seen_questions: set[Path] = set()
    visited_sources: set[Path] = set()

    def add_question(path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Question file does not exist: {resolved}")
        if resolved not in seen_questions:
            seen_questions.add(resolved)
            questions.append(resolved)

    def expand_reference(reference: str | Path, base: Path) -> None:
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if path in visited_sources:
            return
        visited_sources.add(path)

        if path.is_dir():
            if (path / "question.json").is_file():
                add_question(path / "question.json")
                return
            for question_path in _question_paths_from_run(path):
                add_question(question_path)
            return

        if not path.is_file():
            raise FileNotFoundError(f"Confirmation source does not exist: {path}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "choices" in payload:
                add_question(path)
                return
            if isinstance(payload, dict) and "question_ids" in payload and "run_id" in payload:
                for question_path in _question_paths_from_run(path.parent):
                    add_question(question_path)
                return

            if isinstance(payload, list):
                references = payload
            elif isinstance(payload, dict):
                references = None
                for key in ("questions", "question_paths", "sources", "runs"):
                    if key in payload:
                        references = payload[key]
                        break
                if references is None:
                    raise ValueError(
                        f"JSON list must contain questions/question_paths/sources/runs: {path}"
                    )
            else:
                raise ValueError(f"Unsupported JSON confirmation source: {path}")

            if not isinstance(references, list):
                raise TypeError(f"Question references must be a list: {path}")
            for item in references:
                if isinstance(item, str):
                    expand_reference(item, path.parent)
                elif isinstance(item, dict):
                    reference_value = item.get("question_path") or item.get("path")
                    if not reference_value:
                        raise ValueError(f"Question reference has no path: {item!r}")
                    expand_reference(str(reference_value), path.parent)
                else:
                    raise TypeError(f"Unsupported question reference: {item!r}")
            return

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                expand_reference(stripped, path.parent)

    cwd = Path.cwd()
    for source in sources:
        expand_reference(source, cwd)
    if not questions:
        raise ValueError("No questions found in the supplied sources")
    return questions


def _resolve_candidate_path(choice: dict[str, Any], data_root: Path) -> Path:
    path = Path(choice["candidate_path"]).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def _seed_ranges_overlap(base_a: int, count_a: int, base_b: int, count_b: int) -> bool:
    return base_a < base_b + count_b and base_b < base_a + count_a


def _failed_summary(summary: dict[str, Any]) -> bool:
    return bool(summary.get("excluded")) or int(summary.get("failed_seeds", 0)) > 0


def _choice_record(choice: dict[str, Any], candidate_path: str) -> dict[str, Any]:
    return {
        "letter": choice["letter"],
        "candidate_id": choice["candidate_id"],
        "candidate_path": candidate_path,
    }


def _validate(
    summaries: list[dict[str, Any]],
    profile: Any,
    metric: str,
) -> tuple[dict[str, Any] | None, str | None]:
    from architecture_iq.significance.validator import validate_significance

    try:
        return asdict(validate_significance(summaries, profile, metric=metric)), None
    except Exception as exc:  # noqa: BLE001 - surfaced in the durable index
        return None, f"{type(exc).__name__}: {exc}"


def _environment_record() -> dict[str, Any]:
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None

    from architecture_iq.paths import ROOT
    from architecture_iq.util import git_commit_hash

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch": torch_version,
        "worker_device": "cpu",
        "git_commit": git_commit_hash(ROOT),
    }


def _prepare_inputs(
    question_paths: list[Path],
    *,
    data_root: Path,
    profile_override: str | None,
    base_seed: int,
    n_seeds: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from architecture_iq.util import read_json

    question_inputs: list[dict[str, Any]] = []
    candidate_inputs: dict[str, dict[str, Any]] = {}
    original_seed_ranges: set[tuple[int, int]] = set()

    for question_path in question_paths:
        question = read_json(question_path)
        profile_name = profile_override or str(question.get("profile", "v1"))
        choices: list[dict[str, Any]] = []
        dataset_paths: set[str] = set()

        for choice in question["choices"]:
            candidate_path = _resolve_candidate_path(choice, data_root)
            candidate_key = str(candidate_path)
            candidate_spec = read_json(candidate_path / "candidate_spec.json")
            if candidate_spec["candidate_id"] != choice["candidate_id"]:
                raise ValueError(
                    f"Choice {choice['candidate_id']} does not match candidate spec at "
                    f"{candidate_path}"
                )
            if candidate_spec["dataset_id"] != question["dataset_id"]:
                raise ValueError(f"Candidate dataset mismatch at {candidate_path}")
            if candidate_spec["family"] != question["family"]:
                raise ValueError(f"Candidate family mismatch at {candidate_path}")

            # Standard candidate layout is dataset/candidates/set/candidate.
            # The dataset family loader remains responsible for materialized
            # filenames; this only locates the dataset instance directory.
            dataset_path = candidate_path.parents[2].resolve()
            if not (dataset_path / "dataset_spec.json").is_file():
                raise FileNotFoundError(f"Missing dataset_spec.json at {dataset_path}")
            dataset_paths.add(str(dataset_path))
            summary_path = candidate_path / "results" / "summary.json"
            original_summary = read_json(summary_path)

            previous = candidate_inputs.get(candidate_key)
            if previous is not None:
                if previous["dataset_path"] != str(dataset_path):
                    raise ValueError(f"Candidate resolves to multiple datasets: {candidate_path}")
                if previous["profile"] != profile_name:
                    raise ValueError(
                        f"Candidate {candidate_path} appears under multiple profiles; "
                        "pass --profile to choose one confirmation profile"
                    )
                previous["question_paths"].append(str(question_path))
            else:
                candidate_inputs[candidate_key] = {
                    "candidate_path": candidate_key,
                    "candidate_id": candidate_spec["candidate_id"],
                    "dataset_path": str(dataset_path),
                    "profile": profile_name,
                    "question_paths": [str(question_path)],
                    "original_summary_path": str(summary_path.resolve()),
                    "original_summary": original_summary,
                }

            choices.append(
                {
                    "letter": choice["letter"],
                    "candidate_id": choice["candidate_id"],
                    "candidate_path": candidate_key,
                }
            )

            original_base = int(original_summary["base_seed"])
            original_count = int(original_summary["n_seeds"])
            original_seed_ranges.add((original_base, original_count))

        if len(dataset_paths) != 1:
            raise ValueError(f"Question spans multiple datasets: {question_path}")

        evaluation = question.get("evaluation", {})
        if "base_seed" in evaluation and "n_seeds" in evaluation:
            original_seed_ranges.add(
                (int(evaluation["base_seed"]), int(evaluation["n_seeds"]))
            )
        question_inputs.append(
            {
                "question_path": str(question_path.resolve()),
                "question_id": question["question_id"],
                "profile": profile_name,
                "selection_metric": question.get("evaluation", {}).get(
                    "selection_metric", "test_mse"
                ),
                "correct_letter": question["correct_letter"],
                "choices": choices,
            }
        )

    overlaps = [
        (original_base, original_count)
        for original_base, original_count in sorted(original_seed_ranges)
        if _seed_ranges_overlap(base_seed, n_seeds, original_base, original_count)
    ]
    if overlaps:
        ranges = ", ".join(f"[{base}, {base + count})" for base, count in overlaps)
        raise ValueError(
            f"Confirmation seeds [{base_seed}, {base_seed + n_seeds}) overlap "
            f"original GT seeds: {ranges}"
        )
    return question_inputs, candidate_inputs


def _run_workers(
    candidate_inputs: dict[str, dict[str, Any]],
    *,
    base_seed: int,
    n_seeds: int,
    workers: int,
    temp_root: Path | None,
    worker_fn: Worker,
    show_progress: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}
    jobs = list(candidate_inputs.values())
    temp_root_str = str(temp_root.resolve()) if temp_root is not None else None

    def arguments(job: dict[str, Any]) -> tuple[str, str, str, int, int, str | None]:
        return (
            job["candidate_path"],
            job["dataset_path"],
            job["profile"],
            base_seed,
            n_seeds,
            temp_root_str,
        )

    def record_error(candidate_key: str, exc: BaseException) -> None:
        errors[candidate_key] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    if workers == 1:
        for index, job in enumerate(jobs, start=1):
            candidate_key = job["candidate_path"]
            try:
                results[candidate_key] = worker_fn(*arguments(job))
            except Exception as exc:  # noqa: BLE001 - recorded and returned nonzero
                record_error(candidate_key, exc)
            if show_progress:
                status = "ERROR" if candidate_key in errors else "ok"
                print(f"[confirm {index}/{len(jobs)}] {job['candidate_id']}: {status}", flush=True)
        return results, errors

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker_fn, *arguments(job)): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            candidate_key = job["candidate_path"]
            try:
                results[candidate_key] = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded and returned nonzero
                record_error(candidate_key, exc)
            if show_progress:
                status = "ERROR" if candidate_key in errors else "ok"
                print(f"[confirm {index}/{len(jobs)}] {job['candidate_id']}: {status}", flush=True)
    return results, errors


def _assemble_question_results(
    question_inputs: list[dict[str, Any]],
    candidate_inputs: dict[str, dict[str, Any]],
    confirmations: dict[str, dict[str, Any]],
    worker_errors: dict[str, dict[str, str]],
    *,
    base_seed: int,
    n_seeds: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    profile_cache: dict[str, Any] = {}

    for question in question_inputs:
        question_key = question["question_path"]
        choices = question["choices"]
        original_by_letter = {choice["letter"]: choice for choice in choices}
        original_choice = original_by_letter.get(question["correct_letter"])
        original_winner = (
            _choice_record(original_choice, original_choice["candidate_path"])
            if original_choice is not None
            else None
        )
        candidate_keys = [choice["candidate_path"] for choice in choices]
        failed_candidates = [
            key
            for key in candidate_keys
            if key in worker_errors
            or (
                key in confirmations
                and _failed_summary(confirmations[key]["summary"])
            )
        ]

        profile_name = question["profile"]
        if profile_name not in profile_cache:
            profile_cache[profile_name] = _clone_profile(
                profile_name,
                base_seed=base_seed,
                n_seeds=n_seeds,
            )
        profile = profile_cache[profile_name]
        original_summaries = [
            candidate_inputs[key]["original_summary"] for key in candidate_keys
        ]
        original_sig, original_validation_error = _validate(
            original_summaries,
            profile,
            question["selection_metric"],
        )

        confirmation_sig: dict[str, Any] | None = None
        confirmation_validation_error: str | None = None
        confirmation_winner: dict[str, Any] | None = None
        if all(key in confirmations for key in candidate_keys):
            confirmation_summaries = [
                confirmations[key]["summary"] for key in candidate_keys
            ]
            confirmation_sig, confirmation_validation_error = _validate(
                confirmation_summaries,
                profile,
                question["selection_metric"],
            )
            if confirmation_sig is not None and confirmation_sig["winner_index"] >= 0:
                winner_choice = choices[int(confirmation_sig["winner_index"])]
                confirmation_winner = _choice_record(
                    winner_choice,
                    winner_choice["candidate_path"],
                )

        validation_errors = [
            error
            for error in (original_validation_error, confirmation_validation_error)
            if error is not None
        ]
        if validation_errors:
            errors[question_key] = "; ".join(validation_errors)

        winner_matches = bool(
            original_winner is not None
            and confirmation_winner is not None
            and original_winner["candidate_path"] == confirmation_winner["candidate_path"]
        )
        failed = bool(failed_candidates or validation_errors)
        confirmation_significant = bool(
            confirmation_sig is not None and confirmation_sig["passed"]
        )
        results[question_key] = {
            "question_id": question["question_id"],
            "question_path": question_key,
            "selection_metric": question["selection_metric"],
            "candidate_paths": candidate_keys,
            "original_winner": original_winner,
            "confirmation_winner": confirmation_winner,
            "winner_matches": winner_matches,
            "validate_significance": {
                "original": original_sig,
                "confirmation": confirmation_sig,
            },
            "validation_errors": validation_errors,
            "failed_candidates": failed_candidates,
            "failed": failed,
            "confirmed": not failed and confirmation_significant and winner_matches,
        }
    return results, errors


def run_confirmation(
    question_paths: list[Path],
    *,
    data_root: Path,
    base_seed: int,
    n_seeds: int,
    workers: int,
    profile_override: str | None = None,
    temp_root: Path | None = None,
    worker_fn: Worker | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run confirmation and return an in-memory, JSON-serializable index."""

    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    data_root = data_root.resolve()
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    question_inputs, candidate_inputs = _prepare_inputs(
        question_paths,
        data_root=data_root,
        profile_override=profile_override,
        base_seed=base_seed,
        n_seeds=n_seeds,
    )
    confirmations, worker_errors = _run_workers(
        candidate_inputs,
        base_seed=base_seed,
        n_seeds=n_seeds,
        workers=workers,
        temp_root=temp_root,
        worker_fn=worker_fn or _confirm_candidate_worker,
        show_progress=show_progress,
    )
    question_results, question_errors = _assemble_question_results(
        question_inputs,
        candidate_inputs,
        confirmations,
        worker_errors,
        base_seed=base_seed,
        n_seeds=n_seeds,
    )

    candidate_results: dict[str, dict[str, Any]] = {}
    for candidate_key, candidate in candidate_inputs.items():
        confirmation = confirmations.get(candidate_key)
        confirmation_summary = confirmation["summary"] if confirmation else None
        candidate_results[candidate_key] = {
            "candidate_id": candidate["candidate_id"],
            "candidate_path": candidate_key,
            "dataset_path": candidate["dataset_path"],
            "profile": candidate["profile"],
            "question_paths": candidate["question_paths"],
            "original": {
                "summary_path": candidate["original_summary_path"],
                "summary": candidate["original_summary"],
                "failed": _failed_summary(candidate["original_summary"]),
            },
            "confirmation": (
                {
                    "summary": confirmation_summary,
                    "elapsed_seconds": confirmation["elapsed_seconds"],
                    "failed": _failed_summary(confirmation_summary),
                }
                if confirmation is not None
                else None
            ),
            "worker_error": worker_errors.get(candidate_key),
            "failed": candidate_key in worker_errors
            or (confirmation_summary is not None and _failed_summary(confirmation_summary)),
        }

    completed_at = datetime.now(timezone.utc)
    elapsed = time.perf_counter() - started
    confirmed_questions = sum(
        1 for question in question_results.values() if question["confirmed"]
    )
    changed_winners = sum(
        1
        for question in question_results.values()
        if question["confirmation_winner"] is not None and not question["winner_matches"]
    )
    technical_error = bool(worker_errors or question_errors)
    return {
        "schema_version": "architecture_iq.confirmation.v1",
        "status": "error" if technical_error else "complete",
        "configuration": {
            "base_seed": base_seed,
            "n_seeds": n_seeds,
            "workers": workers,
            "profile_override": profile_override,
            "data_root": str(data_root),
        },
        "environment": _environment_record(),
        "timing": {
            "started_at": started_at.replace(microsecond=0).isoformat(),
            "completed_at": completed_at.replace(microsecond=0).isoformat(),
            "elapsed_seconds": elapsed,
        },
        "summary": {
            "questions": len(question_results),
            "unique_candidates": len(candidate_results),
            "confirmed_questions": confirmed_questions,
            "changed_winners": changed_winners,
            "failed_candidates": sum(
                1 for candidate in candidate_results.values() if candidate["failed"]
            ),
            "worker_errors": len(worker_errors),
            "question_errors": len(question_errors),
        },
        # Full, resolved candidate paths are the keys by design.  Candidate IDs
        # alone are not globally unique across arbitrary source lists.
        "candidates": candidate_results,
        "questions": question_results,
        "worker_errors": worker_errors,
        "question_errors": question_errors,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON index without exposing a partial file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_confirmation_to_file(
    question_paths: list[Path],
    *,
    output_path: Path,
    data_root: Path,
    base_seed: int,
    n_seeds: int,
    workers: int,
    profile_override: str | None = None,
    temp_root: Path | None = None,
    worker_fn: Worker | None = None,
    show_progress: bool = False,
) -> int:
    index = run_confirmation(
        question_paths,
        data_root=data_root,
        base_seed=base_seed,
        n_seeds=n_seeds,
        workers=workers,
        profile_override=profile_override,
        temp_root=temp_root,
        worker_fn=worker_fn,
        show_progress=show_progress,
    )
    atomic_write_json(output_path, index)
    return 1 if index["worker_errors"] or index["question_errors"] else 0


def _parser() -> argparse.ArgumentParser:
    from architecture_iq.paths import DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        help="Question-run directories, question paths, or JSON/text path lists",
    )
    parser.add_argument("--output", required=True, type=Path, help="Confirmation index JSON")
    parser.add_argument("--base-seed", required=True, type=int, help="Fresh seed range start")
    parser.add_argument("--n-seeds", required=True, type=int, help="Fresh seeds per candidate")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes (default: all but one logical CPU)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Override each question's profile for confirmation",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=None,
        help="Optional parent for per-candidate temporary directories",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _pin_single_thread()
    args = _parser().parse_args(argv)
    question_paths = collect_question_paths(args.sources)
    print(
        f"[confirmation] {len(question_paths)} questions; discovering unique candidates",
        flush=True,
    )
    exit_code = run_confirmation_to_file(
        question_paths,
        output_path=args.output,
        data_root=args.data_root,
        base_seed=args.base_seed,
        n_seeds=args.n_seeds,
        workers=args.workers,
        profile_override=args.profile,
        temp_root=args.temp_root,
        show_progress=True,
    )
    print(f"[index] {args.output.resolve()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
