"""Post-hoc factorial GT study around mistakes on the frozen 60-question test.

This is explicitly an error-analysis dataset, never a new benchmark score.  For
each question missed by the pre-answer ``cv_champion``, it crosses the three
public architectures with the three public optimizer templates and every LR in
the active profile.  Candidate ground truth follows the canonical
``spec -> generated code -> run_ground_truth`` path and uses the same ten seeds.

The resulting 3 x 3 x 5 grid per missed question exposes architecture/optimizer
crossovers that a marginal heuristic or globally fitted meta-model can miss.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.paths import ROOT
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type
from architecture_iq.significance.validator import mean_metric_key
from architecture_iq.util import read_json, write_json
from tools.meta_model_dataset.build import run_ground_truth_for_experiment
from tools.meta_model_dataset.core import full_candidate_fingerprint, sha256_json


SCHEMA_VERSION = "error_neighborhood_v1"
DEFAULT_STUDY_ROOT = (
    ROOT / "data/meta_model_studies/setting_to_loss_60q_id_v1"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data/meta_model/error_neighborhood_v1"
DEFAULT_QUESTIONS = ROOT / "artifacts/quiz_attempt_60/questions_sanitized.json"
DEFAULT_ANSWER_KEY = ROOT / "artifacts/quiz_attempt_60/answer_key.json"


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"Duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def load_missed_questions(
    study_root: Path,
    questions_path: Path,
    answer_key_path: Path,
) -> list[dict[str, Any]]:
    """Join public settings to already-scored primary errors.

    Reading the answer key is allowed only here, after the frozen external
    prediction phase has completed.  Every output is marked post-hoc.
    """

    score_path = study_root / "external/scores/cv_champion.json"
    score = read_json(score_path)
    questions = _index(read_json(questions_path), "question_id")
    answers = _index(read_json(answer_key_path), "question_id")
    missed = []
    for result in score["questions"]:
        if result["is_correct"]:
            continue
        question_id = str(result["question_id"])
        question = questions[question_id]
        answer = answers[question_id]
        missed.append(
            {
                "question_id": question_id,
                "family": question["family"],
                "dataset_id": question["dataset_id"],
                "selection_metric": question["selection_metric"],
                "predicted_letter": result["predicted_letter"],
                "correct_letter": result["correct_letter"],
                "gt_gap": float(answer["gap"]),
                "choices": deepcopy(question["choices"]),
            }
        )
    if not missed:
        raise ValueError("The frozen primary prediction has no missed questions")
    return missed


def _family_config(
    family: str,
    dataset_id: str,
    dataset_path: Path,
    profile_name: str,
) -> dict[str, Any]:
    profile = load_profile(profile_name)
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    if dataset_spec["family"] != family or dataset_spec["dataset_id"] != dataset_id:
        raise ValueError(f"Dataset mismatch at {dataset_path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": f"posthoc_{family}_{dataset_id}",
        "profile": profile.name,
        "profile_config": profile.raw,
        "dataset_path": _portable(dataset_path),
        "dataset_spec": dataset_spec,
        "ground_truth": {
            "n_seeds": 10,
            "base_seed": 0,
            "fail_threshold_mode": "finite_only",
        },
        "posthoc": True,
        "purpose": "factorial error analysis; never an external benchmark score",
    }


def _candidate_sources(
    question: dict[str, Any],
    learning_rates: Sequence[float],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    choices = question["choices"]
    for architecture_choice in choices:
        for optimizer_choice in choices:
            for learning_rate in learning_rates:
                optimizer = deepcopy(optimizer_choice["optimizer"])
                optimizer["lr"] = float(learning_rate)
                source = {
                    "question_id": question["question_id"],
                    "architecture_letter": architecture_choice["letter"],
                    "architecture_candidate_id": architecture_choice["candidate_id"],
                    "optimizer_template_letter": optimizer_choice["letter"],
                    "optimizer_template_candidate_id": optimizer_choice["candidate_id"],
                    "optimizer_type": optimizer["type"],
                    "learning_rate": float(learning_rate),
                    "is_original_diagonal": bool(
                        architecture_choice["letter"] == optimizer_choice["letter"]
                        and math.isclose(
                            float(learning_rate),
                            float(optimizer_choice["optimizer"]["lr"]),
                        )
                    ),
                }
                setting = {
                    "budget": deepcopy(architecture_choice["budget"]),
                    "model": deepcopy(architecture_choice["model"]),
                    "optimizer": optimizer,
                    "loss": deepcopy(architecture_choice["loss"]),
                }
                yield setting, source


def prepare(
    *,
    study_root: Path,
    output_root: Path,
    questions_path: Path,
    answer_key_path: Path,
    profile_name: str = "v1",
) -> dict[str, Any]:
    ensure_registries()
    profile = load_profile(profile_name)
    missed = load_missed_questions(
        study_root, questions_path, answer_key_path
    )
    learning_rates = sorted(float(value) for value in profile.optimizer_grids["lr"])
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in missed:
        by_family[(question["family"], question["dataset_id"])].append(question)

    experiments = []
    total_sources = 0
    total_unique = 0
    for (family, dataset_id), questions in sorted(by_family.items()):
        dataset_path = ROOT / "data/datasets" / family / dataset_id
        config = _family_config(
            family, dataset_id, dataset_path, profile_name
        )
        experiment_dir = output_root / config["experiment_id"]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        records_by_fingerprint: dict[str, dict[str, Any]] = {}
        for question in questions:
            for setting, source in _candidate_sources(question, learning_rates):
                budget = setting["budget"]
                spec = build_candidate_spec(
                    profile,
                    dataset_id=dataset_id,
                    family=family,
                    budget=int(budget["total_samples_seen"]),
                    batch_size=int(budget["batch_size"]),
                    model=setting["model"],
                    optimizer=setting["optimizer"],
                    loss=setting["loss"],
                )
                fingerprint = full_candidate_fingerprint(spec)
                total_sources += 1
                if fingerprint not in records_by_fingerprint:
                    records_by_fingerprint[fingerprint] = {
                        "sampling_index": len(records_by_fingerprint),
                        "fingerprint": fingerprint,
                        "candidate_id_short": spec["candidate_id"],
                        "artifact_dir": (
                            f"candidates/{spec['candidate_id']}__{fingerprint[:16]}"
                        ),
                        "spec": spec,
                        "sources": [],
                    }
                records_by_fingerprint[fingerprint]["sources"].append(source)

        records = list(records_by_fingerprint.values())
        total_unique += len(records)
        config["design"] = {
            "question_ids": [question["question_id"] for question in questions],
            "architecture_choices_per_question": 3,
            "optimizer_templates_per_question": 3,
            "learning_rates": learning_rates,
            "expected_sources_per_question": 9 * len(learning_rates),
        }
        config_hash = sha256_json(config)
        sampling_path = experiment_dir / "sampling_manifest.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config_hash,
            "config": config,
            "records": records,
        }
        if sampling_path.is_file():
            existing = read_json(sampling_path)
            if existing != payload:
                raise ValueError(
                    f"Existing post-hoc design differs: {sampling_path}; use a new output root"
                )
        else:
            write_json(sampling_path, payload)

        for record in records:
            candidate_dir = experiment_dir / record["artifact_dir"]
            write_candidate(
                record["spec"],
                candidate_dir,
                get_model_type(record["spec"]["model"]["type"]),
            )
        experiments.append(
            {
                "experiment_id": config["experiment_id"],
                "family": family,
                "dataset_id": dataset_id,
                "num_questions": len(questions),
                "num_unique_candidates": len(records),
                "sampling_manifest": _portable(sampling_path),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "posthoc": True,
        "frozen_primary_score_path": _portable(
            study_root / "external/scores/cv_champion.json"
        ),
        "questions_path": _portable(questions_path),
        "answer_key_path": _portable(answer_key_path),
        "num_missed_questions": len(missed),
        "question_ids": [question["question_id"] for question in missed],
        "learning_rates": learning_rates,
        "num_factorial_sources": total_sources,
        "num_unique_candidates": total_unique,
        "experiments": experiments,
    }
    write_json(output_root / "manifest.json", manifest)
    print(
        f"[prepare] errors={len(missed)}, factorial_sources={total_sources}, "
        f"unique_candidates={total_unique}",
        flush=True,
    )
    return manifest


def run_gt(output_root: Path, *, workers: int, limit: int | None = None) -> None:
    manifest = read_json(output_root / "manifest.json")
    for experiment in manifest["experiments"]:
        sampling_path = ROOT / experiment["sampling_manifest"]
        sampling = read_json(sampling_path)
        run_ground_truth_for_experiment(
            config=sampling["config"],
            experiment_dir=sampling_path.parent,
            sampling_manifest=sampling,
            workers=workers,
            limit=limit,
        )


def _loss_for_record(
    experiment_dir: Path,
    record: dict[str, Any],
    selection_metric: str,
) -> tuple[float, float, int]:
    summary = read_json(
        experiment_dir / record["artifact_dir"] / "results/summary.json"
    )
    failed_seeds = int(summary.get("failed_seeds", 0))
    if failed_seeds >= int(summary.get("n_seeds", 0)):
        raise ValueError(f"No successful seeds in {record['artifact_dir']}")
    mean = float(summary[mean_metric_key(selection_metric)])
    std = float(summary[f"std_{selection_metric}"])
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"Invalid mean loss in {record['artifact_dir']}")
    return mean, std, failed_seeds


def _question_analysis(
    question: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    architecture_letters = sorted(
        {row["architecture_letter"] for row in source_rows}
    )
    cells = sorted(
        {
            (
                row["optimizer_template_letter"],
                float(row["learning_rate"]),
            )
            for row in source_rows
        }
    )
    lookup = {
        (
            row["architecture_letter"],
            row["optimizer_template_letter"],
            float(row["learning_rate"]),
        ): float(row["mean_loss"])
        for row in source_rows
    }
    matrix = np.asarray(
        [
            [lookup[(architecture, template, learning_rate)] for template, learning_rate in cells]
            for architecture in architecture_letters
        ],
        dtype=float,
    )
    log_matrix = np.log(matrix)
    grand = float(np.mean(log_matrix))
    architecture_effect = np.mean(log_matrix, axis=1) - grand
    cell_effect = np.mean(log_matrix, axis=0) - grand
    interaction = log_matrix - grand - architecture_effect[:, None] - cell_effect[None, :]

    crossover_pairs = []
    for left_index, left in enumerate(architecture_letters):
        for right_index in range(left_index + 1, len(architecture_letters)):
            right = architecture_letters[right_index]
            difference = log_matrix[left_index] - log_matrix[right_index]
            if np.any(difference < 0.0) and np.any(difference > 0.0):
                crossover_pairs.append(
                    {
                        "architectures": [left, right],
                        "num_cells_left_wins": int(np.sum(difference < 0.0)),
                        "num_cells_right_wins": int(np.sum(difference > 0.0)),
                    }
                )

    original = [row for row in source_rows if row["is_original_diagonal"]]
    original.sort(key=lambda row: row["mean_loss"])
    best_index = np.unravel_index(int(np.argmin(log_matrix)), log_matrix.shape)
    best_architecture = architecture_letters[best_index[0]]
    best_template, best_lr = cells[best_index[1]]
    return {
        "question_id": question["question_id"],
        "family": question["family"],
        "correct_letter": question["correct_letter"],
        "predicted_letter": question["predicted_letter"],
        "gt_gap": question["gt_gap"],
        "diagonal_winner": original[0]["architecture_letter"],
        "diagonal_winner_reproduced": bool(
            original[0]["architecture_letter"] == question["correct_letter"]
        ),
        "original_diagonal": [
            {
                "letter": row["architecture_letter"],
                "mean_loss": row["mean_loss"],
                "std_loss": row["std_loss"],
                "failed_seeds": int(row.get("failed_seeds", 0)),
            }
            for row in original
        ],
        "factorial_best": {
            "architecture_letter": best_architecture,
            "optimizer_template_letter": best_template,
            "learning_rate": best_lr,
            "mean_loss": float(matrix[best_index]),
        },
        "architecture_marginal_order": [
            architecture_letters[index]
            for index in np.argsort(np.mean(log_matrix, axis=1))
        ],
        "num_architecture_crossover_pairs": len(crossover_pairs),
        "architecture_crossovers": crossover_pairs,
        "max_abs_log_interaction": float(np.max(np.abs(interaction))),
        "interaction_rmse_log": float(np.sqrt(np.mean(interaction**2))),
    }


def analyze(
    *,
    study_root: Path,
    output_root: Path,
    questions_path: Path,
    answer_key_path: Path,
) -> dict[str, Any]:
    missed = load_missed_questions(study_root, questions_path, answer_key_path)
    questions = {question["question_id"]: question for question in missed}
    manifest = read_json(output_root / "manifest.json")
    source_rows_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_rows = []
    for experiment in manifest["experiments"]:
        sampling_path = ROOT / experiment["sampling_manifest"]
        sampling = read_json(sampling_path)
        experiment_dir = sampling_path.parent
        metric = sampling["config"]["dataset_spec"]["selection_metric"]
        for record in sampling["records"]:
            mean, std, failed_seeds = _loss_for_record(
                experiment_dir, record, metric
            )
            unique_rows.append(
                {
                    "fingerprint": record["fingerprint"],
                    "candidate_id": record["spec"]["candidate_id"],
                    "family": record["spec"]["family"],
                    "mean_loss": mean,
                    "std_loss": std,
                    "failed_seeds": failed_seeds,
                    "sources": record["sources"],
                }
            )
            for source in record["sources"]:
                source_rows_by_question[source["question_id"]].append(
                    {
                        **source,
                        "mean_loss": mean,
                        "std_loss": std,
                        "failed_seeds": failed_seeds,
                    }
                )

    analyses = [
        _question_analysis(question, source_rows_by_question[question_id])
        for question_id, question in sorted(questions.items())
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "posthoc": True,
        "num_questions": len(analyses),
        "num_unique_candidates": len(unique_rows),
        "questions": analyses,
        "aggregate": {
            "candidates_with_failed_seeds": sum(
                row["failed_seeds"] > 0 for row in unique_rows
            ),
            "diagonal_winners_reproduced": sum(
                row["diagonal_winner_reproduced"] for row in analyses
            ),
            "questions_with_any_architecture_crossover": sum(
                row["num_architecture_crossover_pairs"] > 0 for row in analyses
            ),
            "total_architecture_crossover_pairs": sum(
                row["num_architecture_crossover_pairs"] for row in analyses
            ),
            "median_max_abs_log_interaction": float(
                np.median([row["max_abs_log_interaction"] for row in analyses])
            ),
            "median_interaction_rmse_log": float(
                np.median([row["interaction_rmse_log"] for row in analyses])
            ),
        },
    }
    _jsonl_write(output_root / "rows.jsonl", unique_rows)
    write_json(output_root / "analysis.json", result)

    lines = [
        "# Post-hoc error-neighborhood factorial study",
        "",
        "This dataset is constructed after scoring and is not a benchmark result.",
        "",
        "| Question | Family | True | Predicted | Crossover pairs | Max |interaction| | Factorial best |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in analyses:
        best = row["factorial_best"]
        lines.append(
            f"| {row['question_id']} | {row['family']} | {row['correct_letter']} | "
            f"{row['predicted_letter']} | {row['num_architecture_crossover_pairs']} | "
            f"{row['max_abs_log_interaction']:.4f} | "
            f"arch {best['architecture_letter']} + opt {best['optimizer_template_letter']} "
            f"@ {best['learning_rate']:g} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Questions with architecture rank crossover: "
            f"{result['aggregate']['questions_with_any_architecture_crossover']}/{len(analyses)}",
            f"- Total architecture-pair crossovers: "
            f"{result['aggregate']['total_architecture_crossover_pairs']}",
            f"- Candidates with at least one failed seed: "
            f"{result['aggregate']['candidates_with_failed_seeds']}",
            f"- Original diagonal winners reproduced: "
            f"{result['aggregate']['diagonal_winners_reproduced']}/{len(analyses)}",
            f"- Median max absolute log interaction: "
            f"{result['aggregate']['median_max_abs_log_interaction']:.4f}",
            f"- Median interaction RMSE in log loss: "
            f"{result['aggregate']['median_interaction_rmse_log']:.4f}",
            "",
        ]
    )
    (output_root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "gt", "analyze", "all"))
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        "study_root": args.study_root.resolve(),
        "output_root": args.output_root.resolve(),
        "questions_path": args.questions.resolve(),
        "answer_key_path": args.answer_key.resolve(),
    }
    if args.command in {"prepare", "all"}:
        prepare(**paths)
    if args.command in {"gt", "all"}:
        run_gt(paths["output_root"], workers=args.workers, limit=args.limit)
    if args.command in {"analyze", "all"} and args.limit is None:
        analyze(**paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
