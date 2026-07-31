"""Resumable setting-to-loss meta-model study.

The command deliberately separates the three evaluation phases:

``fit``
    Tunes only on each experiment's 900 training rows, freezes a CV-selected
    champion, and then evaluates every already-fitted method on the locked
    100-row validation split.
``predict-external``
    Reads only the sanitized 60-question bundle and writes hashed, unscored
    predictions.
``score-external``
    Opens the answer key in a separate process and scores the frozen files.

Every fitted method has an independent joblib checkpoint and JSON sidecar.
Rerunning an interrupted command skips checkpoints whose input/code digest
still matches, while stale checkpoints are refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

from tools.meta_model_study.features import load_jsonl
from tools.meta_model_study.metrics import (
    evaluate_predictions,
    ranking_metrics,
    regression_metrics,
    split_half_noise_ceiling,
)
from tools.meta_model_study.models import (
    EnsembleModel,
    FittedModel,
    SearchDefinition,
    fit_definition,
    fit_positive_ensemble,
    search_definitions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data/meta_model/setting_to_loss_60q_id_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/meta_model_studies/setting_to_loss_60q_id_v1"
)
DEFAULT_SANITIZED_QUESTIONS = (
    REPO_ROOT / "artifacts/quiz_attempt_60/questions_sanitized.json"
)
DEFAULT_ANSWER_KEY = REPO_ROOT / "artifacts/quiz_attempt_60/answer_key.json"

SCHEMA_VERSION = "meta_model_study_v1"
EXPERIMENT_FILES = ("train.jsonl", "validation.jsonl", "manifest.json")
MODEL_SOURCE_FILES = (
    "features.py",
    "models.py",
)
StudyModel = FittedModel | EnsembleModel


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_paths(paths: Iterable[Path], extra: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    digest.update(
        json.dumps(_json_safe(extra), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def discover_experiments(dataset_root: Path) -> list[Path]:
    """Return deterministic experiment directories with the full row contract."""

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    experiments = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir()
        and all((path / filename).is_file() for filename in EXPERIMENT_FILES)
    ]
    if not experiments:
        raise ValueError(f"No completed experiments found under {dataset_root}")
    return sorted(experiments, key=lambda path: path.name)


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    expected_split: str,
) -> None:
    if not rows:
        raise ValueError(f"{experiment_id}/{expected_split} is empty")
    fingerprints: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{experiment_id}/{expected_split}[{index}]"
        if row.get("experiment_id") != experiment_id:
            raise ValueError(f"{context} has a mismatched experiment_id")
        if row.get("split") != expected_split:
            raise ValueError(f"{context} has split={row.get('split')!r}")
        if row.get("usable_for_regression") is not True:
            raise ValueError(f"{context} is not usable_for_regression")
        fingerprint = str(row.get("example_fingerprint_sha256", ""))
        is_sha256 = len(fingerprint) == 64 and all(
            character in "0123456789abcdef" for character in fingerprint
        )
        if not is_sha256 or fingerprint in fingerprints:
            raise ValueError(f"{context} has an invalid/duplicate fingerprint")
        fingerprints.add(fingerprint)
        loss = float(row["target"]["mean_loss"])
        log_loss = float(row["target"]["log_mean_loss"])
        if loss <= 0.0 or not math.isfinite(loss):
            raise ValueError(f"{context} has a non-positive/non-finite loss")
        if not math.isclose(math.log(loss), log_loss, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{context} mean_loss/log_mean_loss disagree")
        if not isinstance(row.get("stratum"), str) or not row["stratum"]:
            raise ValueError(f"{context} is missing its predeclared stratum")


def load_experiment_rows(
    experiment_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    experiment_id = experiment_dir.name
    train_rows = load_jsonl(experiment_dir / "train.jsonl")
    validation_rows = load_jsonl(experiment_dir / "validation.jsonl")
    manifest = json.loads((experiment_dir / "manifest.json").read_text("utf-8"))
    _validate_rows(train_rows, experiment_id=experiment_id, expected_split="train")
    _validate_rows(
        validation_rows,
        experiment_id=experiment_id,
        expected_split="validation",
    )
    train_ids = {row["example_fingerprint_sha256"] for row in train_rows}
    validation_ids = {
        row["example_fingerprint_sha256"] for row in validation_rows
    }
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        raise ValueError(f"{experiment_id} train/validation overlap: {len(overlap)}")
    return train_rows, validation_rows, manifest


def make_stratified_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make deterministic folds from the split's predeclared stratum."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    strata = np.asarray([str(row["stratum"]) for row in rows], dtype=object)
    _, counts = np.unique(strata, return_counts=True)
    if counts.size == 0:
        raise ValueError("Cannot make folds from an empty row sequence")
    minimum_count = int(np.min(counts))
    if minimum_count < n_splits:
        raise ValueError(
            f"Every stratum needs at least {n_splits} rows; min={minimum_count}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(rows), dtype=np.int8)
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(dummy, strata)
    ]


def _method_digest(
    experiment_dir: Path,
    definition: SearchDefinition,
    *,
    seed: int,
    n_splits: int,
    include_parameter_count: bool,
) -> str:
    source_dir = Path(__file__).resolve().parent
    files = [experiment_dir / "train.jsonl"] + [
        source_dir / filename for filename in MODEL_SOURCE_FILES
    ]
    return _sha256_paths(
        files,
        {
            "schema_version": SCHEMA_VERSION,
            "method": definition.name,
            "seed": seed,
            "n_splits": n_splits,
            "include_parameter_count": include_parameter_count,
            "sklearn": sklearn.__version__,
        },
    )


def _model_sidecar(model: StudyModel, digest: str, checkpoint: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": model.name,
        "checkpoint": str(checkpoint),
        "input_digest": digest,
        "feature_set": model.feature_set,
        "best_params": model.best_params,
        "interpretable": model.interpretable,
        "cv": {
            "rmse_log": model.cv_rmse_log,
            "mae_log": model.cv_mae_log,
            "r2_log": model.cv_r2_log,
        },
        "search_rows": model.search_rows,
        "written_at": _utc_now(),
    }


def _load_valid_checkpoint(
    checkpoint: Path,
    sidecar: Path,
    *,
    digest: str,
    method: str,
) -> StudyModel | None:
    if not checkpoint.is_file() or not sidecar.is_file():
        return None
    metadata = json.loads(sidecar.read_text("utf-8"))
    if metadata.get("input_digest") != digest or metadata.get("method") != method:
        return None
    model = joblib.load(checkpoint)
    if not isinstance(model, (FittedModel, EnsembleModel)) or model.name != method:
        raise TypeError(f"Invalid checkpoint payload: {checkpoint}")
    return model


def _save_model(
    model: StudyModel,
    *,
    experiment_output: Path,
    digest: str,
) -> None:
    checkpoint = experiment_output / "models" / f"{model.name}.joblib"
    sidecar = experiment_output / "models" / f"{model.name}.json"
    _atomic_joblib_dump(checkpoint, model)
    _atomic_write_json(sidecar, _model_sidecar(model, digest, checkpoint))


def _progress(
    path: Path,
    *,
    experiment_id: str,
    completed: Sequence[str],
    current: str | None,
    total: int,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "completed": list(completed),
            "num_completed": len(completed),
            "num_total": total,
            "current": current,
            "updated_at": _utc_now(),
        },
    )


def _definition_subset(
    definitions: Sequence[SearchDefinition], method_names: set[str] | None
) -> list[SearchDefinition]:
    if method_names is None:
        return list(definitions)
    known = {definition.name for definition in definitions}
    unknown = method_names.difference(known)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
    return [definition for definition in definitions if definition.name in method_names]


def _ensemble_members(models: Sequence[FittedModel], limit: int = 8) -> list[FittedModel]:
    excluded = {"constant_mean", "max_params_heuristic"}
    ordered = sorted(models, key=lambda model: (model.cv_rmse_log, model.name))
    return [model for model in ordered if model.name not in excluded][:limit]


def _evaluate_model(
    model: StudyModel,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    validation_log = model.predict_log(validation_rows)
    cv_metrics = _evaluate_oof_predictions(train_rows, model.oof_predictions)
    validation_metrics = evaluate_predictions(
        validation_rows, validation_log, prediction_space="log"
    )
    return cv_metrics, validation_metrics, validation_log


def _evaluate_oof_predictions(
    rows: Sequence[Mapping[str, Any]], predicted_log: np.ndarray
) -> dict[str, Any]:
    """Evaluate 900-row OOF predictions without enumerating C(900, 3).

    Exact ArchitectureIQ-style three-choice enumeration is reserved for the
    100-row locked holdout.  At 900 rows it would create 121,095,300 triples
    per method and adds no information used for model selection.
    """

    predicted_log = np.asarray(predicted_log, dtype=float)
    true_raw = np.asarray(
        [float(row["target"]["mean_loss"]) for row in rows], dtype=float
    )
    true_log = np.log(true_raw)
    with np.errstate(over="raise", invalid="raise"):
        predicted_raw = np.exp(predicted_log)

    def one(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(np.sum(mask)),
            "raw": regression_metrics(true_raw[mask], predicted_raw[mask]),
            "log": regression_metrics(true_log[mask], predicted_log[mask]),
            "ranking": ranking_metrics(true_raw[mask], predicted_raw[mask]),
            "three_choice": {
                "computed": False,
                "reason": "exact three-choice enumeration is holdout-only",
            },
        }

    all_mask = np.ones(len(rows), dtype=bool)
    eligible_mask = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    return {
        "all": one(all_mask),
        "benchmark_eligible": one(eligible_mask) if np.any(eligible_mask) else None,
    }


def _leaderboard_row(
    model: StudyModel,
    *,
    cv_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    checkpoint: Path,
    selection_eligible: bool,
) -> dict[str, Any]:
    return {
        "method": model.name,
        "feature_set": model.feature_set,
        "interpretable": model.interpretable,
        "selection_eligible": selection_eligible,
        "best_params": model.best_params,
        "checkpoint": str(checkpoint),
        "cv_rmse_log": model.cv_rmse_log,
        "cv_mae_log": model.cv_mae_log,
        "cv_r2_log": model.cv_r2_log,
        "cv": cv_metrics,
        "validation": validation_metrics,
    }


def fit_experiment(
    experiment_dir: Path,
    output_root: Path,
    *,
    jobs: int,
    seed: int,
    n_splits: int,
    method_names: set[str] | None = None,
    force: bool = False,
    include_parameter_count: bool = True,
) -> dict[str, Any]:
    """Fit/checkpoint one experiment and evaluate its locked holdout once."""

    experiment_id = experiment_dir.name
    experiment_output = output_root / "experiments" / experiment_id
    train_rows, validation_rows, dataset_manifest = load_experiment_rows(experiment_dir)
    folds = make_stratified_folds(train_rows, n_splits=n_splits, seed=seed)
    y_log = np.asarray(
        [float(row["target"]["log_mean_loss"]) for row in train_rows], dtype=float
    )
    definitions = _definition_subset(
        search_definitions(seed)
        if include_parameter_count
        else search_definitions(seed, include_parameter_count=False),
        method_names,
    )
    fitted: list[FittedModel] = []
    completed: list[str] = []
    progress_path = experiment_output / "progress.json"

    for definition in definitions:
        digest = _method_digest(
            experiment_dir,
            definition,
            seed=seed,
            n_splits=n_splits,
            include_parameter_count=include_parameter_count,
        )
        checkpoint = experiment_output / "models" / f"{definition.name}.joblib"
        sidecar = experiment_output / "models" / f"{definition.name}.json"
        model = None if force else _load_valid_checkpoint(
            checkpoint, sidecar, digest=digest, method=definition.name
        )
        action = "resume" if model is not None else "fit"
        print(f"[{_utc_now()}] {experiment_id}: {action} {definition.name}", flush=True)
        _progress(
            progress_path,
            experiment_id=experiment_id,
            completed=completed,
            current=definition.name,
            total=len(definitions) + 1,
        )
        if model is None:
            model = fit_definition(
                definition,
                train_rows,
                y_log,
                folds,
                jobs=jobs,
                seed=seed,
            )
            _save_model(model, experiment_output=experiment_output, digest=digest)
        if not isinstance(model, FittedModel):
            raise TypeError(f"Expected FittedModel for {definition.name}")
        fitted.append(model)
        completed.append(model.name)

    if not fitted:
        raise ValueError("At least one base method is required")
    members = _ensemble_members(fitted)
    if len(members) >= 2:
        member_digests = [
            json.loads(
                (
                    experiment_output / "models" / f"{member.name}.json"
                ).read_text("utf-8")
            )["input_digest"]
            for member in members
        ]
        ensemble_digest = hashlib.sha256(
            json.dumps(
                {"members": member_digests, "name": "oof_positive_ensemble"},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        ensemble_path = (
            experiment_output / "models" / "oof_positive_ensemble.joblib"
        )
        ensemble_sidecar = (
            experiment_output / "models" / "oof_positive_ensemble.json"
        )
        ensemble = None if force else _load_valid_checkpoint(
            ensemble_path,
            ensemble_sidecar,
            digest=ensemble_digest,
            method="oof_positive_ensemble",
        )
        if ensemble is None:
            ensemble = fit_positive_ensemble(members, y_log)
            _save_model(
                ensemble,
                experiment_output=experiment_output,
                digest=ensemble_digest,
            )
        if not isinstance(ensemble, EnsembleModel):
            raise TypeError("Invalid ensemble checkpoint")
        all_models: list[StudyModel] = [*fitted, ensemble]
        completed.append(ensemble.name)
    else:
        all_models = list(fitted)

    # The ensemble's combiner is fit on the base OOF matrix, so its apparent
    # meta-fit score is diagnostic and cannot win the pre-holdout CV selection.
    selectable = [model for model in fitted]
    champion = min(selectable, key=lambda model: (model.cv_rmse_log, model.name))
    interpretable = [model for model in selectable if model.interpretable]
    best_interpretable = min(
        interpretable, key=lambda model: (model.cv_rmse_log, model.name)
    )

    leaderboard_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for model in all_models:
        cv_metrics, validation_metrics, validation_log = _evaluate_model(
            model, train_rows, validation_rows
        )
        checkpoint = experiment_output / "models" / f"{model.name}.joblib"
        leaderboard_rows.append(
            _leaderboard_row(
                model,
                cv_metrics=cv_metrics,
                validation_metrics=validation_metrics,
                checkpoint=checkpoint,
                selection_eligible=not isinstance(model, EnsembleModel),
            )
        )
        prediction_records.append(
            {
                "method": model.name,
                "predictions": [
                    {
                        "example_fingerprint_sha256": row[
                            "example_fingerprint_sha256"
                        ],
                        "predicted_log_loss": float(value),
                        "predicted_loss": float(math.exp(float(value))),
                    }
                    for row, value in zip(validation_rows, validation_log)
                ],
            }
        )
    leaderboard_rows.sort(
        key=lambda row: (
            not bool(row["selection_eligible"]),
            float(row["cv"]["all"]["log"]["rmse"]),
            row["method"],
        )
    )
    for rank, row in enumerate(leaderboard_rows, start=1):
        row["rank"] = rank

    noise_path = experiment_output / "noise_ceiling.json"
    if force or not noise_path.is_file():
        noise = split_half_noise_ceiling(validation_rows, experiment_dir.parent)
        _atomic_write_json(noise_path, noise)
    else:
        noise = json.loads(noise_path.read_text("utf-8"))

    try:
        from tools.meta_model_study.interpretation import build_interpretation

        interpretation = build_interpretation(
            train_rows, {model.name: model for model in all_models}
        )
    except ImportError:
        interpretation = {
            "status": "interpretation module unavailable during this run"
        }
    _atomic_write_json(experiment_output / "interpretation.json", interpretation)

    family = str(train_rows[0]["family"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "family": family,
        "dataset_id": train_rows[0]["dataset_id"],
        "selection_metric": train_rows[0]["target"]["selection_metric"],
        "include_parameter_count": include_parameter_count,
        "num_train": len(train_rows),
        "num_validation": len(validation_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "num_train_benchmark_eligible": sum(
            row["target"].get("benchmark_eligible") is True for row in train_rows
        ),
        "num_validation_benchmark_eligible": sum(
            row["target"].get("benchmark_eligible") is True
            for row in validation_rows
        ),
        "cv_champion": champion.name,
        "best_interpretable": best_interpretable.name,
        "ensemble_members": [member.name for member in members],
        "selection_protocol": {
            "target": "log(mean_loss)",
            "folds": n_splits,
            "stratified_by": "row.stratum",
            "champion_selected_by": "minimum train-only CV RMSE in log-loss",
            "validation_used_for_selection": False,
            "ensemble_selection_eligible": False,
            "cv_caveat": (
                "Hyperparameters are selected on the same train-only folds used "
                "for reported selection-CV OOF metrics; the locked holdout and "
                "external blind score are the generalization estimates."
            ),
        },
        "dataset_manifest_config_sha256": dataset_manifest.get("config_sha256"),
        "completed_at": _utc_now(),
    }
    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "family": family,
        "primary_sort": "cv.all.log.rmse",
        "methods": leaderboard_rows,
    }
    _atomic_write_json(experiment_output / "summary.json", summary)
    _atomic_write_json(experiment_output / "leaderboard.json", leaderboard)
    _atomic_write_json(
        experiment_output / "validation_predictions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "methods": prediction_records,
        },
    )
    _progress(
        progress_path,
        experiment_id=experiment_id,
        completed=completed,
        current=None,
        total=len(completed),
    )
    return summary


def fit_study(
    dataset_root: Path,
    output_root: Path,
    *,
    jobs: int,
    seed: int,
    n_splits: int,
    method_names: set[str] | None,
    force: bool,
    include_parameter_count: bool = True,
) -> dict[str, Any]:
    experiments = discover_experiments(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    summaries = []
    for experiment_dir in experiments:
        summaries.append(
            fit_experiment(
                experiment_dir,
                output_root,
                jobs=jobs,
                seed=seed,
                n_splits=n_splits,
                method_names=method_names,
                force=force,
                include_parameter_count=include_parameter_count,
            )
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "seed": seed,
        "n_splits": n_splits,
        "jobs": jobs,
        "methods_filter": sorted(method_names) if method_names else None,
        "include_parameter_count": include_parameter_count,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "experiments": summaries,
    }
    _atomic_write_json(output_root / "study_manifest.json", manifest)
    return manifest


def _load_study_models(
    output_root: Path,
) -> tuple[dict[str, dict[str, StudyModel]], dict[str, dict[str, Any]]]:
    model_map: dict[str, dict[str, StudyModel]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    experiments_root = output_root / "experiments"
    for experiment_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        summary_path = experiment_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text("utf-8"))
        experiment_id = str(summary["experiment_id"])
        summaries[experiment_id] = summary
        models: dict[str, StudyModel] = {}
        for checkpoint in sorted((experiment_dir / "models").glob("*.joblib")):
            model = joblib.load(checkpoint)
            if isinstance(model, (FittedModel, EnsembleModel)):
                models[model.name] = model
        if not models:
            raise ValueError(f"No fitted models found for {experiment_id}")
        model_map[experiment_id] = models
    if not model_map:
        raise ValueError(f"No completed experiment models under {output_root}")
    return model_map, summaries


def _prediction_rows_for_method(
    questions: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[str, StudyModel]],
    method_by_experiment: Mapping[str, str],
) -> list[dict[str, Any]]:
    output = []
    for question in questions:
        experiment_id = str(question["experiment_id"])
        method = method_by_experiment[experiment_id]
        model = models[experiment_id][method]
        choices = list(question["choices"])
        examples = [choice["example"] for choice in choices]
        predicted_log = model.predict_log(examples)
        selected = int(np.argmin(predicted_log))
        output.append(
            {
                "question_id": question["question_id"],
                "family": question["family"],
                "experiment_id": experiment_id,
                "predicted_letter": choices[selected]["letter"],
                "predicted_candidate_id": choices[selected]["candidate_id"],
                "choice_predictions": [
                    {
                        "letter": choice["letter"],
                        "candidate_id": choice["candidate_id"],
                        "predicted_log_loss": float(value),
                    }
                    for choice, value in zip(choices, predicted_log)
                ],
            }
        )
    return output


def predict_external(
    output_root: Path,
    sanitized_questions: Path,
) -> dict[str, Any]:
    """Freeze blind external predictions without opening an answer key."""

    prepared_path = output_root / "external" / "prepared_inputs.json"
    study_manifest_path = output_root / "study_manifest.json"
    study_manifest = (
        json.loads(study_manifest_path.read_text("utf-8"))
        if study_manifest_path.is_file()
        else {}
    )
    include_parameter_count = bool(
        study_manifest.get("include_parameter_count", True)
    )
    prepare_command = [
            sys.executable,
            "-m",
            "tools.meta_model_study.external_prepare",
            "--questions",
            str(sanitized_questions),
            "--output",
            str(prepared_path),
        ]
    if not include_parameter_count:
        prepare_command.append("--exclude-parameter-count")
    subprocess.run(
        prepare_command,
        cwd=REPO_ROOT,
        check=True,
    )
    prepared = json.loads(prepared_path.read_text("utf-8"))
    if prepared.get("questions_sha256") != _sha256_file(sanitized_questions):
        raise ValueError("Prepared external inputs do not match sanitized questions")
    if prepared.get("include_parameter_count", True) != include_parameter_count:
        raise ValueError("Prepared external inputs use the wrong parameter-count policy")
    questions = prepared.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Prepared external input artifact has no questions")

    # Importing this module no longer imports Torch.  The only Torch-using
    # operation (choice parameter counting) happened in the exited helper.
    from tools.meta_model_study.external import write_unscored_predictions

    models, summaries = _load_study_models(output_root)
    experiment_ids = sorted(models)
    question_experiments = {str(question["experiment_id"]) for question in questions}
    if question_experiments != set(experiment_ids):
        raise ValueError(
            "Sanitized questions and fitted experiment sets differ: "
            f"questions={sorted(question_experiments)}, models={experiment_ids}"
        )

    common_methods = set.intersection(
        *(set(models[experiment_id]) for experiment_id in experiment_ids)
    )
    method_maps: dict[str, dict[str, str]] = {
        method: {experiment_id: method for experiment_id in experiment_ids}
        for method in sorted(common_methods)
    }
    method_maps["cv_champion"] = {
        experiment_id: str(summaries[experiment_id]["cv_champion"])
        for experiment_id in experiment_ids
    }
    method_maps["best_interpretable"] = {
        experiment_id: str(summaries[experiment_id]["best_interpretable"])
        for experiment_id in experiment_ids
    }

    unscored_root = output_root / "external" / "unscored"
    artifacts = []
    for method, mapping in sorted(method_maps.items()):
        rows = _prediction_rows_for_method(questions, models, mapping)
        path = unscored_root / f"{method}.json"
        digest = write_unscored_predictions(
            path,
            rows,
            metadata={
                "study_schema_version": SCHEMA_VERSION,
                "method": method,
                "method_by_experiment": mapping,
                "sanitized_questions_path": str(sanitized_questions.resolve()),
                "sanitized_questions_sha256": _sha256_file(sanitized_questions),
                "prepared_inputs_sha256": _sha256_file(prepared_path),
                "answer_key_opened": False,
                "include_parameter_count": include_parameter_count,
                "written_at": _utc_now(),
            },
        )
        artifacts.append(
            {
                "method": method,
                "method_by_experiment": mapping,
                "path": str(path),
                "sha256": digest,
            }
        )
        print(f"froze external predictions: {method} {digest[:12]}", flush=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "unscored_external_predictions",
        "answer_key_opened": False,
        "primary_method": "cv_champion",
        "sanitized_questions_sha256": _sha256_file(sanitized_questions),
        "prepared_inputs_sha256": _sha256_file(prepared_path),
        "num_questions": len(questions),
        "include_parameter_count": include_parameter_count,
        "artifacts": artifacts,
        "completed_at": _utc_now(),
    }
    _atomic_write_json(output_root / "external" / "unscored_manifest.json", manifest)
    return manifest


def score_external(output_root: Path, answer_key: Path) -> dict[str, Any]:
    """Score already-hashed predictions; call only after prediction finishes."""

    from tools.meta_model_study.external import score_predictions

    manifest_path = output_root / "external" / "unscored_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    methods = []
    score_root = output_root / "external" / "scores"
    for artifact in manifest["artifacts"]:
        prediction_path = Path(artifact["path"])
        actual_digest = _sha256_file(prediction_path)
        if actual_digest != artifact["sha256"]:
            raise ValueError(f"Frozen prediction hash changed: {prediction_path}")
        score = score_predictions(prediction_path, answer_key)
        method = str(artifact["method"])
        _atomic_write_json(score_root / f"{method}.json", score)
        methods.append(
            {
                "method": method,
                "method_by_experiment": artifact["method_by_experiment"],
                "predictions_sha256": actual_digest,
                "total": score["total"],
                "by_family": score["by_family"],
                "score_path": str(score_root / f"{method}.json"),
            }
        )
        print(
            f"scored external: {method} "
            f"{score['total']['num_correct']}/{score['total']['num_questions']}",
            flush=True,
        )
    best_posthoc = max(
        methods,
        key=lambda row: (float(row["total"]["accuracy"]), row["method"]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "scored_external_predictions",
        "primary_method": "cv_champion",
        "best_posthoc_method": best_posthoc["method"],
        "posthoc_warning": (
            "best_posthoc_method is a retrospective envelope over many methods; "
            "cv_champion is the pre-answer primary external estimate"
        ),
        "answer_key_path": str(answer_key.resolve()),
        "answer_key_sha256": _sha256_file(answer_key),
        "unscored_manifest_sha256": _sha256_file(manifest_path),
        "methods": sorted(
            methods,
            key=lambda row: (-float(row["total"]["accuracy"]), row["method"]),
        ),
        "completed_at": _utc_now(),
    }
    _atomic_write_json(output_root / "external_score.json", result)
    return result


def _parse_methods(value: str | None) -> set[str] | None:
    if value is None:
        return None
    methods = {item.strip() for item in value.split(",") if item.strip()}
    if not methods:
        raise argparse.ArgumentTypeError("--methods must contain at least one name")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit and validate all experiments")
    fit_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    fit_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    fit_parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    fit_parser.add_argument("--seed", type=int, default=20260713)
    fit_parser.add_argument("--folds", type=int, default=5)
    fit_parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="optional comma-separated subset for a smoke run",
    )
    fit_parser.add_argument("--force", action="store_true")
    fit_parser.add_argument("--exclude-parameter-count", action="store_true")

    predict_parser = subparsers.add_parser(
        "predict-external", help="write blind unscored predictions"
    )
    predict_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    predict_parser.add_argument(
        "--questions", type=Path, default=DEFAULT_SANITIZED_QUESTIONS
    )

    score_parser = subparsers.add_parser(
        "score-external", help="score frozen predictions in a separate phase"
    )
    score_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score_parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY)

    report_parser = subparsers.add_parser(
        "report", help="consolidate completed JSON artifacts into Markdown/JSON"
    )
    report_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        fit_study(
            args.dataset_root.resolve(),
            args.output_root.resolve(),
            jobs=args.jobs,
            seed=args.seed,
            n_splits=args.folds,
            method_names=_parse_methods(args.methods),
            force=args.force,
            include_parameter_count=not args.exclude_parameter_count,
        )
    elif args.command == "predict-external":
        predict_external(args.output_root.resolve(), args.questions.resolve())
    elif args.command == "score-external":
        score_external(args.output_root.resolve(), args.answer_key.resolve())
    elif args.command == "report":
        from tools.meta_model_study.report import generate_report

        generate_report(args.output_root.resolve())
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
