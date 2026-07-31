"""Resumable training and evaluation runner for the wide-v2 meta-model study.

The wide dataset builder owns ground truth.  This module only consumes the
validated exports from :mod:`tools.meta_model_study.wide`, fits meta-models,
and writes reproducible evaluation artifacts.  Four protocols are supported:

``fit-id``
    Per-environment train -> locked-validation evaluation and family-pooled
    train -> locked-validation evaluation.

``fit-grouped``
    Family-conditioned leave-one-environment-out, leave-one-dataset-out, the
    final predeclared ``holdout_candidate`` dataset evaluation, and a true
    leave-one-family-out evaluation.  An outer group is never used for either
    fitting or hyperparameter selection.

All prediction targets are ``log(mean_loss)``.  Hyperparameters are selected
using training-only inner folds through the existing ``fit_definition``
implementation.  Every estimator checkpoint has a digest covering the frozen
wide plan, exact train/test rows and folds, source hashes, and method config.
The runner is deliberately sequential over tasks and caps scikit-learn search
parallelism at four single-threaded workers.
"""

# ruff: noqa: E402 -- thread caps must be set before numerical-library imports.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

# Set these before importing NumPy/scikit-learn.  ``threadpool_limits`` below
# is the runtime backstop when a library was initialized by an earlier import.
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_variable in THREAD_ENVIRONMENT:
    os.environ[_thread_variable] = "1"

import joblib
import numpy as np
import sklearn
from joblib import parallel_backend
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from tools.meta_model_study.features import DATASET_CONDITIONING, FeatureEncoder
from tools.meta_model_study.metrics import (
    log_loss_prediction_metrics,
    loss_prediction_metrics,
    ranking_metrics,
    regression_metrics,
)
from tools.meta_model_study.models import (
    FittedModel,
    SearchDefinition,
    fit_definition,
    search_definitions,
)
from tools.meta_model_study.wide import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_PLAN,
    WideCorpus,
    WideEnvironment,
    group_value,
    load_corpus,
    load_seed_losses,
    load_snapshot,
    validate_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/meta_model_studies/setting_to_loss_wide_v2"
SCHEMA_VERSION = "meta_model_wide_study_v1"
DEFAULT_MAX_JOBS = 4


def _max_jobs_from_environment() -> int:
    """Machine-sharing cap; dedicated hosts may raise it via ARCHIQ_MAX_JOBS."""

    raw = os.environ.get("ARCHIQ_MAX_JOBS")
    if raw is None:
        return DEFAULT_MAX_JOBS
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"ARCHIQ_MAX_JOBS must be an integer, got {raw!r}") from error
    if value < 1:
        raise ValueError(f"ARCHIQ_MAX_JOBS must be >= 1, got {value}")
    return value


MAX_JOBS = _max_jobs_from_environment()
DEFAULT_SEED = 20260714
DEFAULT_FOLDS = 5
ID_SCOPES = ("environment", "dataset", "family", "global")
GROUPED_PROTOCOLS = ("environment", "dataset", "holdout_candidate", "family")
SOURCE_FILES = {
    "wide_loader": Path(__file__).with_name("wide.py"),
    "wide_runner": Path(__file__),
    "feature_encoder": Path(__file__).with_name("features.py"),
    "model_estimators": Path(__file__).with_name("models.py"),
}


@dataclass(frozen=True)
class FoldPlan:
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    manifest: dict[str, Any]


@dataclass
class TaskResult:
    task_id: str
    rows: list[dict[str, Any]]
    predictions: dict[str, np.ndarray]
    leaderboard: dict[str, Any]


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
    if isinstance(value, BaseEstimator):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def validate_jobs(jobs: int) -> int:
    """Enforce the machine-sharing contract at every public entry point."""

    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("jobs must be an integer")
    if not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be between 1 and {MAX_JOBS}, got {jobs}")
    for variable in THREAD_ENVIRONMENT:
        os.environ[variable] = "1"
    return jobs


def _feature_pipeline(
    feature_set: str,
    estimator: BaseEstimator,
    *,
    include_parameter_count: bool = True,
    dataset_conditioning: str = "unaware",
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureEncoder(
                    feature_set=feature_set,
                    include_parameter_count=include_parameter_count,
                    dataset_conditioning=dataset_conditioning,
                ),
            ),
            ("model", estimator),
        ]
    )


def wide_search_definitions(
    seed: int,
    *,
    include_xgboost: bool = False,
    include_parameter_count: bool = True,
    dataset_conditioning: str = "unaware",
) -> list[SearchDefinition]:
    """Return the predeclared wide model ladder.

    OLS and Ridge are separate method names so the unregularized linear rule is
    visible in reports instead of being hidden as one value in a Ridge grid.
    """

    # Reuse the established estimators and search grids.  The only additions
    # are explicit OLS variants and the missing full-feature ElasticNet; this
    # keeps the old and wide studies directly comparable.
    base_definitions = (
        search_definitions(seed, dataset_conditioning=dataset_conditioning)
        if include_parameter_count
        else search_definitions(
            seed,
            include_parameter_count=False,
            dataset_conditioning=dataset_conditioning,
        )
    )
    base = {definition.name: definition for definition in base_definitions}
    required = {
        "constant_mean",
        "compact_ridge",
        "compact_polynomial_ridge",
        "full_ridge",
        "compact_elastic_net",
        "optimizer_lr_lookup",
        "optimizer_lr_ridge",
        "shallow_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "gradient_boosting",
        "rbf_svr",
        "mlp",
    }
    if include_parameter_count:
        required.update(
            {
                "max_params_heuristic",
                "params_ridge",
                "params_polynomial_ridge",
            }
        )
    missing = required.difference(base)
    if missing:  # pragma: no cover - protects the cross-study contract
        raise RuntimeError(
            f"Existing model ladder is missing: {', '.join(sorted(missing))}"
        )
    definitions = [base["constant_mean"]]
    if include_parameter_count:
        params_ridge = base["params_ridge"]
        definitions.extend(
            [
                base["max_params_heuristic"],
                SearchDefinition(
                    name="params_ols",
                    estimator=_feature_pipeline(
                        "params",
                        LinearRegression(n_jobs=1),
                        include_parameter_count=True,
                        dataset_conditioning=dataset_conditioning,
                    ),
                    params={},
                    feature_set="params",
                    interpretable=True,
                ),
                SearchDefinition(
                    name="params_ridge",
                    estimator=params_ridge.estimator,
                    params={
                        "model__alpha": [
                            value
                            for value in params_ridge.params["model__alpha"]
                            if float(value) > 0.0
                        ]
                    },
                    feature_set=params_ridge.feature_set,
                    randomized_iterations=params_ridge.randomized_iterations,
                    interpretable=params_ridge.interpretable,
                ),
                base["params_polynomial_ridge"],
            ]
        )
    definitions.extend(
        [base["optimizer_lr_lookup"], base["optimizer_lr_ridge"]]
    )
    for feature_set in ("compact", "full"):
        definitions.append(
            SearchDefinition(
                name=f"{feature_set}_ols",
                estimator=_feature_pipeline(
                    feature_set,
                    LinearRegression(n_jobs=1),
                    include_parameter_count=include_parameter_count,
                    dataset_conditioning=dataset_conditioning,
                ),
                params={},
                feature_set=feature_set,
                interpretable=True,
            )
        )
        definitions.append(base[f"{feature_set}_ridge"])
        if feature_set == "compact":
            definitions.extend(
                [base["compact_polynomial_ridge"], base["compact_elastic_net"]]
            )
        else:
            compact_elastic = base["compact_elastic_net"]
            definitions.append(
                SearchDefinition(
                    name="full_elastic_net",
                    estimator=_feature_pipeline(
                        "full",
                        ElasticNet(
                            max_iter=20_000,
                            selection="cyclic",
                            random_state=seed,
                        ),
                        include_parameter_count=include_parameter_count,
                        dataset_conditioning=dataset_conditioning,
                    ),
                    params=compact_elastic.params,
                    feature_set="full",
                    randomized_iterations=compact_elastic.randomized_iterations,
                    interpretable=True,
                )
            )
    definitions.extend(
        base[name]
        for name in (
            "shallow_tree",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "gradient_boosting",
            "rbf_svr",
            "mlp",
        )
    )
    fixed_configs = {
        "compact_ridge": {},
        "full_ridge": {},
        "random_forest": {
            "model__n_estimators": 300,
            "model__max_depth": 12,
            "model__min_samples_leaf": 5,
            "model__max_features": 0.7,
        },
        "extra_trees": {
            "model__n_estimators": 300,
            "model__max_depth": 16,
            "model__min_samples_leaf": 5,
            "model__max_features": 0.7,
        },
        "mlp": {
            "model__hidden_layer_sizes": (128, 64),
            "model__activation": "relu",
            "model__alpha": 0.01,
            "model__learning_rate_init": 0.001,
        },
        "xgboost": {
            "model__n_estimators": 400,
            "model__learning_rate": 0.03,
            "model__max_depth": 3,
            "model__min_child_weight": 5,
            "model__subsample": 0.9,
            "model__colsample_bytree": 0.8,
            "model__reg_alpha": 0.0,
            "model__reg_lambda": 10.0,
        },
    }
    for name, params in fixed_configs.items():
        if name not in base:
            continue
        estimator = sklearn.base.clone(base[name].estimator)
        if params:
            estimator.set_params(**params)
        definitions.append(
            SearchDefinition(
                name=f"{name}_fixed",
                estimator=estimator,
                params={},
                feature_set=base[name].feature_set,
                interpretable=base[name].interpretable,
                dataset_conditioning=dataset_conditioning,
            )
        )
    if include_xgboost:
        if "xgboost" not in base:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "--include-xgboost was requested but xgboost is not installed"
            )
        definitions.append(base["xgboost"])
    return [
        replace(definition, dataset_conditioning=dataset_conditioning)
        for definition in definitions
    ]


def _definition_subset(
    definitions: Sequence[SearchDefinition], method_names: set[str] | None
) -> list[SearchDefinition]:
    if method_names is None:
        return list(definitions)
    known = {definition.name for definition in definitions}
    unknown = method_names.difference(known)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
    selected = [
        definition for definition in definitions if definition.name in method_names
    ]
    if not selected:
        raise ValueError("At least one method is required")
    return selected


def _row_fingerprints(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["example_fingerprint_sha256"]) for row in rows]


def _stratified_fold_plan(
    rows: Sequence[Mapping[str, Any]], *, n_splits: int, seed: int
) -> FoldPlan:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if len(rows) < 2:
        raise ValueError("At least two training rows are required")
    strata = np.asarray([str(row["stratum"]) for row in rows], dtype=object)
    _, counts = np.unique(strata, return_counts=True)
    maximum_stratified = int(np.min(counts)) if counts.size else 0
    effective = min(n_splits, maximum_stratified)
    dummy = np.zeros(len(rows), dtype=np.int8)
    if effective >= 2:
        splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=seed)
        raw_folds = splitter.split(dummy, strata)
        kind = "stratified_by_row_stratum"
    else:
        effective = min(n_splits, len(rows))
        if effective < 2:
            raise ValueError("At least two rows are required for inner CV")
        splitter = KFold(n_splits=effective, shuffle=True, random_state=seed)
        raw_folds = splitter.split(dummy)
        kind = "kfold_fallback_insufficient_stratum_count"
    folds = tuple(
        (
            np.asarray(train, dtype=np.int64),
            np.asarray(validation, dtype=np.int64),
        )
        for train, validation in raw_folds
    )
    return FoldPlan(
        folds=folds,
        manifest={
            "kind": kind,
            "requested_splits": n_splits,
            "effective_splits": len(folds),
            "seed": seed,
        },
    )


def _group_fold_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    n_splits: int,
    seed: int,
) -> FoldPlan:
    groups = np.asarray([group_value(row, axis) for row in rows], dtype=object)
    unique_groups = sorted(set(str(value) for value in groups))
    if len(unique_groups) < 2:
        fallback = _stratified_fold_plan(rows, n_splits=n_splits, seed=seed)
        return FoldPlan(
            folds=fallback.folds,
            manifest={
                **fallback.manifest,
                "fallback_from": f"group_kfold:{axis}",
                "n_groups": len(unique_groups),
            },
        )
    effective = min(n_splits, len(unique_groups))
    dummy = np.zeros(len(rows), dtype=np.int8)
    splitter = GroupKFold(n_splits=effective)
    folds = tuple(
        (
            np.asarray(train, dtype=np.int64),
            np.asarray(validation, dtype=np.int64),
        )
        for train, validation in splitter.split(dummy, groups=groups)
    )
    for train, validation in folds:
        if set(groups[train]).intersection(set(groups[validation])):
            raise AssertionError(f"Inner {axis} group leakage")
    return FoldPlan(
        folds=folds,
        manifest={
            "kind": "group_kfold",
            "axis": axis,
            "requested_splits": n_splits,
            "effective_splits": len(folds),
            "n_groups": len(unique_groups),
            "groups": unique_groups,
            "seed": None,
        },
    )


def _fold_digest(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> str:
    fingerprints = _row_fingerprints(rows)
    payload = [
        {
            "train": [fingerprints[index] for index in train],
            "validation": [fingerprints[index] for index in validation],
        }
        for train, validation in folds
    ]
    return _sha256_value(payload)


def _definition_config(definition: SearchDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "estimator": _json_safe(definition.estimator),
        "params": _json_safe(definition.params),
        "feature_set": definition.feature_set,
        "randomized_iterations": definition.randomized_iterations,
        "interpretable": definition.interpretable,
        "dataset_conditioning": definition.dataset_conditioning,
    }


def _source_hashes() -> dict[str, str]:
    return {name: _sha256_file(path.resolve()) for name, path in SOURCE_FILES.items()}


def _checkpoint_contract(
    *,
    definition: SearchDefinition,
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    fold_plan: FoldPlan,
    protocol: Mapping[str, Any],
    seed: int,
    data_sha256: str | None = None,
    fold_sha256: str | None = None,
) -> dict[str, Any]:
    method_config = _definition_config(definition)
    if data_sha256 is None:
        data_sha256 = _sha256_value({"train": train_rows, "test": test_rows})
    if fold_sha256 is None:
        fold_sha256 = _fold_digest(train_rows, fold_plan.folds)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "source_hashes": dict(source_hashes),
        "method_config": method_config,
        "data_sha256": data_sha256,
        "fold_sha256": fold_sha256,
        "fold_manifest": fold_plan.manifest,
        "protocol": protocol,
        "seed": seed,
        "scikit_learn": sklearn.__version__,
        "thread_limit": 1,
    }
    return {**payload, "input_digest": _sha256_value(payload)}


def _load_valid_checkpoint(
    checkpoint: Path,
    sidecar: Path,
    *,
    digest: str,
    method: str,
) -> FittedModel | None:
    if not checkpoint.is_file() or not sidecar.is_file():
        return None
    metadata = json.loads(sidecar.read_text("utf-8"))
    if metadata.get("input_digest") != digest or metadata.get("method") != method:
        return None
    model = joblib.load(checkpoint)
    if not isinstance(model, FittedModel) or model.name != method:
        raise TypeError(f"Invalid checkpoint payload: {checkpoint}")
    return model


def _regret_counts(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[Counter[float], int]:
    """Return exact regret multiplicities for every three-choice subset.

    Sorting by prediction identifies the selected row for every triple.  For
    each selected row, grouping the later rows by true loss counts all pairs by
    their minimum true loss in O(n^2 log n) time and O(n^2) distinct output at
    worst, without allocating the C(n, 3) triple matrix.
    """

    n_rows = int(truth.size)
    if n_rows < 3:
        return Counter(), 0
    order = np.lexsort((np.arange(n_rows), prediction))
    ordered_truth = truth[order]
    counts: Counter[float] = Counter()
    for position in range(n_rows - 2):
        selected = float(ordered_truth[position])
        values, multiplicities = np.unique(
            ordered_truth[position + 1 :], return_counts=True
        )
        greater = int(np.sum(multiplicities))
        for value, multiplicity_raw in zip(values, multiplicities):
            multiplicity = int(multiplicity_raw)
            greater -= multiplicity
            pair_count = multiplicity * greater + math.comb(multiplicity, 2)
            regret = max(selected - float(value), 0.0)
            counts[regret] += pair_count
    n_groups = math.comb(n_rows, 3)
    if sum(counts.values()) != n_groups:
        raise AssertionError("Three-choice regret counts do not cover every triple")
    return counts, n_groups


def _weighted_median(counts: Mapping[float, int]) -> float | None:
    total = sum(counts.values())
    if total == 0:
        return None
    left_rank = (total - 1) // 2
    right_rank = total // 2
    cumulative = 0
    left_value: float | None = None
    right_value: float | None = None
    for value, count in sorted(counts.items()):
        next_cumulative = cumulative + count
        if left_value is None and left_rank < next_cumulative:
            left_value = float(value)
        if right_rank < next_cumulative:
            right_value = float(value)
            break
        cumulative = next_cumulative
    if left_value is None or right_value is None:  # pragma: no cover - defensive
        raise AssertionError("Weighted median ranks were not found")
    return (left_value + right_value) / 2.0


def _regret_summary(counts: Mapping[float, int]) -> dict[str, float | None]:
    total = sum(counts.values())
    if total == 0:
        return {"mean": None, "median": None}
    return {
        "mean": float(sum(value * count for value, count in counts.items()) / total),
        "median": _weighted_median(counts),
    }


def _score_subset(
    rows: Sequence[Mapping[str, Any]], predicted_log_loss: np.ndarray
) -> dict[str, Any]:
    truth_log = np.asarray(
        [float(row["target"]["log_mean_loss"]) for row in rows], dtype=np.float64
    )
    truth_raw = np.asarray(
        [float(row["target"]["mean_loss"]) for row in rows], dtype=np.float64
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_value(row, "environment")].append(index)

    pair_numerator = 0.0
    comparable_pairs = 0
    spearman_values: list[float] = []
    log_regrets: Counter[float] = Counter()
    raw_regrets: Counter[float] = Counter()
    n_groups = 0
    n_correct = 0
    gap_groups = 0
    gap_correct = 0.0
    gap_raw_regret_sum = 0.0
    gap_log_regret_sum = 0.0
    environment_sizes: dict[str, int] = {}
    environment_metrics: dict[str, Any] = {}
    macro_log_rmse: list[float] = []
    macro_spearman: list[float] = []
    macro_three_choice: list[float] = []
    macro_gap_three_choice: list[float] = []
    for environment in sorted(grouped):
        indices = np.asarray(grouped[environment], dtype=np.int64)
        environment_sizes[environment] = int(indices.size)
        environment_truth_log = truth_log[indices]
        environment_truth_raw = truth_raw[indices]
        environment_prediction = predicted_log_loss[indices]
        detailed = log_loss_prediction_metrics(
            environment_truth_raw,
            environment_prediction,
            gap_threshold=0.05,
        )
        environment_metrics[environment] = detailed
        ranking = detailed["ranking"]
        comparable = int(ranking["n_comparable_pairs"])
        if ranking["pair_concordance"] is not None:
            comparable_pairs += comparable
            pair_numerator += float(ranking["pair_concordance"]) * comparable
        if ranking["spearman"] is not None:
            spearman_values.append(float(ranking["spearman"]))
            macro_spearman.append(float(ranking["spearman"]))
        macro_log_rmse.append(float(detailed["log"]["rmse"]))
        all_choice = detailed["three_choice"]["all"]
        if all_choice["accuracy"] is not None:
            macro_three_choice.append(float(all_choice["accuracy"]))
        gap_choice = detailed["three_choice"]["gap_ge_0_05"]
        if gap_choice["accuracy"] is not None:
            macro_gap_three_choice.append(float(gap_choice["accuracy"]))
            group_count = int(gap_choice["n_groups"])
            gap_groups += group_count
            gap_correct += float(gap_choice["accuracy"]) * group_count
            gap_raw_regret_sum += (
                float(gap_choice["regret"]["raw"]["mean"]) * group_count
            )
            gap_log_regret_sum += (
                float(gap_choice["regret"]["log"]["mean"]) * group_count
            )

        log_counts, group_count = _regret_counts(
            environment_truth_log, environment_prediction
        )
        raw_counts, raw_group_count = _regret_counts(
            environment_truth_raw, environment_prediction
        )
        if raw_group_count != group_count or raw_counts.get(0.0, 0) != log_counts.get(
            0.0, 0
        ):
            raise AssertionError("Raw/log winner accounting disagrees")
        log_regrets.update(log_counts)
        raw_regrets.update(raw_counts)
        n_groups += group_count
        n_correct += log_counts.get(0.0, 0)

    result = {
        "n": len(rows),
        "log": regression_metrics(truth_log, predicted_log_loss),
        "ranking": ranking_metrics(truth_log, predicted_log_loss),
        "per_environment": environment_metrics,
        "raw_regression_aggregation": (
            "reported_per_environment_only; raw CE/MSE scales are not pooled"
        ),
        "within_environment": {
            "scope": "same_environment_only",
            "n_environments": len(grouped),
            "environment_sizes": environment_sizes,
            "spearman_macro": (
                float(np.mean(spearman_values)) if spearman_values else None
            ),
            "pair_concordance": (
                pair_numerator / comparable_pairs if comparable_pairs else None
            ),
            "n_comparable_pairs": comparable_pairs,
            "macro": {
                "log_rmse": float(np.mean(macro_log_rmse)),
                "spearman": (
                    float(np.mean(macro_spearman)) if macro_spearman else None
                ),
                "three_choice_accuracy": (
                    float(np.mean(macro_three_choice))
                    if macro_three_choice
                    else None
                ),
                "gap_ge_0_05_three_choice_accuracy": (
                    float(np.mean(macro_gap_three_choice))
                    if macro_gap_three_choice
                    else None
                ),
            },
            "three_choice": {
                "scope": "all_triples_within_each_environment",
                "n_groups": n_groups,
                "n_correct": n_correct,
                "accuracy": n_correct / n_groups if n_groups else None,
                "log_regret": _regret_summary(log_regrets),
                "raw_regret": _regret_summary(raw_regrets),
                "gap_ge_0_05": {
                    "threshold": 0.05,
                    "gap_definition": "second_best_raw_loss - best_raw_loss",
                    "n_groups": gap_groups,
                    "accuracy": gap_correct / gap_groups if gap_groups else None,
                    "raw_regret_mean": (
                        gap_raw_regret_sum / gap_groups if gap_groups else None
                    ),
                    "log_regret_mean": (
                        gap_log_regret_sum / gap_groups if gap_groups else None
                    ),
                    "median_note": (
                        "exact medians are retained in per_environment metrics"
                    ),
                },
            },
        },
    }
    if len(environment_metrics) == 1:
        only = next(iter(environment_metrics.values()))
        result["raw"] = only["raw"]
        result["three_choice_detailed"] = only["three_choice"]
    return result


def score_wide_predictions(
    rows: Sequence[Mapping[str, Any]],
    predicted_log_loss: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Score log loss and within-environment ranking/choice metrics."""

    if not rows:
        raise ValueError("At least one row is required")
    predictions = np.asarray(predicted_log_loss, dtype=np.float64)
    if predictions.shape != (len(rows),) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"Expected {len(rows)} finite log-loss predictions")
    eligible = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    return {
        "target": "log(mean_loss)",
        "all": _score_subset(rows, predictions),
        "benchmark_eligible": (
            _score_subset(
                [row for row, keep in zip(rows, eligible) if keep],
                predictions[eligible],
            )
            if np.any(eligible)
            else None
        ),
    }


def _median_metric_trees(trees: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not trees:
        raise ValueError("At least one metric tree is required")
    result: dict[str, Any] = {}
    for key in trees[0]:
        values = [tree[key] for tree in trees]
        first = values[0]
        if isinstance(first, Mapping):
            result[key] = _median_metric_trees(values)  # type: ignore[arg-type]
        elif key == "n" or key.startswith("n_") or key == "threshold":
            result[key] = (
                first
                if all(value == first for value in values[1:])
                else float(np.median([float(value) for value in values]))
            )
        else:
            numeric = [float(value) for value in values if value is not None]
            result[key] = float(np.median(numeric)) if numeric else None
    return result


def dynamic_split_noise_ceiling(environment: WideEnvironment) -> dict[str, Any]:
    """Compute exhaustive complementary split reliability for any seed count.

    Five-seed wide-v2 rows use all ten 2/3 partitions in both directions.  For
    an even seed count, requiring seed zero in the first half removes duplicate
    complementary partitions, matching the original 10-seed protocol.
    """

    final_losses = load_seed_losses(environment, split="validation")
    n_seeds = environment.n_seeds
    if n_seeds < 2:
        raise ValueError("Noise ceiling requires at least two seeds")
    smaller = n_seeds // 2
    larger = n_seeds - smaller
    if smaller == larger:
        partitions = [
            subset for subset in combinations(range(n_seeds), smaller) if 0 in subset
        ]
    else:
        partitions = list(combinations(range(n_seeds), smaller))
    all_indices = set(range(n_seeds))
    eligible = np.asarray(
        [
            row["target"].get("benchmark_eligible") is True
            for row in environment.validation_rows
        ],
        dtype=bool,
    )
    all_metrics: list[dict[str, Any]] = []
    eligible_metrics: list[dict[str, Any]] = []
    for first_tuple in partitions:
        first = np.asarray(first_tuple, dtype=np.int64)
        second = np.asarray(sorted(all_indices.difference(first_tuple)), dtype=np.int64)
        mean_first = np.mean(final_losses[:, first], axis=1)
        mean_second = np.mean(final_losses[:, second], axis=1)
        for truth, prediction in (
            (mean_second, mean_first),
            (mean_first, mean_second),
        ):
            all_metrics.append(loss_prediction_metrics(truth, prediction))
            if np.any(eligible):
                eligible_metrics.append(
                    loss_prediction_metrics(truth[eligible], prediction[eligible])
                )
    return {
        "status": "computed",
        "n_seeds": n_seeds,
        "split_sizes": [smaller, larger],
        "n_complementary_partitions": len(partitions),
        "n_directed_comparisons": len(all_metrics),
        "all": {
            "n_rows": len(environment.validation_rows),
            "median_metrics": _median_metric_trees(all_metrics),
        },
        "benchmark_eligible": (
            {
                "n_rows": int(np.sum(eligible)),
                "median_metrics": _median_metric_trees(eligible_metrics),
            }
            if eligible_metrics
            else None
        ),
    }


def noise_ceiling_for_environment(
    environment: WideEnvironment, *, mode: str
) -> dict[str, Any]:
    """Compute, skip, or require the stored-curve ceiling without guessing."""

    if mode not in {"auto", "skip", "require"}:
        raise ValueError("noise ceiling mode must be auto, skip, or require")
    if mode == "skip":
        return {
            "status": "skipped",
            "reason": "disabled_by_user",
            "n_seeds": environment.n_seeds,
        }
    try:
        return dynamic_split_noise_ceiling(environment)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        if mode == "require":
            raise
        return {
            "status": "skipped",
            "reason": f"stored_seed_curves_unavailable: {exc}",
            "n_seeds": environment.n_seeds,
        }


def _run_task(
    *,
    task_id: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    output_dir: Path,
    fold_plan: FoldPlan,
    definitions: Sequence[SearchDefinition],
    protocol: dict[str, Any],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    force: bool,
) -> TaskResult:
    validate_jobs(jobs)
    if not train_rows or not test_rows:
        raise ValueError(f"{task_id} needs non-empty train and test rows")
    overlap = set(_row_fingerprints(train_rows)).intersection(
        _row_fingerprints(test_rows)
    )
    if overlap:
        raise ValueError(f"{task_id} train/test overlap: {len(overlap)} rows")
    y_log = np.asarray(
        [float(row["target"]["log_mean_loss"]) for row in train_rows], dtype=float
    )
    methods: list[dict[str, Any]] = []
    prediction_payload: list[dict[str, Any]] = []
    prediction_vectors: dict[str, np.ndarray] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "completed": [],
            "current": None,
            "n_total": len(definitions),
            "updated_at": _utc_now(),
        },
    )
    completed: list[str] = []
    # These are task-invariant but can cover thousands of rows.  Compute them
    # once instead of reserializing the same data for every method digest.
    task_data_sha256 = _sha256_value({"train": train_rows, "test": test_rows})
    task_fold_sha256 = _fold_digest(train_rows, fold_plan.folds)
    for definition in definitions:
        contract = _checkpoint_contract(
            definition=definition,
            plan_sha256=plan_sha256,
            source_hashes=source_hashes,
            train_rows=train_rows,
            test_rows=test_rows,
            fold_plan=fold_plan,
            protocol=protocol,
            seed=seed,
            data_sha256=task_data_sha256,
            fold_sha256=task_fold_sha256,
        )
        checkpoint = output_dir / "models" / f"{definition.name}.joblib"
        sidecar = output_dir / "models" / f"{definition.name}.json"
        model = (
            None
            if force
            else _load_valid_checkpoint(
                checkpoint,
                sidecar,
                digest=contract["input_digest"],
                method=definition.name,
            )
        )
        action = "resume" if model is not None else "fit"
        print(f"[{_utc_now()}] {task_id}: {action} {definition.name}", flush=True)
        _atomic_write_json(
            output_dir / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "completed": completed,
                "current": definition.name,
                "n_total": len(definitions),
                "updated_at": _utc_now(),
            },
        )
        if model is None:
            with (
                parallel_backend("loky", n_jobs=jobs, inner_max_num_threads=1),
                threadpool_limits(limits=1),
            ):
                model = fit_definition(
                    definition,
                    train_rows,
                    y_log,
                    list(fold_plan.folds),
                    jobs=jobs,
                    seed=seed,
                )
            _atomic_joblib_dump(checkpoint, model)
            _atomic_write_json(
                sidecar,
                {
                    "schema_version": SCHEMA_VERSION,
                    "method": model.name,
                    "checkpoint": str(checkpoint.resolve()),
                    **contract,
                    "best_params": model.best_params,
                    "feature_set": model.feature_set,
                    "cv": {
                        "rmse_log": model.cv_rmse_log,
                        "mae_log": model.cv_mae_log,
                        "r2_log": model.cv_r2_log,
                    },
                    "written_at": _utc_now(),
                },
            )
        predicted = model.predict_log(test_rows)
        if predicted.shape != (len(test_rows),) or not np.all(np.isfinite(predicted)):
            raise ValueError(f"{task_id}/{model.name} produced invalid predictions")
        oof_metrics = score_wide_predictions(train_rows, model.oof_predictions)
        test_metrics = score_wide_predictions(test_rows, predicted)
        methods.append(
            {
                "method": model.name,
                "feature_set": model.feature_set,
                "interpretable": model.interpretable,
                "best_params": model.best_params,
                "checkpoint": str(checkpoint.resolve()),
                "input_digest": contract["input_digest"],
                "cv_rmse_log": model.cv_rmse_log,
                "cv_mae_log": model.cv_mae_log,
                "cv_r2_log": model.cv_r2_log,
                "oof": oof_metrics,
                "test": test_metrics,
            }
        )
        prediction_vectors[model.name] = predicted
        prediction_payload.append(
            {
                "method": model.name,
                "predictions": [
                    {
                        "example_fingerprint_sha256": row["example_fingerprint_sha256"],
                        "predicted_log_loss": float(value),
                    }
                    for row, value in zip(test_rows, predicted)
                ],
            }
        )
        completed.append(model.name)

    methods.sort(key=lambda item: (float(item["cv_rmse_log"]), item["method"]))
    for rank, item in enumerate(methods, start=1):
        item["cv_rank"] = rank
    champion = methods[0]["method"]
    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "target": "log(mean_loss)",
        "champion_selected_by": "minimum training-only inner-CV log RMSE",
        "test_used_for_selection": False,
        "cv_champion": champion,
        "protocol": protocol,
        "inner_cv": fold_plan.manifest,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "methods": methods,
    }
    _atomic_write_json(output_dir / "leaderboard.json", leaderboard)
    _atomic_write_json(
        output_dir / "predictions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "target": "log(mean_loss)",
            "methods": prediction_payload,
        },
    )
    _atomic_write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "completed": completed,
            "current": None,
            "n_total": len(definitions),
            "updated_at": _utc_now(),
        },
    )
    return TaskResult(task_id, test_rows, prediction_vectors, leaderboard)


def _aggregate_results(
    results: Sequence[TaskResult], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one completed task is required")
    rows = [row for result in results for row in result.rows]
    method_names = set(results[0].predictions)
    if any(set(result.predictions) != method_names for result in results[1:]):
        raise ValueError("Tasks disagree about evaluated methods")
    methods = []
    for method in sorted(method_names):
        predictions = np.concatenate([result.predictions[method] for result in results])
        cv_values = [
            float(
                next(
                    item["cv_rmse_log"]
                    for item in result.leaderboard["methods"]
                    if item["method"] == method
                )
            )
            for result in results
        ]
        methods.append(
            {
                "method": method,
                "n_tasks": len(results),
                "mean_task_cv_rmse_log": float(np.mean(cv_values)),
                "test": score_wide_predictions(rows, predictions),
            }
        )
    methods.sort(
        key=lambda item: (float(item["mean_task_cv_rmse_log"]), item["method"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(protocol),
        "target": "log(mean_loss)",
        "n_tasks": len(results),
        "n_test": len(rows),
        "task_ids": [result.task_id for result in results],
        "methods": methods,
    }


def _families(corpus: WideCorpus) -> list[str]:
    return sorted({environment.family for environment in corpus.environments})


def _datasets(corpus: WideCorpus) -> list[str]:
    return sorted({environment.dataset_id for environment in corpus.environments})


def _rows_for_family(
    rows: Iterable[dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["family"]) == family]


def _rows_for_dataset(
    rows: Iterable[dict[str, Any]], dataset_id: str
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["dataset_id"]) == dataset_id]


def fit_per_environment_id(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
    noise_ceiling: str,
) -> dict[str, Any]:
    """Fit each completed environment and open only its locked validation."""

    protocol = {
        "name": "per_environment_id",
        "train": "environment.train_rows",
        "test": "same environment.validation_rows",
        "locked_validation": True,
    }
    results: list[TaskResult] = []
    ceilings: dict[str, Any] = {}
    for environment in corpus.environments:
        fold_plan = _stratified_fold_plan(
            environment.train_rows, n_splits=n_splits, seed=seed
        )
        result = _run_task(
            task_id=environment.experiment_id,
            train_rows=list(environment.train_rows),
            test_rows=list(environment.validation_rows),
            output_dir=output_root / "id" / "environment" / environment.experiment_id,
            fold_plan=fold_plan,
            definitions=definitions,
            protocol={
                **protocol,
                "family": environment.family,
                "dataset": environment.dataset_id,
            },
            plan_sha256=plan_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            force=force,
        )
        ceiling = noise_ceiling_for_environment(environment, mode=noise_ceiling)
        _atomic_write_json(
            output_root
            / "id"
            / "environment"
            / environment.experiment_id
            / "noise_ceiling.json",
            ceiling,
        )
        ceilings[environment.experiment_id] = {
            "status": ceiling["status"],
            "n_seeds": ceiling["n_seeds"],
        }
        results.append(result)
    aggregate = _aggregate_results(results, protocol=protocol)
    aggregate["noise_ceilings"] = ceilings
    _atomic_write_json(output_root / "id" / "environment" / "aggregate.json", aggregate)
    return aggregate


def fit_family_pooled_id(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    """Fit one ID model per family and score all locked validation rows."""

    protocol = {
        "name": "family_pooled_id",
        "train": "all family train_rows",
        "test": "all family validation_rows",
        "locked_validation": True,
        "family_conditioned": True,
    }
    results: list[TaskResult] = []
    for family in _families(corpus):
        train_rows = _rows_for_family(corpus.train_rows, family)
        test_rows = _rows_for_family(corpus.validation_rows, family)
        fold_plan = _stratified_fold_plan(train_rows, n_splits=n_splits, seed=seed)
        results.append(
            _run_task(
                task_id=f"family={family}",
                train_rows=train_rows,
                test_rows=test_rows,
                output_dir=output_root / "id" / "family" / family,
                fold_plan=fold_plan,
                definitions=definitions,
                protocol={**protocol, "family": family},
                plan_sha256=plan_sha256,
                source_hashes=source_hashes,
                jobs=jobs,
                seed=seed,
                force=force,
            )
        )
    aggregate = _aggregate_results(results, protocol=protocol)
    _atomic_write_json(output_root / "id" / "family" / "aggregate.json", aggregate)
    return aggregate


def fit_dataset_pooled_id(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    """Fit one ID model per dataset and score its locked validation rows."""

    protocol = {
        "name": "dataset_pooled_id",
        "train": "all dataset train_rows",
        "test": "all dataset validation_rows",
        "locked_validation": True,
        "dataset_conditioned": True,
    }
    results: list[TaskResult] = []
    for dataset_id in _datasets(corpus):
        train_rows = _rows_for_dataset(corpus.train_rows, dataset_id)
        test_rows = _rows_for_dataset(corpus.validation_rows, dataset_id)
        fold_plan = _stratified_fold_plan(train_rows, n_splits=n_splits, seed=seed)
        results.append(
            _run_task(
                task_id=f"dataset={dataset_id}",
                train_rows=train_rows,
                test_rows=test_rows,
                output_dir=output_root / "id" / "dataset" / dataset_id,
                fold_plan=fold_plan,
                definitions=definitions,
                protocol={**protocol, "dataset": dataset_id},
                plan_sha256=plan_sha256,
                source_hashes=source_hashes,
                jobs=jobs,
                seed=seed,
                force=force,
            )
        )
    aggregate = _aggregate_results(results, protocol=protocol)
    _atomic_write_json(output_root / "id" / "dataset" / "aggregate.json", aggregate)
    return aggregate


def fit_global_pooled_id(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    """Fit one shared model on every training row and score locked validation."""

    protocol = {
        "name": "global_pooled_id",
        "train": "all corpus train_rows",
        "test": "all corpus validation_rows",
        "locked_validation": True,
        "dataset_conditioned": False,
        "shared_model_count": 1,
    }
    train_rows = list(corpus.train_rows)
    test_rows = list(corpus.validation_rows)
    fold_plan = _stratified_fold_plan(train_rows, n_splits=n_splits, seed=seed)
    result = _run_task(
        task_id="global",
        train_rows=train_rows,
        test_rows=test_rows,
        output_dir=output_root / "id" / "global",
        fold_plan=fold_plan,
        definitions=definitions,
        protocol=protocol,
        plan_sha256=plan_sha256,
        source_hashes=source_hashes,
        jobs=jobs,
        seed=seed,
        force=force,
    )
    aggregate = _aggregate_results([result], protocol=protocol)
    _atomic_write_json(output_root / "id" / "global" / "aggregate.json", aggregate)
    return aggregate


def _fit_logo_protocol(
    corpus: WideCorpus,
    output_root: Path,
    *,
    axis: str,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    protocol = {
        "name": f"leave_one_{axis}_out",
        "outer_axis": axis,
        "family_conditioned": True,
        "train": f"other {axis} groups' train_rows",
        "test": f"held-out {axis} group's all_rows",
        "outer_test_used_for_selection": False,
    }
    results: list[TaskResult] = []
    for family in _families(corpus):
        family_train = _rows_for_family(corpus.train_rows, family)
        family_all = _rows_for_family(corpus.all_rows, family)
        groups = sorted({group_value(row, axis) for row in family_all})
        if len(groups) < 2:
            raise ValueError(f"{family} needs at least two {axis} groups for LOGO")
        for held_out in groups:
            train_rows = [
                row for row in family_train if group_value(row, axis) != held_out
            ]
            test_rows = [
                row for row in family_all if group_value(row, axis) == held_out
            ]
            fold_plan = _group_fold_plan(
                train_rows, axis=axis, n_splits=n_splits, seed=seed
            )
            results.append(
                _run_task(
                    task_id=f"{family}:{axis}={held_out}",
                    train_rows=train_rows,
                    test_rows=test_rows,
                    output_dir=output_root / "ood" / f"{axis}_logo" / family / held_out,
                    fold_plan=fold_plan,
                    definitions=definitions,
                    protocol={
                        **protocol,
                        "family": family,
                        "held_out_group": held_out,
                    },
                    plan_sha256=plan_sha256,
                    source_hashes=source_hashes,
                    jobs=jobs,
                    seed=seed,
                    force=force,
                )
            )
    aggregate = _aggregate_results(results, protocol=protocol)
    _atomic_write_json(
        output_root / "ood" / f"{axis}_logo" / "aggregate.json", aggregate
    )
    return aggregate


def fit_family_logo_ood(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    """Fit on complete families and evaluate a genuinely unseen family.

    This is deliberately separate from :func:`fit_family_pooled_id`.  The ID
    protocol fits one model *inside* every family, whereas this protocol holds
    one complete family out of both fitting and training-only model selection.
    Predictions are written for every held-family row.  A second aggregate over
    only the preassigned locked-validation rows is stored alongside the all-row
    aggregate so the corrected result can be compared to the ID headline
    without retraining or changing the outer split.
    """

    protocol = {
        "name": "leave_one_family_out",
        "outer_axis": "family",
        "family_conditioned": False,
        "cross_family": True,
        "train": "other families' train_rows",
        "test": "held-out family's all_rows",
        "inner_cv": "leave-one-training-family-out",
        "outer_test_used_for_selection": False,
    }
    families = _families(corpus)
    if len(families) < 3:
        raise ValueError(
            "leave-one-family-out needs at least three families so training-only "
            "inner CV has at least two family groups"
        )

    results: list[TaskResult] = []
    for held_out in families:
        training_families = [family for family in families if family != held_out]
        train_rows = [
            row
            for row in corpus.train_rows
            if group_value(row, "family") != held_out
        ]
        test_rows = [
            row for row in corpus.all_rows if group_value(row, "family") == held_out
        ]
        if {group_value(row, "family") for row in train_rows}.intersection(
            {held_out}
        ):
            raise AssertionError("Held-out family leaked into training")
        fold_plan = _group_fold_plan(
            train_rows, axis="family", n_splits=n_splits, seed=seed
        )
        results.append(
            _run_task(
                task_id=f"family={held_out}",
                train_rows=train_rows,
                test_rows=test_rows,
                output_dir=output_root / "ood" / "family_logo" / held_out,
                fold_plan=fold_plan,
                definitions=definitions,
                protocol={
                    **protocol,
                    "held_out_family": held_out,
                    "training_families": training_families,
                },
                plan_sha256=plan_sha256,
                source_hashes=source_hashes,
                jobs=jobs,
                seed=seed,
                force=force,
            )
        )

    aggregate = _aggregate_results(results, protocol=protocol)
    aggregate_path = output_root / "ood" / "family_logo" / "aggregate.json"
    _atomic_write_json(aggregate_path, aggregate)

    validation_results: list[TaskResult] = []
    for result in results:
        keep = np.asarray(
            [str(row.get("split")) == "validation" for row in result.rows],
            dtype=bool,
        )
        validation_results.append(
            TaskResult(
                task_id=result.task_id,
                rows=[row for row, selected in zip(result.rows, keep) if selected],
                predictions={
                    method: predictions[keep]
                    for method, predictions in result.predictions.items()
                },
                leaderboard=result.leaderboard,
            )
        )
    validation_protocol = {
        **protocol,
        "name": "leave_one_family_out_locked_validation",
        "test": "held-out family's locked validation_rows",
        "derived_from": str(aggregate_path.resolve()),
    }
    validation_aggregate = _aggregate_results(
        validation_results, protocol=validation_protocol
    )
    _atomic_write_json(
        output_root
        / "ood"
        / "family_logo"
        / "locked_validation_aggregate.json",
        validation_aggregate,
    )
    aggregate["locked_validation"] = {
        "n_test": validation_aggregate["n_test"],
        "path": str(
            (
                output_root
                / "ood"
                / "family_logo"
                / "locked_validation_aggregate.json"
            ).resolve()
        ),
    }
    _atomic_write_json(aggregate_path, aggregate)
    return aggregate


def _dataset_cohort(row: Mapping[str, Any]) -> str:
    labels = row.get("group_labels")
    if not isinstance(labels, Mapping):
        raise ValueError("row.group_labels must be a mapping")
    value = labels.get("dataset_cohort")
    if not isinstance(value, str) or not value:
        raise ValueError("row.group_labels.dataset_cohort must be a non-empty string")
    return value


def fit_holdout_candidate_ood(
    corpus: WideCorpus,
    output_root: Path,
    *,
    definitions: Sequence[SearchDefinition],
    plan_sha256: str,
    source_hashes: Mapping[str, str],
    jobs: int,
    seed: int,
    n_splits: int,
    force: bool,
) -> dict[str, Any]:
    """Run the final predeclared holdout-candidate dataset evaluation."""

    protocol = {
        "name": "holdout_candidate_dataset_ood",
        "outer_axis": "dataset",
        "outer_label": "dataset_cohort=holdout_candidate",
        "family_conditioned": True,
        "train": "non-holdout-candidate datasets' train_rows",
        "test": "holdout-candidate datasets' all_rows",
        "outer_test_used_for_selection": False,
    }
    # A dataset must have exactly one frozen cohort across every environment.
    cohorts_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in corpus.all_rows:
        cohorts_by_dataset[group_value(row, "dataset")].add(_dataset_cohort(row))
    inconsistent = {
        dataset: sorted(cohorts)
        for dataset, cohorts in cohorts_by_dataset.items()
        if len(cohorts) != 1
    }
    if inconsistent:
        raise ValueError(f"Datasets mix dataset_cohort labels: {inconsistent}")

    results: list[TaskResult] = []
    for family in _families(corpus):
        family_train = _rows_for_family(corpus.train_rows, family)
        family_all = _rows_for_family(corpus.all_rows, family)
        held_out_datasets = sorted(
            {
                group_value(row, "dataset")
                for row in family_all
                if _dataset_cohort(row) == "holdout_candidate"
            }
        )
        if not held_out_datasets:
            raise ValueError(f"{family} has no holdout_candidate dataset")
        held_set = set(held_out_datasets)
        train_rows = [
            row for row in family_train if group_value(row, "dataset") not in held_set
        ]
        test_rows = [
            row for row in family_all if group_value(row, "dataset") in held_set
        ]
        if {group_value(row, "dataset") for row in train_rows}.intersection(held_set):
            raise AssertionError("Holdout-candidate dataset leaked into training")
        fold_plan = _group_fold_plan(
            train_rows, axis="dataset", n_splits=n_splits, seed=seed
        )
        results.append(
            _run_task(
                task_id=f"{family}:holdout_candidate",
                train_rows=train_rows,
                test_rows=test_rows,
                output_dir=output_root / "ood" / "holdout_candidate" / family,
                fold_plan=fold_plan,
                definitions=definitions,
                protocol={
                    **protocol,
                    "family": family,
                    "held_out_datasets": held_out_datasets,
                },
                plan_sha256=plan_sha256,
                source_hashes=source_hashes,
                jobs=jobs,
                seed=seed,
                force=force,
            )
        )
    aggregate = _aggregate_results(results, protocol=protocol)
    _atomic_write_json(
        output_root / "ood" / "holdout_candidate" / "aggregate.json", aggregate
    )
    return aggregate


def _load_run_inputs(
    dataset_root: Path | None,
    plan_path: Path | None,
    *,
    require_complete: bool,
    phases: set[str] | None = None,
    snapshot_manifest: Path | None = None,
) -> tuple[
    WideCorpus,
    str,
    dict[str, str],
    dict[str, Any],
    dict[str, str | None],
]:
    if snapshot_manifest is not None:
        if dataset_root is not None or plan_path is not None:
            raise ValueError(
                "snapshot_manifest is mutually exclusive with dataset_root/plan_path"
            )
        if phases:
            raise ValueError("snapshot_manifest is mutually exclusive with phases")
        snapshot = load_snapshot(snapshot_manifest)
        corpus = snapshot.corpus
        validation = {
            "status": "complete",
            "source": "snapshot_manifest",
            "snapshot_manifest_path": str(snapshot.path),
            "snapshot_manifest_sha256": snapshot.sha256,
            "counts": dict(snapshot.manifest["counts"]),
            "fingerprints_sha256": snapshot.manifest["fingerprints_sha256"],
            "completed_environments": [
                environment.experiment_id for environment in corpus.environments
            ],
        }
        provenance = {
            "dataset_root": None,
            "plan_path": None,
            "plan_sha256": None,
            "snapshot_manifest_path": str(snapshot.path),
            "snapshot_manifest_sha256": snapshot.sha256,
        }
        return corpus, snapshot.sha256, _source_hashes(), validation, provenance

    if dataset_root is None or plan_path is None:
        raise ValueError("dataset_root and plan_path are required without a snapshot")
    report = validate_root(dataset_root, plan_path=plan_path)
    if report["status"] == "invalid":
        raise ValueError(
            "Wide dataset validation failed: "
            + json.dumps(report["validation_errors"], ensure_ascii=False)
        )
    if require_complete and report["status"] != "complete":
        raise ValueError(
            f"Wide dataset is {report['status']}; --require-complete was requested"
        )
    corpus = load_corpus(
        dataset_root,
        expected_n_seeds=report["expected"]["n_seeds"],
        require_no_partial=require_complete,
    )
    if phases:
        plan = json.loads(plan_path.read_text("utf-8"))
        default_phase = str(plan["defaults"]["phase"])
        expected = {
            str(item["experiment_id"])
            for item in plan["experiments"]
            if str(item.get("phase", default_phase)) in phases
        }
        environments = tuple(
            environment
            for environment in corpus.environments
            if str(environment.manifest["config"]["phase"]) in phases
        )
        actual = {environment.experiment_id for environment in environments}
        if actual != expected:
            raise ValueError(
                "Selected phases are incomplete: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        corpus = WideCorpus(
            root=corpus.root,
            environments=environments,
            n_seeds=corpus.n_seeds,
            all_rows=tuple(
                row for environment in environments for row in environment.all_rows
            ),
            train_rows=tuple(
                row for environment in environments for row in environment.train_rows
            ),
            validation_rows=tuple(
                row
                for environment in environments
                for row in environment.validation_rows
            ),
        )
        report = {
            **report,
            "selected_phases": sorted(phases),
            "selected_phase_environments": sorted(actual),
        }
    plan_sha256 = _sha256_file(plan_path.resolve())
    provenance = {
        "dataset_root": str(dataset_root.resolve()),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "snapshot_manifest_path": None,
        "snapshot_manifest_sha256": None,
    }
    return corpus, plan_sha256, _source_hashes(), report, provenance


def fit_wide_id(
    dataset_root: Path | None,
    output_root: Path,
    plan_path: Path | None,
    *,
    jobs: int,
    seed: int,
    n_splits: int,
    method_names: set[str] | None,
    include_xgboost: bool,
    scopes: set[str],
    force: bool,
    noise_ceiling: str,
    require_complete: bool,
    phases: set[str] | None = None,
    include_parameter_count: bool = True,
    snapshot_manifest: Path | None = None,
    dataset_conditioning: str = "unaware",
) -> dict[str, Any]:
    validate_jobs(jobs)
    unknown_scopes = scopes.difference(ID_SCOPES)
    if unknown_scopes:
        raise ValueError(f"Unknown ID scopes: {', '.join(sorted(unknown_scopes))}")
    corpus, contract_sha256, source_hashes, validation, provenance = _load_run_inputs(
        dataset_root,
        plan_path,
        require_complete=require_complete,
        phases=phases,
        snapshot_manifest=snapshot_manifest,
    )
    definitions = _definition_subset(
        wide_search_definitions(
            seed,
            include_xgboost=include_xgboost,
            include_parameter_count=include_parameter_count,
            dataset_conditioning=dataset_conditioning,
        ),
        method_names,
    )
    started = _utc_now()
    results: dict[str, Any] = {}
    if "environment" in scopes:
        results["environment"] = fit_per_environment_id(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
            noise_ceiling=noise_ceiling,
        )
    if "dataset" in scopes:
        results["dataset"] = fit_dataset_pooled_id(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
        )
    if "family" in scopes:
        results["family"] = fit_family_pooled_id(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
        )
    if "global" in scopes:
        results["global"] = fit_global_pooled_id(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "command": "fit-id",
        "dataset_root": provenance["dataset_root"],
        "output_root": str(output_root.resolve()),
        "plan_path": provenance["plan_path"],
        "plan_sha256": provenance["plan_sha256"],
        "snapshot_manifest_path": provenance["snapshot_manifest_path"],
        "snapshot_manifest_sha256": provenance["snapshot_manifest_sha256"],
        "source_hashes": source_hashes,
        "dataset_validation": validation,
        "n_seeds": corpus.n_seeds,
        "n_environments": len(corpus.environments),
        "phases": sorted(phases) if phases else None,
        "include_parameter_count": include_parameter_count,
        "dataset_conditioning": dataset_conditioning,
        "methods": [definition.name for definition in definitions],
        "scopes": sorted(scopes),
        "jobs": jobs,
        "thread_limit": 1,
        "seed": seed,
        "n_splits": n_splits,
        "noise_ceiling": noise_ceiling,
        "started_at": started,
        "completed_at": _utc_now(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "results": {
            name: {
                "n_tasks": result["n_tasks"],
                "n_test": result["n_test"],
            }
            for name, result in results.items()
        },
    }
    _atomic_write_json(output_root / "id" / "manifest.json", manifest)
    return manifest


def fit_wide_grouped(
    dataset_root: Path | None,
    output_root: Path,
    plan_path: Path | None,
    *,
    jobs: int,
    seed: int,
    n_splits: int,
    method_names: set[str] | None,
    include_xgboost: bool,
    protocols: set[str],
    force: bool,
    require_complete: bool,
    phases: set[str] | None = None,
    include_parameter_count: bool = True,
    snapshot_manifest: Path | None = None,
    dataset_conditioning: str = "unaware",
) -> dict[str, Any]:
    validate_jobs(jobs)
    unknown = protocols.difference(GROUPED_PROTOCOLS)
    if unknown:
        raise ValueError(f"Unknown grouped protocols: {', '.join(sorted(unknown))}")
    corpus, contract_sha256, source_hashes, validation, provenance = _load_run_inputs(
        dataset_root,
        plan_path,
        require_complete=require_complete,
        phases=phases,
        snapshot_manifest=snapshot_manifest,
    )
    definitions = _definition_subset(
        wide_search_definitions(
            seed,
            include_xgboost=include_xgboost,
            include_parameter_count=include_parameter_count,
            dataset_conditioning=dataset_conditioning,
        ),
        method_names,
    )
    started = _utc_now()
    results: dict[str, Any] = {}
    for axis in ("environment", "dataset"):
        if axis in protocols:
            results[axis] = _fit_logo_protocol(
                corpus,
                output_root,
                axis=axis,
                definitions=definitions,
                plan_sha256=contract_sha256,
                source_hashes=source_hashes,
                jobs=jobs,
                seed=seed,
                n_splits=n_splits,
                force=force,
            )
    if "holdout_candidate" in protocols:
        results["holdout_candidate"] = fit_holdout_candidate_ood(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
        )
    if "family" in protocols:
        results["family"] = fit_family_logo_ood(
            corpus,
            output_root,
            definitions=definitions,
            plan_sha256=contract_sha256,
            source_hashes=source_hashes,
            jobs=jobs,
            seed=seed,
            n_splits=n_splits,
            force=force,
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "command": "fit-grouped",
        "dataset_root": provenance["dataset_root"],
        "output_root": str(output_root.resolve()),
        "plan_path": provenance["plan_path"],
        "plan_sha256": provenance["plan_sha256"],
        "snapshot_manifest_path": provenance["snapshot_manifest_path"],
        "snapshot_manifest_sha256": provenance["snapshot_manifest_sha256"],
        "source_hashes": source_hashes,
        "dataset_validation": validation,
        "n_seeds": corpus.n_seeds,
        "n_environments": len(corpus.environments),
        "phases": sorted(phases) if phases else None,
        "include_parameter_count": include_parameter_count,
        "dataset_conditioning": dataset_conditioning,
        "methods": [definition.name for definition in definitions],
        "protocols": sorted(protocols),
        "jobs": jobs,
        "thread_limit": 1,
        "seed": seed,
        "n_splits": n_splits,
        "started_at": started,
        "completed_at": _utc_now(),
        "results": {
            name: {
                "n_tasks": result["n_tasks"],
                "n_test": result["n_test"],
            }
            for name, result in results.items()
        },
    }
    _atomic_write_json(output_root / "ood" / "manifest.json", manifest)
    return manifest


def _parse_names(value: str | None, *, defaults: Sequence[str]) -> set[str]:
    if value is None:
        return set(defaults)
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names:
        raise argparse.ArgumentTypeError("comma-separated list must not be empty")
    return names


def _parse_methods(value: str | None) -> set[str] | None:
    if value is None:
        return None
    methods = {item.strip() for item in value.split(",") if item.strip()}
    if not methods:
        raise argparse.ArgumentTypeError("--methods must not be empty")
    return methods


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=None)
    source.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=None,
        help="frozen cross-root completed-GT snapshot manifest",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=MAX_JOBS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--methods", help="optional comma-separated method subset for smoke/resume"
    )
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--phases",
        default=None,
        help="optional comma-separated completed phases; every planned environment in the selected phases must exist",
    )
    parser.add_argument("--exclude-parameter-count", action="store_true")
    parser.add_argument(
        "--dataset-conditioning",
        choices=DATASET_CONDITIONING,
        default="unaware",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_id = subparsers.add_parser(
        "fit-id", help="fit per-environment/family ID models"
    )
    _add_common_arguments(fit_id)
    fit_id.add_argument(
        "--scopes",
        default=",".join(ID_SCOPES),
        help="environment,dataset,family,global (comma-separated)",
    )
    fit_id.add_argument(
        "--noise-ceiling",
        choices=("auto", "skip", "require"),
        default="auto",
    )

    grouped = subparsers.add_parser(
        "fit-grouped",
        help="fit environment/dataset/family LOGO and final holdout",
    )
    _add_common_arguments(grouped)
    grouped.add_argument(
        "--protocols",
        default=",".join(GROUPED_PROTOCOLS),
        help="environment,dataset,holdout_candidate,family (comma-separated)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_jobs(args.jobs)
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    method_names = _parse_methods(args.methods)
    phases = _parse_names(args.phases, defaults=()) if args.phases else None
    if args.snapshot_manifest is not None and phases:
        parser.error("--snapshot-manifest is mutually exclusive with --phases")
    if args.snapshot_manifest is not None and args.plan is not None:
        parser.error("--snapshot-manifest does not use --plan")
    if args.snapshot_manifest is not None:
        dataset_root = None
        plan_path = None
        snapshot_manifest = args.snapshot_manifest.resolve()
    else:
        dataset_root = (args.dataset_root or DEFAULT_DATASET_ROOT).resolve()
        plan_path = (args.plan or DEFAULT_PLAN).resolve()
        snapshot_manifest = None
    if args.command == "fit-id":
        fit_wide_id(
            dataset_root,
            args.output_root.resolve(),
            plan_path,
            jobs=args.jobs,
            seed=args.seed,
            n_splits=args.folds,
            method_names=method_names,
            include_xgboost=args.include_xgboost,
            scopes=_parse_names(args.scopes, defaults=ID_SCOPES),
            force=args.force,
            noise_ceiling=args.noise_ceiling,
            require_complete=args.require_complete,
            phases=phases,
            include_parameter_count=not args.exclude_parameter_count,
            snapshot_manifest=snapshot_manifest,
            dataset_conditioning=args.dataset_conditioning,
        )
    elif args.command == "fit-grouped":
        fit_wide_grouped(
            dataset_root,
            args.output_root.resolve(),
            plan_path,
            jobs=args.jobs,
            seed=args.seed,
            n_splits=args.folds,
            method_names=method_names,
            include_xgboost=args.include_xgboost,
            protocols=_parse_names(args.protocols, defaults=GROUPED_PROTOCOLS),
            force=args.force,
            require_complete=args.require_complete,
            phases=phases,
            include_parameter_count=not args.exclude_parameter_count,
            snapshot_manifest=snapshot_manifest,
            dataset_conditioning=args.dataset_conditioning,
        )
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
