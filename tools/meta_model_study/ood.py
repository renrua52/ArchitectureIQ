"""Group-held-out evaluation for setting-to-loss meta-models.

This study asks a stricter question than an IID row split: can a meta-model
predict a setting whose *entire optimizer, learning rate, architecture size,
or optimizer-by-learning-rate cell* was absent from fitting?  Each protocol
is exhaustive group cross-fitting over the 1,000 selected rows of one dataset
family.  A row is a test example exactly once per protocol, and the estimator
for its fold is fitted only on rows outside the held group.

The four estimator configurations are fixed in this source before any OOD
score is observed.  There is intentionally no hyperparameter search: an inner
ordinary random split could still expose the held-out group, while a nested
group search would make this already-large diagnostic needlessly expensive.
The target is always ``log(mean_loss)``.

Run from the repository root with::

    .venv/bin/python -m tools.meta_model_study.ood

Results are written to ``data/meta_model_studies/setting_to_loss_60q_id_v1/ood``.
Per-fold prediction checkpoints make the command safe to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from tools.meta_model_study.features import FeatureEncoder, load_jsonl
from tools.meta_model_study.metrics import ranking_metrics, regression_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/meta_model/setting_to_loss_60q_id_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/meta_model_studies/setting_to_loss_60q_id_v1/ood"
)

SCHEMA_VERSION = "meta_model_ood_v1"
PROTOCOLS = (
    "leave_one_optimizer_out",
    "leave_one_lr_out",
    "leave_one_size_out",
    "leave_one_optimizer_lr_cell_out",
)
METHODS = (
    "params_ridge",
    "compact_polynomial_ridge",
    "extra_trees",
    "xgboost",
)

OPTIMIZERS = ("Adagrad", "Adam", "AdamW", "RMSprop", "SGD")
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003, 0.01)
SIZES_BY_FAMILY: dict[str, tuple[int, ...]] = {
    "univariate_regression": (16, 32, 64, 128, 256),
    "multivariate_regression": (16, 32, 64, 128, 256),
    "bigram_lm": (32, 64, 128),
}

# These settings are deliberately shared across every family and OOD fold.
# They are a runtime-conscious, regularized middle of the original search
# grids, not values selected by looking at any OOD result.
FIXED_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "params_ridge": {
        "feature_set": "params",
        "alpha": 1.0,
    },
    "compact_polynomial_ridge": {
        "feature_set": "compact",
        "degree": 2,
        "include_bias": False,
        "alpha": 100.0,
    },
    "extra_trees": {
        "feature_set": "full",
        "n_estimators": 256,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
    "xgboost": {
        "feature_set": "full",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
    },
}


@dataclass(frozen=True)
class GroupFold:
    """One exhaustive held-group fold."""

    group: str
    train_indices: np.ndarray
    test_indices: np.ndarray


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
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
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


def _lr_text(value: float) -> str:
    return format(float(value), ".12g")


def _optimizer(row: Mapping[str, Any]) -> str:
    return str(row["setting"]["optimizer"]["type"])


def _learning_rate(row: Mapping[str, Any]) -> float:
    return float(row["setting"]["optimizer"]["lr"])


def _size_field_and_value(row: Mapping[str, Any]) -> tuple[str, int]:
    model = row["setting"]["model"]
    if "width" in model:
        return "width", int(model["width"])
    if "d_model" in model:
        return "d_model", int(model["d_model"])
    raise KeyError("setting.model must contain width or d_model")


def group_value(row: Mapping[str, Any], protocol: str) -> str:
    """Return the target-free group key for one predeclared protocol."""

    if protocol == "leave_one_optimizer_out":
        return f"optimizer={_optimizer(row)}"
    if protocol == "leave_one_lr_out":
        return f"lr={_lr_text(_learning_rate(row))}"
    if protocol == "leave_one_size_out":
        field, value = _size_field_and_value(row)
        return f"{field}={value}"
    if protocol == "leave_one_optimizer_lr_cell_out":
        return f"optimizer={_optimizer(row)}|lr={_lr_text(_learning_rate(row))}"
    known = ", ".join(PROTOCOLS)
    raise ValueError(f"Unknown OOD protocol {protocol!r}; expected one of: {known}")


def expected_groups(family: str, protocol: str) -> tuple[str, ...]:
    """Enumerate groups from the profile grid without consulting targets."""

    if family not in SIZES_BY_FAMILY:
        raise ValueError(f"No predeclared size grid for family {family!r}")
    if protocol == "leave_one_optimizer_out":
        return tuple(f"optimizer={value}" for value in OPTIMIZERS)
    if protocol == "leave_one_lr_out":
        return tuple(f"lr={_lr_text(value)}" for value in LEARNING_RATES)
    if protocol == "leave_one_size_out":
        field = "d_model" if family == "bigram_lm" else "width"
        return tuple(f"{field}={value}" for value in SIZES_BY_FAMILY[family])
    if protocol == "leave_one_optimizer_lr_cell_out":
        return tuple(
            f"optimizer={optimizer}|lr={_lr_text(lr)}"
            for optimizer in OPTIMIZERS
            for lr in LEARNING_RATES
        )
    known = ", ".join(PROTOCOLS)
    raise ValueError(f"Unknown OOD protocol {protocol!r}; expected one of: {known}")


def make_group_folds(
    rows: Sequence[Mapping[str, Any]],
    protocol: str,
    *,
    declared_groups: Sequence[str] | None = None,
) -> list[GroupFold]:
    """Make exhaustive leave-one-group-out folds and audit their partition."""

    if not rows:
        raise ValueError("Cannot build OOD folds from an empty row sequence")
    keys = np.asarray([group_value(row, protocol) for row in rows], dtype=object)
    observed = set(str(value) for value in keys)
    if declared_groups is None:
        groups = tuple(sorted(observed))
    else:
        groups = tuple(declared_groups)
        if len(groups) != len(set(groups)):
            raise ValueError(f"{protocol} declared_groups contains duplicates")
        missing = set(groups) - observed
        unexpected = observed - set(groups)
        if missing or unexpected:
            raise ValueError(
                f"{protocol} group grid mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    folds: list[GroupFold] = []
    all_test_indices: list[np.ndarray] = []
    for group in groups:
        test = np.flatnonzero(keys == group).astype(np.int64, copy=False)
        train = np.flatnonzero(keys != group).astype(np.int64, copy=False)
        if test.size == 0 or train.size == 0:
            raise ValueError(
                f"{protocol}/{group} must have non-empty train and test partitions"
            )
        if set(keys[train]).intersection(set(keys[test])):
            raise AssertionError(f"Group leakage in {protocol}/{group}")
        folds.append(GroupFold(group, train, test))
        all_test_indices.append(test)

    concatenated = np.concatenate(all_test_indices)
    if not np.array_equal(np.sort(concatenated), np.arange(len(rows))):
        raise AssertionError(f"{protocol} test folds are not an exact row partition")
    return folds


def exact_three_choice_accuracy(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, int | float | None]:
    """Count exact three-choice winners without materializing all triples.

    Sort by predicted loss (and original row index for deterministic predicted
    ties).  A row is selected for every triple containing it and two rows
    later in this order.  Such a triple is correct exactly when both later
    rows have true loss at least as large.  This gives an exact O(n^2) count
    and preserves full credit for true-loss ties.
    """

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("y_true and y_pred must be finite")
    n_rows = int(truth.size)
    if n_rows < 3:
        return {"n_groups": 0, "n_correct": 0, "accuracy": None}

    original_order = np.arange(n_rows)
    predicted_order = np.lexsort((original_order, prediction))
    ordered_truth = truth[predicted_order]
    n_correct = 0
    for position in range(n_rows - 2):
        later_not_better = int(
            np.count_nonzero(ordered_truth[position + 1 :] >= ordered_truth[position])
        )
        if later_not_better >= 2:
            n_correct += math.comb(later_not_better, 2)
    n_groups = math.comb(n_rows, 3)
    return {
        "n_groups": n_groups,
        "n_correct": n_correct,
        "accuracy": n_correct / n_groups,
    }


def prediction_metrics(
    y_true_log: Sequence[float] | np.ndarray,
    y_pred_log: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Score one held group in log-loss space."""

    regression = regression_metrics(y_true_log, y_pred_log)
    ranking = ranking_metrics(y_true_log, y_pred_log)
    return {
        "n": len(y_true_log),
        "log_mae": regression["mae"],
        "log_rmse": regression["rmse"],
        "log_r2": regression["r2"],
        "spearman": ranking["spearman"],
        "three_choice": exact_three_choice_accuracy(y_true_log, y_pred_log),
    }


def fixed_estimators(*, seed: int, jobs: int) -> dict[str, BaseEstimator]:
    """Construct the four fixed, leakage-safe comparison estimators."""

    if jobs < 1:
        raise ValueError("jobs must be at least one")
    estimators: dict[str, BaseEstimator] = {
        "params_ridge": Pipeline(
            [
                ("features", FeatureEncoder(feature_set="params")),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "compact_polynomial_ridge": Pipeline(
            [
                ("features", FeatureEncoder(feature_set="compact")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=100.0)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("features", FeatureEncoder(feature_set="full")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=256,
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features=0.7,
                        random_state=seed,
                        n_jobs=jobs,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - exercised by deployment only
        raise RuntimeError(
            "XGBoost is required; install the project's meta-model extra"
        ) from exc
    estimators["xgboost"] = Pipeline(
        [
            ("features", FeatureEncoder(feature_set="full")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=4,
                    min_child_weight=5,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=10.0,
                    random_state=seed,
                    n_jobs=jobs,
                    verbosity=0,
                ),
            ),
        ]
    )
    return estimators


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def summarize_folds(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[GroupFold],
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine fold predictions without forming cross-group three-choice items."""

    if len(folds) != len(fold_results):
        raise ValueError("folds and fold_results must have the same length")
    truth = np.asarray(
        [float(row["target"]["log_mean_loss"]) for row in rows], dtype=float
    )
    oof = np.full(len(rows), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold, result in zip(folds, fold_results):
        if result["group"] != fold.group:
            raise ValueError("Fold result order/group mismatch")
        predictions = np.asarray(result["predictions_log"], dtype=float)
        if predictions.shape != (fold.test_indices.size,):
            raise ValueError(f"Prediction shape mismatch for {fold.group}")
        oof[fold.test_indices] = predictions
        metrics = prediction_metrics(truth[fold.test_indices], predictions)
        fold_metrics.append(
            {
                "group": fold.group,
                "n_train": int(fold.train_indices.size),
                "n_test": int(fold.test_indices.size),
                "metrics": metrics,
            }
        )
    if not np.all(np.isfinite(oof)):
        raise AssertionError("OOD folds did not produce one finite prediction per row")

    micro = prediction_metrics(truth, oof)
    # A global triple could mix three separately held groups and therefore does
    # not correspond to any one trained estimator.  Replace that number by the
    # exact weighted total across within-fold test triples.
    n_three = sum(
        int(item["metrics"]["three_choice"]["n_groups"])
        for item in fold_metrics
    )
    correct_three = sum(
        int(item["metrics"]["three_choice"]["n_correct"])
        for item in fold_metrics
    )
    micro["three_choice"] = {
        "n_groups": n_three,
        "n_correct": correct_three,
        "accuracy": correct_three / n_three if n_three else None,
        "scope": "within_held_group_only",
    }
    macro = {
        metric: _mean_defined(
            [item["metrics"][metric] for item in fold_metrics]
        )
        for metric in ("log_mae", "log_rmse", "log_r2", "spearman")
    }
    macro["three_choice_accuracy"] = _mean_defined(
        [item["metrics"]["three_choice"]["accuracy"] for item in fold_metrics]
    )
    return {
        "micro": micro,
        "macro_fold_mean": macro,
        "folds": fold_metrics,
        "predictions": [
            {
                "example_fingerprint_sha256": row[
                    "example_fingerprint_sha256"
                ],
                "prediction_log_mean_loss": float(prediction),
            }
            for row, prediction in zip(rows, oof)
        ],
    }


def _checkpoint_slug(group: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9]+", "_", group).strip("_")[:60]
    suffix = hashlib.sha256(group.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{suffix}"


def _fold_digest(
    *,
    data_digest: str,
    protocol: str,
    group: str,
    method: str,
    seed: int,
) -> str:
    source_dir = Path(__file__).resolve().parent
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_sha256": data_digest,
        "ood_source_sha256": _sha256_file(Path(__file__)),
        "features_source_sha256": _sha256_file(source_dir / "features.py"),
        "protocol": protocol,
        "group": group,
        "method": method,
        "model_config": FIXED_MODEL_CONFIGS[method],
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fit_or_load_fold(
    *,
    rows: Sequence[Mapping[str, Any]],
    fold: GroupFold,
    estimator: BaseEstimator,
    checkpoint: Path,
    digest: str,
    force: bool,
) -> dict[str, Any]:
    expected_fingerprints = [
        rows[index]["example_fingerprint_sha256"] for index in fold.test_indices
    ]
    if checkpoint.is_file() and not force:
        saved = json.loads(checkpoint.read_text("utf-8"))
        if (
            saved.get("input_digest") == digest
            and saved.get("group") == fold.group
            and saved.get("test_fingerprints") == expected_fingerprints
        ):
            predictions = np.asarray(saved.get("predictions_log"), dtype=float)
            if predictions.shape == (fold.test_indices.size,) and np.all(
                np.isfinite(predictions)
            ):
                return saved

    train_rows = [rows[index] for index in fold.train_indices]
    test_rows = [rows[index] for index in fold.test_indices]
    y_train = np.asarray(
        [float(row["target"]["log_mean_loss"]) for row in train_rows], dtype=float
    )
    fitted = clone(estimator).fit(np.asarray(train_rows, dtype=object), y_train)
    predictions = np.asarray(
        fitted.predict(np.asarray(test_rows, dtype=object)), dtype=float
    )
    if predictions.shape != (len(test_rows),) or not np.all(np.isfinite(predictions)):
        raise ValueError(f"Non-finite/invalid predictions for held group {fold.group}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_digest": digest,
        "group": fold.group,
        "test_fingerprints": expected_fingerprints,
        "predictions_log": predictions.tolist(),
        "written_at": _utc_now(),
    }
    _atomic_write_json(checkpoint, result)
    return result


def _validate_dataset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    expected_row_count: int | None,
) -> str:
    if expected_row_count is not None and len(rows) != expected_row_count:
        raise ValueError(
            f"{experiment_id} contains {len(rows)} rows; expected {expected_row_count}"
        )
    if not rows:
        raise ValueError(f"{experiment_id} has no selected rows")
    families = {str(row.get("family")) for row in rows}
    if len(families) != 1:
        raise ValueError(f"{experiment_id} contains multiple families: {families}")
    fingerprints: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{experiment_id}[{index}]"
        if row.get("experiment_id") != experiment_id:
            raise ValueError(f"{context} has mismatched experiment_id")
        if row.get("usable_for_regression") is not True:
            raise ValueError(f"{context} is not usable_for_regression")
        fingerprint = str(row.get("example_fingerprint_sha256", ""))
        if len(fingerprint) != 64 or fingerprint in fingerprints:
            raise ValueError(f"{context} has invalid/duplicate fingerprint")
        fingerprints.add(fingerprint)
        mean_loss = float(row["target"]["mean_loss"])
        log_loss = float(row["target"]["log_mean_loss"])
        if not math.isfinite(mean_loss) or mean_loss <= 0.0:
            raise ValueError(f"{context} has invalid mean_loss")
        if not math.isclose(math.log(mean_loss), log_loss, abs_tol=1e-12):
            raise ValueError(f"{context} mean_loss/log_mean_loss disagree")
    return next(iter(families))


def discover_experiments(dataset_root: Path) -> list[Path]:
    """Find completed experiment directories containing all 1,000 rows."""

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    experiments = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir()
            and (path / "all.jsonl").is_file()
            and (path / "manifest.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not experiments:
        raise ValueError(f"No completed experiments found under {dataset_root}")
    return experiments


def evaluate_experiment(
    experiment_dir: Path,
    *,
    output_root: Path,
    protocols: Sequence[str] = PROTOCOLS,
    methods: Sequence[str] = METHODS,
    seed: int = 20260714,
    jobs: int = 1,
    expected_row_count: int | None = 1000,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run all requested group-held-out fits for one dataset family."""

    unknown_protocols = set(protocols) - set(PROTOCOLS)
    unknown_methods = set(methods) - set(METHODS)
    if unknown_protocols:
        raise ValueError(f"Unknown protocols: {sorted(unknown_protocols)}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    rows = load_jsonl(experiment_dir / "all.jsonl")
    experiment_id = experiment_dir.name
    family = _validate_dataset_rows(
        rows,
        experiment_id=experiment_id,
        expected_row_count=expected_row_count,
    )
    estimators = fixed_estimators(seed=seed, jobs=jobs)
    data_digest = _sha256_file(experiment_dir / "all.jsonl")
    protocol_results: dict[str, Any] = {}

    for protocol in protocols:
        declared = expected_groups(family, protocol)
        folds = make_group_folds(rows, protocol, declared_groups=declared)
        method_results: dict[str, Any] = {}
        for method in methods:
            fold_results: list[dict[str, Any]] = []
            for fold_number, fold in enumerate(folds, start=1):
                if progress is not None:
                    progress(
                        f"{experiment_id} | {protocol} | {method} | "
                        f"{fold_number}/{len(folds)} {fold.group}"
                    )
                digest = _fold_digest(
                    data_digest=data_digest,
                    protocol=protocol,
                    group=fold.group,
                    method=method,
                    seed=seed,
                )
                checkpoint = (
                    output_root
                    / "checkpoints"
                    / experiment_id
                    / protocol
                    / method
                    / f"{_checkpoint_slug(fold.group)}.json"
                )
                fold_results.append(
                    _fit_or_load_fold(
                        rows=rows,
                        fold=fold,
                        estimator=estimators[method],
                        checkpoint=checkpoint,
                        digest=digest,
                        force=force,
                    )
                )
            method_results[method] = summarize_folds(rows, folds, fold_results)
        protocol_results[protocol] = {
            "group_axis": protocol,
            "declared_groups": list(declared),
            "n_folds": len(folds),
            "methods": method_results,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "family": family,
        "n_rows": len(rows),
        "all_jsonl_sha256": data_digest,
        "protocols": protocol_results,
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact, auditable report from the machine-readable result."""

    lines = [
        "# Setting-to-loss grouped OOD evaluation",
        "",
        "## Protocol",
        "",
        (
            "This is exhaustive group cross-fitting over all 1,000 selected rows "
            "per family. Each fold fits only on rows outside one optimizer, LR, "
            "width/`d_model`, or optimizer×LR cell and predicts every row in that "
            "held group. Every row is tested exactly once per protocol."
        ),
        "",
        (
            "The target is `log(mean_loss)`. All four model configurations are "
            "fixed globally before OOD evaluation; there is no holdout-aware "
            "hyperparameter selection. Three-choice accuracy is exact over every "
            "triple contained within each held group (never across folds)."
        ),
        "",
        (
            "This reuses the former 900/100 split as one 1,000-row group-cross-fit "
            "dataset. It measures compositional interpolation/extrapolation and is "
            "not an additional untouched holdout."
        ),
        "",
        "### Fixed estimators",
        "",
        "```json",
        json.dumps(FIXED_MODEL_CONFIGS, indent=2, sort_keys=True),
        "```",
        "",
        "## Aggregate results",
        "",
        (
            "| Family | Held-out axis | Method | Folds | Log RMSE | Log R² | "
            "Spearman | Exact 3-choice |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    experiments = result["experiments"]
    for experiment_id in sorted(experiments):
        experiment = experiments[experiment_id]
        family = experiment["family"]
        for protocol in PROTOCOLS:
            if protocol not in experiment["protocols"]:
                continue
            protocol_result = experiment["protocols"][protocol]
            for method in METHODS:
                if method not in protocol_result["methods"]:
                    continue
                metrics = protocol_result["methods"][method]["micro"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            family,
                            protocol,
                            method,
                            str(protocol_result["n_folds"]),
                            _format_number(metrics["log_rmse"]),
                            _format_number(metrics["log_r2"]),
                            _format_number(metrics["spearman"]),
                            _format_number(
                                metrics["three_choice"]["accuracy"] * 100.0,
                                2,
                            )
                            + "%",
                        ]
                    )
                    + " |"
                )

    lines.extend(["", "## Best tested method by family and axis", ""])
    for experiment_id in sorted(experiments):
        experiment = experiments[experiment_id]
        lines.extend([f"### {experiment['family']}", ""])
        for protocol in PROTOCOLS:
            if protocol not in experiment["protocols"]:
                continue
            method_results = experiment["protocols"][protocol]["methods"]
            winner = min(
                method_results,
                key=lambda method: method_results[method]["micro"]["log_rmse"],
            )
            metrics = method_results[winner]["micro"]
            worst_fold = max(
                method_results[winner]["folds"],
                key=lambda item: item["metrics"]["log_rmse"],
            )
            lines.append(
                f"- `{protocol}`: **{winner}**, log RMSE "
                f"{_format_number(metrics['log_rmse'])}, exact three-choice "
                f"{_format_number(metrics['three_choice']['accuracy'] * 100, 2)}%; "
                f"hardest group `{worst_fold['group']}` "
                f"(RMSE {_format_number(worst_fold['metrics']['log_rmse'])})."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation cautions",
            "",
            (
                "- Leave-one-optimizer-out is true category extrapolation. An "
                "unseen optimizer's one-hot and optimizer-specific LR interaction "
                "cannot be learned, so a low score is expected and informative."
            ),
            (
                "- Leave-one-LR/size-out tests interpolation at interior grid "
                "values and extrapolation at grid endpoints; inspect fold rows "
                "before calling either result a general OOD law."
            ),
            (
                "- Cell holdout is the cleanest compositional test: the optimizer "
                "and LR are individually observed during training, but their exact "
                "combination is not."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_study(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    protocols: Sequence[str] = PROTOCOLS,
    methods: Sequence[str] = METHODS,
    experiment_names: set[str] | None = None,
    seed: int = 20260714,
    jobs: int = 1,
    expected_row_count: int | None = 1000,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the complete grouped OOD study and write JSON plus Markdown."""

    experiments = discover_experiments(dataset_root)
    if experiment_names is not None:
        experiments = [path for path in experiments if path.name in experiment_names]
        missing = experiment_names - {path.name for path in experiments}
        if missing:
            raise ValueError(f"Unknown experiment directories: {sorted(missing)}")

    experiment_results: dict[str, Any] = {}
    for experiment_dir in experiments:
        experiment_results[experiment_dir.name] = evaluate_experiment(
            experiment_dir,
            output_root=output_root,
            protocols=protocols,
            methods=methods,
            seed=seed,
            jobs=jobs,
            expected_row_count=expected_row_count,
            force=force,
            progress=progress,
        )
        partial = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "dataset_root": str(dataset_root.resolve()),
            "seed": seed,
            "target": "log_mean_loss",
            "fixed_model_configs": FIXED_MODEL_CONFIGS,
            "experiments": experiment_results,
        }
        _atomic_write_json(output_root / "results.partial.json", partial)

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "dataset_root": str(dataset_root.resolve()),
        "seed": seed,
        "target": "log_mean_loss",
        "protocol": {
            "split": "exhaustive_leave_one_group_out",
            "hyperparameters": "globally_fixed_before_ood_evaluation",
            "three_choice_scope": "all_triples_within_each_held_group",
            "uses_all_original_splits": True,
        },
        "fixed_model_configs": FIXED_MODEL_CONFIGS,
        "experiments": experiment_results,
    }
    _atomic_write_json(output_root / "results.json", result)
    _atomic_write_text(output_root / "report.md", render_markdown(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional experiment directory names; defaults to all completed ones",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--expected-row-count",
        type=int,
        default=1000,
        help="Set to 0 to disable the production 1,000-row audit",
    )
    parser.add_argument("--force", action="store_true", help="Ignore fold checkpoints")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least one")
    expected_rows = args.expected_row_count or None
    run_study(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        protocols=tuple(args.protocols),
        methods=tuple(args.methods),
        experiment_names=set(args.experiments) if args.experiments else None,
        seed=args.seed,
        jobs=args.jobs,
        expected_row_count=expected_rows,
        force=args.force,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(output_root := args.output_root / "results.json")
    print(args.output_root / "report.md")
    return 0 if output_root.is_file() else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
