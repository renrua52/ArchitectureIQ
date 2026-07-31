"""Train-only interpretation artifacts for setting-to-loss meta-models.

The study selects and evaluates models elsewhere.  This module deliberately
accepts only the training rows and already-fitted models, so descriptive rules
cannot accidentally inspect validation or external-question labels.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from tools.meta_model_study.models import (
    EnsembleModel,
    FittedModel,
    OptimizerLrLookupRegressor,
)


InterpretationModel = FittedModel | EnsembleModel


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _row_values(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    log_params: list[float] = []
    total_params: list[float] = []
    raw_losses: list[float] = []
    log_losses: list[float] = []
    optimizers: list[str] = []
    learning_rates: list[float] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"train row {index} must be a mapping, got {type(row).__name__}"
            )
        derived = row.get("derived")
        target = row.get("target")
        setting = row.get("setting")
        if not isinstance(derived, Mapping):
            raise TypeError(f"train row {index}.derived must be a mapping")
        if not isinstance(target, Mapping):
            raise TypeError(f"train row {index}.target must be a mapping")
        if not isinstance(setting, Mapping):
            raise TypeError(f"train row {index}.setting must be a mapping")

        supplied_total = derived.get("total_params")
        supplied_log_params = derived.get("log_total_params")
        if supplied_log_params is None and supplied_total is None:
            raise KeyError(
                f"train row {index}.derived must contain log_total_params "
                "or total_params"
            )
        if supplied_total is not None:
            total = _finite_float(
                supplied_total,
                name=f"train row {index}.derived.total_params",
            )
            if total <= 0.0:
                raise ValueError(
                    f"train row {index}.derived.total_params must be positive"
                )
        else:
            total = math.exp(
                _finite_float(
                    supplied_log_params,
                    name=f"train row {index}.derived.log_total_params",
                )
            )
        log_param = (
            math.log(total)
            if supplied_log_params is None
            else _finite_float(
                supplied_log_params,
                name=f"train row {index}.derived.log_total_params",
            )
        )
        if not math.isclose(log_param, math.log(total), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"train row {index} has inconsistent total_params and "
                "log_total_params"
            )

        supplied_raw_loss = target.get("mean_loss")
        supplied_log_loss = target.get("log_mean_loss")
        if supplied_raw_loss is None and supplied_log_loss is None:
            raise KeyError(
                f"train row {index}.target must contain mean_loss or log_mean_loss"
            )
        if supplied_raw_loss is not None:
            raw_loss = _finite_float(
                supplied_raw_loss,
                name=f"train row {index}.target.mean_loss",
            )
            if raw_loss <= 0.0:
                raise ValueError(f"train row {index}.target.mean_loss must be positive")
        else:
            raw_loss = math.exp(
                _finite_float(
                    supplied_log_loss,
                    name=f"train row {index}.target.log_mean_loss",
                )
            )
        log_loss = (
            math.log(raw_loss)
            if supplied_log_loss is None
            else _finite_float(
                supplied_log_loss,
                name=f"train row {index}.target.log_mean_loss",
            )
        )
        if not math.isclose(log_loss, math.log(raw_loss), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(
                f"train row {index} has inconsistent mean_loss and log_mean_loss"
            )

        optimizer = setting.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise TypeError(f"train row {index}.setting.optimizer must be a mapping")
        optimizer_type = optimizer.get("type")
        if not isinstance(optimizer_type, str) or not optimizer_type:
            raise ValueError(
                f"train row {index}.setting.optimizer.type must be a non-empty string"
            )
        learning_rate = _finite_float(
            optimizer.get("lr"),
            name=f"train row {index}.setting.optimizer.lr",
        )
        if learning_rate <= 0.0:
            raise ValueError(
                f"train row {index}.setting.optimizer.lr must be positive"
            )

        log_params.append(log_param)
        total_params.append(total)
        raw_losses.append(raw_loss)
        log_losses.append(log_loss)
        optimizers.append(optimizer_type)
        learning_rates.append(learning_rate)

    return (
        np.asarray(log_params, dtype=float),
        np.asarray(total_params, dtype=float),
        np.asarray(raw_losses, dtype=float),
        optimizers,
        np.asarray(learning_rates, dtype=float),
        np.asarray(log_losses, dtype=float),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    left_ranks = np.asarray(rankdata(left, method="average"), dtype=float)
    right_ranks = np.asarray(rankdata(right, method="average"), dtype=float)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return None
    value = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    return value if math.isfinite(value) else None


def _optimizer_lr_summary(
    optimizers: list[str],
    learning_rates: np.ndarray,
    raw_losses: np.ndarray,
    log_losses: np.ndarray,
) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, (optimizer, learning_rate) in enumerate(
        zip(optimizers, learning_rates)
    ):
        groups[(optimizer, float(learning_rate))].append(index)

    cells = []
    for (optimizer, learning_rate), indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=int)
        mean_log_loss = float(np.mean(log_losses[indices]))
        cells.append(
            {
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "log10_learning_rate": math.log10(learning_rate),
                "count": int(indices.size),
                "mean_log_loss": mean_log_loss,
                "mean_raw_loss": float(np.mean(raw_losses[indices])),
                "geometric_mean_raw_loss": math.exp(mean_log_loss),
            }
        )

    best_lr = []
    for optimizer in sorted(set(optimizers)):
        optimizer_cells = [cell for cell in cells if cell["optimizer"] == optimizer]
        winner = min(
            optimizer_cells,
            key=lambda cell: (
                cell["mean_log_loss"],
                cell["mean_raw_loss"],
                cell["learning_rate"],
            ),
        )
        best_lr.append(dict(winner))
    overall_best = min(
        cells,
        key=lambda cell: (
            cell["mean_log_loss"],
            cell["mean_raw_loss"],
            cell["optimizer"],
            cell["learning_rate"],
        ),
    )
    return {
        "selection_rule": "minimum training mean_log_loss",
        "cells": cells,
        "best_lr_by_optimizer": best_lr,
        "overall_best_cell": dict(overall_best),
    }


def _parameter_correlations(
    log_params: np.ndarray,
    raw_losses: np.ndarray,
    log_losses: np.ndarray,
    optimizers: list[str],
) -> dict[str, Any]:
    def one(indices: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(indices.size),
            "spearman_vs_log_loss": _spearman(
                log_params[indices], log_losses[indices]
            ),
            "spearman_vs_raw_loss": _spearman(
                log_params[indices], raw_losses[indices]
            ),
        }

    all_indices = np.arange(log_params.size, dtype=int)
    by_optimizer = []
    optimizer_array = np.asarray(optimizers, dtype=object)
    for optimizer in sorted(set(optimizers)):
        indices = np.flatnonzero(optimizer_array == optimizer)
        by_optimizer.append({"optimizer": optimizer, **one(indices)})
    return {
        "x": "derived.log_total_params",
        "overall": one(all_indices),
        "by_optimizer": by_optimizer,
    }


def _parameter_quantile_summary(
    log_params: np.ndarray,
    total_params: np.ndarray,
    raw_losses: np.ndarray,
    log_losses: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, Any]:
    if n_bins < 1:
        raise ValueError(f"parameter_bins must be positive, got {n_bins}")
    quantiles = np.linspace(0.0, 1.0, min(n_bins, log_params.size) + 1)
    edges = np.unique(np.quantile(log_params, quantiles))
    if edges.size == 1:
        assignments = np.zeros(log_params.size, dtype=int)
    else:
        assignments = np.searchsorted(edges[1:-1], log_params, side="right")

    bins = []
    for bin_index in range(max(1, edges.size - 1)):
        indices = np.flatnonzero(assignments == bin_index)
        if indices.size == 0:
            continue
        bins.append(
            {
                "bin_index": int(bin_index),
                "count": int(indices.size),
                "min_log_total_params": float(np.min(log_params[indices])),
                "max_log_total_params": float(np.max(log_params[indices])),
                "mean_log_total_params": float(np.mean(log_params[indices])),
                "min_total_params": int(round(float(np.min(total_params[indices])))),
                "max_total_params": int(round(float(np.max(total_params[indices])))),
                "mean_total_params": float(np.mean(total_params[indices])),
                "mean_log_loss": float(np.mean(log_losses[indices])),
                "mean_raw_loss": float(np.mean(raw_losses[indices])),
                "geometric_mean_raw_loss": math.exp(
                    float(np.mean(log_losses[indices]))
                ),
            }
        )
    return {
        "requested_bins": int(n_bins),
        "observed_bins": len(bins),
        "method": "empirical quantiles of derived.log_total_params; duplicate edges dropped",
        "bins": bins,
    }


def summarize_simple_rules(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    parameter_bins: int = 5,
) -> dict[str, Any]:
    """Summarize simple setting/loss relationships using training labels only."""

    rows = list(train_rows)
    if not rows:
        raise ValueError("summarize_simple_rules requires at least one training row")
    (
        log_params,
        total_params,
        raw_losses,
        optimizers,
        learning_rates,
        log_losses,
    ) = _row_values(rows)
    return {
        "optimizer_lr": _optimizer_lr_summary(
            optimizers,
            learning_rates,
            raw_losses,
            log_losses,
        ),
        "log_params_loss_spearman": _parameter_correlations(
            log_params,
            raw_losses,
            log_losses,
            optimizers,
        ),
        "parameter_quantile_bins": _parameter_quantile_summary(
            log_params,
            total_params,
            raw_losses,
            log_losses,
            n_bins=parameter_bins,
        ),
    }


def _narrative_findings(simple_rules: Mapping[str, Any]) -> list[str]:
    """Render a few auditable one-line rules from the train-only summaries."""

    optimizer_lr = simple_rules["optimizer_lr"]
    overall_best = optimizer_lr["overall_best_cell"]
    best_cells = optimizer_lr["best_lr_by_optimizer"]
    learning_rates = ", ".join(
        f"{cell['optimizer']}={float(cell['learning_rate']):g}"
        for cell in best_cells
    )
    correlation = simple_rules["log_params_loss_spearman"]["overall"][
        "spearman_vs_log_loss"
    ]
    if correlation is None:
        parameter_text = "Parameter-count rank association is undefined."
    elif abs(float(correlation)) < 0.1:
        parameter_text = (
            f"Parameter count is almost rank-neutral overall (Spearman "
            f"rho={float(correlation):.3f})."
        )
    elif correlation < 0:
        parameter_text = (
            f"Larger parameter count tends to reduce loss overall (Spearman "
            f"rho={float(correlation):.3f}; negative is better)."
        )
    else:
        parameter_text = (
            f"Larger parameter count tends to increase loss overall (Spearman "
            f"rho={float(correlation):.3f})."
        )
    return [
        (
            "Optimizer and learning rate should be treated jointly; the best "
            "training cell is "
            f"{overall_best['optimizer']} at lr={float(overall_best['learning_rate']):g}."
        ),
        f"Best train-only learning rate within each optimizer: {learning_rates}.",
        parameter_text,
    ]


def _pipeline_feature_names(pipeline: Pipeline, n_features: int) -> list[str]:
    try:
        names = pipeline[:-1].get_feature_names_out()
    except (AttributeError, TypeError, ValueError):
        return [f"feature_{index}" for index in range(n_features)]
    result = [str(name) for name in np.asarray(names, dtype=object).tolist()]
    if len(result) != n_features:
        return [f"feature_{index}" for index in range(n_features)]
    return result


def _ranked_values(
    names: list[str],
    values: np.ndarray,
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    pairs = zip(names, np.asarray(values, dtype=float).reshape(-1))
    ranked = sorted(pairs, key=lambda pair: (-abs(float(pair[1])), pair[0]))
    return [
        {
            "feature": name,
            value_key: float(value),
            f"absolute_{value_key}": abs(float(value)),
        }
        for name, value in ranked
    ]


def _lookup_table(estimator: OptimizerLrLookupRegressor) -> dict[str, Any]:
    if not hasattr(estimator, "lookup_") or not hasattr(estimator, "counts_"):
        return {"fitted": False}
    cells = []
    for key, value in sorted(estimator.lookup_.items()):
        optimizer, learning_rate = key
        mean_log_loss = _finite_float(value, name="lookup value")
        cells.append(
            {
                "optimizer": str(optimizer),
                "learning_rate": float(learning_rate),
                "count": int(estimator.counts_[key]),
                "predicted_log_loss": mean_log_loss,
                "predicted_raw_loss": math.exp(mean_log_loss),
            }
        )
    global_mean = _finite_float(estimator.global_mean_, name="lookup global mean")
    return {
        "fitted": True,
        "shrinkage": float(estimator.shrinkage),
        "global_mean_log_loss": global_mean,
        "global_mean_raw_loss": math.exp(global_mean),
        "cells": cells,
    }


def extract_model_interpretation(model: InterpretationModel) -> dict[str, Any]:
    """Extract JSON-friendly internals from one already-fitted study model."""

    if isinstance(model, EnsembleModel):
        return {
            "kind": "ensemble",
            "name": model.name,
            "intercept": float(model.intercept),
            "members": [
                {"name": member.name, "weight": float(weight)}
                for member, weight in zip(model.models, model.weights)
            ],
        }
    if not isinstance(model, FittedModel):
        raise TypeError(
            "model must be a FittedModel or EnsembleModel, "
            f"got {type(model).__name__}"
        )

    estimator = model.estimator
    final_estimator = (
        estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    )
    result: dict[str, Any] = {
        "kind": "fitted_model",
        "name": model.name,
        "estimator_type": type(final_estimator).__name__,
        "feature_set": model.feature_set,
    }

    if isinstance(final_estimator, OptimizerLrLookupRegressor):
        result["optimizer_lr_lookup"] = _lookup_table(final_estimator)

    coefficient_values = getattr(final_estimator, "coef_", None)
    if (
        isinstance(final_estimator, (Ridge, ElasticNet))
        and coefficient_values is not None
    ):
        coefficients = np.asarray(coefficient_values, dtype=float).reshape(-1)
        feature_names = (
            _pipeline_feature_names(estimator, coefficients.size)
            if isinstance(estimator, Pipeline)
            else [f"feature_{index}" for index in range(coefficients.size)]
        )
        intercept = np.asarray(final_estimator.intercept_, dtype=float).reshape(-1)
        result["linear_model"] = {
            "n_features": int(coefficients.size),
            "intercept": (
                float(intercept[0])
                if intercept.size == 1
                else [float(value) for value in intercept]
            ),
            "coefficients": _ranked_values(
                feature_names,
                coefficients,
                value_key="coefficient",
            ),
        }

    importance_values = getattr(final_estimator, "feature_importances_", None)
    if importance_values is not None:
        importances = np.asarray(importance_values, dtype=float).reshape(-1)
        feature_names = (
            _pipeline_feature_names(estimator, importances.size)
            if isinstance(estimator, Pipeline)
            else [f"feature_{index}" for index in range(importances.size)]
        )
        result["feature_importances"] = {
            "n_features": int(importances.size),
            "values": _ranked_values(
                feature_names,
                importances,
                value_key="importance",
            ),
        }
        if isinstance(final_estimator, DecisionTreeRegressor):
            result["shallow_tree_text"] = export_text(
                final_estimator,
                feature_names=feature_names,
                decimals=6,
            )
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Interpretation value is not JSON-safe: {type(value).__name__}")


def build_interpretation(
    train_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, InterpretationModel],
    *,
    parameter_bins: int = 5,
) -> dict[str, Any]:
    """Build the complete train-only interpretation artifact.

    The mapping key is retained as the stable study identifier even when it
    differs from the fitted object's display name.
    """

    rows = list(train_rows)
    simple_rules = summarize_simple_rules(
        rows,
        parameter_bins=parameter_bins,
    )
    first_row = rows[0] if rows else {}
    result = {
        "schema_version": 1,
        "label_source": "training rows only",
        "n_train_rows": len(rows),
        "experiment_id": first_row.get("experiment_id"),
        "family": first_row.get("family"),
        "simple_rules": simple_rules,
        "findings": _narrative_findings(simple_rules),
        "models": {
            str(name): extract_model_interpretation(model)
            for name, model in sorted(models.items())
        },
    }
    safe = _json_safe(result)
    # This assertion catches accidental NaN/Infinity or unsupported objects at
    # the module boundary instead of producing a subtly non-portable artifact.
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


__all__ = [
    "build_interpretation",
    "extract_model_interpretation",
    "summarize_simple_rules",
]
