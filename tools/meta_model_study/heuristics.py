"""Human-readable heuristic formulas with small linear calibrations.

This module implements a deliberately narrow meta-model baseline.  An LLM
supplies a fixed, low-dimensional formula whose terms have an explicit prior
direction (capacity, optimizer/LR mismatch, and a few architecture-shape
terms).  Labels may then be used in only three simple ways:

``positive_affine``
    Fit an intercept and a non-negative scale for the complete fixed score.
    This calibrates units without changing a single ranking.
``component_nnls``
    Fit an intercept and non-negative weights for the ten formula terms.
    Every learned term therefore retains its prior direction.
``component_ridge``
    Fit all ten term weights with Ridge.  Its alpha is selected using only
    five-fold CV inside the 900-row training split.

The locked 100-row validation split is never used for fitting, method
selection, or alpha selection.  The external-prediction command reads the
already prepared, target-free inputs and has no answer-key argument.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from tools.meta_model_study.external import sha256_file, write_unscored_predictions
from tools.meta_model_study.metrics import (
    evaluate_predictions,
    ranking_metrics,
    regression_metrics,
)
from tools.meta_model_study.run import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    discover_experiments,
    load_experiment_rows,
    make_stratified_folds,
)


SCHEMA_VERSION = "meta_model_heuristic_formula_v2"
METHODS = (
    "fixed_zero_shot",
    "positive_affine",
    "component_nnls",
    "component_ridge",
)
COMPONENT_NAMES = (
    "capacity",
    "optimizer_prior",
    "optimizer_lr_quadratic",
    "weight_decay",
    "depth",
    "width",
    "residual",
    "layer_norm",
    "activation",
    "architecture_shape",
)
DEFAULT_ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_PREPARED_INPUTS = DEFAULT_OUTPUT_ROOT / "external/prepared_inputs.json"
DEFAULT_ARTIFACT_DIR = DEFAULT_OUTPUT_ROOT / "heuristics/heuristic_formula_v2"
DEFAULT_EXTERNAL_OUTPUT = (
    DEFAULT_OUTPUT_ROOT / "external/unscored/heuristic_formula_v2.json"
)


@dataclass(frozen=True)
class FormulaPrior:
    """Frozen coefficients chosen for interpretability, not fitted labels."""

    family: str
    model_type: str
    intercept_log_loss: float
    reference_log_params: float
    capacity_slope: float
    optimizer_penalty: Mapping[str, float]
    optimizer_log10_lr_optimum: Mapping[str, float]
    lr_curvature: float
    reference_log10_weight_decay: float
    weight_decay_curvature: float
    target_depth: float
    depth_curvature: float
    target_log2_width: float
    width_curvature: float
    residual_effect: float
    layer_norm_effect: float
    activation_penalty: Mapping[str, float]
    shape_curvature: float


_OPTIMIZER_LR_BIGRAM = {
    "Adagrad": -2.0,
    "Adam": -3.0,
    "AdamW": -3.0,
    "RMSprop": -3.3,
    "SGD": -2.0,
}
_OPTIMIZER_LR_MULTIVARIATE = {
    "Adagrad": -2.0,
    "Adam": -2.5,
    "AdamW": -2.5,
    "RMSprop": -3.3,
    "SGD": -2.0,
}
_OPTIMIZER_LR_UNIVARIATE = {
    "Adagrad": -2.0,
    "Adam": -2.0,
    "AdamW": -3.0,
    "RMSprop": -3.3,
    "SGD": -2.0,
}

# These are intentionally few, round-number priors.  They encode the familiar
# expectations that more capacity usually helps; LR has an optimizer-specific
# log-scale sweet spot; tiny-budget MLPs prefer moderate depth/width; and
# normalization/residual structure can matter.  They are constants in source
# and ``formula_components`` never accepts a target.
FORMULA_PRIORS: dict[str, FormulaPrior] = {
    "bigram_lm": FormulaPrior(
        family="bigram_lm",
        model_type="transformer_lm",
        intercept_log_loss=1.20,
        reference_log_params=math.log(50_000),
        capacity_slope=0.008,
        optimizer_penalty={
            "Adagrad": 0.02,
            "Adam": 0.0,
            "AdamW": 0.005,
            "RMSprop": 0.01,
            "SGD": 0.08,
        },
        optimizer_log10_lr_optimum=_OPTIMIZER_LR_BIGRAM,
        lr_curvature=0.025,
        reference_log10_weight_decay=-4.0,
        weight_decay_curvature=0.001,
        target_depth=2.0,
        depth_curvature=0.002,
        target_log2_width=6.0,
        width_curvature=0.002,
        residual_effect=0.0,
        layer_norm_effect=0.0,
        activation_penalty={},
        shape_curvature=0.002,
    ),
    "multivariate_regression": FormulaPrior(
        family="multivariate_regression",
        model_type="mlp",
        intercept_log_loss=-0.50,
        reference_log_params=math.log(50_000),
        capacity_slope=0.10,
        optimizer_penalty={
            "Adagrad": 0.15,
            "Adam": 0.0,
            "AdamW": 0.0,
            "RMSprop": 0.02,
            "SGD": 0.12,
        },
        optimizer_log10_lr_optimum=_OPTIMIZER_LR_MULTIVARIATE,
        lr_curvature=0.22,
        reference_log10_weight_decay=-4.0,
        weight_decay_curvature=0.005,
        target_depth=5.0,
        depth_curvature=0.025,
        target_log2_width=7.0,
        width_curvature=0.025,
        residual_effect=-0.05,
        layer_norm_effect=-0.06,
        activation_penalty={
            "gelu": 0.0,
            "silu": 0.0,
            "relu": 0.015,
            "leaky_relu": 0.02,
            "tanh": 0.04,
            "sigmoid": 0.08,
        },
        shape_curvature=0.0,
    ),
    "univariate_regression": FormulaPrior(
        family="univariate_regression",
        model_type="mlp",
        intercept_log_loss=-2.00,
        reference_log_params=math.log(50_000),
        capacity_slope=0.13,
        optimizer_penalty={
            "Adagrad": 0.25,
            "Adam": 0.0,
            "AdamW": 0.05,
            "RMSprop": 0.15,
            "SGD": 0.35,
        },
        optimizer_log10_lr_optimum=_OPTIMIZER_LR_UNIVARIATE,
        lr_curvature=0.30,
        reference_log10_weight_decay=-4.0,
        weight_decay_curvature=0.005,
        target_depth=5.0,
        depth_curvature=0.04,
        target_log2_width=7.0,
        width_curvature=0.04,
        residual_effect=0.18,
        layer_norm_effect=-0.08,
        activation_penalty={
            "gelu": 0.0,
            "silu": 0.0,
            "relu": 0.02,
            "leaky_relu": 0.025,
            "tanh": 0.05,
            "sigmoid": 0.10,
        },
        shape_curvature=0.0,
    ),
}


@dataclass(frozen=True)
class CalibratedFormula:
    """A fixed formula or a train-only linear calibration of its components."""

    method: str
    family: str
    intercept: float
    coefficients: tuple[float, ...]
    alpha: float | None = None
    affine_slope: float | None = None

    def predict_components(self, components: np.ndarray) -> np.ndarray:
        matrix = np.asarray(components, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(COMPONENT_NAMES):
            raise ValueError(
                f"components must have shape (n, {len(COMPONENT_NAMES)}), "
                f"got {matrix.shape}"
            )
        result = self.intercept + matrix @ np.asarray(self.coefficients)
        if not np.all(np.isfinite(result)):
            raise ValueError("Formula produced non-finite predictions")
        return result

    def predict(self, examples: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = component_matrix(examples, self.family)
        return self.predict_components(matrix)

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "family": self.family,
            "intercept": self.intercept,
            "coefficients": dict(zip(COMPONENT_NAMES, self.coefficients)),
            "alpha": self.alpha,
            "affine_slope": self.affine_slope,
            "equation": equation_text(self),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> CalibratedFormula:
        coefficients = value.get("coefficients")
        if not isinstance(coefficients, Mapping):
            raise TypeError("formula coefficients must be an object")
        return cls(
            method=str(value["method"]),
            family=str(value["family"]),
            intercept=float(value["intercept"]),
            coefficients=tuple(float(coefficients[name]) for name in COMPONENT_NAMES),
            alpha=(float(value["alpha"]) if value.get("alpha") is not None else None),
            affine_slope=(
                float(value["affine_slope"])
                if value.get("affine_slope") is not None
                else None
            ),
        )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _example_parts(example: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    # Prepared external choices add exactly one target-free ``example`` layer.
    if "setting" not in example and isinstance(example.get("example"), Mapping):
        example = example["example"]
    setting = _mapping(example.get("setting"), name="example.setting")
    derived = _mapping(example.get("derived"), name="example.derived")
    return setting, derived


def _log_params(derived: Mapping[str, Any]) -> float:
    if "log_total_params" in derived:
        result = _finite(derived["log_total_params"], name="log_total_params")
    else:
        total = _finite(derived.get("total_params"), name="total_params")
        if total <= 0.0:
            raise ValueError("total_params must be positive")
        result = math.log(total)
    return result


def _log10_positive(value: Any, *, name: str) -> float:
    number = _finite(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return math.log10(number)


def _activation_component(model: Mapping[str, Any], prior: FormulaPrior) -> float:
    raw = model.get("activations", [])
    if not isinstance(raw, (list, tuple)):
        raise TypeError("model.activations must be a list")
    if not raw:
        return 0.0
    penalties = []
    for activation in raw:
        if not isinstance(activation, str):
            raise TypeError("model activation names must be strings")
        penalties.append(float(prior.activation_penalty.get(activation, 0.04)))
    return float(np.mean(penalties))


def formula_components(
    example: Mapping[str, Any], family: str
) -> dict[str, float]:
    """Return the ten target-free fixed-formula contributions for one setting."""

    try:
        prior = FORMULA_PRIORS[family]
    except KeyError as exc:
        raise ValueError(f"No heuristic prior for family {family!r}") from exc
    setting, derived = _example_parts(example)
    model = _mapping(setting.get("model"), name="setting.model")
    optimizer = _mapping(setting.get("optimizer"), name="setting.optimizer")
    if model.get("type") != prior.model_type:
        raise ValueError(
            f"Family {family!r} expects model.type={prior.model_type!r}, "
            f"got {model.get('type')!r}"
        )

    optimizer_type = str(optimizer.get("type"))
    if optimizer_type not in prior.optimizer_log10_lr_optimum:
        raise ValueError(f"Unsupported optimizer for formula: {optimizer_type!r}")
    log_lr = _log10_positive(optimizer.get("lr"), name="optimizer.lr")
    lr_delta = log_lr - prior.optimizer_log10_lr_optimum[optimizer_type]

    weight_decay = _finite(
        optimizer.get("weight_decay", 0.0), name="optimizer.weight_decay"
    )
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    # Treat exact zero as the common 1e-6 floor; this keeps the term finite and
    # expresses a mild preference, not a hard requirement for weight decay.
    log_weight_decay = math.log10(max(weight_decay, 1e-6))

    components = {name: 0.0 for name in COMPONENT_NAMES}
    components["capacity"] = -prior.capacity_slope * (
        _log_params(derived) - prior.reference_log_params
    )
    components["optimizer_prior"] = float(
        prior.optimizer_penalty.get(optimizer_type, 0.20)
    )
    components["optimizer_lr_quadratic"] = prior.lr_curvature * lr_delta**2
    components["weight_decay"] = prior.weight_decay_curvature * (
        log_weight_decay - prior.reference_log10_weight_decay
    ) ** 2

    if prior.model_type == "mlp":
        depth = _finite(model.get("depth"), name="model.depth")
        width = _finite(model.get("width"), name="model.width")
        if width <= 0.0:
            raise ValueError("model.width must be positive")
        components["depth"] = prior.depth_curvature * (depth - prior.target_depth) ** 2
        components["width"] = prior.width_curvature * (
            math.log2(width) - prior.target_log2_width
        ) ** 2
        components["residual"] = prior.residual_effect * float(
            bool(model.get("residual", False))
        )
        layer_norm = model.get("layer_norm", [])
        if not isinstance(layer_norm, (list, tuple)):
            raise TypeError("model.layer_norm must be a list")
        layer_norm_fraction = (
            sum(bool(value) for value in layer_norm) / len(layer_norm)
            if layer_norm
            else 0.0
        )
        components["layer_norm"] = prior.layer_norm_effect * layer_norm_fraction
        components["activation"] = _activation_component(model, prior)
    else:
        depth = _finite(model.get("num_layers"), name="model.num_layers")
        width = _finite(model.get("d_model"), name="model.d_model")
        d_ff = _finite(model.get("d_ff"), name="model.d_ff")
        heads = _finite(model.get("num_heads"), name="model.num_heads")
        if min(width, d_ff, heads) <= 0.0:
            raise ValueError("Transformer widths and head count must be positive")
        components["depth"] = prior.depth_curvature * (depth - prior.target_depth) ** 2
        components["width"] = prior.width_curvature * (
            math.log2(width) - prior.target_log2_width
        ) ** 2
        ff_ratio_delta = math.log2(d_ff / width) - 2.0
        head_dim_delta = math.log2(width / heads) - 5.0
        components["architecture_shape"] = prior.shape_curvature * (
            ff_ratio_delta**2 + head_dim_delta**2
        )

    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("Heuristic components must all be finite")
    return components


def component_matrix(
    examples: Sequence[Mapping[str, Any]], family: str
) -> np.ndarray:
    """Return row-aligned fixed contributions in ``COMPONENT_NAMES`` order."""

    if not examples:
        return np.empty((0, len(COMPONENT_NAMES)), dtype=np.float64)
    rows = []
    for example in examples:
        components = formula_components(example, family)
        rows.append([components[name] for name in COMPONENT_NAMES])
    return np.asarray(rows, dtype=np.float64)


def fixed_formula(family: str) -> CalibratedFormula:
    prior = FORMULA_PRIORS[family]
    return CalibratedFormula(
        method="fixed_zero_shot",
        family=family,
        intercept=prior.intercept_log_loss,
        coefficients=(1.0,) * len(COMPONENT_NAMES),
    )


def _positive_affine(
    matrix: np.ndarray, y: np.ndarray, family: str
) -> CalibratedFormula:
    base_model = fixed_formula(family)
    score = base_model.predict_components(matrix)
    centered_score = score - np.mean(score)
    denominator = float(np.dot(centered_score, centered_score))
    slope = (
        max(0.0, float(np.dot(centered_score, y - np.mean(y)) / denominator))
        if denominator > 0.0
        else 0.0
    )
    intercept = float(np.mean(y) - slope * np.mean(score))
    # Fold the prior intercept into an equivalent component-linear equation.
    return CalibratedFormula(
        method="positive_affine",
        family=family,
        intercept=intercept + slope * base_model.intercept,
        coefficients=(slope,) * len(COMPONENT_NAMES),
        affine_slope=slope,
    )


def _component_nnls(
    matrix: np.ndarray, y: np.ndarray, family: str
) -> CalibratedFormula:
    estimator = LinearRegression(positive=True).fit(matrix, y)
    return CalibratedFormula(
        method="component_nnls",
        family=family,
        intercept=float(estimator.intercept_),
        coefficients=tuple(float(value) for value in estimator.coef_),
    )


def _component_ridge(
    matrix: np.ndarray, y: np.ndarray, family: str, alpha: float
) -> CalibratedFormula:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Ridge alpha must be finite and positive")
    scale = np.std(matrix, axis=0)
    scale[scale == 0.0] = 1.0
    mean = np.mean(matrix, axis=0)
    normalized = (matrix - mean) / scale
    estimator = Ridge(alpha=alpha).fit(normalized, y)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64) / scale
    intercept = float(estimator.intercept_ - np.dot(coefficients, mean))
    return CalibratedFormula(
        method="component_ridge",
        family=family,
        intercept=intercept,
        coefficients=tuple(float(value) for value in coefficients),
        alpha=float(alpha),
    )


def fit_method(
    method: str,
    matrix: np.ndarray,
    y: np.ndarray,
    family: str,
    *,
    alpha: float | None = None,
) -> CalibratedFormula:
    """Fit one calibration; ``fixed_zero_shot`` ignores labels by design."""

    matrix = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (y.size, len(COMPONENT_NAMES)):
        raise ValueError("matrix and target have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(y)):
        raise ValueError("matrix and target must be finite")
    if method == "fixed_zero_shot":
        return fixed_formula(family)
    if method == "positive_affine":
        return _positive_affine(matrix, y, family)
    if method == "component_nnls":
        return _component_nnls(matrix, y, family)
    if method == "component_ridge":
        if alpha is None:
            raise ValueError("component_ridge requires alpha")
        return _component_ridge(matrix, y, family, alpha)
    raise ValueError(f"Unknown heuristic method: {method!r}")


def _cv_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "log_regression": regression_metrics(y, predictions),
        "ranking": ranking_metrics(y, predictions),
    }


def cross_validate_formulas(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    *,
    n_splits: int = 5,
    seed: int = 20260713,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> tuple[dict[str, CalibratedFormula], dict[str, dict[str, Any]]]:
    """Tune and compare formulas using only the supplied training rows."""

    matrix = component_matrix(rows, family)
    y = np.asarray([row["target"]["log_mean_loss"] for row in rows], dtype=float)
    folds = make_stratified_folds(rows, n_splits=n_splits, seed=seed)
    oof_by_method: dict[str, np.ndarray] = {}

    # The fixed formula has no fitted state, so its OOF vector is simply its
    # target-free prediction.  The other two non-Ridge calibrations are fit
    # independently inside every fold.
    oof_by_method["fixed_zero_shot"] = fixed_formula(family).predict_components(
        matrix
    )
    for method in ("positive_affine", "component_nnls"):
        oof = np.empty(y.size, dtype=float)
        for train_indices, test_indices in folds:
            fitted = fit_method(
                method,
                matrix[train_indices],
                y[train_indices],
                family,
            )
            oof[test_indices] = fitted.predict_components(matrix[test_indices])
        oof_by_method[method] = oof

    alpha_rows: list[dict[str, float]] = []
    ridge_oof: dict[float, np.ndarray] = {}
    for raw_alpha in alpha_grid:
        alpha = float(raw_alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("Every alpha must be finite and positive")
        oof = np.empty(y.size, dtype=float)
        for train_indices, test_indices in folds:
            fitted = fit_method(
                "component_ridge",
                matrix[train_indices],
                y[train_indices],
                family,
                alpha=alpha,
            )
            oof[test_indices] = fitted.predict_components(matrix[test_indices])
        ridge_oof[alpha] = oof
        metrics = regression_metrics(y, oof)
        alpha_rows.append(
            {
                "alpha": alpha,
                "rmse_log": metrics["rmse"],
                "mae_log": metrics["mae"],
            }
        )
    best_alpha = min(alpha_rows, key=lambda row: (row["rmse_log"], row["alpha"]))[
        "alpha"
    ]
    oof_by_method["component_ridge"] = ridge_oof[best_alpha]

    fitted_models = {
        method: fit_method(
            method,
            matrix,
            y,
            family,
            alpha=best_alpha if method == "component_ridge" else None,
        )
        for method in METHODS
    }
    results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        result = _cv_metrics(y, oof_by_method[method])
        result["oof_predictions"] = oof_by_method[method].tolist()
        if method == "component_ridge":
            result["alpha_search"] = alpha_rows
            result["selected_alpha"] = best_alpha
        results[method] = result
    return fitted_models, results


def equation_text(model: CalibratedFormula) -> str:
    terms = [f"{model.intercept:.8g}"]
    for name, coefficient in zip(COMPONENT_NAMES, model.coefficients):
        terms.append(f"({coefficient:.8g} * {name})")
    return "predicted_log_loss = " + " + ".join(terms)


def formula_definition(family: str) -> dict[str, Any]:
    prior = FORMULA_PRIORS[family]
    return {
        "prior": asdict(prior),
        "component_order": list(COMPONENT_NAMES),
        "components": {
            "capacity": "-capacity_slope * (ln(total_params) - ln(reference_params))",
            "optimizer_prior": "fixed family-specific optimizer penalty",
            "optimizer_lr_quadratic": (
                "lr_curvature * (log10(lr) - optimizer_log10_lr_optimum)^2"
            ),
            "weight_decay": (
                "weight_decay_curvature * "
                "(log10(max(weight_decay, 1e-6)) - reference)^2"
            ),
            "depth": "depth_curvature * (depth - target_depth)^2",
            "width": "width_curvature * (log2(width) - target_log2_width)^2",
            "residual": "residual_effect * I[residual] (MLP only)",
            "layer_norm": "layer_norm_effect * normalized-layer fraction (MLP only)",
            "activation": "mean fixed activation penalty (MLP only)",
            "architecture_shape": (
                "shape_curvature * ((log2(d_ff/d_model)-2)^2 + "
                "(log2(head_dim)-5)^2) (Transformer only)"
            ),
        },
        "fixed_equation": equation_text(fixed_formula(family)),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
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
    ).encode()
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


def fit_study(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    n_splits: int = 5,
    seed: int = 20260713,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> dict[str, Any]:
    """Fit all family formulas, then evaluate each locked holdout exactly once."""

    experiments: list[dict[str, Any]] = []
    for experiment_dir in discover_experiments(dataset_root):
        train_rows, validation_rows, _manifest = load_experiment_rows(experiment_dir)
        families = {str(row["family"]) for row in train_rows + validation_rows}
        if len(families) != 1:
            raise ValueError(f"{experiment_dir.name} mixes families: {families}")
        family = families.pop()
        models, cv_results = cross_validate_formulas(
            train_rows,
            family,
            n_splits=n_splits,
            seed=seed,
            alpha_grid=alpha_grid,
        )
        # Selection is frozen from training-only CV before any holdout metric is
        # computed.  RMSE is appropriate because the declared target is log loss.
        selected_method = min(
            METHODS,
            key=lambda method: (
                cv_results[method]["log_regression"]["rmse"],
                METHODS.index(method),
            ),
        )
        validation_results = {}
        validation_predictions = {}
        for method in METHODS:
            predictions = models[method].predict(validation_rows)
            validation_predictions[method] = predictions.tolist()
            validation_results[method] = evaluate_predictions(
                validation_rows,
                predictions,
                prediction_space="log",
            )
        experiment_result = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_dir.name,
            "family": family,
            "num_train": len(train_rows),
            "num_validation": len(validation_rows),
            "selection_protocol": {
                "target": "log(mean_loss)",
                "method_selected_by": "minimum train-only CV RMSE in log-loss",
                "ridge_alpha_selected_by": "minimum train-only CV RMSE in log-loss",
                "folds": n_splits,
                "stratified_by": "row.stratum",
                "validation_used_for_selection": False,
            },
            "selected_method": selected_method,
            "formula_definition": formula_definition(family),
            "models": {method: models[method].to_json() for method in METHODS},
            "cv": cv_results,
            "validation": validation_results,
            "validation_predictions_log_loss": validation_predictions,
            "written_at": _utc_now(),
        }
        experiment_path = artifact_dir / "experiments" / f"{experiment_dir.name}.json"
        _atomic_write_json(experiment_path, experiment_result)
        experiments.append(
            {
                "experiment_id": experiment_dir.name,
                "family": family,
                "selected_method": selected_method,
                "artifact": str(experiment_path.resolve()),
                "artifact_sha256": sha256_file(experiment_path),
                "cv": {
                    method: cv_results[method]["log_regression"]
                    for method in METHODS
                },
                "validation": {
                    method: {
                        "log": validation_results[method]["all"]["log"],
                        "ranking": validation_results[method]["all"]["ranking"],
                        "three_choice": validation_results[method]["all"][
                            "three_choice"
                        ],
                    }
                    for method in METHODS
                },
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "num_experiments": len(experiments),
        "methods": list(METHODS),
        "experiments": experiments,
        "completed_at": _utc_now(),
    }
    _atomic_write_json(artifact_dir / "summary.json", summary)
    return summary


def _load_fitted_formulas(
    artifact_dir: Path,
) -> tuple[dict[str, CalibratedFormula], dict[str, str]]:
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models: dict[str, CalibratedFormula] = {}
    methods: dict[str, str] = {}
    for entry in summary["experiments"]:
        experiment_id = str(entry["experiment_id"])
        artifact_path = Path(str(entry["artifact"]))
        result = json.loads(artifact_path.read_text(encoding="utf-8"))
        method = str(result["selected_method"])
        models[experiment_id] = CalibratedFormula.from_json(result["models"][method])
        methods[experiment_id] = method
    return models, methods


def predict_external(
    prepared_inputs_path: Path = DEFAULT_PREPARED_INPUTS,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_EXTERNAL_OUTPUT,
) -> dict[str, Any]:
    """Write blind predictions from prepared inputs; no answer key is accepted."""

    prepared = json.loads(prepared_inputs_path.read_text(encoding="utf-8"))
    questions = prepared.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("prepared inputs must contain a non-empty questions list")
    models, method_by_experiment = _load_fitted_formulas(artifact_dir)
    predictions: list[dict[str, Any]] = []
    for question in questions:
        experiment_id = str(question["experiment_id"])
        try:
            model = models[experiment_id]
        except KeyError as exc:
            raise ValueError(f"No fitted formula for {experiment_id!r}") from exc
        choices = question["choices"]
        scores = model.predict([choice["example"] for choice in choices])
        selected = int(np.argmin(scores))
        predictions.append(
            {
                "question_id": str(question["question_id"]),
                "family": str(question["family"]),
                "experiment_id": experiment_id,
                "predicted_letter": str(choices[selected]["letter"]),
                "predicted_candidate_id": str(choices[selected]["candidate_id"]),
                "choice_predictions": [
                    {
                        "letter": str(choice["letter"]),
                        "candidate_id": str(choice["candidate_id"]),
                        "predicted_log_loss": float(score),
                    }
                    for choice, score in zip(choices, scores)
                ],
            }
        )
    metadata = {
        "method": "heuristic_formula_v2",
        "answer_key_opened": False,
        "method_by_experiment": method_by_experiment,
        "prepared_inputs_path": str(prepared_inputs_path.resolve()),
        "prepared_inputs_sha256": sha256_file(prepared_inputs_path),
        "formula_summary_path": str((artifact_dir / "summary.json").resolve()),
        "formula_summary_sha256": sha256_file(artifact_dir / "summary.json"),
        "written_at": _utc_now(),
    }
    digest = write_unscored_predictions(
        output_path,
        predictions,
        metadata=metadata,
    )
    return {
        "path": str(output_path.resolve()),
        "sha256": digest,
        "num_questions": len(predictions),
        "method_by_experiment": method_by_experiment,
        "answer_key_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="fit and evaluate formulas")
    fit_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    fit_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    fit_parser.add_argument("--folds", type=int, default=5)
    fit_parser.add_argument("--seed", type=int, default=20260713)

    predict_parser = subparsers.add_parser(
        "predict-external", help="write blind predictions from prepared inputs"
    )
    predict_parser.add_argument(
        "--prepared-inputs", type=Path, default=DEFAULT_PREPARED_INPUTS
    )
    predict_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_OUTPUT)

    all_parser = subparsers.add_parser("all", help="fit, evaluate, and predict blind")
    all_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    all_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    all_parser.add_argument("--folds", type=int, default=5)
    all_parser.add_argument("--seed", type=int, default=20260713)
    all_parser.add_argument(
        "--prepared-inputs", type=Path, default=DEFAULT_PREPARED_INPUTS
    )
    all_parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"fit", "all"}:
        fit_study(
            args.dataset_root.resolve(),
            args.artifact_dir.resolve(),
            n_splits=args.folds,
            seed=args.seed,
        )
    if args.command in {"predict-external", "all"}:
        result = predict_external(
            args.prepared_inputs.resolve(),
            args.artifact_dir.resolve(),
            args.output.resolve(),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
