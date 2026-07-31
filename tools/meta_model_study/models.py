"""Model search space and fitted-model containers for the meta-model study."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from tools.meta_model_study.features import FeatureEncoder


def _as_rows(values: Any) -> list[dict[str, Any]]:
    if isinstance(values, np.ndarray):
        return list(values.tolist())
    return list(values)


class OptimizerLrLookupRegressor(RegressorMixin, BaseEstimator):
    """Mean log-loss in each optimizer/lr grid cell, with safe fallbacks."""

    def __init__(self, shrinkage: float = 0.0) -> None:
        self.shrinkage = shrinkage

    @staticmethod
    def _key(row: dict[str, Any]) -> tuple[str, float]:
        optimizer = row["setting"]["optimizer"]
        return str(optimizer["type"]), float(optimizer["lr"])

    def fit(self, X: Any, y: np.ndarray) -> OptimizerLrLookupRegressor:
        rows = _as_rows(X)
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {targets.shape}")
        if len(rows) != targets.size:
            raise ValueError(
                "X and y must contain the same number of rows, got "
                f"{len(rows)} and {targets.size}"
            )
        if not rows:
            raise ValueError("OptimizerLrLookupRegressor.fit requires at least one row")
        if not np.all(np.isfinite(targets)):
            raise ValueError("y contains non-finite values")
        if not math.isfinite(float(self.shrinkage)) or self.shrinkage < 0.0:
            raise ValueError("shrinkage must be finite and non-negative")
        self.global_mean_ = float(np.mean(targets))
        grouped: dict[tuple[str, float], list[float]] = {}
        for row, target in zip(rows, targets):
            grouped.setdefault(self._key(row), []).append(float(target))
        self.lookup_ = {}
        self.counts_ = {}
        for key, values in grouped.items():
            count = len(values)
            mean = float(np.mean(values))
            weight = count / (count + float(self.shrinkage))
            self.lookup_[key] = weight * mean + (1.0 - weight) * self.global_mean_
            self.counts_[key] = count
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("global_mean_", "lookup_", "counts_"))
        return np.asarray(
            [self.lookup_.get(self._key(row), self.global_mean_) for row in _as_rows(X)],
            dtype=float,
        )


class MaxParamsHeuristic(RegressorMixin, BaseEstimator):
    """A no-label ranking rule: more parameters predicts lower loss."""

    def fit(self, X: Any, y: np.ndarray | None = None) -> MaxParamsHeuristic:
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(
            [-float(row["derived"]["log_total_params"]) for row in _as_rows(X)],
            dtype=float,
        )


@dataclass(frozen=True)
class SearchDefinition:
    name: str
    estimator: BaseEstimator
    params: dict[str, list[Any]]
    feature_set: str | None = None
    randomized_iterations: int | None = None
    interpretable: bool = False
    dataset_conditioning: str = "unaware"


@dataclass
class FittedModel:
    name: str
    estimator: BaseEstimator
    feature_set: str | None
    best_params: dict[str, Any]
    cv_rmse_log: float
    cv_mae_log: float
    cv_r2_log: float
    oof_predictions: np.ndarray
    search_rows: list[dict[str, Any]]
    interpretable: bool = False
    dataset_conditioning: str = "unaware"

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(np.asarray(rows, dtype=object)),
            dtype=float,
        )


@dataclass
class EnsembleModel:
    """Positive linear stack over base-model OOF prediction columns.

    ``cv_*`` and ``oof_predictions`` are apparent fit diagnostics from fitting
    and evaluating the combiner on the same OOF matrix.  They are not
    meta-level cross-validation estimates and must not be used for champion
    selection; locked holdout/external predictions provide ensemble evidence.
    """

    name: str
    models: list[FittedModel]
    weights: np.ndarray
    intercept: float
    cv_rmse_log: float
    cv_mae_log: float
    cv_r2_log: float
    oof_predictions: np.ndarray
    interpretable: bool = False

    @property
    def feature_set(self) -> str:
        return "ensemble"

    @property
    def best_params(self) -> dict[str, Any]:
        return {
            "members": [model.name for model in self.models],
            "weights": [float(value) for value in self.weights],
            "intercept": float(self.intercept),
        }

    @property
    def search_rows(self) -> list[dict[str, Any]]:
        return []

    def predict_log(self, rows: list[dict[str, Any]]) -> np.ndarray:
        matrix = np.column_stack([model.predict_log(rows) for model in self.models])
        return self.intercept + matrix @ self.weights


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


def search_definitions(
    seed: int,
    *,
    include_parameter_count: bool = True,
    dataset_conditioning: str = "unaware",
) -> list[SearchDefinition]:
    """Pre-registered model ladder, ordered from simplest to most flexible."""

    def feature_pipeline(feature_set: str, estimator: BaseEstimator) -> Pipeline:
        return _feature_pipeline(
            feature_set,
            estimator,
            include_parameter_count=include_parameter_count,
            dataset_conditioning=dataset_conditioning,
        )

    definitions = [
        SearchDefinition(
            name="constant_mean",
            estimator=DummyRegressor(strategy="mean"),
            params={},
            interpretable=True,
        ),
        SearchDefinition(
            name="max_params_heuristic",
            estimator=MaxParamsHeuristic(),
            params={},
            interpretable=True,
        ),
        SearchDefinition(
            name="params_ridge",
            estimator=feature_pipeline("params", Ridge()),
            params={"model__alpha": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]},
            feature_set="params",
            interpretable=True,
        ),
        SearchDefinition(
            name="params_polynomial_ridge",
            estimator=Pipeline(
                [
                    (
                        "features",
                        FeatureEncoder(
                            feature_set="params",
                            include_parameter_count=include_parameter_count,
                            dataset_conditioning=dataset_conditioning,
                        ),
                    ),
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ("model", Ridge()),
                ]
            ),
            params={
                "poly__degree": [2, 3, 4],
                "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            },
            feature_set="params",
            interpretable=True,
        ),
        SearchDefinition(
            name="optimizer_lr_lookup",
            estimator=OptimizerLrLookupRegressor(),
            params={"shrinkage": [0.0, 1.0, 3.0, 10.0, 30.0]},
            interpretable=True,
        ),
        SearchDefinition(
            name="optimizer_lr_ridge",
            estimator=feature_pipeline("optimizer_lr", Ridge()),
            params={"model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]},
            feature_set="optimizer_lr",
            interpretable=True,
        ),
        SearchDefinition(
            name="compact_ridge",
            estimator=feature_pipeline("compact", Ridge()),
            params={"model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]},
            feature_set="compact",
            interpretable=True,
        ),
        SearchDefinition(
            name="compact_polynomial_ridge",
            estimator=Pipeline(
                [
                    (
                        "features",
                        FeatureEncoder(
                            feature_set="compact",
                            include_parameter_count=include_parameter_count,
                            dataset_conditioning=dataset_conditioning,
                        ),
                    ),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("model", Ridge()),
                ]
            ),
            params={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            feature_set="compact",
            interpretable=True,
        ),
        SearchDefinition(
            name="full_ridge",
            estimator=feature_pipeline("full", Ridge()),
            params={"model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]},
            feature_set="full",
            interpretable=True,
        ),
        SearchDefinition(
            name="compact_elastic_net",
            estimator=feature_pipeline(
                "compact",
                ElasticNet(max_iter=20_000, selection="cyclic", random_state=seed),
            ),
            params={
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
                "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
            },
            feature_set="compact",
            randomized_iterations=16,
            interpretable=True,
        ),
        SearchDefinition(
            name="shallow_tree",
            estimator=feature_pipeline(
                "compact",
                DecisionTreeRegressor(random_state=seed),
            ),
            params={
                "model__max_depth": [1, 2, 3, 4, 5],
                "model__min_samples_leaf": [15, 30, 50],
                "model__ccp_alpha": [0.0, 1e-4, 1e-3],
            },
            feature_set="compact",
            randomized_iterations=30,
            interpretable=True,
        ),
        SearchDefinition(
            name="random_forest",
            estimator=feature_pipeline(
                "full",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            params={
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": [0.4, 0.7, 1.0],
            },
            feature_set="full",
            randomized_iterations=24,
        ),
        SearchDefinition(
            name="extra_trees",
            estimator=feature_pipeline(
                "full",
                ExtraTreesRegressor(
                    n_estimators=500,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            params={
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": [0.4, 0.7, 1.0],
            },
            feature_set="full",
            randomized_iterations=24,
        ),
        SearchDefinition(
            name="hist_gradient_boosting",
            estimator=feature_pipeline(
                "full",
                HistGradientBoostingRegressor(
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
            params={
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_iter": [200, 400, 800],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__min_samples_leaf": [10, 20, 40],
                "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
            },
            feature_set="full",
            randomized_iterations=32,
        ),
        SearchDefinition(
            name="gradient_boosting",
            estimator=feature_pipeline(
                "compact",
                GradientBoostingRegressor(random_state=seed),
            ),
            params={
                "model__n_estimators": [100, 300, 600],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [5, 15, 30],
                "model__loss": ["squared_error", "huber"],
            },
            feature_set="compact",
            randomized_iterations=28,
        ),
        SearchDefinition(
            name="rbf_svr",
            estimator=feature_pipeline("compact", SVR(kernel="rbf")),
            params={
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
                "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
                "model__epsilon": [0.001, 0.01, 0.03, 0.1],
            },
            feature_set="compact",
            randomized_iterations=32,
        ),
        SearchDefinition(
            name="mlp",
            estimator=feature_pipeline(
                "full",
                MLPRegressor(
                    max_iter=2_000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=seed,
                ),
            ),
            params={
                "model__hidden_layer_sizes": [
                    (16,),
                    (32,),
                    (64,),
                    (32, 16),
                    (64, 32),
                    (128, 64, 32),
                ],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
                "model__learning_rate_init": [0.0003, 0.001, 0.003],
            },
            feature_set="full",
            randomized_iterations=28,
        ),
    ]

    try:
        from xgboost import XGBRegressor

        definitions.append(
            SearchDefinition(
                name="xgboost",
                estimator=feature_pipeline(
                    "full",
                    XGBRegressor(
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=1,
                        random_state=seed,
                        verbosity=0,
                    ),
                ),
                params={
                    "model__n_estimators": [200, 400, 800, 1200],
                    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "model__max_depth": [1, 2, 3, 4, 6],
                    "model__min_child_weight": [1, 5, 20],
                    "model__subsample": [0.7, 0.9, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                    "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
                    "model__reg_lambda": [0.1, 1.0, 10.0, 100.0],
                },
                feature_set="full",
                randomized_iterations=40,
            )
        )
    except ImportError:
        pass
    if not include_parameter_count:
        parameter_only = {
            "max_params_heuristic",
            "params_ridge",
            "params_polynomial_ridge",
        }
        definitions = [
            definition
            for definition in definitions
            if definition.name not in parameter_only
        ]
    return [
        replace(definition, dataset_conditioning=dataset_conditioning)
        for definition in definitions
    ]


def _search_rows(search: GridSearchCV | RandomizedSearchCV) -> list[dict[str, Any]]:
    results = search.cv_results_
    order = np.argsort(results["rank_test_rmse"])
    rows = []
    for index in order[: min(20, len(order))]:
        rows.append(
            {
                "rank": int(results["rank_test_rmse"][index]),
                "rmse_log": float(-results["mean_test_rmse"][index]),
                "rmse_log_std": float(results["std_test_rmse"][index]),
                "mae_log": float(-results["mean_test_mae"][index]),
                "r2_log": float(results["mean_test_r2"][index]),
                "params": results["params"][index],
            }
        )
    return rows


def fit_definition(
    definition: SearchDefinition,
    rows: list[dict[str, Any]],
    y_log: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    jobs: int,
    seed: int,
) -> FittedModel:
    """Tune on CV folds, refit all training rows, and retain OOF predictions.

    Hyperparameters are selected globally with ``folds`` before generating
    fixed-hyperparameter OOF predictions with the same folds.  Consequently
    the stored ``cv_*`` values are selection-CV diagnostics rather than an
    unbiased nested-CV estimate.  Locked holdout results are the final model
    comparison.
    """

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict

    X = np.asarray(rows, dtype=object)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    if definition.params:
        if definition.randomized_iterations is None:
            search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
                definition.estimator,
                definition.params,
                scoring=scoring,
                refit="rmse",
                cv=folds,
                n_jobs=jobs,
                error_score="raise",
                return_train_score=False,
            )
        else:
            search = RandomizedSearchCV(
                definition.estimator,
                definition.params,
                n_iter=definition.randomized_iterations,
                scoring=scoring,
                refit="rmse",
                cv=folds,
                n_jobs=jobs,
                random_state=seed,
                error_score="raise",
                return_train_score=False,
            )
        search.fit(X, y_log)
        estimator = search.best_estimator_
        best_params = search.best_params_
        search_rows = _search_rows(search)
    else:
        estimator = clone(definition.estimator).fit(X, y_log)
        best_params = {}
        search_rows = []

    oof = cross_val_predict(
        clone(estimator),
        X,
        y_log,
        cv=folds,
        n_jobs=jobs,
        method="predict",
    )
    estimator.fit(X, y_log)
    return FittedModel(
        name=definition.name,
        estimator=estimator,
        feature_set=definition.feature_set,
        best_params=best_params,
        cv_rmse_log=float(math.sqrt(mean_squared_error(y_log, oof))),
        cv_mae_log=float(mean_absolute_error(y_log, oof)),
        cv_r2_log=float(r2_score(y_log, oof)),
        oof_predictions=np.asarray(oof, dtype=float),
        search_rows=search_rows,
        interpretable=definition.interpretable,
        dataset_conditioning=definition.dataset_conditioning,
    )


def fit_positive_ensemble(
    models: list[FittedModel],
    y_log: np.ndarray,
    *,
    name: str = "oof_positive_ensemble",
) -> EnsembleModel:
    """Fit non-negative stacking weights on base OOF prediction columns.

    The returned ``cv_*`` fields score the combiner on the same OOF matrix used
    to fit its weights.  These apparent meta-fit values are diagnostic only and
    must not participate in CV champion selection.  This function never
    touches locked holdout labels; holdout/external evaluation remains honest.
    """

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    targets = np.asarray(y_log, dtype=float)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("y_log must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(targets)):
        raise ValueError("y_log contains non-finite values")
    if not models:
        raise ValueError("fit_positive_ensemble requires at least one model")
    columns = []
    for model in models:
        column = np.asarray(model.oof_predictions, dtype=float)
        if column.ndim != 1 or column.size != targets.size:
            raise ValueError(
                f"Model {model.name!r} has OOF shape {column.shape}; "
                f"expected ({targets.size},)"
            )
        if not np.all(np.isfinite(column)):
            raise ValueError(f"Model {model.name!r} OOF predictions are non-finite")
        columns.append(column)
    matrix = np.column_stack(columns)
    combiner = LinearRegression(positive=True).fit(matrix, targets)
    weights = np.asarray(combiner.coef_, dtype=float)
    # An intercept-only solution is valid when no base OOF column adds signal.
    # Replacing it with arbitrary uniform weights would discard the fitted mean.
    intercept = float(combiner.intercept_)
    predictions = intercept + matrix @ weights
    return EnsembleModel(
        name=name,
        models=models,
        weights=weights,
        intercept=intercept,
        cv_rmse_log=float(math.sqrt(mean_squared_error(targets, predictions))),
        cv_mae_log=float(mean_absolute_error(targets, predictions)),
        cv_r2_log=float(r2_score(targets, predictions)),
        oof_predictions=np.asarray(predictions, dtype=float),
    )
