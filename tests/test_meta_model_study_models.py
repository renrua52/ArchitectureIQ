from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError

from tools.meta_model_study.models import (
    FittedModel,
    MaxParamsHeuristic,
    OptimizerLrLookupRegressor,
    SearchDefinition,
    fit_definition,
    fit_positive_ensemble,
    search_definitions,
)


def _row(
    index: int,
    *,
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    params: int | None = None,
) -> dict[str, Any]:
    total_params = params if params is not None else 100 + index
    return {
        "setting": {
            "optimizer": {"type": optimizer, "lr": learning_rate},
            "model": {
                "type": "mlp",
                "input_dim": 1,
                "depth": 1,
                "width": 8,
                "residual": False,
                "activations": ["relu"],
                "layer_norm": [False],
            },
            "loss": {"loss_id": "mse"},
            "budget": {
                "training_steps": 4,
                "batch_size": 8,
                "total_samples_seen": 32,
            },
        },
        "derived": {
            "total_params": total_params,
            "trainable_params": total_params,
            "log_total_params": math.log(total_params),
        },
        # This deliberately extreme value must never be read by an estimator.
        "target": {"mean_loss": 1e100 + index},
    }


def test_lookup_regressor_shrinks_cells_and_falls_back_for_unseen_key() -> None:
    rows = [
        _row(0),
        _row(1),
        _row(2, optimizer="SGD", learning_rate=0.1),
    ]
    targets = np.asarray([1.0, 3.0, 10.0])
    model = OptimizerLrLookupRegressor(shrinkage=2.0).fit(rows, targets)

    predictions = model.predict(
        [rows[0], rows[2], _row(3, optimizer="RMSprop", learning_rate=0.03)]
    )

    global_mean = 14.0 / 3.0
    assert predictions[0] == pytest.approx(0.5 * 2.0 + 0.5 * global_mean)
    assert predictions[1] == pytest.approx((1.0 / 3.0) * 10.0 + (2.0 / 3.0) * global_mean)
    assert predictions[2] == pytest.approx(global_mean)
    assert model.counts_ == {("Adam", 0.001): 2, ("SGD", 0.1): 1}


def test_lookup_regressor_obeys_sklearn_input_and_fitted_state_contract() -> None:
    model = OptimizerLrLookupRegressor()

    with pytest.raises(NotFittedError):
        model.predict([_row(0)])
    with pytest.raises(ValueError, match="same number"):
        model.fit([_row(0), _row(1)], np.asarray([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        OptimizerLrLookupRegressor(shrinkage=-1.0).fit(
            [_row(0)], np.asarray([1.0])
        )


def test_max_params_heuristic_is_cloneable_and_ignores_targets() -> None:
    rows = [_row(0, params=100), _row(1, params=10_000)]
    model = clone(MaxParamsHeuristic()).fit(rows, np.asarray([1000.0, -1000.0]))

    prediction = model.predict(rows)

    assert prediction.tolist() == pytest.approx([-math.log(100), -math.log(10_000)])
    assert prediction[1] < prediction[0]


def _three_folds() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
        (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
    ]


def test_fit_definition_retains_true_fixed_model_oof_predictions() -> None:
    rows = [_row(index) for index in range(6)]
    y_log = np.arange(6, dtype=float)
    definition = SearchDefinition(
        name="constant",
        estimator=DummyRegressor(strategy="mean"),
        params={},
        interpretable=True,
    )

    fitted = fit_definition(
        definition,
        rows,
        y_log,
        _three_folds(),
        jobs=1,
        seed=7,
    )

    # Each pair is predicted by the mean of the other four rows.  Predictions
    # from a model refitted on all six rows would instead all equal 2.5.
    assert fitted.oof_predictions.tolist() == pytest.approx(
        [3.5, 3.5, 2.5, 2.5, 1.5, 1.5]
    )
    assert fitted.predict_log([_row(10), _row(11)]).tolist() == [2.5, 2.5]
    assert fitted.cv_rmse_log == pytest.approx(
        np.sqrt(np.mean((y_log - fitted.oof_predictions) ** 2))
    )
    assert fitted.cv_mae_log == pytest.approx(
        np.mean(np.abs(y_log - fitted.oof_predictions))
    )
    assert fitted.feature_set is None
    assert fitted.interpretable is True


@pytest.mark.parametrize("randomized_iterations", [None, 1])
def test_fit_definition_supports_grid_and_randomized_search(
    randomized_iterations: int | None,
) -> None:
    rows = [
        _row(index, optimizer="Adam" if index % 2 else "SGD")
        for index in range(6)
    ]
    y_log = np.asarray([1.0, 4.0, 1.2, 4.2, 0.8, 3.8])
    definition = SearchDefinition(
        name="lookup_search",
        estimator=OptimizerLrLookupRegressor(),
        params={"shrinkage": [0.0, 10.0]},
        randomized_iterations=randomized_iterations,
    )

    fitted = fit_definition(
        definition,
        rows,
        y_log,
        _three_folds(),
        jobs=1,
        seed=11,
    )

    assert fitted.best_params["shrinkage"] in {0.0, 10.0}
    assert len(fitted.search_rows) == (2 if randomized_iterations is None else 1)
    assert fitted.search_rows[0]["rank"] == 1
    assert fitted.oof_predictions.shape == (6,)


def test_fit_definition_integrates_fold_local_feature_pipeline() -> None:
    rows = [_row(index, params=2 ** (index + 5)) for index in range(12)]
    y_log = np.asarray(
        [0.4 * row["derived"]["log_total_params"] for row in rows]
    )
    definition = next(
        item for item in search_definitions(seed=3) if item.name == "params_ridge"
    )
    folds = []
    indices = np.arange(12)
    for start in (0, 4, 8):
        test = indices[start : start + 4]
        train = np.setdiff1d(indices, test)
        folds.append((train, test))

    fitted = fit_definition(
        definition,
        rows,
        y_log,
        folds,
        jobs=1,
        seed=3,
    )

    encoder = fitted.estimator.named_steps["features"]
    assert encoder.feature_names == ["derived.log_total_params"]
    assert fitted.oof_predictions.shape == (12,)
    assert fitted.predict_log([_row(20, params=2**20)]).shape == (1,)


class RowValueRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, key: str = "x", scale: float = 1.0) -> None:
        self.key = key
        self.scale = scale

    def fit(self, X: Any, y: Any = None) -> RowValueRegressor:
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray([float(row[self.key]) * self.scale for row in X])


def _fitted_member(
    name: str,
    oof: np.ndarray,
    *,
    key: str,
    scale: float = 1.0,
) -> FittedModel:
    return FittedModel(
        name=name,
        estimator=RowValueRegressor(key=key, scale=scale),
        feature_set=None,
        best_params={},
        cv_rmse_log=0.0,
        cv_mae_log=0.0,
        cv_r2_log=0.0,
        oof_predictions=np.asarray(oof, dtype=float),
        search_rows=[],
    )


def test_positive_ensemble_uses_oof_columns_and_final_member_predictions() -> None:
    x = np.arange(8, dtype=float)
    z = np.asarray([0.0, 1.0] * 4)
    y_log = 0.7 + 2.0 * x + 3.0 * z
    members = [
        _fitted_member("x", x, key="x"),
        _fitted_member("z", z, key="z"),
    ]

    ensemble = fit_positive_ensemble(members, y_log)

    assert ensemble.weights.tolist() == pytest.approx([2.0, 3.0])
    assert ensemble.intercept == pytest.approx(0.7)
    assert ensemble.oof_predictions == pytest.approx(y_log)
    assert ensemble.cv_rmse_log == pytest.approx(0.0, abs=1e-12)
    assert ensemble.predict_log([{"x": 2.0, "z": 1.0}]) == pytest.approx([7.7])
    assert ensemble.best_params["members"] == ["x", "z"]


def test_positive_ensemble_preserves_valid_intercept_only_solution() -> None:
    y_log = np.asarray([1.0, 2.0, 3.0, 4.0])
    zero = np.zeros_like(y_log)
    members = [
        _fitted_member("zero_a", zero, key="x", scale=0.0),
        _fitted_member("zero_b", zero, key="z", scale=0.0),
    ]

    ensemble = fit_positive_ensemble(members, y_log)

    assert ensemble.weights.tolist() == [0.0, 0.0]
    assert ensemble.intercept == pytest.approx(np.mean(y_log))
    assert ensemble.oof_predictions.tolist() == [2.5, 2.5, 2.5, 2.5]
    assert ensemble.predict_log([{"x": 99.0, "z": -5.0}]).tolist() == [2.5]


def test_positive_ensemble_rejects_empty_or_misaligned_members() -> None:
    y_log = np.asarray([1.0, 2.0, 3.0])
    member = _fitted_member("short", np.asarray([1.0, 2.0]), key="x")

    with pytest.raises(ValueError, match="at least one"):
        fit_positive_ensemble([], y_log)
    with pytest.raises(ValueError, match="OOF"):
        fit_positive_ensemble([member], y_log)


def test_registered_search_definitions_have_unique_names_and_valid_params() -> None:
    definitions = search_definitions(seed=5)
    names = [definition.name for definition in definitions]

    assert len(names) == len(set(names))
    for definition in definitions:
        clone(definition.estimator)
        known_params = definition.estimator.get_params(deep=True)
        assert set(definition.params).issubset(known_params)
        for values in definition.params.values():
            assert len(values) == len({repr(value) for value in values})
        if definition.randomized_iterations is not None:
            assert definition.randomized_iterations > 0
def test_search_definitions_record_dataset_conditioning_in_encoders() -> None:
    definitions = search_definitions(seed=3, dataset_conditioning="description")
    encoded = [item for item in definitions if item.feature_set is not None]
    assert encoded
    for definition in encoded:
        assert definition.estimator.named_steps["features"].dataset_conditioning == "description"
