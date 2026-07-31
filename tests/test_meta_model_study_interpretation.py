from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from tools.meta_model_study.features import FeatureEncoder
from tools.meta_model_study.interpretation import (
    build_interpretation,
    extract_model_interpretation,
    summarize_simple_rules,
)
from tools.meta_model_study.models import (
    EnsembleModel,
    FittedModel,
    OptimizerLrLookupRegressor,
)


def _row(
    index: int,
    *,
    optimizer: str,
    learning_rate: float,
    params: int,
    raw_loss: float,
) -> dict[str, Any]:
    return {
        "setting": {
            "model": {
                "type": "mlp",
                "input_dim": 2,
                "depth": 1 + index % 3,
                "width": 8 * (1 + index % 4),
                "residual": bool(index % 2),
                "activations": ["relu"] * (1 + index % 3),
                "layer_norm": [bool(index % 2)] * (1 + index % 3),
            },
            "optimizer": {"type": optimizer, "lr": learning_rate},
            "loss": {"loss_id": "mse"},
            "budget": {
                "batch_size": 32,
                "training_steps": 64,
                "total_samples_seen": 2048,
            },
        },
        "derived": {
            "total_params": params,
            "trainable_params": params,
            "log_total_params": math.log(params),
        },
        "target": {
            "mean_loss": raw_loss,
            "log_mean_loss": math.log(raw_loss),
        },
        # Interpretation must ignore arbitrary non-training metadata.
        "provenance": {"sentinel": 10**100},
    }


def _rows() -> list[dict[str, Any]]:
    return [
        _row(0, optimizer="Adam", learning_rate=1e-3, params=100, raw_loss=8.0),
        _row(1, optimizer="Adam", learning_rate=1e-3, params=200, raw_loss=4.0),
        _row(2, optimizer="Adam", learning_rate=1e-2, params=300, raw_loss=2.0),
        _row(3, optimizer="Adam", learning_rate=1e-2, params=400, raw_loss=1.0),
        _row(4, optimizer="SGD", learning_rate=1e-2, params=500, raw_loss=1.0),
        _row(5, optimizer="SGD", learning_rate=1e-2, params=600, raw_loss=2.0),
        _row(6, optimizer="SGD", learning_rate=1e-1, params=700, raw_loss=4.0),
        _row(7, optimizer="SGD", learning_rate=1e-1, params=800, raw_loss=8.0),
    ]


def _fitted(
    name: str,
    estimator: Any,
    *,
    feature_set: str | None,
) -> FittedModel:
    return FittedModel(
        name=name,
        estimator=estimator,
        feature_set=feature_set,
        best_params={},
        cv_rmse_log=0.0,
        cv_mae_log=0.0,
        cv_r2_log=0.0,
        oof_predictions=np.zeros(8),
        search_rows=[],
        interpretable=True,
    )


def test_simple_rules_report_optimizer_lr_correlations_and_parameter_bins() -> None:
    rules = summarize_simple_rules(_rows(), parameter_bins=4)

    optimizer_lr = rules["optimizer_lr"]
    assert sum(cell["count"] for cell in optimizer_lr["cells"]) == 8
    best = {
        cell["optimizer"]: cell["learning_rate"]
        for cell in optimizer_lr["best_lr_by_optimizer"]
    }
    assert best == {"Adam": pytest.approx(1e-2), "SGD": pytest.approx(1e-2)}

    correlations = rules["log_params_loss_spearman"]
    assert correlations["overall"]["count"] == 8
    by_optimizer = {
        row["optimizer"]: row for row in correlations["by_optimizer"]
    }
    assert by_optimizer["Adam"]["spearman_vs_log_loss"] == pytest.approx(-1.0)
    assert by_optimizer["SGD"]["spearman_vs_log_loss"] == pytest.approx(1.0)
    assert by_optimizer["Adam"]["spearman_vs_raw_loss"] == pytest.approx(-1.0)

    parameter_bins = rules["parameter_quantile_bins"]
    assert parameter_bins["observed_bins"] == 4
    assert [item["count"] for item in parameter_bins["bins"]] == [2, 2, 2, 2]
    assert parameter_bins["bins"][0]["mean_raw_loss"] == pytest.approx(6.0)


def test_extracts_linear_coefficients_tree_and_optimizer_lookup() -> None:
    rows = _rows()
    X = np.asarray(rows, dtype=object)
    y_log = np.asarray([row["target"]["log_mean_loss"] for row in rows])

    ridge_pipeline = Pipeline(
        [("features", FeatureEncoder("optimizer_lr")), ("model", Ridge(alpha=1.0))]
    ).fit(X, y_log)
    elastic_pipeline = Pipeline(
        [
            ("features", FeatureEncoder("compact")),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20_000)),
        ]
    ).fit(X, y_log)
    tree_pipeline = Pipeline(
        [
            ("features", FeatureEncoder("compact")),
            ("model", DecisionTreeRegressor(max_depth=2, random_state=7)),
        ]
    ).fit(X, y_log)
    lookup = OptimizerLrLookupRegressor(shrinkage=1.0).fit(X, y_log)

    ridge = extract_model_interpretation(
        _fitted("ridge", ridge_pipeline, feature_set="optimizer_lr")
    )
    elastic = extract_model_interpretation(
        _fitted("elastic", elastic_pipeline, feature_set="compact")
    )
    tree = extract_model_interpretation(
        _fitted("tree", tree_pipeline, feature_set="compact")
    )
    lookup_result = extract_model_interpretation(
        _fitted("lookup", lookup, feature_set=None)
    )

    ridge_linear = ridge["linear_model"]
    assert ridge_linear["n_features"] == len(ridge_linear["coefficients"])
    assert any(
        item["feature"] == "derived.log_total_params"
        for item in ridge_linear["coefficients"]
    )
    assert elastic["linear_model"]["n_features"] > ridge_linear["n_features"]
    assert tree["feature_importances"]["n_features"] > 0
    assert tree["shallow_tree_text"].startswith("|---")
    assert "class:" not in tree["shallow_tree_text"]

    table = lookup_result["optimizer_lr_lookup"]
    assert table["fitted"] is True
    assert table["shrinkage"] == 1.0
    assert len(table["cells"]) == 4
    assert sum(cell["count"] for cell in table["cells"]) == 8
    assert all(cell["predicted_raw_loss"] > 0 for cell in table["cells"])


def test_complete_artifact_includes_ensemble_and_is_strict_json() -> None:
    rows = _rows()
    X = np.asarray(rows, dtype=object)
    y_log = np.asarray([row["target"]["log_mean_loss"] for row in rows])
    lookup_estimator = OptimizerLrLookupRegressor().fit(X, y_log)
    lookup = _fitted("lookup", lookup_estimator, feature_set=None)
    ensemble = EnsembleModel(
        name="stack",
        models=[lookup],
        weights=np.asarray([0.75]),
        intercept=0.25,
        cv_rmse_log=0.0,
        cv_mae_log=0.0,
        cv_r2_log=1.0,
        oof_predictions=y_log,
    )

    result = build_interpretation(
        rows,
        {"lookup_key": lookup, "ensemble_key": ensemble},
        parameter_bins=4,
    )

    assert result["label_source"] == "training rows only"
    assert result["n_train_rows"] == 8
    assert result["models"]["ensemble_key"]["members"] == [
        {"name": "lookup", "weight": 0.75}
    ]
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "sentinel" not in encoded


def test_rules_reject_empty_rows_and_inconsistent_targets() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_simple_rules([])
    bad = _rows()[0]
    bad["target"]["log_mean_loss"] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_simple_rules([bad])
