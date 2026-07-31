from __future__ import annotations

import pytest

from tools.meta_model_study.analyze_wide_report import (
    environment_bootstrap_ci,
    metric_view,
)
from tools.meta_model_study.render_setting_to_loss_report import environment_rows


def _method(accuracy: float) -> dict:
    detail = {
        "raw": {"rmse": 2.0},
        "log": {"rmse": 0.2},
        "ranking": {"spearman": 0.7},
        "three_choice": {
            "all": {
                "accuracy": accuracy,
                "regret": {"log": {"mean": 0.03}},
            },
            "gap_ge_0_05": {"accuracy": min(1.0, accuracy + 0.1)},
        },
    }
    return {
        "test": {
            "all": {
                "n": 33,
                "raw": detail["raw"],
                "log": detail["log"],
                "ranking": detail["ranking"],
                "per_environment": {"env_1": detail},
                "within_environment": {
                    "macro": {
                        "log_rmse": 0.2,
                        "spearman": 0.7,
                        "three_choice_accuracy": accuracy,
                    },
                    "three_choice": {"accuracy": accuracy},
                },
            }
        }
    }


def test_report_view_keeps_per_environment_metrics_at_their_real_schema_level() -> None:
    with_metrics = metric_view(_method(0.6))
    without_metrics = metric_view(_method(0.7))

    assert with_metrics["per_environment"]["env_1"]["raw"]["rmse"] == 2.0
    ci = environment_bootstrap_ci({"family": with_metrics}, draws=100)
    assert ci is not None
    assert ci["lower"] == pytest.approx(0.6)
    assert ci["upper"] == pytest.approx(0.6)

    rendered = environment_rows(
        {
            "per_environment": [
                {
                    "task_id": "env_1",
                    "family": "univariate_regression",
                    "with_parameter_count": {"method": "xgboost", **with_metrics},
                    "without_parameter_count": {
                        "method": "extra_trees",
                        **without_metrics,
                    },
                    "three_choice_delta_without_minus_with": 0.1,
                }
            ]
        }
    )
    assert "2.000" in rendered
    assert "60.0%" in rendered
    assert "70.0%" in rendered
