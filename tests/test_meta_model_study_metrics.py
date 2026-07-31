from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from tools.meta_model_study.metrics import (
    evaluate_environment_log_predictions,
    evaluate_predictions,
    log_loss_prediction_metrics,
    loss_prediction_metrics,
    ranking_metrics,
    regression_metrics,
    split_half_noise_ceiling,
    winner_metrics_3choice,
)


def test_raw_and_log_regression_metrics() -> None:
    truth = np.asarray([1.0, 2.0, 4.0])
    prediction = np.asarray([1.0, 3.0, 5.0])

    raw = regression_metrics(truth, prediction)
    combined = loss_prediction_metrics(truth, prediction)

    assert raw["mae"] == pytest.approx(2.0 / 3.0)
    assert raw["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert raw["r2"] == pytest.approx(4.0 / 7.0)
    assert combined["raw"] == raw
    expected_log_residual = np.log(truth) - np.log(prediction)
    assert combined["log"]["mae"] == pytest.approx(
        np.mean(np.abs(expected_log_residual))
    )
    assert combined["log"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(expected_log_residual**2))
    )


def test_ranking_metrics_handle_reverse_order_and_ties() -> None:
    reverse = ranking_metrics([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    tied_prediction = ranking_metrics([1.0, 2.0, 3.0], [7.0, 7.0, 7.0])

    assert reverse["spearman"] == pytest.approx(-1.0)
    assert reverse["kendall_tau_b"] == pytest.approx(-1.0)
    assert reverse["pair_concordance"] == pytest.approx(0.0)
    assert reverse["n_pairs"] == 3
    assert tied_prediction["spearman"] is None
    assert tied_prediction["kendall_tau_b"] is None
    assert tied_prediction["pair_concordance"] == pytest.approx(0.5)


def _brute_three_choice(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    correct: list[float] = []
    regret: list[float] = []
    gap_correct: list[float] = []
    gap_regret: list[float] = []
    for triple in combinations(range(truth.size), 3):
        indices = np.asarray(triple)
        true_group = truth[indices]
        selected_loss = float(true_group[np.argmin(prediction[indices])])
        best = float(np.min(true_group))
        is_correct = float(selected_loss == best)
        item_regret = max(selected_loss - best, 0.0)
        correct.append(is_correct)
        regret.append(item_regret)
        ordered = np.sort(true_group)
        if ordered[1] - ordered[0] >= threshold:
            gap_correct.append(is_correct)
            gap_regret.append(item_regret)

    def summary(accuracy: list[float], regrets: list[float]) -> dict[str, float | int]:
        return {
            "n_groups": len(regrets),
            "accuracy": float(np.mean(accuracy)),
            "mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets)),
        }

    return summary(correct, regret), summary(gap_correct, gap_regret)


def test_vectorized_three_choice_metrics_equal_brute_force_exactly() -> None:
    rng = np.random.default_rng(9)
    truth = rng.lognormal(size=17)
    prediction = truth + rng.normal(scale=0.7, size=truth.size)
    expected, expected_gap = _brute_three_choice(
        truth, prediction, threshold=0.05
    )

    actual = winner_metrics_3choice(truth, prediction)

    assert actual["n_groups"] == expected["n_groups"] == 680
    assert actual["accuracy"] == pytest.approx(expected["accuracy"])
    assert actual["mean_regret"] == pytest.approx(expected["mean_regret"])
    assert actual["median_regret"] == pytest.approx(expected["median_regret"])
    assert actual["gap_ge_0_05"]["threshold"] == 0.05
    for key, value in expected_gap.items():
        assert actual["gap_ge_0_05"][key] == pytest.approx(value)


def _row(index: int, loss: float, *, eligible: bool) -> dict:
    return {
        "example_fingerprint_sha256": f"fingerprint-{index}",
        "target": {
            "mean_loss": loss,
            "benchmark_eligible": eligible,
        },
        "provenance": {"candidate_path": f"candidate-{index}"},
    }


def test_evaluate_predictions_reports_all_and_benchmark_eligible_subset() -> None:
    rows = [
        _row(0, 1.0, eligible=True),
        _row(1, 2.0, eligible=False),
        _row(2, 3.0, eligible=True),
        _row(3, 4.0, eligible=True),
    ]
    predictions = {
        row["example_fingerprint_sha256"]: np.log(row["target"]["mean_loss"])
        for row in rows
    }

    result = evaluate_predictions(rows, predictions, prediction_space="log")

    assert result["all"]["n"] == 4
    assert result["benchmark_eligible"]["n"] == 3
    assert result["all"]["raw"]["mae"] == pytest.approx(0.0, abs=1e-15)
    assert result["all"]["raw"]["rmse"] == pytest.approx(0.0, abs=1e-15)
    assert result["all"]["raw"]["r2"] == pytest.approx(1.0)
    assert result["benchmark_eligible"]["ranking"]["pair_concordance"] == 1.0


def test_log_loss_prediction_metrics_exponentiates_for_raw_regression() -> None:
    truth = np.asarray([0.5, 2.0, 8.0])
    prediction_log = np.log(truth)

    result = log_loss_prediction_metrics(truth, prediction_log)

    assert result["prediction_space"] == "log"
    assert result["raw"]["mae"] == pytest.approx(0.0, abs=1e-14)
    assert result["raw"]["rmse"] == pytest.approx(0.0, abs=1e-14)
    assert result["raw"]["r2"] == pytest.approx(1.0)
    assert result["log"] == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    assert result["ranking"]["spearman"] == pytest.approx(1.0)
    assert result["three_choice"]["all"]["accuracy"] == pytest.approx(1.0)


def test_log_loss_prediction_metrics_filters_gap_and_reports_both_regrets() -> None:
    truth = np.asarray([1.0, 1.04, 2.0, 4.0])
    prediction_log = np.log([1.5, 1.0, 2.0, 4.0])

    result = log_loss_prediction_metrics(truth, prediction_log)
    all_groups = result["three_choice"]["all"]
    gap_groups = result["three_choice"]["gap_ge_0_05"]

    assert all_groups["n_groups"] == 4
    assert all_groups["accuracy"] == pytest.approx(0.5)
    assert all_groups["regret"]["raw"]["mean"] == pytest.approx(0.02)
    assert all_groups["regret"]["log"]["mean"] == pytest.approx(
        np.log(1.04) / 2.0
    )
    assert gap_groups["threshold"] == 0.05
    assert gap_groups["n_groups"] == 2
    assert gap_groups["accuracy"] == pytest.approx(1.0)
    assert gap_groups["regret"]["raw"]["mean"] == pytest.approx(0.0)
    assert gap_groups["regret"]["log"]["mean"] == pytest.approx(0.0)


def _environment_row(
    index: int,
    loss: float,
    *,
    experiment_id: str = "env-1",
    dataset_id: str = "dataset-1",
) -> dict:
    row = _row(index, loss, eligible=True)
    row.update(
        {
            "experiment_id": experiment_id,
            "family": "univariate_regression",
            "dataset_id": dataset_id,
            "group_labels": {
                "environment": experiment_id,
                "family": "univariate_regression",
                "dataset": dataset_id,
            },
        }
    )
    row["target"]["selection_metric"] = "test_mse"
    return row


def test_environment_log_metrics_reject_cross_dataset_mixing() -> None:
    rows = [
        _environment_row(0, 1.0),
        _environment_row(1, 2.0, experiment_id="env-2", dataset_id="dataset-2"),
        _environment_row(2, 3.0),
    ]

    with pytest.raises(ValueError, match="exactly one"):
        evaluate_environment_log_predictions(rows, np.log([1.0, 2.0, 3.0]))


def test_split_half_noise_ceiling_uses_all_complementary_partitions(
    tmp_path: Path,
) -> None:
    rows = []
    for index, loss in enumerate([0.1, 0.2, 0.4, 0.8, 1.6]):
        row = _row(index, loss, eligible=index != 4)
        rows.append(row)
        results_dir = tmp_path / row["provenance"]["candidate_path"] / "results"
        results_dir.mkdir(parents=True)
        curves = np.column_stack(
            [np.full(10, loss * 2.0), np.full(10, loss)]
        )
        np.savez(
            results_dir / "curves.npz",
            curves=curves,
            samples=np.asarray([1, 2]),
            batch_size=1,
        )

    ceiling = split_half_noise_ceiling(rows, tmp_path)

    assert ceiling["n_complementary_partitions"] == 126
    assert ceiling["n_directed_comparisons"] == 252
    assert ceiling["all"]["n_rows"] == 5
    assert ceiling["benchmark_eligible"]["n_rows"] == 4
    all_median = ceiling["all"]["median_metrics"]
    assert all_median["raw"] == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    assert all_median["ranking"]["spearman"] == pytest.approx(1.0)
    assert all_median["three_choice"]["accuracy"] == pytest.approx(1.0)
    assert all_median["three_choice"]["n_groups"] == 10
    assert (
        ceiling["benchmark_eligible"]["median_metrics"]["three_choice"][
            "n_groups"
        ]
        == 4
    )


def test_split_half_rejects_curves_that_do_not_match_stored_target(
    tmp_path: Path,
) -> None:
    row = _row(0, 1.0, eligible=True)
    results_dir = tmp_path / row["provenance"]["candidate_path"] / "results"
    results_dir.mkdir(parents=True)
    np.savez(results_dir / "curves.npz", curves=np.full((10, 1), 2.0))

    with pytest.raises(ValueError, match="do not reproduce"):
        split_half_noise_ceiling([row], tmp_path)
