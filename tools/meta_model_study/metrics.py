"""Metrics for setting-to-loss meta-model experiments.

All losses in this module use the benchmark convention that lower is better.
The three-choice metrics are exact over every :math:`\binom{n}{3}` validation
group; they do not sample synthetic questions.  The split-half ceiling reads
the final per-seed values from the stored ``curves.npz`` files, preserving the
repository's generated-code-to-ground-truth contract.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MetricValue = float | int | None


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired_vectors(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = _finite_vector(y_true, name="y_true")
    prediction = _finite_vector(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{truth.shape} and {prediction.shape}"
        )
    if truth.size == 0:
        raise ValueError("At least one prediction is required")
    return truth, prediction


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, and the finite-sample R-squared score.

    For a constant target, R-squared follows scikit-learn's finite convention:
    a perfect prediction scores 1 and every imperfect prediction scores 0.
    """

    truth, prediction = _paired_vectors(y_true, y_pred)
    residual = truth - prediction
    squared_error = float(np.dot(residual, residual))
    total_error = float(np.dot(truth - np.mean(truth), truth - np.mean(truth)))
    if total_error == 0.0:
        r2 = 1.0 if squared_error == 0.0 else 0.0
    else:
        r2 = 1.0 - squared_error / total_error
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(squared_error / truth.size)),
        "r2": float(r2),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Equivalent to scipy.stats.rankdata(method="average"), without SciPy."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Ranks are conventionally one-based; correlation is translation invariant.
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def ranking_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, MetricValue]:
    """Return Spearman rho, Kendall tau-b, and pair concordance.

    Pair concordance is the C-index over pairs with distinct true losses.
    Correctly ordered pairs receive one point and predicted ties receive half a
    point.  True-loss ties are not comparable and are omitted from its
    denominator.  Undefined correlations (for example a constant prediction)
    are represented as ``None`` rather than a fabricated numeric score.
    """

    truth, prediction = _paired_vectors(y_true, y_pred)
    spearman = _pearson(_average_ranks(truth), _average_ranks(prediction))

    concordant = 0
    discordant = 0
    tied_true = 0
    tied_prediction = 0
    comparable = 0
    concordance_score = 0.0
    for left, right in combinations(range(truth.size), 2):
        true_delta = truth[left] - truth[right]
        pred_delta = prediction[left] - prediction[right]
        if true_delta == 0.0:
            tied_true += 1
        else:
            comparable += 1
            if pred_delta == 0.0:
                concordance_score += 0.5
            elif true_delta * pred_delta > 0.0:
                concordance_score += 1.0

        if pred_delta == 0.0:
            tied_prediction += 1
        if true_delta != 0.0 and pred_delta != 0.0:
            if true_delta * pred_delta > 0.0:
                concordant += 1
            else:
                discordant += 1

    total_pairs = math.comb(truth.size, 2)
    kendall_denominator = math.sqrt(
        (total_pairs - tied_true) * (total_pairs - tied_prediction)
    )
    kendall = (
        (concordant - discordant) / kendall_denominator
        if kendall_denominator > 0.0
        else None
    )
    pair_concordance = concordance_score / comparable if comparable else None
    return {
        "spearman": spearman,
        "kendall_tau_b": float(kendall) if kendall is not None else None,
        "pair_concordance": (
            float(pair_concordance) if pair_concordance is not None else None
        ),
        "n_pairs": total_pairs,
        "n_comparable_pairs": comparable,
    }


@lru_cache(maxsize=32)
def _three_choice_indices(n_rows: int) -> np.ndarray:
    """Return a cached, read-only array containing every index triple."""

    if n_rows < 3:
        result = np.empty((0, 3), dtype=np.int64)
    else:
        result = np.fromiter(
            (index for triple in combinations(range(n_rows), 3) for index in triple),
            dtype=np.int64,
            count=3 * math.comb(n_rows, 3),
        ).reshape(-1, 3)
    result.setflags(write=False)
    return result


def _winner_summary(
    true_groups: np.ndarray,
    selected_losses: np.ndarray,
) -> dict[str, MetricValue]:
    n_groups = int(true_groups.shape[0])
    if n_groups == 0:
        return {
            "n_groups": 0,
            "accuracy": None,
            "mean_regret": None,
            "median_regret": None,
        }
    best_losses = np.min(true_groups, axis=1)
    # A prediction selecting either member of an exact GT tie is correct.
    correct = selected_losses == best_losses
    regrets = np.maximum(selected_losses - best_losses, 0.0)
    return {
        "n_groups": n_groups,
        "accuracy": float(np.mean(correct)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
    }


def winner_metrics_3choice(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    gap_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate winner selection and regret over all three-choice groups.

    The gap-filtered subset uses the absolute true-loss difference between the
    best and second-best choices.  Prediction ties are broken by the original
    row order (NumPy's stable first-minimum behavior); exact true-winner ties
    receive full credit.
    """

    truth, prediction = _paired_vectors(y_true, y_pred)
    if not math.isfinite(gap_threshold) or gap_threshold < 0.0:
        raise ValueError("gap_threshold must be a finite non-negative number")
    triples = _three_choice_indices(truth.size)
    if triples.shape[0] == 0:
        summary = _winner_summary(np.empty((0, 3)), np.empty(0))
        return {
            **summary,
            "gap_ge_0_05": {"threshold": float(gap_threshold), **summary},
        }

    true_groups = truth[triples]
    predicted_groups = prediction[triples]
    selected_positions = np.argmin(predicted_groups, axis=1)
    selected_losses = true_groups[np.arange(triples.shape[0]), selected_positions]
    result = _winner_summary(true_groups, selected_losses)

    two_smallest = np.partition(true_groups, kth=1, axis=1)[:, :2]
    gaps = np.max(two_smallest, axis=1) - np.min(two_smallest, axis=1)
    gap_mask = gaps >= gap_threshold
    return {
        **result,
        "gap_ge_0_05": {
            "threshold": float(gap_threshold),
            **_winner_summary(true_groups[gap_mask], selected_losses[gap_mask]),
        },
    }


def _dual_scale_winner_summary(
    true_groups: np.ndarray,
    selected_losses: np.ndarray,
) -> dict[str, Any]:
    """Summarize winner accuracy and regret in raw- and log-loss space."""

    n_groups = int(true_groups.shape[0])
    if n_groups == 0:
        return {
            "n_groups": 0,
            "accuracy": None,
            "regret": {
                "raw": {"mean": None, "median": None},
                "log": {"mean": None, "median": None},
            },
        }

    best_losses = np.min(true_groups, axis=1)
    correct = selected_losses == best_losses
    raw_regrets = np.maximum(selected_losses - best_losses, 0.0)
    log_regrets = np.maximum(
        np.log(selected_losses) - np.log(best_losses),
        0.0,
    )
    return {
        "n_groups": n_groups,
        "accuracy": float(np.mean(correct)),
        "regret": {
            "raw": {
                "mean": float(np.mean(raw_regrets)),
                "median": float(np.median(raw_regrets)),
            },
            "log": {
                "mean": float(np.mean(log_regrets)),
                "median": float(np.median(log_regrets)),
            },
        },
    }


def log_loss_prediction_metrics(
    y_true_raw: Sequence[float] | np.ndarray,
    y_pred_log: Sequence[float] | np.ndarray,
    *,
    gap_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate one environment from predictions of ``log(mean_loss)``.

    Predictions are exponentiated exactly once for raw-loss regression.  The
    same aligned rows are then evaluated in log space, for within-environment
    ranking, and over every three-choice combination.  Three-choice regret is
    reported both as an absolute raw-loss difference and as a log loss ratio.

    The gap subset uses the absolute raw-loss difference between the best and
    second-best ground-truth choices.  Callers must not combine environments;
    :func:`evaluate_environment_log_predictions` enforces that condition for
    exported setting rows.
    """

    truth = _finite_vector(y_true_raw, name="y_true_raw")
    prediction_log = _finite_vector(y_pred_log, name="y_pred_log")
    if truth.shape != prediction_log.shape:
        raise ValueError(
            "y_true_raw and y_pred_log must have the same shape, got "
            f"{truth.shape} and {prediction_log.shape}"
        )
    if truth.size == 0:
        raise ValueError("At least one prediction is required")
    if np.any(truth <= 0.0):
        raise ValueError("Raw ground-truth losses must be strictly positive")
    if not math.isfinite(gap_threshold) or gap_threshold < 0.0:
        raise ValueError("gap_threshold must be a finite non-negative number")

    with np.errstate(over="raise", invalid="raise"):
        try:
            prediction_raw = np.exp(prediction_log)
        except FloatingPointError as error:
            raise ValueError("Log-space predictions overflowed") from error

    triples = _three_choice_indices(truth.size)
    if triples.shape[0] == 0:
        empty_groups = np.empty((0, 3), dtype=np.float64)
        empty_losses = np.empty(0, dtype=np.float64)
        all_summary = _dual_scale_winner_summary(empty_groups, empty_losses)
        gap_summary = _dual_scale_winner_summary(empty_groups, empty_losses)
    else:
        true_groups = truth[triples]
        predicted_groups = prediction_raw[triples]
        selected_positions = np.argmin(predicted_groups, axis=1)
        selected_losses = true_groups[
            np.arange(triples.shape[0]), selected_positions
        ]
        all_summary = _dual_scale_winner_summary(true_groups, selected_losses)

        two_smallest = np.partition(true_groups, kth=1, axis=1)[:, :2]
        gaps = np.max(two_smallest, axis=1) - np.min(two_smallest, axis=1)
        gap_mask = gaps >= gap_threshold
        gap_summary = _dual_scale_winner_summary(
            true_groups[gap_mask], selected_losses[gap_mask]
        )

    return {
        "n": int(truth.size),
        "prediction_space": "log",
        "raw": regression_metrics(truth, prediction_raw),
        "log": regression_metrics(np.log(truth), prediction_log),
        "ranking": ranking_metrics(truth, prediction_raw),
        "three_choice": {
            "all": all_summary,
            "gap_ge_0_05": {
                "threshold": float(gap_threshold),
                "gap_definition": "second_best_raw_loss - best_raw_loss",
                **gap_summary,
            },
        },
    }


def loss_prediction_metrics(
    y_true_raw: Sequence[float] | np.ndarray,
    y_pred_raw: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Return raw/log regression, ranking, and exact three-choice metrics."""

    truth, prediction = _paired_vectors(y_true_raw, y_pred_raw)
    if np.any(truth <= 0.0):
        raise ValueError("Raw ground-truth losses must be strictly positive")
    raw = regression_metrics(truth, prediction)
    if np.any(prediction <= 0.0):
        log_metrics: dict[str, float | None] = {
            "mae": None,
            "rmse": None,
            "r2": None,
        }
    else:
        log_metrics = regression_metrics(np.log(truth), np.log(prediction))
    return {
        "n": int(truth.size),
        "raw": raw,
        "log": log_metrics,
        "ranking": ranking_metrics(truth, prediction),
        "three_choice": winner_metrics_3choice(truth, prediction),
    }


def _prediction_vector(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float] | np.ndarray | Mapping[str, float],
) -> np.ndarray:
    if isinstance(predictions, Mapping):
        missing = [
            str(row["example_fingerprint_sha256"])
            for row in rows
            if str(row["example_fingerprint_sha256"]) not in predictions
        ]
        if missing:
            raise KeyError(f"Missing predictions for {len(missing)} rows: {missing[:3]}")
        values = [
            predictions[str(row["example_fingerprint_sha256"])] for row in rows
        ]
    else:
        values = predictions
    vector = _finite_vector(values, name="predictions")
    if vector.size != len(rows):
        raise ValueError(f"Expected {len(rows)} predictions, got {vector.size}")
    return vector


def _single_environment_identity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Return the shared environment identity or reject mixed row groups."""

    identities: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows):
        group_labels = row.get("group_labels")
        labels = group_labels if isinstance(group_labels, Mapping) else {}
        experiment_id = row.get("experiment_id")
        group_environment = labels.get("environment")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError(f"rows[{index}] has no experiment_id")
        if group_environment is not None and group_environment != experiment_id:
            raise ValueError(
                f"rows[{index}] group environment disagrees with experiment_id"
            )

        family = row.get("family")
        dataset_id = row.get("dataset_id")
        target = row.get("target")
        selection_metric = (
            target.get("selection_metric") if isinstance(target, Mapping) else None
        )
        for field, value in (
            ("family", family),
            ("dataset_id", dataset_id),
            ("target.selection_metric", selection_metric),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"rows[{index}] has no {field}")

        identities.add(
            (
                experiment_id,
                family,
                dataset_id,
                selection_metric,
            )
        )

    if len(identities) != 1:
        raise ValueError(
            "Environment metrics require rows from exactly one "
            "experiment_id/dataset_id/family/selection_metric group"
        )
    experiment_id, family, dataset_id, selection_metric = identities.pop()
    return {
        "experiment_id": experiment_id,
        "family": family,
        "dataset_id": dataset_id,
        "selection_metric": selection_metric,
    }


def evaluate_environment_log_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float] | np.ndarray | Mapping[str, float],
    *,
    gap_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate log-loss predictions for exactly one exported environment."""

    if not rows:
        raise ValueError("At least one validation row is required")
    identity = _single_environment_identity(rows)
    prediction_log = _prediction_vector(rows, predictions)
    truth = _finite_vector(
        [float(row["target"]["mean_loss"]) for row in rows],
        name="target.mean_loss",
    )
    eligible_mask = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    result: dict[str, Any] = {
        "environment": identity,
        "all": log_loss_prediction_metrics(
            truth,
            prediction_log,
            gap_threshold=gap_threshold,
        ),
        "benchmark_eligible": None,
    }
    if np.any(eligible_mask):
        result["benchmark_eligible"] = log_loss_prediction_metrics(
            truth[eligible_mask],
            prediction_log[eligible_mask],
            gap_threshold=gap_threshold,
        )
    return result


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float] | np.ndarray | Mapping[str, float],
    *,
    prediction_space: str = "raw",
) -> dict[str, Any]:
    """Evaluate aligned validation predictions, including the eligible subset.

    ``predictions`` may be row-aligned values or a mapping keyed by the full
    example fingerprint.  With ``prediction_space="log"``, values are safely
    exponentiated before all metrics are evaluated in both raw and log space.
    """

    if not rows:
        raise ValueError("At least one validation row is required")
    predicted = _prediction_vector(rows, predictions)
    if prediction_space == "log":
        with np.errstate(over="raise", invalid="raise"):
            try:
                predicted = np.exp(predicted)
            except FloatingPointError as error:
                raise ValueError("Log-space predictions overflowed") from error
    elif prediction_space != "raw":
        raise ValueError("prediction_space must be 'raw' or 'log'")

    truth = _finite_vector(
        [float(row["target"]["mean_loss"]) for row in rows],
        name="target.mean_loss",
    )
    eligible_mask = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    result: dict[str, Any] = {
        "all": loss_prediction_metrics(truth, predicted),
        "benchmark_eligible": None,
    }
    if np.any(eligible_mask):
        result["benchmark_eligible"] = loss_prediction_metrics(
            truth[eligible_mask], predicted[eligible_mask]
        )
    return result


def load_final_seed_losses(
    rows: Sequence[Mapping[str, Any]],
    base_dir: str | Path,
    *,
    expected_n_seeds: int = 10,
) -> np.ndarray:
    """Load the final metric for every seed from each row's ``curves.npz``."""

    root = Path(base_dir)
    final_losses: list[np.ndarray] = []
    for row in rows:
        candidate_path = Path(str(row["provenance"]["candidate_path"]))
        if not candidate_path.is_absolute():
            candidate_path = root / candidate_path
        curves_path = candidate_path / "results" / "curves.npz"
        if not curves_path.is_file():
            raise FileNotFoundError(f"Missing stored GT curves: {curves_path}")
        with np.load(curves_path, allow_pickle=False) as archive:
            if "curves" not in archive:
                raise ValueError(f"Missing 'curves' array in {curves_path}")
            curves = np.asarray(archive["curves"], dtype=np.float64)
        if curves.ndim != 2 or curves.shape[1] == 0:
            raise ValueError(
                f"Expected a non-empty 2D curves array in {curves_path}, "
                f"got {curves.shape}"
            )
        if curves.shape[0] != expected_n_seeds:
            raise ValueError(
                f"Expected {expected_n_seeds} seeds in {curves_path}, "
                f"got {curves.shape[0]}"
            )
        final = curves[:, -1]
        if not np.all(np.isfinite(final)) or np.any(final <= 0.0):
            raise ValueError(f"Final seed losses must be finite and positive: {curves_path}")
        stored_mean = float(row["target"]["mean_loss"])
        if not math.isclose(
            float(np.mean(final)), stored_mean, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"Final curves do not reproduce target.mean_loss for {curves_path}"
            )
        final_losses.append(final)
    if not final_losses:
        raise ValueError("At least one validation row is required")
    return np.stack(final_losses, axis=0)


def _median_metric_trees(trees: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Take leafwise medians while preserving invariant count metadata."""

    if not trees:
        raise ValueError("At least one metric tree is required")
    output: dict[str, Any] = {}
    for key in trees[0]:
        values = [tree[key] for tree in trees]
        first = values[0]
        if isinstance(first, Mapping):
            output[key] = _median_metric_trees(values)  # type: ignore[arg-type]
            continue
        if key == "n" or key.startswith("n_") or key == "threshold":
            if all(value == first for value in values[1:]):
                output[key] = first
            else:
                # Gap-filtered group counts can legitimately change because
                # each seed half supplies an independently noisy GT gap.
                output[key] = float(np.median([float(value) for value in values]))
            continue
        numeric = [float(value) for value in values if value is not None]
        output[key] = float(np.median(numeric)) if numeric else None
    return output


def split_half_noise_ceiling(
    rows: Sequence[Mapping[str, Any]],
    base_dir: str | Path,
) -> dict[str, Any]:
    """Estimate a 10-seed complementary 5/5 split-half noise ceiling.

    All 126 unique complementary partitions are used.  Each partition is
    evaluated in both directions (half A predicting half B and vice versa), so
    asymmetric metrics such as R-squared and regret are not biased by an
    arbitrary orientation.  The returned metric tree contains the leafwise
    median over all 252 directed comparisons.
    """

    final_losses = load_final_seed_losses(rows, base_dir, expected_n_seeds=10)
    eligible_mask = np.asarray(
        [row["target"].get("benchmark_eligible") is True for row in rows],
        dtype=bool,
    )
    # Requiring seed index zero in A chooses one representative from each
    # unordered pair {A, complement(A)}: C(9, 4) == 126.
    half_indices = [
        (0, *tail) for tail in combinations(range(1, 10), 4)
    ]
    all_metrics: list[dict[str, Any]] = []
    eligible_metrics: list[dict[str, Any]] = []
    all_seed_indices = set(range(10))
    for half_a_tuple in half_indices:
        half_a = np.asarray(half_a_tuple, dtype=np.int64)
        half_b = np.asarray(sorted(all_seed_indices.difference(half_a_tuple)))
        mean_a = np.mean(final_losses[:, half_a], axis=1)
        mean_b = np.mean(final_losses[:, half_b], axis=1)
        all_metrics.append(loss_prediction_metrics(mean_b, mean_a))
        all_metrics.append(loss_prediction_metrics(mean_a, mean_b))
        if np.any(eligible_mask):
            eligible_metrics.append(
                loss_prediction_metrics(mean_b[eligible_mask], mean_a[eligible_mask])
            )
            eligible_metrics.append(
                loss_prediction_metrics(mean_a[eligible_mask], mean_b[eligible_mask])
            )

    return {
        "n_seeds": 10,
        "half_size": 5,
        "n_complementary_partitions": len(half_indices),
        "n_directed_comparisons": len(all_metrics),
        "all": {
            "n_rows": len(rows),
            "median_metrics": _median_metric_trees(all_metrics),
        },
        "benchmark_eligible": (
            {
                "n_rows": int(np.sum(eligible_mask)),
                "median_metrics": _median_metric_trees(eligible_metrics),
            }
            if eligible_metrics
            else None
        ),
    }
