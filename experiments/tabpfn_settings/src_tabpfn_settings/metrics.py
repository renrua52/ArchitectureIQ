"""Evaluation helpers for setting→metric prediction."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def pairwise_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, lower_is_better: bool = True) -> float:
    """Fraction of pairs where predicted order matches true order."""
    n = len(y_true)
    if n < 2:
        return float("nan")
    correct = 0
    total = 0
    for i, j in itertools.combinations(range(n), 2):
        ti, tj = float(y_true[i]), float(y_true[j])
        pi, pj = float(y_pred[i]), float(y_pred[j])
        if ti == tj:
            continue
        true_ij = ti < tj if lower_is_better else ti > tj
        pred_ij = pi < pj if lower_is_better else pi > pj
        correct += int(true_ij == pred_ij)
        total += 1
    return correct / total if total else float("nan")


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, *, lower_is_better: bool = True) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        spearman = float("nan")
    else:
        # Pearson on ranks
        rt = y_true.argsort().argsort().astype(float)
        rp = y_pred.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rt, rp)[0, 1])
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "pairwise_acc": pairwise_ranking_accuracy(y_true, y_pred, lower_is_better=lower_is_better),
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
    }
