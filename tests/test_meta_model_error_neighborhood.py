from __future__ import annotations

import math

import pytest

from tools.meta_model_study.error_neighborhood import (
    _candidate_sources,
    _question_analysis,
)


def _choice(letter: str, *, width: int, optimizer: str, lr: float) -> dict:
    optimizer_spec = {"type": optimizer, "lr": lr, "weight_decay": 0.0}
    if optimizer == "SGD":
        optimizer_spec["momentum"] = 0.9
    if optimizer in {"Adam", "AdamW"}:
        optimizer_spec["betas"] = [0.9, 0.999]
    return {
        "letter": letter,
        "candidate_id": f"c_{letter.lower()}",
        "budget": {
            "training_steps": 4,
            "batch_size": 8,
            "total_samples_seen": 32,
        },
        "model": {
            "type": "mlp",
            "input_dim": 1,
            "depth": 1,
            "width": width,
            "residual": False,
            "activations": ["relu"],
            "layer_norm": [False],
        },
        "optimizer": optimizer_spec,
        "loss": {"loss_id": "mse"},
    }


def _question() -> dict:
    return {
        "question_id": "q_test",
        "family": "univariate_regression",
        "correct_letter": "B",
        "predicted_letter": "A",
        "gt_gap": 0.1,
        "choices": [
            _choice("A", width=8, optimizer="Adam", lr=1e-3),
            _choice("B", width=16, optimizer="SGD", lr=1e-2),
            _choice("C", width=32, optimizer="RMSprop", lr=3e-4),
        ],
    }


def test_candidate_sources_form_complete_factorial_and_mark_diagonal() -> None:
    rows = list(_candidate_sources(_question(), [1e-3, 1e-2]))

    assert len(rows) == 3 * 3 * 2
    assert {
        (source["architecture_letter"], source["optimizer_template_letter"])
        for _, source in rows
    } == {
        (architecture, optimizer)
        for architecture in "ABC"
        for optimizer in "ABC"
    }
    diagonal = [source for _, source in rows if source["is_original_diagonal"]]
    assert {(row["architecture_letter"], row["learning_rate"]) for row in diagonal} == {
        ("A", 1e-3),
        ("B", 1e-2),
    }
    # C's original 3e-4 is intentionally absent from the requested two-LR grid.
    assert all(setting["budget"]["training_steps"] == 4 for setting, _ in rows)


def test_question_analysis_detects_architecture_rank_crossovers() -> None:
    source_rows = []
    # A beats B in cell 1 but loses in cell 2; C is always worse.
    losses = {
        "A": [1.0, 4.0],
        "B": [2.0, 1.5],
        "C": [5.0, 5.0],
    }
    for architecture, values in losses.items():
        for index, mean_loss in enumerate(values):
            source_rows.append(
                {
                    "question_id": "q_test",
                    "architecture_letter": architecture,
                    "optimizer_template_letter": "A",
                    "learning_rate": [1e-3, 1e-2][index],
                    "is_original_diagonal": architecture == "A" and index == 0,
                    "mean_loss": mean_loss,
                    "std_loss": 0.1,
                }
            )

    result = _question_analysis(_question(), source_rows)

    assert result["num_architecture_crossover_pairs"] == 1
    assert result["architecture_crossovers"][0]["architectures"] == ["A", "B"]
    assert result["factorial_best"] == {
        "architecture_letter": "A",
        "optimizer_template_letter": "A",
        "learning_rate": 1e-3,
        "mean_loss": 1.0,
    }
    assert result["max_abs_log_interaction"] > 0.0
    assert math.isfinite(result["interaction_rmse_log"])


def test_question_analysis_requires_complete_grid() -> None:
    with pytest.raises(KeyError):
        _question_analysis(
            _question(),
            [
                {
                    "architecture_letter": "A",
                    "optimizer_template_letter": "A",
                    "learning_rate": 1e-3,
                    "is_original_diagonal": True,
                    "mean_loss": 1.0,
                    "std_loss": 0.1,
                },
                {
                    "architecture_letter": "B",
                    "optimizer_template_letter": "A",
                    "learning_rate": 1e-2,
                    "is_original_diagonal": False,
                    "mean_loss": 2.0,
                    "std_loss": 0.1,
                },
            ],
        )
