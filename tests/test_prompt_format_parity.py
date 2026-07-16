"""Parity between package formatters and inspector mirrors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from architecture_iq.prompts import formatters as pkg

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))
import prompt_format as insp  # noqa: E402


@pytest.mark.parametrize(
    "name",
    [
        "format_mlp_nl",
        "format_kan_nl",
        "format_optimizer_nl",
        "format_loss_nl",
        "format_training_schedule",
        "format_dataset_protocol",
        "format_synthetic_tabular_classification_rule",
        "format_ranking_protocol",
    ],
)
def test_formatter_parity(name: str) -> None:
    assert getattr(pkg, name) is not None
    assert getattr(insp, name) is not None


def test_mlp_nl_parity_output() -> None:
    model = {
        "depth": 2,
        "width": 64,
        "residual": False,
        "layer_norm": [True, True],
        "activations": ["leaky_relu", "gelu"],
    }
    assert pkg.format_mlp_nl(model) == insp.format_mlp_nl(model)


def test_kan_nl_parity_output() -> None:
    model = {
        "type": "kan",
        "input_dim": 2,
        "output_dim": 1,
        "depth": 2,
        "width": 8,
        "grid_size": 5,
        "spline_order": 3,
        "grid_range": [-1.0, 1.0],
        "base_activation": "silu",
    }
    assert pkg.format_kan_nl(model) == insp.format_kan_nl(model)


@pytest.mark.parametrize(
    "rule_family, active_features, interaction_pairs, weights, breakpoint",
    [
        ("smooth_additive", [0, 2], [], [-1.0, 0.75], 0.0),
        ("sparse_interaction", [0, 2, 3], [[0, 2], [2, 3]], [-1.0, 0.75], 0.0),
        ("piecewise_boundary", [0, 2], [], [-1.0, 0.75, 0.5], -0.25),
    ],
)
def test_classification_rule_card_parity(
    rule_family: str,
    active_features: list[int],
    interaction_pairs: list[list[int]],
    weights: list[float],
    breakpoint: float,
) -> None:
    params = {
        "input_dim": 4,
        "rule_family": rule_family,
        "active_features": active_features,
        "interaction_pairs": interaction_pairs,
        "rule_weights": weights,
        "piecewise_breakpoint": breakpoint,
        "noise_std": 0.1,
        "decision_threshold": 0.125,
        "point_sampling": {"seed": 11},
        "calibration": {"seed": 22, "size": 4096, "target_positive_rate": 0.5},
    }
    text = pkg.format_synthetic_tabular_classification_rule(params)
    assert text == insp.format_synthetic_tabular_classification_rule(params)
    assert "Latent score" in text
    assert "Label rule" in text
    assert "Bayes decision boundary" in text
    assert "def " not in text


def test_ranking_protocol() -> None:
    text = pkg.format_ranking_protocol(n_seeds=10, base_seed=0, selection_metric="test_mse")
    assert "seeds" in text
    assert "0" in text and "9" in text
    assert "mean" in text
