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
        "format_gru_lm_nl",
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
        "layer_norm": [True, False],
        "activation": "gelu",
    }
    assert pkg.format_mlp_nl(model) == insp.format_mlp_nl(model)


def test_gru_lm_nl_parity_output() -> None:
    model = {
        "type": "gru_lm",
        "vocab_size": 32,
        "context_length": 16,
        "d_model": 64,
        "num_layers": 2,
    }
    assert pkg.format_gru_lm_nl(model) == insp.format_gru_lm_nl(model)
    assert pkg.format_model_nl(model) == insp.format_model_nl(model)


def test_gru_lm_residual_nl_parity_output() -> None:
    model = {
        "type": "gru_lm",
        "vocab_size": 32,
        "context_length": 16,
        "d_model": 64,
        "num_layers": 2,
        "layer_residual": True,
    }
    expected = "Layer residual connections: enabled; after each GRU layer, h = h + GRU_layer(h)."
    assert pkg.format_gru_lm_nl(model) == insp.format_gru_lm_nl(model)
    assert expected in pkg.format_gru_lm_nl(model)


@pytest.mark.parametrize(
    "rule_family, active_features, interaction_pairs, weights, breakpoint",
    [
        ("smooth_additive", [0, 2], [], [-1.0, 0.75], 0.0),
        ("sparse_interaction", [0, 2, 3], [[0, 2], [2, 3]], [-1.0, 0.75], 0.0),
        ("xor", [0, 2], [[0, 2]], [-1.0], 0.0),
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
    if rule_family == "xor":
        assert "s(x) = -x_0·x_2" in text
        assert "nominal only" in text
        assert "need not follow" in text


def test_spiral_rule_card_parity() -> None:
    params = {
        "input_dim": 2,
        "rule_family": "spiral",
        "active_features": [0, 1],
        "interaction_pairs": [],
        "rule_weights": [1.0],
        "piecewise_breakpoint": 0.0,
        "spiral_turns": 2.0,
        "noise_std": 0.05,
        "decision_threshold": 0.0,
        "point_sampling": {"distribution": "two_spirals", "seed": 11, "turns": 2.0},
        "calibration": {"seed": 22, "size": 0, "target_positive_rate": 0.5},
    }
    text = pkg.format_synthetic_tabular_classification_rule(params)
    assert text == insp.format_synthetic_tabular_classification_rule(params)
    assert "two-spirals" in text
    # The arm's parameter range is printed as a multiple of pi, not as 13.0665.
    assert "[0.5, 0.5 + 4π]" in text
    assert "Label rule" in text
    assert "Bayes decision boundary" in text
    assert "phase = π" in text
    assert "def " not in text


def test_exact_xor_rule_card_parity() -> None:
    """A zero cut-off with no label noise makes the quadrant rule exact.

    The two hedging lines the calibrated-threshold version needs ("nominal
    only", "need not follow") must then be gone, because they would be telling
    the reader to distrust a rule that now holds exactly.
    """
    params = {
        "input_dim": 4,
        "rule_family": "xor",
        "active_features": [0, 2],
        "interaction_pairs": [[0, 2]],
        "rule_weights": [-1.0],
        "piecewise_breakpoint": 0.0,
        "decision_threshold": 0.0,
        "point_sampling": {"seed": 11},
        "calibration": {
            "seed": 22,
            "size": 4096,
            "target_positive_rate": 0.5,
            "realized_positive_rate": 0.4993,
        },
    }
    text = pkg.format_synthetic_tabular_classification_rule(params)
    assert text == insp.format_synthetic_tabular_classification_rule(params)
    assert "class 1 exactly when its two active coordinates have opposite signs" in text
    assert "nominal only" not in text
    assert "need not follow" not in text
    # The stated balance is the one the cut-off achieves, not the target.
    assert "labels 49.9% of them class 1" in text
    assert "def " not in text


def test_classification_dataset_protocol_uses_family_dispatch_without_rule_family() -> None:
    params = {
        "input_dim": 2,
        "train_size": 256,
        "test_size": 256,
    }
    package_text = pkg.format_dataset_protocol(
        params,
        family="synthetic_tabular_classification",
        device="cpu",
    )
    inspector_text = insp.format_dataset_protocol(
        params,
        family="synthetic_tabular_classification",
        device="cpu",
    )

    assert package_text == inspector_text
    assert "binary classification" in package_text
    assert "test cross-entropy" in package_text


def test_ranking_protocol() -> None:
    text = pkg.format_ranking_protocol(
        n_seeds=10,
        base_seed=0,
        selection_metric="test_mse",
    )
    assert "seeds" in text
    assert "0" in text and "9" in text
    assert "mean" in text
