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
        "format_optimizer_nl",
        "format_loss_nl",
        "format_training_schedule",
        "format_dataset_protocol",
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


def test_ranking_protocol() -> None:
    text = pkg.format_ranking_protocol(n_seeds=10, base_seed=0, selection_metric="test_mse")
    assert "seeds" in text
    assert "0" in text and "9" in text
    assert "mean" in text
