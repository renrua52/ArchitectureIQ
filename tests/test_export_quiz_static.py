import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_quiz_static", ROOT / "tools" / "export_quiz_static.py"
)
assert SPEC and SPEC.loader
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


def test_classification_export_uses_a_smoothed_compact_projection() -> None:
    train = {
        "x": torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]]),
        "y": torch.tensor([0, 0, 1, 1]),
    }
    test = {
        "x": torch.tensor([[0.15, 0.15], [0.85, 0.85]]),
        "y": torch.tensor([0, 1]),
    }

    plot = EXPORTER._classification_plot(
        train, test, {"active_features": [0, 1], "rule_family": "smooth_additive"}
    )

    probability = np.asarray(plot["probability"], dtype=float)
    finite = probability[np.isfinite(probability)]
    assert len(plot["xEdges"]) == EXPORTER.CLASSIFICATION_BINS + 1
    assert len(plot["yEdges"]) == EXPORTER.CLASSIFICATION_BINS + 1
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert len(plot["train"]) <= 2 * EXPORTER.CLASSIFICATION_TRAIN_POINTS_PER_CLASS
    assert len(plot["test"]) <= 2 * EXPORTER.CLASSIFICATION_TEST_POINTS_PER_CLASS
    assert "blue low, red high" in plot["legend"]


def test_classification_probability_grid_can_keep_empty_bins_unknown() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    y = np.asarray([0, 1])

    _, _, probability = EXPORTER._classification_probability_grid(
        x, y, 0, 1, bins=3, prior_strength=0.0
    )

    assert np.any(np.isnan(probability))
