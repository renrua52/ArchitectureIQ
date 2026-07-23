from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "classification_calibration",
    ROOT / "tools" / "generate_kan_mlp_classification_calibration.py",
)
assert SPEC and SPEC.loader
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def test_classification_pairs_cover_dimensions_and_match_parameters() -> None:
    for input_dim in (4, 8, 16):
        pairs = CAL.pair_specs(input_dim)
        assert len(pairs) == 2
        for name, kan, mlp in pairs:
            assert name.startswith("kan_cls_")
            assert kan["output_dim"] == mlp["output_dim"] == 2
            assert kan["input_dim"] == mlp["input_dim"] == input_dim
            ratio = CAL.parameter_count(kan) / CAL.parameter_count(mlp)
            assert 0.95 <= ratio <= 1.05


def test_classification_pair_rejects_uncalibrated_dimension() -> None:
    with pytest.raises(ValueError, match="expects input_dim"):
        CAL.pair_specs(3)
