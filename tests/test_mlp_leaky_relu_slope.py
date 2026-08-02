from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch import nn

from architecture_iq.interactive import assemble_model_spec
from architecture_iq.models.mlp import MlpModelFamily
from architecture_iq.profile import load_profile
from architecture_iq.prompts.formatters import format_mlp_nl
from architecture_iq.runtime.loader import load_module_from_file


def _model_spec(slope: float | None = None) -> dict:
    spec = {
        "type": "mlp",
        "depth": 2,
        "width": 8,
        "residual": False,
        "layer_norm": [False, False],
        "activations": ["leaky_relu", "relu"],
        "input_dim": 3,
    }
    if slope is not None:
        spec["leaky_relu_slope"] = slope
    return spec


def _leaky_relu(module: nn.Module) -> nn.LeakyReLU:
    return next(layer for layer in module.modules() if isinstance(layer, nn.LeakyReLU))


def test_sample_spec_records_profile_leaky_relu_slope() -> None:
    profile = load_profile("v1")
    spec = MlpModelFamily().sample_spec(profile, random.Random(0))
    assert spec["leaky_relu_slope"] == pytest.approx(0.01)


def test_interactive_mlp_spec_records_profile_leaky_relu_slope() -> None:
    profile = load_profile("v1")
    spec = assemble_model_spec(
        profile,
        random.Random(0),
        depth=2,
        width=8,
        residual=False,
        activations=["relu", "gelu"],
        layer_norm=[False, False],
    )
    assert spec["leaky_relu_slope"] == pytest.approx(0.01)


def test_build_module_uses_explicit_leaky_relu_slope() -> None:
    module = MlpModelFamily().build_module(_model_spec(0.01))
    assert _leaky_relu(module).negative_slope == pytest.approx(0.01)


def test_legacy_spec_without_slope_keeps_legacy_build_and_render_behavior() -> None:
    family = MlpModelFamily()
    spec = _model_spec()
    module = family.build_module(spec)
    rendered = family.render_model_py(spec)
    assert _leaky_relu(module).negative_slope == pytest.approx(0.1)
    assert "nn.LeakyReLU(0.1)" in rendered


def test_rendered_model_uses_slope_and_imports_and_executes(tmp_path: Path) -> None:
    family = MlpModelFamily()
    spec = _model_spec(0.01)
    rendered = family.render_model_py(spec)
    assert "nn.LeakyReLU(0.01)" in rendered
    assert "nn.LeakyReLU(0.1)" not in rendered

    model_path = tmp_path / "model.py"
    model_path.write_text(rendered, encoding="utf-8")
    module = load_module_from_file(model_path, "rendered_mlp_model")
    model = module.Model()
    output = model(torch.randn(4, 3))
    assert output.shape == (4, 1)
    assert _leaky_relu(model).negative_slope == pytest.approx(0.01)


def test_prompt_formatter_mentions_slope_only_when_used() -> None:
    leaky_text = format_mlp_nl({**_model_spec(0.01), "activations": ["leaky_relu", "relu"]})
    assert "LeakyReLU negative slope: 0.01" in leaky_text

    non_leaky_text = format_mlp_nl({**_model_spec(0.01), "activations": ["relu", "gelu"]})
    assert "negative slope" not in non_leaky_text