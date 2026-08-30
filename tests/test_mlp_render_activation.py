"""Only the network's own activation appears in the generated model.py."""

from __future__ import annotations

import pytest
import torch

from architecture_iq.models.mlp import MlpModelFamily, _activation_module
from architecture_iq.runtime.loader import load_module_from_file

# Every activation the profile pool can draw, paired with the constructors that
# must NOT appear in its rendered file.
UNUSED = {
    "relu": ("nn.LeakyReLU", "nn.GELU", "nn.SiLU"),
    "leaky_relu": ("nn.ReLU()", "nn.GELU", "nn.SiLU"),
    "gelu": ("nn.ReLU()", "nn.LeakyReLU", "nn.SiLU"),
    "silu": ("nn.ReLU()", "nn.LeakyReLU", "nn.GELU"),
}


def _spec(activation: str) -> dict:
    spec = {
        "type": "mlp",
        "depth": 2,
        "width": 8,
        "residual": False,
        "activation": activation,
        "layer_norm": [False, True],
        "input_dim": 3,
    }
    if activation == "leaky_relu":
        spec["leaky_relu_slope"] = 0.01
    return spec


@pytest.mark.parametrize("activation", sorted(UNUSED))
def test_rendered_model_shows_only_the_activation_it_uses(activation: str, tmp_path) -> None:
    """A gelu network used to display `nn.LeakyReLU(0.1)`: a slope that applies to
    nothing, and a constant contradicting the profile's leaky_relu_slope."""
    rendered = MlpModelFamily().render_model_py(_spec(activation))
    for absent in UNUSED[activation]:
        assert absent not in rendered, (activation, absent)

    # The prompt excerpts this file, and ground truth imports it, so a narrowed
    # mapping still has to import, build, and run.
    path = tmp_path / "model.py"
    path.write_text(rendered, encoding="utf-8")
    module = load_module_from_file(path, f"rendered_mlp_{activation}")
    assert module.Model()(torch.randn(4, 3)).shape == (4, 1)


@pytest.mark.parametrize("activation", sorted(UNUSED))
def test_rendered_activation_matches_the_in_process_module(activation: str) -> None:
    """render_model_py must construct what _activation_module constructs."""
    spec = _spec(activation)
    expected = _activation_module(activation, spec.get("leaky_relu_slope", 0.1))
    rendered = MlpModelFamily().render_model_py(spec)
    assert f"nn.{type(expected).__name__}(" in rendered
    if activation == "leaky_relu":
        assert f"nn.LeakyReLU({expected.negative_slope!r})" in rendered
