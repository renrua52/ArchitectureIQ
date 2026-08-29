"""Structural invariants for the MLP model (v1.1 review fixes A1/A2/D4)."""

from __future__ import annotations

import random

import pytest
import torch

from architecture_iq.candidates.generator import (
    build_candidate_spec,
    sample_loss,
    sample_optimizer,
    trainable_parameter_count,
)
from architecture_iq.models.mlp import MLP
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type


def _spec(residual: bool, layer_norm: bool) -> dict:
    return {
        "type": "mlp",
        "depth": 4,
        "width": 32,
        "residual": residual,
        "layer_norm": [layer_norm] * 4,
        "activations": ["relu"] * 4,
        "input_dim": 8,
        "output_dim": 1,
    }


def _linears_and_acts(module: torch.nn.Module) -> list[str]:
    seq: list[str] = []
    for child in module.net:
        if isinstance(child, torch.nn.Linear):
            seq.append("L")
        elif isinstance(child, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.LeakyReLU)):
            seq.append("A")
        else:
            seq.append("B")
    return seq


def test_no_adjacent_linear_layers() -> None:
    # A1: the stem Linear must be followed by an activation so it cannot
    # collapse into the first block's Linear.
    for residual in (False, True):
        model = MLP(
            input_dim=8, depth=4, width=32, activations=["relu"] * 4,
            layer_norm=[True] * 4, residual=residual,
        )
        kinds = _linears_and_acts(model)
        assert kinds[0] == "L" and kinds[1] == "A", kinds
        # Top level: stem Linear + activation, 4 block containers, head Linear.
        assert kinds == ["L", "A", "B", "B", "B", "B", "L"]


def test_residual_branch_has_down_projection() -> None:
    # A2: standard pre-activation residual, y = x + W2*act(W1*norm(x)).
    model = MLP(
        input_dim=8, depth=6, width=32, activations=["relu"] * 6,
        layer_norm=[False] * 6, residual=True,
    )
    blocks = [m for m in model.net if isinstance(m, torch.nn.Module) and hasattr(m, "linear2")]
    assert len(blocks) == 6
    for block in blocks:
        assert block.linear2 is not None

    h = torch.randn(256, 8)
    norms = [h.norm().item()]
    for child in model.net:
        h = child(h)
        norms.append(h.norm().item())
    # With a down-projection the stream may shrink as well as grow; the old
    # projection-free branch grew monotonically.
    growth = [b - a for a, b in zip(norms, norms[1:])]
    assert any(g < 0 for g in growth), norms


def test_non_residual_block_has_no_projection() -> None:
    model = MLP(
        input_dim=8, depth=3, width=32, activations=["relu"] * 3,
        layer_norm=[True] * 3, residual=False,
    )
    for child in model.net:
        if isinstance(child, torch.nn.Module) and hasattr(child, "linear2"):
            assert child.linear2 is None


def test_rendered_model_py_matches_class_structure() -> None:
    ensure_registries()
    family = get_model_type("mlp")
    for residual in (False, True):
        spec = _spec(residual=residual, layer_norm=True)
        code = family.render_model_py(spec)
        module_globals: dict = {}
        exec(compile(code, "model.py", "exec"), module_globals)
        rendered = module_globals["Model"]()
        reference = family.build_module(spec)
        torch.manual_seed(0)
        with torch.no_grad():
            for p_rendered, p_reference in zip(
                sorted(rendered.parameters(), key=lambda p: p.shape),
                sorted(reference.parameters(), key=lambda p: p.shape),
            ):
                assert p_rendered.shape == p_reference.shape
        kinds = _linears_and_acts(rendered)
        assert kinds[0] == "L" and kinds[1] == "A"


def test_trainable_parameter_count_matches_module() -> None:
    ensure_registries()
    spec = _spec(residual=True, layer_norm=True)
    assert trainable_parameter_count(spec) == sum(
        p.numel() for p in get_model_type("mlp").build_module(spec).parameters()
    )


def test_new_specs_omit_leaky_relu_slope() -> None:
    # D4: the dead field is no longer written into fresh specs.
    ensure_registries()
    profile = load_profile("v1.1")
    spec = get_model_type("mlp").sample_spec(
        profile, random.Random(0), dataset_params={"input_dim": 8}
    )
    assert "leaky_relu_slope" not in spec


def test_double_regularization_guard_zeroes_weight_decay() -> None:
    # A6: a legacy lambda loss together with weight_decay must not reach GT.
    ensure_registries()
    profile = load_profile("v1.1")
    model = get_model_type("mlp").sample_spec(
        profile, random.Random(1), dataset_params={"input_dim": 8}
    )
    optimizer = sample_optimizer(profile, random.Random(2)) | {"weight_decay": 1e-3}
    loss = {"loss_id": "mse_l2", "lambda": 1e-2}
    spec = build_candidate_spec(
        profile,
        dataset_id="ds_test",
        family="univariate_regression",
        budget=4096,
        batch_size=64,
        model=model,
        optimizer=optimizer,
        loss=loss,
    )
    assert spec["optimizer"]["weight_decay"] == 0
    assert spec["loss"]["loss_id"] == "mse_l2"


def test_sample_loss_never_returns_regularized() -> None:
    ensure_registries()
    profile = load_profile("v1.1")
    rng = random.Random(3)
    for _ in range(30):
        loss = sample_loss(profile, "univariate_regression", rng)
        assert loss == {"loss_id": "mse"}


@pytest.mark.parametrize("residual", [False, True])
def test_forward_shapes(residual: bool) -> None:
    model = MLP(
        input_dim=8, depth=4, width=32, activations=["relu"] * 4,
        layer_norm=[True] * 4, residual=residual,
    )
    y = model(torch.randn(16, 8))
    assert y.shape == (16, 1)
