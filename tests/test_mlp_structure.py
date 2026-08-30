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
from architecture_iq.models.mlp import MLP, MLPBlock
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type


def _spec(residual: bool, layer_norm: bool) -> dict:
    return {
        "type": "mlp",
        "depth": 4,
        "width": 32,
        "residual": residual,
        "layer_norm": [layer_norm] * 4,
        "activation": "relu",
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
            input_dim=8, depth=4, width=32, activation="relu",
            layer_norm=[True] * 4, residual=residual,
        )
        kinds = _linears_and_acts(model)
        assert kinds[0] == "L" and kinds[1] == "A", kinds
        # Top level: stem Linear + activation, 4 block containers, head Linear.
        assert kinds == ["L", "A", "B", "B", "B", "B", "L"]


def _linear_count(module: torch.nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))


def test_depth_counts_hidden_linear_layers() -> None:
    # `depth` is the number of hidden Linear layers; with the input projection
    # and the output head that is depth + 2 nn.Linear modules -- identical for
    # residual and non-residual, so the two never differ on parameter count.
    for depth in (1, 2, 3, 6):
        built = [
            MLP(
                input_dim=8, depth=depth, width=32, activation="relu",
                layer_norm=[True] * depth, residual=residual,
            )
            for residual in (False, True)
        ]
        for model in built:
            assert _linear_count(model) == depth + 2
        params = [sum(p.numel() for p in m.parameters()) for m in built]
        assert params[0] == params[1], (depth, params)


def test_residual_adds_skip_before_the_activation() -> None:
    # act(x + Linear(norm(x))): the pre-activation sum lets the branch pull the
    # stream down. The old act(Linear(norm(x))) + x form could only add
    # non-negative values, so the residual stream grew monotonically.
    torch.manual_seed(0)
    model = MLP(
        input_dim=8, depth=6, width=32, activation="relu",
        layer_norm=[False] * 6, residual=True,
    )
    h = torch.randn(256, 8)
    norms = [h.norm().item()]
    for child in model.net:
        h = child(h)
        norms.append(h.norm().item())
    growth = [b - a for a, b in zip(norms, norms[1:])]
    assert any(g < 0 for g in growth), norms


def test_residual_block_matches_formula() -> None:
    torch.manual_seed(0)
    block = MLPBlock(width=16, activation="relu", use_layer_norm=True, use_residual=True)
    x = torch.randn(32, 16)
    expected = torch.relu(x + block.linear(block.norm(x)))
    assert torch.allclose(block(x), expected)


def test_non_residual_block_matches_formula() -> None:
    torch.manual_seed(0)
    block = MLPBlock(width=16, activation="relu", use_layer_norm=True, use_residual=False)
    x = torch.randn(32, 16)
    expected = torch.relu(block.linear(block.norm(x)))
    assert torch.allclose(block(x), expected)


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


def test_rendered_model_py_writes_activation_and_skip_in_plain_text() -> None:
    # The generated file is what the prompt shows and what GT executes, so the
    # two network-wide choices are written out literally: the activation as its
    # constructor call, the skip as the line it is. No module constant, no
    # name -> module mapping, no runtime flag standing in for a fixed value.
    ensure_registries()
    family = get_model_type("mlp")
    for activation, ctor in (
        ("relu", "nn.ReLU()"),
        ("gelu", "nn.GELU()"),
        ("silu", "nn.SiLU()"),
    ):
        for residual in (False, True):
            spec = _spec(residual=residual, layer_norm=True) | {"activation": activation}
            code = family.render_model_py(spec)
            assert f"self.act = {ctor}" in code, code
            assert f"            {ctor},\n" in code, code
            for banned in ("ACTIVATION", "USE_RESIDUAL", "def _activation", "mapping["):
                assert banned not in code, (banned, code)
            assert ("        h = h + x\n" in code) is residual, code
            assert "self.use_residual" not in code
            # layer_norm is the one axis that really does vary per layer, so it
            # stays an argument -- one call per block, with its flag inline.
            assert code.count("MLPBlock(width=32, use_layer_norm=True)") == spec["depth"]


def test_rendered_leaky_relu_carries_its_slope_and_nothing_else() -> None:
    ensure_registries()
    spec = _spec(residual=True, layer_norm=False) | {
        "activation": "leaky_relu",
        "leaky_relu_slope": 0.01,
    }
    code = get_model_type("mlp").render_model_py(spec)
    assert "self.act = nn.LeakyReLU(0.01)" in code
    assert "nn.ReLU()" not in code and "nn.GELU()" not in code


def test_rendered_model_py_forward_is_bit_identical_to_the_reference() -> None:
    # Plain text must not mean "different network": the rendered file and the
    # in-process module have to build the same parameters in the same order and
    # compute the same forward pass from the same seed.
    ensure_registries()
    family = get_model_type("mlp")
    for residual in (False, True):
        for layer_norm in (False, True):
            spec = _spec(residual=residual, layer_norm=layer_norm)
            code = family.render_model_py(spec)
            module_globals: dict = {}
            exec(compile(code, "model.py", "exec"), module_globals)
            torch.manual_seed(0)
            rendered = module_globals["Model"]()
            torch.manual_seed(0)
            reference = family.build_module(spec)
            x = torch.randn(16, 8)
            with torch.no_grad():
                assert torch.equal(rendered(x), reference(x)), (residual, layer_norm)
            assert sum(p.numel() for p in rendered.parameters()) == sum(
                p.numel() for p in reference.parameters()
            )


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
        input_dim=8, depth=4, width=32, activation="relu",
        layer_norm=[True] * 4, residual=residual,
    )
    y = model(torch.randn(16, 8))
    assert y.shape == (16, 1)
