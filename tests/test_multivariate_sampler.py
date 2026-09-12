from __future__ import annotations

import re

import numpy as np
import torch

from architecture_iq.families.symbolic_expr import (
    CONST_ABS_MAX,
    CONST_QUANTUM,
    NodeKind,
    iter_nodes,
)
from architecture_iq.families.multivariate_regression.sampler import (
    eval_node_mv,
    sample_symbolic_expression,
    used_dimensions,
)


def test_multivariate_uses_all_input_dims() -> None:
    for input_dim in (2, 3, 5, 8):
        for seed in range(20):
            sampled = sample_symbolic_expression(seed=seed, input_dim=input_dim, max_depth=3)
            assert used_dimensions(sampled.tree) == set(range(input_dim))
            for dim in range(input_dim):
                assert f"x{dim}" in sampled.expression


def test_multivariate_has_interaction_when_n_gt_2() -> None:
    sampled = sample_symbolic_expression(seed=0, input_dim=4, max_depth=3)
    # Several additive terms, counting both separators: canonicalization emits a
    # negative term as `a - b`, so counting only "+" undercounts the sum.
    assert len(re.findall(r" [+-] ", sampled.expression)) >= 3


def test_multivariate_reproducible() -> None:
    a = sample_symbolic_expression(seed=11, input_dim=4, max_depth=3)
    b = sample_symbolic_expression(seed=11, input_dim=4, max_depth=3)
    assert a.expression == b.expression


def test_multivariate_renders_pi_symbolically() -> None:
    saw_trig = False
    for input_dim in (2, 4, 8):
        for seed in range(15):
            sampled = sample_symbolic_expression(seed=seed, input_dim=input_dim, max_depth=3)
            assert "6.283185307179586" not in sampled.expression
            assert "6.283185307179586" not in sampled.torch_expression
            if "sin(" in sampled.expression or "cos(" in sampled.expression:
                saw_trig = True
                assert "pi" in sampled.expression
                assert "torch.pi" in sampled.torch_expression
    assert saw_trig, "no trig sampled; the assertions above never ran"


def test_multivariate_expressions_are_clean() -> None:
    for input_dim in (2, 4, 8):
        for seed in range(15):
            sampled = sample_symbolic_expression(seed=seed, input_dim=input_dim, max_depth=3)
            expression = sampled.expression
            assert not re.search(r"\d+\.\d{4,}", expression), expression
            for pattern in ("+ -", "- -", "* -", "/ -", "0 - ", "--"):
                assert pattern not in expression, (pattern, expression)
            assert "clamp" not in sampled.torch_expression
            for node in iter_nodes(sampled.tree):
                if node.kind is NodeKind.CONST:
                    value = float(node.value)
                    assert abs(value) <= CONST_ABS_MAX, expression
                    assert (
                        abs(value / CONST_QUANTUM - round(value / CONST_QUANTUM)) < 1e-9
                    ), expression
                if node.kind is NodeKind.POW:
                    # ((x)**2)**3 collapses to x**6.
                    assert node.left.kind is not NodeKind.POW, expression


def test_multivariate_torch_form_matches_eval() -> None:
    """synthesize.py runs torch_expression; validation used eval_node_mv.

    The torch form is evaluated exactly as the generated `target(x)` does: one
    `[N, D]` tensor named `x`, sliced per coordinate. Binding `x0`..`xN` here
    instead would let a renderer that emits prose names pass a test that the
    generated file then fails with NameError.
    """
    for input_dim in (2, 4, 8):
        for seed in range(10):
            sampled = sample_symbolic_expression(seed=seed, input_dim=input_dim, max_depth=3)
            assert f"x{input_dim - 1}" not in sampled.torch_expression
            rng = np.random.default_rng(seed)
            points = rng.uniform(0.0, 1.0, size=(64, input_dim))
            reference = eval_node_mv(sampled.tree, points)
            executed = eval(
                sampled.torch_expression,
                {"torch": torch},
                {"x": torch.tensor(points, dtype=torch.float32)},
            )
            np.testing.assert_allclose(
                reference, executed.detach().numpy(), atol=2e-4, rtol=2e-4
            )
