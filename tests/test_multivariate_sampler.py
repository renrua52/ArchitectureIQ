from __future__ import annotations

import re

from architecture_iq.families.multivariate_regression.sampler import (
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
    # At least two distinct variables appear outside isolated xN tokens via + terms
    assert sampled.expression.count("+") >= 3


def test_multivariate_reproducible() -> None:
    a = sample_symbolic_expression(seed=11, input_dim=4, max_depth=3)
    b = sample_symbolic_expression(seed=11, input_dim=4, max_depth=3)
    assert a.expression == b.expression
