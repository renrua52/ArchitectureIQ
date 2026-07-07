from __future__ import annotations

import random

import pytest

from architecture_iq.families.univariate_regression.sampler import (
    sample_symbolic_expression,
    validate_expression,
)


def test_sampler_reproducible():
    a = sample_symbolic_expression(seed=7, max_depth=3)
    b = sample_symbolic_expression(seed=7, max_depth=3)
    assert a.expression == b.expression
    assert a.retry == b.retry


def test_sampler_expression_has_nonlinear():
    for seed in range(20):
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        assert sampled.tree.has_nonlinear()
        assert validate_expression(sampled.tree, (0.0, 1.0))


def test_sampler_not_trivial_constant():
    rng = random.Random(0)
    seen = set()
    for seed in range(30):
        sampled = sample_symbolic_expression(seed=seed)
        seen.add(sampled.expression)
    assert len(seen) > 10
