"""Univariate target-expression sampler.

The tree grammar, canonicalization and rendering all live in
`architecture_iq.families.symbolic_expr`; this module only samples raw trees and
applies the univariate acceptance criteria. `x` is scalar, so the single
coordinate renders as plain `x`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from architecture_iq.families.symbolic_expr import (
    CONSTANTS,
    ExprNode,
    NodeKind,
    canonicalize,
    context_1d,
    denominators_are_safe,
    eval_tree,
    is_const_only,
    render_infix,
    render_torch,
    sample_binary_kind,
    sample_constant_node,
    sample_unary_node,
    trees_equal,
    used_dimensions,
)

__all__ = [
    "CONSTANTS",
    "ExprNode",
    "NodeKind",
    "SampledExpression",
    "eval_node",
    "sample_symbolic_expression",
    "sample_tree",
    "to_infix",
    "to_torch_expr",
    "used_dimensions",
    "validate_expression",
]


def _var_name(_dim: int) -> str:
    return "x"


def to_infix(node: ExprNode) -> str:
    return render_infix(node, var=_var_name)


def to_torch_expr(node: ExprNode) -> str:
    return render_torch(node, var=_var_name)


# eval_node is the historical name; the shared implementation handles both the
# 1-D and the multivariate layouts.
eval_node = eval_tree

# Kept exported under their old private names: tests and the multivariate
# sampler both reach for them.
_is_const_only = is_const_only
_trees_equal = trees_equal


def _sample_leaf(rng: random.Random) -> ExprNode:
    if rng.random() < 0.55:
        return ExprNode(NodeKind.X)
    return sample_constant_node(rng)


def sample_tree(rng: random.Random, max_depth: int, depth: int = 0) -> ExprNode:
    if depth >= max_depth:
        return _sample_leaf(rng)
    roll = rng.random()
    if depth == 0 or roll < 0.45:
        return ExprNode(
            sample_binary_kind(rng),
            left=sample_tree(rng, max_depth, depth + 1),
            right=sample_tree(rng, max_depth, depth + 1),
        )
    if roll < 0.75:
        return sample_unary_node(rng, sample_tree(rng, max_depth, depth + 1))
    return _sample_leaf(rng)


@dataclass
class SampledExpression:
    tree: ExprNode
    expression: str
    torch_expression: str
    sampler_seed: int
    retry: int


def _affine_residual(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    design = np.stack([np.ones_like(xs), xs], axis=1)
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    return ys - design @ coef


def validate_expression(
    tree: ExprNode,
    domain: tuple[float, float],
    *,
    min_range: float = 0.4,
    max_abs: float = 5.0,
    near_singular_abs: float = 4.5,
    near_singular_frac: float = 0.95,
    grid_points: int = 128,
    min_nonlinear_fraction: float = 0.3,
) -> bool:
    xs = np.linspace(domain[0], domain[1], grid_points)
    ys = eval_tree(tree, xs)
    if not np.all(np.isfinite(ys)):
        return False
    # Numeric nonlinearity: an affine fit must leave a substantial residual. A
    # syntactic check is fooled by nonlinear nodes applied to constant subtrees
    # (abs(1.5), tanh(2*0.5), 2**3) -- canonicalization now folds those away,
    # but the numeric test is still the honest criterion.
    residual = _affine_residual(xs, ys)
    if float(np.max(np.abs(residual))) <= 1e-3:
        return False
    y_range = float(np.max(ys) - np.min(ys))
    if y_range < min_range:
        return False
    # The nonlinear component must carry a substantial share of the output
    # range, so expressions are neither constant- nor affine-dominated.
    residual_range = float(np.max(residual) - np.min(residual))
    if residual_range < min_nonlinear_fraction * y_range:
        return False
    if not denominators_are_safe(tree, context_1d(domain)):
        return False
    if float(np.max(np.abs(ys))) > max_abs:
        return False
    frac_ok = float(np.mean(np.abs(ys) <= near_singular_abs))
    return frac_ok >= near_singular_frac


def sample_symbolic_expression(
    seed: int,
    max_depth: int = 3,
    max_retries: int = 200,
    domain: tuple[float, float] = (0.0, 1.0),
) -> SampledExpression:
    rng = random.Random(seed)
    ctx = context_1d(domain)
    for retry in range(max_retries):
        tree = canonicalize(sample_tree(rng, max_depth), ctx)
        if tree is None:
            continue
        if not validate_expression(tree, domain):
            continue
        return SampledExpression(
            tree=tree,
            expression=to_infix(tree),
            torch_expression=to_torch_expr(tree),
            sampler_seed=seed,
            retry=retry,
        )
    raise RuntimeError(
        f"Failed to sample valid expression after {max_retries} retries (seed={seed})"
    )
