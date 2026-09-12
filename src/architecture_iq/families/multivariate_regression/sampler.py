"""Symbolic expression sampler for R^n -> R targets.

The tree grammar, canonicalization and rendering live in
`architecture_iq.families.symbolic_expr`; this module contributes the
multivariate tree shape (one nonlinear term per coordinate plus a few
cross-variable interactions) and the multivariate acceptance criteria. Each
coordinate renders as `x0`, `x1`, ....
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from architecture_iq.families.symbolic_expr import (
    CONSTANTS,
    ExprNode,
    NodeKind,
    canonicalize,
    context_nd,
    denominators_are_safe,
    eval_tree,
    render_infix,
    render_torch,
    sample_binary_kind,
    sample_constant_node,
    sample_unary_node,
    used_dimensions,
)

__all__ = [
    "CONSTANTS",
    "ExprNode",
    "NodeKind",
    "SampledExpression",
    "eval_node_mv",
    "sample_symbolic_expression",
    "sample_tree_mv",
    "to_infix_mv",
    "to_torch_expr_mv",
    "used_dimensions",
    "validate_expression_mv",
]


def _var_name(dim: int) -> str:
    """Prose name of a coordinate: `x0`, `x1`, ..."""
    return f"x{dim}"


def _torch_var_name(dim: int) -> str:
    """Executable name of a coordinate.

    The generated `synthesize.py` defines `target(x)` over the whole `[N, D]`
    batch, so the torch form has to slice rather than reference `x0` names that
    exist nowhere at runtime. Indexing binds tighter than any operator the
    renderer emits, so a slice is safe as an atom.
    """
    return f"x[:, {dim}]"


def to_infix_mv(node: ExprNode) -> str:
    return render_infix(node, var=_var_name)


def to_torch_expr_mv(node: ExprNode) -> str:
    return render_torch(node, var=_torch_var_name)


# eval_node_mv is the historical name; the shared implementation dispatches on
# the input rank, so `[N, D]` rows and 1-D samples share one code path.
eval_node_mv = eval_tree


def _x_node(dim: int) -> ExprNode:
    return ExprNode(NodeKind.X, value=float(dim))


def _sample_dim_term(rng: random.Random, max_depth: int, dim: int) -> ExprNode:
    """Nonlinear subtree that always depends on x_dim."""
    x = _x_node(dim)
    if max_depth <= 1:
        return sample_unary_node(rng, x)

    roll = rng.random()
    if roll < 0.45:
        inner = x
        if rng.random() < 0.4:
            inner = _sample_dim_term(rng, max_depth - 1, dim)
        return sample_unary_node(rng, inner)
    if roll < 0.8:
        kind = sample_binary_kind(rng)
        left = sample_unary_node(rng, x) if rng.random() < 0.65 else x
        return ExprNode(kind, left=left, right=sample_constant_node(rng))
    inner = _sample_dim_term(rng, max_depth - 1, dim)
    return sample_unary_node(rng, inner)


def _sample_interaction_term(
    rng: random.Random,
    max_depth: int,
    i: int,
    j: int,
) -> ExprNode:
    xi = _x_node(i)
    xj = _x_node(j)
    if rng.random() < 0.55:
        if rng.random() < 0.5:
            xi = sample_unary_node(rng, xi)
        if rng.random() < 0.5:
            xj = sample_unary_node(rng, xj)
        kind = NodeKind.MUL if rng.random() < 0.7 else sample_binary_kind(rng)
        return ExprNode(kind, left=xi, right=xj)
    if max_depth > 1 and rng.random() < 0.5:
        return ExprNode(
            NodeKind.ADD,
            left=_sample_dim_term(rng, max_depth - 1, i),
            right=_sample_dim_term(rng, max_depth - 1, j),
        )
    return ExprNode(
        NodeKind.MUL,
        left=ExprNode(NodeKind.SIN2PI, left=xi),
        right=ExprNode(NodeKind.COS2PI, left=xj),
    )


def _fold_add(nodes: list[ExprNode]) -> ExprNode:
    acc = nodes[0]
    for node in nodes[1:]:
        acc = ExprNode(NodeKind.ADD, left=acc, right=node)
    return acc


def sample_tree_mv(rng: random.Random, max_depth: int, input_dim: int) -> ExprNode:
    """Build f(x) as a sum of per-coordinate nonlinear terms plus interactions."""
    per_dim_depth = max(2, max_depth)
    terms = [_sample_dim_term(rng, per_dim_depth, dim) for dim in range(input_dim)]

    if input_dim >= 2:
        n_interactions = 1 if input_dim == 2 else rng.randint(1, min(2, input_dim - 1))
        dims = list(range(input_dim))
        rng.shuffle(dims)
        for k in range(n_interactions):
            i = dims[k % input_dim]
            j = dims[(k + 1) % input_dim]
            if i == j:
                j = (j + 1) % input_dim
            terms.append(_sample_interaction_term(rng, per_dim_depth, i, j))

    return _fold_add(terms)


@dataclass
class SampledExpression:
    tree: ExprNode
    expression: str
    torch_expression: str
    sampler_seed: int
    retry: int


def validate_expression_mv(
    tree: ExprNode,
    input_dim: int,
    domain: tuple[float, float],
    *,
    min_range: float = 0.4,
    max_abs: float = 5.0,
    near_singular_abs: float = 4.5,
    near_singular_frac: float = 0.95,
    grid_points: int = 64,
    min_dim_fraction: float = 0.05,
) -> bool:
    dims = used_dimensions(tree)
    if dims != set(range(input_dim)):
        return False
    rng = np.random.default_rng(0)
    xs = rng.uniform(domain[0], domain[1], size=(grid_points, input_dim))
    ys = eval_tree(tree, xs)
    if not np.all(np.isfinite(ys)):
        return False
    y_range = float(np.max(ys) - np.min(ys))
    scaled_min_range = min_range * math.sqrt(input_dim)
    if y_range < scaled_min_range:
        return False
    if not denominators_are_safe(tree, context_nd(domain, input_dim)):
        return False
    # Numeric dead-dimension check: freezing one coordinate at the domain
    # midpoint must change the output by a visible share of the range. A
    # syntactic check is fooled by nonlinear nodes over constant subtrees.
    mid = (domain[0] + domain[1]) / 2.0
    for dim in range(input_dim):
        frozen = xs.copy()
        frozen[:, dim] = mid
        ys_frozen = eval_tree(tree, frozen)
        contribution = float(np.std(ys - ys_frozen))
        if contribution < min_dim_fraction * max(y_range, 1e-9):
            return False
    if float(np.max(np.abs(ys))) > max_abs * math.sqrt(input_dim):
        return False
    frac_ok = float(np.mean(np.abs(ys) <= near_singular_abs * math.sqrt(input_dim)))
    return frac_ok >= near_singular_frac


def sample_symbolic_expression(
    seed: int,
    *,
    input_dim: int,
    max_depth: int = 3,
    max_retries: int = 200,
    domain: tuple[float, float] = (0.0, 1.0),
) -> SampledExpression:
    rng = random.Random(seed)
    ctx = context_nd(domain, input_dim)
    for retry in range(max_retries):
        tree = canonicalize(sample_tree_mv(rng, max_depth, input_dim), ctx)
        if tree is None:
            continue
        # Canonicalization can merge or cancel terms, so the every-coordinate
        # requirement is re-checked on the canonical tree rather than the raw one.
        if not validate_expression_mv(tree, input_dim, domain):
            continue
        return SampledExpression(
            tree=tree,
            expression=to_infix_mv(tree),
            torch_expression=to_torch_expr_mv(tree),
            sampler_seed=seed,
            retry=retry,
        )
    raise RuntimeError(
        f"Failed to sample valid multivariate expression after {max_retries} retries "
        f"(seed={seed}, input_dim={input_dim})"
    )
