"""Symbolic expression sampler for R^n -> R targets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from architecture_iq.families.univariate_regression.sampler import (
    CONSTANTS,
    ExprNode,
    NodeKind,
    to_torch_expr,
)


def used_dimensions(node: ExprNode) -> set[int]:
    if node.kind == NodeKind.X:
        return {int(node.value or 0)}
    dims: set[int] = set()
    if node.left is not None:
        dims |= used_dimensions(node.left)
    if node.right is not None:
        dims |= used_dimensions(node.right)
    return dims


def _sample_unary(rng: random.Random) -> NodeKind:
    return rng.choice(
        [
            NodeKind.SIN2PI,
            NodeKind.COS2PI,
            NodeKind.TANH2,
            NodeKind.ABS,
            NodeKind.SQUARE,
            NodeKind.CUBE,
        ]
    )


def _sample_binary(rng: random.Random) -> NodeKind:
    return rng.choice([NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV])


def _x_node(dim: int) -> ExprNode:
    return ExprNode(NodeKind.X, value=float(dim))


def _const_node(rng: random.Random) -> ExprNode:
    return ExprNode(NodeKind.CONST, value=rng.choice(CONSTANTS))


def _sample_dim_term(rng: random.Random, max_depth: int, dim: int) -> ExprNode:
    """Nonlinear subtree that always depends on x_dim."""
    x = _x_node(dim)
    if max_depth <= 1:
        return ExprNode(_sample_unary(rng), left=x)

    roll = rng.random()
    if roll < 0.45:
        inner = x
        if rng.random() < 0.4:
            inner = _sample_dim_term(rng, max_depth - 1, dim)
        return ExprNode(_sample_unary(rng), left=inner)
    if roll < 0.8:
        kind = _sample_binary(rng)
        left = ExprNode(_sample_unary(rng), left=x) if rng.random() < 0.65 else x
        return ExprNode(kind, left=left, right=_const_node(rng))
    inner = _sample_dim_term(rng, max_depth - 1, dim)
    return ExprNode(_sample_unary(rng), left=inner)


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
            xi = ExprNode(_sample_unary(rng), left=xi)
        if rng.random() < 0.5:
            xj = ExprNode(_sample_unary(rng), left=xj)
        kind = NodeKind.MUL if rng.random() < 0.7 else _sample_binary(rng)
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
    if len(nodes) == 1:
        return nodes[0]
    acc = nodes[0]
    for node in nodes[1:]:
        acc = ExprNode(NodeKind.ADD, left=acc, right=node)
    return acc


def sample_tree_mv(rng: random.Random, max_depth: int, input_dim: int) -> ExprNode:
    """Build f(x) as sum of per-coordinate nonlinear terms plus cross-variable interactions."""
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


def eval_node_mv(node: ExprNode, x: np.ndarray) -> np.ndarray:
    if node.kind == NodeKind.X:
        dim = int(node.value or 0)
        return x[:, dim]
    if node.kind == NodeKind.CONST:
        return np.full(x.shape[0], float(node.value), dtype=np.float64)
    left = eval_node_mv(node.left, x) if node.left is not None else None
    if node.kind in {
        NodeKind.SIN2PI,
        NodeKind.COS2PI,
        NodeKind.TANH2,
        NodeKind.ABS,
        NodeKind.SQUARE,
        NodeKind.CUBE,
    }:
        assert left is not None
        if node.kind == NodeKind.SIN2PI:
            return np.sin(2 * math.pi * left)
        if node.kind == NodeKind.COS2PI:
            return np.cos(2 * math.pi * left)
        if node.kind == NodeKind.TANH2:
            return np.tanh(2 * left)
        if node.kind == NodeKind.ABS:
            return np.abs(left)
        if node.kind == NodeKind.SQUARE:
            return left ** 2
        if node.kind == NodeKind.CUBE:
            return left ** 3
    assert node.right is not None and left is not None
    right = eval_node_mv(node.right, x)
    if node.kind == NodeKind.ADD:
        return left + right
    if node.kind == NodeKind.SUB:
        return left - right
    if node.kind == NodeKind.MUL:
        return left * right
    if node.kind == NodeKind.DIV:
        denom = np.maximum(np.abs(right), 0.1) * np.sign(right + 1e-12)
        denom = np.where(np.abs(denom) < 0.1, 0.1, denom)
        return left / denom
    raise ValueError(node.kind)


def _fmt_float(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4g}".rstrip("0").rstrip(".")


def to_infix_mv(node: ExprNode, parent_prec: int = 0) -> str:
    if node.kind == NodeKind.X:
        return f"x{int(node.value or 0)}"
    if node.kind == NodeKind.CONST:
        return _fmt_float(float(node.value))
    if node.kind in {
        NodeKind.SIN2PI,
        NodeKind.COS2PI,
        NodeKind.TANH2,
        NodeKind.ABS,
        NodeKind.SQUARE,
        NodeKind.CUBE,
    }:
        inner = to_infix_mv(node.left, 99)  # type: ignore[arg-type]
        if node.kind == NodeKind.SIN2PI:
            return f"sin(6.283185307179586*{inner})"
        if node.kind == NodeKind.COS2PI:
            return f"cos(6.283185307179586*{inner})"
        if node.kind == NodeKind.TANH2:
            return f"tanh(2*{inner})"
        if node.kind == NodeKind.ABS:
            return f"abs({inner})"
        if node.kind == NodeKind.SQUARE:
            return f"({inner})**2"
        if node.kind == NodeKind.CUBE:
            return f"({inner})**3"
    assert node.left is not None and node.right is not None
    op = {NodeKind.ADD: "+", NodeKind.SUB: "-", NodeKind.MUL: "*", NodeKind.DIV: "/"}[node.kind]
    prec = {NodeKind.ADD: 1, NodeKind.SUB: 1, NodeKind.MUL: 2, NodeKind.DIV: 2}[node.kind]
    left = to_infix_mv(node.left, prec)
    right = to_infix_mv(node.right, prec + 1)
    expr = f"{left} {op} {right}"
    if prec < parent_prec:
        return f"({expr})"
    return expr


def to_torch_expr_mv(node: ExprNode) -> str:
    if node.kind == NodeKind.X:
        dim = int(node.value or 0)
        return f"x[:, {dim}]"
    if node.kind == NodeKind.CONST:
        return f"torch.tensor({_fmt_float(float(node.value))}, dtype=x.dtype, device=x.device)"
    if node.kind in {
        NodeKind.SIN2PI,
        NodeKind.COS2PI,
        NodeKind.TANH2,
        NodeKind.ABS,
        NodeKind.SQUARE,
        NodeKind.CUBE,
    }:
        inner = to_torch_expr_mv(node.left)  # type: ignore[arg-type]
        if node.kind == NodeKind.SIN2PI:
            return f"torch.sin(6.283185307179586 * ({inner}))"
        if node.kind == NodeKind.COS2PI:
            return f"torch.cos(6.283185307179586 * ({inner}))"
        if node.kind == NodeKind.TANH2:
            return f"torch.tanh(2 * ({inner}))"
        if node.kind == NodeKind.ABS:
            return f"torch.abs({inner})"
        if node.kind == NodeKind.SQUARE:
            return f"({inner}) ** 2"
        if node.kind == NodeKind.CUBE:
            return f"({inner}) ** 3"
    left = to_torch_expr_mv(node.left)  # type: ignore[arg-type]
    right = to_torch_expr_mv(node.right)  # type: ignore[arg-type]
    if node.kind == NodeKind.ADD:
        return f"({left} + {right})"
    if node.kind == NodeKind.SUB:
        return f"({left} - {right})"
    if node.kind == NodeKind.MUL:
        return f"({left} * {right})"
    if node.kind == NodeKind.DIV:
        return (
            f"({left} / torch.clamp(torch.abs({right}), min=0.1) "
            f"* torch.sign({right} + 1e-12))"
        )
    raise ValueError(node.kind)


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
) -> bool:
    dims = used_dimensions(tree)
    if dims != set(range(input_dim)):
        return False
    rng = np.random.default_rng(0)
    xs = rng.uniform(domain[0], domain[1], size=(grid_points, input_dim))
    ys = eval_node_mv(tree, xs)
    if not np.all(np.isfinite(ys)):
        return False
    if not tree.has_nonlinear():
        return False
    y_range = float(np.max(ys) - np.min(ys))
    scaled_min_range = min_range * math.sqrt(input_dim)
    if y_range < scaled_min_range:
        return False
    if float(np.max(np.abs(ys))) > max_abs * math.sqrt(input_dim):
        return False
    frac_ok = float(np.mean(np.abs(ys) <= near_singular_abs * math.sqrt(input_dim)))
    return frac_ok >= near_singular_frac


@dataclass
class SampledExpression:
    tree: ExprNode
    expression: str
    torch_expression: str
    sampler_seed: int
    retry: int


def sample_symbolic_expression(
    seed: int,
    *,
    input_dim: int,
    max_depth: int = 3,
    max_retries: int = 200,
    domain: tuple[float, float] = (0.0, 1.0),
) -> SampledExpression:
    rng = random.Random(seed)
    for retry in range(max_retries):
        tree = sample_tree_mv(rng, max_depth, input_dim)
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
