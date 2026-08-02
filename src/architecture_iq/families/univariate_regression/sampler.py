from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class NodeKind(Enum):
    X = auto()
    CONST = auto()
    SIN2PI = auto()
    COS2PI = auto()
    TANH2 = auto()
    ABS = auto()
    SQUARE = auto()
    CUBE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


@dataclass
class ExprNode:
    kind: NodeKind
    value: float | None = None
    left: ExprNode | None = None
    right: ExprNode | None = None

    def has_nonlinear(self) -> bool:
        nonlinear = {
            NodeKind.SIN2PI,
            NodeKind.COS2PI,
            NodeKind.TANH2,
            NodeKind.ABS,
            NodeKind.SQUARE,
            NodeKind.CUBE,
            NodeKind.DIV,
        }
        if self.kind in nonlinear:
            return True
        if self.left and self.left.has_nonlinear():
            return True
        if self.right and self.right.has_nonlinear():
            return True
        return False


CONSTANTS = [round(i / 6, 4) for i in range(-12, 13)]


def _sample_leaf(rng: random.Random) -> ExprNode:
    if rng.random() < 0.55:
        return ExprNode(NodeKind.X)
    return ExprNode(NodeKind.CONST, value=rng.choice(CONSTANTS))


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


def sample_tree(rng: random.Random, max_depth: int, depth: int = 0) -> ExprNode:
    if depth >= max_depth:
        return _sample_leaf(rng)
    roll = rng.random()
    if depth == 0 or roll < 0.45:
        kind = _sample_binary(rng)
        return ExprNode(
            kind,
            left=sample_tree(rng, max_depth, depth + 1),
            right=sample_tree(rng, max_depth, depth + 1),
        )
    if roll < 0.75:
        kind = _sample_unary(rng)
        return ExprNode(kind, left=sample_tree(rng, max_depth, depth + 1))
    return _sample_leaf(rng)


def eval_node(node: ExprNode, x: np.ndarray) -> np.ndarray:
    if node.kind == NodeKind.X:
        return x
    if node.kind == NodeKind.CONST:
        return np.full_like(x, float(node.value), dtype=np.float64)
    assert node.left is not None
    left = eval_node(node.left, x)
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
    assert node.right is not None
    right = eval_node(node.right, x)
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
    raise ValueError(f"Unknown node kind {node.kind}")


def _prec(kind: NodeKind) -> int:
    if kind in {NodeKind.ADD, NodeKind.SUB}:
        return 1
    if kind in {NodeKind.MUL, NodeKind.DIV}:
        return 2
    return 3


def _fmt_float(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    text = f"{v:.4f}".rstrip("0").rstrip(".")
    return text


def to_infix(node: ExprNode, parent_prec: int = 0) -> str:
    if node.kind == NodeKind.X:
        return "x"
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
        assert node.left is not None
        inner = to_infix(node.left, 99)
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
    op = {NodeKind.ADD: "+", NodeKind.SUB: "-", NodeKind.MUL: "*", NodeKind.DIV: "/"}[
        node.kind
    ]
    prec = _prec(node.kind)
    left = to_infix(node.left, prec)
    right = to_infix(node.right, prec + 1)
    expr = f"{left} {op} {right}"
    if prec < parent_prec:
        return f"({expr})"
    return expr


def to_torch_expr(node: ExprNode) -> str:
    if node.kind == NodeKind.X:
        return "x"
    if node.kind == NodeKind.CONST:
        return f"torch.tensor({_fmt_float(float(node.value))}, dtype=x.dtype, device=x.device)"
    if node.kind == NodeKind.SIN2PI:
        return f"torch.sin(6.283185307179586 * ({to_torch_expr(node.left)}))"
    if node.kind == NodeKind.COS2PI:
        return f"torch.cos(6.283185307179586 * ({to_torch_expr(node.left)}))"
    if node.kind == NodeKind.TANH2:
        return f"torch.tanh(2 * ({to_torch_expr(node.left)}))"
    if node.kind == NodeKind.ABS:
        return f"torch.abs({to_torch_expr(node.left)})"
    if node.kind == NodeKind.SQUARE:
        inner = to_torch_expr(node.left)
        return f"({inner}) ** 2"
    if node.kind == NodeKind.CUBE:
        inner = to_torch_expr(node.left)
        return f"({inner}) ** 3"
    left = to_torch_expr(node.left)
    right = to_torch_expr(node.right)
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


@dataclass
class SampledExpression:
    tree: ExprNode
    expression: str
    torch_expression: str
    sampler_seed: int
    retry: int


def validate_expression(
    tree: ExprNode,
    domain: tuple[float, float],
    *,
    min_range: float = 0.4,
    max_abs: float = 5.0,
    near_singular_abs: float = 4.5,
    near_singular_frac: float = 0.95,
    grid_points: int = 128,
) -> bool:
    xs = np.linspace(domain[0], domain[1], grid_points)
    ys = eval_node(tree, xs)
    if not np.all(np.isfinite(ys)):
        return False
    if not tree.has_nonlinear():
        return False
    y_range = float(np.max(ys) - np.min(ys))
    if y_range < min_range:
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
    for retry in range(max_retries):
        tree = sample_tree(rng, max_depth)
        if not validate_expression(tree, domain):
            continue
        expression = to_infix(tree)
        torch_expression = to_torch_expr(tree)
        return SampledExpression(
            tree=tree,
            expression=expression,
            torch_expression=torch_expression,
            sampler_seed=seed,
            retry=retry,
        )
    raise RuntimeError(
        f"Failed to sample valid expression after {max_retries} retries (seed={seed})"
    )
