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


CONSTANTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, -0.5, -1.0, -2.0, -3.0, -4.0]


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


def _is_const_only(node: ExprNode) -> bool:
    if node.kind == NodeKind.X:
        return False
    if node.kind == NodeKind.CONST:
        return True
    if node.left is not None and not _is_const_only(node.left):
        return False
    if node.right is not None and not _is_const_only(node.right):
        return False
    return True


def _trees_equal(a: ExprNode, b: ExprNode) -> bool:
    if a.kind != b.kind or a.value != b.value:
        return False
    if (a.left is None) != (b.left is None):
        return False
    if a.left is not None and not _trees_equal(a.left, b.left):
        return False
    if (a.right is None) != (b.right is None):
        return False
    if a.right is not None and not _trees_equal(a.right, b.right):
        return False
    return True


def _const_value(node: ExprNode) -> float | None:
    if not _is_const_only(node):
        return None
    ys = eval_node(node, np.array([0.5]))
    if not np.all(np.isfinite(ys)):
        return None
    return float(ys[0])


def simplify_tree(node: ExprNode) -> ExprNode | None:
    """Constant folding plus degenerate-pattern elimination.

    Returns None when the tree is structurally invalid (e.g. division by a
    zero constant) and the sample should be rejected. Dimensional X nodes
    (value = dim index) are preserved for multivariate reuse.
    """
    if node.kind == NodeKind.X:
        return ExprNode(NodeKind.X, value=node.value)
    if node.kind == NodeKind.CONST:
        return ExprNode(NodeKind.CONST, value=float(node.value))

    left = simplify_tree(node.left) if node.left is not None else None
    if left is None and node.left is not None:
        return None
    right = simplify_tree(node.right) if node.right is not None else None
    if right is None and node.right is not None:
        return None

    node = ExprNode(node.kind, value=node.value, left=left, right=right)

    # Constant folding: collapse const-only subtrees to a single CONST leaf,
    # but only when the folded value is a clean multiple of 0.5 — folded
    # values like tanh(1)=0.7616 or 0.5/3=0.1667 would reintroduce the
    # arbitrary-decimal constants this sampler is supposed to avoid.
    folded = _const_value(node)
    if folded is not None:
        snapped = round(folded * 2) / 2
        if abs(folded - snapped) > 1e-9:
            return None
        return ExprNode(NodeKind.CONST, value=snapped)

    if node.kind in {NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV}:
        assert node.left is not None and node.right is not None
        lk = node.left.kind
        rk = node.right.kind
        lv = node.left.value if lk == NodeKind.CONST else None
        rv = node.right.value if rk == NodeKind.CONST else None
        if node.kind == NodeKind.ADD:
            if lv == 0.0:
                return node.right
            if rv == 0.0:
                return node.left
        elif node.kind == NodeKind.SUB:
            if rv == 0.0:
                return node.left
            if _trees_equal(node.left, node.right):
                return ExprNode(NodeKind.CONST, value=0.0)
        elif node.kind == NodeKind.MUL:
            if lv == 0.0 or rv == 0.0:
                return ExprNode(NodeKind.CONST, value=0.0)
            if lv == 1.0:
                return node.right
            if rv == 1.0:
                return node.left
        else:  # DIV
            if rk == NodeKind.CONST and rv == 0.0:
                return None
            if lk == NodeKind.CONST and lv == 0.0:
                return ExprNode(NodeKind.CONST, value=0.0)
            if rk == NodeKind.CONST and rv == 1.0:
                return node.left
    return node


def _div_guard_ok(
    node: ExprNode,
    domain: tuple[float, float],
    *,
    min_denominator: float = 0.1,
    grid_points: int = 1024,
) -> bool:
    """Every DIV right operand must stay above the clamp floor on the domain.

    eval_node applies a silent |denominator| >= 0.1 clamp; when the sampled
    expression never triggers it, the rendered formula and the executed
    target coincide exactly.
    """
    xs = np.linspace(domain[0], domain[1], grid_points)
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.kind == NodeKind.DIV:
            assert cur.right is not None
            denom = eval_node(cur.right, xs)
            if not np.all(np.isfinite(denom)):
                return False
            if float(np.min(np.abs(denom))) < min_denominator:
                return False
        if cur.left is not None:
            stack.append(cur.left)
        if cur.right is not None:
            stack.append(cur.right)
    return True


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
    ys = eval_node(tree, xs)
    if not np.all(np.isfinite(ys)):
        return False
    # Numeric nonlinearity: an affine fit must leave a substantial residual.
    # The previous syntactic check was fooled by nonlinear nodes applied to
    # constant subtrees (e.g. abs(1.5), tanh(2*0.5), 2**3).
    residual = _affine_residual(xs, ys)
    if float(np.max(np.abs(residual))) <= 1e-3:
        return False
    y_range = float(np.max(ys) - np.min(ys))
    if y_range < min_range:
        return False
    # The nonlinear component must carry a substantial share of the output
    # range so expressions are neither constant- nor affine-dominated.
    residual_range = float(np.max(residual) - np.min(residual))
    if residual_range < min_nonlinear_fraction * y_range:
        return False
    if not _div_guard_ok(tree, domain):
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
        tree = simplify_tree(tree)
        if tree is None or tree.kind == NodeKind.CONST:
            continue
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
