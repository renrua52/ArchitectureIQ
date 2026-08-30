"""Shared symbolic-expression core for the regression dataset families.

One tree, three consumers: `eval_tree` (validation), `render_infix` (the prose
`expression` shown in the prompt) and `render_torch` (the body of
`synthesize.py`, which is what actually runs). Keeping all three here is what
stops the prose from drifting from the executed target — the univariate and
multivariate samplers used to carry two copies of the renderer, and the copies
had already diverged in how they spelled pi.

The substantial piece is `canonicalize`. Raw sampled trees are full of things a
human would never write: `x * x`, `x + x`, `0 - x**3`, `x + -1`, `x / 0.5`,
`((x)**2)**3`, `sin(2*pi*(4 - x))`, `abs(x)` on a domain where `x >= 0`, and
folded constants like `-27`. `canonicalize` rewrites a tree into a normal form —
a sum of coefficient-times-power-product terms, with reduced trig arguments and
no provably-redundant `abs` — or returns None when the normal form would need a
constant outside the clean table, in which case the sampler just retries.

Products are deliberately *not* expanded: `2*x*(x - 3)` reads better than
`2*x**2 - 6*x`, and expansion would also let a degree-2 tree explode into a
long polynomial.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np


class NodeKind(Enum):
    X = auto()
    CONST = auto()
    SIN2PI = auto()
    COS2PI = auto()
    TANH2 = auto()
    ABS = auto()
    POW = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


@dataclass
class ExprNode:
    """`value` carries the CONST value, the X coordinate index, or the POW exponent."""

    kind: NodeKind
    value: float | None = None
    left: ExprNode | None = None
    right: ExprNode | None = None


UNARY_KINDS = frozenset(
    {NodeKind.SIN2PI, NodeKind.COS2PI, NodeKind.TANH2, NodeKind.ABS, NodeKind.POW}
)
BINARY_KINDS = frozenset({NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV})
TRIG_KINDS = frozenset({NodeKind.SIN2PI, NodeKind.COS2PI})

# 0.0 is deliberately absent: every expression it can build (`x + 0`, `x * 0`,
# `0 - x`, `x / 0`) is degenerate, and canonicalize would rewrite or reject it
# anyway, so sampling it only burns retries.
CONSTANTS = [0.5, 1.0, 2.0, 3.0, 4.0, -0.5, -1.0, -2.0, -3.0, -4.0]

# Every numeric literal that survives into a rendered expression has to read
# like something a person would write. Coefficients are snapped to this grid and
# rejected outside this magnitude, which keeps `-27` (from `(-3)**3`), `0.0156`
# (from repeated halving) and `0.1667` (from `0.5 / 3`) out of the prompt.
# Eighths, not quarters: 1/8 arises constantly from products of the 0.5 constant
# and reads perfectly well, whereas 1/16 and 1/3 do not.
CONST_QUANTUM = 0.125
CONST_ABS_MAX = 12.0
# x**9 reads as noise rather than as a target worth reasoning about.
MAX_POW_EXPONENT = 6
# Denominator floor. eval_tree divides without a clamp, so this margin is the
# whole reason the rendered formula and the executed target agree.
MIN_DENOMINATOR = 0.25

_EPS = 1e-9


def clean_constant(value: float) -> float | None:
    """Snap to the clean grid, or None when the value does not belong on it."""
    if not math.isfinite(value):
        return None
    snapped = round(value / CONST_QUANTUM) * CONST_QUANTUM
    if abs(value - snapped) > 1e-9:
        return None
    if abs(snapped) > CONST_ABS_MAX:
        return None
    return snapped + 0.0  # normalise -0.0


# --------------------------------------------------------------------------- #
# tree utilities
# --------------------------------------------------------------------------- #


def node_key(node: ExprNode) -> str:
    """Deterministic serialization; identifies equal factors and equal terms."""
    value = "" if node.value is None else f"{float(node.value):.12g}"
    left = node_key(node.left) if node.left is not None else ""
    right = node_key(node.right) if node.right is not None else ""
    return f"{node.kind.name}[{value}|{left}|{right}]"


def trees_equal(a: ExprNode, b: ExprNode) -> bool:
    return node_key(a) == node_key(b)


def is_const_only(node: ExprNode) -> bool:
    if node.kind is NodeKind.X:
        return False
    if node.kind is NodeKind.CONST:
        return True
    if node.left is not None and not is_const_only(node.left):
        return False
    if node.right is not None and not is_const_only(node.right):
        return False
    return True


def used_dimensions(node: ExprNode) -> set[int]:
    if node.kind is NodeKind.X:
        return {int(node.value or 0)}
    dims: set[int] = set()
    if node.left is not None:
        dims |= used_dimensions(node.left)
    if node.right is not None:
        dims |= used_dimensions(node.right)
    return dims


def iter_nodes(node: ExprNode):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if cur.left is not None:
            stack.append(cur.left)
        if cur.right is not None:
            stack.append(cur.right)


def constants_are_clean(node: ExprNode) -> bool:
    """No literal or exponent in the tree falls outside the clean tables."""
    for cur in iter_nodes(node):
        if cur.kind is NodeKind.CONST:
            if clean_constant(float(cur.value)) is None:
                return False
        elif cur.kind is NodeKind.POW:
            exponent = float(cur.value)
            if exponent != int(exponent):
                return False
            if not 2 <= int(exponent) <= MAX_POW_EXPONENT:
                return False
    return True


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def _eval(node: ExprNode, x: np.ndarray) -> np.ndarray:
    if node.kind is NodeKind.X:
        if x.ndim == 1:
            return x
        return x[:, int(node.value or 0)]
    if node.kind is NodeKind.CONST:
        return np.full(x.shape[0], float(node.value), dtype=np.float64)
    assert node.left is not None
    left = _eval(node.left, x)
    if node.kind is NodeKind.SIN2PI:
        return np.sin(2 * math.pi * left)
    if node.kind is NodeKind.COS2PI:
        return np.cos(2 * math.pi * left)
    if node.kind is NodeKind.TANH2:
        return np.tanh(2 * left)
    if node.kind is NodeKind.ABS:
        return np.abs(left)
    if node.kind is NodeKind.POW:
        return left.astype(np.float64) ** int(node.value)
    assert node.right is not None
    right = _eval(node.right, x)
    if node.kind is NodeKind.ADD:
        return left + right
    if node.kind is NodeKind.SUB:
        return left - right
    if node.kind is NodeKind.MUL:
        return left * right
    if node.kind is NodeKind.DIV:
        return left / right
    raise ValueError(f"Unknown node kind {node.kind}")


def eval_tree(node: ExprNode, x: np.ndarray) -> np.ndarray:
    """Evaluate on 1-D samples (shape `[N]`) or multivariate rows (`[N, D]`).

    Division is plain division: canonicalize rejects any tree whose denominator
    can approach zero on the domain, so there is no hidden clamp here to make
    the executed target differ from the rendered formula. Non-finite values are
    a legitimate intermediate result while probing a candidate, so overflow and
    divide-by-zero are silenced and screened by the callers instead.
    """
    with np.errstate(all="ignore"):
        return _eval(node, np.asarray(x, dtype=np.float64))


# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #

# A term is a coefficient times a product of powers of atomic factors. Atomic
# means X, a unary application whose own argument is already canonical, or a
# whole multi-term sum appearing inside a product.
Factors = dict[str, tuple[ExprNode, int]]
Term = tuple[float, Factors]


@dataclass(frozen=True)
class CanonContext:
    """Domain probe used to settle sign questions (currently only `abs`)."""

    samples: np.ndarray


def context_1d(domain: tuple[float, float], *, grid_points: int = 2049) -> CanonContext:
    return CanonContext(samples=np.linspace(domain[0], domain[1], grid_points))


def context_nd(
    domain: tuple[float, float],
    input_dim: int,
    *,
    grid_points: int = 4096,
) -> CanonContext:
    # Interior draws plus the corners and the centre, so a sign test cannot be
    # fooled by a region the random sample happens to miss.
    rng = np.random.default_rng(12345)
    interior = rng.uniform(domain[0], domain[1], size=(grid_points, input_dim))
    landmarks = np.array(
        [
            [domain[0]] * input_dim,
            [domain[1]] * input_dim,
            [(domain[0] + domain[1]) / 2.0] * input_dim,
        ]
    )
    return CanonContext(samples=np.concatenate([interior, landmarks], axis=0))


def _term_key(factors: Factors) -> tuple:
    return tuple(sorted((key, exp) for key, (_, exp) in factors.items()))


def _term_degree(factors: Factors) -> int:
    return sum(exp for _, exp in factors.values())


def _sorted_terms(terms: list[Term]) -> list[Term]:
    """Canonical print order: highest degree first, then by factor key."""
    return sorted(
        terms, key=lambda term: (-_term_degree(term[1]), _term_key(term[1]), term[0])
    )


def _lead_with_positive(ordered: list[Term]) -> list[Term]:
    """Promote the first positive term so a sum rarely opens with a minus sign.

    `3 - x**4` and `sin(2*pi*x) - cos(2*pi*x)` read better than `-x**4 + 3` and
    `-cos(2*pi*x) + sin(2*pi*x)`; a sum whose every term is negative still has
    to open with one.
    """
    if not ordered or ordered[0][0] > 0:
        return ordered
    for index, (coeff, _) in enumerate(ordered):
        if coeff > 0:
            return [ordered[index], *ordered[:index], *ordered[index + 1 :]]
    return ordered


def _merge(terms: list[Term]) -> list[Term] | None:
    merged: dict[tuple, tuple[float, Factors]] = {}
    for coeff, factors in terms:
        key = _term_key(factors)
        if key in merged:
            existing_coeff, existing_factors = merged[key]
            merged[key] = (existing_coeff + coeff, existing_factors)
        else:
            merged[key] = (coeff, factors)
    out: list[Term] = []
    for coeff, factors in merged.values():
        cleaned = clean_constant(coeff)
        if cleaned is None:
            return None
        if cleaned == 0.0:
            continue
        out.append((cleaned, factors))
    return out


def _scale(terms: list[Term], factor: float) -> list[Term] | None:
    scaled: list[Term] = []
    for coeff, factors in terms:
        cleaned = clean_constant(coeff * factor)
        if cleaned is None:
            return None
        if cleaned == 0.0:
            continue
        scaled.append((cleaned, factors))
    return scaled


def _negate(terms: list[Term]) -> list[Term] | None:
    return _scale(terms, -1.0)


def _pure_constant(terms: list[Term]) -> float | None:
    """The value when the sum is a bare number (an empty sum is 0)."""
    if not terms:
        return 0.0
    if len(terms) == 1 and not terms[0][1]:
        return terms[0][0]
    return None


def _constant_terms(value: float) -> list[Term] | None:
    cleaned = clean_constant(value)
    if cleaned is None:
        return None
    return [] if cleaned == 0.0 else [(cleaned, {})]


def _split_constant(terms: list[Term]) -> tuple[float, list[Term]]:
    shift = 0.0
    rest: list[Term] = []
    for coeff, factors in terms:
        if factors:
            rest.append((coeff, factors))
        else:
            shift += coeff
    return shift, rest


def _combine_factors(a: Factors, b: Factors) -> Factors | None:
    out: Factors = dict(a)
    for key, (node, exp) in b.items():
        if key in out:
            merged_exp = out[key][1] + exp
            if merged_exp == 0:
                del out[key]
                continue
            out[key] = (node, merged_exp)
        else:
            out[key] = (node, exp)
    for _, exp in out.values():
        if abs(exp) > MAX_POW_EXPONENT:
            return None
    return out


def _atom_factors(node: ExprNode) -> Factors:
    return {node_key(node): (node, 1)}


def _atom_term(node: ExprNode, *, coeff: float = 1.0) -> list[Term]:
    return [(coeff, _atom_factors(node))]


def _as_atom(terms: list[Term]) -> Factors | None:
    """Freeze a multi-term sum so it can join a product unexpanded."""
    node = sum_to_node(terms)
    return None if node is None else _atom_factors(node)


def _multiply(left: list[Term], right: list[Term]) -> list[Term] | None:
    left_const = _pure_constant(left)
    if left_const is not None:
        return _scale(right, left_const)
    right_const = _pure_constant(right)
    if right_const is not None:
        return _scale(left, right_const)
    coeff = 1.0
    if len(left) == 1:
        coeff *= left[0][0]
        left_factors: Factors | None = left[0][1]
    else:
        left_factors = _as_atom(left)
    if len(right) == 1:
        coeff *= right[0][0]
        right_factors: Factors | None = right[0][1]
    else:
        right_factors = _as_atom(right)
    if left_factors is None or right_factors is None:
        return None
    factors = _combine_factors(left_factors, right_factors)
    if factors is None:
        return None
    return _merge([(coeff, factors)])


def _invert(terms: list[Term]) -> list[Term] | None:
    """1 / sum, as one term carrying negative exponents where it can."""
    const = _pure_constant(terms)
    if const is not None:
        if const == 0.0:
            return None
        return _constant_terms(1.0 / const)
    if len(terms) == 1:
        coeff, factors = terms[0]
        cleaned = clean_constant(1.0 / coeff)
        if cleaned is None:
            return None
        inverted: Factors = {}
        for key, (node, exp) in factors.items():
            inverted[key] = (node, -exp)
        return [(cleaned, inverted)]
    atom = _as_atom(terms)
    if atom is None:
        return None
    key, (node, _) = next(iter(atom.items()))
    return [(1.0, {key: (node, -1)})]


def _power(terms: list[Term], exponent: int) -> list[Term] | None:
    if exponent < 2:
        return None
    const = _pure_constant(terms)
    if const is not None:
        return _constant_terms(const**exponent)
    if len(terms) == 1:
        coeff, factors = terms[0]
        cleaned = clean_constant(coeff**exponent)
        if cleaned is None:
            return None
        raised: Factors = {}
        for key, (node, exp) in factors.items():
            new_exp = exp * exponent
            if abs(new_exp) > MAX_POW_EXPONENT:
                return None
            raised[key] = (node, new_exp)
        return _merge([(cleaned, raised)])
    if exponent > MAX_POW_EXPONENT:
        return None
    atom = _as_atom(terms)
    if atom is None:
        return None
    key, (node, _) = next(iter(atom.items()))
    return [(1.0, {key: (node, exponent)})]


def _leading_coeff(terms: list[Term]) -> float:
    ordered = _sorted_terms(terms)
    return ordered[0][0] if ordered else 0.0


def _is_nonnegative(node: ExprNode, ctx: CanonContext) -> bool:
    """Structurally or numerically nonnegative everywhere on the domain."""
    if node.kind is NodeKind.ABS:
        return True
    if node.kind is NodeKind.POW and int(node.value) % 2 == 0:
        return True
    values = eval_tree(node, ctx.samples)
    return bool(np.all(np.isfinite(values))) and float(np.min(values)) >= 0.0


def _is_nonpositive(node: ExprNode, ctx: CanonContext) -> bool:
    values = eval_tree(node, ctx.samples)
    return bool(np.all(np.isfinite(values))) and float(np.max(values)) <= 0.0


def _canon_abs(inner: list[Term], ctx: CanonContext) -> list[Term] | None:
    node = sum_to_node(inner)
    if node is None:
        return None
    # abs() is a no-op wherever its argument keeps one sign, and the target is
    # only ever evaluated on the domain. `abs(x)` on [0, 1] and `abs(x**2)` are
    # the common cases; a same-sign argument would otherwise leave the prompt
    # showing an operation that does nothing.
    if _is_nonnegative(node, ctx):
        return inner
    if _is_nonpositive(node, ctx):
        return _negate(inner)
    return _atom_term(ExprNode(NodeKind.ABS, left=node))


def _canon_trig(kind: NodeKind, inner: list[Term]) -> list[Term] | None:
    shift, rest = _split_constant(inner)
    if not rest:
        raw = (
            math.sin(2 * math.pi * shift)
            if kind is NodeKind.SIN2PI
            else math.cos(2 * math.pi * shift)
        )
        return _constant_terms(raw)
    sign = 1.0
    if _leading_coeff(rest) < 0:
        negated = _negate(rest)
        if negated is None:
            return None
        rest = negated
        shift = -shift
        if kind is NodeKind.SIN2PI:
            sign = -sign  # sin is odd; cos is even
    # sin/cos of 2*pi*u have period 1 in u, and a half-period shift is a sign
    # flip, so `sin(2*pi*(4 - x))` collapses to `-sin(2*pi*x)`.
    shift -= math.floor(shift)
    if abs(shift - 0.5) < _EPS:
        shift = 0.0
        sign = -sign
    elif shift < _EPS or abs(shift - 1.0) < _EPS:
        shift = 0.0
    terms = list(rest)
    if shift != 0.0:
        cleaned = clean_constant(shift)
        if cleaned is None:
            return None
        terms.append((cleaned, {}))
    merged = _merge(terms)
    if not merged:
        return None
    node = sum_to_node(merged)
    if node is None:
        return None
    return _atom_term(ExprNode(kind, left=node), coeff=sign)


def _canon_tanh(inner: list[Term]) -> list[Term] | None:
    if _pure_constant(inner) is not None:
        return _constant_terms(math.tanh(2 * _pure_constant(inner)))
    sign = 1.0
    if _leading_coeff(inner) < 0:
        negated = _negate(inner)
        if negated is None:
            return None
        inner = negated
        sign = -1.0  # tanh is odd
    node = sum_to_node(inner)
    if node is None:
        return None
    return _atom_term(ExprNode(NodeKind.TANH2, left=node), coeff=sign)


def _canon(node: ExprNode, ctx: CanonContext) -> list[Term] | None:
    if node.kind is NodeKind.X:
        return _atom_term(ExprNode(NodeKind.X, value=node.value))
    if node.kind is NodeKind.CONST:
        return _constant_terms(float(node.value))

    if node.kind in BINARY_KINDS:
        assert node.left is not None and node.right is not None
        left = _canon(node.left, ctx)
        right = _canon(node.right, ctx)
        if left is None or right is None:
            return None
        if node.kind is NodeKind.ADD:
            return _merge(left + right)
        if node.kind is NodeKind.SUB:
            negated = _negate(right)
            if negated is None:
                return None
            return _merge(left + negated)
        if node.kind is NodeKind.MUL:
            return _multiply(left, right)
        inverted = _invert(right)
        if inverted is None:
            return None
        return _multiply(left, inverted)

    assert node.left is not None
    inner = _canon(node.left, ctx)
    if inner is None:
        return None
    if node.kind is NodeKind.POW:
        return _power(inner, int(node.value))
    if node.kind is NodeKind.ABS:
        return _canon_abs(inner, ctx)
    if node.kind is NodeKind.TANH2:
        return _canon_tanh(inner)
    if node.kind in TRIG_KINDS:
        return _canon_trig(node.kind, inner)
    raise ValueError(f"Unknown node kind {node.kind}")


def _factor_node(node: ExprNode, exponent: int) -> ExprNode:
    if exponent == 1:
        return node
    return ExprNode(NodeKind.POW, value=float(exponent), left=node)


def _product(nodes: list[ExprNode]) -> ExprNode:
    acc = nodes[0]
    for node in nodes[1:]:
        acc = ExprNode(NodeKind.MUL, left=acc, right=node)
    return acc


def _term_to_node(coeff: float, factors: Factors) -> ExprNode:
    """`coeff * prod(f**e)`, with any negative exponents folded into one DIV."""
    ordered = sorted(factors.items(), key=lambda item: item[0])
    numerator = [_factor_node(node, exp) for _, (node, exp) in ordered if exp > 0]
    denominator = [_factor_node(node, -exp) for _, (node, exp) in ordered if exp < 0]
    if not numerator:
        body: ExprNode = ExprNode(NodeKind.CONST, value=coeff)
    else:
        # The coefficient joins the factor list rather than wrapping it, so the
        # product stays one left-associated chain and needs no inner parentheses:
        # `0.25*tanh(2*x)*x`, not `0.25*(tanh(2*x)*x)`. A coefficient of -1 is
        # kept as an explicit CONST child; the renderer prints it as unary minus.
        factor_nodes = list(numerator)
        if coeff != 1.0:
            factor_nodes.insert(0, ExprNode(NodeKind.CONST, value=coeff))
        body = _product(factor_nodes)
    if denominator:
        return ExprNode(NodeKind.DIV, left=body, right=_product(denominator))
    return body


def sum_to_node(terms: list[Term]) -> ExprNode | None:
    """Rebuild a tree from canonical terms.

    Negative terms after the first become SUB nodes, so a rendered expression
    never contains `+ -3` or `- -3`; only the leading term may be negative, and
    the renderer prints that as a unary minus.
    """
    if not terms:
        return ExprNode(NodeKind.CONST, value=0.0)
    ordered = _lead_with_positive(_sorted_terms(terms))
    first_coeff, first_factors = ordered[0]
    acc = _term_to_node(first_coeff, first_factors)
    for coeff, factors in ordered[1:]:
        kind = NodeKind.SUB if coeff < 0 else NodeKind.ADD
        acc = ExprNode(kind, left=acc, right=_term_to_node(abs(coeff), factors))
    return acc


def denominators_are_safe(
    node: ExprNode,
    ctx: CanonContext,
    *,
    min_denominator: float = MIN_DENOMINATOR,
) -> bool:
    """No DIV denominator may approach zero anywhere on the domain."""
    for cur in iter_nodes(node):
        if cur.kind is not NodeKind.DIV:
            continue
        assert cur.right is not None
        values = eval_tree(cur.right, ctx.samples)
        if not np.all(np.isfinite(values)):
            return False
        if float(np.min(np.abs(values))) < min_denominator:
            return False
    return True


def canonicalize(node: ExprNode, ctx: CanonContext) -> ExprNode | None:
    """Normal form, or None when the sample should be rejected and retried."""
    terms = _canon(node, ctx)
    if not terms:
        return None
    canonical = sum_to_node(terms)
    if canonical is None or canonical.kind is NodeKind.CONST:
        return None
    if not constants_are_clean(canonical):
        return None
    if not denominators_are_safe(canonical, ctx):
        return None
    return canonical


# --------------------------------------------------------------------------- #
# sampling helpers shared by both families
# --------------------------------------------------------------------------- #

# POW twice so integer powers stay about as common as they were when SQUARE and
# CUBE were separate kinds.
_UNARY_CHOICES = (
    NodeKind.SIN2PI,
    NodeKind.COS2PI,
    NodeKind.TANH2,
    NodeKind.ABS,
    NodeKind.POW,
    NodeKind.POW,
)
_BINARY_CHOICES = (NodeKind.ADD, NodeKind.SUB, NodeKind.MUL, NodeKind.DIV)


def sample_unary_node(rng: random.Random, child: ExprNode) -> ExprNode:
    kind = rng.choice(_UNARY_CHOICES)
    if kind is NodeKind.POW:
        return ExprNode(NodeKind.POW, value=float(rng.choice([2, 3])), left=child)
    return ExprNode(kind, left=child)


def sample_binary_kind(rng: random.Random) -> NodeKind:
    return rng.choice(_BINARY_CHOICES)


def sample_constant_node(rng: random.Random) -> ExprNode:
    return ExprNode(NodeKind.CONST, value=rng.choice(CONSTANTS))


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Backend:
    """How one target language spells the pieces of an expression."""

    var: Callable[[int], str]
    call: Callable[[str, str], str]
    pi: str
    mul: str
    power: str


def _prec(kind: NodeKind) -> int:
    if kind in {NodeKind.ADD, NodeKind.SUB}:
        return 1
    if kind in {NodeKind.MUL, NodeKind.DIV}:
        return 2
    if kind is NodeKind.POW:
        return 3
    return 4


def _scaled_argument(
    node: ExprNode, scale: float, backend: Backend, *, use_pi: bool
) -> str:
    """Render `scale * node` (times pi), folding a leading numeric factor in.

    This is what turns `sin(2*pi*(2*x))` into `sin(4*pi*x)` and `tanh(2*(0.5*x))`
    into `tanh(x)`.
    """
    inner = node
    if (
        node.kind is NodeKind.MUL
        and node.left is not None
        and node.left.kind is NodeKind.CONST
    ):
        scale *= float(node.left.value)
        assert node.right is not None
        inner = node.right
    pieces: list[str] = []
    if scale != 1.0:
        pieces.append(format_number(scale))
    if use_pi:
        pieces.append(backend.pi)
    pieces.append(_render(inner, _prec(NodeKind.MUL) + 1, backend))
    return backend.mul.join(pieces)


def _render(node: ExprNode, parent_prec: int, backend: Backend) -> str:
    if node.kind is NodeKind.X:
        return backend.var(int(node.value or 0))
    if node.kind is NodeKind.CONST:
        return format_number(float(node.value))
    if node.kind in TRIG_KINDS:
        assert node.left is not None
        name = "sin" if node.kind is NodeKind.SIN2PI else "cos"
        return backend.call(name, _scaled_argument(node.left, 2.0, backend, use_pi=True))
    if node.kind is NodeKind.TANH2:
        assert node.left is not None
        return backend.call(
            "tanh", _scaled_argument(node.left, 2.0, backend, use_pi=False)
        )
    if node.kind is NodeKind.ABS:
        assert node.left is not None
        return backend.call("abs", _render(node.left, 0, backend))
    if node.kind is NodeKind.POW:
        assert node.left is not None
        base = _render(node.left, _prec(NodeKind.POW) + 1, backend)
        text = f"{base}{backend.power}{int(node.value)}"
        return f"({text})" if _prec(NodeKind.POW) < parent_prec else text

    assert node.left is not None and node.right is not None
    prec = _prec(node.kind)
    if node.kind is NodeKind.MUL:
        if node.left.kind is NodeKind.CONST and float(node.left.value) == -1.0:
            # A leading -1 coefficient reads as a unary minus. In both prose and
            # Python `-x ** 2` means -(x ** 2), which is the intended term.
            text = f"-{_render(node.right, prec + 1, backend)}"
        else:
            left = _render(node.left, prec, backend)
            right = _render(node.right, prec + 1, backend)
            text = f"{left}{backend.mul}{right}"
    else:
        left = _render(node.left, prec, backend)
        right = _render(node.right, prec + 1, backend)
        op = {NodeKind.ADD: "+", NodeKind.SUB: "-", NodeKind.DIV: "/"}[node.kind]
        text = f"{left} {op} {right}"
    return f"({text})" if prec < parent_prec else text


def render_infix(node: ExprNode, *, var: Callable[[int], str]) -> str:
    """The prose form shown in the prompt, e.g. `2*x**3 - sin(2*pi*x)`."""
    backend = Backend(
        var=var,
        call=lambda name, arg: f"{name}({arg})",
        pi="pi",
        mul="*",
        power="**",
    )
    return _render(node, 0, backend)


def render_torch(node: ExprNode, *, var: Callable[[int], str]) -> str:
    """The executable form written into `synthesize.py`."""
    backend = Backend(
        var=var,
        call=lambda name, arg: f"torch.{name}({arg})",
        pi="torch.pi",
        mul=" * ",
        power=" ** ",
    )
    return _render(node, 0, backend)
