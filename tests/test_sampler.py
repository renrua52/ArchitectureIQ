from __future__ import annotations

import math
import re

import numpy as np
import torch

from architecture_iq.families.symbolic_expr import (
    CONST_ABS_MAX,
    CONST_QUANTUM,
    MAX_POW_EXPONENT,
    MIN_DENOMINATOR,
    ExprNode,
    NodeKind,
    canonicalize,
    context_1d,
    eval_tree,
    is_const_only,
    iter_nodes,
    trees_equal,
)
from architecture_iq.families.univariate_regression.sampler import (
    eval_node,
    sample_symbolic_expression,
    to_infix,
    validate_expression,
)

SEEDS = range(200)
CTX = context_1d((0.0, 1.0))


def _canon(node: ExprNode) -> ExprNode | None:
    return canonicalize(node, CTX)


def _x() -> ExprNode:
    return ExprNode(NodeKind.X)


def _const(value: float) -> ExprNode:
    return ExprNode(NodeKind.CONST, value=value)


def test_sampler_reproducible():
    a = sample_symbolic_expression(seed=7, max_depth=3)
    b = sample_symbolic_expression(seed=7, max_depth=3)
    assert a.expression == b.expression
    assert a.retry == b.retry


def test_sampler_expression_is_numerically_nonlinear():
    # Nonlinearity is decided numerically (affine residual), so genuine
    # polynomials pass while abs(const) tricks fail.
    for seed in range(20):
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        xs = np.linspace(0.0, 1.0, 256)
        ys = eval_node(sampled.tree, xs)
        design = np.stack([np.ones_like(xs), xs], axis=1)
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        residual = ys - design @ coef
        assert float(np.max(np.abs(residual))) > 1e-3, sampled.expression
        assert validate_expression(sampled.tree, (0.0, 1.0))


def test_sampler_not_trivial_constant():
    seen = {sample_symbolic_expression(seed=seed).expression for seed in range(30)}
    assert len(seen) > 10


def test_sampler_renders_pi_symbolically():
    """`pi` / `torch.pi`, never the 6.283185307179586 literal.

    A numeric 2*pi in the prompt reads as an arbitrary magic constant, and it was
    also the one place the old renderer had to be exempted from the clean-constant
    check.
    """
    saw_trig = False
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        assert "6.283185307179586" not in sampled.expression
        assert "6.283185307179586" not in sampled.torch_expression
        if "sin(" in sampled.expression or "cos(" in sampled.expression:
            saw_trig = True
            assert "pi" in sampled.expression
            assert "torch.pi" in sampled.torch_expression
    assert saw_trig, "no trig sampled; the assertions above never ran"


def test_sampler_constants_are_clean():
    """Every literal is a multiple of an eighth with a bounded magnitude."""
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        # No long decimal tails such as 0.1667 or 0.0156.
        assert not re.search(r"\d+\.\d{4,}", sampled.expression), sampled.expression
        for node in iter_nodes(sampled.tree):
            if node.kind is not NodeKind.CONST:
                continue
            value = float(node.value)
            assert abs(value) <= CONST_ABS_MAX, sampled.expression
            assert abs(value / CONST_QUANTUM - round(value / CONST_QUANTUM)) < 1e-9, (
                sampled.expression
            )


def test_sampler_no_double_signs():
    """Sign normalization: no `+ -3`, `- -3`, or `0 - u` reaches the prompt."""
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        for pattern in ("+ -", "- -", "* -", "/ -", "0 - ", "--"):
            assert pattern not in sampled.expression, (pattern, sampled.expression)


def test_sampler_no_degenerate_patterns():
    """x*0, x-x, x/1, x*x and const-only subtrees are gone before rendering."""
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        node = sampled.tree
        for cur in iter_nodes(node):
            if cur.kind is NodeKind.MUL:
                for side in (cur.left, cur.right):
                    assert not (
                        side is not None
                        and side.kind is NodeKind.CONST
                        and float(side.value) == 0.0
                    ), f"zero-multiplication in {sampled.expression}"
                # x*x must have collapsed into x**2.
                assert not trees_equal(cur.left, cur.right), (
                    f"unmerged square in {sampled.expression}"
                )
            if cur.kind is NodeKind.SUB:
                assert not trees_equal(cur.left, cur.right), (
                    f"x-x in {sampled.expression}"
                )
            if cur.kind is NodeKind.DIV:
                assert not (
                    cur.right is not None
                    and cur.right.kind is NodeKind.CONST
                    and float(cur.right.value) in (1.0, -1.0)
                ), f"x/1 in {sampled.expression}"
            if cur.kind is NodeKind.POW:
                # No ((x)**2)**3 -- nested powers multiply their exponents.
                assert cur.left.kind is not NodeKind.POW, sampled.expression
                assert 2 <= int(cur.value) <= MAX_POW_EXPONENT, sampled.expression
            assert not is_const_only(cur) or cur.kind is NodeKind.CONST, (
                f"unfolded const-only subtree in {sampled.expression}"
            )


def test_sampler_no_redundant_parentheses_around_powers():
    """`x**2`, not `(x)**2`; a compound base still keeps its parentheses."""
    for seed in SEEDS:
        expression = sample_symbolic_expression(seed=seed, max_depth=3).expression
        assert "(x)" not in expression, expression
    assert to_infix(_canon(ExprNode(NodeKind.POW, value=2.0, left=_x()))) == "x**2"
    squared_sum = ExprNode(
        NodeKind.POW,
        value=2.0,
        left=ExprNode(NodeKind.ADD, left=_x(), right=_const(1.0)),
    )
    assert to_infix(_canon(squared_sum)) == "(x + 1)**2"


def test_sampler_div_guard_holds_without_a_clamp():
    """Denominators stay clear of zero, so no hidden clamp is needed.

    eval_tree now divides plainly: the executed target, the prose expression and
    the validation all evaluate the identical function.
    """
    xs = np.linspace(0.0, 1.0, 4096)
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        assert "clamp" not in sampled.torch_expression
        for node in iter_nodes(sampled.tree):
            if node.kind is not NodeKind.DIV:
                continue
            denom = eval_node(node.right, xs)
            assert float(np.min(np.abs(denom))) >= MIN_DENOMINATOR - 1e-9, (
                sampled.expression
            )


def test_prose_torch_and_eval_agree():
    """The three renderings of one tree must be the same function.

    This is the invariant the shared core exists to protect: the prompt shows
    `expression`, synthesize.py runs `torch_expression`, and validation used
    eval_tree. Two hand-maintained renderers had already drifted once.
    """
    for seed in SEEDS:
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        xs = np.linspace(0.0, 1.0, 257)
        reference = eval_node(sampled.tree, xs)

        executed = eval(
            sampled.torch_expression,
            {"torch": torch},
            {"x": torch.tensor(xs, dtype=torch.float32)},
        )
        np.testing.assert_allclose(
            reference, executed.detach().numpy(), atol=2e-5, rtol=2e-5
        )

        prose = re.sub(r"\b(sin|cos|tanh|abs)\(", r"np.\1(", sampled.expression)
        np.testing.assert_allclose(
            reference,
            eval(prose, {"np": np, "pi": math.pi}, {"x": xs}),
            atol=1e-9,
            rtol=1e-9,
        )


# --------------------------------------------------------------------------- #
# canonicalize, on hand-built trees for each defect the sampler used to emit
# --------------------------------------------------------------------------- #


def test_canonicalize_merges_repeated_factors_and_terms():
    assert to_infix(_canon(ExprNode(NodeKind.MUL, left=_x(), right=_x()))) == "x**2"
    assert to_infix(_canon(ExprNode(NodeKind.ADD, left=_x(), right=_x()))) == "2*x"
    nested = ExprNode(
        NodeKind.POW,
        value=3.0,
        left=ExprNode(NodeKind.POW, value=2.0, left=_x()),
    )
    assert to_infix(_canon(nested)) == "x**6"


def test_canonicalize_folds_constant_division_and_multiplication():
    # x / 0.5 is 2*x, and x * 1 is x.
    assert to_infix(_canon(ExprNode(NodeKind.DIV, left=_x(), right=_const(0.5)))) == "2*x"
    cubed = ExprNode(NodeKind.POW, value=3.0, left=_x())
    assert to_infix(_canon(ExprNode(NodeKind.MUL, left=cubed, right=_const(1.0)))) == "x**3"


def test_canonicalize_rewrites_zero_minus_u_as_a_leading_minus():
    """`0 - x**3` used to survive: simplify_tree had no SUB lv == 0 case."""
    cubed = ExprNode(NodeKind.POW, value=3.0, left=_x())
    assert to_infix(_canon(ExprNode(NodeKind.SUB, left=_const(0.0), right=cubed))) == "-x**3"


def test_canonicalize_leads_with_a_positive_term_when_one_exists():
    # `3 - x**4` rather than `-x**4 + 3`.
    quartic = ExprNode(NodeKind.POW, value=4.0, left=_x())
    tree = ExprNode(NodeKind.SUB, left=_const(3.0), right=quartic)
    assert to_infix(_canon(tree)) == "3 - x**4"


def test_canonicalize_reduces_trig_shifts_and_signs():
    """sin/cos of 2*pi*u have period 1, and a half period is a sign flip."""

    def sin_of(inner: ExprNode) -> str:
        return to_infix(_canon(ExprNode(NodeKind.SIN2PI, left=inner)))

    # sin(2*pi*(x + 1)) == sin(2*pi*x)
    assert sin_of(ExprNode(NodeKind.ADD, left=_x(), right=_const(1.0))) == "sin(2*pi*x)"
    # sin(2*pi*(x + 0.5)) == -sin(2*pi*x)
    assert sin_of(ExprNode(NodeKind.ADD, left=_x(), right=_const(0.5))) == "-sin(2*pi*x)"
    # sin(2*pi*(4 - x)) == -sin(2*pi*x): the leading sign comes out, sin is odd.
    assert sin_of(ExprNode(NodeKind.SUB, left=_const(4.0), right=_x())) == "-sin(2*pi*x)"
    # cos is even, so extracting the leading sign does not flip it.
    cos_tree = ExprNode(
        NodeKind.COS2PI, left=ExprNode(NodeKind.SUB, left=_const(4.0), right=_x())
    )
    assert to_infix(_canon(cos_tree)) == "cos(2*pi*x)"
    # The inner coefficient folds into the pi multiplier: sin(2*pi*(2*x)).
    doubled = ExprNode(NodeKind.ADD, left=_x(), right=_x())
    assert sin_of(doubled) == "sin(4*pi*x)"


def test_canonicalize_drops_abs_on_a_same_sign_argument():
    """On [0, 1] `abs(x)` is `x`, and `abs(u**2)` is `u**2` for any domain."""
    assert to_infix(_canon(ExprNode(NodeKind.ABS, left=_x()))) == "x"
    squared = ExprNode(NodeKind.POW, value=2.0, left=_x())
    assert to_infix(_canon(ExprNode(NodeKind.ABS, left=squared))) == "x**2"
    # A genuinely sign-changing argument keeps its abs.
    sin_x = ExprNode(NodeKind.SIN2PI, left=_x())
    assert to_infix(_canon(ExprNode(NodeKind.ABS, left=sin_x))) == "abs(sin(2*pi*x))"


def test_canonicalize_rejects_unclean_constants():
    """(-3)**3 = -27 and 0.5/3 = 0.1667 are rejected rather than rendered."""
    cube_of_neg3 = ExprNode(NodeKind.POW, value=3.0, left=_const(-3.0))
    assert _canon(ExprNode(NodeKind.ADD, left=_x(), right=cube_of_neg3)) is None
    third = ExprNode(NodeKind.DIV, left=_const(0.5), right=_const(3.0))
    assert _canon(ExprNode(NodeKind.MUL, left=_x(), right=third)) is None


def test_canonicalize_rejects_constant_and_near_singular_trees():
    assert _canon(ExprNode(NodeKind.SUB, left=_x(), right=_x())) is None
    assert _canon(_const(2.0)) is None
    # 1 / x is unbounded at x = 0, so the denominator guard rejects it.
    assert _canon(ExprNode(NodeKind.DIV, left=_const(1.0), right=_x())) is None


def test_canonicalize_does_not_expand_products():
    """`(4 - x)*x**3` stays factored; expansion would bloat the prompt."""
    tree = ExprNode(
        NodeKind.MUL,
        left=ExprNode(NodeKind.SUB, left=_const(4.0), right=_x()),
        right=ExprNode(NodeKind.POW, value=3.0, left=_x()),
    )
    assert to_infix(_canon(tree)) == "(4 - x)*x**3"


def test_eval_tree_divides_without_clamping():
    """The clamp is gone; eval_tree is the plain arithmetic the prompt shows."""
    tree = ExprNode(NodeKind.DIV, left=_const(1.0), right=_x())
    values = eval_tree(tree, np.array([0.05, 0.5, 1.0]))
    np.testing.assert_allclose(values, [20.0, 2.0, 1.0])
