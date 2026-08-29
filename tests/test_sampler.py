from __future__ import annotations



from architecture_iq.families.univariate_regression.sampler import (
    sample_symbolic_expression,
    validate_expression,
)


def test_sampler_reproducible():
    a = sample_symbolic_expression(seed=7, max_depth=3)
    b = sample_symbolic_expression(seed=7, max_depth=3)
    assert a.expression == b.expression
    assert a.retry == b.retry


def test_sampler_expression_is_numerically_nonlinear():
    # B4: nonlinearity is decided numerically (affine residual), so genuine
    # polynomials like (x-3)*(x+x)*(x-0.5) pass while abs(const) tricks fail.
    import numpy as np

    from architecture_iq.families.univariate_regression.sampler import eval_node

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
    seen = set()
    for seed in range(30):
        sampled = sample_symbolic_expression(seed=seed)
        seen.add(sampled.expression)
    assert len(seen) > 10


def test_sampler_no_dirty_decimal_constants():
    # B1: constants come from a clean table; folded constants must be clean
    # multiples of 0.5 too (PI in trig rendering is exempt).
    import re

    for seed in range(200):
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        rendered = sampled.expression.replace("6.283185307179586", "PI")
        assert not re.search(r"\d+\.\d{3,}", rendered), sampled.expression


def test_sampler_no_degenerate_patterns():
    # B3: x*0, x-x, x/1 and const-only trees are eliminated before rendering.
    from architecture_iq.families.univariate_regression.sampler import (
        NodeKind,
        _is_const_only,
        _trees_equal,
    )

    def check(node):
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.kind == NodeKind.MUL:
                for side in (cur.left, cur.right):
                    assert not (
                        side is not None
                        and side.kind == NodeKind.CONST
                        and float(side.value) == 0.0
                    ), f"zero-multiplication in {node!r}"
            if cur.kind == NodeKind.SUB:
                assert not _trees_equal(cur.left, cur.right), f"x-x in {node!r}"
            if cur.kind == NodeKind.DIV:
                assert not (
                    cur.right is not None
                    and cur.right.kind == NodeKind.CONST
                    and float(cur.right.value) == 1.0
                ), f"x/1 in {node!r}"
            assert not _is_const_only(cur) or cur.kind == NodeKind.CONST, (
                f"unfolded const-only subtree in {node!r}"
            )
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)

    for seed in range(200):
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        check(sampled.tree)


def test_sampler_div_guard_never_triggers_clamp():
    # B2: every DIV denominator stays above the 0.1 clamp floor on the whole
    # domain, so the rendered formula equals the executed target exactly.
    import numpy as np

    from architecture_iq.families.univariate_regression.sampler import (
        NodeKind,
        eval_node,
    )

    xs = np.linspace(0.0, 1.0, 4096)
    for seed in range(200):
        sampled = sample_symbolic_expression(seed=seed, max_depth=3)
        stack = [sampled.tree]
        while stack:
            node = stack.pop()
            if node.kind == NodeKind.DIV:
                denom = eval_node(node.right, xs)
                assert float(np.min(np.abs(denom))) >= 0.1 - 1e-12, sampled.expression
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
