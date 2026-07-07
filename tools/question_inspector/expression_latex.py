"""Convert dataset infix expressions to LaTeX for the inspector UI."""

from __future__ import annotations

import re

TWO_PI = "6.283185307179586"

_TRANSFORMATIONS = None


def _transformations():
    global _TRANSFORMATIONS
    if _TRANSFORMATIONS is None:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            standard_transformations,
        )

        _TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
    return _TRANSFORMATIONS


def _prepare_for_sympy(expression: str) -> str:
    text = expression.strip()
    text = text.replace(TWO_PI, "2*pi")
    text = re.sub(r"\babs\b", "Abs", text)
    # x0, x1, ... must become x_0, x_1 before implicit-multiplication parsing (else x0 -> x*0).
    text = re.sub(r"\bx(\d+)\b", r"x_\1", text)
    return text


def expression_to_latex(expression: str) -> str:
    """Render a dataset expression string as LaTeX."""
    if not expression or expression == "—":
        return r"\text{—}"

    from sympy import latex
    from sympy.parsing.sympy_parser import parse_expr

    try:
        parsed = parse_expr(_prepare_for_sympy(expression), transformations=_transformations())
        return latex(parsed)
    except Exception:
        escaped = (
            expression.replace("\\", r"\\textbackslash ")
            .replace("_", r"\_")
            .replace("^", r"\^{}")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )
        return rf"\texttt{{{escaped}}}"
