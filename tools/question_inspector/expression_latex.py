"""Convert dataset infix expressions to LaTeX for the inspector UI."""

from __future__ import annotations

from html import escape, unescape
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


def expression_to_mathml(expression: str) -> str:
    """Render a dataset expression as self-contained presentation MathML.

    MathML lets the static exporter show formulas offline without fetching a
    JavaScript renderer such as MathJax or KaTeX.
    """
    if not expression or expression == "—":
        return '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>—</mtext></math>'

    from sympy import mathml
    from sympy.parsing.sympy_parser import parse_expr

    try:
        parsed = parse_expr(_prepare_for_sympy(expression), transformations=_transformations())
        body = unescape(mathml(parsed, printer="presentation"))
    except Exception:
        body = f"<mtext>{escape(expression)}</mtext>"
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{body}</math>'
