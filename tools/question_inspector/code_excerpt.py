"""Code excerpt helpers (mirror of architecture_iq.prompts.code_excerpt)."""

from __future__ import annotations

import ast


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"Could not extract source for {type(node).__name__}")
    return segment.strip()


def extract_class_definitions(source: str) -> str:
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            parts.append(_source_segment(source, node))
    if not parts:
        raise ValueError("No class definitions found in source")
    return "\n\n\n".join(parts)


def extract_function_definitions(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            parts.append(_source_segment(source, node))
    if not parts:
        raise ValueError(f"No matching functions found: {names}")
    return "\n\n\n".join(parts)


def excerpt_model_py(source: str) -> str:
    parts: list[str] = []
    try:
        parts.append(extract_function_definitions(source, {"_activation", "_make_grid", "_bspline_bases"}))
    except ValueError:
        pass
    parts.append(extract_class_definitions(source))
    return "\n\n\n".join(parts)


def excerpt_loss_py(source: str) -> str:
    return extract_function_definitions(source, {"loss_fn"})


def excerpt_optimizer_py(source: str) -> str:
    return extract_function_definitions(source, {"build_optimizer"})


def excerpt_synthesize_py(source: str) -> str:
    """Return dataset synthesis definitions needed to reproduce materialized data."""
    names = {"target", "build_transition", "synthesize"}
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            parts.append(_source_segment(source, node))
    if not parts:
        raise ValueError(f"No matching functions found: {names}")
    return "\n\n\n".join(parts)
