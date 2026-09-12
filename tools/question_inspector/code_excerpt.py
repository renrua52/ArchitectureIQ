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


def extract_module_constants(source: str) -> str:
    """Return top-level constant assignments, in source order.

    No current renderer emits module constants -- the MLP writes its activation
    and its skip directly into the class bodies -- but artifacts generated
    before that (and any future renderer that hoists a shared value) do, and a
    class body referencing a dropped constant would be an excerpt that cannot
    run.
    """
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            parts.append(_source_segment(source, node))
    return "\n".join(parts)


def excerpt_model_py(source: str) -> str:
    parts: list[str] = []
    constants = extract_module_constants(source)
    if constants:
        parts.append(constants)
    try:
        # Helper names only legacy artifacts carry; the current MLP renderer has
        # no _activation indirection, so this simply finds nothing.
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
