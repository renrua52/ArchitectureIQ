from __future__ import annotations

import ast


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"Could not extract source for {type(node).__name__}")
    return segment.strip()


def extract_class_definitions(source: str) -> str:
    """Return all top-level class definitions (e.g. nn.Module subclasses)."""
    tree = ast.parse(source)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            parts.append(_source_segment(source, node))
    if not parts:
        raise ValueError("No class definitions found in source")
    return "\n\n\n".join(parts)


def extract_function_definitions(source: str, names: set[str]) -> str:
    """Return top-level function definitions whose names are in *names*."""
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
        parts.append(extract_function_definitions(source, {"_activation"}))
    except ValueError:
        pass
    parts.append(extract_class_definitions(source))
    return "\n\n\n".join(parts)


def excerpt_loss_py(source: str) -> str:
    return extract_function_definitions(source, {"loss_fn"})


def excerpt_optimizer_py(source: str) -> str:
    return extract_function_definitions(source, {"build_optimizer"})


def excerpt_synthesize_py(source: str) -> str:
    return extract_function_definitions(source, {"target"})
