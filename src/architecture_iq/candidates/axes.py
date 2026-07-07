"""Axis inference and compatibility for candidate sets and questions."""

from __future__ import annotations

import json
from typing import Any

CANDIDATE_AXES = ("model", "optimizer", "loss", "batch_size")

SINGLE_AXIS_TYPES = frozenset({"architecture_only", "optimizer_only", "loss_only"})

VARYING_AXIS_CHOICES = frozenset({"model", "optimizer", "loss"})


def spec_axis_json(spec: dict[str, Any], axis: str) -> str:
    if axis == "batch_size":
        return json.dumps(spec["budget"]["batch_size"], sort_keys=True)
    return json.dumps(spec[axis], sort_keys=True)


def infer_axes(specs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return (invariant_axes, varying_axes) comparing candidate specs."""
    invariant: list[str] = []
    varying: list[str] = []
    for axis in CANDIDATE_AXES:
        values = {spec_axis_json(spec, axis) for spec in specs}
        if len(values) == 1:
            invariant.append(axis)
        else:
            varying.append(axis)
    return invariant, varying


def choices_have_contrast(specs: list[dict[str, Any]]) -> bool:
    """True when at least one axis varies across specs."""
    _, varying = infer_axes(specs)
    return bool(varying)


def infer_question_type(specs: list[dict[str, Any]]) -> str:
    """Map varying axes to a legacy question type label for prompts."""
    _, varying = infer_axes(specs)
    training_varying = frozenset(a for a in varying if a in VARYING_AXIS_CHOICES)
    if training_varying == {"model"}:
        return "architecture_only"
    if training_varying == {"optimizer"}:
        return "optimizer_only"
    if training_varying == {"loss"}:
        return "loss_only"
    return "mixed"


def choices_compatible(specs: list[dict[str, Any]], question_type: str | None = None) -> bool:
    """Return whether specs form a valid comparison set."""
    if len(specs) < 2:
        return False
    if not choices_have_contrast(specs):
        return False
    if question_type is None:
        return True
    if question_type == "mixed":
        return True
    if question_type not in SINGLE_AXIS_TYPES:
        raise ValueError(f"Unknown question type: {question_type}")
    if infer_question_type(specs) != question_type:
        return False
    _, varying = infer_axes(specs)
    return "batch_size" not in varying
