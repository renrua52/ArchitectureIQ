"""Stable question versioning shared by quiz bundle tooling.

Keep this implementation in sync with ``question_inspector/feedback.py``.
It intentionally has no reverse import into the inspector so publishing stays
usable as a small, standard-library-only maintenance tool.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


QUESTION_VERSION_ALGORITHM = "qv1"
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


class QuestionVersionError(ValueError):
    """Raised when a question cannot be represented as canonical JSON."""


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(item)
            for key, item in value.items()
            if not str(key).startswith("_inspector_")
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _contains_unicode_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_json_interoperability(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str):
            if _contains_unicode_surrogate(item):
                raise QuestionVersionError(
                    "question cannot contain Unicode surrogate code points"
                )
            continue
        if isinstance(item, int):
            if not -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER:
                raise QuestionVersionError(
                    "question integer-valued JSON numbers must be within the "
                    "JavaScript safe-integer range"
                )
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise QuestionVersionError(
                    "question must contain only finite JSON values"
                )
            if item.is_integer() and not (
                -MAX_SAFE_JSON_INTEGER <= item <= MAX_SAFE_JSON_INTEGER
            ):
                raise QuestionVersionError(
                    "question integer-valued JSON numbers must be within the "
                    "JavaScript safe-integer range"
                )
            continue
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def normalize_question(question: Mapping[str, Any]) -> dict[str, Any]:
    """Return the detached canonical JSON value used for question hashing."""
    if not isinstance(question, Mapping):
        raise QuestionVersionError("question must be a mapping")
    try:
        encoded = json.dumps(
            _normalize_value(question),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise QuestionVersionError(f"question is not canonical JSON: {exc}") from exc
    if not isinstance(
        normalized, dict
    ):  # Defensive; Mapping always normalizes to dict.
        raise QuestionVersionError("question must normalize to a JSON object")
    _validate_json_interoperability(normalized)
    return normalized


def compute_question_version(question: Mapping[str, Any]) -> str:
    """Hash normalized question JSON into an algorithm-tagged version."""
    canonical = json.dumps(
        normalize_question(question),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{QUESTION_VERSION_ALGORITHM}_{digest}"


question_version = compute_question_version
