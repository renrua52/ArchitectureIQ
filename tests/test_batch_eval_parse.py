"""Tests for the eval-side answer parser (backend/eval/batch_eval.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.eval.batch_eval import parse_answer  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("b", "B"),
        ("Answer: B", "B"),
        ("Answer: **B**", "B"),
        ("Given the refs, B seems most likely. Answer: **F**", "F"),
        ("The answer is **B**.", "B"),
        ("Pick C.", "C"),
        ("Comparing all options, B has the largest width...\nB", "B"),
        ("option C uses AdamW which is best.\nAnswer: C", "C"),
        ("", None),
        (None, None),
    ],
)
def test_parse_answer(text: str | None, expected: str | None) -> None:
    assert parse_answer(text) == expected


def test_parse_answer_prefers_final_answer_over_first_letter() -> None:
    # old parser fell back to the FIRST standalone letter ("A" from the first
    # option listed in reasoning), ignoring a bolded final answer.
    text = (
        "- A uses AdamW with LR 0.003, likely worse.\n"
        "- B is closest to the best reference.\n"
        "Answer: **B**"
    )
    assert parse_answer(text) == "B"
