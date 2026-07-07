from __future__ import annotations

from architecture_iq.prompts.formatters import format_training_schedule
from architecture_iq.prompts.renderer import _question_total_samples_seen


def test_question_total_samples_seen_dict() -> None:
    assert _question_total_samples_seen({"total_samples_seen": 512}) == 512


def test_question_total_samples_seen_int() -> None:
    assert _question_total_samples_seen(1024) == 1024


def test_format_training_schedule() -> None:
    text = format_training_schedule(
        {"training_steps": 32, "batch_size": 16, "total_samples_seen": 512}
    )
    assert "training_steps: 32" in text
    assert "batch_size: 16" in text
    assert "total_samples_seen: 512" in text
