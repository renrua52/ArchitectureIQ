from __future__ import annotations

from pathlib import Path

from architecture_iq.prompts.formatters import (
    format_mlp_nl,
    format_training_schedule,
)
from architecture_iq.prompts.renderer import render_prompt


def test_format_mlp_nl_lists_activations() -> None:
    text = format_mlp_nl(
        {
            "depth": 2,
            "width": 64,
            "residual": False,
            "layer_norm": [True, True],
            "activations": ["leaky_relu", "gelu"],
        }
    )
    assert "- Activations: [leaky_relu, gelu]" in text


def test_format_training_schedule_includes_product() -> None:
    text = format_training_schedule(
        {"training_steps": 8, "batch_size": 64, "total_samples_seen": 512}
    )
    assert "training_steps: 8" in text
    assert "batch_size: 64" in text
    assert "training_steps × batch_size" in text


def test_render_prompt_includes_reproduction_protocol() -> None:
    q_path = Path("data/questions/q_547c83")
    if not q_path.exists():
        return
    text = render_prompt(q_path)
    assert "Data splits and training protocol" in text
    assert "Target expression (canonical)" in text
    assert "with replacement" in text
    assert "Ground-truth ranking" in text or "Evaluation metric" in text
    assert "Activations: [" in text
    assert "def _activation" in text
    assert "## Sample budget (same for all choices)" in text


def test_render_prompt_matches_prompt_txt() -> None:
    q_path = Path("data/questions/q_547c83")
    if not q_path.exists():
        return
    from architecture_iq.util import read_json

    q = read_json(q_path / "question.json")
    if "evaluation" not in q:
        return
    rendered = render_prompt(q_path)
    on_disk = (q_path / "prompt.txt").read_text(encoding="utf-8")
    assert rendered == on_disk
