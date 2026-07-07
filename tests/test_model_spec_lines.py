from __future__ import annotations

import sys
from pathlib import Path

from architecture_iq.prompts.formatters import format_model_spec_lines

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))
import prompt_format as insp  # noqa: E402


def test_transformer_lm_spec_lines() -> None:
    model = {
        "type": "transformer_lm",
        "vocab_size": 32,
        "context_length": 16,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
    }
    lines = format_model_spec_lines(model)
    text = "\n".join(lines)
    assert "d_model: 64" in text
    assert "num_layers: 2" in text
    assert "num_heads: 4" in text
    assert "d_ff: 128" in text
    assert "dropout" not in text.lower()
    assert "Depth" not in text
    assert "None" not in text


def test_transformer_lm_spec_lines_legacy_keys() -> None:
    model = {
        "type": "transformer_lm",
        "vocab_size": 32,
        "context_length": 16,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
    }
    lines = format_model_spec_lines(model)
    assert "d_model: 64" in "\n".join(lines)
    assert "d_ff: 128" in "\n".join(lines)


def test_inspector_transformer_lm_spec_lines_match_package() -> None:
    model = {
        "type": "transformer_lm",
        "vocab_size": 32,
        "context_length": 16,
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 128,
    }
    assert insp.format_model_spec_lines(model) == format_model_spec_lines(model)
