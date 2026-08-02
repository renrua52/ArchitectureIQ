from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from architecture_iq import cli


def test_generate_candidates_omits_empty_model_quota(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate_candidate_set(*args: Any, **kwargs: Any) -> Path:
        captured.update(kwargs)
        return tmp_path / "set"

    monkeypatch.setattr(cli, "generate_candidate_set", fake_generate_candidate_set)
    result = CliRunner().invoke(
        cli.app,
        [
            "generate-candidates",
            str(tmp_path / "dataset"),
            "--profile",
            "v2.4-gru-architecture-pilot",
            "--budget",
            "5120",
            "--count",
            "2",
            "--vary",
            "model",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model_type_counts"] is None


def test_generate_candidates_forwards_model_quotas(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate_candidate_set(*args: Any, **kwargs: Any) -> Path:
        captured.update(kwargs)
        return tmp_path / "set"

    monkeypatch.setattr(cli, "generate_candidate_set", fake_generate_candidate_set)
    result = CliRunner().invoke(
        cli.app,
        [
            "generate-candidates",
            str(tmp_path / "dataset"),
            "--profile",
            "v2.4-gru-architecture-pilot",
            "--budget",
            "5120",
            "--count",
            "2",
            "--vary",
            "model",
            "--model-type-count",
            "transformer_lm=1",
            "--model-type-count",
            "gru_lm=1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model_type_counts"] == {"transformer_lm": 1, "gru_lm": 1}