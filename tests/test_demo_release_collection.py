"""Regression checks for the tracked demo-release collection contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_demo_release_collection.py"
SPEC = importlib.util.spec_from_file_location("build_demo_release_collection", MODULE_PATH)
assert SPEC and SPEC.loader
release_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_builder)


def test_release_spec_cannot_escape_its_data_root() -> None:
    with pytest.raises(ValueError, match="stay inside data_root"):
        release_builder.build(
            [{"source_run": "../outside", "track": "test"}],
            title="test release",
            data_root=ROOT / "data",
            release_spec=True,
        )