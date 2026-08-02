"""BakeFile contract checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "contracts" / "quiz_bake.schema.json"
MINI = ROOT / "contracts" / "examples" / "mini_bake.json"
DEMO = ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def _validate(path: Path, schema: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    # catalog <-> byId
    assert set(payload["byId"]) >= {row["id"] for row in payload["questions"]}


def test_mini_bake_matches_schema(schema: dict) -> None:
    _validate(MINI, schema)


def test_demo_bake_matches_schema(schema: dict) -> None:
    if not DEMO.is_file():
        pytest.skip("demo bake missing")
    _validate(DEMO, schema)
