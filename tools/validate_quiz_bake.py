#!/usr/bin/env python3
"""Validate BakeFile JSON documents against contracts/quiz_bake.schema.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "contracts" / "quiz_bake.schema.json"
DEFAULT_TARGETS = [
    ROOT / "contracts" / "examples" / "mini_bake.json",
    ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json",
]


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _cross_check(payload: dict) -> list[str]:
    """Semantic checks beyond JSON Schema."""
    errors: list[str] = []
    questions = payload.get("questions")
    by_id = payload.get("byId")
    if not isinstance(questions, list) or not isinstance(by_id, dict):
        return errors
    catalog_ids = []
    for index, row in enumerate(questions):
        if not isinstance(row, dict):
            errors.append(f"questions[{index}] is not an object")
            continue
        qid = row.get("id")
        catalog_ids.append(qid)
        if qid not in by_id:
            errors.append(f"questions[{index}].id={qid!r} missing from byId")
    for qid, item in by_id.items():
        if not isinstance(item, dict):
            errors.append(f"byId[{qid!r}] is not an object")
            continue
        if item.get("id") != qid:
            errors.append(f"byId key {qid!r} != question.id {item.get('id')!r}")
        detail = item.get("detail") or {}
        choices = detail.get("choices") if isinstance(detail, dict) else None
        letters = set()
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and "letter" in choice:
                    letters.add(choice["letter"])
        reveal = item.get("reveal") or {}
        correct = reveal.get("correctLetter") if isinstance(reveal, dict) else None
        if correct is not None and letters and correct not in letters:
            errors.append(f"{qid}: reveal.correctLetter {correct!r} not in choice letters {sorted(letters)}")
    missing_catalog = [qid for qid in by_id if qid not in catalog_ids]
    if missing_catalog:
        errors.append(f"byId has ids not listed in questions: {missing_catalog[:5]}")
    return errors


def validate_file(path: Path, schema: dict) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "jsonschema is required. Install with: pip install -e '.[dev]'"
        ) from exc

    payload = _load(path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = [
        f"{e.json_path}: {e.message}" if hasattr(e, "json_path") else f"{list(e.absolute_path)}: {e.message}"
        for e in sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    ]
    if isinstance(payload, dict):
        errors.extend(_cross_check(payload))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="BakeFile JSON paths (default: mini bake + public demo bake)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="JSON Schema path",
    )
    args = parser.parse_args()
    schema_path = args.schema.resolve()
    if not schema_path.is_file():
        raise SystemExit(f"Schema not found: {schema_path}")
    schema = _load(schema_path)
    targets = [p.resolve() for p in args.paths] if args.paths else list(DEFAULT_TARGETS)
    failed = 0
    for path in targets:
        if not path.is_file():
            print(f"SKIP {path} (missing)")
            continue
        errors = validate_file(path, schema)
        if errors:
            failed += 1
            print(f"FAIL {path} ({len(errors)} issue(s))")
            for line in errors[:40]:
                print(f"  - {line}")
            if len(errors) > 40:
                print(f"  … {len(errors) - 40} more")
        else:
            print(f"OK   {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
