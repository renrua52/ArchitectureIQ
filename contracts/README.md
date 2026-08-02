# contracts/ — BakeFile interface

This directory is the **only agreed exchange format** between the question
pipeline (backend) and the React quiz (frontend).

## What lives here

| Path | Purpose |
|------|---------|
| [`quiz_bake.schema.json`](./quiz_bake.schema.json) | JSON Schema for a BakeFile (`schema_version: 1`) |
| [`examples/mini_bake.json`](./examples/mini_bake.json) | Small 4-question fixture (one per dataset family) |
| [`CHANGELOG.md`](./CHANGELOG.md) | Schema version history |

Telemetry POST shapes and optional quiz HTTP APIs are **out of scope** here for
now (telemetry is a separate hosted ingest; static bake is enough for the quiz).

## Rules

1. Frontend consumes a **BakeFile** only. It must not read `data/datasets/**/q_*`
   directories or import `architecture_iq`.
2. Backend may keep raw `q_xxxxxx` artifacts internally, but must **export** a
   BakeFile that validates against this schema (see `tools/export_quiz_static.py`).
3. Contract changes land in `contracts/` first (schema + mini example + changelog),
   with frontend **and** backend review, before either side ships incompatible code.
4. Additive optional fields may keep `schema_version: 1` if old clients can ignore
   them. Removing/renaming/changing meaning of required fields requires a version bump.

## Validate

```bash
# from repo root
.venv/bin/python -m pip install -e ".[dev]"   # includes jsonschema
.venv/bin/python tools/validate_quiz_bake.py
.venv/bin/python tools/validate_quiz_bake.py frontend/quiz/public/data/questions.json
```

## Ownership

See [`.github/CODEOWNERS`](../.github/CODEOWNERS). Update GitHub usernames when
FE/BE owners are assigned. Collaboration norms:
[`docs/FRONTEND_BACKEND.md`](../docs/FRONTEND_BACKEND.md).
