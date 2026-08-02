# BakeFile changelog

## schema_version 1 — 2026-07-24

Initial frozen contract extracted from the React quiz BakeFile and
`tools/export_quiz_static.py` output.

- Top level: `schema_version`, `questions`, `byId`; optional `ordered`, `collection`
- Per question: `detail` (prompt / shared / dataset / choices) + `reveal`
- Question ids match `q_[0-9a-f]+`
- Fixture: `examples/mini_bake.json` (4 questions, one per registered demo family)

Breaking changes after this point must bump `schema_version` and add a new section here.
