# Frontend / backend split

ArchitectureIQ product quiz development is split so FE and BE can work in
separate directories and merge with minimal conflict.

## Boundaries

| Role | Owns | Delivers |
|------|------|----------|
| **Backend (pipeline)** | `src/architecture_iq/`, `profiles/`, `tools/export_quiz_static.py`, demo bundles under `examples/` | Valid **BakeFile** JSON (`q_*` stays internal) |
| **Frontend (quiz)** | `frontend/quiz/` | UI that renders a BakeFile; optional telemetry POSTs to the hosted ingest |
| **Contracts** | `contracts/` | Schema + mini fixture + changelog — shared, dual review |
| **Streamlit inspector** | `tools/question_inspector/`, `tools/start_quiz.py` | **Frozen** — no new product features |

## Interface

- Document: [`contracts/README.md`](../contracts/README.md)
- Schema: [`contracts/quiz_bake.schema.json`](../contracts/quiz_bake.schema.json)
- Dev fixture: [`contracts/examples/mini_bake.json`](../contracts/examples/mini_bake.json)

Frontend must not import `architecture_iq` or read `data/datasets/**`.

## PR rules

1. Prefer `feat/fe-*`, `feat/be-*`, `feat/contracts-*` branch names.
2. Changing BakeFile shape: open a **contracts** PR first (schema + mini example + changelog), get FE and BE approval, then follow-up PRs.
3. Frontend CI should not require PyTorch. Backend CI should validate exports with `tools/validate_quiz_bake.py`.
4. Who changes questions updates the published bake (`frontend/quiz/public/data/questions.json` or agreed release path) and notes `collection_id` / question count in the PR.

## Local quick starts

Backend / export:

```bash
.venv/bin/python tools/export_quiz_static.py
.venv/bin/python tools/validate_quiz_bake.py
```

Frontend:

```bash
cd frontend/quiz
# optional: cp ../../contracts/examples/mini_bake.json public/data/questions.json
npm install && npm run dev
```
