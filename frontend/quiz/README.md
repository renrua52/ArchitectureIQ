# ArchitectureIQ quiz frontend

React quiz (Observe → Compare → Reveal) fed by a **static BakeFile**.

**Contract:** [`contracts/README.md`](../../contracts/README.md) · split rules:
[`docs/FRONTEND_BACKEND.md`](../../docs/FRONTEND_BACKEND.md)

This package must not import `architecture_iq` or read raw `q_*` artifact dirs.

Deployment notes / limitations: [`DEPLOY.md`](./DEPLOY.md).

## Prerequisites

1. Node 20+ recommended.
2. A BakeFile at `public/data/questions.json` (repo ships the 46-question demo bake).
   For a tiny fixture:

```bash
cp ../../contracts/examples/mini_bake.json public/data/questions.json
```

3. Optional telemetry: copy [`.env.example`](../../.env.example) → repo-root `.env`
   and set `VITE_TELEMETRY_URL` / `VITE_TELEMETRY_KEY` (hosted Edge Function or local
   FastAPI). Scoring works with telemetry omitted.

Backend-owned bake regeneration (not required for FE-only work):

```bash
# from repo root
.venv/bin/python tools/export_quiz_static.py
.venv/bin/python tools/validate_quiz_bake.py
```

## Dev

```bash
cd frontend/quiz
npm install
# optional: set -a && source ../../.env && set +a
npm run dev
```

Open http://127.0.0.1:5173/

## Env used by the browser

| Variable | Purpose |
|----------|---------|
| `VITE_TELEMETRY_URL` | Telemetry ingest base or full Edge Function URL |
| `VITE_TELEMETRY_KEY` | Bearer token for ingest POSTs |

Events fail open if ingest is down.
