# ArchitectureIQ public quiz frontend

Staged React quiz (Observe → Compare → Reveal) fed by a **static bake** of
question artifacts. Telemetry posts to a **Supabase Edge Function** (preferred)
or local FastAPI ingest.

Public ship steps: [`DEPLOY.md`](./DEPLOY.md).

## Prerequisites

1. Apply [`services/telemetry_api/schema.sql`](../../services/telemetry_api/schema.sql) in Supabase.
2. Deploy Edge Function `telemetry` (see DEPLOY.md) **or** run local FastAPI.
3. Copy [`.env.example`](../../.env.example) → repo-root `.env` and fill secrets.
3. Bake questions:

```bash
# from repo root
.venv/bin/python tools/export_quiz_static.py
# optional: only univariate run
.venv/bin/python tools/export_quiz_static.py --run run_20q_3c_b09206
```

## Dev

Terminal A — telemetry ingest:

```bash
pip install -r services/telemetry_api/requirements.txt
set -a && source .env && set +a
uvicorn services.telemetry_api.app:app --host 127.0.0.1 --port 8080
```

Terminal B — frontend (load Vite env from root `.env` by exporting `VITE_*`):

```bash
cd frontend/quiz
npm install
set -a && source ../../.env && set +a
npm run dev
```

Open http://127.0.0.1:5173/

## Env used by the browser

| Variable | Purpose |
|----------|---------|
| `VITE_TELEMETRY_URL` | Full Edge Function URL, **or** FastAPI base `http://127.0.0.1:8080` |
| `VITE_TELEMETRY_KEY` | Same value as `TELEMETRY_API_KEY` |

Quiz scoring works without telemetry; events fail open if the API is down.
