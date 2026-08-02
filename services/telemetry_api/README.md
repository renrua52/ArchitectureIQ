# Telemetry ingest API

Thin FastAPI service for ArchitectureIQ quiz analytics (**local / optional**).

The current telemetry design is for local/internal use. The experimental
Supabase Edge Function is not approved for external deployment because a
browser-visible `VITE_*` value cannot be used as a server credential. See
[`frontend/quiz/DEPLOY.md`](../../frontend/quiz/DEPLOY.md) for the required
hardening before revisiting that path.

## Local

```bash
# from repo root
python -m venv .venv-telemetry
source .venv-telemetry/bin/activate
pip install -r services/telemetry_api/requirements.txt

export DATABASE_URL='postgresql://...'
export TELEMETRY_API_KEY='...'
export CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'

uvicorn services.telemetry_api.app:app --host 127.0.0.1 --port 8080
```

Or run as a module path from `services/telemetry_api`:

```bash
cd services/telemetry_api
uvicorn app:app --host 127.0.0.1 --port 8080
```

## Schema

Apply [`schema.sql`](./schema.sql) in the Supabase SQL Editor once.

## Env

| Variable | Required | Purpose |
|----------|----------|---------|
| `DATABASE_URL` | yes | Postgres URI (Supabase connection string) |
| `TELEMETRY_API_KEY` | yes | Bearer token for POST |
| `CORS_ORIGINS` | no | Comma-separated origins (default localhost Vite) |
| `TELEMETRY_RATE_LIMIT_PER_MINUTE` | no | Default 120 |

## Smoke test

```bash
curl -s -X POST "http://127.0.0.1:8080/api/telemetry/events" \
  -H "Authorization: Bearer $TELEMETRY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"testsession01","event_type":"session_start","payload":{"schema_version":1}}'
```

Then check the `quiz_events` table in Supabase.

The quiz also accepts `audit_feedback` events. Their payload records
`confidence` (1–5), `decision` (`keep`, `revise`, or `reject`), the selected
answer, and question provenance (`track`, `profile`, `profile_hash`).
