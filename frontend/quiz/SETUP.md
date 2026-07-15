# ArchitectureIQ quiz frontend + telemetry

## Quick start (local)

1. Put secrets in repo-root `.env` (see `.env.example`). **Never commit `.env`.**
2. In Supabase SQL Editor, run [`services/telemetry_api/schema.sql`](services/telemetry_api/schema.sql).
3. Bake demo questions:

```bash
.venv/bin/python tools/export_quiz_static.py
```

4. Start ingest + UI:

```bash
# terminal 1
set -a && source .env && set +a
.venv/bin/pip install -r services/telemetry_api/requirements.txt
.venv/bin/uvicorn services.telemetry_api.app:app --host 127.0.0.1 --port 8080

# terminal 2
cd frontend/quiz && npm install
set -a && source ../../.env && set +a
npm run dev
```

Open http://127.0.0.1:5173/

## DATABASE_URL tips (Supabase)

Use **Project Settings → Database → Connection string → URI**.

- Must look like:  
  `postgresql://postgres:PASSWORD@db.<ref>.supabase.co:5432/postgres?sslmode=require`
- If the password contains `@ : / # ?` etc., **URL-encode** it (e.g. `@` → `%40`).
- Do not paste the Project URL (`https://….supabase.co`) as `DATABASE_URL`.

## Layout

| Path | Role |
|------|------|
| `frontend/quiz/` | Staged React quiz (static bake) |
| `tools/export_quiz_static.py` | Bake `questions.json` from quiz_demo/data |
| `services/telemetry_api/` | Thin ingest → Postgres `quiz_events` |
