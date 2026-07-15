# Public deploy checklist

Architecture: **static quiz (Pages)** + **thin ingest API (Render)** + **Postgres (Supabase)**.

## 0. Repo prep (already done in code)

- [x] `/data/` gitignore is root-only; `frontend/quiz/public/data/questions.json` can be committed
- [ ] Commit and push: frontend, telemetry service, baked `questions.json`

```bash
# from repo root — refresh bake before push if needed
.venv/bin/python tools/export_quiz_static.py
git add .gitignore frontend/quiz services/telemetry_api tools/export_quiz_static.py
git add frontend/quiz/public/data/questions.json
# …plus any other quiz-related files you intend to ship
git status   # confirm .env is NOT staged
```

Never commit `.env`.

## 1. Supabase (you likely finished this)

- [x] Project created  
- [x] Run [`services/telemetry_api/schema.sql`](../../services/telemetry_api/schema.sql)  
- [x] Keep `DATABASE_URL` private (URL-encoded password + `?sslmode=require`)

Check data later in **Table Editor → `quiz_events`**.

## 2. Deploy ingest API (Render — required for回流)

1. Sign up at [render.com](https://render.com), connect the GitHub repo.
2. **New → Web Service**
   - Runtime: Python
   - Build: `pip install -r services/telemetry_api/requirements.txt`
   - Start: `uvicorn services.telemetry_api.app:app --host 0.0.0.0 --port $PORT`
3. Environment variables:

| Name | Value |
|------|--------|
| `DATABASE_URL` | Supabase URI |
| `TELEMETRY_API_KEY` | same as local `.env` |
| `CORS_ORIGINS` | temporary: `http://127.0.0.1:5173` — replace after Pages URL exists |

4. Save the public URL, e.g. `https://YOUR-SERVICE.onrender.com`
5. Smoke test:

```bash
curl -s -X POST "https://YOUR-SERVICE.onrender.com/api/telemetry/events" \
  -H "Authorization: Bearer $TELEMETRY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"pubtest0001","event_type":"session_start","payload":{"schema_version":1}}'
```

Expect `{"accepted":1}` and a new row in Supabase.

## 3. Deploy static frontend (Cloudflare Pages or Netlify)

1. New project → link same GitHub repo  
2. Root directory: `frontend/quiz`  
3. Build command: `npm install && npm run build`  
4. Output directory: `dist`  
5. Build environment variables:

| Name | Value |
|------|--------|
| `VITE_TELEMETRY_URL` | `https://YOUR-SERVICE.onrender.com` (no trailing slash) |
| `VITE_TELEMETRY_KEY` | same as `TELEMETRY_API_KEY` |

6. Deploy → copy site URL, e.g. `https://xxx.pages.dev`

## 4. Wire CORS

In Render, set:

```text
CORS_ORIGINS=https://xxx.pages.dev,http://127.0.0.1:5173
```

Redeploy / restart the ingest service.

## 5. Verify end-to-end

1. Open the public site, answer one question  
2. Supabase `quiz_events` shows `session_start`, `question_view`, `answer_submit`  
3. Browser Network tab: POST to ingest returns **202**

## Optional later

- Custom domain (Pages + DNS; optionally `api.yourdomain.com` → Render)  
- Upgrade Render plan if cold starts are annoying  
- Rotate `TELEMETRY_API_KEY` if it leaks
