# Public deploy checklist

Architecture: **static quiz (Pages/Netlify)** + **Supabase Edge Function ingest** + **Postgres (`quiz_events`)**.

No Render / no credit card required for ingest.

---

## 1. Supabase table (you likely finished)

In **SQL Editor**, run [`services/telemetry_api/schema.sql`](../../services/telemetry_api/schema.sql).

Confirm **Table Editor → `quiz_events`** exists.

---

## 2. Deploy Edge Function `telemetry` (Dashboard)

### 2.1 Create the function

1. Open [Supabase Dashboard](https://supabase.com/dashboard) → your project.
2. Left sidebar → **Edge Functions**.
3. **Deploy a new function** / **Create function**.
4. Name: `telemetry` (exact).
5. Paste the full contents of
   [`supabase/functions/telemetry/index.ts`](../../supabase/functions/telemetry/index.ts)
   into the editor (or use CLI below).
6. **Critical:** turn **Enforce JWT verification** / **Verify JWT** **OFF**.
   Auth is our own `TELEMETRY_API_KEY`, not a Supabase user JWT.
7. Deploy.

Public URL will look like:

```text
https://YOUR_PROJECT_REF.supabase.co/functions/v1/telemetry
```

(`YOUR_PROJECT_REF` = Project Settings → General → Reference ID, same host as the project URL.)

### 2.2 Set secrets (Dashboard)

Still under **Edge Functions** → **Secrets** (project-wide) / **Manage secrets**:

| Secret name | Value |
|-------------|--------|
| `TELEMETRY_API_KEY` | Same random string as in your local `.env` (and later `VITE_TELEMETRY_KEY`) |
| `CORS_ORIGINS` | For now: `http://127.0.0.1:5173,http://localhost:5173` — after the static site is live, add `https://your-site.pages.dev` |

Do **not** paste `DATABASE_URL` or service role into the browser env.
`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are injected automatically by Supabase for Edge Functions.

### 2.3 Smoke test

```bash
export TELEMETRY_API_KEY='your-key-from-local-env'
export FN='https://YOUR_PROJECT_REF.supabase.co/functions/v1/telemetry'

# health
curl -s "$FN"

# insert one event
curl -s -i -X POST "$FN" \
  -H "Authorization: Bearer $TELEMETRY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"edgetest0001","event_type":"session_start","payload":{"schema_version":1}}'
```

Expect HTTP **202** and body `{"accepted":1}`, plus a new row in **`quiz_events`**.

Common failures:

| Symptom | Fix |
|---------|-----|
| 401 | Wrong `TELEMETRY_API_KEY`, or JWT verify still ON |
| CORS error in browser later | Add the site origin to `CORS_ORIGINS` secret, redeploy not always required for secrets—wait a few seconds and retry |
| 500 Database error | Table missing / typo; re-run `schema.sql` |

### Optional: deploy via CLI instead of paste

```bash
# once: npm i -g supabase  OR  brew install supabase/tap/supabase
supabase login
supabase link --project-ref YOUR_PROJECT_REF
supabase secrets set TELEMETRY_API_KEY='...' CORS_ORIGINS='http://127.0.0.1:5173,http://localhost:5173'
supabase functions deploy telemetry --no-verify-jwt
```

---

## 3. Point local frontend at the Edge Function

In repo-root `.env`:

```bash
VITE_TELEMETRY_URL=https://YOUR_PROJECT_REF.supabase.co/functions/v1/telemetry
VITE_TELEMETRY_KEY=same-as-TELEMETRY_API_KEY
```

Restart `npm run dev`. Answer one question → Network POST → **202** → row in Supabase.

(Local FastAPI still works if `VITE_TELEMETRY_URL=http://127.0.0.1:8080` — client appends `/api/telemetry/events`.)

---

## 4. Deploy static frontend (Cloudflare Pages or Netlify)

1. Link GitHub repo.
2. Root directory: `frontend/quiz`
3. Build: `npm install && npm run build`
4. Output: `dist`
5. Build env:

| Name | Value |
|------|--------|
| `VITE_TELEMETRY_URL` | `https://YOUR_PROJECT_REF.supabase.co/functions/v1/telemetry` |
| `VITE_TELEMETRY_KEY` | same as `TELEMETRY_API_KEY` |

6. Deploy → copy site URL, e.g. `https://xxx.pages.dev`

---

## 5. Update CORS after the site is live

In Supabase → Edge Functions → Secrets, set:

```text
CORS_ORIGINS=https://xxx.pages.dev,http://127.0.0.1:5173,http://localhost:5173
```

Retest from the public site: Network → POST → **202**.

---

## 6. End-to-end check

1. Open public quiz, answer one question.
2. Table Editor → `quiz_events`: `session_start`, `question_view`, `answer_submit`, …
3. Browser never sees the DB password or service role key.

---

## Notes

- FastAPI under `services/telemetry_api/` remains useful for **local** smoke tests without deploying a function.
- Rotate `TELEMETRY_API_KEY` if it leaks; update the Edge Secret + rebuild frontend `VITE_TELEMETRY_KEY`.
