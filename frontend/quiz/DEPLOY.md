# Deployment status

The quiz is currently supported for **local and internal demonstration only**.
It is not a public-deployment checklist or a release runbook.

Do not deploy the present static BakeFile or set `VITE_TELEMETRY_KEY` on a
hosted site:

- the BakeFile currently contains answer-reveal fields for internal review;
- every `VITE_*` value is bundled into browser JavaScript, so it cannot be a
  server-side telemetry credential.

Before any external deployment, the project needs all of the following:

1. A public-only question schema and an automated check that static assets and
   GET responses contain no reveal/correct-answer fields.
2. API-only answer reveal after a recorded submission.
3. A browser-safe telemetry authorization design (for example short-lived,
   scoped tokens or authenticated users), plus rate limiting and key rotation.
4. A reproducible, versioned distribution path for the frozen question/data
   assets.

For current local setup, use [`SETUP.md`](./SETUP.md) and the local FastAPI
telemetry service in [`services/telemetry_api/`](../../services/telemetry_api/).

## User system + session recordings (2026-09)

The user/recording backend addresses hardening point 3 for its own surface:

- **Schema**: [`supabase/schema.sql`](../../supabase/schema.sql) — three tables
  (`quiz_users`, `quiz_sessions`, `recording_chunks`) with RLS enabled and
  zero anon policies; the browser-facing anon key cannot read or write any
  table directly (verified by `frontend/quiz/e2e_recording_test.py` + manual
  REST probes).
- **Browser surface**: exactly three SECURITY DEFINER RPCs.
  `quiz_register(username, shared_password)` issues a per-user upload token;
  the shared password is verified server-side and never ships to the browser.
  `quiz_upsert_session` / `quiz_ingest_chunk` require that token.
- **Reads**: no browser read path at all. Users export their own session
  (results + full recording) locally via the Export button; admins pull
  everything over a direct Postgres connection with
  [`tools/export_recordings.py`](../../tools/export_recordings.py).
- **Env**: `VITE_SUPABASE_URL` + `VITE_SUPABASE_ANON_KEY` in
  `.env.production.local` (gitignored). The anon key is public-by-design;
  all authorization happens inside the RPCs.
- **Recorder**: `src/recorder.ts` captures pointer moves (~16 Hz), clicks,
  scrolls, tab-visibility switches and quiz events as permille-of-viewport
  coordinates; uploads in 10 s chunks with a localStorage-backed retry
  queue; `keepalive` fetch on pagehide. `src/replay.tsx` replays any
  exported recording file in-app.
