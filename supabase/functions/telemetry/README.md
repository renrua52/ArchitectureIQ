# Edge Function: `telemetry`

Preferred public ingest (no Render / no credit card). Writes to `quiz_events` via the service role.

Dashboard deploy steps: see [`../../frontend/quiz/DEPLOY.md`](../../frontend/quiz/DEPLOY.md).

**JWT verification must be OFF** — auth is `Authorization: Bearer <TELEMETRY_API_KEY>`.
