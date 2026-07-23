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
