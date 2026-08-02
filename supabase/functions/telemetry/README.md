# Edge Function: `telemetry`

Experimental telemetry ingest for internal development. It is not enabled as a
supported external-deployment path.

The current browser-token authentication scheme is unsuitable for a hosted
site: `VITE_*` values are visible to every browser. Before this function is
used outside a trusted internal environment, replace that scheme with
browser-safe authorization and complete the conditions in
[`frontend/quiz/DEPLOY.md`](../../../frontend/quiz/DEPLOY.md).
