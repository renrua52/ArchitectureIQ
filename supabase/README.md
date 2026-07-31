# ArchitectureIQ feedback ingestion and reports on Supabase

This directory is the deployable receiver for the question inspector's existing
`session_trace` POST contract. It stores one append-only row per event and works
for every inspector upload path:

- **Upload comment** sends a one-event `session_trace` envelope.
- **Surprised / As expected** sends its immutable reaction as a one-event
  envelope when transport is configured; otherwise it stays pending.
- **Upload all session events** sends the same envelope with multiple events.

Both paths use `event_id` as the durable idempotency key. The Edge Function calls
the service-role-only `feedback_ingest_events` RPC; it no longer inserts directly
through the `feedback_events` REST table. The RPC compares the stored and incoming
logical event as exact JSONB over seven fields: `schema_version`, `event_id`,
`event_type`, `session_id`, `question_id`, `question_version`, and the complete
recursive `payload`. JSON object key order is irrelevant, while array order and
missing-versus-null values remain significant. `occurred_at`, `sequence`, and all
trace/request/receive metadata are intentionally excluded, matching the Python
trace's logical-idempotency contract.

An exact replay returns HTTP 200 as a verified `duplicate`. Reusing an ID for
different logical content returns HTTP 409 `EVENT_ID_CONFLICT`, preserves the
immutable first write, and inserts no new feedback event from that request. The
same rule applies atomically to mixed batches: if one event conflicts, all
otherwise-new events in that batch are withheld as rejected.
Transaction-scoped advisory locks serialize overlapping ID sets in a
deterministic lock-key order, so
simultaneous first writes, exact retries, and conflicting writes receive the same
first-write-wins classification.

## What is included

- `migrations/20260711000000_feedback_events.sql` creates the single JSONB event
  table, indexes, append-only trigger, RLS posture, and reporting views.
- `functions/feedback-ingest/index.ts` authenticates, validates, and inserts a
  one-event or full-trace envelope.
- `migrations/20260712000000_feedback_reports.sql` adds four protected,
  parameterized report RPCs that filter raw events before aggregation.
- `migrations/20260712010000_feedback_ingest_observability.sql` adds a separate,
  private, append-only ingestion request-outcome table for OBS-001 phase one.
- `migrations/20260712011000_feedback_ingest_observability_report.sql` adds the
  private, single-row `feedback_report_ingestion_summary` RPC for OBS-001B.
- `migrations/20260712012000_feedback_event_conflicts.sql` adds exact logical
  comparison, the atomic ingest RPC, the private conflict audit, and outcome
  schema 1.1.
- `migrations/20260712012500_feedback_event_writer_lockdown.sql` removes direct
  service-role inserts into `feedback_events` after the RPC-aware Edge rollout.
- `migrations/20260712013000_feedback_conflict_observability_report.sql` extends
  ingestion reporting with verified retry, legacy-unclassified, conflict, and
  event-ID reuse metrics.
- `migrations/20260712013500_feedback_raw_view_hardening.sql` preserves the
  existing service-role raw-view column prefixes while correcting attempt,
  question-version, and known-answer statistics and making optional proposal
  seed conversion fail-safe.
- `migrations/20260712014000_feedback_question_registry.sql` adds private,
  append-only release/question/choice facts, dynamic event attribution,
  Registry quality, and exact-event resolution. Its deferred checks and
  question-version advisory lock are designed to reject incomplete or
  inconsistent inserts. The service role has SELECT only; it cannot register
  or change answers.
- `migrations/20260712014500_feedback_question_registry_release_4e752a.sql`
  registers the current attested 60-question / 180-choice release using three
  insert-only statements generated from `registries/release_4e752a....json`.
- `migrations/20260712015000_feedback_authoritative_reports.sql` preserves the
  four business RPC schemas while replacing client-reported dimensions and
  correctness with registry-derived values. It also adds the single-row
  aggregate-authority status marker.
- `migrations/20260712016000_feedback_detail_reports.sql` adds paginated,
  registry-matched answer and proposed/rejected-setting RPCs, then recreates
  `feedback_report_authority_status` as the final seven-column
  `registry_v1/detail_v1` aggregate-plus-detail cutover marker.
- `migrations/20260712017000_feedback_business_snapshot.sql` adds the
  service-role-only, single-statement `business_snapshot_v1` RPC that returns
  all six business pages from one PostgreSQL MVCC snapshot.
- `migrations/20260712018000_feedback_session_attempt_filters.sql` is the
  forward-only REPORT-002 cutover that recreates the six business RPCs and
  snapshot with appended `session_id`/`attempt_id` parameters while preserving
  every historical positional prefix.
- `migrations/20260712019000_question_reactions.sql` expands the existing event
  enum and atomic ingest RPC for strict `question_presented` decisions and
  post-result surprise reactions. Presentation payloads require exact
  attempt/release/decision/policy context, mode, propensity, source, and
  position. Reactions require `reaction=surprise`, a boolean value,
  `timing=after_reveal`, and an attempt ID. Both preserve schema `1.0`,
  append-only storage, and conflict idempotency.
- `migrations/20260712020000_feedback_surprise_report.sql` adds the two
  service-role-only SURPRISE-002 RPCs: authoritative per-question first-rating
  counts/coverage/Beta(1,1) posterior and a conserved raw reaction-quality
  classification. It does not change the six-page business snapshot.
- `functions/feedback-report/index.ts` exposes all thirteen reporting RPCs through
  a read-only GET endpoint authenticated by a dedicated report token.
- `../tools/feedback_postgres_acceptance.py` is an explicit, rollback-only
  staging database verifier for the migration catalog, function/table ACLs,
  forced RLS, trigger shapes, constraints, and exact current registry.

No consent or privacy workflow is included; this receiver is intended for the
current internal deployment.

Everything in this inventory describes repository-local deployment inputs and
their intended state **after** the migrations are applied in order. None of the
registry/detail/snapshot/identity-filter/surprise migrations, Edge revisions,
PostgreSQL constraints/RLS grants, database verifier, or endpoint verifier has
been confirmed on a real Supabase project yet.

## Request contract

Send `POST` with both headers:

```text
Authorization: Bearer <FEEDBACK_INGEST_TOKEN>
Content-Type: application/json
```

The request must be at most **1 MiB** and contain **1–500 events**. Envelope and
event objects use exact field allowlists; unknown top-level fields are rejected.
The supported envelope is:

```json
{
  "schema_version": "1.0",
  "envelope_type": "session_trace",
  "trace_id": "trace_...",
  "session_id": "anon_...",
  "created_at": "2026-07-11T08:00:00.000Z",
  "event_count": 1,
  "events": [
    {
      "schema_version": "1.0",
      "event_id": "evt_...",
      "event_type": "comment_submitted",
      "occurred_at": "2026-07-11T08:00:01.000Z",
      "session_id": "anon_...",
      "question_id": "q_example",
      "question_version": "qv1_...",
      "payload": {
        "category": "suggestion",
        "text": "Please expose the training seed.",
        "attempt_id": "attempt_...",
        "release_id": "quiz_...",
        "family": "univariate_regression",
        "question_type": "architecture_only"
      },
      "sequence": 1
    }
  ]
}
```

Validation also requires:

- `event_count` equals `events.length`;
- event `sequence` values are integers from 1 through 2,147,483,647 and strictly
  increasing in request order;
  they are stable session sequence numbers, so a one-event comment upload keeps
  its original sequence (it does not have to be 1);
- every event uses the envelope `session_id`;
- every `event_id` is unique within the request;
- identifiers are trimmed, non-empty, at most 200 Unicode code points, and
  contain no newlines;
- timestamps are RFC 3339 values;
- `event_type` is one of `answer_submitted`, `question_presented`,
  `question_reaction_submitted`, `custom_setting_proposed`,
  `custom_setting_rejected`, `custom_run_completed`, `custom_run_failed`, or
  `comment_submitted`;
- type-specific payload fields match the Python client: an answer has
  `selected_letter`; a presentation has attempt/release/decision/policy IDs,
  `exploit|explore|fallback|manual` mode, finite propensity in `(0, 1]`, a known
  navigation source, and positive position; a reaction has
  `reaction=surprise`, boolean `value`, `timing=after_reveal`, and attempt ID; a
  proposed/rejected setting has an object `setting`; a run event has an object
  `run` whose status matches `completed`/`failed`; and a comment has an allowed
  `category` plus 1–2000 trimmed Unicode code points of `text`;
- every recursive JSON string is valid Unicode without an unpaired surrogate;
  and every integer-valued JSON number is within JavaScript's lossless range
  `[-9007199254740991, 9007199254740991]`. This is checked before the Edge
  Function forwards data to PostgreSQL, so an accepted receipt cannot hide
  JavaScript integer rounding.

Payload objects may retain additional JSON properties produced through the
client's `extra` mechanism. Known reporting context (`attempt_id`, `release_id`,
`family`, `dataset_id`, `question_type`, `selection_metric`, `budget`, and
`is_correct`) is type-checked when present.
Ingestion deliberately does not reject missing or unknown registry claims: raw
events must remain recoverable, including events that arrive before a release is
registered. STATS-003 resolves them dynamically at report time using exact
`payload.release_id + question_id + question_version` membership. Business
reports never use payload `is_correct`, family, dataset, or question type as an
authority; unmatched data stays visible only through raw/Registry quality
surfaces until an owner-reviewed registry migration makes it resolvable.

Every response includes the same receipt counters and an `X-Request-ID` header:

```json
{
  "accepted": 1,
  "duplicate": 0,
  "conflict": 0,
  "rejected": 0,
  "request_id": "72aee12d-7742-44ea-b3d9-f056ae5c8ac2"
}
```

The JSON `request_id` and `X-Request-ID` header carry the same server-generated
UUID. It is distinct from every event ID and is also the primary key of that
request's sanitized outcome row, so trusted reporting tools can correlate one
ingestion receipt with one persisted outcome exactly.

`accepted` is newly inserted events. `duplicate` is a confirmed non-conflicting
reuse whose seven-field logical JSONB exactly matches the stored first write; a
duplicate-only request is a successful idempotent ingestion. `conflict` counts
IDs whose logical content differs from the first write. It is zero on a normal
HTTP 200 receipt.

A conflict response is HTTP 409 with `error.code = EVENT_ID_CONFLICT`. Its
`accepted` count is zero, `duplicate` still counts exact matches already stored,
and `rejected` counts both conflicting events and otherwise-new events withheld
by atomic batch rejection; `conflict` is the conflicting subset. Thus
`accepted + duplicate + rejected` equals the requested batch size. Validation
errors return HTTP 400, oversize bodies 413, authentication failures 401, and
retryable storage failures 502.

For example, a three-event batch containing one exact replay, one conflicting
ID, and one otherwise-new event returns no event insert from that request:

```json
{
  "accepted": 0,
  "duplicate": 1,
  "conflict": 1,
  "rejected": 2,
  "request_id": "72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
  "error": {
    "code": "EVENT_ID_CONFLICT",
    "message": "One or more event IDs already store different logical content; the batch was not inserted"
  }
}
```

A storage 502 returns all four numeric receipt counters as zero for
compatibility; it does **not** claim that all requested events were rejected.
The separate outcome record keeps accepted/duplicate/conflict/rejected counters
nullable when the storage response was lost or invalid and records whether
storage was `unknown`, `not_committed`, or `not_attempted`. Error responses add
a short `error.code` and safe message; bearer tokens, service-role keys, request
bodies, and PostgREST response bodies are never returned or logged.

## OBS-001 ingestion request outcomes

`feedback_ingest_request_outcomes` is deliberately separate from
`feedback_events`. It records one sanitized request result without changing
accepted-event history or quiz/report aggregates. The table has forced RLS,
grants only service-role `SELECT`/`INSERT`, and rejects update, delete, and
truncate operations.

Only authenticated POST requests enter the persisted phase-one denominator:

- `success` covers accepted-only, mixed accepted/duplicate, and duplicate-only
  idempotent requests whose duplicate IDs were confirmed by JSONB comparison;
- `client_rejection` covers authenticated oversize, unreadable, or invalid
  envelopes for which storage was not attempted, plus confirmed event-ID
  conflicts returned as non-retryable HTTP 409 after atomic batch rejection;
- `service_failure` covers authenticated internal/storage failures and marks
  them retryable with `storage_state` `unknown`, `not_committed`, or
  `not_attempted`.

The upgraded Edge writes outcome schema `1.1` with observer revision `obs2`.
Confirmed success rows have `conflicting_event_count = 0`; conflict rows have a
positive count and `storage_state = confirmed`; validation and storage-unknown
rows keep it `null` because no comparison result was confirmed. Append-only
schema `1.0` rows remain valid without a rewrite. Their duplicate counts predate
server comparison and are reported as legacy/unclassified, not retroactively
called exact retries or conflicts.

Method 405, authentication 401, and missing-function-configuration 503 outcomes
are written only as safe structured logs. They are not persisted and are not
included in the rate denominator. Outcome rows never store tokens, request
bodies, IP addresses, comments, settings, or other event payload content; they
contain only request timing, HTTP status, classification, nullable counters,
storage state, and an observer revision.

Outcome persistence is observability-only and fail-open. It runs as a bounded
background task through `EdgeRuntime.waitUntil`, so a slow or failed outcome
write is logged safely but cannot delay, turn into an error, or alter the
original ingestion receipt. The main event-storage request has its own bounded
deadline so a hung PostgREST call reaches the retryable/unknown path. This also
defines the current coverage limit: the same-database table cannot see requests
that never reached the Edge Function, and it may be unable to record an outcome
during a full database outage.

OBS-001B exposes the persisted subset through the private
`feedback_report_ingestion_summary` RPC. It aggregates only outcome rows where
`included_in_rate = true`. Optional `from` and `to` bounds apply to the
server-controlled request `started_at`, not the client event `occurred_at`, and
use the half-open interval `started_at >= from AND started_at < to`. An optional
UUID `request_id` performs an exact equality filter on the persisted request
outcome and may be combined with the time bounds. This filter is accepted only
for the ingestion-summary RPC; the six business RPCs reject it. Release,
family, question-type, and question-ID filters remain unsupported. The RPC
always returns exactly one row, including when the selected window or exact
request ID contains no recorded outcome.

The normal Reports UI does not expose `request_id`; its observability tab uses
only the date range. The exact filter is reserved for trusted operator tooling
and the hosted roundtrip verifier, rather than being a user-facing analytics
dimension. Its value is the ingestion POST receipt UUID. The `feedback-report`
Edge Function also puts its own correlation ID in each report response envelope;
that separate report-request ID is not the ingestion filter value.

The conflict-aware report keeps explicit counts for conflict requests, accepted
events, all non-conflicting duplicate results, verified idempotent duplicates,
legacy/unclassified duplicates, conflicting events, correlated private-audit
rows, total event-ID reuse, and `classified_event_count`. The classified count
sums every event in successful or conflict-classified RPC results, including
otherwise-new events withheld by an atomic conflict rejection.
`conflict_audit_event_count` joins the private sidecar to the selected recorded
outcome request IDs; the strict client requires it to equal
`conflicting_event_count`.
Its rates have explicit denominators:

- `request_failure_rate = (client_rejection_count + service_failure_count) /
  recorded_request_count`;
- `duplicate_event_rate = duplicate_event_count /
  (accepted_event_count + duplicate_event_count)`;
- `event_id_reuse_rate = event_id_reuse_count / classified_event_count`;
- `classified_conflicting_event_rate = conflicting_event_count /
  (idempotent_duplicate_event_count + conflicting_event_count)`.

The classified conflict rate deliberately excludes legacy/unclassified
duplicates. `duplicate_event_count` remains the total non-conflicting duplicate
result count across both revisions; it is split into verified-idempotent and
legacy-unclassified components for interpretation.

Each rate is `null` (shown as N/A) when its own denominator is zero.
`recorded_rate_available` is true only when `recorded_request_count > 0`; it
does not assert that every end-to-end request was observed. The RPC deliberately
returns `end_to_end_coverage_available = false` for every window.

The function is `SECURITY INVOKER`; execution is revoked from `public`, `anon`,
and `authenticated` and granted only to `service_role`. It returns aggregate
counts, rates, and server timestamps, not raw outcome rows or event content.
The underlying outcome table remains private, forced-RLS, and append-only.

This narrower recorded-subset report does not redefine the business Summary
metric. `feedback_report_summary` still returns
`ingestion_failure_rate = null` and
`ingestion_failure_rate_available = false`. Those fields remain N/A until the
hosted Postgres/RLS deployment and end-to-end outcome coverage are validated.

## Deploy to a Supabase project

The following commands are instructions for a machine with the Supabase CLI;
they have not been run against a real hosted project for this revision. Local
contract tests and preflight output do not establish hosted migration or
runtime state.

Before each rollout boundary, run the repository-local, read-only preflight for
the phase you are about to enter:

```bash
.venv/bin/python tools/feedback_rollout_preflight.py --phase expand
.venv/bin/python tools/feedback_rollout_preflight.py --phase ingest-cutover
.venv/bin/python tools/feedback_rollout_preflight.py --phase lockdown-report
.venv/bin/python tools/feedback_rollout_preflight.py --phase report-app
```

Checks are cumulative: a later phase includes every earlier deployment input
and contract. The four earlier migrations are included as compatibility inputs
so the checked revision is reproducible; that is not an instruction to reapply
them to an existing database. The tool reads files and Git metadata only. It
never accepts tokens, database URLs, or project references; it does not contact
Supabase, apply SQL, deploy functions, or write hosted probe data. It prints the
Git SHA and an exact SHA-256 fingerprint of the enumerated cumulative rollout
inputs—including the production Inspector feedback client/outbox/recovery UI,
runtime release attestation, rollback-only database verifier, and hosted
roundtrip verifier—so an operator can bind the
review, deployment, and smoke evidence to the same revision. Use `--json` for
machine-readable `PASS` / `FAIL` /
`UNVERIFIED` results.

`hosted.acceptance` is deliberately `UNVERIFIED`, and `deploy_ready` is always
false, because a local static check cannot prove the target's migration state,
deployed Edge revisions, secrets, RLS/grants, or runtime behavior. Adding
`--require-hosted` makes that missing evidence fail the command with exit code
2. A static failure exits 1. Without `--require-hosted`, a clean static pass can
exit 0 for use as a repository gate, but the output still says **NOT
DEPLOY-READY**. Untracked, staged, unstaged, deleted, missing, symlinked,
non-regular, or invalid UTF-8 checked inputs fail closed; the checker itself and
the exact migration inventory are also covered. The current shared working tree
is therefore not a deployable revision until its scoped inputs are reviewed and
committed.

The phases correspond to this compatibility sequence:

- `expand`: apply through `20260712012000_feedback_event_conflicts.sql`.
- `ingest-cutover`: deploy and smoke the RPC-aware `feedback-ingest`.
- `lockdown-report`: enter a Reports maintenance window, apply `12500`, `13000`,
  the backward-compatible raw-view hardening in `13500`, registry schema
  `14000`, reviewed release data `14500`, and authoritative report cutover
  `15000`, followed by authoritative answer/proposal details in `16000` and the
  atomic six-view business snapshot in `17000`, the forward-only
  session/attempt filter cutover in `18000`, strict presentation/reaction storage
  in `19000`, and the additive surprise reports in `20000`; keep each exact
  SQL/Python contract and registry artifact together.
- `report-app`: deploy the matching `feedback-report` function and private
  Reports app, then perform hosted acceptance against the recorded
  fingerprints.

After a real deployment exists, preserve the clean preflight, rollback-only
PostgreSQL acceptance JSON, authoritative hosted-roundtrip JSON, and reviewed
provider control-plane capture through the retrospective ledger documented in
[`deployments/README.md`](../deployments/README.md). The ledger tool/docs are
`report-app` fingerprint inputs; `deployments/ledger.jsonl` and post-deploy
evidence are intentionally not, because the provider deploy ID cannot exist
until after that source commit is deployed. All three evidence classes must be
reviewed against one deployment context. `ACTIVATED_REVIEWED` is an audit state,
not a provider-issued cryptographic attestation. No real ledger event has been
recorded for the current working tree.

1. Log in and link this repository to the target project:

   ```bash
   supabase login
   supabase link --project-ref YOUR_PROJECT_REF
   ```

2. Choose the migration rollout appropriate to the target. A new staging
   project with no live writer can apply every migration normally:

   ```bash
   supabase db push
   ```

   An existing project running the old direct-insert Edge must use two schema
   phases. First apply migrations only through
   `20260712012000_feedback_event_conflicts.sql`; this adds the RPC while the old
   path still has `INSERT`. Deploy and smoke-test the RPC-aware
   `feedback-ingest` function next. Only then apply `12500`, `13000`, `13500`,
   `14000`, the reviewed current-release `14500`, `15000`, `16000`, `17000`,
   `18000`, `19000`, and `20000` in order. Do not deploy the
   presentation/reaction-emitting Inspector
   until `19000` is installed. `12500`
   revokes direct event inserts; `13000` changes ingestion observability; `13500`
   hardens raw views; `14000/14500` install immutable registry facts; `15000`
   cuts the original four business reports over to server authority; `16000`
   adds authoritative answer/proposal details and the final dual-revision status;
   `17000` adds the single-statement `business_snapshot_v1` RPC used by the app;
   `18000` recreates it and all six business RPCs with appended server-side
   session/attempt identity filters while preserving old positional prefixes;
   `19000` adds presentation/reaction enum and payload constraints to the store
   and ingest RPC; `20000` adds the two independent surprise report RPCs.
   Deploy the matching strict
   Reports client/app with this maintenance-window sequence; a client/schema
   mismatch fails closed. Use a staged
   deployment revision or migration runner that can stop at that boundary; do
   not run one uncoordinated all-pending `db push` while the old Edge is serving
   traffic. A briefly stale Edge instance after lockdown fails closed and its
   request can be retried.

   For every later quiz release, generate a new registry JSON and timestamped
   data migration from the complete bundle—never hand-write answer rows:

   ```bash
   .venv/bin/python tools/export_feedback_registry.py \
     --bundle examples/quiz_demo/bundle \
     --json-output supabase/registries/release_<64hex>.json \
     --sql-output supabase/migrations/<timestamp>_feedback_question_registry_release_<prefix>.sql
   ```

   Review both files and run the same command with `--check` before applying.
   Migrations execute as the database owner; do not grant registry INSERT or an
   importer RPC to `service_role`, because both Edge Functions hold that key.

3. Generate a dedicated ingest token, store it in your password manager, then
   set it as an Edge Function secret. Do not reuse the service-role key.

   ```bash
   openssl rand -base64 32
   supabase secrets set FEEDBACK_INGEST_TOKEN='PASTE_THE_GENERATED_VALUE'
   ```

   Hosted Edge Functions already receive `SUPABASE_URL` and
   `SUPABASE_SERVICE_ROLE_KEY`; do not copy either value into Streamlit.

4. Generate a separate report token. It must not equal the ingest token:

   ```bash
   openssl rand -base64 32
   supabase secrets set FEEDBACK_REPORT_TOKEN='PASTE_A_DIFFERENT_GENERATED_VALUE'
   ```

5. Deploy both functions with Supabase JWT verification disabled. Each
   function performs its own Bearer-token authentication. On an upgraded live
   project, first move to the RPC-aware `feedback-ingest` at the `12000`
   boundary, then install through `19000` before exposing the
   presentation/reaction-enabled Inspector. Deploy `feedback-report` only after
   `20000` plus the `18000` identity-filter migration, the `17000` business
   snapshot, the `16000` detail-report cutover, and the final dual-revision
   authority-status marker are installed:

   ```bash
   supabase functions deploy feedback-ingest --no-verify-jwt
   supabase functions deploy feedback-report --no-verify-jwt
   ```

6. The two endpoints are:

   ```text
   https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-ingest
   https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-report
   ```

The gateway must remain in `--no-verify-jwt` mode; the functions still reject
requests without their exact dedicated Bearer token. Hosted Edge Functions
receive `SUPABASE_SERVICE_ROLE_KEY` automatically. It stays inside the Edge
environment and is never a Streamlit or browser credential.

## PostgreSQL staging acceptance

After applying the migrations to a staging project, and before treating an
endpoint roundtrip as deployment evidence, inspect the actual database with the
rollback-only verifier. Install its optional driver once:

```bash
.venv/bin/python -m pip install -e '.[postgres-acceptance]'
```

Inject an owner/admin PostgreSQL DSN from the team's secret manager. The tool
does not accept a DSN argument and never prints it. The target label is an
operator assertion, not a cryptographic binding to the DSN. `--confirm-staging`
is still mandatory, and production-like labels are rejected:

```bash
export ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN='INJECT_STAGING_ADMIN_DSN'
export ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_TARGET='architecture-iq-staging'
.venv/bin/python tools/feedback_postgres_acceptance.py \
  --confirm-staging > postgres-staging-acceptance.json
unset ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN
```

Use a staging database owner or equivalent trusted migration role. The
application `service_role` intentionally lacks the mutation privileges needed
to exercise every protected boundary. The verifier checks the exact reviewed
migration range; all 13 application function signatures, attributes, empty
search paths, and execute grants; six forced-RLS/no-policy tables and their
least-privilege grants; exact trigger event/timing/level masks; all stable named
PK/UK/FK/CHECK constraints; and the current 60-question/180-choice registry.
It reconstructs the registry identity hash from every hosted question/choice
row instead of trusting only stored counts and `registry_id`. Safe
statement-level UPDATE/DELETE/TRUNCATE probes and immediate/deferred invalid
registry inserts run behind savepoints. The dedicated connection always calls
`rollback()` and `close()` and never commits.

Exit 0 means every checked database condition passed and rollback was
confirmed; exit 2 is configuration/driver refusal; exit 3 is a database or
acceptance failure. Save the sanitized JSON beside the matching Git SHA and
`report-app` preflight fingerprint. A pass does not automatically change the
offline preflight's `hosted_verified=false`, and it does not prove that either
Edge Function is deployed, endpoint authentication/receipts, persistence after
restart, concurrent advisory-lock behavior, or snapshot behavior under real
write/load. The permanent hosted roundtrip below covers the endpoint path;
concurrency and load remain separate staging gates.

## Configure Streamlit

In the deployed Streamlit app's secrets (or local `.streamlit/secrets.toml`),
add:

```toml
[feedback]
endpoint = "https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-ingest"
token = "THE_SAME_FEEDBACK_INGEST_TOKEN"
timeout_seconds = 10

[reports]
endpoint = "https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-report"
token = "THE_SAME_FEEDBACK_REPORT_TOKEN"
timeout_seconds = 10
```

Only the two dedicated application tokens belong in their respective
Streamlit apps. Do not reuse one for the other. Never put
`SUPABASE_SERVICE_ROLE_KEY` in Streamlit secrets, client code, Git, curl history,
or a downloadable trace.

Equivalent process environment variables are
`ARCHITECTURE_IQ_FEEDBACK_ENDPOINT`, `ARCHITECTURE_IQ_FEEDBACK_TOKEN`, and
optionally `ARCHITECTURE_IQ_FEEDBACK_TIMEOUT`.

The separate Reports app uses `ARCHITECTURE_IQ_REPORTS_URL`,
`ARCHITECTURE_IQ_REPORTS_READ_TOKEN`, and optionally
`ARCHITECTURE_IQ_REPORTS_TIMEOUT`. See
[`tools/feedback_reports/README.md`](../tools/feedback_reports/README.md).
Its report token protects the server-to-server Edge request, not the Reports
web page. Deploy that entrypoint as a separate app with Streamlit's
platform-level private access enabled and restricted to maintainers; an
anonymous Reports URL is not an acceptable deployment.

Once both hosted endpoints and private page access are configured, prove the
write/read roundtrip with the explicit permanent-write verifier (prefer a
staging project):

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write
```

The CLI first builds a feedback registry from `--bundle` (defaulting to the
checked-in quiz bundle), which runs runtime manifest/question attestation and
publisher validation, then selects a real canonical question and expected
`registry_id`. Before any permanent POST it requires all of these hosted
preconditions:

- `feedback_report_authority_status` returns exactly one seven-column row with
  `authority_revision='registry_v1'`,
  `business_reports_authoritative=true`, registered release/question/choice
  counts that cover at least the locally attested bundle,
  `detail_revision='detail_v1'`, and
  `detail_reports_authoritative=true`. This jointly proves the `15000`
  aggregate and `16000` detail cutovers.
- `feedback_report_answers` and `feedback_report_proposals` each return a
  complete empty page for a random nonexistent question ID. These protected
  negative controls prove that both `16000` RPCs are on the Edge allowlist and
  reachable without depending on historical report rows.
- `feedback_report_business_snapshot` returns `business_snapshot_v1` for a
  different random nonexistent question ID before the first write. Its server
  `snapshot_at`, embedded `registry_v1`/`detail_v1` authority facts and counts
  must validate; Summary must be the exact all-zero singleton and the other
  five pages must be complete and empty. A passing run records
  `business_snapshot_verified=true`.
- exact event resolution for a random valid event ID returns the canonical
  one-row `not_found` shape. This exercises the `14000` projection without
  relying on aggregate history.
- ingestion summary for a random request UUID returns the all-zero/null row. A
  post-write negative control catches a backend that silently ignores this
  filter.

The normal probe then writes one comment for that registered question. HTTP 200
must carry the same canonical UUID in the receipt header and body. A fresh run
requires `accepted=1, duplicate=0, conflict=0, rejected=0`; resume requires
`accepted=0, duplicate=1, conflict=0, rejected=0`. The exact request outcome and
event-resolution row must agree with that receipt.

The default registry-aware successful-batch probe writes an answer, proposed
setting, and comment in the same session. The first fresh write must be
`3/0/0/0`; replaying the identical trace must be `0/3/0/0`, with a distinct
canonical request UUID and exact outcome for each request. The probe deliberately
sends incorrect client family/question-type and inverted answer `is_correct`.
Exact event resolution must instead return canonical registry identity and
server-derived correctness. It then scans every Answers/Proposals page in each
event's `[occurred_at, occurred_at + 1 ms)` authoritative window, matches the
event ID exactly once, and validates the complete answer row plus structured
proposal setting/inheritance and seed facts. A resume reports whether this
invocation actually performed the first write instead of presenting existing
duplicates as new write evidence. `--skip-successful-batch-probe` explicitly
leaves the successful-batch/detail proof flags false. The legacy fixture path
retains its disjoint two-event answer/comment trace but is not the default CLI
hosted proof.

The authoritative default then verifies the `18000` identity filters. A snapshot
for the uploaded `session_id + attempt_id` must contain the exact Summary,
Sessions, Questions, Answers, Proposals, and Comments facts for that trace. A
random wrong session paired with the real attempt, and the real session paired
with a random wrong attempt, must each return an exact empty six-view snapshot.
Only this positive hit plus both negative controls sets hosted JSON evidence
`session_attempt_filters_verified=true`.

By default, the conflict phase submits the same comment event ID with only its
text changed. It must raise the production client's structured conflict error
with HTTP 409, `error.code=EVENT_ID_CONFLICT`, a new canonical header/body UUID,
and `accepted=0, duplicate=0, conflict=1, rejected=1`. Exact outcome correlation
must show the conflict and private audit, while exact event resolution must
remain the original registered comment. This jointly checks 409 handling,
sidecar correlation, and first-write-wins for the single-event probe.

To additionally exercise one all-or-none batch shape, opt in explicitly:

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write \
  --include-mixed-batch-probe
```

After the normal and single-conflict phases, this optional phase submits a two-event
trace in the order `[existing changed-text conflict, deterministic new comment]`.
It requires another structured HTTP 409 with
`accepted=0, duplicate=0, conflict=1, rejected=2`; its canonical request UUID
must be distinct from both earlier request UUIDs. Exact outcome correlation must
show one client-rejection/conflict request, one conflicting event, one private
audit event, no accepted or duplicate result, `classified_event_count=2`,
`event_id_reuse_rate=0.5`, failure and classified-conflict rates of one, a null
duplicate rate, and incomplete end-to-end coverage. Exact resolution for the
existing event must remain unchanged, while the withheld new event must still
return `not_found`.

The verifier supports deterministic `--run-id ... --resume`. A fresh run proves
new first writes; resume proves the expected existing events and exact logical
replays without claiming a new first write. Both modes run a new single-conflict
probe by default, and both also run the mixed batch when
`--include-mixed-batch-probe` is present. For a compatibility or
interrupted-recovery check only, pass
`--skip-conflict-probe`; the result then reports `conflict_verified=false` and
`mixed_batch_verified=false` and is not a complete conflict acceptance.
`--skip-conflict-probe` and `--include-mixed-batch-probe` are mutually exclusive.

The tool accepts no token arguments and cannot clean up accepted events, request
outcomes, or conflict audit. Fresh default execution permanently adds the normal
and successful-batch events, and every invocation records new request outcomes;
conflict phases also leave rejected outcomes and audit rows. This is true for
`--resume`, so these probes intentionally affect global quality metrics. Run
them in staging first and retain every printed request UUID. If fail-open outcome
persistence is lost, a 409 and its audit may already have committed even though
polling times out; a retry creates another request/audit footprint.

The opt-in trace proves all-or-none handling only for the tested
`new + conflict` order; it
does not prove `duplicate + conflict`, arbitrary concurrency, requests that never
reached Edge, or end-to-end coverage. No real hosted acceptance run has been
completed yet: the authority-status, registry resolution, ingestion, RLS, and
detail/constraint claims above remain local contracts until this command succeeds
against the reviewed hosted revision.

## Run and curl locally

With the Supabase CLI installed, start the local stack and apply migrations:

```bash
supabase start
supabase db reset
```

Create an untracked local file such as `supabase/.env.local`:

```dotenv
FEEDBACK_INGEST_TOKEN=local-feedback-token-change-me
FEEDBACK_REPORT_TOKEN=local-report-token-change-me
```

Serve both functions. The local CLI supplies its local `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` automatically:

```bash
supabase functions serve \
  --no-verify-jwt \
  --env-file supabase/.env.local
```

In another shell, upload a one-comment envelope:

```bash
curl --fail-with-body \
  --request POST \
  --url http://127.0.0.1:54321/functions/v1/feedback-ingest \
  --header 'Authorization: Bearer local-feedback-token-change-me' \
  --header 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "schema_version": "1.0",
  "envelope_type": "session_trace",
  "trace_id": "trace_local_comment",
  "session_id": "anon_local",
  "created_at": "2026-07-11T08:00:00.000Z",
  "event_count": 1,
  "events": [
    {
      "schema_version": "1.0",
      "event_id": "evt_local_comment_1",
      "event_type": "comment_submitted",
      "occurred_at": "2026-07-11T08:00:01.000Z",
      "session_id": "anon_local",
      "question_id": "q_local",
      "question_version": "qv1_local",
      "payload": {
        "category": "suggestion",
        "text": "Local ingest smoke test",
        "attempt_id": "attempt_local",
        "family": "univariate_regression",
        "question_type": "architecture_only"
      },
      "sequence": 1
    }
  ]
}
JSON
```

Run the same command again: the expected HTTP 200 receipt changes from
`accepted: 1, duplicate: 0, conflict: 0, rejected: 0` to
`accepted: 0, duplicate: 1, conflict: 0, rejected: 0`. Changing any of the seven
logical fields while retaining `evt_local_comment_1` instead returns HTTP 409
`EVENT_ID_CONFLICT`; the original row remains unchanged. To test a full
inspector trace, download its JSON and send the file unchanged:

```bash
curl --fail-with-body \
  --request POST \
  --url http://127.0.0.1:54321/functions/v1/feedback-ingest \
  --header 'Authorization: Bearer local-feedback-token-change-me' \
  --header 'Content-Type: application/json' \
  --data-binary @architectureiq-anon_example.json
```

Query a protected report after uploading events:

```bash
curl --fail-with-body --get \
  --url http://127.0.0.1:54321/functions/v1/feedback-report \
  --header 'Authorization: Bearer local-report-token-change-me' \
  --data-urlencode 'view=feedback_report_summary' \
  --data-urlencode 'limit=200' \
  --data-urlencode 'offset=0'
```

## REPORT-001/REPORT-002 views and protected RPCs

The migration provides read-only building blocks for a Kahoot-style internal
dashboard:

- `feedback_session_summary`: one row per session/`attempt_id`, with known-answer
  accuracy, distinct question-version, proposed/rejected setting,
  completed/failed run, and comment counts;
- `feedback_question_stats`: attempts, sessions, accuracy, comments, and custom
  work (including separate rejected-setting and failed-run counts) grouped by
  question version, `family`, and `question_type`;
- `feedback_comments`: categorized comment messages with session, attempt,
  family, and question context;
- `feedback_proposals`: proposed and rejected setting JSON, a `setting_status`,
  plus inheritance and seed context.

These are legacy/raw diagnostic views over browser payloads. They remain for
compatibility and trusted investigation, but their dimensions or `is_correct`
values are not the authority used by the post-`15000`/`16000` business RPCs.

Migration `13500` hardens these original service-role views without changing any
existing named column's name, order, or type. It appends
`known_answer_count`, `incorrect_answer_count`, and `unknown_answer_count` to the
two aggregate views, and computes `accuracy` over known boolean correctness only.
Session question counts use `(question_id, question_version)`; question attempt
counts use `(session_id, coalesced attempt_id)`, so IDs reused in another session
do not collapse and a missing attempt ID still counts once per session.

`feedback_proposals.n_seeds` and `base_seed` remain nullable `integer` columns for
compatibility. A JSON string, fractional number, or number outside the int32
range now produces `NULL` for that optional field instead of making the entire
view fail with a cast error. Because the two aggregate views append columns,
consumers that require an exact `SELECT *` shape should update their schema
expectation; consumers selecting the pre-existing named columns remain
compatible. The protected `feedback_report_*` RPC and Python client schemas are
unchanged by `13500`.

After applying `13500` in **staging**, verify the catalog and execute the
counterexamples below in one rollback-only SQL Editor transaction. The expected
question row is `attempt_count=2`, `known=1`, `unknown=1`, `accuracy=1.0000`;
the session row has two question versions; every displayed proposal seed is
`NULL`. Do not use synthetic rows against production.

```sql
select c.relname, a.attnum, a.attname, format_type(a.atttypid, a.atttypmod)
from pg_class c
join pg_attribute a on a.attrelid = c.oid
where c.relname in (
    'feedback_session_summary',
    'feedback_question_stats',
    'feedback_proposals'
)
  and a.attnum > 0
  and not a.attisdropped
order by c.relname, a.attnum;

select c.relname, c.reloptions,
       has_table_privilege('anon', c.oid, 'select') as anon_select,
       has_table_privilege('authenticated', c.oid, 'select') as authenticated_select,
       has_table_privilege('service_role', c.oid, 'select') as service_role_select
from pg_class c
where c.relname in (
    'feedback_session_summary',
    'feedback_question_stats',
    'feedback_proposals'
)
order by c.relname;

begin;

insert into public.feedback_events (
    event_id, schema_version, trace_id, trace_created_at, session_id,
    question_id, question_version, event_type, occurred_at, sequence,
    payload, ingest_request_id, received_at
)
values
    (
        'raw_hardening_answer_a', '1.0', 'trace_raw_hardening', now(),
        'raw_session_a', 'q_raw_hardening', 'qv_raw_1', 'answer_submitted',
        now(), 1,
        '{"attempt_id":"attempt_shared","selected_letter":"A","is_correct":true}'::jsonb,
        '00000000-0000-4000-8000-000000000001', now()
    ),
    (
        'raw_hardening_answer_b', '1.0', 'trace_raw_hardening', now(),
        'raw_session_b', 'q_raw_hardening', 'qv_raw_1', 'answer_submitted',
        now(), 1,
        '{"attempt_id":"attempt_shared","selected_letter":"B"}'::jsonb,
        '00000000-0000-4000-8000-000000000002', now()
    ),
    (
        'raw_hardening_comment_v2', '1.0', 'trace_raw_hardening', now(),
        'raw_session_a', 'q_raw_hardening', 'qv_raw_2', 'comment_submitted',
        now(), 2,
        '{"attempt_id":"attempt_shared","category":"other","text":"version probe"}'::jsonb,
        '00000000-0000-4000-8000-000000000003', now()
    ),
    (
        'raw_hardening_proposal_large', '1.0', 'trace_raw_hardening', now(),
        'raw_session_a', 'q_raw_hardening', 'qv_raw_1',
        'custom_setting_proposed', now(), 3,
        '{"attempt_id":"attempt_shared","setting":{},"n_seeds":2147483648,"base_seed":"legacy"}'::jsonb,
        '00000000-0000-4000-8000-000000000004', now()
    ),
    (
        'raw_hardening_proposal_fractional', '1.0', 'trace_raw_hardening', now(),
        'raw_session_a', 'q_raw_hardening', 'qv_raw_1',
        'custom_setting_rejected', now(), 4,
        '{"attempt_id":"attempt_shared","setting":{},"n_seeds":1.5,"base_seed":-2147483649}'::jsonb,
        '00000000-0000-4000-8000-000000000005', now()
    );

select attempt_count, answer_count, known_answer_count,
       incorrect_answer_count, unknown_answer_count, accuracy
from public.feedback_question_stats
where question_id = 'q_raw_hardening' and question_version = 'qv_raw_1';

select session_id, attempt_id, question_count
from public.feedback_session_summary
where session_id = 'raw_session_a' and attempt_id = 'attempt_shared';

select event_id, n_seeds, base_seed
from public.feedback_proposals
where event_id like 'raw_hardening_proposal_%'
order by event_id;

rollback;
```

For example, run these from the Supabase SQL editor or another trusted
service-role/reporting connection:

```sql
select *
from public.feedback_question_stats
order by answer_count desc, question_id;

select *
from public.feedback_session_summary
order by last_event_at desc;

select *
from public.feedback_comments
order by occurred_at desc;
```

These views are the **REPORT-001** starting point, not a public API. They use
`security_invoker`, have no grants to `anon` or `authenticated`, and the base
table has forced RLS with no client policy. The reporting endpoint does not
expose or grant them to browsers.

Migration `14000` adds `feedback_authoritative_events`, a dynamic projection
that exact-joins each raw event on client-claimed `payload.release_id` plus the
top-level `question_id` and `question_version`. It never guesses a release from
the question ID and never mutates the raw event. Registering a release later can
therefore make historical events resolvable. For matched answers, the selected
letter is resolved through the registered choice and correctness is derived by
comparison with the registered correct letter; payload dimensions and
`is_correct` are retained only as mismatch diagnostics.

Migration `15000` replaces the four business RPC implementations while
preserving their parameters, return columns, and column order:
`feedback_report_summary`, `feedback_report_sessions`,
`feedback_report_questions`, and `feedback_report_comments`. They aggregate only
rows whose dynamic `registry_status` is `matched`. Their `release_id`, `family`,
`dataset_id`, `question_type`, and correctness come only from registry facts;
there is no fallback to client claims. The RPCs accept scalar `release_id`,
`family`, `question_type`, and `question_id` filters plus optional `from`/`to`
timestamps, and the comment RPC also accepts `category`. Dimension filters use
the authoritative columns. Business time bounds still filter client
`occurred_at`; `from` is inclusive and `to` is exclusive, and the Reports UI
sends UTC midnight boundaries.

Migration `16000` adds the fifth and sixth business RPCs,
`feedback_report_answers` and `feedback_report_proposals`, with the same common
content/date filters and authoritative dimensions. Both return only
registry-matched events ordered by descending `occurred_at`, then `event_id`.
Answer rows expose selected letter, optional client candidate/correctness,
canonical candidate, answer status, server-derived correctness, and context/
correctness mismatch flags. Proposal rows cover both proposed and rejected
settings and expose status, label, object-valued setting/inheritance as JSON
text, fail-safe nullable int32 seed fields, and rejection error type. The strict
Python client validates the exact schemas and JSON; the UI formats JSON safely,
paginates both surfaces, and refuses CSV export from a truncated first page.

Migration `17000` adds the eleventh protected RPC,
`feedback_report_business_snapshot`. One Streamlit refresh sends one GET to the
report Edge Function; the Edge makes one PostgREST call; and this stable SQL
function assembles Summary, Sessions, Questions, Answers, Proposals, and
Comments in one statement and therefore one PostgreSQL MVCC snapshot. The
singleton row carries `business_snapshot_v1`, server `snapshot_at`, the embedded
`registry_v1`/`detail_v1` booleans and registry counts, and one
`pages_json` text value containing exactly six page envelopes. Text is used at
the RPC boundary; Edge parses it for basic shape but embeds the validated
PostgREST array bytes unchanged, so outer bigint registry counts and inner
report numbers never round-trip through JavaScript `Number`. Six bounded
materialized staging CTEs retain exact totals while allowing only the first
`p_limit` stable rows per view to reach JSON conversion and UTF-8 byte
measurement. Complete-row page budgets are 64 KiB Summary, 256 KiB each for
Sessions/Questions/Answers, 2.5 MiB Proposals, and 128 KiB Comments, with a
final 4 MiB `pages_json` cap. The strict Python client checks the exact view set,
schemas, common limit/zero offsets, and cross-page count conservation, then the
app replaces all six saved pages atomically. A failed refresh keeps the previous
complete business snapshot and its authority metadata. Row- or byte-truncated
inner pages retain exact totals and remain ineligible for CSV export.

Migration `18000` is a forward-only replacement of the six business functions
and the snapshot. It appends `p_session_id` and `p_attempt_id` after each
historical positional prefix—after `p_category` for Comments and after `p_limit`
for the snapshot—so old callers that omit the new arguments retain unfiltered
behavior. All six business functions filter authoritative `events.session_id`
and normalized `events.report_attempt_id`; the snapshot passes both values to
all six lateral calls. The migration drops the previous exact signatures before
recreating them, avoiding ambiguous defaulted overloads in PostgREST.

Migration `19000` extends the append-only event contract without changing wire
schema `1.0`. `question_presented` stores the exact release/attempt,
decision/policy IDs, mode, propensity, source, and position emitted by initial,
Next, Random, or manual navigation. `question_reaction_submitted` stores one
strict post-reveal surprise boolean with attempt context. The table constraints,
atomic ingest RPC, Edge validator, exact-retry behavior, and content-conflict
behavior agree on both types. Presentation rows are durable evaluation facts;
the current repository does not yet expose a presentation/propensity aggregate
report.

Migration `20000` adds `feedback_report_surprise_questions` and
`feedback_report_surprise_quality` as the twelfth and thirteenth protected RPCs.
The question page uses only registry-matched events and counts the first valid
post-answer rating for each exact session/attempt/release/question/version. It
returns answered attempts, yes/no/rating counts, rating coverage,
`observed_surprise_rate`, and a Beta(1,1) `posterior_mean`, with no answer key or
GT metric. The quality row classifies every raw reaction as valid, orphan, or
duplicate; orphans split into registry-unmatched, invalid-payload, and
missing-prior-answer counts, with explicit conservation booleans. Both RPCs
accept the eight release/family/type/question/time/session/attempt filters and
are independent SQL statements—not members of `business_snapshot_v1` and not a
shared snapshot with each other.

The OBS-001B migration adds the ingestion RPC,
`feedback_report_ingestion_summary`, and the STATS-002B reporting migration
replaces it under the same signature with conflict-aware columns. It accepts
optional `from`/`to` bounds, applies them to the server request `started_at`, and
also accepts an optional UUID `request_id` for exact outcome correlation.
`request_id` is not accepted by the business RPCs and is not exposed in the
Reports UI. The ingestion RPC returns one aggregate row. Content filters are
intentionally unsupported because rejected requests do not have trustworthy
release or question dimensions.

That exclusion includes `session_id` and `attempt_id`: ingestion observability
and Registry quality are unavailable while either identity filter is active.
Authority status supports no filters at all, and exact-event resolution supports
only its required `event_id`; neither auxiliary verifier surface accepts an
identity filter.

`14000` also adds `feedback_report_registry_quality`, a single-row aggregate for
missing/unknown release, question membership, invalid letter, candidate/context/
correctness mismatch, and unmatched comment/proposal signals. The exact
`feedback_report_event_resolution(event_id)` RPC returns one canonical row even
for a nonexistent ID and is reserved for verifier/operator use; the Reports UI
does not expose event-ID lookup. `15000` initially adds
`feedback_report_authority_status` for its aggregate cutover; `16000` drops and
recreates that function in the same transaction as both detail RPCs. The final
exact seven-column row identifies `registry_v1`, current registry counts, and
`detail_v1`, with explicit authoritative booleans for both cutovers. It is a
hosted preflight marker, not a browser dashboard tab.

Only `service_role` may execute all thirteen RPCs. The `feedback-report` Edge
Function holds that role, checks `FEEDBACK_REPORT_TOKEN`, allowlists query
parameters, and returns a strict page envelope with `view`, `rows`, exact
`total`, `limit`, and `offset`. Table pages are capped at 1,000 rows per request.
The Streamlit app disables CSV whenever `total` exceeds the rows in its first
page, so a partial result is never exported as complete.

The business Summary RPC continues to return `ingestion_failure_rate = null`
and `ingestion_failure_rate_available = false`. The separate ingestion-summary
RPC exposes only its explicitly recorded subset and fixes
`end_to_end_coverage_available = false`; it does not make the business field an
end-to-end metric.

The repository-local private Streamlit Reports app has ten tabs: Summary,
Sessions, Questions, Answers, Proposals, Comments,
**Ingestion observability**, **Registry quality**, **Surprise**, and **Data
quality**. The
first six use the shared authoritative filters. Registry quality comes from its
dedicated RPC instead of inferring unknown releases from an empty business
filter. Data quality combines authoritative/registry signals with recorded
rejection/failure, verified idempotent retry, legacy-unclassified duplicate,
confirmed conflict, and coverage signals. The app no longer requests the status
RPC separately before rendering: its strict snapshot parser requires the
embedded `business_snapshot_v1`, `registry_v1`/`detail_v1` authoritative facts,
counts, `snapshot_at`, and all six page contracts. Ingestion observability and
Registry quality remain independent requests, do not share that MVCC snapshot,
and are skipped while business content or identity filters are active. Surprise
question/quality requests are also independent of the business snapshot and one
another; both use the active eight shared filters and display their own
load/error state. The app cannot turn
persisted-subset metrics into end-to-end health. This describes repository-local
deployment inputs and UI behavior; neither `17000`–`20000` nor the hosted Reports
stack has been accepted on a real PostgreSQL/Supabase project yet.

## Storage and security behavior

- `event_id` is the primary key. Exact seven-field logical replays are verified
  idempotent duplicates; different logical content is a 409 conflict, preserves
  the first write, and atomically rejects every new event in that batch.
- `sequence` stores the original stable session sequence from each event,
  including when that event first arrives in a one-event envelope.
- `feedback-ingest` calls the SECURITY DEFINER `feedback_ingest_events` RPC with
  the service-role credential held in its environment. After writer lockdown,
  the service role cannot insert directly into `feedback_events` but retains
  `SELECT`; every application write therefore participates in the RPC's ordered
  transaction locks and exact JSONB comparison. `feedback-report` uses the same
  environment-only role for read-only RPC calls.
- Confirmed conflicts are written to the private, forced-RLS, append-only
  `feedback_event_conflicts` sidecar. Its only facts are `request_id`,
  `event_id`, `first_ingest_request_id`, `comparison_revision`, and
  server-controlled `detected_at`; it stores no payload or content hash and
  cannot rewrite the accepted event.
- Sanitized request outcomes live in their own private, append-only table. A
  failed outcome write is fail-open and cannot change the ingestion receipt;
  schema 1.1/`obs2` supplies conflict classification while legacy 1.0 duplicate
  counts stay unclassified.
- The event table grants the service role `SELECT` but, after lockdown, not
  direct `INSERT`. The separate outcome table still grants the Edge the minimal
  `SELECT`/`INSERT` needed for fail-open observability.
- A database trigger rejects `UPDATE`, `DELETE`, and `TRUNCATE`, keeping event
  history append-only even if application code accidentally attempts mutation.
- RLS is enabled and forced, with no `anon` or `authenticated` policies.
- Report views preserve underlying RLS with `security_invoker` and explicitly
  revoke public browser roles.

The SQL editor/project owner can deliberately change the schema through a new
migration when retention or administrative deletion is later required.
