# ArchitectureIQ Feedback Reports

This is the separate, internal, read-only Streamlit dashboard for uploaded
ArchitectureIQ feedback. It does not run inside the question inspector and it
does not read the inspector's browser-session state. Its data source is the
protected `feedback-report` endpoint backed by Supabase.

The code, protected backend contract, client, UI, and tests are available
locally. They have **not** been deployed to a hosted Supabase project or a
private Streamlit app yet, so the repository alone does not provide live data
or cross-session persistence.

## Run the app

Install the inspector dependencies, then start the dedicated entrypoint:

```bash
pip install -e ".[inspector]"
.venv/bin/python -m streamlit run tools/feedback_reports/app.py
```

Without configuration, the page remains read-only and shows setup guidance; it
does not fall back to demo data or claim that there are zero production events.

## Configure read access

Configure the app in an untracked `.streamlit/secrets.toml` or the corresponding
Streamlit Cloud secrets:

```toml
[reports]
endpoint = "https://YOUR_PROJECT_REF.supabase.co/functions/v1/feedback-report"
token = "THE_DEDICATED_REPORT_TOKEN"
timeout_seconds = 10
```

Equivalent environment variables are:

```text
ARCHITECTURE_IQ_REPORTS_URL
ARCHITECTURE_IQ_REPORTS_READ_TOKEN
ARCHITECTURE_IQ_REPORTS_TIMEOUT
```

The value supplied as `[reports].token` / `ARCHITECTURE_IQ_REPORTS_READ_TOKEN`
must match the Edge Function's `FEEDBACK_REPORT_TOKEN`. This is a dedicated
read token: do not reuse `FEEDBACK_INGEST_TOKEN`. The Supabase service-role key
stays only in the hosted Edge Function environment and must never be copied to
Streamlit secrets, client code, Git, a URL, or a CSV.

The report token authenticates **Streamlit to the Edge Function**; it does not
authenticate a person opening the Streamlit page. The Reports app has no
built-in maintainer login. A hosted deployment must therefore use a separate
private Streamlit app with platform-level access restricted to the maintainers.
Do not publish this entrypoint at an anonymously accessible URL: any page
visitor could otherwise view and export the returned feedback. Private page
access is a hard deployment acceptance condition, not an optional privacy
enhancement.

The Python client also refuses HTTP redirects so its Bearer token cannot be
forwarded to a different origin. Configure the final Edge Function URL rather
than a vanity or redirecting URL.

Deployment instructions for the migration and Edge Function are in
[`supabase/README.md`](../../supabase/README.md).
For an existing live receiver, deployment is deliberately staged: apply the
atomic-RPC/outcome-1.1 migration while legacy inserts are still allowed, deploy
and smoke-test the RPC-aware ingestion Edge Function, then apply the direct
writer lockdown and report migrations in order through `13500` raw-view
hardening, `14000` registry schema, `14500` reviewed release data, `15000`
authoritative aggregates, `16000` authoritative answer/proposal details, and
`17000` atomic six-view business snapshots, then the forward-only `18000`
session/attempt filter cutover, `19000` post-result reactions, and `20000`
surprise aggregates. Deploy the matching report Edge/client/app only after
`20000`. Applying lockdown
before the new ingestion Edge makes stale instances fail closed; it is safe to
retry but creates avoidable downtime. None of these hosted steps has been
accepted against a real project yet.

## Dashboard behavior

The dashboard has six business tabs, three independent quality/observability
tabs, and a tenth locally derived quality tab:

- **Summary** — sessions, attempts, answers, authoritative accuracy, propose
  usage, comments, and supporting setting/run counts;
- **Sessions** — one row per `session_id` / `attempt_id` after filtering;
- **Questions** — one row per question version and reporting context;
- **Answers** — individual registry-matched answer events with canonical choice,
  server-derived correctness, client claims, and mismatch flags;
- **Proposals** — individual proposed/rejected setting events with safe setting
  JSON, inheritance, seed, label, and rejection context;
- **Comments** — categorized individual comment rows;
- **Ingestion observability** — one aggregate row for the persisted subset of
  authenticated POST request outcomes;
- **Registry quality** — authority coverage, unresolved identities/answers, and
  client/server mismatch counts over raw events;
- **Surprise** — per-question post-result yes/no ratings, response coverage,
  observed rate, Beta(1,1) posterior, and a separate raw reaction quality row;
- **Data quality** — severity-labeled findings derived from the current Summary
  plus the saved ingestion and registry snapshots without another HTTP request.

The protected endpoint allowlists thirteen RPCs. Six are the underlying business
views; the atomic business-snapshot RPC is the app's refresh surface; ingestion
and registry quality back their own tabs; two surprise RPCs back one tab; and
authority status plus exact-event resolution remain verifier/operator surfaces
rather than dashboard tabs.

The six business tabs use the same registry-derived filter set before
aggregation.
The sidebar accepts one scalar value for each of `release_id`, `family`,
`question_type`, `question_id`, `session_id`, and `attempt_id`; empty values mean
no identity/content filter. Session and attempt are global drilldown filters:
when both are present, every matching event must satisfy both values.
**Apply filters** fetches a new six-view business snapshot, **Refresh** retries
the currently applied filter set, and **Reset** returns to all events.

Every business refresh is one Streamlit GET for
`feedback_report_business_snapshot`. The Edge Function makes one PostgREST RPC
call. Migration `17000` introduced the single-statement snapshot; forward
migration `18000` recreates it and all six underlying business RPCs with
`session_id`/`attempt_id` forwarding, so Summary and every detail page still
share one PostgreSQL MVCC snapshot.
The singleton row identifies `business_snapshot_v1`, exposes the server
`snapshot_at`, and embeds the exact `registry_v1`/
`business_reports_authoritative=true`, internally consistent registry counts,
and `detail_v1`/`detail_reports_authoritative=true` authority contract. A
`15000`- or `16000`-only backend cannot satisfy that response.

The six inner envelopes cross the PostgREST/Edge boundary in `pages_json` text.
Migration `17000` computes exact totals and deterministic ranks over all matches,
then materializes at most the requested first N complete rows per view before
JSON conversion and UTF-8 byte measurement. Per-page budgets are 64 KiB for
Summary, 256 KiB each for Sessions/Questions/Answers, 2.5 MiB for Proposals, and
128 KiB for Comments; the final `pages_json` has an exact 4 MiB cap. No field or
row is shortened. The Edge validates the PostgREST array and embeds its original
JSON bytes, preserving outer PostgreSQL bigint authority counts as well as the
inner text without a JavaScript `Number` round trip. The Python client then
parses strict JSON and requires exactly the six business views, one common
requested limit, zero offsets, valid page schemas, and summary/detail count
conservation before the app can replace any saved business state.

STATS-003 registers immutable release/question/choice facts from an attested
bundle. Business filters and dimensions use only exact release + question ID +
question-version membership, and answer correctness is derived from the
registered choice map. Client `is_correct`, family, dataset, and question type
remain raw audit claims; they never become fallback authority. Unmatched events
stay visible through Registry quality but are excluded from business accuracy.

REPORT-002 migration `16000` adds paginated `feedback_report_answers` and
`feedback_report_proposals`. Both read only registry-matched events, use the
same authoritative content/date filters as the aggregate tabs, and sort by
newest event then event ID. Answer rows expose the client choice/correctness
beside the canonical candidate, server result, answer status, and mismatch
flags. Proposal rows cover proposed and rejected settings; object-valued setting
and inheritance payloads are returned as JSON text, optional seeds fail safe to
nullable PostgreSQL integers, and the strict client/UI validates and safely
formats the JSON. CSV remains disabled for a truncated first page.

Forward migration `20260712018000_feedback_session_attempt_filters.sql` adds
server-authoritative `session_id` and normalized report `attempt_id` predicates
to Summary, Sessions, Questions, Answers, Proposals, and Comments, and forwards
the same two filters through `feedback_report_business_snapshot`. The parameters
are appended after each historical positional prefix (after `category` for
Comments and after `limit` for the snapshot), so callers that omit them retain
the previous unfiltered behavior. The migration drops and recreates one exact
signature per function to avoid ambiguous PostgREST overload selection.

Only after the complete snapshot and its embedded authority metadata validate
does the app atomically replace all six saved business pages and the displayed
server `snapshot_at`. A failed business refresh reports the error and continues
to render the previous complete snapshot; it never mixes old and new pages or
clears the authority metadata attached to that prior snapshot.

After a successful business swap, the app requests Ingestion observability and
Registry quality independently. Failure of either auxiliary request clears only
that auxiliary result and does not roll back or hide the validated business
snapshot. Those two quality RPCs do not share the business MVCC snapshot or
necessarily share a snapshot with each other. Data quality reuses the saved
business and auxiliary rows and creates no extra request.

The app also requests Surprise questions and Surprise quality independently
after a successful business swap, and offers a Surprise-only refresh. These two
statements are not part of `business_snapshot_v1` and need not be atomic with
the business snapshot or with each other. Both accept the same eight
release/family/type/question/session/attempt/time filters. The question RPC is
paginated only by PostgREST `limit`/`offset` with an exact total; the quality RPC
is one aggregate row and has no SQL limit parameter.

Surprise questions contain exactly the authoritative question identity,
answered-attempt count, first valid yes/no rating counts, coverage, raw observed
rate, Beta(1,1) posterior mean, and first/last rating times. Not clicking is
missing feedback, never an implicit “not surprised” vote; therefore the raw
observed rate is N/A when no rating exists, while the neutral posterior is 50%.
The quality row proves both `raw = valid + orphan + duplicate` and
`orphan = registry-unmatched + invalid-payload + missing-prior-answer`, while
showing unknown-release reactions as a subset of registry-unmatched reactions.
Neither response exposes answer keys, GT metrics, or private cold-start feature
composition.

Each Surprise response has its own validated CSV. A complete page can be
exported with formula-safe UTF-8 CSV serialization; a partial question page is
shown but its CSV button remains disabled until filters make the result
complete.

If any release, family, question-type, question-ID, session-ID, or attempt-ID
filter is active, the app does not request ingestion observability or all-event
Registry quality. Their tabs explain that they are unavailable until those
filters are cleared. Authority status supports no filters, and exact-event
resolution supports only `event_id`; neither verifier/operator surface accepts
session/attempt identity filters. This
avoids attributing an authenticated rejection or unresolved raw event to a
particular authoritative release/question filter.

The Data quality tab continues to show business-only findings when ingestion is
skipped or fails, and explicitly marks unavailable auxiliary evidence. It warns
on unresolved registry identity/answers, client/server context or correctness
disagreement, and ordinary recorded client rejection,
errors on recorded service failure or confirmed conflicting event IDs, reports
verified exact retries as information, and warns separately about legacy
duplicate IDs that predate server comparison and remain unclassified. Confirmed
conflicts preserve the first write and reject the whole incoming event batch.
The tab never produces a green overall-health result: coverage is currently
incomplete, and the two independent auxiliary rows remain non-atomic relative
to the six-view business snapshot and use different time semantics. Frequency
thresholds remain future STATS-002 work.

The optional date control creates UTC bounds shared by both query paths. Start
dates are inclusive at `00:00:00Z`. The user-facing end date is inclusive as a
calendar day and is sent to the endpoint as the exclusive start of the next
UTC day. For example, July 11 through July 12 becomes:

```text
from=2026-07-11T00:00:00Z
to=2026-07-13T00:00:00Z
```

The six business RPCs apply those bounds to event time:
`occurred_at >= from` and `occurred_at < to`. Accuracy uses only answers resolved
through the registered letter/candidate map and server-derived correctness;
unresolved answers are counted separately outside the denominator. Propose
usage is attempts with a proposal divided by solve attempts.

The dashboard sends only `from` and `to` to the ingestion RPC and applies the
same half-open bounds to the server-controlled request `started_at`. The
protected RPC and Python client additionally support a strict UUID `request_id`
filter only for ingestion summary. It performs exact outcome correlation and is
reserved for operator tooling and the hosted verifier; the Streamlit UI has no
request-ID input. The filter value is the ingestion POST receipt UUID, not the
separate correlation ID in a `feedback-report` response envelope. Business
report views reject that filter.

The ingestion RPC always returns one row and aggregates only persisted outcomes
with `included_in_rate = true`: success (including verified duplicate-only),
client rejection (including HTTP 409 event-ID conflicts), and service failure.
A valid but unmatched `request_id` therefore returns the normal all-zero row
with null timestamps/rates and false availability.

Server classification uses exact PostgreSQL JSONB equality over
`schema_version`, `event_id`, `event_type`, `session_id`, `question_id`,
`question_version`, and the full recursive `payload`. It excludes
`occurred_at`, `sequence`, and trace/request/receive metadata. A matching reuse
is an idempotent duplicate; different logical JSONB is a conflict whose HTTP
409 batch inserts no new feedback events and preserves the first write. The
private append-only `feedback_event_conflicts` audit stores only request/event
correlation and the first ingest request ID, never payload or a content hash.

Outcome schema 1.1 with observer revision `obs2` distinguishes exact retries
from conflicts. It reports conflict-request count, accepted events,
non-conflicting duplicate results, verified idempotent duplicates, conflicting
events, correlated private-audit rows, and total event-ID reuse. Append-only 1.0
outcomes remain queryable;
their duplicate counts are split into `unclassified_duplicate_event_count`
because the old receiver did not compare logical content. They are not
retroactively counted as either verified retries or conflicts.
`classified_event_count` sums all events in successful or conflict-classified
requests, including new events withheld when another event made the batch fail
atomically. Its rates are:

- recorded request failure rate = `(client_rejection_count +
  service_failure_count) / recorded_request_count`;
- duplicate event-ID rate = `duplicate_event_count /
  (accepted_event_count + duplicate_event_count)`;
- event-ID reuse rate = `event_id_reuse_count / classified_event_count`;
- classified conflict rate = `conflicting_event_count /
  (idempotent_duplicate_event_count + conflicting_event_count)`.

`duplicate_event_count` contains non-conflicting duplicates from both revisions
and equals verified-idempotent plus legacy-unclassified duplicates. The
classified conflict denominator intentionally excludes legacy rows whose content
was never compared.
`conflict_audit_event_count` joins the private append-only sidecar to the selected
recorded outcome request IDs and must equal `conflicting_event_count`.

Each rate is N/A when its own denominator is zero. `recorded_rate_available`
only says whether the recorded-request denominator is present;
`end_to_end_coverage_available` is always false.

These are recorded-subset metrics, not an end-to-end ingestion rate. The
business Summary fields `ingestion_failure_rate` and
`ingestion_failure_rate_available` remain N/A/false. Coverage excludes
authentication 401, method 405, missing configuration, requests that never
reached the Edge Function, and outcome writes lost to timeout, schema/HTTP
failure, or database outage. Schema 1.1 duplicates are verified matches;
schema 1.0 duplicates remain visibly unclassified. Outcome rows contain no
tokens, request bodies, IP addresses, comments, or settings, and their fail-open
persistence cannot alter the original ingestion receipt.

## Page limits and CSV downloads

The app requests the first 100, 250, 500, or 1,000 matching rows per table; the
endpoint returns an exact `total`. The stable prefix may contain fewer rows than
the requested limit when the view's byte budget is reached. A page is complete
only when its returned row count equals that total. Whether truncated by row
limit or byte budget, the UI reports `shown of total` and disables CSV export.
Narrow the filters until the result is complete; the app never labels a partial
page as a full CSV.

Complete results can be downloaded from each tab. CSV files use stable columns,
UTF-8 with a BOM for spreadsheet compatibility, deterministic UTC filenames,
and canonical JSON for array values. User-controlled strings beginning with
spreadsheet formula prefixes (`=`, `+`, `-`, or `@`, ignoring leading spaces)
are prefixed with an apostrophe during export. This changes only the downloaded
cell, not the stored comment or identifier.

## Verification

Focused offline checks cover the strict client/envelope contract, bearer-token
redaction, RPC/query allowlists, UI rendering, date semantics, page truncation,
and CSV safety:

```bash
.venv/bin/python -m pytest -q \
  tests/test_feedback_reports.py \
  tests/test_feedback_reports_ui.py \
  tests/test_feedback_reports_app.py \
  tests/test_feedback_report_backend.py \
  tests/test_feedback_rollout_preflight.py
```

These tests use fixtures and mocked/local endpoints. Passing them does not mean
the migration or functions have been deployed to a hosted project.

### Rollout preflight

Before deploying this app or its report endpoint, run the cumulative local
static check for the final rollout phase from the repository root:

```bash
.venv/bin/python tools/feedback_rollout_preflight.py \
  --phase report-app
```

The report-app phase also rechecks the earlier expand, ingest-cutover, and
writer-lockdown inputs. In particular, it dynamically compares the SQL return
columns in migration `13000` with the strict Python client, verifies the
allowlisted report views and read-only Edge wiring, and verifies that migration
`13500` preserves every existing raw-view named-column prefix while appending
only the three answer-quality columns and keeping the proposal view's exact
16-column shape. It also verifies the attested `14000/14500` registry pair,
the four authoritative aggregate RPCs and status marker in `15000`, both strict
detail RPCs in `16000`, the final seven-column dual-revision status schema, and
the `17000` single-statement business snapshot RPC. It also verifies the
forward-only `18000` signature replacement, preserved historical positional
prefixes, identity predicates/forwarding, auxiliary-view exclusions, and hosted
evidence flag, plus the strict
`business_snapshot_v1`/`pages_json` client/app contract, and exact thirteen-view
Edge/Python allowlist. It fingerprints every enumerated cumulative
rollout/compatibility input, including the production feedback client,
rollback-only database verifier, hosted roundtrip verifier, Inspector
pending/recovery/comment implementation, runtime
release attestation, checker, and dependency metadata. Baseline migrations are
fingerprinted for reproducibility, not as an
instruction to reapply them. The current ingestion-summary contract has 22
columns, including
`conflict_audit_event_count`; old and new strict clients are not mutually
compatible at that `13000` schema boundary. The `13500` migration does not alter
the protected Python client schema. Static parsing does not prove PostgreSQL
column types, grants, RLS, triggers, constraints, or deployed registry content;
run the staging database verifier below against the actual target.

This command performs no network request and accepts no credentials. A static
pass is still `overall=UNVERIFIED`, `hosted_verified=false`, and
`deploy_ready=false`. Use `--json` for automation and `--require-hosted` when a
pipeline must fail until separate hosted evidence has been attached; the latter
returns exit code 2 because this intentionally offline tool can never produce
that evidence. See the staged rollout sequence in
[`supabase/README.md`](../../supabase/README.md).

### PostgreSQL staging acceptance

After migrations are applied, use a staging owner/admin DSN to run the direct
catalog and rollback-only constraint verifier before the permanent endpoint
roundtrip:

```bash
.venv/bin/python -m pip install -e '.[postgres-acceptance]'
export ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN='INJECT_STAGING_ADMIN_DSN'
export ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_TARGET='architecture-iq-staging'
.venv/bin/python tools/feedback_postgres_acceptance.py \
  --confirm-staging > postgres-staging-acceptance.json
unset ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN
```

The DSN is environment-only and is omitted from JSON and diagnostics. The
explicit target label cannot itself prove which database the DSN names, so the
operator must verify that mapping. The tool checks migration order, 13
application-function signatures/attributes/ACLs, forced RLS with no policies,
table grants, exact
trigger masks, named constraints, exact hosted 60-question/180-choice registry
content hash, and dual authority revisions. Append-only and deferred-registry
counterexamples are isolated by savepoints, and the complete transaction is
rolled back and closed without a commit. Keep its sanitized JSON with the same
Git SHA and preflight fingerprint used for deployment.

This proof is database-scoped. It does not prove Edge deployment, endpoint
authentication or receipts, restart persistence, advisory-lock concurrency,
or real-load MVCC behavior. The hosted roundtrip covers the application path;
the remaining concurrency/load cases need separate staging runs. Full command,
exit-code, role, and evidence-boundary details are in
[`supabase/README.md`](../../supabase/README.md).

### Hosted roundtrip

After deploying both Edge Functions and the private Reports app, run the opt-in
hosted verifier. It reuses the production feedback and Reports clients, and it
refuses to write until the local quiz bundle passes both runtime release
attestation and the feedback-registry exporter validation:

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write
```

The default bundle is `examples/quiz_demo/bundle`; use `--bundle PATH` for a
different fully published release. `build_feedback_registry()` rebuilds and
validates the publisher output, while the runtime manifest and selected
`question.json` are attested again. The first registry row in canonical sorted
order is the deterministic probe membership. The CLI never constructs a fake
release, question ID, version, answer, or candidate. The resulting
`registry_id` is printed and included in JSON output.

The tool reads only the four documented feedback/report environment variables;
token arguments are intentionally unsupported. `--confirm-permanent-write` is
mandatory because accepted events, request outcomes, and private conflict
audits cannot be cleaned up by this tool. Run against staging first.

Before the first POST, the dedicated `feedback_report_authority_status` row must
use the exact seven-column contract and report
`authority_revision=registry_v1`,
`business_reports_authoritative=true`, registered inventory counts large
enough to contain the locally attested bundle,
`detail_revision=detail_v1`, and
`detail_reports_authoritative=true`. The first pair proves the `15000` aggregate
cutover; the second proves that `16000` recreated the marker together with both
authoritative detail RPCs. The verifier also queries Answers and Proposals with
a random nonexistent question ID and requires complete empty pages, proving the
protected Edge allowlist and RPC reachability without depending on historical
rows. It then requests one `business_snapshot_v1` with a different random
nonexistent question ID and limit one. The embedded authority/count metadata and
server `snapshot_at` must validate, Summary must be the exact all-zero singleton,
and the other five pages must be complete and empty. This pre-write negative
control proves the protected `17000` route without relying on existing business
data; a successful result emits `business_snapshot_verified=true`. Then
`feedback_report_event_resolution` must
return its exact single-row `not_found/not_found` sentinel for every event ID
that should be new.
`--resume` instead requires the original comment ID to resolve to the complete
expected registry/session identity; a missing or different event fails before a
write. The answer, proposal, and comment batch IDs must be either all absent or
all exact, never a partial batch. An opt-in mixed-batch withheld ID must be
absent. A random request UUID must also return the all-zero/null ingestion row.

After the authoritative successful batch is visible, the verifier performs the
REPORT-002 identity-filter proof against the recreated `18000` snapshot. The
real uploaded `session_id + attempt_id` pair must return the exact Summary,
Sessions, Questions, Answers, Proposals, and Comments rows for that trace. A
random wrong session with the real attempt, and the real session with a random
wrong attempt, must each return the exact empty six-view snapshot. Only then may
the JSON evidence set `session_attempt_filters_verified=true`.

The event payload deliberately claims the correct release but a wrong client
family and question type. The answer additionally claims the inverse
`is_correct` value. Successful verification therefore requires the exact-event
RPC—not the client payload—to return:

- the attested `registry_id`, release, question ID/version, family, dataset, and
  question type;
- `client_context_mismatch=true` for comments and answers;
- `answer_status=not_answer` with no answer fields for comments; and
- `answer_status=resolved`, the canonical candidate, server-derived
  `authoritative_is_correct`, and `client_correctness_mismatch=true` for the
  answer.

This exact event-ID proof remains valid when the real question already has
historical answers. The verifier intentionally does not require any of the six
business reports to equal one or two.

The first comment upload requires HTTP 200 and one canonical request UUID in
both header and body. Fresh mode requires `1/0/0/0`; resume requires
`0/1/0/0` for accepted/duplicate/conflict/rejected. Its exact request outcome,
exact authoritative event resolution, and a nonexistent request-ID negative
control are polled together until they converge.

The next default phase sends one production `SessionTrace` containing an answer,
a proposed setting, and a comment in the same session and under the same attested
membership. Fresh mode requires `3/0/0/0`; the **same envelope unchanged** is
then replayed and must return `0/3/0/0`. Each POST gets a distinct canonical request UUID and an
independently persisted outcome. All three exact event resolutions must remain
unchanged across replay. If resume finds the complete batch already present,
both requests are duplicate-only and the result reports
`successful_batch_verified=true` but
`successful_batch_first_write_verified=false`. An interrupted run with only the
original comment can still observe the batch's fresh acceptance. Explicit
`--skip-successful-batch-probe` leaves both proof fields false.

The default conflict phase reuses the original comment ID with changed text. It
requires structured HTTP 409 `EVENT_ID_CONFLICT`, counters `0/0/1/1`, a new
canonical request UUID, the exact persisted conflict/audit outcome, and a
byte-for-byte unchanged authoritative resolution for the first write.

For an additional all-or-none check, opt in to one more request:

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write \
  --include-mixed-batch-probe
```

That request is one two-event trace ordered as
`[existing changed-text conflict, deterministic new comment]`. It must return a
canonical request UUID distinct from every earlier request and structured HTTP
409 counters `0/0/1/2`. Its exact outcome must retain the existing conflict and
audit metrics. The original event resolution must remain unchanged and the
fresh withheld event ID must still return `not_found`, proving all-or-none
rejection without relying on shared aggregates.

Immediately before the first permanent POST, the verifier prints the safe
`run_id` to stderr and flushes it. Keep that value: if the upload succeeds but
the terminal, network response, or report polling is interrupted, it is the
identifier required for recovery. Events record their actual UTC construction
time. Resume relies on deterministic `event_id` values and the
database's first-write-wins idempotency; it does not backdate future smoke runs
to a fixed timestamp.

If a request or terminal is interrupted, rerun the exact deterministic event:

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write \
  --run-id RUN_ID_FROM_THE_FIRST_ATTEMPT \
  --resume
```

Compatibility/recovery without the conflict request must be explicit:

```bash
.venv/bin/python tools/feedback_reports/verify_hosted_roundtrip.py \
  --confirm-permanent-write \
  --run-id RUN_ID_FROM_THE_FIRST_ATTEMPT \
  --resume \
  --skip-conflict-probe
```

Resume first verifies the server-derived registry/session identity, then safely
replays the same event ID. A generic 2xx or newly accepted original event is not
enough. Only a first run whose exact-event preflight was `not_found` can prove
that the currently configured ingestion and report endpoints share the same
durable store for a new write.

Resume still performs two successful-batch requests (duplicate-only when the
batch already exists) and a new conflict request by default; it is not a
read-only recovery. Adding
`--include-mixed-batch-probe` to the resume command also submits a new mixed
trace and requires `mixed_batch_verified=true`. To reduce the recovery
footprint, explicitly add `--skip-successful-batch-probe` and/or
`--skip-conflict-probe`. The corresponding verification fields remain false and
the run does not count as complete acceptance. The conflict skip and mixed
include flags are mutually exclusive.

A fresh default run permanently adds four events attached to the selected real
registry membership: the original comment plus the answer/proposal/comment
batch. It records the original single-event request, two successful batch
outcomes, and a rejected
conflict outcome/private audit row; the optional mixed phase records another
rejected conflict outcome/private audit row.
Fresh and resume runs therefore affect global duplicate/failure/conflict
metrics. Save all four default request UUIDs (five with the mixed probe). If
fail-open outcome persistence is lost, a write or 409/audit may have committed
even though verification times out, and retrying creates another permanent
outcome footprint.

The default probe proves one registered membership, deliberate client/server
disagreement, single-event ingestion, a normal three-event accepted/idempotent
replay, exact authoritative Answers/Proposals rows, and single-event 409
preservation. The opt-in trace proves all-or-none rejection only for its tested
`new + conflict` batch; it does not prove
`duplicate + conflict`, arbitrary concurrency, requests that never reached
Edge, or end-to-end coverage. No real hosted deployment acceptance run has been
completed yet.

The exact-event RPC attests individual registry projection, while the strict
seven-column authority-status row fingerprints both the `15000` aggregate and
`16000` detail cutovers. The two protected detail negative controls and the
`17000` empty six-view snapshot negative control must also pass.
`business_snapshot_verified=true` records only that hosted pre-write proof;
`detail_reports_verified=true` is emitted only after the successful batch's
real answer and proposal rows match their uploaded events;
`session_attempt_filters_verified=true` additionally requires the positive
identity drilldown and both negative empty-snapshot controls above. Hosted
catalog and
grants/RLS count as evidence only when the direct staging database verifier has
passed for the same deployed revision; concurrency and real-load
detail/snapshot behavior still require separate deployment runs.

For a real rollout, retain the authoritative JSON output together with the
rollback-only PostgreSQL acceptance and reviewed provider control-plane capture
using the retrospective ledger in
[`deployments/README.md`](../../deployments/README.md). The evidence must share
one release/source/provider/backend/origin deployment context and must not be
reused across deployment keys. Ledger activation is deliberately named
`ACTIVATED_REVIEWED`: the hosted verifier proves behavior over the configured
endpoints, while its current JSON does not carry endpoint origins. Mapping that
run to the declared origin hashes and provider project/deploy metadata is
therefore a separately reviewed operational binding, not provider-signed proof.
No real hosted evidence or ledger event exists for the current working tree.
