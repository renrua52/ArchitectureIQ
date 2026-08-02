# Question Inspector

> **Status: frozen.** No new product features. Use `frontend/quiz/` + BakeFile
> (`contracts/`) for the quiz. Collaboration notes:
> [`docs/FRONTEND_BACKEND.md`](../../docs/FRONTEND_BACKEND.md).

Streamlit UI for inspecting ArchitectureIQ questions and trying custom training
settings. Existing question and candidate artifacts are never modified.

## Install

From the repo root:

```bash
pip install -e ".[inspector]"
```

## Run

```bash
.venv/bin/python tools/start_quiz.py
```

The launcher passes an explicit question path under `data/`; on a fresh clone
it materializes the bundled demo there first. Existing generated questions are
left unchanged. This explicit local workflow is separate from the hosted
default described below.

Optional: open a specific question or run first:

```bash
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX/q_XXXXXX
.venv/bin/python tools/start_quiz.py --question-run data/datasets/univariate_regression/sym_XXXXXX/questions/run_5q_2c_XXXXXX
```

Or run Streamlit directly:

```bash
streamlit run tools/question_inspector/app.py
```

Direct and hosted launches with no question argument read the
version-controlled `examples/quiz_demo/bundle/` **in place**. They do not copy
it into Streamlit's writable `data/` directory, so a stale local snapshot
cannot silently shadow the deployed release. Set `ARCHITECTURE_IQ_DATA_ROOT`
or use the sidebar's **Data root** field to open a different local root.

## Features

- **Question** tab — dataset panel, candidate cards, quiz flow, and file inspector
- **Prompt** tab — full rendered benchmark prompt (`prompt.txt`)
- **My session** tab — session-wide answer/accuracy, proposed/rejected setting,
  completed/failed custom-run, and comment summaries; safe answer, setting, and
  comment tables; full trace JSON download; and pending-versus-acknowledged
  upload status
- **Sidebar** — pick from existing questions, use surprise-aware **Next**, or
  choose **Random**; recommendation failure falls back to sequential navigation
- **Release attestation** — show the full release ID, raw manifest SHA-256,
  verified artifact count, fixed Streamlit entry path, and an allowlisted
  runtime Git SHA derived from the checkout and any deployment declaration;
  published questions follow manifest order
- **Quiz** — click **Select** on a choice to lock in your answer; metrics and ranked results appear immediately
- **Custom settings** — while solving, choose architecture, optimizer, loss, budget,
  batch size, and seed parameters, or inherit all editable values from Choice A/B/C;
  confirm to train and add a new curve
- **After answering** — use **View** or the info button on any choice to browse files (`summary.json` included once answered)
- **Surprise reaction** — after the result is revealed, record exactly one
  **Surprised / As expected** reaction for that question and quiz attempt;
  correctness, surprise, comments, and future likes remain separate signals
- **Session data** — presentations, answers, surprise reactions, and proposed
  settings are recorded in an append-only trace; download the complete JSON or
  upload it to an authenticated receiver
- **Recovery upload** — select a previously downloaded session JSON, validate it
  locally, and resume only that file's still-pending events in a separate outbox
- **Per-question comments** — add one categorized message to the trace, or upload
  that event immediately when the endpoint is configured

Set **Data root** to the directory containing `datasets/`. The no-argument
default is the version-controlled bundled release; an explicit `tools/start_quiz.py`
question path selects its containing local `data/` root. Questions from
dataset-scoped runs and legacy `<root>/questions/` appear in the sidebar.

Every present `quiz_manifest.json` is a complete release claim, not advisory
metadata. Before serving its questions, the inspector checks the exact manifest
schema, recomputes `release_id` without `generated_at`, validates source-run and
question identities, requires a sorted one-to-one physical artifact inventory,
rejects symlinks and special files, and hashes every regular file to verify its
declared size and SHA-256. It also strictly parses every published
`question.json` and recomputes its question version. This lightweight check runs
from file bytes rather than trusting manifest or artifact mtimes, so a same-size,
same-mtime edit is still detected.

The default bundled root requires a valid manifest and stops with no question
pool when attestation fails. Any other root may omit the manifest for local,
unversioned development; if a manifest is present there but invalid, it also
fails closed instead of silently serving mislabeled content. Unversioned local
questions never attach a release ID to feedback.

## Surprise-aware Next and post-result reactions

The current default release attests 60 questions. Its private cold-start catalog
iterates only `manifest.question_dirs()`—not every question under local `data/`—
and strictly reads each published `question.json`, choice `candidate_spec.json`,
and stored `results/summary.json`. It does not import generated code, rerun GT,
or write any score back into the bundle. Correct-letter/significance and
failed/excluded-seed checks form a non-negotiable validity gate. Parameter count,
depth, width, and optimizer-aggressiveness shortcuts support the current MLP and
Transformer specs; tied shortcut winners split one vote, all-equal shortcuts are
omitted, and unknown model plugins are not treated as zero-sized models.

On an attested release, **Next** excludes the current and already answered exact
release/question/version identities. It avoids repeating the current family when
another family remains, exploits the highest cold-start posterior 80% of the
time, and explores uniformly among the least-presented eligible questions 20% of
the time. Stable identity tie-breaking makes the result independent of choice or
filesystem ordering. If catalog validation, selection, or path resolution fails,
the UI keeps working by using sequential Next.

Every published-question navigation records a local `question_presented` event
with attempt/release, a random decision ID, policy version, mode, exact mixture
propensity, navigation source, and position. Initial, recommended Next, Random,
and manual picker navigation use explicit policy/source values. These events
provide an exposure denominator; an answer is not used as a proxy for exposure.
Current Next uses only the offline catalog and this attempt's local presentation
counts. It does not fetch SURPRISE-002 aggregates, personalize across sessions,
or claim a measured recommendation lift.

After an answer and its result are revealed, the player may record exactly one
**Surprised / As expected** value for that attempt. Surprise remains independent
of correctness, comments, and any future like signal. Presentation and reaction
events use the same trace, browser outbox, download/recovery, pending upload,
idempotency, and quarantine paths as the existing feedback events.

Runtime Git identity is read from the actual checkout with a fixed, read-only
`git rev-parse` call under a minimal environment that ignores inherited
`GIT_*` redirection and replacement objects. It is cross-checked against any
full 40-hex value supplied in
`ARCHITECTURE_IQ_GIT_SHA`, `GIT_COMMIT`, `COMMIT_SHA`, or `SOURCE_VERSION`.
Conflicting declarations, malformed configured values, an unavailable Git
checkout, or a malformed Git result display as `N/A`; an environment variable
cannot silently override a different checkout commit. This identifies source
bytes more reliably, but it still does not prove a Cloud branch, provider
deployment ID, deploy time, or environment URL. Those belong in the separate
post-deployment evidence ledger documented in
[`deployments/README.md`](../../deployments/README.md). No real ledger event has
been recorded for the current working tree.

Custom runs are stored in a per-session temporary directory (not under the shared
question artifact tree). Switching to another question clears that question's custom
runs for your session. Legacy on-disk folders under
`<question>/custom_settings/` are removed on question load when present. Every run
receives a unique sequence id and display name. At most two runs are retained per
question session: the newest run and the historical run with the lowest final loss.
Custom runs do not alter the choices, answer key, or score.
Selecting Choice A/B/C applies inheritance immediately, including the question's
ground-truth seed count and base seed; no separate apply action is required.

## My session

The **My session** tab summarizes the append-only trace across every question and
attempt in the current Streamlit browser session. Presentation decisions,
answer totals and accuracy, surprise yes/total, unique question versions,
setting proposals and validation rejections, completed and failed custom runs,
and comments remain visible when navigating between questions. The tables expose
only the fields needed for review rather than raw
event payloads, generated code, local paths, or exception traces.

**Download session JSON** exports the complete versioned trace. Batch upload
sends only events that are still **pending**, splitting them at the receiver's
500-event and 1 MiB limits. An event remains pending until the configured
endpoint returns the complete STORE-001 receipt: all four
non-negative integer `accepted` / `duplicate` / `conflict` / `rejected`
counters, no conflicts or rejections, a total matching the sent event count,
and one canonical RFC UUID (variant RFC 4122, version 1–5) repeated exactly in
the response header and body. Generic, partial, malformed, or mismatched 2xx responses leave events
pending for safe retry. **Acknowledged** means the current browser session
observed that strict confirmation—it is not itself a durable server-side
record.

If a batch contains an event-ID content conflict, the uploader retries that
batch one event at a time. Exact/new events continue to storage; only confirmed
conflicting IDs become **quarantined** and are excluded from later batches so
they cannot block new answers or settings. The sidebar exposes a separate JSON
download for quarantined events and retains their request IDs without placing
event payloads in status metadata.

The live trace, current attempt, acknowledgement IDs, and quarantine metadata
are mirrored into a checksum-protected IndexedDB browser outbox; endpoint URLs
and tokens are never stored there. A fresh Streamlit session restores a valid
copy, including presentation and reaction events, and fails closed with a
diagnostic download when the browser copy is corrupt. This mirror is saved at
Streamlit run boundaries rather than being a synchronous write-ahead log, so a
sudden crash can still precede the latest save. Use the JSON download for a
manual copy; authoritative cross-browser/server persistence still requires a
configured STORE-001-compatible ingestion endpoint.

## Restore a downloaded session JSON — locally implemented

The inspector can recover upload from a previously saved **Download session
JSON** file. Recovery is independent of the live browser trace and follows these
rules:

- the uploaded-file size passes the 10 MiB gate before `getvalue()` is called;
  no network request occurs until the complete document passes strict UTF-8 and
  JSON parsing, the exact wire schema, RFC 3339 timestamps, event-count,
  trace-ID, per-event schema, and session-ID checks;
- the file never imports acknowledged or quarantined state from an earlier
  browser. Its events enter a separate recovery outbox;
- canonical full-file content determines `recovery_id`. The current browser
  session stores pending, acknowledged, and quarantined progress under that ID,
  so selecting or retrying the same content sends only events still pending;
- a file that reuses an event ID with different logical content has a different
  `recovery_id` and is still sent to the authoritative receiver instead of being
  suppressed by the first file's browser state; and
- pending recovered events use the same 500-event / 1 MiB chunking, Bearer
  authentication, complete strict receipt checks, and HTTP 409 event-by-event
  isolation as the live uploader. Only confirmed conflicts are quarantined;
  retryable failures remain pending.

The focused recovery, feedback, and outbox test slice passes 167 tests, and the
Streamlit AppTest widget smoke passes. This is a local delivery only: it has not
been deployed to the public app or accepted against the real hosted receiver.

## Feedback upload

The inspector uses one versioned `session_trace` JSON envelope for every upload path:

- **Upload pending session events** sends only unacknowledged, non-quarantined
  events in receiver-sized chunks.
- **Upload comment** sends a one-event trace to the same endpoint.
- **Surprised / As expected** records one immutable reaction and immediately
  attempts the same one-event upload when transport is configured.
- **Recover a downloaded session** validates the complete saved envelope and
  sends only pending events from its content-scoped recovery outbox.

Every event has a stable `event_id`; both requests set `Idempotency-Key`, and the
ArchitectureIQ receiver classifies that ID against exact logical content. The
logical identity is the JSONB value of `schema_version`, `event_id`,
`event_type`, `session_id`, `question_id`, `question_version`, and the complete
recursive `payload`. It excludes `occurred_at`, `sequence`, and envelope,
request, and receive metadata, matching the inspector's in-memory idempotency
rule. JSON object key order does not matter; arrays and missing-versus-null
values do. Recursive JSON uses the cross-runtime lossless subset: integer-valued
numbers stay within ±9,007,199,254,740,991, strings contain no unpaired Unicode
surrogates, and identifier/comment limits count Unicode code points rather than
UTF-16 code units. Python rejects violations before opening the network, and the
Edge receiver repeats the check before calling its storage RPC. The session trace covers
`answer_submitted`, `question_presented`, `question_reaction_submitted`, `custom_setting_proposed`,
`custom_setting_rejected`, `custom_run_completed`, `custom_run_failed`, and
`comment_submitted`. Question reactions require `reaction="surprise"`, a strict
boolean value, `timing="after_reveal"`, and an attempt ID. Their deterministic
event ID makes the first response immutable and exact retries idempotent.
Question presentations require attempt/release/decision/policy identifiers,
`exploit|explore|fallback|manual` mode, a finite propensity in `(0, 1]`,
`initial|next|random|picker` source, and a positive position. It contains
structured specs and summaries, not generated Python files, local paths, or
exception traces. A setting rejected during form/spec validation is separate
from a valid setting whose training run later fails.
For a published question, the event payload includes `release_id` only when the
active data root, question path, question ID, and recomputed question version all
match the manifest.

The deployed receiver's normal HTTP 200 receipt contains non-negative
`accepted`, `duplicate`, `conflict`, and `rejected` counters plus one canonical
request UUID in both the response header and body. The low-level compatibility
helper accepts that complete strict structure on any 2xx response, while the
production STORE-001 receiver and hosted acceptance verifier expect HTTP 200.
A first upload is accepted;
replaying the same logical event is a verified duplicate with `conflict=0`.
Reusing an ID for different logical content returns HTTP 409
`EVENT_ID_CONFLICT`. The immutable first write is preserved, and a mixed batch
containing any conflict inserts none of its otherwise-new events. The inspector
isolates that batch: conflicting IDs are quarantined while withheld new events
are retried and uploaded under their unchanged IDs. Generic, partial, malformed,
or retryable failures remain pending. Repeating an unchanged HTTP 200 upload
remains safe. Generic counter-free 2xx acknowledgment exists only as an explicit
low-level compatibility opt-in for legacy collectors; the inspector does not use
it.

Configure the receiver in Streamlit secrets:

```toml
[feedback]
endpoint = "https://feedback.example.internal/v1/events"
token = "replace-with-a-server-side-bearer-token"
timeout_seconds = 10
```

Or set both `ARCHITECTURE_IQ_FEEDBACK_ENDPOINT` and
`ARCHITECTURE_IQ_FEEDBACK_TOKEN`, plus optional
`ARCHITECTURE_IQ_FEEDBACK_TIMEOUT`. The inspector considers upload configured
only when both a non-empty endpoint and Bearer token are available. If either is
missing, event capture and JSON download still work and the UI does not claim
that data was uploaded. Streamlit Community Cloud's local disk is not a durable
receiver.
The client refuses HTTP redirects so the Bearer token is never forwarded to a
different origin; configure the final ingestion URL directly.

A ready-to-deploy Supabase table and Edge Function for this contract live in
[`supabase/`](../../supabase/README.md). The local implementation includes a
private conflict audit and conflict-aware reports, but it has not yet completed
real hosted Supabase roundtrip/conflict acceptance.
When upgrading an existing receiver, apply the atomic ingest-RPC migration
first, deploy and verify the RPC-aware Edge Function second, and revoke direct
event-table inserts only afterward; the Supabase guide documents the staged
migration/Edge/lockdown sequence.

Deployment status (2026-07-12): this document describes the current local
implementation and contracts. The public Streamlit site still serves the old UI,
and neither runtime release attestation nor the upload, 409 quarantine, recovery,
surprise-aware Next/reactions, or Reports flows have completed real hosted
acceptance or been deployed there.
