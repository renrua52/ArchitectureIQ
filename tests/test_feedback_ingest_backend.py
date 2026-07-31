"""Executable storage and retry contract tests for feedback-ingest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "supabase/migrations/20260711000000_feedback_events.sql"
CONFLICT_MIGRATION = (
    REPO / "supabase/migrations/20260712012000_feedback_event_conflicts.sql"
)
EDGE_FUNCTION = REPO / "supabase/functions/feedback-ingest/index.ts"


def _node_with_type_stripping() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    major = int(
        subprocess.run(
            [node, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.lstrip("v")
        .split(".", maxsplit=1)[0]
    )
    if major < 22:
        pytest.skip("Node.js type stripping requires Node 22+")
    return node


def test_feedback_event_store_declares_durable_idempotency_invariants() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    conflict_sql = CONFLICT_MIGRATION.read_text(encoding="utf-8")
    source = EDGE_FUNCTION.read_text(encoding="utf-8")

    assert "event_id text primary key" in sql
    assert "alter table public.feedback_events force row level security" in sql
    assert "before update or delete or truncate" in sql
    assert "feedback_events is append-only" in sql
    assert "create function public.feedback_ingest_events(" in conflict_sql
    assert "all-or-none feedback-event insertion on conflict" in conflict_sql
    assert 'new URL("/rest/v1/rpc/feedback_ingest_events", supabaseUrl)' in source
    assert "p_events: envelope.events" in source


def test_feedback_ingest_executes_retry_and_idempotency_contract() -> None:
    node = _node_with_type_stripping()
    script = r"""
import assert from "node:assert/strict";

let handler;
let mode = "normal";
let outcomeMode = "normal";
const stored = new Map();
const outcomes = new Map();
const rpcCalls = [];
const backgroundTasks = [];

globalThis.EdgeRuntime = {
  waitUntil(promise) { backgroundTasks.push(promise); },
};

const flushBackground = async () => {
  const pending = backgroundTasks.splice(0, backgroundTasks.length);
  await Promise.all(pending);
};

globalThis.Deno = {
  env: {
    get(name) {
      return {
        FEEDBACK_INGEST_TOKEN: "ingest-secret",
        SUPABASE_URL: "https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY: "service-secret",
      }[name] ?? null;
    },
  },
  serve(callback) { handler = callback; },
};

globalThis.fetch = async (input, init) => {
  const endpoint = new URL(String(input));
  if (endpoint.pathname === "/rest/v1/feedback_ingest_request_outcomes") {
    assert.equal(endpoint.searchParams.get("on_conflict"), "request_id");
    assert.equal(init.method, "POST");
    assert.equal(init.headers.apikey, "service-secret");
    assert.equal(init.headers.Authorization, "Bearer service-secret");
    assert.equal(
      init.headers.Prefer,
      "resolution=ignore-duplicates,return=minimal",
    );
    const row = JSON.parse(init.body);
    assert.equal(typeof row.request_id, "string");
    assert.equal(row.schema_version, "1.1");
    assert.equal(row.method, "POST");
    assert.equal(row.authenticated, true);
    assert.equal(row.included_in_rate, true);
    assert.equal(row.observer_revision, "obs2");
    assert.ok(Object.hasOwn(row, "conflicting_event_count"));
    const serialized = JSON.stringify(row);
    for (const forbidden of [
      "ingest-secret",
      "service-secret",
      "Please clarify the evaluation wording.",
      "The curve failed to render.",
      "evt_",
      "anon_ingest_test",
    ]) {
      assert.ok(!serialized.includes(forbidden));
    }
    if (outcomeMode === "slow-fail") {
      await new Promise((resolve) => setTimeout(resolve, 250));
      return new Response("", { status: 503 });
    }
    if (!outcomes.has(row.request_id)) {
      outcomes.set(row.request_id, structuredClone(row));
    }
    return new Response("", { status: 201 });
  }

  assert.equal(endpoint.pathname, "/rest/v1/rpc/feedback_ingest_events");
  assert.equal(endpoint.search, "");
  assert.equal(init.method, "POST");
  assert.equal(init.headers.apikey, "service-secret");
  assert.equal(init.headers.Authorization, "Bearer service-secret");
  assert.equal(init.headers["Content-Type"], "application/json");
  assert.equal(init.headers.Prefer, undefined);

  const args = JSON.parse(init.body);
  assert.deepEqual(Object.keys(args).sort(), [
    "p_events",
    "p_request_id",
    "p_trace_created_at",
    "p_trace_id",
  ]);
  assert.equal(typeof args.p_request_id, "string");
  assert.equal(typeof args.p_trace_id, "string");
  assert.equal(typeof args.p_trace_created_at, "string");
  assert.ok(Array.isArray(args.p_events));

  const logical = (event) => ({
    schema_version: event.schema_version,
    event_id: event.event_id,
    event_type: event.event_type,
    session_id: event.session_id,
    question_id: event.question_id,
    question_version: event.question_version,
    payload: event.payload,
  });
  const jsonEqual = (left, right) => {
    if (typeof left === "number" || typeof right === "number") {
      return typeof left === "number" && typeof right === "number" &&
        left === right;
    }
    if (left === null || right === null) return left === right;
    if (typeof left !== typeof right) return false;
    if (Array.isArray(left) || Array.isArray(right)) {
      return Array.isArray(left) && Array.isArray(right) &&
        left.length === right.length &&
        left.every((item, index) => jsonEqual(item, right[index]));
    }
    if (typeof left === "object") {
      const leftKeys = Object.keys(left).sort();
      const rightKeys = Object.keys(right).sort();
      return jsonEqual(leftKeys, rightKeys) &&
        leftKeys.every((key) => jsonEqual(left[key], right[key]));
    }
    return left === right;
  };

  const newEvents = [];
  let duplicate = 0;
  let conflicting = 0;
  for (const event of args.p_events) {
    assert.equal(typeof event.event_id, "string");
    const existing = stored.get(event.event_id);
    if (existing === undefined) {
      newEvents.push(event);
    } else if (jsonEqual(logical(existing), logical(event))) {
      duplicate += 1;
    } else {
      conflicting += 1;
    }
  }

  const committed = conflicting === 0;
  if (committed) {
    for (const event of newEvents) {
      stored.set(event.event_id, {
        ...structuredClone(event),
        trace_id: args.p_trace_id,
        trace_created_at: args.p_trace_created_at,
        ingest_request_id: args.p_request_id,
      });
    }
  }
  const result = {
    requested_event_count: args.p_events.length,
    new_event_count: newEvents.length,
    accepted_event_count: committed ? newEvents.length : 0,
    duplicate_event_count: duplicate,
    conflicting_event_count: conflicting,
    rejected_event_count: committed ? 0 : newEvents.length + conflicting,
    committed,
  };
  rpcCalls.push({ ...structuredClone(result), mode });

  if (mode === "commit-then-fail") {
    assert.equal(committed, true);
    throw new TypeError("simulated response loss after commit");
  }
  return new Response(JSON.stringify([result]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

await import(__EDGE_FUNCTION_URI__);
assert.equal(typeof handler, "function");

const event = (eventId, eventType, sequence, payload, questionId = "q_test") => ({
  schema_version: "1.0",
  event_id: eventId,
  event_type: eventType,
  occurred_at: `2026-07-12T00:00:0${sequence}.000Z`,
  session_id: "anon_ingest_test",
  question_id: questionId,
  question_version: `qv1_${questionId}`,
  payload,
  sequence,
});

const envelope = (traceId, events) => ({
  schema_version: "1.0",
  envelope_type: "session_trace",
  trace_id: traceId,
  session_id: "anon_ingest_test",
  created_at: "2026-07-12T00:00:00.000Z",
  event_count: events.length,
  events,
});

const invoke = async (trace) => {
  const handlerStarted = performance.now();
  const response = await handler(new Request(
    "https://edge.example/functions/v1/feedback-ingest",
    {
      method: "POST",
      headers: {
        Authorization: "Bearer ingest-secret",
        "Content-Type": "application/json; charset=utf-8",
        "Idempotency-Key": trace.trace_id,
      },
      body: JSON.stringify(trace),
    },
  ));
  const handlerElapsedMs = performance.now() - handlerStarted;
  const result = {
    status: response.status,
    body: await response.json(),
    handlerElapsedMs,
  };
  await flushBackground();
  return result;
};

const assertReceipt = (result, status, counters) => {
  assert.equal(result.status, status);
  assert.equal(result.body.accepted, counters.accepted);
  assert.equal(result.body.duplicate, counters.duplicate);
  assert.equal(result.body.conflict, counters.conflict);
  assert.equal(result.body.rejected, counters.rejected);
  assert.equal(typeof result.body.request_id, "string");
};

// 1. A complete session trace is durably inserted on its first upload.
const firstTrace = envelope("trace_first", [
  event("evt_first_answer", "answer_submitted", 1, {
    attempt_id: "attempt_first",
    selected_letter: "A",
    selected_candidate_id: "c_a",
    is_correct: true,
  }),
  event("evt_first_proposal", "custom_setting_proposed", 2, {
    attempt_id: "attempt_first",
    setting: { model: { type: "mlp", width: 64 } },
    label: "Wide MLP",
    n_seeds: 2,
    base_seed: 0,
    inherited_from: null,
  }),
]);
let result = await invoke(firstTrace);
assertReceipt(result, 200, {
  accepted: 2, duplicate: 0, conflict: 0, rejected: 0,
});
assert.equal(stored.size, 2);

// 2. A byte-equivalent retry is acknowledged entirely as duplicate.
result = await invoke(firstTrace);
assertReceipt(result, 200, {
  accepted: 0, duplicate: 2, conflict: 0, rejected: 0,
});
assert.equal(stored.size, 2);

// 3. The RPC distinguishes an exact replay from event-ID reuse with different
// logical content. The original row remains first-write-wins.
const conflictingTrace = envelope("trace_conflicting_payload", [
  event("evt_first_answer", "answer_submitted", 1, {
    attempt_id: "attempt_first",
    selected_letter: "Z",
    selected_candidate_id: "c_different",
    is_correct: false,
  }),
]);
result = await invoke(conflictingTrace);
assertReceipt(result, 409, {
  accepted: 0, duplicate: 0, conflict: 1, rejected: 1,
});
assert.equal(result.body.error.code, "EVENT_ID_CONFLICT");
assert.equal(stored.get("evt_first_answer").payload.selected_letter, "A");
assert.equal(stored.size, 2);

// 4. A batch containing one new event, one exact duplicate, and one conflict
// is rejected atomically. In particular, the otherwise valid new event is not
// inserted and the existing conflicting row is not changed.
const mixedTrace = envelope("trace_mixed_conflict", [
  firstTrace.events[0],
  event(
    "evt_mixed_new",
    "comment_submitted",
    2,
    {
      attempt_id: "attempt_mixed",
      category: "suggestion",
      text: "This row must not be partially inserted.",
    },
    "q_mixed",
  ),
  event("evt_first_proposal", "custom_setting_proposed", 3, {
    attempt_id: "attempt_first",
    setting: { model: { type: "mlp", width: 999 } },
    label: "Conflicting MLP",
    n_seeds: 2,
    base_seed: 0,
    inherited_from: null,
  }),
]);
result = await invoke(mixedTrace);
assertReceipt(result, 409, {
  accepted: 0, duplicate: 1, conflict: 1, rejected: 2,
});
assert.equal(result.body.error.code, "EVENT_ID_CONFLICT");
assert.equal(stored.has("evt_mixed_new"), false);
assert.equal(stored.get("evt_first_proposal").payload.label, "Wide MLP");
assert.equal(stored.size, 2);

// 5. Model the ambiguous failure that matters for retry safety: the atomic RPC
// commits both rows and its response is then lost. Replay classifies both rows
// as exact duplicates rather than conflicts.
const ambiguousTrace = envelope("trace_ambiguous", [
  event(
    "evt_ambiguous_answer",
    "answer_submitted",
    1,
    {
      attempt_id: "attempt_ambiguous",
      selected_letter: "A",
      is_correct: true,
    },
    "q_ambiguous",
  ),
  event(
    "evt_ambiguous_comment",
    "comment_submitted",
    2,
    {
      attempt_id: "attempt_ambiguous",
      category: "bug",
      text: "The curve failed to render.",
    },
    "q_ambiguous",
  ),
]);
mode = "commit-then-fail";
result = await invoke(ambiguousTrace);
assertReceipt(result, 502, {
  accepted: 0, duplicate: 0, conflict: 0, rejected: 0,
});
assert.equal(result.body.error.code, "STORAGE_UNAVAILABLE");
assert.match(result.body.error.message, /retry is safe/);
assert.equal(stored.size, 4);
mode = "normal";

// 6. Outcome persistence remains observability-only. A slow failed outcome
// write cannot delay or turn a durably committed event into an ingest failure.
const telemetryTrace = envelope("trace_telemetry_failure", [
  event(
    "evt_telemetry_answer",
    "answer_submitted",
    1,
    { attempt_id: "attempt_telemetry", selected_letter: "C" },
    "q_telemetry",
  ),
]);
outcomeMode = "slow-fail";
result = await invoke(telemetryTrace);
assertReceipt(result, 200, {
  accepted: 1, duplicate: 0, conflict: 0, rejected: 0,
});
assert.ok(result.handlerElapsedMs < 225);
outcomeMode = "normal";
assert.equal(stored.size, 5);

result = await invoke(ambiguousTrace);
assertReceipt(result, 200, {
  accepted: 0, duplicate: 2, conflict: 0, rejected: 0,
});
assert.equal(stored.size, 5);

assert.deepEqual(
  [...stored.keys()].sort(),
  [
    "evt_ambiguous_answer",
    "evt_ambiguous_comment",
    "evt_first_answer",
    "evt_first_proposal",
    "evt_telemetry_answer",
  ],
);
assert.deepEqual(
  rpcCalls.map((call) => ({
    requested: call.requested_event_count,
    newEvents: call.new_event_count,
    accepted: call.accepted_event_count,
    duplicate: call.duplicate_event_count,
    conflicting: call.conflicting_event_count,
    rejected: call.rejected_event_count,
    committed: call.committed,
    mode: call.mode,
  })),
  [
    {
      requested: 2, newEvents: 2, accepted: 2, duplicate: 0,
      conflicting: 0, rejected: 0, committed: true, mode: "normal",
    },
    {
      requested: 2, newEvents: 0, accepted: 0, duplicate: 2,
      conflicting: 0, rejected: 0, committed: true, mode: "normal",
    },
    {
      requested: 1, newEvents: 0, accepted: 0, duplicate: 0,
      conflicting: 1, rejected: 1, committed: false, mode: "normal",
    },
    {
      requested: 3, newEvents: 1, accepted: 0, duplicate: 1,
      conflicting: 1, rejected: 2, committed: false, mode: "normal",
    },
    {
      requested: 2, newEvents: 2, accepted: 2, duplicate: 0,
      conflicting: 0, rejected: 0, committed: true,
      mode: "commit-then-fail",
    },
    {
      requested: 1, newEvents: 1, accepted: 1, duplicate: 0,
      conflicting: 0, rejected: 0, committed: true, mode: "normal",
    },
    {
      requested: 2, newEvents: 0, accepted: 0, duplicate: 2,
      conflicting: 0, rejected: 0, committed: true, mode: "normal",
    },
  ],
);

const recordedOutcomes = [...outcomes.values()];
assert.equal(recordedOutcomes.length, 6);
assert.deepEqual(
  recordedOutcomes.map((row) => row.outcome_code),
  [
    "accepted_only",
    "duplicate_only",
    "event_id_conflict",
    "event_id_conflict",
    "storage_unavailable",
    "duplicate_only",
  ],
);
assert.deepEqual(
  recordedOutcomes.map((row) => row.storage_state),
  [
    "confirmed",
    "confirmed",
    "confirmed",
    "confirmed",
    "unknown",
    "confirmed",
  ],
);
assert.deepEqual(
  recordedOutcomes.map((row) => row.conflicting_event_count),
  [0, 0, 1, 1, null, 0],
);
assert.deepEqual(
  recordedOutcomes.map((row) => row.rejected_event_count),
  [0, 0, 1, 2, null, 0],
);
assert.deepEqual(
  recordedOutcomes.map((row) => row.outcome_class),
  [
    "success",
    "success",
    "client_rejection",
    "client_rejection",
    "service_failure",
    "success",
  ],
);
const unknown = recordedOutcomes.find(
  (row) => row.outcome_code === "storage_unavailable",
);
assert.equal(unknown.requested_event_count, 2);
assert.equal(unknown.accepted_event_count, null);
assert.equal(unknown.duplicate_event_count, null);
assert.equal(unknown.conflicting_event_count, null);
assert.equal(unknown.rejected_event_count, null);
assert.equal(unknown.retryable, true);
""".replace("__EDGE_FUNCTION_URI__", repr(EDGE_FUNCTION.as_uri()))

    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_feedback_ingest_enforces_cross_runtime_json_and_unicode_contract() -> None:
    node = _node_with_type_stripping()
    script = r"""
import assert from "node:assert/strict";

let handler;
const rpcPayloads = [];

globalThis.EdgeRuntime = { waitUntil() {} };
globalThis.Deno = {
  env: {
    get(name) {
      return {
        FEEDBACK_INGEST_TOKEN: "ingest-secret",
        SUPABASE_URL: "https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY: "service-secret",
      }[name] ?? null;
    },
  },
  serve(callback) { handler = callback; },
};

globalThis.fetch = async (input, init) => {
  const endpoint = new URL(String(input));
  if (endpoint.pathname === "/rest/v1/feedback_ingest_request_outcomes") {
    return new Response("", { status: 201 });
  }
  assert.equal(endpoint.pathname, "/rest/v1/rpc/feedback_ingest_events");
  const args = JSON.parse(init.body);
  rpcPayloads.push(structuredClone(args));
  const count = args.p_events.length;
  return new Response(JSON.stringify([{
    requested_event_count: count,
    new_event_count: count,
    accepted_event_count: count,
    duplicate_event_count: 0,
    conflicting_event_count: 0,
    rejected_event_count: 0,
    committed: true,
  }]), { status: 200, headers: { "Content-Type": "application/json" } });
};

await import(__EDGE_FUNCTION_URI__);

const event = (eventType, payload, sessionId = "anon_interop") => ({
  schema_version: "1.0",
  event_id: "evt_interop",
  event_type: eventType,
  occurred_at: "2026-07-12T00:00:01.000Z",
  session_id: sessionId,
  question_id: "q_interop",
  question_version: "qv1_interop",
  payload,
  sequence: 1,
});
const envelope = (item, sessionId = item.session_id) => ({
  schema_version: "1.0",
  envelope_type: "session_trace",
  trace_id: "trace_interop",
  session_id: sessionId,
  created_at: "2026-07-12T00:00:00.000Z",
  event_count: 1,
  events: [item],
});
const invoke = async (body) => {
  const response = await handler(new Request(
    "https://edge.example/functions/v1/feedback-ingest",
    {
      method: "POST",
      headers: {
        Authorization: "Bearer ingest-secret",
        "Content-Type": "application/json",
      },
      body,
    },
  ));
  return { status: response.status, body: await response.json() };
};

const safe = envelope(event("custom_setting_proposed", {
  setting: {
    limits: [Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER],
  },
}));
let result = await invoke(JSON.stringify(safe));
assert.equal(result.status, 200);
assert.deepEqual(
  rpcPayloads[0].p_events[0].payload.setting.limits,
  [Number.MAX_SAFE_INTEGER, -Number.MAX_SAFE_INTEGER],
);

const positiveUnsafe = JSON.stringify(safe).replace(
  String(Number.MAX_SAFE_INTEGER),
  "9007199254740992",
);
let rpcCount = rpcPayloads.length;
result = await invoke(positiveUnsafe);
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_REQUEST");
assert.equal(rpcPayloads.length, rpcCount);

const negativeUnsafe = JSON.stringify(safe).replace(
  String(-Number.MAX_SAFE_INTEGER),
  "-9007199254740992",
);
result = await invoke(negativeUnsafe);
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_REQUEST");
assert.equal(rpcPayloads.length, rpcCount);

const exponentUnsafe = envelope(event("custom_setting_proposed", {
  setting: { value: 1e20 },
}));
result = await invoke(JSON.stringify(exponentUnsafe));
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_REQUEST");
assert.equal(rpcPayloads.length, rpcCount);

const emoji = "😀";
const maxEmojiSession = emoji.repeat(200);
const maxEmojiComment = envelope(event(
  "comment_submitted",
  { category: "other", text: emoji.repeat(2_000) },
  maxEmojiSession,
));
result = await invoke(JSON.stringify(maxEmojiComment));
assert.equal(result.status, 200);
assert.equal(rpcPayloads.at(-1).p_events[0].session_id, maxEmojiSession);
assert.equal(rpcPayloads.at(-1).p_events[0].payload.text, emoji.repeat(2_000));

rpcCount = rpcPayloads.length;
const oversizedEmojiIdentifier = structuredClone(maxEmojiComment);
oversizedEmojiIdentifier.session_id = emoji.repeat(201);
oversizedEmojiIdentifier.events[0].session_id = emoji.repeat(201);
result = await invoke(JSON.stringify(oversizedEmojiIdentifier));
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_ENVELOPE");
assert.equal(rpcPayloads.length, rpcCount);

const oversizedEmojiComment = structuredClone(maxEmojiComment);
oversizedEmojiComment.events[0].payload.text = emoji.repeat(2_001);
result = await invoke(JSON.stringify(oversizedEmojiComment));
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_ENVELOPE");
assert.equal(rpcPayloads.length, rpcCount);

for (const surrogate of ["\ud800", "\udc00"]) {
  const loneSurrogate = structuredClone(safe);
  loneSurrogate.events[0].payload.setting.note = surrogate;
  result = await invoke(JSON.stringify(loneSurrogate));
  assert.equal(result.status, 400);
  assert.equal(result.body.error.code, "INVALID_REQUEST");
  assert.equal(rpcPayloads.length, rpcCount);
}
""".replace("__EDGE_FUNCTION_URI__", repr(EDGE_FUNCTION.as_uri()))

    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_feedback_ingest_records_eligible_terminal_branches_without_secrets() -> None:
    node = _node_with_type_stripping()
    script = r"""
import assert from "node:assert/strict";

let handler;
let configured = true;
let storageMode = "normal";
const outcomes = [];
const logs = [];
const backgroundTasks = [];
const originalConsoleError = console.error;
console.error = (...values) => logs.push(values.map(String).join(" "));

globalThis.EdgeRuntime = {
  waitUntil(promise) { backgroundTasks.push(promise); },
};

const flushBackground = async () => {
  const pending = backgroundTasks.splice(0, backgroundTasks.length);
  await Promise.all(pending);
};

globalThis.Deno = {
  env: {
    get(name) {
      if (!configured) return null;
      return {
        FEEDBACK_INGEST_TOKEN: "ingest-secret",
        SUPABASE_URL: "https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY: "service-secret",
      }[name] ?? null;
    },
  },
  serve(callback) { handler = callback; },
};

globalThis.fetch = async (input, init) => {
  const endpoint = new URL(String(input));
  if (endpoint.pathname === "/rest/v1/feedback_ingest_request_outcomes") {
    const row = JSON.parse(init.body);
    assert.equal(row.schema_version, "1.1");
    assert.equal(row.observer_revision, "obs2");
    assert.ok(Object.hasOwn(row, "conflicting_event_count"));
    outcomes.push(row);
    return new Response("", { status: 201 });
  }
  assert.equal(endpoint.pathname, "/rest/v1/rpc/feedback_ingest_events");
  const args = JSON.parse(init.body);
  assert.deepEqual(Object.keys(args).sort(), [
    "p_events",
    "p_request_id",
    "p_trace_created_at",
    "p_trace_id",
  ]);
  assert.ok(Array.isArray(args.p_events));
  if (storageMode === "reject") {
    return new Response('{"private":"database diagnostic"}', { status: 503 });
  }
  if (storageMode === "client-reject") {
    return new Response('{"code":"constraint"}', { status: 409 });
  }
  const conflicting = storageMode === "conflict" ? 1 : 0;
  const committed = conflicting === 0;
  const result = {
    requested_event_count: args.p_events.length,
    new_event_count: committed ? args.p_events.length : 0,
    accepted_event_count: committed ? args.p_events.length : 0,
    duplicate_event_count: 0,
    conflicting_event_count: conflicting,
    rejected_event_count: conflicting,
    committed,
  };
  return new Response(JSON.stringify([result]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

await import(__EDGE_FUNCTION_URI__);

const invoke = async ({
  method = "POST",
  token = "ingest-secret",
  body = null,
  contentType = "application/json",
} = {}) => {
  const headers = {};
  if (token !== null) headers.Authorization = `Bearer ${token}`;
  if (contentType !== null) headers["Content-Type"] = contentType;
  const response = await handler(new Request(
    "https://edge.example/functions/v1/feedback-ingest",
    { method, headers, body: method === "POST" ? body : null },
  ));
  const result = { status: response.status, body: await response.json() };
  await flushBackground();
  return result;
};

// Excluded requests never amplify untrusted traffic into database writes.
let result = await invoke({ method: "GET", token: null });
assert.equal(result.status, 405);
configured = false;
result = await invoke({ body: "{}" });
assert.equal(result.status, 503);
configured = true;
result = await invoke({ token: "wrong-token", body: "{}" });
assert.equal(result.status, 401);
assert.equal(outcomes.length, 0);

// Authenticated parse, size, and envelope rejections enter the denominator.
result = await invoke({ body: '{"text":"TOP_SECRET_COMMENT"' });
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_REQUEST");

result = await invoke({
  body: JSON.stringify({ events: [{}, {}] }),
});
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_ENVELOPE");
assert.equal(result.body.rejected, 2);

result = await invoke({ body: "x".repeat(1_048_577) });
assert.equal(result.status, 413);
assert.equal(result.body.error.code, "REQUEST_TOO_LARGE");

const event = {
  schema_version: "1.0",
  event_id: "evt_storage",
  event_type: "answer_submitted",
  occurred_at: "2026-07-12T00:00:01.000Z",
  session_id: "anon_storage",
  question_id: "q_storage",
  question_version: "qv1_storage",
  payload: { selected_letter: "A" },
  sequence: 1,
};
const envelope = {
  schema_version: "1.0",
  envelope_type: "session_trace",
  trace_id: "trace_storage",
  session_id: "anon_storage",
  created_at: "2026-07-12T00:00:00.000Z",
  event_count: 1,
  events: [event],
};
const oversizedSequenceEnvelope = structuredClone(envelope);
oversizedSequenceEnvelope.trace_id = "trace_oversized_sequence";
oversizedSequenceEnvelope.events[0].sequence = 2_147_483_648;
result = await invoke({ body: JSON.stringify(oversizedSequenceEnvelope) });
assert.equal(result.status, 400);
assert.equal(result.body.error.code, "INVALID_ENVELOPE");
assert.deepEqual(
  [result.body.accepted, result.body.duplicate, result.body.conflict, result.body.rejected],
  [0, 0, 0, 1],
);

storageMode = "conflict";
result = await invoke({ body: JSON.stringify(envelope) });
assert.equal(result.status, 409);
assert.equal(result.body.error.code, "EVENT_ID_CONFLICT");
assert.deepEqual(
  [
    result.body.accepted,
    result.body.duplicate,
    result.body.conflict,
    result.body.rejected,
  ],
  [0, 0, 1, 1],
);

storageMode = "reject";
result = await invoke({ body: JSON.stringify(envelope) });
assert.equal(result.status, 502);
assert.deepEqual(
  [
    result.body.accepted,
    result.body.duplicate,
    result.body.conflict,
    result.body.rejected,
  ],
  [0, 0, 0, 0],
);

assert.deepEqual(
  outcomes.map((row) => row.outcome_code),
  [
    "invalid_request",
    "invalid_envelope",
    "request_too_large",
    "invalid_envelope",
    "event_id_conflict",
    "storage_unavailable",
  ],
);
assert.deepEqual(
  outcomes.map((row) => row.outcome_class),
  [
    "client_rejection",
    "client_rejection",
    "client_rejection",
    "client_rejection",
    "client_rejection",
    "service_failure",
  ],
);
assert.equal(outcomes[0].requested_event_count, null);
assert.equal(outcomes[1].requested_event_count, 2);
assert.equal(outcomes[1].rejected_event_count, 2);
assert.equal(outcomes[2].requested_event_count, null);
assert.equal(outcomes[3].requested_event_count, 1);
assert.equal(outcomes[3].accepted_event_count, 0);
assert.equal(outcomes[3].conflicting_event_count, null);
assert.equal(outcomes[3].rejected_event_count, 1);
assert.equal(outcomes[3].storage_state, "not_attempted");
assert.equal(outcomes[3].retryable, false);
assert.equal(outcomes[4].requested_event_count, 1);
assert.equal(outcomes[4].accepted_event_count, 0);
assert.equal(outcomes[4].conflicting_event_count, 1);
assert.equal(outcomes[4].rejected_event_count, 1);
assert.equal(outcomes[4].storage_state, "confirmed");
assert.equal(outcomes[4].retryable, false);
assert.equal(outcomes[5].requested_event_count, 1);
assert.equal(outcomes[5].accepted_event_count, null);
assert.equal(outcomes[5].conflicting_event_count, null);
assert.equal(outcomes[5].storage_state, "unknown");
assert.equal(outcomes[5].retryable, true);

storageMode = "client-reject";
result = await invoke({ body: JSON.stringify(envelope) });
assert.equal(result.status, 502);
assert.equal(result.body.conflict, 0);
assert.equal(outcomes[6].outcome_code, "storage_unavailable");
assert.equal(outcomes[6].conflicting_event_count, null);
assert.equal(outcomes[6].storage_state, "not_committed");

const serializedOutcomes = JSON.stringify(outcomes);
const serializedLogs = logs.join("\n");
for (const forbidden of [
  "TOP_SECRET_COMMENT",
  "ingest-secret",
  "service-secret",
  "wrong-token",
  "database diagnostic",
]) {
  assert.ok(!serializedOutcomes.includes(forbidden));
  assert.ok(!serializedLogs.includes(forbidden));
}
assert.ok(logs.some((line) => line.includes('"outcome_code":"method_not_allowed"')));
assert.ok(logs.some((line) => line.includes('"outcome_code":"unauthorized"')));
assert.ok(logs.some((line) => line.includes('"outcome_code":"service_unavailable"')));
console.error = originalConsoleError;
""".replace("__EDGE_FUNCTION_URI__", repr(EDGE_FUNCTION.as_uri()))

    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
