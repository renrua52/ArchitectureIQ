"""Contracts for structured post-reveal question reactions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "supabase/migrations/20260712019000_question_reactions.sql"
CONFLICT_MIGRATION = (
    REPO / "supabase/migrations/20260712012000_feedback_event_conflicts.sql"
)
EDGE_FUNCTION = REPO / "supabase/functions/feedback-ingest/index.ts"

ORIGINAL_EVENT_TYPES = (
    "answer_submitted",
    "custom_setting_proposed",
    "custom_setting_rejected",
    "custom_run_completed",
    "custom_run_failed",
    "comment_submitted",
)
REACTION_EVENT_TYPE = "question_reaction_submitted"


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


def test_reaction_migration_is_forward_only_and_preserves_store_guards() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert MIGRATION.name > "20260712018000_feedback_session_attempt_filters.sql"
    assert sql.startswith("begin;\n")
    assert sql.rstrip().endswith("commit;")
    assert "drop constraint feedback_events_event_type_check" in sql
    assert "add constraint feedback_events_event_type_check" in sql
    for event_type in (*ORIGINAL_EVENT_TYPES, REACTION_EVENT_TYPE):
        assert sql.count(f"'{event_type}'") >= 2

    assert "create or replace function public.feedback_ingest_events(" in sql
    assert "input.event_type not in (" in sql
    assert "pg_catalog.pg_advisory_xact_lock" in sql
    assert "public.feedback_logical_event_v1(" in sql
    assert "insert into public.feedback_event_conflicts" in sql
    assert "feedback event writer race detected; retry the request" in sql
    assert (
        "revoke all on function public.feedback_ingest_events(\n"
        "    uuid,\n"
        "    text,\n"
        "    timestamptz,\n"
        "    jsonb\n"
        ")\n"
        "from public, anon, authenticated, service_role;"
    ) in sql
    assert (
        "grant execute on function public.feedback_ingest_events(\n"
        "    uuid,\n"
        "    text,\n"
        "    timestamptz,\n"
        "    jsonb\n"
        ")\n"
        "to service_role;"
    ) in sql

    lowered = sql.lower()
    assert "disable row level security" not in lowered
    assert "drop trigger" not in lowered
    assert "grant insert on table public.feedback_events" not in lowered
    assert "update public.feedback_events" not in lowered
    assert "delete from public.feedback_events" not in lowered


def test_reaction_payload_constraint_is_strict_and_release_is_optional() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    constraint = sql.split(
        "add constraint feedback_events_question_reaction_payload_check",
        maxsplit=1,
    )[1].split(
        "add constraint feedback_events_question_presented_payload_check",
        maxsplit=1,
    )[0]

    assert "event_type <> 'question_reaction_submitted'" in constraint
    assert "payload ?& array[" in constraint
    for key in ("reaction", "value", "timing", "attempt_id"):
        assert f"'{key}'" in constraint
    assert "payload ->> 'reaction' = 'surprise'" in constraint
    assert "jsonb_typeof(payload -> 'value') = 'boolean'" in constraint
    assert "payload ->> 'timing' = 'after_reveal'" in constraint
    assert "jsonb_typeof(payload -> 'attempt_id') = 'string'" in constraint
    assert "length(payload ->> 'attempt_id')" in constraint
    assert "between 1 and 200" in constraint
    assert "not (payload ? 'release_id')" in constraint
    assert (
        "jsonb_typeof(\n                                    payload -> 'release_id'"
        in constraint
    )
    # COALESCE(..., false) prevents missing/null required values from passing a
    # PostgreSQL CHECK through SQL's unknown truth value.
    assert constraint.count("coalesce(") == 2
    assert "pg_catalog.coalesce" not in constraint
    assert constraint.count("false") == 2


def test_reaction_rpc_changes_only_the_supported_event_enum() -> None:
    original = CONFLICT_MIGRATION.read_text(encoding="utf-8")
    upgraded = MIGRATION.read_text(encoding="utf-8")
    original_rpc = original[
        original.index(
            "create function public.feedback_ingest_events("
        ) : original.index(
            "\n$function$;",
            original.index("create function public.feedback_ingest_events("),
        )
        + len("\n$function$;")
    ]
    upgraded_rpc = upgraded[
        upgraded.index(
            "create or replace function public.feedback_ingest_events("
        ) : upgraded.index(
            "\n$function$;",
            upgraded.index("create or replace function public.feedback_ingest_events("),
        )
        + len("\n$function$;")
    ]
    upgraded_without_enum = upgraded_rpc.replace(
        "create or replace function",
        "create function",
        1,
    ).replace(
        "                'comment_submitted',\n"
        "                'question_presented',\n"
        "                'question_reaction_submitted'",
        "                'comment_submitted'",
        1,
    )

    assert upgraded_without_enum == original_rpc


def test_feedback_ingest_strictly_validates_question_reactions() -> None:
    node = _node_with_type_stripping()
    script = r"""
import assert from "node:assert/strict";

let handler;
const rpcPayloads = [];

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
  return new Response(JSON.stringify([{
    requested_event_count: args.p_events.length,
    new_event_count: args.p_events.length,
    accepted_event_count: args.p_events.length,
    duplicate_event_count: 0,
    conflicting_event_count: 0,
    rejected_event_count: 0,
    committed: true,
  }]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

await import(__EDGE_FUNCTION_URI__);
assert.equal(typeof handler, "function");

let eventNumber = 0;
const envelope = (payload) => {
  eventNumber += 1;
  const event = {
    schema_version: "1.0",
    event_id: `evt_reaction_${eventNumber}`,
    event_type: "question_reaction_submitted",
    occurred_at: "2026-07-12T08:00:01.000Z",
    session_id: "anon_reaction_test",
    question_id: "q_reaction",
    question_version: `qv1_${"a".repeat(64)}`,
    payload,
    sequence: eventNumber,
  };
  return {
    schema_version: "1.0",
    envelope_type: "session_trace",
    trace_id: "trace_reaction_test",
    session_id: "anon_reaction_test",
    created_at: "2026-07-12T08:00:00.000Z",
    event_count: 1,
    events: [event],
  };
};

const invoke = async (payload) => {
  const response = await handler(new Request("https://edge.test/feedback-ingest", {
    method: "POST",
    headers: {
      Authorization: "Bearer ingest-secret",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(envelope(payload)),
  }));
  return { status: response.status, body: await response.json() };
};

const released = {
  reaction: "surprise",
  value: true,
  timing: "after_reveal",
  attempt_id: "attempt_1",
  release_id: `release_${"b".repeat(64)}`,
  family: "multivariate_regression",
};
let result = await invoke(released);
assert.equal(result.status, 200);
assert.deepEqual(
  [result.body.accepted, result.body.duplicate, result.body.rejected],
  [1, 0, 0],
);
assert.deepEqual(rpcPayloads[0].p_events[0].payload, released);

const unreleased = {
  reaction: "surprise",
  value: false,
  timing: "after_reveal",
  attempt_id: "attempt_2",
};
result = await invoke(unreleased);
assert.equal(result.status, 200);
assert.deepEqual(rpcPayloads[1].p_events[0].payload, unreleased);

const invalidPayloads = [
  { value: true, timing: "after_reveal", attempt_id: "attempt_3" },
  { reaction: "like", value: true, timing: "after_reveal", attempt_id: "attempt_3" },
  { reaction: "surprise", timing: "after_reveal", attempt_id: "attempt_3" },
  { reaction: "surprise", value: 1, timing: "after_reveal", attempt_id: "attempt_3" },
  { reaction: "surprise", value: "true", timing: "after_reveal", attempt_id: "attempt_3" },
  { reaction: "surprise", value: true, attempt_id: "attempt_3" },
  { reaction: "surprise", value: true, timing: "before_reveal", attempt_id: "attempt_3" },
  { reaction: "surprise", value: true, timing: "after_reveal" },
  { reaction: "surprise", value: true, timing: "after_reveal", attempt_id: null },
  { reaction: "surprise", value: true, timing: "after_reveal", attempt_id: "" },
  { reaction: "surprise", value: true, timing: "after_reveal", attempt_id: " attempt_3" },
  {
    reaction: "surprise",
    value: true,
    timing: "after_reveal",
    attempt_id: "attempt_3",
    release_id: null,
  },
  {
    reaction: "surprise",
    value: true,
    timing: "after_reveal",
    attempt_id: "attempt_3",
    release_id: "",
  },
];

for (const payload of invalidPayloads) {
  result = await invoke(payload);
  assert.equal(result.status, 400);
  assert.equal(result.body.error.code, "INVALID_ENVELOPE");
}
assert.equal(rpcPayloads.length, 2);
""".replace("__EDGE_FUNCTION_URI__", repr(EDGE_FUNCTION.as_uri()))

    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO,
    )
