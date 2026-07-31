"""Backend contracts for auditable question presentation events."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "supabase/migrations/20260712019000_question_reactions.sql"
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


def test_presentation_store_constraint_and_rpc_enum_are_strict() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    constraint = sql.split(
        "add constraint feedback_events_question_presented_payload_check",
        maxsplit=1,
    )[1].split("-- STATS-002B", maxsplit=1)[0]

    assert sql.count("'question_presented'") >= 2
    assert "event_type <> 'question_presented'" in constraint
    for field in (
        "attempt_id",
        "release_id",
        "decision_id",
        "policy_version",
        "mode",
        "propensity",
        "source",
        "position",
    ):
        assert f"'{field}'" in constraint
    assert "'exploit', 'explore', 'fallback', 'manual'" in constraint
    assert "'initial', 'next', 'random', 'picker'" in constraint
    assert "jsonb_typeof(payload -> 'propensity') = 'number'" in constraint
    assert "(payload ->> 'propensity')::numeric > 0" in constraint
    assert "(payload ->> 'propensity')::numeric <= 1" in constraint
    assert "jsonb_typeof(payload -> 'position') = 'number'" in constraint
    assert "between 1 and 9007199254740991" in constraint


def test_feedback_ingest_validates_question_presentations() -> None:
    node = _node_with_type_stripping()
    script = r"""
import assert from "node:assert/strict";

let handler;
const rpcPayloads = [];
globalThis.Deno = {
  env: { get(name) { return {
    FEEDBACK_INGEST_TOKEN: "ingest-secret",
    SUPABASE_URL: "https://project.supabase.co",
    SUPABASE_SERVICE_ROLE_KEY: "service-secret",
  }[name] ?? null; } },
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
    requested_event_count: 1,
    new_event_count: 1,
    accepted_event_count: 1,
    duplicate_event_count: 0,
    conflicting_event_count: 0,
    rejected_event_count: 0,
    committed: true,
  }]), { status: 200, headers: { "Content-Type": "application/json" } });
};

await import(__EDGE_FUNCTION_URI__);
let number = 0;
const invoke = async (payload) => {
  number += 1;
  const event = {
    schema_version: "1.0",
    event_id: `evt_presented_${number}`,
    event_type: "question_presented",
    occurred_at: "2026-07-12T08:00:01.000Z",
    session_id: "anon_presented",
    question_id: "q_presented",
    question_version: `qv1_${"a".repeat(64)}`,
    payload,
    sequence: number,
  };
  const response = await handler(new Request("https://edge.test/feedback-ingest", {
    method: "POST",
    headers: {
      Authorization: "Bearer ingest-secret",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      schema_version: "1.0",
      envelope_type: "session_trace",
      trace_id: "trace_presented",
      session_id: "anon_presented",
      created_at: "2026-07-12T08:00:00.000Z",
      event_count: 1,
      events: [event],
    }),
  }));
  return { status: response.status, body: await response.json() };
};

const valid = {
  attempt_id: "attempt_1",
  release_id: `release_${"b".repeat(64)}`,
  decision_id: "decision_1",
  policy_version: "surprise_policy_v1",
  mode: "exploit",
  propensity: 0.85,
  source: "next",
  position: 2,
};
let result = await invoke(valid);
assert.equal(result.status, 200);
assert.deepEqual(rpcPayloads[0].p_events[0].payload, valid);

const invalid = [
  { ...valid, attempt_id: "" },
  { ...valid, release_id: null },
  { ...valid, decision_id: " decision_1" },
  { ...valid, policy_version: "" },
  { ...valid, mode: "unknown" },
  { ...valid, propensity: 0 },
  { ...valid, propensity: 1.1 },
  { ...valid, propensity: "0.5" },
  { ...valid, source: "unknown" },
  { ...valid, position: 0 },
  { ...valid, position: 1.5 },
  { ...valid, position: "2" },
];
for (const payload of invalid) {
  result = await invoke(payload);
  assert.equal(result.status, 400);
  assert.equal(result.body.error.code, "INVALID_ENVELOPE");
}
assert.equal(rpcPayloads.length, 1);
""".replace("__EDGE_FUNCTION_URI__", repr(EDGE_FUNCTION.as_uri()))

    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO,
    )
