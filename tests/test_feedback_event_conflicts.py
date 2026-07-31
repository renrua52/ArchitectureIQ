"""Static contracts for atomic event-id conflict detection and writer lockdown."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFLICT_MIGRATION = (
    REPO / "supabase/migrations/20260712012000_feedback_event_conflicts.sql"
)
LOCKDOWN_MIGRATION = (
    REPO / "supabase/migrations/20260712012500_feedback_event_writer_lockdown.sql"
)
EDGE_FUNCTION = REPO / "supabase/functions/feedback-ingest/index.ts"


def _sql() -> str:
    return CONFLICT_MIGRATION.read_text(encoding="utf-8")


def _function(sql: str, name: str) -> str:
    start = sql.index(f"create function public.{name}(")
    return sql[start : sql.index("\n$function$;", start) + len("\n$function$;")]


def _table_definition(sql: str, name: str) -> str:
    marker = f"create table public.{name} ("
    return sql.split(marker, maxsplit=1)[1].split("\n);", maxsplit=1)[0]


def _normalized(value: str) -> str:
    return " ".join(value.split())


def test_conflict_migrations_are_forward_transactional_steps() -> None:
    conflict_sql = _sql()
    lockdown_sql = LOCKDOWN_MIGRATION.read_text(encoding="utf-8")

    for sql in (conflict_sql, lockdown_sql):
        assert sql.startswith("begin;\n")
        assert sql.rstrip().endswith("commit;")
    assert CONFLICT_MIGRATION.name < LOCKDOWN_MIGRATION.name


def test_conflict_sidecar_has_only_sanitized_correlation_columns() -> None:
    definition = _table_definition(_sql(), "feedback_event_conflicts")
    declared = tuple(
        match.group(1)
        for line in definition.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*)\s+", line))
        and match.group(1) != "constraint"
    )

    assert declared == (
        "request_id",
        "event_id",
        "first_ingest_request_id",
        "comparison_revision",
        "detected_at",
    )
    assert "primary key (request_id, event_id)" in definition
    assert "references public.feedback_events (event_id)" in definition
    assert "comparison_revision = 'logical_event_v1'" in definition
    assert "payload" not in definition
    assert "fingerprint" not in definition
    assert "hash" not in definition


def test_conflict_sidecar_is_private_indexed_and_append_only() -> None:
    sql = _sql()

    assert sql.count("create index feedback_event_conflicts_") == 2
    assert (
        "alter table public.feedback_event_conflicts enable row level security;" in sql
    )
    assert (
        "alter table public.feedback_event_conflicts force row level security;" in sql
    )
    assert (
        "revoke all on table public.feedback_event_conflicts\n"
        "    from public, anon, authenticated, service_role;"
    ) in sql
    assert (
        "grant select on table public.feedback_event_conflicts to service_role;" in sql
    )
    assert "grant insert on table public.feedback_event_conflicts" not in sql
    assert "feedback_event_conflicts is append-only" in sql
    assert (
        "before update or delete or truncate on public.feedback_event_conflicts" in sql
    )


def test_logical_event_v1_is_exact_seven_field_jsonb_projection() -> None:
    helper = _function(_sql(), "feedback_logical_event_v1")
    body = helper.split("as $function$\n", maxsplit=1)[1]
    projected = tuple(
        match.group(1) for match in re.finditer(r"    '([a-z_]+)', p_[a-z_]+", body)
    )

    assert projected == (
        "schema_version",
        "event_id",
        "event_type",
        "session_id",
        "question_id",
        "question_version",
        "payload",
    )
    for excluded in (
        "occurred_at",
        "sequence",
        "trace_id",
        "trace_created_at",
        "ingest_request_id",
        "received_at",
        "request_id",
    ):
        assert excluded not in body
    assert "returns jsonb" in helper
    assert "immutable" in helper
    assert "strict" in helper
    assert "pg_catalog.jsonb_build_object" in helper


def test_ingest_rpc_signature_result_order_and_acl_are_fixed() -> None:
    sql = _sql()
    rpc = _function(sql, "feedback_ingest_events")
    header = rpc.split("language plpgsql", maxsplit=1)[0]

    assert (
        "create function public.feedback_ingest_events(\n"
        "    p_request_id uuid,\n"
        "    p_trace_id text,\n"
        "    p_trace_created_at timestamptz,\n"
        "    p_events jsonb\n"
        ")"
    ) in header
    expected_returns = (
        "returns table ( requested_event_count integer, new_event_count integer, "
        "accepted_event_count integer, duplicate_event_count integer, "
        "conflicting_event_count integer, rejected_event_count integer, "
        "committed boolean )"
    )
    assert expected_returns in _normalized(header)
    assert "security definer" in rpc
    assert "set search_path = ''" in rpc
    assert (
        "revoke all on function public.feedback_ingest_events(\n"
        "    uuid,\n"
        "    text,\n"
        "    timestamptz,\n"
        "    jsonb\n"
        ")\nfrom public, anon, authenticated, service_role;"
    ) in sql
    assert (
        "grant execute on function public.feedback_ingest_events(\n"
        "    uuid,\n"
        "    text,\n"
        "    timestamptz,\n"
        "    jsonb\n"
        ")\nto service_role;"
    ) in sql


def test_ingest_rpc_defends_batch_shape_size_identity_and_order() -> None:
    rpc = _function(_sql(), "feedback_ingest_events")

    assert "v_requested not between 1 and 500" in rpc
    assert "jsonb_typeof(p_events) is distinct from 'array'" in rpc
    assert "p_events items must use the exact event field set" in rpc
    for field in (
        "schema_version",
        "event_id",
        "event_type",
        "occurred_at",
        "session_id",
        "question_id",
        "question_version",
        "payload",
        "sequence",
    ):
        assert f"'{field}'" in rpc
    assert "v_distinct_event_ids <> v_requested" in rpc
    assert "v_session_count <> 1" in rpc
    assert "sequence values must be strictly increasing" in rpc
    assert rpc.count("sequence numeric") == 5
    assert "sequence integer" not in rpc
    assert "input.sequence <> pg_catalog.trunc(input.sequence)" in rpc
    assert "input.sequence not between 1 and 2147483647" in rpc
    assert "new_rows.sequence::integer" in rpc


def test_edge_rejects_sequences_that_postgres_integer_cannot_store() -> None:
    source = EDGE_FUNCTION.read_text(encoding="utf-8")

    assert "const MAX_EVENT_SEQUENCE = 2_147_483_647" in source
    assert "sequence > MAX_EVENT_SEQUENCE" in source
    assert "event.sequence must be between 1 and ${MAX_EVENT_SEQUENCE}" in source


def test_ingest_rpc_acquires_transaction_locks_in_lock_key_order() -> None:
    rpc = _function(_sql(), "feedback_ingest_events")
    lock_section = rpc.split("for v_lock_key in", maxsplit=1)[1].split(
        "    end loop;", maxsplit=1
    )[0]

    assert "pg_catalog.hashtextextended" in lock_section
    assert "select distinct" in lock_section
    assert "order by lock_keys.lock_key" in lock_section
    assert "pg_catalog.pg_advisory_xact_lock(v_lock_key)" in lock_section
    assert lock_section.index("order by lock_keys.lock_key") < lock_section.index(
        "pg_catalog.pg_advisory_xact_lock(v_lock_key)"
    )
    assert "pg_advisory_lock(" not in lock_section


def test_ingest_rpc_uses_exact_db_comparison_and_rejects_whole_conflict_batch() -> None:
    rpc = _function(_sql(), "feedback_ingest_events")
    comparison_start = rpc.index("), classified as (")
    conflict_start = rpc.index("if v_conflicting > 0 then")
    conflict_end = rpc.index("    end if;", conflict_start)
    conflict_branch = rpc[conflict_start:conflict_end]
    event_insert = rpc.index("insert into public.feedback_events", conflict_end)

    assert (
        rpc[comparison_start:conflict_start].count("public.feedback_logical_event_v1(")
        == 2
    )
    assert "insert into public.feedback_event_conflicts" in conflict_branch
    assert "insert into public.feedback_events" not in conflict_branch
    assert _normalized(
        "return query select v_requested, v_new, 0, v_duplicate, "
        "v_conflicting, v_requested - v_duplicate, false;"
    ) in _normalized(conflict_branch)
    assert conflict_end < event_insert
    assert "on conflict (request_id, event_id) do nothing" in conflict_branch


def test_ingest_rpc_bulk_insert_fails_closed_on_bypass_writer_race() -> None:
    rpc = _function(_sql(), "feedback_ingest_events")

    assert "insert into public.feedback_events" in rpc
    assert "on conflict (event_id) do nothing" in rpc
    assert "returning event_id" in rpc
    assert "if v_inserted <> v_new then" in rpc
    assert "errcode = '40001'" in rpc
    assert "feedback event writer race detected; retry the request" in rpc
    assert _normalized(
        "return query select v_requested, v_new, v_inserted, v_duplicate, 0, 0, true;"
    ) in _normalized(rpc)
    assert "on conflict do update" not in _normalized(rpc).lower()
    assert "update public.feedback_events" not in rpc.lower()


def test_outcome_forward_revision_preserves_legacy_and_adds_conflict_409() -> None:
    sql = _sql()
    outcome = sql.split(
        "alter table public.feedback_ingest_request_outcomes\n"
        "    add column conflicting_event_count integer;",
        maxsplit=1,
    )[1]

    assert "check (schema_version in ('1.0', '1.1'))" in outcome
    assert "schema_version = '1.0'" in outcome
    assert "conflicting_event_count is null" in outcome
    assert "schema_version = '1.1'" in outcome
    null_comparison_group = outcome.split(
        "outcome_code in (\n                            'request_too_large',",
        maxsplit=1,
    )[1].split("and conflicting_event_count is null", maxsplit=1)[0]
    for code in (
        "invalid_request",
        "invalid_envelope",
        "storage_unavailable",
        "internal_error",
    ):
        assert f"'{code}'" in null_comparison_group
    assert "outcome_code = 'event_id_conflict'" in outcome
    assert "http_status = 409" in outcome
    assert "outcome_class = 'client_rejection'" in outcome
    assert "conflicting_event_count > 0" in outcome
    assert "accepted_event_count = 0" in outcome
    assert (
        "requested_event_count\n                    = duplicate_event_count + rejected_event_count"
        in outcome
    )
    assert "conflicting_event_count <= rejected_event_count" in outcome
    assert "storage_state = 'confirmed'" in outcome
    assert "and not retryable" in outcome
    assert "update public.feedback_ingest_request_outcomes" not in outcome.lower()


def test_writer_lockdown_removes_only_direct_event_insert_from_service_role() -> None:
    sql = LOCKDOWN_MIGRATION.read_text(encoding="utf-8")

    assert "revoke insert on table public.feedback_events from service_role;" in sql
    assert "grant select on table public.feedback_events to service_role;" in sql
    assert "revoke select" not in sql.lower()
    assert "grant insert" not in sql.lower()
    assert "rollout order matters" in sql.lower()
