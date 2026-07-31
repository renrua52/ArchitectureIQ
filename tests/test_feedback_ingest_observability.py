"""Static contracts for the append-only ingestion outcome store."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MIGRATION = (
    REPO / "supabase/migrations/20260712010000_feedback_ingest_observability.sql"
)
CONFLICT_MIGRATION = (
    REPO / "supabase/migrations/20260712012000_feedback_event_conflicts.sql"
)
EDGE_FUNCTION = REPO / "supabase/functions/feedback-ingest/index.ts"

TABLE = "public.feedback_ingest_request_outcomes"


def _migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def _table_definition(sql: str) -> str:
    marker = f"create table {TABLE} ("
    return sql.split(marker, maxsplit=1)[1].split("\n);\n\ncreate index", maxsplit=1)[0]


def test_outcome_table_has_exact_sanitized_columns() -> None:
    sql = _migration_sql()
    definition = _table_definition(sql)
    expected_columns = (
        "request_id",
        "schema_version",
        "started_at",
        "finished_at",
        "duration_ms",
        "method",
        "authenticated",
        "included_in_rate",
        "outcome_class",
        "outcome_code",
        "http_status",
        "submission_kind",
        "requested_event_count",
        "accepted_event_count",
        "duplicate_event_count",
        "rejected_event_count",
        "storage_state",
        "retryable",
        "observer_revision",
        "recorded_at",
    )
    declared_columns = tuple(
        match.group(1)
        for line in definition.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*)\s+", line))
        and match.group(1) != "constraint"
    )

    assert declared_columns == expected_columns
    assert "request_id uuid primary key" in definition
    assert "recorded_at timestamptz not null default now()" in definition

    forbidden_columns = {
        "authorization",
        "body",
        "error_message",
        "error_stack",
        "headers",
        "ip",
        "ip_address",
        "payload",
        "token",
        "url",
        "user_agent",
    }
    assert forbidden_columns.isdisjoint(declared_columns)


def test_edge_outcome_row_matches_database_columns_exactly() -> None:
    definition = _table_definition(_migration_sql())
    database_columns = {
        match.group(1)
        for line in definition.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*)\s+", line))
        and match.group(1) != "constraint"
    }
    conflict_sql = CONFLICT_MIGRATION.read_text(encoding="utf-8")
    assert "add column conflicting_event_count integer;" in conflict_sql
    database_columns.add("conflicting_event_count")
    source = EDGE_FUNCTION.read_text(encoding="utf-8")
    row_literal = source.split("  const row = {\n", maxsplit=1)[1].split(
        "\n  };", maxsplit=1
    )[0]
    edge_columns = {
        match.group(1)
        for line in row_literal.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*):", line))
    }

    assert edge_columns == database_columns - {"recorded_at"}
    assert "http_status" in edge_columns


def test_outcome_table_enforces_class_code_count_and_time_consistency() -> None:
    sql = _migration_sql()
    definition = _table_definition(sql)

    assert "check (schema_version = '1.0')" in definition
    assert "check (finished_at >= started_at and duration_ms >= 0)" in definition
    for outcome_class in (
        "success",
        "client_rejection",
        "service_failure",
        "excluded",
    ):
        assert f"'{outcome_class}'" in definition
    for outcome_code in (
        "accepted_only",
        "duplicate_only",
        "mixed_success",
        "request_too_large",
        "invalid_request",
        "invalid_envelope",
        "storage_unavailable",
        "internal_error",
        "method_not_allowed",
        "unauthorized",
        "service_unavailable",
    ):
        assert f"'{outcome_code}'" in definition
    for submission_kind in ("session_trace", "single_comment", "unknown"):
        assert f"'{submission_kind}'" in definition
    for storage_state in (
        "confirmed",
        "not_attempted",
        "not_committed",
        "unknown",
    ):
        assert f"'{storage_state}'" in definition

    assert "requested_event_count > 0" in definition
    assert definition.count("requested_event_count is not null") >= 2
    assert definition.count("accepted_event_count is not null") >= 2
    assert definition.count("duplicate_event_count is not null") >= 2
    assert definition.count("rejected_event_count is not null") >= 2
    assert definition.count("accepted_event_count is null") >= 2
    assert definition.count("duplicate_event_count is null") >= 2
    assert definition.count("rejected_event_count is null") >= 2
    assert "= accepted_event_count + duplicate_event_count" in definition
    assert "requested_event_count = rejected_event_count" in definition
    assert "and authenticated" in definition
    assert "and included_in_rate" in definition
    assert "and not authenticated" in definition
    assert "and not included_in_rate" in definition
    for status in (200, 400, 401, 405, 413, 500, 502, 503):
        assert f"http_status = {status}" in definition
    assert "observer_revision !~ E'[\\r\\n]'" in definition


def test_outcome_store_is_private_indexed_and_append_only() -> None:
    sql = _migration_sql()

    assert sql.startswith("begin;\n")
    assert sql.rstrip().endswith("commit;")
    assert sql.count("create index feedback_ingest_outcomes_") == 4
    assert "where included_in_rate;" in sql
    assert f"alter table {TABLE} enable row level security;" in sql
    assert f"alter table {TABLE} force row level security;" in sql
    assert (
        f"revoke all on table {TABLE}\n"
        "    from public, anon, authenticated, service_role;"
    ) in sql
    assert (f"grant select, insert on table {TABLE}\n    to service_role;") in sql
    assert "grant all" not in sql.lower()
    assert "grant update" not in sql.lower()
    assert "grant delete" not in sql.lower()

    assert "create function public.reject_feedback_ingest_outcome_mutation()" in sql
    assert "feedback_ingest_request_outcomes is append-only" in sql
    assert "before update or delete or truncate" in sql
    assert f"on {TABLE}" in sql
    assert "for each statement execute function" in sql


def test_observability_migration_does_not_mutate_accepted_events_or_reports() -> None:
    sql = _migration_sql().lower()

    assert "alter table public.feedback_events" not in sql
    assert "insert into public.feedback_events" not in sql
    assert "update public.feedback_events" not in sql
    assert "delete from public.feedback_events" not in sql
    assert "feedback_report_summary" not in sql
    assert "ingestion_failure_rate" not in sql
