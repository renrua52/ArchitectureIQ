"""Static contracts for the private OBS-001B aggregation RPC."""

from __future__ import annotations

import re
from pathlib import Path

from tools.feedback_reports import copy_view_columns


REPO = Path(__file__).resolve().parents[1]
MIGRATION = (
    REPO
    / "supabase/migrations/20260712013000_feedback_conflict_observability_report.sql"
)
FUNCTION = "public.feedback_report_ingestion_summary"
VIEW = "feedback_report_ingestion_summary"


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_ingestion_summary_has_stable_single_row_contract() -> None:
    sql = _sql()

    assert f"create function {FUNCTION}(" in sql
    assert "p_from timestamptz default null" in sql
    assert "p_to timestamptz default null" in sql
    assert "p_request_id uuid default null" in sql
    expected_fields = (
        "recorded_request_count bigint",
        "first_started_at timestamptz",
        "last_finished_at timestamptz",
        "success_request_count bigint",
        "client_rejection_count bigint",
        "service_failure_count bigint",
        "event_id_conflict_request_count bigint",
        "accepted_event_count bigint",
        "duplicate_event_count bigint",
        "idempotent_duplicate_event_count bigint",
        "unclassified_duplicate_event_count bigint",
        "conflicting_event_count bigint",
        "conflict_audit_event_count bigint",
        "event_id_reuse_count bigint",
        "classified_event_count bigint",
        "known_event_result_count bigint",
        "request_failure_rate numeric",
        "duplicate_event_rate numeric",
        "event_id_reuse_rate numeric",
        "classified_conflicting_event_rate numeric",
        "recorded_rate_available boolean",
        "end_to_end_coverage_available boolean",
    )
    for field in expected_fields:
        assert field in sql

    assert "count(*) as recorded_request_count" in sql
    assert "min(started_at) as first_started_at" in sql
    assert "max(finished_at) as last_finished_at" in sql
    assert "from derived" in sql
    assert "cross join conflict_audit_metrics;" in sql
    assert "group by" not in sql.lower()
    assert "drop function public.feedback_report_ingestion_summary(" in sql


def test_ingestion_summary_columns_match_strict_python_client() -> None:
    sql = _sql()
    returned = sql.split("returns table (", maxsplit=1)[1].split(
        "\n)\nlanguage sql", maxsplit=1
    )[0]
    database_columns = tuple(
        match.group(1)
        for line in returned.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*)\s+", line))
    )

    assert database_columns == copy_view_columns()[VIEW]
    query_source = (
        REPO / "supabase/functions/feedback-report/report_query.ts"
    ).read_text(encoding="utf-8")
    assert f'"{VIEW}"' in query_source


def test_ingestion_summary_uses_inclusive_exclusive_server_time_filter() -> None:
    sql = _sql()

    assert "from public.feedback_ingest_request_outcomes as outcomes" in sql
    assert "where outcomes.included_in_rate" in sql
    assert "p_request_id is null or outcomes.request_id = p_request_id" in sql
    assert "outcomes.started_at >= p_from" in sql
    assert "outcomes.started_at < p_to" in sql
    assert "p_from is null or p_to is null or p_from < p_to" in sql

    # The range guard belongs inside the filtered CTE. Equal or reversed bounds
    # therefore feed zero rows to the aggregate instead of bypassing filtering.
    filtered = sql.split("with filtered as (", maxsplit=1)[1].split(
        "),\nmetrics as (", maxsplit=1
    )[0]
    assert "p_from < p_to" in filtered
    assert " or p_from >= p_to" not in filtered
    assert " or p_from = p_to" not in filtered


def test_ingestion_summary_calculates_documented_rates_and_zero_denominators() -> None:
    sql = _sql()

    assert "where outcome_class = 'success'" in sql
    assert "where outcome_class = 'client_rejection'" in sql
    assert "where outcome_class = 'service_failure'" in sql
    assert "where outcome_code = 'event_id_conflict'" in sql
    assert "coalesce(sum(accepted_event_count), 0)" in sql
    assert "coalesce(sum(duplicate_event_count), 0)" in sql
    assert "where conflicting_event_count is not null" in sql
    assert "coalesce(sum(conflicting_event_count), 0)" in sql
    assert "from public.feedback_event_conflicts as conflicts" in sql
    assert "filtered.request_id = conflicts.request_id" in sql
    assert "count(*) as conflict_audit_event_count" in sql
    assert "sum(requested_event_count) filter (" in sql
    for outcome_code in (
        "accepted_only",
        "duplicate_only",
        "mixed_success",
        "event_id_conflict",
    ):
        assert f"'{outcome_code}'" in sql
    assert (
        "metrics.duplicate_event_count\n"
        "            - metrics.idempotent_duplicate_event_count\n"
        "            as unclassified_duplicate_event_count"
    ) in sql
    assert (
        "metrics.duplicate_event_count\n"
        "            + metrics.conflicting_event_count\n"
        "            as event_id_reuse_count"
    ) in sql
    assert (
        "metrics.accepted_event_count\n"
        "            + metrics.duplicate_event_count\n"
        "            as known_event_result_count"
    ) in sql

    assert (
        "derived.client_rejection_count\n            + derived.service_failure_count"
    ) in sql
    assert "nullif(derived.recorded_request_count, 0)" in sql
    assert "nullif(derived.known_event_result_count, 0)" in sql
    assert "nullif(derived.classified_event_count, 0)" in sql
    assert (
        "derived.idempotent_duplicate_event_count\n"
        "                + derived.conflicting_event_count"
    ) in sql
    assert sql.count("        4\n") == 4
    assert ("derived.recorded_request_count > 0 as recorded_rate_available") in sql
    assert "false as end_to_end_coverage_available" in sql


def test_ingestion_summary_is_private_read_only_and_does_not_touch_events() -> None:
    sql = _sql()
    lowered = sql.lower()

    assert "language sql" in sql
    assert "stable" in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert (
        f"revoke all on function {FUNCTION}(\n"
        "    timestamptz,\n"
        "    timestamptz,\n"
        "    uuid\n"
        ")\n"
        "from public, anon, authenticated, service_role;"
    ) in sql
    assert (
        f"grant execute on function {FUNCTION}(\n"
        "    timestamptz,\n"
        "    timestamptz,\n"
        "    uuid\n"
        ")\n"
        "to service_role;"
    ) in sql

    assert "public.feedback_events" not in lowered
    assert "public.feedback_event_conflicts" in lowered
    assert "insert into" not in lowered
    assert "update " not in lowered
    assert "delete from" not in lowered
    assert "alter table" not in lowered
