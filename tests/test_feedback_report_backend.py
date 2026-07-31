"""Focused contract checks for the protected Supabase report backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "supabase/migrations/20260712000000_feedback_reports.sql"
RAW_VIEW_HARDENING_MIGRATION = (
    REPO / "supabase/migrations/20260712013500_feedback_raw_view_hardening.sql"
)
DETAIL_REPORT_MIGRATION = (
    REPO / "supabase/migrations/20260712016000_feedback_detail_reports.sql"
)
BUSINESS_SNAPSHOT_MIGRATION = (
    REPO / "supabase/migrations/20260712017000_feedback_business_snapshot.sql"
)
FUNCTION_DIR = REPO / "supabase/functions/feedback-report"
QUERY_MODULE = FUNCTION_DIR / "report_query.ts"
EDGE_FUNCTION = FUNCTION_DIR / "index.ts"


def test_report_migration_has_private_prefiltered_rpc_contract() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    for name in (
        "feedback_report_summary",
        "feedback_report_sessions",
        "feedback_report_questions",
        "feedback_report_comments",
    ):
        assert f"create function public.{name}(" in sql
        assert f"grant execute on function public.{name}(" in sql
    assert sql.count("security invoker") == 4
    assert sql.count("from public.feedback_events as events") == 4
    assert "jsonb_typeof(payload -> 'is_correct') = 'boolean'" in sql
    assert "solve_attempt_count bigint" in sql
    assert "metrics.first_event_at" in sql
    assert "/ nullif(metrics.solve_attempt_count, 0)" in sql
    assert "/ nullif(grouped.solve_attempt_count, 0)" in sql
    assert "null::numeric as ingestion_failure_rate" in sql
    assert "false as ingestion_failure_rate_available" in sql
    assert "from public, anon, authenticated, service_role" in sql

    # Category is a comment-row filter, not a misleading aggregate filter.
    summary_sql, comments_sql = sql.split(
        "create function public.feedback_report_comments(", maxsplit=1
    )
    assert "p_category" not in summary_sql
    assert "p_category" in comments_sql


def test_raw_view_hardening_is_additive_private_and_consistent() -> None:
    sql = RAW_VIEW_HARDENING_MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.split()).lower()
    session_sql, remainder = normalized.split(
        "create or replace view public.feedback_question_stats", maxsplit=1
    )
    question_sql, proposals_sql = remainder.split(
        "create or replace view public.feedback_proposals", maxsplit=1
    )

    assert normalized.count("create or replace view public.") == 3
    assert (
        normalized.count("with (security_invoker = true, security_barrier = true)") == 3
    )
    assert "count(distinct (question_id, question_version)) as question_count" in (
        session_sql
    )
    assert "count(distinct ( session_id, coalesce(" in question_sql
    assert "nullif(payload ->> 'attempt_id', ''), '') )) as attempt_count" in (
        question_sql
    )

    # Existing output columns stay in place; the new quality columns are append-only.
    for section in (session_sql, question_sql):
        positions = [
            section.index(f"as {column}")
            for column in (
                "accuracy",
                "proposal_count",
                "comment_count",
                "known_answer_count",
                "incorrect_answer_count",
                "unknown_answer_count",
            )
        ]
        assert positions == sorted(positions)
        accuracy_prefix = section[: section.index("as accuracy")]
        assert "jsonb_typeof(payload -> 'is_correct') = 'boolean'" in accuracy_prefix

    assert "from public, anon, authenticated" in normalized
    assert "grant select on table" in normalized
    assert "to service_role" in normalized


def test_raw_proposal_view_defensively_handles_json_number_counterexamples() -> None:
    sql = RAW_VIEW_HARDENING_MIGRATION.read_text(encoding="utf-8")
    proposals_sql = " ".join(
        sql.split("create or replace view public.feedback_proposals", maxsplit=1)[1]
        .split("revoke all on table", maxsplit=1)[0]
        .split()
    ).lower()

    for field_name in ("n_seeds", "base_seed"):
        assert (
            f"when jsonb_typeof(payload -> '{field_name}') = 'number' then case"
            in proposals_sql
        )
        assert (
            f"(payload ->> '{field_name}')::numeric = "
            f"trunc((payload ->> '{field_name}')::numeric)" in proposals_sql
        )
        assert "between -2147483648 and 2147483647" in proposals_sql
        assert (
            f"then ((payload ->> '{field_name}')::numeric)::integer else null end "
            f"else null end as {field_name}" in proposals_sql
        )
        assert f"then (payload ->> '{field_name}')::integer" not in proposals_sql


def _return_columns(sql: str, function_name: str) -> tuple[str, ...]:
    function_sql = sql.split(f"create function public.{function_name}(", maxsplit=1)[1]
    declaration = function_sql.split("language sql", maxsplit=1)[0]
    returns = declaration.split("returns table (", maxsplit=1)[1].rsplit(")", 1)[0]
    return tuple(
        line.strip().split()[0].rstrip(",")
        for line in returns.splitlines()
        if line.strip()
    )


def test_detail_report_migration_is_authoritative_private_and_exact() -> None:
    sql = DETAIL_REPORT_MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.split()).lower()
    expected = {
        "feedback_report_answers": (
            "event_id",
            "occurred_at",
            "received_at",
            "session_id",
            "attempt_id",
            "question_id",
            "question_version",
            "release_id",
            "family",
            "dataset_id",
            "question_type",
            "selected_letter",
            "client_selected_candidate_id",
            "selected_candidate_id",
            "answer_status",
            "is_correct",
            "client_is_correct",
            "client_context_mismatch",
            "client_correctness_mismatch",
        ),
        "feedback_report_proposals": (
            "event_id",
            "occurred_at",
            "received_at",
            "session_id",
            "attempt_id",
            "question_id",
            "question_version",
            "release_id",
            "family",
            "dataset_id",
            "question_type",
            "setting_status",
            "label",
            "setting_json",
            "inherited_from_json",
            "n_seeds",
            "base_seed",
            "error_type",
        ),
        "feedback_report_authority_status": (
            "authority_revision",
            "business_reports_authoritative",
            "registered_release_count",
            "registered_question_count",
            "registered_choice_count",
            "detail_revision",
            "detail_reports_authoritative",
        ),
    }
    for function_name, columns in expected.items():
        assert _return_columns(sql, function_name) == columns
        assert f"create function public.{function_name}(" in sql
        assert f"grant execute on function public.{function_name}(" in sql
        assert f"revoke all on function public.{function_name}(" in sql

    assert normalized.count("security invoker") == 3
    assert normalized.count("from public.feedback_authoritative_events as events") == 2
    assert normalized.count("events.registry_status = 'matched'") == 2
    for canonical_field in (
        "authoritative_release_id",
        "authoritative_question_id",
        "authoritative_question_version",
        "authoritative_family",
        "authoritative_dataset_id",
        "authoritative_question_type",
    ):
        assert normalized.count(canonical_field) >= 2
    assert "events.authoritative_is_correct as is_correct" in normalized
    assert "events.payload -> 'is_correct' = 'true'::jsonb" in normalized
    assert normalized.count("order by events.occurred_at desc, events.event_id") == 2
    assert normalized.count("from public, anon, authenticated, service_role") == 3
    assert "drop function public.feedback_report_authority_status()" in normalized
    assert "'detail_v1'::text as detail_revision" in normalized
    assert "true as detail_reports_authoritative" in normalized


def test_detail_proposals_are_setting_events_with_defensive_json_and_integers() -> None:
    sql = " ".join(
        DETAIL_REPORT_MIGRATION.read_text(encoding="utf-8")
        .split("create function public.feedback_report_proposals(", maxsplit=1)[1]
        .split("revoke all on function", maxsplit=1)[0]
        .split()
    ).lower()

    assert "'custom_setting_proposed', 'custom_setting_rejected'" in sql
    assert "custom_run_completed" not in sql
    assert "custom_run_failed" not in sql
    assert "when 'custom_setting_proposed' then 'proposed'" in sql
    assert "when 'custom_setting_rejected' then 'rejected'" in sql
    assert "jsonb_typeof(events.payload -> 'setting') = 'object'" in sql
    assert "then (events.payload -> 'setting')::text" in sql
    assert "jsonb_typeof( events.payload -> 'inherited_from' ) = 'object'" in sql
    assert "then (events.payload -> 'inherited_from')::text" in sql
    for field_name in ("n_seeds", "base_seed"):
        assert f"jsonb_typeof(events.payload -> '{field_name}') = 'number'" in sql
        assert f"(events.payload ->> '{field_name}')::numeric" in sql
    assert sql.count("between -2147483648 and 2147483647") == 2


def test_business_snapshot_migration_is_single_statement_private_and_exact() -> None:
    sql = BUSINESS_SNAPSHOT_MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.split()).lower()

    assert _return_columns(sql, "feedback_report_business_snapshot") == (
        "snapshot_revision",
        "snapshot_at",
        "authority_revision",
        "business_reports_authoritative",
        "registered_release_count",
        "registered_question_count",
        "registered_choice_count",
        "detail_revision",
        "detail_reports_authoritative",
        "pages_json",
    )
    assert normalized.startswith("begin;")
    assert "create function public.feedback_report_business_snapshot(" in normalized
    assert normalized.endswith("commit;")
    assert "language sql stable security invoker set search_path = ''" in normalized
    function_body = sql.split("as $function$", maxsplit=1)[1].split(
        "$function$;", maxsplit=1
    )[0]
    assert function_body.count(";") == 1
    assert "'business_snapshot_v1'::text as snapshot_revision" in normalized
    assert "pg_catalog.statement_timestamp() as snapshot_at" in normalized
    assert (
        "from parameters cross join lateral ( select status.* from "
        "public.feedback_report_authority_status() as status limit "
        "parameters.page_limit ) as status"
    ) in normalized
    assert "p_limit integer default 200" in normalized
    assert "where p_limit between 1 and 1000" in normalized
    assert (
        normalized.count(
            "parameters.release_id, parameters.family, "
            "parameters.question_type, parameters.question_id, "
            "parameters.from_at, parameters.to_at"
        )
        == 6
    )
    assert (
        normalized.count("from parameters cross join lateral public.feedback_report_")
        == 6
    )
    assert "parameters.to_at, null ) as comments" in normalized
    assert "pg_catalog.jsonb_build_object(" in normalized
    assert "pages_document.document::text as pages_json" in normalized
    assert "pages_json jsonb" not in normalized

    assert "report_rows as not materialized" in normalized
    assert "sized_rows as not materialized" in normalized
    assert "budgeted_rows as not materialized" in normalized
    assert "page_results as materialized" in normalized
    bounded_staging_ctes = (
        "summary_page_rows",
        "session_page_rows",
        "question_page_rows",
        "answer_page_rows",
        "proposal_page_rows",
        "comment_page_rows",
    )
    for cte_name in bounded_staging_ctes:
        assert f"{cte_name} as materialized" in normalized
    for removed_wide_cte in (
        "summary_rows as materialized",
        "session_rows as materialized",
        "question_rows as materialized",
        "answer_rows as materialized",
        "proposal_rows as materialized",
        "comment_rows as materialized",
    ):
        assert removed_wide_cte not in normalized

    page_budgets = {65536: 1, 262144: 3, 2621440: 1, 131072: 1}
    assert (
        sum(
            byte_budget * expected_count
            for byte_budget, expected_count in page_budgets.items()
        )
        == 3_604_480
    )
    for byte_budget, expected_count in page_budgets.items():
        assert normalized.count(f"{byte_budget}::bigint") == expected_count
    assert "4194304::bigint as snapshot_pages_bytes" in normalized
    assert "pg_catalog.convert_to(rows.row_json::text, 'utf8')" in normalized
    assert "rows between unbounded preceding and current row" in normalized
    assert "partition by rows.view_name order by rows.page_rank" in normalized
    assert (
        "rows.cumulative_page_bytes + 1 <= definitions.page_byte_budget" in normalized
    )
    assert "pg_catalog.jsonb_agg( rows.row_json order by rows.page_rank )" in normalized
    assert (
        normalized.count(
            "where ranked.snapshot_page_rank <= ( "
            "select parameters.page_limit from parameters )"
        )
        == 6
    )
    staging_sql, report_sql = normalized.split(
        "), report_rows as not materialized (", maxsplit=1
    )
    assert "pg_catalog.to_jsonb(" not in staging_sql
    assert (
        report_sql.count(
            "pg_catalog.to_jsonb(rows) - 'snapshot_page_rank' - 'snapshot_exact_total'"
        )
        == 6
    )
    assert "require an o(n) scan/window for exact totals" in normalized
    assert (
        "pg_catalog.convert_to(pages_document.document::text, 'utf8') ) <= "
        "byte_budgets.snapshot_pages_bytes"
    ) in normalized

    for function_name in (
        "feedback_report_summary",
        "feedback_report_sessions",
        "feedback_report_questions",
        "feedback_report_answers",
        "feedback_report_proposals",
        "feedback_report_comments",
    ):
        assert f"lateral public.{function_name}(" in normalized
        assert normalized.count(f"public.{function_name}(") == 1
        assert normalized.count(f"'{function_name}'") == 6

    assert normalized.count("'rows',") == 6
    assert normalized.count("'total',") == 6
    assert normalized.count("'limit', parameters.page_limit") == 6
    assert normalized.count("'offset', 0") == 6
    assert "sessions.attempt_id asc nulls first" in normalized
    assert (
        "questions.answer_count desc, questions.question_id asc, "
        "questions.question_version asc, questions.release_id asc nulls first, "
        "questions.family asc nulls first, questions.dataset_id asc nulls first, "
        "questions.question_type asc nulls first"
    ) in normalized
    assert (
        normalized.count("order by answers.occurred_at desc, answers.event_id asc") == 1
    )
    assert (
        normalized.count("order by proposals.occurred_at desc, proposals.event_id asc")
        == 1
    )
    assert (
        normalized.count("order by comments.occurred_at desc, comments.event_id asc")
        == 1
    )

    signature = (
        "public.feedback_report_business_snapshot( text, text, text, text, "
        "timestamptz, timestamptz, integer )"
    )
    assert f"revoke all on function {signature}" in normalized
    assert f"grant execute on function {signature}" in normalized
    assert "from public, anon, authenticated, service_role" in normalized
    assert "to service_role" in normalized
    assert "security definer" not in normalized
    assert "insert into" not in normalized
    assert " update " not in normalized
    assert " delete " not in normalized


@pytest.mark.parametrize(
    ("row_bytes", "limit", "budget", "expected_ranks"),
    [
        ((), 5, 2, ()),
        ((3,), 1, 5, (1,)),
        ((3,), 1, 4, ()),
        ((3, 2), 2, 8, (1, 2)),
        ((3, 2), 2, 7, (1,)),
        ((8, 1), 2, 5, ()),
        ((3, 2), 1, 100, (1,)),
    ],
)
def test_business_snapshot_byte_budget_models_exact_ordered_prefix(
    row_bytes: tuple[int, ...],
    limit: int,
    budget: int,
    expected_ranks: tuple[int, ...],
) -> None:
    """Model the SQL prefix and its exact comma/bracket byte accounting."""
    exact_total = len(row_bytes)
    staged = [
        (rank, serialized_bytes, exact_total)
        for rank, serialized_bytes in enumerate(row_bytes[:limit], start=1)
    ]
    cumulative = 0
    selected: list[tuple[int, int]] = []
    for rank, serialized_bytes, _ in staged:
        cumulative += serialized_bytes + 1
        if cumulative + 1 <= budget:
            selected.append((rank, serialized_bytes))

    assert tuple(rank for rank, _ in selected) == expected_ranks
    page_total = max((total for _, _, total in staged), default=0)
    assert page_total == exact_total
    if selected:
        actual_array_bytes = (
            sum(serialized_bytes for _, serialized_bytes in selected)
            + len(selected)
            - 1
            + 2
        )
        assert actual_array_bytes <= budget
    assert expected_ranks == tuple(range(1, len(expected_ranks) + 1))


def test_edge_function_keeps_credentials_private_and_returns_agreed_envelope() -> None:
    source = EDGE_FUNCTION.read_text(encoding="utf-8")

    assert 'request.method !== "GET"' in source
    assert 'Deno.env.get("FEEDBACK_REPORT_TOKEN")' in source
    assert 'Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")' in source
    assert "FEEDBACK_INGEST_TOKEN" not in source
    assert '"Cache-Control": "no-store"' in source
    assert "crypto.subtle.digest" in source
    assert "left[index] ^ right[index]" in source
    assert "`/rest/v1/rpc/${query.view}`" in source
    assert 'query.view === "feedback_report_business_snapshot"' in source
    assert 'query.view === "feedback_report_surprise_questions"' in source
    assert 'query.view === "feedback_report_surprise_quality"' in source
    assert "rawRowsJson = await response.text()" in source
    assert '"rows":${rowsJson}' in source
    assert '"feedback_report_ingestion_summary"' in source
    assert '"feedback_report_authority_status"' in source
    assert '"feedback_report_business_snapshot"' in source
    assert '"feedback_report_registry_quality"' in source
    assert '"feedback_report_surprise_quality"' in source
    assert '"feedback_report_event_resolution"' in source
    assert "feedback_report_answers:" in source
    assert "feedback_report_proposals:" in source
    assert "feedback_report_surprise_questions:" in source
    assert 'Prefer: "count=exact"' in source
    assert 'response.headers.get("Content-Range")' in source
    for field in ("view", "rows", "total", "limit", "offset"):
        assert f'"{field}":' in source


def test_report_query_parser_and_rpc_parameters_execute_under_node() -> None:
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

    script = f"""
import assert from "node:assert/strict";
import {{ parseReportQuery, reportRpcParameters }} from {QUERY_MODULE.as_uri()!r};

const query = parseReportQuery(
  "https://report.example/internal?view=feedback_report_comments" +
  "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
  "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
  "&from=2026-07-01T00%3A00%3A00Z" +
  "&to=2026-07-02T00%3A00%3A00Z&category=bug" +
  "&limit=1000&offset=1000000"
);
assert.equal(query.limit, 1000);
assert.equal(query.offset, 1000000);
assert.equal(query.category, "bug");
assert.deepEqual(Object.keys(reportRpcParameters(query)).sort(), [
  "p_attempt_id", "p_category", "p_family", "p_from",
  "p_question_id", "p_question_type", "p_release_id", "p_session_id", "p_to"
]);

const summary = parseReportQuery(
  "https://report.example/internal?view=feedback_report_summary"
);
assert.equal(summary.limit, 200);
assert.equal(summary.offset, 0);
assert.deepEqual(Object.keys(reportRpcParameters(summary)).sort(), [
  "p_attempt_id", "p_family", "p_from", "p_question_id", "p_question_type",
  "p_release_id", "p_session_id", "p_to"
]);
for (const detailView of [
  "feedback_report_answers",
  "feedback_report_proposals",
]) {{
  const detail = parseReportQuery(
    `https://report.example/internal?view=${{detailView}}` +
    "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
    "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
    "&from=2026-07-01T00%3A00%3A00Z" +
    "&to=2026-07-02T00%3A00%3A00Z&limit=7&offset=3"
  );
  assert.equal(detail.limit, 7);
  assert.equal(detail.offset, 3);
  assert.deepEqual(reportRpcParameters(detail), {{
    p_release_id: "release_abc",
    p_family: "bigram_lm",
    p_question_type: "mixed",
    p_question_id: "q_1",
    p_session_id: "anon_1",
    p_attempt_id: "attempt_1",
    p_from: "2026-07-01T00:00:00Z",
    p_to: "2026-07-02T00:00:00Z",
  }});
}}
parseReportQuery(
  "https://report.example/internal?view=feedback_report_summary" +
  "&from=2026-07-01T00%3A00%3A00.000000001Z" +
  "&to=2026-07-01T00%3A00%3A00.000000002Z"
);

const ingestion = parseReportQuery(
  "https://report.example/internal?view=feedback_report_ingestion_summary" +
  "&from=2026-07-01T00%3A00%3A00Z&to=2026-07-02T00%3A00%3A00Z" +
  "&request_id=72aee12d-7742-44ea-b3d9-f056ae5c8ac2"
);
assert.equal(ingestion.offset, 0);
assert.deepEqual(reportRpcParameters(ingestion), {{
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_request_id: "72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
}});
const unfilteredIngestion = parseReportQuery(
  "https://report.example/internal?view=feedback_report_ingestion_summary"
);
assert.deepEqual(reportRpcParameters(unfilteredIngestion), {{
  p_from: null,
  p_to: null,
  p_request_id: null,
}});

const authority = parseReportQuery(
  "https://report.example/internal?view=feedback_report_authority_status"
);
assert.deepEqual(reportRpcParameters(authority), {{}});

const snapshot = parseReportQuery(
  "https://report.example/internal?view=feedback_report_business_snapshot" +
  "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
  "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
  "&from=2026-07-01T00%3A00%3A00Z" +
  "&to=2026-07-02T00%3A00%3A00Z&limit=17"
);
assert.equal(snapshot.limit, 17);
assert.equal(snapshot.offset, 0);
assert.deepEqual(reportRpcParameters(snapshot), {{
  p_release_id: "release_abc",
  p_family: "bigram_lm",
  p_question_type: "mixed",
  p_question_id: "q_1",
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_limit: 17,
  p_session_id: "anon_1",
  p_attempt_id: "attempt_1",
}});

for (const count of [100, 200]) {{
  const emojiIdentifier = "😀".repeat(count);
  const emojiQuery = parseReportQuery(
    "https://report.example/internal?view=feedback_report_summary&family=" +
      encodeURIComponent(emojiIdentifier)
  );
  assert.equal(emojiQuery.family, emojiIdentifier);
}}
assert.throws(() => parseReportQuery(
  "https://report.example/internal?view=feedback_report_summary&family=" +
    encodeURIComponent("😀".repeat(201))
));

const registry = parseReportQuery(
  "https://report.example/internal?view=feedback_report_registry_quality" +
  "&from=2026-07-01T00%3A00%3A00Z&to=2026-07-02T00%3A00%3A00Z"
);
assert.deepEqual(reportRpcParameters(registry), {{
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
}});

const surpriseQuestions = parseReportQuery(
  "https://report.example/internal?view=feedback_report_surprise_questions" +
  "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
  "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
  "&from=2026-07-01T00%3A00%3A00Z" +
  "&to=2026-07-02T00%3A00%3A00Z&limit=17&offset=3"
);
assert.deepEqual(reportRpcParameters(surpriseQuestions), {{
  p_release_id: "release_abc",
  p_family: "bigram_lm",
  p_question_type: "mixed",
  p_question_id: "q_1",
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_session_id: "anon_1",
  p_attempt_id: "attempt_1",
}});

const surpriseQuality = parseReportQuery(
  "https://report.example/internal?view=feedback_report_surprise_quality" +
  "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
  "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
  "&from=2026-07-01T00%3A00%3A00Z" +
  "&to=2026-07-02T00%3A00%3A00Z"
);
assert.deepEqual(reportRpcParameters(surpriseQuality), {{
  p_release_id: "release_abc",
  p_family: "bigram_lm",
  p_question_type: "mixed",
  p_question_id: "q_1",
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_session_id: "anon_1",
  p_attempt_id: "attempt_1",
}});

const resolution = parseReportQuery(
  "https://report.example/internal?view=feedback_report_event_resolution" +
  "&event_id=evt_exact_1"
);
assert.deepEqual(reportRpcParameters(resolution), {{ p_event_id: "evt_exact_1" }});

for (const value of [
  "?view=unknown",
  "?view=feedback_report_summary&unknown=x",
  "?view=feedback_report_summary&family=a&family=b",
  "?view=feedback_report_summary&limit=1001",
  "?view=feedback_report_summary&offset=1000001",
  "?view=feedback_report_summary&family=%20bad",
  "?view=feedback_report_summary&category=bug",
  "?view=feedback_report_comments&category=invalid",
  "?view=feedback_report_summary&from=2026-02-30T00%3A00%3A00Z",
  "?view=feedback_report_summary&from=2026-07-02T00%3A00%3A00Z" +
    "&to=2026-07-01T00%3A00%3A00Z",
  "?view=feedback_report_ingestion_summary&release_id=release_1",
  "?view=feedback_report_ingestion_summary&family=bigram_lm",
  "?view=feedback_report_ingestion_summary&session_id=anon_1",
  "?view=feedback_report_ingestion_summary&offset=1",
  "?view=feedback_report_authority_status&offset=1",
  "?view=feedback_report_authority_status&from=2026-07-01T00%3A00%3A00Z",
  "?view=feedback_report_authority_status&attempt_id=attempt_1",
  "?view=feedback_report_business_snapshot&offset=1",
  "?view=feedback_report_summary&request_id=72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
  "?view=feedback_report_ingestion_summary&request_id=not-a-uuid",
  "?view=feedback_report_ingestion_summary&request_id=72aee12d-7742-04ea-b3d9-f056ae5c8ac2",
  "?view=feedback_report_ingestion_summary&request_id=72aee12d-7742-44ea-73d9-f056ae5c8ac2",
  "?view=feedback_report_ingestion_summary" +
    "&request_id=72aee12d-7742-44ea-b3d9-f056ae5c8ac2" +
    "&request_id=72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
  "?view=feedback_report_registry_quality&release_id=release_1",
  "?view=feedback_report_registry_quality&session_id=anon_1",
  "?view=feedback_report_registry_quality&request_id=" +
    "72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
  "?view=feedback_report_surprise_quality&offset=1",
  "?view=feedback_report_event_resolution",
  "?view=feedback_report_event_resolution&event_id=evt_1&from=" +
    "2026-07-01T00%3A00%3A00Z",
  "?view=feedback_report_event_resolution&event_id=evt_1&attempt_id=attempt_1",
  "?view=feedback_report_summary&event_id=evt_1",
]) {{
  assert.throws(() => parseReportQuery(`https://report.example/internal${{value}}`));
}}
"""
    subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_edge_report_contract_executes_with_mocked_postgrest() -> None:
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

    script = f"""
import assert from "node:assert/strict";

let handler;
let mode = "page";
let fetchCall = null;
let fetchCount = 0;
const snapshotPages =
  '{{"feedback_report_summary":{{"total":9007199254740993}}}}';
const emptySnapshotPage = (view) => ({{
  view,
  rows: [],
  total: 0,
  limit: 1000,
  offset: 0,
}});
const completeSettingJson = JSON.stringify({{
  note: '"'.repeat(500_000),
}});
const largeSnapshotPages = JSON.stringify({{
  feedback_report_summary: {{
    view: "feedback_report_summary",
    rows: [{{ event_count: 2, proposal_count: 2 }}],
    total: 1,
    limit: 1000,
    offset: 0,
  }},
  feedback_report_sessions: emptySnapshotPage("feedback_report_sessions"),
  feedback_report_questions: emptySnapshotPage("feedback_report_questions"),
  feedback_report_answers: emptySnapshotPage("feedback_report_answers"),
  feedback_report_proposals: {{
    view: "feedback_report_proposals",
    rows: [{{
      event_id: "evt_large_complete_setting",
      setting_json: completeSettingJson,
    }}],
    total: 2,
    limit: 1000,
    offset: 0,
  }},
  feedback_report_comments: emptySnapshotPage("feedback_report_comments"),
}});
globalThis.Deno = {{
  env: {{
    get(name) {{
      return {{
        FEEDBACK_REPORT_TOKEN: "report-secret",
        SUPABASE_URL: "https://project.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY: "service-secret",
      }}[name] ?? null;
    }},
  }},
  serve(callback) {{ handler = callback; }},
}};
globalThis.fetch = async (input, init) => {{
  fetchCount += 1;
  fetchCall = {{ input: String(input), init }};
  if (mode === "past-end") {{
    return new Response('{{"code":"PGRST103"}}', {{
      status: 416,
      headers: {{ "Content-Range": "*/9" }},
    }});
  }}
  if (mode === "summary") {{
    return new Response('[{{"event_count":3}}]', {{ status: 200 }});
  }}
  if (mode === "surprise-precision") {{
    return new Response('[{{"rating_count":9007199254740993}}]', {{
      status: 200,
      headers: {{ "Content-Range": "0-0/1" }},
    }});
  }}
  if (mode === "ingestion") {{
    return new Response('[{{"recorded_request_count":3}}]', {{ status: 200 }});
  }}
  if (mode === "authority") {{
    return new Response(
      '[{{"authority_revision":"registry_v1",' +
      '"business_reports_authoritative":true,' +
      '"registered_release_count":1,"registered_question_count":60,' +
      '"registered_choice_count":180,"detail_revision":"detail_v1",' +
      '"detail_reports_authoritative":true}}]',
      {{ status: 200 }},
    );
  }}
  if (mode === "snapshot") {{
    return new Response(
      '[{{"snapshot_revision":"business_snapshot_v1",' +
      '"snapshot_at":"2026-07-12T00:00:00Z",' +
      '"authority_revision":"registry_v1",' +
      '"business_reports_authoritative":true,' +
      '"registered_release_count":9007199254740993,' +
      '"registered_question_count":9007199254740995,' +
      '"registered_choice_count":18014398509481990,' +
      '"detail_revision":"detail_v1",' +
      '"detail_reports_authoritative":true,' +
      '"pages_json":' + JSON.stringify(snapshotPages) + '}}]',
      {{ status: 200 }},
    );
  }}
  if (mode === "snapshot-large") {{
    return new Response(JSON.stringify([{{
      snapshot_revision: "business_snapshot_v1",
      snapshot_at: "2026-07-12T00:00:00Z",
      authority_revision: "registry_v1",
      business_reports_authoritative: true,
      registered_release_count: 1,
      registered_question_count: 60,
      registered_choice_count: 180,
      detail_revision: "detail_v1",
      detail_reports_authoritative: true,
      pages_json: largeSnapshotPages,
    }}]), {{ status: 200 }});
  }}
  if (mode === "singleton-empty") {{
    return new Response('[]', {{ status: 200 }});
  }}
  if (mode === "singleton-two") {{
    return new Response('[{{}},{{}}]', {{ status: 200 }});
  }}
  return new Response('[{{"session_id":"anon_1"}}]', {{
    status: 206,
    headers: {{ "Content-Range": "5-5/9" }},
  }});
}};
await import({EDGE_FUNCTION.as_uri()!r});

const headers = {{ Authorization: "Bearer report-secret" }};
let response = await handler(new Request(
  "https://edge.example?view=feedback_report_sessions&family=bigram_lm" +
  "&session_id=anon_1&attempt_id=attempt_1&limit=2&offset=5",
  {{ headers }},
));
assert.equal(response.status, 200);
let body = await response.json();
assert.deepEqual(Object.keys(body).sort(), [
  "limit", "offset", "request_id", "rows", "total", "view"
]);
assert.equal(body.total, 9);
assert.equal(body.limit, 2);
assert.equal(body.offset, 5);
assert.deepEqual(body.rows, [{{ session_id: "anon_1" }}]);
assert.ok(fetchCall.input.includes("rpc/feedback_report_sessions"));
assert.match(fetchCall.input, /limit=2/);
assert.match(fetchCall.input, /offset=5/);
assert.equal(fetchCall.init.headers.Prefer, "count=exact");
assert.equal(fetchCall.init.headers.Authorization, "Bearer service-secret");
const rpcBody = JSON.parse(fetchCall.init.body);
assert.equal(rpcBody.p_family, "bigram_lm");
assert.equal(rpcBody.p_session_id, "anon_1");
assert.equal(rpcBody.p_attempt_id, "attempt_1");
assert.ok(!("p_limit" in rpcBody));
assert.ok(!("p_offset" in rpcBody));

for (const detailView of [
  "feedback_report_answers",
  "feedback_report_proposals",
]) {{
  response = await handler(new Request(
    `https://edge.example?view=${{detailView}}&question_id=q_1` +
    "&session_id=anon_1&attempt_id=attempt_1&limit=3&offset=2",
    {{ headers }},
  ));
  body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(body.total, 9);
  const upstream = new URL(fetchCall.input);
  assert.ok(upstream.pathname.endsWith(`/rpc/${{detailView}}`));
  assert.equal(upstream.searchParams.get("order"), "occurred_at.desc,event_id.asc");
  assert.equal(upstream.searchParams.get("limit"), "3");
  assert.equal(upstream.searchParams.get("offset"), "2");
  const detailRpcBody = JSON.parse(fetchCall.init.body);
  assert.equal(detailRpcBody.p_question_id, "q_1");
  assert.equal(detailRpcBody.p_session_id, "anon_1");
  assert.equal(detailRpcBody.p_attempt_id, "attempt_1");
}}

response = await handler(new Request(
  "https://edge.example?view=feedback_report_surprise_questions" +
  "&release_id=release_1&question_id=q_1&session_id=anon_1" +
  "&attempt_id=attempt_1&limit=3&offset=2",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 9);
let upstream = new URL(fetchCall.input);
assert.ok(upstream.pathname.endsWith("/rpc/feedback_report_surprise_questions"));
assert.equal(
  upstream.searchParams.get("order"),
  "posterior_mean.desc,rating_count.desc,release_id.asc," +
    "question_id.asc,question_version.asc",
);
assert.equal(upstream.searchParams.get("limit"), "3");
assert.equal(upstream.searchParams.get("offset"), "2");
let surpriseRpcBody = JSON.parse(fetchCall.init.body);
assert.ok(!("p_limit" in surpriseRpcBody));
assert.equal(surpriseRpcBody.p_session_id, "anon_1");
assert.equal(surpriseRpcBody.p_attempt_id, "attempt_1");

mode = "surprise-precision";
response = await handler(new Request(
  "https://edge.example?view=feedback_report_surprise_questions&limit=3",
  {{ headers }},
));
const surpriseResponseText = await response.text();
assert.match(surpriseResponseText, /"rating_count":9007199254740993(?:,|}})/);
body = JSON.parse(surpriseResponseText);
assert.equal(response.status, 200);
assert.equal(body.total, 1);

mode = "past-end";
response = await handler(new Request(
  "https://edge.example?view=feedback_report_comments&limit=2&offset=20",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 9);
assert.deepEqual(body.rows, []);

mode = "summary";
response = await handler(new Request(
  "https://edge.example?view=feedback_report_summary",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 1);
assert.equal(body.limit, 200);
assert.equal(body.offset, 0);
assert.deepEqual(body.rows, [{{ event_count: 3 }}]);
assert.ok(!fetchCall.input.includes("limit="));
assert.ok(!fetchCall.input.includes("offset="));

response = await handler(new Request(
  "https://edge.example?view=feedback_report_surprise_quality" +
  "&release_id=release_1&question_id=q_1&session_id=anon_1" +
  "&attempt_id=attempt_1",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 1);
upstream = new URL(fetchCall.input);
assert.ok(upstream.pathname.endsWith("/rpc/feedback_report_surprise_quality"));
assert.ok(!upstream.searchParams.has("limit"));
assert.ok(!upstream.searchParams.has("offset"));
surpriseRpcBody = JSON.parse(fetchCall.init.body);
assert.ok(!("p_limit" in surpriseRpcBody));
assert.equal(surpriseRpcBody.p_session_id, "anon_1");
assert.equal(surpriseRpcBody.p_attempt_id, "attempt_1");

mode = "ingestion";
response = await handler(new Request(
  "https://edge.example?view=feedback_report_ingestion_summary" +
  "&from=2026-07-01T00%3A00%3A00Z&to=2026-07-02T00%3A00%3A00Z" +
  "&request_id=72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 1);
assert.deepEqual(body.rows, [{{ recorded_request_count: 3 }}]);
assert.ok(fetchCall.input.includes("rpc/feedback_report_ingestion_summary"));
assert.ok(!fetchCall.input.includes("limit="));
assert.ok(!fetchCall.input.includes("offset="));
assert.ok(!fetchCall.input.includes("order="));
assert.ok(!("Prefer" in fetchCall.init.headers));
assert.deepEqual(JSON.parse(fetchCall.init.body), {{
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_request_id: "72aee12d-7742-44ea-b3d9-f056ae5c8ac2",
}});

mode = "authority";
response = await handler(new Request(
  "https://edge.example?view=feedback_report_authority_status",
  {{ headers }},
));
body = await response.json();
assert.equal(response.status, 200);
assert.equal(body.total, 1);
assert.deepEqual(body.rows, [{{
  authority_revision: "registry_v1",
  business_reports_authoritative: true,
  registered_release_count: 1,
  registered_question_count: 60,
  registered_choice_count: 180,
  detail_revision: "detail_v1",
  detail_reports_authoritative: true,
}}]);
assert.ok(fetchCall.input.includes("rpc/feedback_report_authority_status"));
assert.ok(!fetchCall.input.includes("limit="));
assert.ok(!fetchCall.input.includes("offset="));
assert.ok(!fetchCall.input.includes("order="));
assert.ok(!("Prefer" in fetchCall.init.headers));
assert.deepEqual(JSON.parse(fetchCall.init.body), {{}});

mode = "snapshot";
const fetchCountBeforeSnapshot = fetchCount;
response = await handler(new Request(
  "https://edge.example?view=feedback_report_business_snapshot" +
  "&release_id=release_abc&family=bigram_lm&question_type=mixed" +
  "&question_id=q_1&session_id=anon_1&attempt_id=attempt_1" +
  "&from=2026-07-01T00%3A00%3A00Z" +
  "&to=2026-07-02T00%3A00%3A00Z&limit=17",
  {{ headers }},
));
const snapshotResponseText = await response.text();
assert.match(
  snapshotResponseText,
  /"registered_release_count":9007199254740993(?:,|}})/,
);
assert.match(
  snapshotResponseText,
  /"registered_question_count":9007199254740995(?:,|}})/,
);
assert.match(
  snapshotResponseText,
  /"registered_choice_count":18014398509481990(?:,|}})/,
);
body = JSON.parse(snapshotResponseText);
assert.equal(response.status, 200);
assert.equal(fetchCount, fetchCountBeforeSnapshot + 1);
assert.equal(body.total, 1);
assert.equal(body.limit, 17);
assert.equal(body.offset, 0);
assert.equal(body.rows[0].snapshot_revision, "business_snapshot_v1");
assert.equal(body.rows[0].pages_json, snapshotPages);
assert.ok(fetchCall.input.includes("rpc/feedback_report_business_snapshot"));
assert.ok(!fetchCall.input.includes("limit="));
assert.ok(!fetchCall.input.includes("offset="));
assert.ok(!fetchCall.input.includes("order="));
assert.ok(!("Prefer" in fetchCall.init.headers));
assert.deepEqual(JSON.parse(fetchCall.init.body), {{
  p_release_id: "release_abc",
  p_family: "bigram_lm",
  p_question_type: "mixed",
  p_question_id: "q_1",
  p_from: "2026-07-01T00:00:00Z",
  p_to: "2026-07-02T00:00:00Z",
  p_limit: 17,
  p_session_id: "anon_1",
  p_attempt_id: "attempt_1",
}});

mode = "snapshot-large";
const fetchCountBeforeLargeSnapshot = fetchCount;
response = await handler(new Request(
  "https://edge.example?view=feedback_report_business_snapshot&limit=1000",
  {{ headers }},
));
const largeSnapshotResponseBytes = new Uint8Array(await response.arrayBuffer());
assert.equal(response.status, 200);
assert.equal(fetchCount, fetchCountBeforeLargeSnapshot + 1);
assert.ok(completeSettingJson.length > 1_000_000);
assert.ok(new TextEncoder().encode(completeSettingJson).byteLength < 1024 * 1024);
assert.ok(new TextEncoder().encode(largeSnapshotPages).byteLength < 4 * 1024 * 1024);
assert.ok(largeSnapshotResponseBytes.byteLength < 10 * 1024 * 1024);
assert.ok(largeSnapshotResponseBytes.byteLength > largeSnapshotPages.length);
body = JSON.parse(new TextDecoder().decode(largeSnapshotResponseBytes));
const parsedLargePages = JSON.parse(body.rows[0].pages_json);
const largeProposalPage = parsedLargePages.feedback_report_proposals;
assert.equal(largeProposalPage.limit, 1000);
assert.equal(largeProposalPage.offset, 0);
assert.equal(largeProposalPage.total, 2);
assert.equal(largeProposalPage.rows.length, 1);
assert.equal(
  largeProposalPage.rows[0].setting_json,
  completeSettingJson,
);
assert.equal(
  largeProposalPage.rows[0].event_id,
  "evt_large_complete_setting",
);

for (const invalidMode of ["singleton-empty", "singleton-two"]) {{
  mode = invalidMode;
  response = await handler(new Request(
    "https://edge.example?view=feedback_report_ingestion_summary",
    {{ headers }},
  ));
  assert.equal(response.status, 502);
  body = await response.json();
  assert.equal(body.error.code, "REPORT_UNAVAILABLE");
}}

response = await handler(new Request(
  "https://edge.example?view=feedback_report_summary",
));
assert.equal(response.status, 401);
"""
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", [QUERY_MODULE, EDGE_FUNCTION])
def test_report_typescript_is_parseable_by_node(path: Path) -> None:
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
    subprocess.run(
        [node, "--experimental-strip-types", "--check", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
