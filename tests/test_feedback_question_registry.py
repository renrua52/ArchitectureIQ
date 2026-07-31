"""Static contracts for STATS-003 authoritative question attribution."""

from __future__ import annotations

from pathlib import Path

from tools import feedback_rollout_preflight as preflight
from tools.feedback_reports import copy_view_columns


REPO = Path(__file__).resolve().parents[1]
REGISTRY_MIGRATION = (
    REPO / "supabase/migrations/20260712014000_feedback_question_registry.sql"
)
AUTHORITATIVE_REPORT_MIGRATION = (
    REPO / "supabase/migrations/20260712015000_feedback_authoritative_reports.sql"
)


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).lower()


def test_registry_tables_are_private_append_only_and_complete() -> None:
    sql = _normalized(REGISTRY_MIGRATION)

    for table in (
        "feedback_quiz_releases",
        "feedback_quiz_questions",
        "feedback_quiz_choices",
    ):
        assert f"create table public.{table}" in sql
        assert f"alter table public.{table} enable row level security" in sql
        assert f"alter table public.{table} force row level security" in sql
        assert f"create trigger {table}_append_only" in sql

    assert "from public, anon, authenticated, service_role" in sql
    assert "grant select on table" in sql
    assert "to service_role" in sql
    assert "grant insert" not in sql
    assert "grant update" not in sql
    assert "grant delete" not in sql

    assert "deferrable initially deferred" in sql
    assert "feedback_quiz_questions_correct_choice_fkey" in sql
    assert "feedback_quiz_question_inventory_complete" in sql
    assert "feedback_quiz_choice_inventory_complete" in sql
    assert "feedback_quiz_release_inventory_complete" in sql
    assert "feedback_quiz_question_release_inventory_complete" in sql
    assert "feedback_quiz_choice_release_inventory_complete" in sql
    assert "feedback_quiz_question_version_lock" in sql
    assert "pg_catalog.pg_advisory_xact_lock" in sql
    assert "before insert on public.feedback_quiz_questions" in sql
    assert "after insert on public.feedback_quiz_choices" in sql
    assert "question_version metadata differs across releases" in sql
    assert "question_version choices differ across releases" in sql


def test_authority_projection_never_guesses_or_mutates_raw_events() -> None:
    sql = _normalized(REGISTRY_MIGRATION)
    view = sql.split("create view public.feedback_authoritative_events", maxsplit=1)[
        1
    ].split("revoke all on table public.feedback_authoritative_events", maxsplit=1)[0]

    assert "with (security_invoker = true, security_barrier = true)" in view
    assert "from public.feedback_events as events" in view
    assert "releases.release_id = nullif(events.payload ->> 'release_id', '')" in view
    assert "questions.question_id = events.question_id" in view
    assert "questions.question_version = events.question_version" in view
    assert "choices.letter = nullif(events.payload ->> 'selected_letter', '')" in view
    assert "missing_release" in view
    assert "unknown_release" in view
    assert "question_not_in_release" in view
    assert "invalid_selected_letter" in view
    assert "selected_candidate_mismatch" in view
    assert "authoritative_is_correct" in view
    assert "coalesce(" in view
    assert "pg_catalog.coalesce(" not in view
    assert "false ) as client_correctness_mismatch" in view

    assert "update public.feedback_events" not in sql
    assert "alter table public.feedback_events" not in sql
    assert "coalesce(events.payload ->> 'release_id'" not in view
    assert "coalesce(events.payload ->> 'family'" not in view


def test_registry_quality_and_exact_resolution_are_additive_private_rpcs() -> None:
    sql = REGISTRY_MIGRATION.read_text(encoding="utf-8")
    quality_columns = preflight.extract_sql_return_columns(
        sql, "feedback_report_registry_quality"
    )
    resolution_columns = preflight.extract_sql_return_columns(
        sql, "feedback_report_event_resolution"
    )

    assert quality_columns == (
        "registered_release_count",
        "registered_question_count",
        "registered_choice_count",
        "registry_available",
        "raw_event_count",
        "authoritative_event_count",
        "excluded_event_count",
        "missing_release_event_count",
        "unknown_release_event_count",
        "question_not_in_release_event_count",
        "raw_answer_count",
        "authoritative_answer_count",
        "unresolved_answer_count",
        "invalid_selected_letter_answer_count",
        "selected_candidate_mismatch_answer_count",
        "unmatched_comment_count",
        "unmatched_proposal_count",
        "client_context_mismatch_event_count",
        "client_correctness_mismatch_answer_count",
        "registry_match_rate",
        "answer_resolution_rate",
    )
    assert resolution_columns == (
        "event_id",
        "event_type",
        "occurred_at",
        "received_at",
        "session_id",
        "attempt_id",
        "client_release_id",
        "registry_status",
        "answer_status",
        "registry_id",
        "release_id",
        "question_id",
        "question_version",
        "family",
        "dataset_id",
        "question_type",
        "selected_letter",
        "client_selected_candidate_id",
        "selected_candidate_id",
        "authoritative_is_correct",
        "client_is_correct",
        "client_context_mismatch",
        "client_correctness_mismatch",
    )

    normalized = " ".join(sql.split()).lower()
    assert "p_from timestamptz default null, p_to timestamptz default null" in (
        normalized
    )
    assert (
        "p_release_id"
        not in normalized.split(
            "create function public.feedback_report_registry_quality", maxsplit=1
        )[1].split(
            "create function public.feedback_report_event_resolution", maxsplit=1
        )[0]
    )
    assert "grant execute on function public.feedback_report_registry_quality" in (
        normalized
    )
    assert "grant execute on function public.feedback_report_event_resolution" in (
        normalized
    )
    assert "'not_found'::text" in normalized
    assert "where not exists (select 1 from matched)" in normalized


def test_business_report_schema_is_stable_but_semantics_are_authoritative() -> None:
    sql = AUTHORITATIVE_REPORT_MIGRATION.read_text(encoding="utf-8")
    expected = copy_view_columns()
    for function, view in (
        ("feedback_report_summary", "feedback_report_summary"),
        ("feedback_report_sessions", "feedback_report_sessions"),
        ("feedback_report_questions", "feedback_report_questions"),
        ("feedback_report_comments", "feedback_report_comments"),
    ):
        assert preflight.extract_sql_return_columns(sql, function) == expected[view]

    assert (
        preflight.extract_sql_return_columns(sql, "feedback_report_authority_status")
        == expected["feedback_report_authority_status"][:5]
    )

    normalized = " ".join(sql.split()).lower()
    assert normalized.count("from public.feedback_authoritative_events as events") == 4
    assert normalized.count("events.registry_status = 'matched'") == 4
    assert "authoritative_is_correct" in normalized
    assert "authoritative_release_id" in normalized
    assert "authoritative_family" in normalized
    assert "authoritative_question_type" in normalized

    # Raw context remains available in the private projection for audit, but
    # none of the four business RPCs may use it for dimensions/correctness.
    assert "payload ->> 'release_id'" not in normalized
    assert "payload ->> 'family'" not in normalized
    assert "payload ->> 'dataset_id'" not in normalized
    assert "payload ->> 'question_type'" not in normalized
    assert "payload -> 'is_correct'" not in normalized
    assert "'registry_v1'::text as authority_revision" in normalized
    assert "true as business_reports_authoritative" in normalized
    assert "grant execute on function public.feedback_report_authority_status()" in (
        normalized
    )
