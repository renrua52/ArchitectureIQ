"""Contracts and local semantics for the SURPRISE-002 report migration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "supabase/migrations/20260712020000_feedback_surprise_report.sql"
QUESTIONS_RPC = "feedback_report_surprise_questions"
QUALITY_RPC = "feedback_report_surprise_quality"


def _function_sql(sql: str, function_name: str) -> str:
    start = sql.index(f"create function public.{function_name}(")
    end = sql.index("\n$function$;", start) + len("\n$function$;")
    return sql[start:end]


def _return_columns(sql: str, function_name: str) -> tuple[str, ...]:
    function_sql = _function_sql(sql, function_name)
    declaration = function_sql.split("language sql", maxsplit=1)[0]
    returns = declaration.split("returns table (", maxsplit=1)[1].rsplit(")", 1)[0]
    return tuple(
        line.strip().split()[0].rstrip(",")
        for line in returns.splitlines()
        if line.strip()
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def test_surprise_report_is_additive_forward_only_and_has_exact_schemas() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = _normalize(sql)

    assert MIGRATION.name > "20260712019000_question_reactions.sql"
    assert sql.startswith("begin;\n")
    assert sql.rstrip().endswith("commit;")
    assert normalized.count("create function public.") == 2
    assert "create or replace" not in normalized
    assert "drop function" not in normalized
    assert "alter table" not in normalized
    assert "feedback_report_business_snapshot(" not in normalized
    assert "business_snapshot_v1 contract unchanged" in normalized

    assert _return_columns(sql, QUESTIONS_RPC) == (
        "question_id",
        "question_version",
        "release_id",
        "family",
        "dataset_id",
        "question_type",
        "answered_attempt_count",
        "rating_count",
        "surprised_count",
        "not_surprised_count",
        "rating_coverage_rate",
        "observed_surprise_rate",
        "posterior_mean",
        "first_rating_at",
        "last_rating_at",
    )
    assert _return_columns(sql, QUALITY_RPC) == (
        "raw_reaction_count",
        "valid_reaction_count",
        "orphan_reaction_count",
        "duplicate_reaction_count",
        "registry_unmatched_reaction_count",
        "invalid_payload_reaction_count",
        "missing_prior_answer_reaction_count",
        "unknown_release_reaction_count",
        "counts_conserved",
        "orphan_breakdown_conserved",
    )


def test_surprise_question_rpc_has_fixed_filters_and_authority() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    function = _normalize(_function_sql(sql, QUESTIONS_RPC))
    signature = (
        "create function public.feedback_report_surprise_questions( "
        "p_release_id text default null, p_family text default null, "
        "p_question_type text default null, p_question_id text default null, "
        "p_from timestamptz default null, p_to timestamptz default null, "
        "p_session_id text default null, p_attempt_id text default null )"
    )

    assert function.startswith(signature)
    assert "from public.feedback_authoritative_events as events" in function
    assert "events.registry_status = 'matched'" in function
    for canonical_dimension in (
        "authoritative_release_id",
        "authoritative_question_id",
        "authoritative_question_version",
        "authoritative_family",
        "authoritative_dataset_id",
        "authoritative_question_type",
    ):
        assert canonical_dimension in function
    for filter_expression in (
        "events.authoritative_release_id = p_release_id",
        "events.authoritative_family = p_family",
        "events.authoritative_question_type = p_question_type",
        "events.authoritative_question_id = p_question_id",
        "events.session_id = p_session_id",
        "events.report_attempt_id = p_attempt_id",
    ):
        assert filter_expression in function
    assert "occurred_at >= p_from" in function
    assert "occurred_at < p_to" in function
    assert "p_from < p_to" in function
    assert "p_limit" not in function
    assert "limit p_limit" not in function
    assert function.endswith("reported.question_version; $function$;")


def test_valid_vote_requires_post_reveal_payload_and_prior_matched_answer() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    questions = _normalize(_function_sql(sql, QUESTIONS_RPC))
    quality = _normalize(_function_sql(sql, QUALITY_RPC))

    for function in (questions, quality):
        assert "event_type = 'question_reaction_submitted'" in function
        assert "payload ->> 'reaction'" in function
        assert "'surprise'" in function
        assert "jsonb_typeof(" in function
        assert "payload -> 'value'" in function
        assert "'boolean'" in function
        assert "payload ->> 'timing'" in function
        assert "'after_reveal'" in function
        assert "event_type = 'answer_submitted'" in function
        assert "registry_status = 'matched'" in function
        assert "answers.session_id = reactions.session_id" in function
        assert "answers.report_attempt_id = reactions.report_attempt_id" in function
        assert (
            "answers.authoritative_release_id = reactions.authoritative_release_id"
            in function
        )
        assert (
            "answers.authoritative_question_id = reactions.authoritative_question_id"
            in function
        )
        assert (
            "answers.authoritative_question_version = "
            "reactions.authoritative_question_version" in function
        )
        assert (
            "( answers.occurred_at, answers.sequence, answers.event_id ) < "
            "( reactions.occurred_at, reactions.sequence, reactions.event_id )"
            in function
        )


def test_vote_identity_has_global_stable_first_valid_deduplication() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    for function_name in (QUESTIONS_RPC, QUALITY_RPC):
        function = _normalize(_function_sql(sql, function_name))
        window = (
            "partition by reactions.session_id, reactions.report_attempt_id, "
            "reactions.authoritative_release_id, "
            "reactions.authoritative_question_id, "
            "reactions.authoritative_question_version order by "
            "reactions.occurred_at, reactions.sequence, reactions.event_id"
        )
        assert "row_number() over" in function
        assert window in function

    questions = _normalize(_function_sql(sql, QUESTIONS_RPC))
    assert questions.index("row_number() over") < questions.index(
        "p_from is null or reactions.occurred_at >= p_from"
    )
    assert "where reactions.reaction_rank = 1" in questions

    quality = _normalize(_function_sql(sql, QUALITY_RPC))
    assert quality.index("eligible_ranked as") < quality.index("windowed as")
    assert "when ranked.reaction_rank = 1 then 'valid'" in quality
    assert "else 'duplicate'" in quality
    assert "where reactions.orphan_reason is null" in quality


def test_counts_rates_beta_prior_and_quality_conservation_are_explicit() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    questions = _normalize(_function_sql(sql, QUESTIONS_RPC))
    quality = _normalize(_function_sql(sql, QUALITY_RPC))

    # A rating is constructed from the exhaustive boolean yes/no counts.
    assert (
        "coalesce(ratings.surprised_count, 0) + "
        "coalesce(ratings.not_surprised_count, 0) )::bigint as rating_count"
        in questions
    )
    assert "/ nullif(answered.answered_attempt_count, 0)" in questions
    assert "as rating_coverage_rate" in questions
    assert "as observed_surprise_rate" in questions
    assert (
        "(1 + coalesce(ratings.surprised_count, 0))::numeric / "
        "( 2 + coalesce(ratings.surprised_count, 0) + "
        "coalesce(ratings.not_surprised_count, 0) )" in questions
    )
    assert "as posterior_mean" in questions

    assert (
        "raw_reaction_count = metrics.valid_reaction_count + "
        "metrics.orphan_reaction_count + metrics.duplicate_reaction_count" in quality
    )
    assert (
        "orphan_reaction_count = metrics.registry_unmatched_reaction_count + "
        "metrics.invalid_payload_reaction_count + "
        "metrics.missing_prior_answer_reaction_count" in quality
    )
    assert "registry_status = 'unknown_release'" in quality
    assert "coalesce( events.authoritative_release_id, " in quality
    assert "events.claimed_release_id" in quality


def test_surprise_rpcs_are_service_role_only_security_invokers() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = _normalize(sql)

    assert normalized.count("stable security invoker set search_path = ''") == 2
    assert "security definer" not in normalized
    assert "disable row level security" not in normalized
    assert "bypassrls" not in normalized
    for name, signature in (
        (
            QUESTIONS_RPC,
            "text, text, text, text, timestamptz, timestamptz, text, text",
        ),
        (
            QUALITY_RPC,
            "text, text, text, text, timestamptz, timestamptz, text, text",
        ),
    ):
        assert (
            f"revoke all on function public.{name}( {signature} ) "
            "from public, anon, authenticated, service_role;"
        ) in normalized
        assert (
            f"grant execute on function public.{name}( {signature} ) to service_role;"
        ) in normalized
    assert "grant execute" in normalized
    assert "to public" not in normalized
    assert "to anon" not in normalized
    assert "to authenticated" not in normalized


@dataclass(frozen=True)
class _Event:
    event_id: str
    event_type: str
    occurred_at: int
    sequence: int
    session_id: str
    attempt_id: str | None
    release_id: str
    question_id: str
    question_version: str
    registry_status: str = "matched"
    payload: dict[str, Any] | None = None

    @property
    def identity(self) -> tuple[str, str | None, str, str, str]:
        return (
            self.session_id,
            self.attempt_id,
            self.release_id,
            self.question_id,
            self.question_version,
        )

    @property
    def order(self) -> tuple[int, int, str]:
        return self.occurred_at, self.sequence, self.event_id


_Status = Literal["valid", "orphan", "duplicate"]


def _payload_is_valid(event: _Event) -> bool:
    payload = event.payload or {}
    return (
        event.attempt_id is not None
        and payload.get("reaction") == "surprise"
        and type(payload.get("value")) is bool
        and payload.get("timing") == "after_reveal"
    )


def _classify_reactions(
    events: list[_Event],
) -> dict[str, tuple[_Status, str | None]]:
    answers = [
        event
        for event in events
        if event.event_type == "answer_submitted"
        and event.registry_status == "matched"
        and event.attempt_id is not None
    ]
    reactions = [
        event for event in events if event.event_type == "question_reaction_submitted"
    ]
    classified: dict[str, tuple[_Status, str | None]] = {}
    eligible: defaultdict[tuple[str, str | None, str, str, str], list[_Event]] = (
        defaultdict(list)
    )

    for reaction in reactions:
        if reaction.registry_status != "matched":
            classified[reaction.event_id] = ("orphan", "registry_unmatched")
        elif not _payload_is_valid(reaction):
            classified[reaction.event_id] = ("orphan", "invalid_payload")
        elif not any(
            answer.identity == reaction.identity and answer.order < reaction.order
            for answer in answers
        ):
            classified[reaction.event_id] = ("orphan", "missing_prior_answer")
        else:
            eligible[reaction.identity].append(reaction)

    for identity_reactions in eligible.values():
        for rank, reaction in enumerate(
            sorted(identity_reactions, key=lambda event: event.order), start=1
        ):
            classified[reaction.event_id] = (
                "valid" if rank == 1 else "duplicate",
                None,
            )
    return classified


def _event(
    event_id: str,
    event_type: str,
    occurred_at: int,
    *,
    session: str,
    attempt: str,
    sequence: int | None = None,
    registry_status: str = "matched",
    payload: dict[str, Any] | None = None,
) -> _Event:
    return _Event(
        event_id=event_id,
        event_type=event_type,
        occurred_at=occurred_at,
        sequence=occurred_at if sequence is None else sequence,
        session_id=session,
        attempt_id=attempt,
        release_id="release_1" if registry_status == "matched" else "release_unknown",
        question_id="q_1",
        question_version="qv_1",
        registry_status=registry_status,
        payload=payload,
    )


def _reaction_payload(value: bool) -> dict[str, Any]:
    return {"reaction": "surprise", "value": value, "timing": "after_reveal"}


def _semantic_fixture() -> list[_Event]:
    return [
        # An early orphan does not permanently occupy the identity.  The first
        # reaction after the answer is valid and the next eligible one duplicates it.
        _event(
            "a_orphan_early",
            "question_reaction_submitted",
            5,
            session="s_a",
            attempt="a_a",
            payload=_reaction_payload(True),
        ),
        _event("a_answer", "answer_submitted", 10, session="s_a", attempt="a_a"),
        _event(
            "a_valid",
            "question_reaction_submitted",
            11,
            session="s_a",
            attempt="a_a",
            payload=_reaction_payload(False),
        ),
        _event(
            "a_duplicate",
            "question_reaction_submitted",
            12,
            session="s_a",
            attempt="a_a",
            payload=_reaction_payload(True),
        ),
        _event("b_answer", "answer_submitted", 20, session="s_b", attempt="a_b"),
        _event(
            "b_valid",
            "question_reaction_submitted",
            21,
            session="s_b",
            attempt="a_b",
            payload=_reaction_payload(True),
        ),
        _event(
            "c_orphan_no_answer",
            "question_reaction_submitted",
            30,
            session="s_c",
            attempt="a_c",
            payload=_reaction_payload(True),
        ),
        _event(
            "d_unknown_release",
            "question_reaction_submitted",
            40,
            session="s_d",
            attempt="a_d",
            registry_status="unknown_release",
            payload=_reaction_payload(True),
        ),
        _event("e_answer", "answer_submitted", 50, session="s_e", attempt="a_e"),
        _event(
            "e_invalid_payload",
            "question_reaction_submitted",
            51,
            session="s_e",
            attempt="a_e",
            payload={
                "reaction": "surprise",
                "value": "yes",
                "timing": "after_reveal",
            },
        ),
        # Repeated answers still contribute one answered-attempt identity.
        _event("f_answer_1", "answer_submitted", 60, session="s_f", attempt="a_f"),
        _event("f_answer_2", "answer_submitted", 61, session="s_f", attempt="a_f"),
        _event(
            "f_valid",
            "question_reaction_submitted",
            62,
            session="s_f",
            attempt="a_f",
            payload=_reaction_payload(True),
        ),
        # Answered but unrated: it belongs in the coverage denominator.
        _event("g_answer", "answer_submitted", 70, session="s_g", attempt="a_g"),
    ]


def test_local_semantics_conserve_bad_votes_and_use_first_valid_reaction() -> None:
    events = _semantic_fixture()
    classified = _classify_reactions(events)
    status_counts = Counter(status for status, _reason in classified.values())
    reason_counts = Counter(
        reason for _status, reason in classified.values() if reason is not None
    )

    assert classified["a_orphan_early"] == ("orphan", "missing_prior_answer")
    assert classified["a_valid"] == ("valid", None)
    assert classified["a_duplicate"] == ("duplicate", None)
    assert status_counts == {"valid": 3, "orphan": 4, "duplicate": 1}
    assert reason_counts == {
        "missing_prior_answer": 2,
        "registry_unmatched": 1,
        "invalid_payload": 1,
    }
    assert len(classified) == sum(status_counts.values())
    assert status_counts["orphan"] == sum(reason_counts.values())

    # Ranking is global, so selecting a later time window cannot promote a
    # previously classified duplicate to valid.
    late_window = {
        event.event_id: classified[event.event_id]
        for event in events
        if event.event_type == "question_reaction_submitted"
        and 12 <= event.occurred_at < 20
    }
    assert late_window == {"a_duplicate": ("duplicate", None)}


def test_local_question_aggregate_has_unique_denominator_and_beta_11_prior() -> None:
    events = _semantic_fixture()
    classified = _classify_reactions(events)
    answered_identities = {
        event.identity
        for event in events
        if event.event_type == "answer_submitted" and event.registry_status == "matched"
    }
    valid_ratings = [
        event
        for event in events
        if classified.get(event.event_id) == ("valid", None)
        and event.identity in answered_identities
    ]
    surprised = sum(event.payload == _reaction_payload(True) for event in valid_ratings)
    not_surprised = sum(
        event.payload == _reaction_payload(False) for event in valid_ratings
    )
    rating_count = surprised + not_surprised
    answered_attempt_count = len(answered_identities)

    assert answered_attempt_count == 5
    assert rating_count == 3 == surprised + not_surprised
    assert (surprised, not_surprised) == (2, 1)
    assert rating_count / answered_attempt_count == 0.6
    assert surprised / rating_count == 2 / 3
    assert (1 + surprised) / (2 + rating_count) == 0.6
    assert (1 + 0) / (2 + 0) == 0.5
