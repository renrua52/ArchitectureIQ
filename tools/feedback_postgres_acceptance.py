#!/usr/bin/env python3
"""Explicit, rollback-only PostgreSQL staging acceptance for feedback storage.

This is deliberately separate from the endpoint roundtrip verifier.  It checks
the deployed PostgreSQL catalog and exercises database constraints directly.
The DSN is accepted only through ``ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN`` and
is never included in output or exception text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Protocol, TextIO


DSN_ENV: Final = "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN"
TARGET_ENV: Final = "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_TARGET"
EVIDENCE_SCHEMA_VERSION: Final = "1.0"
EVIDENCE_TYPE: Final = "architecture_iq_postgres_staging_acceptance"

EXIT_OK: Final = 0
EXIT_CONFIGURATION: Final = 2
EXIT_ACCEPTANCE_FAILED: Final = 3

PASS: Final = "PASS"
FAIL: Final = "FAIL"

EXPECTED_MIGRATION_VERSIONS: Final = (
    "20260711000000",
    "20260712000000",
    "20260712010000",
    "20260712011000",
    "20260712012000",
    "20260712012500",
    "20260712013000",
    "20260712013500",
    "20260712014000",
    "20260712014500",
    "20260712015000",
    "20260712016000",
    "20260712017000",
    "20260712018000",
    "20260712019000",
    "20260712020000",
)

CURRENT_RELEASE_ID: Final = (
    "release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f"
)
CURRENT_MANIFEST_SHA256: Final = (
    "9fa3c9e28aa81dffd7ea751be40245d1f62f01c252b91024e62de0d8bb230005"
)
CURRENT_REGISTRY_ID: Final = (
    "registry_db3f1a166af0b526e08d4eff49539c6a2150653d1940b0fcccbdbfbe0b525131"
)
CURRENT_QUESTION_COUNT: Final = 60
CURRENT_CHOICE_COUNT: Final = 180

_SAFE_TARGET_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_NON_PRODUCTION_TARGET_TOKENS: Final = frozenset(
    {
        "dev",
        "development",
        "preview",
        "qa",
        "sandbox",
        "stage",
        "staging",
        "test",
        "testing",
    }
)
_PRODUCTION_TARGET_TOKENS: Final = frozenset({"live", "main", "prod", "production"})


class AcceptanceConfigurationError(ValueError):
    """Raised before a database connection when staging consent is unsafe."""


class AcceptanceExecutionError(RuntimeError):
    """A sanitized database execution failure safe to expose in evidence."""


class CursorLike(Protocol):
    def execute(
        self,
        query: str,
        params: Sequence[Any] | None = None,
    ) -> Any: ...

    def fetchone(self) -> Sequence[Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...

    def __enter__(self) -> CursorLike: ...

    def __exit__(self, *args: Any) -> None: ...


class ConnectionLike(Protocol):
    def cursor(self) -> CursorLike: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


Connector = Callable[[str], ConnectionLike]


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    argument_types: str
    language: str
    volatility: str
    security_definer: bool
    returns_set: bool
    result_type: str


_SIX_FILTER_ARGS = (
    "text, text, text, text, timestamp with time zone, timestamp with time zone"
)
_EIGHT_FILTER_ARGS = f"{_SIX_FILTER_ARGS}, text, text"
EXPECTED_FUNCTIONS: Final = (
    FunctionSpec(
        "feedback_ingest_events",
        "uuid, text, timestamp with time zone, jsonb",
        "plpgsql",
        "v",
        True,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_logical_event_v1",
        "text, text, text, text, text, text, jsonb",
        "sql",
        "i",
        False,
        False,
        "jsonb",
    ),
    FunctionSpec(
        "feedback_report_summary",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_sessions",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_questions",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_answers",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_proposals",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_comments",
        f"{_SIX_FILTER_ARGS}, text, text, text",
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_ingestion_summary",
        "timestamp with time zone, timestamp with time zone, uuid",
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_registry_quality",
        "timestamp with time zone, timestamp with time zone",
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_event_resolution", "text", "sql", "s", False, True, "record"
    ),
    FunctionSpec(
        "feedback_report_authority_status", "", "sql", "s", False, True, "record"
    ),
    FunctionSpec(
        "feedback_report_business_snapshot",
        f"{_SIX_FILTER_ARGS}, integer, text, text",
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_surprise_questions",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
    FunctionSpec(
        "feedback_report_surprise_quality",
        _EIGHT_FILTER_ARGS,
        "sql",
        "s",
        False,
        True,
        "record",
    ),
)

EXPECTED_TABLE_GRANTS: Final = {
    "feedback_events": (True, False),
    "feedback_ingest_request_outcomes": (True, True),
    "feedback_event_conflicts": (True, False),
    "feedback_quiz_releases": (True, False),
    "feedback_quiz_questions": (True, False),
    "feedback_quiz_choices": (True, False),
}

EXPECTED_TRIGGERS: Final = {
    ("feedback_events", "feedback_events_append_only"): (58, False, False),
    (
        "feedback_ingest_request_outcomes",
        "feedback_ingest_request_outcomes_append_only",
    ): (58, False, False),
    ("feedback_event_conflicts", "feedback_event_conflicts_append_only"): (
        58,
        False,
        False,
    ),
    ("feedback_quiz_releases", "feedback_quiz_releases_append_only"): (
        58,
        False,
        False,
    ),
    ("feedback_quiz_questions", "feedback_quiz_questions_append_only"): (
        58,
        False,
        False,
    ),
    ("feedback_quiz_choices", "feedback_quiz_choices_append_only"): (
        58,
        False,
        False,
    ),
    ("feedback_quiz_questions", "feedback_quiz_question_version_lock"): (
        7,
        False,
        False,
    ),
    ("feedback_quiz_questions", "feedback_quiz_question_inventory_complete"): (
        5,
        True,
        True,
    ),
    ("feedback_quiz_choices", "feedback_quiz_choice_inventory_complete"): (
        5,
        True,
        True,
    ),
    ("feedback_quiz_releases", "feedback_quiz_release_inventory_complete"): (
        5,
        True,
        True,
    ),
    (
        "feedback_quiz_questions",
        "feedback_quiz_question_release_inventory_complete",
    ): (5, True, True),
    ("feedback_quiz_choices", "feedback_quiz_choice_release_inventory_complete"): (
        5,
        True,
        True,
    ),
}

_EXPECTED_CHECK_CONSTRAINTS: Final = {
    "feedback_events": (
        "feedback_events_schema_version_check",
        "feedback_events_event_type_check",
        "feedback_events_question_presented_payload_check",
        "feedback_events_question_reaction_payload_check",
        "feedback_events_sequence_check",
        "feedback_events_payload_object_check",
        "feedback_events_event_id_check",
        "feedback_events_trace_id_check",
        "feedback_events_session_id_check",
        "feedback_events_question_id_check",
        "feedback_events_question_version_check",
    ),
    "feedback_ingest_request_outcomes": (
        "feedback_ingest_outcomes_schema_version_check",
        "feedback_ingest_outcomes_time_check",
        "feedback_ingest_outcomes_method_check",
        "feedback_ingest_outcomes_class_check",
        "feedback_ingest_outcomes_code_check",
        "feedback_ingest_outcomes_submission_kind_check",
        "feedback_ingest_outcomes_count_check",
        "feedback_ingest_outcomes_storage_state_check",
        "feedback_ingest_outcomes_observer_revision_check",
        "feedback_ingest_outcomes_revision_conflict_check",
        "feedback_ingest_outcomes_classification_check",
    ),
    "feedback_event_conflicts": (
        "feedback_event_conflicts_event_id_check",
        "feedback_event_conflicts_revision_check",
    ),
    "feedback_quiz_releases": (
        "feedback_quiz_releases_schema_check",
        "feedback_quiz_releases_release_id_check",
        "feedback_quiz_releases_manifest_sha256_check",
        "feedback_quiz_releases_registry_id_check",
        "feedback_quiz_releases_question_count_check",
        "feedback_quiz_releases_choice_count_check",
    ),
    "feedback_quiz_questions": (
        "feedback_quiz_questions_version_check",
        "feedback_quiz_questions_correct_letter_check",
        "feedback_quiz_questions_choice_count_check",
        "feedback_quiz_questions_identifiers_check",
    ),
    "feedback_quiz_choices": (
        "feedback_quiz_choices_letter_check",
        "feedback_quiz_choices_candidate_id_check",
    ),
}

EXPECTED_CONSTRAINTS: Final = {
    ("feedback_events", "feedback_events_pkey"): ("p", False, False),
    (
        "feedback_ingest_request_outcomes",
        "feedback_ingest_request_outcomes_pkey",
    ): ("p", False, False),
    ("feedback_event_conflicts", "feedback_event_conflicts_pkey"): (
        "p",
        False,
        False,
    ),
    ("feedback_event_conflicts", "feedback_event_conflicts_event_id_fkey"): (
        "f",
        False,
        False,
    ),
    ("feedback_quiz_releases", "feedback_quiz_releases_pkey"): (
        "p",
        False,
        False,
    ),
    ("feedback_quiz_releases", "feedback_quiz_releases_registry_id_key"): (
        "u",
        False,
        False,
    ),
    ("feedback_quiz_questions", "feedback_quiz_questions_pkey"): (
        "p",
        False,
        False,
    ),
    (
        "feedback_quiz_questions",
        "feedback_quiz_questions_release_question_key",
    ): ("u", False, False),
    ("feedback_quiz_questions", "feedback_quiz_questions_release_fkey"): (
        "f",
        False,
        False,
    ),
    (
        "feedback_quiz_questions",
        "feedback_quiz_questions_correct_choice_fkey",
    ): ("f", True, True),
    ("feedback_quiz_choices", "feedback_quiz_choices_pkey"): (
        "p",
        False,
        False,
    ),
    ("feedback_quiz_choices", "feedback_quiz_choices_candidate_key"): (
        "u",
        False,
        False,
    ),
    ("feedback_quiz_choices", "feedback_quiz_choices_letter_candidate_key"): (
        "u",
        False,
        False,
    ),
    ("feedback_quiz_choices", "feedback_quiz_choices_question_fkey"): (
        "f",
        False,
        False,
    ),
    **{
        (table_name, constraint_name): ("c", False, False)
        for table_name, constraint_names in _EXPECTED_CHECK_CONSTRAINTS.items()
        for constraint_name in constraint_names
    },
}


IDENTITY_SQL: Final = """
/* architecture_iq_acceptance:server_identity */
select
    pg_catalog.statement_timestamp(),
    pg_catalog.current_database(),
    current_user,
    pg_catalog.current_setting('server_version_num')::integer,
    pg_catalog.pg_is_in_recovery()
"""

MIGRATIONS_SQL: Final = """
/* architecture_iq_acceptance:migrations */
select migrations.version::text
from supabase_migrations.schema_migrations as migrations
where migrations.version::text between %s and %s
order by migrations.version::text
"""

FUNCTIONS_SQL: Final = """
/* architecture_iq_acceptance:functions */
select
    procedures.proname,
    pg_catalog.oidvectortypes(procedures.proargtypes),
    languages.lanname,
    procedures.provolatile,
    procedures.prosecdef,
    exists (
        select 1
        from pg_catalog.unnest(
            coalesce(procedures.proconfig, array[]::text[])
        ) as options(setting)
        where options.setting in ('search_path=', 'search_path=""')
    ) as empty_search_path,
    procedures.proretset,
    pg_catalog.format_type(procedures.prorettype, null),
    pg_catalog.has_function_privilege('anon', procedures.oid, 'EXECUTE'),
    pg_catalog.has_function_privilege('authenticated', procedures.oid, 'EXECUTE'),
    pg_catalog.has_function_privilege('service_role', procedures.oid, 'EXECUTE')
from pg_catalog.pg_proc as procedures
join pg_catalog.pg_namespace as namespaces
    on namespaces.oid = procedures.pronamespace
join pg_catalog.pg_language as languages
    on languages.oid = procedures.prolang
where namespaces.nspname = 'public'
  and procedures.proname = any(%s::text[])
order by procedures.proname, pg_catalog.oidvectortypes(procedures.proargtypes)
"""

RLS_SQL: Final = """
/* architecture_iq_acceptance:rls */
select
    relations.relname,
    relations.relrowsecurity,
    relations.relforcerowsecurity,
    (
        select pg_catalog.count(*)
        from pg_catalog.pg_policy as policies
        where policies.polrelid = relations.oid
    ) as policy_count
from pg_catalog.pg_class as relations
join pg_catalog.pg_namespace as namespaces on namespaces.oid = relations.relnamespace
where namespaces.nspname = 'public'
  and relations.relname = any(%s::text[])
  and relations.relkind = 'r'
order by relations.relname
"""

TABLE_GRANTS_SQL: Final = """
/* architecture_iq_acceptance:table_grants */
select
    relations.relname,
    roles.rolname,
    pg_catalog.has_table_privilege(roles.oid, relations.oid, 'SELECT'),
    pg_catalog.has_table_privilege(roles.oid, relations.oid, 'INSERT'),
    pg_catalog.has_table_privilege(roles.oid, relations.oid, 'UPDATE'),
    pg_catalog.has_table_privilege(roles.oid, relations.oid, 'DELETE'),
    pg_catalog.has_table_privilege(roles.oid, relations.oid, 'TRUNCATE')
from pg_catalog.pg_class as relations
join pg_catalog.pg_namespace as namespaces on namespaces.oid = relations.relnamespace
cross join pg_catalog.pg_roles as roles
where namespaces.nspname = 'public'
  and relations.relname = any(%s::text[])
  and roles.rolname = any(%s::text[])
order by relations.relname, roles.rolname
"""

TRIGGERS_SQL: Final = """
/* architecture_iq_acceptance:triggers */
select
    relations.relname,
    triggers.tgname,
    triggers.tgenabled,
    triggers.tgisinternal,
    triggers.tgtype::integer,
    coalesce(constraints.condeferrable, false),
    coalesce(constraints.condeferred, false)
from pg_catalog.pg_trigger as triggers
join pg_catalog.pg_class as relations on relations.oid = triggers.tgrelid
join pg_catalog.pg_namespace as namespaces on namespaces.oid = relations.relnamespace
left join pg_catalog.pg_constraint as constraints
    on constraints.oid = triggers.tgconstraint
where namespaces.nspname = 'public'
  and relations.relname = any(%s::text[])
  and not triggers.tgisinternal
order by relations.relname, triggers.tgname
"""

CONSTRAINTS_SQL: Final = """
/* architecture_iq_acceptance:constraints */
select
    relations.relname,
    constraints.conname,
    constraints.contype,
    constraints.condeferrable,
    constraints.condeferred,
    constraints.convalidated
from pg_catalog.pg_constraint as constraints
join pg_catalog.pg_class as relations on relations.oid = constraints.conrelid
join pg_catalog.pg_namespace as namespaces on namespaces.oid = relations.relnamespace
where namespaces.nspname = 'public'
  and constraints.conname = any(%s::text[])
order by relations.relname, constraints.conname
"""

REGISTRY_SQL: Final = """
/* architecture_iq_acceptance:registry_authority */
select
    releases.release_id,
    releases.registry_schema_version,
    releases.manifest_sha256,
    releases.registry_id,
    releases.question_count,
    releases.choice_count,
    (
        select pg_catalog.count(*)
        from public.feedback_quiz_questions as questions
        where questions.release_id = releases.release_id
    ) as actual_question_count,
    (
        select pg_catalog.count(*)
        from public.feedback_quiz_choices as choices
        where choices.release_id = releases.release_id
    ) as actual_choice_count,
    authority.authority_revision,
    authority.business_reports_authoritative,
    authority.registered_release_count,
    authority.registered_question_count,
    authority.registered_choice_count,
    authority.detail_revision,
    authority.detail_reports_authoritative
from public.feedback_quiz_releases as releases
cross join lateral public.feedback_report_authority_status() as authority
where releases.release_id = %s
"""

REGISTRY_CONTENT_SQL: Final = """
/* architecture_iq_acceptance:registry_content */
select
    questions.question_id,
    questions.question_version,
    questions.family,
    questions.dataset_id,
    questions.question_type,
    questions.correct_letter,
    questions.correct_candidate_id,
    questions.choice_count,
    choices.letter,
    choices.candidate_id
from public.feedback_quiz_questions as questions
join public.feedback_quiz_choices as choices
  on choices.release_id = questions.release_id
 and choices.question_id = questions.question_id
 and choices.question_version = questions.question_version
where questions.release_id = %s
order by questions.question_id, questions.question_version, choices.letter
"""

PROBE_RESIDUE_SQL: Final = """
/* architecture_iq_acceptance:probe_residue */
select pg_catalog.count(*)
from public.feedback_quiz_releases
where release_id = any(%s::text[])
"""


@dataclass(frozen=True)
class CheckResult:
    code: str
    status: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "status": self.status, "summary": self.summary}


@dataclass(frozen=True)
class AcceptanceEvidence:
    accepted: bool
    target_label: str | None
    database_contacted: bool
    transaction_rolled_back: bool
    server: Mapping[str, Any] | None
    registry: Mapping[str, Any] | None
    checks: tuple[CheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        passed = sum(check.status == PASS for check in self.checks)
        failed = sum(check.status == FAIL for check in self.checks)
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence_type": EVIDENCE_TYPE,
            "accepted": self.accepted,
            "target_label": self.target_label,
            "database_contacted": self.database_contacted,
            "transaction_rolled_back": self.transaction_rolled_back,
            "server": dict(self.server) if self.server is not None else None,
            "registry": dict(self.registry) if self.registry is not None else None,
            "checks": [check.to_dict() for check in self.checks],
            "summary": {"pass": passed, "fail": failed},
        }


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise AcceptanceConfigurationError("invalid command-line arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run rollback-only ArchitectureIQ PostgreSQL staging acceptance."
    )
    parser.add_argument("--confirm-staging", action="store_true")
    parser.add_argument("--target-label")
    return parser


def _safe_target_label(value: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _SAFE_TARGET_PATTERN.fullmatch(value) is None
    ):
        raise AcceptanceConfigurationError(
            "target label must be 1-64 lowercase ASCII letters, digits, '.', '_', or '-'"
        )
    tokens = frozenset(re.split(r"[._-]+", value))
    if tokens & _PRODUCTION_TARGET_TOKENS or any(
        token.startswith(("live", "main", "prod")) for token in tokens
    ):
        raise AcceptanceConfigurationError(
            "production-like target labels are forbidden"
        )
    if not tokens & _NON_PRODUCTION_TARGET_TOKENS:
        raise AcceptanceConfigurationError(
            "target label must explicitly identify staging, test, dev, sandbox, preview, or qa"
        )
    return value


def _resolve_configuration(
    argv: Sequence[str] | None,
    environ: Mapping[str, str],
) -> tuple[str, str]:
    args = _parser().parse_args(argv)
    if not args.confirm_staging:
        raise AcceptanceConfigurationError("--confirm-staging is required")

    dsn = environ.get(DSN_ENV)
    if (
        not isinstance(dsn, str)
        or not dsn
        or dsn != dsn.strip()
        or len(dsn) > 8192
        or any(character in dsn for character in ("\x00", "\r", "\n"))
    ):
        raise AcceptanceConfigurationError(
            f"{DSN_ENV} must contain one safe non-empty DSN"
        )

    argument_label = args.target_label
    environment_label = environ.get(TARGET_ENV)
    if (
        argument_label is not None
        and environment_label is not None
        and argument_label != environment_label
    ):
        raise AcceptanceConfigurationError(
            "target label argument and environment disagree"
        )
    target_label = argument_label or environment_label
    if target_label is None:
        raise AcceptanceConfigurationError(
            f"--target-label or {TARGET_ENV} is required"
        )
    return dsn, _safe_target_label(target_label)


def _load_psycopg_connector(
    importer: Callable[[str], Any] = importlib.import_module,
) -> Connector:
    try:
        psycopg = importer("psycopg")
    except (ImportError, ModuleNotFoundError) as exc:
        raise AcceptanceConfigurationError(
            "optional PostgreSQL driver is missing; install with "
            '`python -m pip install "psycopg[binary]>=3.1"`'
        ) from exc
    connect = getattr(psycopg, "connect", None)
    if not callable(connect):
        raise AcceptanceConfigurationError(
            "installed psycopg package has no connect API"
        )

    def connector(dsn: str) -> ConnectionLike:
        return connect(
            dsn,
            connect_timeout=10,
            application_name="architecture_iq_postgres_acceptance",
            autocommit=False,
        )

    return connector


def _check(code: str, passed: bool, success: str, failure: str) -> CheckResult:
    return CheckResult(code, PASS if passed else FAIL, success if passed else failure)


def _safe_identity_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 200:
        return None
    if value != value.strip() or any(
        character in value for character in ("\x00", "\r", "\n")
    ):
        return None
    return value


def _rfc3339(value: Any) -> str | None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _registry_content_id(rows: Sequence[Sequence[Any]]) -> str | None:
    """Rebuild the publisher's registry identity core from hosted scalar rows."""
    questions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if len(row) != 10:
            return None
        string_values = (*row[:7], *row[8:10])
        if any(not isinstance(value, str) or not value for value in string_values):
            return None
        choice_count = row[7]
        if (
            not isinstance(choice_count, int)
            or isinstance(choice_count, bool)
            or not 2 <= choice_count <= 26
        ):
            return None
        key = (row[0], row[1])
        metadata = tuple(row[:8])
        question = questions.get(key)
        if question is None:
            question = {
                "metadata": metadata,
                "choices": {},
            }
            questions[key] = question
        elif question["metadata"] != metadata:
            return None
        choices = question["choices"]
        if row[8] in choices:
            return None
        choices[row[8]] = row[9]

    question_documents: list[dict[str, Any]] = []
    for key in sorted(questions):
        question = questions[key]
        metadata = question["metadata"]
        choices = question["choices"]
        if len(choices) != metadata[7]:
            return None
        question_documents.append(
            {
                "question_id": metadata[0],
                "question_version": metadata[1],
                "family": metadata[2],
                "dataset_id": metadata[3],
                "question_type": metadata[4],
                "correct_letter": metadata[5],
                "correct_candidate_id": metadata[6],
                "choices": {letter: choices[letter] for letter in sorted(choices)},
            }
        )
    if (
        len(question_documents) != CURRENT_QUESTION_COUNT
        or sum(len(question["choices"]) for question in question_documents)
        != CURRENT_CHOICE_COUNT
    ):
        return None
    core = {
        "schema_version": "1.0",
        "release_id": CURRENT_RELEASE_ID,
        "question_count": CURRENT_QUESTION_COUNT,
        "choice_count": CURRENT_CHOICE_COUNT,
        "questions": question_documents,
    }
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"registry_{hashlib.sha256(canonical).hexdigest()}"


def _sqlstate(error: BaseException) -> str | None:
    value = getattr(error, "sqlstate", None)
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[0-9A-Z]{5}", value)
        else None
    )


def _expected_failure_probe(
    cursor: CursorLike,
    *,
    index: int,
    statements: Sequence[tuple[str, Sequence[Any] | None]],
    expected_sqlstate: str,
) -> bool:
    savepoint = f"architecture_iq_acceptance_{index}"
    cursor.execute(f"SAVEPOINT {savepoint}")
    observed_sqlstate: str | None = None
    try:
        for statement, params in statements:
            cursor.execute(statement, params)
    except Exception as exc:  # Database adapters expose SQLSTATE on exceptions.
        observed_sqlstate = _sqlstate(exc)
    finally:
        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
    return observed_sqlstate == expected_sqlstate


def _append_only_probes() -> tuple[tuple[str, tuple[tuple[str, None], ...]], ...]:
    # Every protected migration installs a FOR EACH STATEMENT trigger.  A
    # zero-row UPDATE/DELETE therefore exercises the trigger without requiring
    # fixture rows or risking a transient rewrite of accepted facts.
    key_columns = {
        "feedback_events": "event_id",
        "feedback_ingest_request_outcomes": "request_id",
        "feedback_event_conflicts": "request_id",
        "feedback_quiz_releases": "release_id",
        "feedback_quiz_questions": "release_id",
        "feedback_quiz_choices": "release_id",
    }
    probes: list[tuple[str, tuple[tuple[str, None], ...]]] = []
    for table_name, key_column in key_columns.items():
        for operation, statement in (
            (
                "update",
                f"/* architecture_iq_acceptance:probe:{table_name}:update */ "
                f"update public.{table_name} set {key_column} = {key_column} where false",
            ),
            (
                "delete",
                f"/* architecture_iq_acceptance:probe:{table_name}:delete */ "
                f"delete from public.{table_name} where false",
            ),
        ):
            probes.append((f"{table_name}.{operation}", ((statement, None),)))
    for table_name in (
        "feedback_ingest_request_outcomes",
        "feedback_event_conflicts",
    ):
        probes.append(
            (
                f"{table_name}.truncate",
                (
                    (
                        f"/* architecture_iq_acceptance:probe:{table_name}:truncate */ "
                        f"truncate table public.{table_name}",
                        None,
                    ),
                ),
            )
        )
    return tuple(probes)


def _catalog_checks(
    cursor: CursorLike,
) -> tuple[
    list[CheckResult],
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    checks: list[CheckResult] = []

    cursor.execute(IDENTITY_SQL)
    identity_row = cursor.fetchone()
    server: Mapping[str, Any] | None = None
    identity_ok = identity_row is not None and len(identity_row) == 5
    if identity_ok:
        observed_at = _rfc3339(identity_row[0])
        database = _safe_identity_text(identity_row[1])
        role = _safe_identity_text(identity_row[2])
        version_num = identity_row[3]
        in_recovery = identity_row[4]
        identity_ok = (
            observed_at is not None
            and database is not None
            and role is not None
            and isinstance(version_num, int)
            and not isinstance(version_num, bool)
            and version_num >= 140000
            and isinstance(in_recovery, bool)
            and not in_recovery
        )
        if identity_ok:
            server = {
                "observed_at": observed_at,
                "database": database,
                "role": role,
                "server_version_num": version_num,
                "in_recovery": in_recovery,
            }
    checks.append(
        _check(
            "postgres.identity",
            identity_ok,
            "Server time and writable-primary identity were captured.",
            "Server identity is missing, unsafe, unsupported, or in recovery.",
        )
    )

    cursor.execute(
        MIGRATIONS_SQL,
        (EXPECTED_MIGRATION_VERSIONS[0], EXPECTED_MIGRATION_VERSIONS[-1]),
    )
    migration_rows = cursor.fetchall()
    migration_versions = tuple(str(row[0]) for row in migration_rows if len(row) == 1)
    checks.append(
        _check(
            "postgres.migrations",
            len(migration_rows) == len(migration_versions)
            and migration_versions == EXPECTED_MIGRATION_VERSIONS,
            "The hosted migration range exactly matches the reviewed staged order.",
            "The hosted migration range is missing, reordered, duplicated, or unclassified.",
        )
    )

    function_names = sorted({spec.name for spec in EXPECTED_FUNCTIONS})
    cursor.execute(FUNCTIONS_SQL, (function_names,))
    function_rows = cursor.fetchall()
    observed_functions: dict[tuple[str, str], Sequence[Any]] = {}
    function_rows_valid = True
    for row in function_rows:
        if len(row) != 11 or not isinstance(row[0], str) or not isinstance(row[1], str):
            function_rows_valid = False
            continue
        key = (row[0], row[1])
        if key in observed_functions:
            function_rows_valid = False
        observed_functions[key] = row
    expected_function_keys = {
        (spec.name, spec.argument_types) for spec in EXPECTED_FUNCTIONS
    }
    function_shape_ok = (
        function_rows_valid and set(observed_functions) == expected_function_keys
    )
    function_acl_ok = function_shape_ok
    for spec in EXPECTED_FUNCTIONS:
        row = observed_functions.get((spec.name, spec.argument_types))
        if row is None:
            function_shape_ok = False
            function_acl_ok = False
            continue
        function_shape_ok = function_shape_ok and tuple(row[2:8]) == (
            spec.language,
            spec.volatility,
            spec.security_definer,
            True,
            spec.returns_set,
            spec.result_type,
        )
        function_acl_ok = function_acl_ok and tuple(row[8:11]) == (False, False, True)
    checks.append(
        _check(
            "postgres.functions",
            function_shape_ok,
            "All application RPC signatures and execution attributes match.",
            "An application RPC signature, language, volatility, security, or search path differs.",
        )
    )
    checks.append(
        _check(
            "postgres.function_grants",
            function_acl_ok,
            "Application RPC execution is service-role-only.",
            "An application RPC has missing service access or browser-role execution access.",
        )
    )

    table_names = sorted(EXPECTED_TABLE_GRANTS)
    cursor.execute(RLS_SQL, (table_names,))
    rls_rows = cursor.fetchall()
    observed_rls = {str(row[0]): tuple(row[1:4]) for row in rls_rows if len(row) == 4}
    rls_ok = set(observed_rls) == set(table_names) and all(
        observed_rls[name] == (True, True, 0) for name in table_names
    )
    checks.append(
        _check(
            "postgres.rls",
            rls_ok,
            "Every private fact table has forced RLS and no policy.",
            "A private fact table is missing forced RLS or has an unexpected policy.",
        )
    )

    roles = ("anon", "authenticated", "service_role")
    cursor.execute(TABLE_GRANTS_SQL, (table_names, list(roles)))
    grant_rows = cursor.fetchall()
    observed_grants = {
        (str(row[0]), str(row[1])): tuple(row[2:7])
        for row in grant_rows
        if len(row) == 7
    }
    grants_ok = set(observed_grants) == {
        (table_name, role) for table_name in table_names for role in roles
    }
    for table_name, (service_select, service_insert) in EXPECTED_TABLE_GRANTS.items():
        grants_ok = grants_ok and observed_grants.get((table_name, "anon")) == (
            False,
            False,
            False,
            False,
            False,
        )
        grants_ok = grants_ok and observed_grants.get(
            (table_name, "authenticated")
        ) == (False, False, False, False, False)
        grants_ok = grants_ok and observed_grants.get((table_name, "service_role")) == (
            service_select,
            service_insert,
            False,
            False,
            False,
        )
    checks.append(
        _check(
            "postgres.table_grants",
            grants_ok,
            "Private table privileges match the least-privilege service-role matrix.",
            "A private table privilege differs for anon, authenticated, or service_role.",
        )
    )

    cursor.execute(TRIGGERS_SQL, (table_names,))
    trigger_rows = cursor.fetchall()
    observed_triggers = {
        (str(row[0]), str(row[1])): tuple(row[2:7])
        for row in trigger_rows
        if len(row) == 7
    }
    triggers_ok = set(observed_triggers) == set(EXPECTED_TRIGGERS)
    for key, (trigger_type, deferrable, deferred) in EXPECTED_TRIGGERS.items():
        triggers_ok = triggers_ok and observed_triggers.get(key) == (
            "O",
            False,
            trigger_type,
            deferrable,
            deferred,
        )
    checks.append(
        _check(
            "postgres.triggers",
            triggers_ok,
            "Append-only and deferred registry triggers are present and enabled.",
            "An expected append-only or deferred registry trigger is missing or altered.",
        )
    )

    constraint_names = sorted(name for _, name in EXPECTED_CONSTRAINTS)
    cursor.execute(CONSTRAINTS_SQL, (constraint_names,))
    constraint_rows = cursor.fetchall()
    observed_constraints = {
        (str(row[0]), str(row[1])): tuple(row[2:6])
        for row in constraint_rows
        if len(row) == 6
    }
    constraints_ok = set(observed_constraints) == set(EXPECTED_CONSTRAINTS)
    for key, (constraint_type, deferrable, deferred) in EXPECTED_CONSTRAINTS.items():
        constraints_ok = constraints_ok and observed_constraints.get(key) == (
            constraint_type,
            deferrable,
            deferred,
            True,
        )
    checks.append(
        _check(
            "postgres.constraints",
            constraints_ok,
            "Critical event, outcome, conflict, and registry constraints are validated.",
            "A critical database constraint is missing, invalid, or has different deferral semantics.",
        )
    )

    cursor.execute(REGISTRY_SQL, (CURRENT_RELEASE_ID,))
    registry_rows = cursor.fetchall()
    cursor.execute(REGISTRY_CONTENT_SQL, (CURRENT_RELEASE_ID,))
    content_registry_id = _registry_content_id(cursor.fetchall())
    registry: Mapping[str, Any] | None = None
    registry_ok = len(registry_rows) == 1 and len(registry_rows[0]) == 15
    if registry_ok:
        row = registry_rows[0]
        registry_ok = tuple(row[:10]) == (
            CURRENT_RELEASE_ID,
            "1.0",
            CURRENT_MANIFEST_SHA256,
            CURRENT_REGISTRY_ID,
            CURRENT_QUESTION_COUNT,
            CURRENT_CHOICE_COUNT,
            CURRENT_QUESTION_COUNT,
            CURRENT_CHOICE_COUNT,
            "registry_v1",
            True,
        )
        registry_ok = registry_ok and content_registry_id == CURRENT_REGISTRY_ID
        registry_ok = registry_ok and (
            isinstance(row[10], int)
            and not isinstance(row[10], bool)
            and row[10] >= 1
            and isinstance(row[11], int)
            and not isinstance(row[11], bool)
            and row[11] >= CURRENT_QUESTION_COUNT
            and isinstance(row[12], int)
            and not isinstance(row[12], bool)
            and row[12] >= CURRENT_CHOICE_COUNT
            and row[13] == "detail_v1"
            and row[14] is True
        )
        if registry_ok:
            registry = {
                "release_id": CURRENT_RELEASE_ID,
                "registry_id": CURRENT_REGISTRY_ID,
                "question_count": CURRENT_QUESTION_COUNT,
                "choice_count": CURRENT_CHOICE_COUNT,
                "registered_release_count": row[10],
                "registered_question_count": row[11],
                "registered_choice_count": row[12],
                "authority_revision": row[8],
                "detail_revision": row[13],
            }
    checks.append(
        _check(
            "postgres.registry_authority",
            registry_ok,
            "The exact attested 60-question/180-choice content and dual authority revisions match.",
            "The current registry content, release, child counts, or authority revisions differ.",
        )
    )
    return checks, server, registry


def _mutation_checks(cursor: CursorLike) -> list[CheckResult]:
    append_results = []
    next_index = 1
    for _name, statements in _append_only_probes():
        append_results.append(
            _expected_failure_probe(
                cursor,
                index=next_index,
                statements=statements,
                expected_sqlstate="55000",
            )
        )
        next_index += 1

    invalid_token = secrets.token_hex(32)
    invalid_release_id = f"release_{invalid_token}"
    invalid_registry_id = f"registry_{secrets.token_hex(32)}"
    invalid_insert = """
/* architecture_iq_acceptance:probe:registry_check */
insert into public.feedback_quiz_releases (
    release_id, registry_schema_version, manifest_sha256, registry_id,
    question_count, choice_count
) values (%s, '1.0', %s, %s, 0, 0)
"""
    invalid_ok = _expected_failure_probe(
        cursor,
        index=next_index,
        statements=(
            (invalid_insert, (invalid_release_id, invalid_token, invalid_registry_id)),
        ),
        expected_sqlstate="23514",
    )
    next_index += 1

    incomplete_token = secrets.token_hex(32)
    incomplete_release_id = f"release_{incomplete_token}"
    incomplete_registry_id = f"registry_{secrets.token_hex(32)}"
    incomplete_insert = """
/* architecture_iq_acceptance:probe:registry_deferred */
insert into public.feedback_quiz_releases (
    release_id, registry_schema_version, manifest_sha256, registry_id,
    question_count, choice_count
) values (%s, '1.0', %s, %s, 1, 2)
"""
    incomplete_ok = _expected_failure_probe(
        cursor,
        index=next_index,
        statements=(
            (
                incomplete_insert,
                (incomplete_release_id, incomplete_token, incomplete_registry_id),
            ),
            (
                "/* architecture_iq_acceptance:probe:registry_deferred_force */ "
                "set constraints all immediate",
                None,
            ),
        ),
        expected_sqlstate="23514",
    )

    cursor.execute(PROBE_RESIDUE_SQL, ([invalid_release_id, incomplete_release_id],))
    residue_row = cursor.fetchone()
    no_residue = residue_row is not None and tuple(residue_row) == (0,)
    return [
        _check(
            "postgres.append_only_probes",
            all(append_results),
            "Rollback-only UPDATE/DELETE/TRUNCATE probes were rejected by append-only triggers.",
            "At least one append-only mutation was not rejected with the required SQLSTATE.",
        ),
        _check(
            "postgres.registry_counterexamples",
            invalid_ok and incomplete_ok and no_residue,
            "Immediate and deferred invalid registry inserts failed without residue.",
            "A registry counterexample succeeded, raised the wrong SQLSTATE, or left residue.",
        ),
    ]


def run_acceptance(
    *,
    dsn: str,
    target_label: str,
    connector: Connector,
) -> AcceptanceEvidence:
    """Connect, inspect, probe, and always roll back the acceptance transaction."""
    target_label = _safe_target_label(target_label)
    try:
        connection = connector(dsn)
    except Exception:
        raise AcceptanceExecutionError(
            "database connection failed before identity could be captured"
        ) from None

    checks: list[CheckResult] = []
    server: Mapping[str, Any] | None = None
    registry: Mapping[str, Any] | None = None
    database_contacted = True
    rolled_back = False
    execution_failed = False
    try:
        with connection.cursor() as cursor:
            cursor.execute("set local statement_timeout = '15s'")
            cursor.execute("set local lock_timeout = '2s'")
            cursor.execute("set local idle_in_transaction_session_timeout = '30s'")
            catalog_checks, server, registry = _catalog_checks(cursor)
            checks.extend(catalog_checks)
            database_contacted = server is not None
            checks.extend(_mutation_checks(cursor))
    except Exception:
        execution_failed = True
        checks.append(
            CheckResult(
                "postgres.execution",
                FAIL,
                "A database statement failed outside an expected rollback-only probe.",
            )
        )
    finally:
        try:
            connection.rollback()
            rolled_back = True
        except Exception:
            checks.append(
                CheckResult(
                    "postgres.rollback",
                    FAIL,
                    "The acceptance transaction could not be confirmed rolled back.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "postgres.rollback",
                    PASS,
                    "The complete acceptance transaction was rolled back.",
                )
            )
        try:
            connection.close()
        except Exception:
            execution_failed = True
            checks.append(
                CheckResult(
                    "postgres.close",
                    FAIL,
                    "The dedicated acceptance connection did not close cleanly.",
                )
            )

    accepted = (
        database_contacted
        and rolled_back
        and not execution_failed
        and checks
        and all(check.status == PASS for check in checks)
    )
    return AcceptanceEvidence(
        accepted=bool(accepted),
        target_label=target_label,
        database_contacted=database_contacted,
        transaction_rolled_back=rolled_back,
        server=server,
        registry=registry,
        checks=tuple(checks),
    )


def _error_evidence(
    *,
    target_label: str | None,
    code: str,
    summary: str,
) -> AcceptanceEvidence:
    return AcceptanceEvidence(
        accepted=False,
        target_label=target_label,
        database_contacted=False,
        transaction_rolled_back=False,
        server=None,
        registry=None,
        checks=(CheckResult(code, FAIL, summary),),
    )


def _emit(evidence: AcceptanceEvidence, stream: TextIO) -> None:
    print(
        json.dumps(
            evidence.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    connector: Connector | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the explicit staging acceptance and emit one sanitized JSON object."""
    resolved_environ = os.environ if environ is None else environ
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    target_label: str | None = None
    try:
        dsn, target_label = _resolve_configuration(argv, resolved_environ)
        resolved_connector = connector or _load_psycopg_connector()
    except AcceptanceConfigurationError as exc:
        evidence = _error_evidence(
            target_label=target_label,
            code="configuration",
            summary=str(exc),
        )
        _emit(evidence, output)
        print("PostgreSQL staging acceptance was not started.", file=errors)
        return EXIT_CONFIGURATION

    try:
        evidence = run_acceptance(
            dsn=dsn,
            target_label=target_label,
            connector=resolved_connector,
        )
    except AcceptanceExecutionError:
        evidence = _error_evidence(
            target_label=target_label,
            code="postgres.connection",
            summary="Database connection failed before acceptance evidence was captured.",
        )
    _emit(evidence, output)
    if evidence.accepted:
        print("PostgreSQL staging acceptance passed.", file=errors)
        return EXIT_OK
    print("PostgreSQL staging acceptance failed.", file=errors)
    return EXIT_ACCEPTANCE_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
