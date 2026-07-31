"""Offline tests for the explicit PostgreSQL staging acceptance verifier."""

from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import feedback_postgres_acceptance as acceptance


SECRET_DSN = "postgresql://acceptance:super-secret@db.example.test/postgres"
REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    REPO / "supabase" / "registries" / f"{acceptance.CURRENT_RELEASE_ID}.json"
)

FUNCTION_MIGRATIONS = {
    "feedback_ingest_events": "20260712012000_feedback_event_conflicts.sql",
    "feedback_logical_event_v1": "20260712012000_feedback_event_conflicts.sql",
    "feedback_report_summary": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_sessions": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_questions": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_comments": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_ingestion_summary": (
        "20260712013000_feedback_conflict_observability_report.sql"
    ),
    "feedback_report_registry_quality": "20260712014000_feedback_question_registry.sql",
    "feedback_report_event_resolution": "20260712014000_feedback_question_registry.sql",
    "feedback_report_answers": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_proposals": "20260712018000_feedback_session_attempt_filters.sql",
    "feedback_report_authority_status": "20260712016000_feedback_detail_reports.sql",
    "feedback_report_business_snapshot": (
        "20260712018000_feedback_session_attempt_filters.sql"
    ),
    "feedback_report_surprise_questions": (
        "20260712020000_feedback_surprise_report.sql"
    ),
    "feedback_report_surprise_quality": ("20260712020000_feedback_surprise_report.sql"),
}


def _function_declaration(sql: str, function_name: str) -> tuple[str, str]:
    marker = f"create function public.{function_name}("
    assert marker in sql
    tail = sql.rsplit(marker, maxsplit=1)[1]
    arguments, remainder = tail.split(")\nreturns", maxsplit=1)
    segment = (
        f"create function public.{function_name}({arguments})\nreturns" + remainder
    )
    return arguments, segment.split("$function$;", maxsplit=1)[0]


def _canonical_argument_types(arguments: str) -> str:
    if not arguments.strip():
        return ""
    resolved = []
    for parameter in arguments.split(","):
        declaration = re.split(r"\s+default\s+", parameter.strip(), maxsplit=1)[0]
        _name, type_name = declaration.split(maxsplit=1)
        resolved.append(
            "timestamp with time zone" if type_name == "timestamptz" else type_name
        )
    return ", ".join(resolved)


def _registry_content_rows() -> list[tuple[Any, ...]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = []
    for question in registry["questions"]:
        for letter, candidate_id in question["choices"].items():
            rows.append(
                (
                    question["question_id"],
                    question["question_version"],
                    question["family"],
                    question["dataset_id"],
                    question["question_type"],
                    question["correct_letter"],
                    question["correct_candidate_id"],
                    len(question["choices"]),
                    letter,
                    candidate_id,
                )
            )
    return sorted(rows, key=lambda row: (row[0], row[1], row[8]))


class FakeDatabaseError(Exception):
    def __init__(self, sqlstate: str | None = None) -> None:
        super().__init__("diagnostic may contain postgresql://leaked:secret@host/db")
        self.sqlstate = sqlstate


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(
        self,
        query: str,
        params: Any = None,
    ) -> None:
        self.connection.queries.append((query, params))
        normalized = " ".join(query.split()).lower()
        self.rows = []
        if normalized.startswith(
            ("savepoint ", "rollback to savepoint ", "release savepoint ")
        ):
            return
        if normalized.startswith("set local "):
            return
        if "architecture_iq_acceptance:server_identity" in normalized:
            if self.connection.fault == "identity":
                self.rows = [("not-a-timestamp", "postgres", "postgres", 150000, False)]
            else:
                self.rows = [
                    (
                        datetime(2026, 7, 12, 12, 34, 56, tzinfo=timezone.utc),
                        "postgres",
                        "postgres",
                        150008,
                        False,
                    )
                ]
            return
        if "architecture_iq_acceptance:migrations" in normalized:
            if self.connection.fault == "execution":
                raise FakeDatabaseError()
            versions = list(acceptance.EXPECTED_MIGRATION_VERSIONS)
            if self.connection.fault == "migrations":
                versions.pop()
            self.rows = [(version,) for version in versions]
            return
        if "architecture_iq_acceptance:functions" in normalized:
            rows = [
                (
                    spec.name,
                    spec.argument_types,
                    spec.language,
                    spec.volatility,
                    spec.security_definer,
                    True,
                    spec.returns_set,
                    spec.result_type,
                    False,
                    False,
                    True,
                )
                for spec in acceptance.EXPECTED_FUNCTIONS
            ]
            if self.connection.fault == "functions":
                rows[0] = (*rows[0][:3], "s", *rows[0][4:])
            if self.connection.fault == "function_overload":
                rows.append(
                    (
                        "feedback_report_summary",
                        acceptance._SIX_FILTER_ARGS,
                        "sql",
                        "s",
                        False,
                        True,
                        True,
                        "record",
                        False,
                        False,
                        True,
                    )
                )
            if self.connection.fault == "function_grants":
                rows[0] = (*rows[0][:10], False)
            self.rows = rows
            return
        if "architecture_iq_acceptance:rls" in normalized:
            self.rows = [
                (name, True, self.connection.fault != "rls" or index != 0, 0)
                for index, name in enumerate(sorted(acceptance.EXPECTED_TABLE_GRANTS))
            ]
            return
        if "architecture_iq_acceptance:table_grants" in normalized:
            rows = []
            for table_name in sorted(acceptance.EXPECTED_TABLE_GRANTS):
                service_select, service_insert = acceptance.EXPECTED_TABLE_GRANTS[
                    table_name
                ]
                rows.extend(
                    [
                        (table_name, "anon", False, False, False, False, False),
                        (
                            table_name,
                            "authenticated",
                            False,
                            False,
                            False,
                            False,
                            False,
                        ),
                        (
                            table_name,
                            "service_role",
                            service_select,
                            service_insert,
                            False,
                            False,
                            False,
                        ),
                    ]
                )
            if self.connection.fault == "table_grants":
                rows[-1] = (*rows[-1][:-1], True)
            self.rows = rows
            return
        if "architecture_iq_acceptance:triggers" in normalized:
            rows = [
                (
                    table_name,
                    trigger_name,
                    "O",
                    False,
                    trigger_type,
                    deferrable,
                    deferred,
                )
                for (table_name, trigger_name), (
                    trigger_type,
                    deferrable,
                    deferred,
                ) in acceptance.EXPECTED_TRIGGERS.items()
            ]
            if self.connection.fault == "triggers":
                rows.pop()
            self.rows = rows
            return
        if "architecture_iq_acceptance:constraints" in normalized:
            rows = [
                (
                    table_name,
                    constraint_name,
                    constraint_type,
                    deferrable,
                    deferred,
                    True,
                )
                for (table_name, constraint_name), (
                    constraint_type,
                    deferrable,
                    deferred,
                ) in acceptance.EXPECTED_CONSTRAINTS.items()
            ]
            if self.connection.fault == "constraints":
                rows[0] = (*rows[0][:-1], False)
            self.rows = rows
            return
        if "architecture_iq_acceptance:registry_authority" in normalized:
            question_count = (
                acceptance.CURRENT_QUESTION_COUNT - 1
                if self.connection.fault == "registry"
                else acceptance.CURRENT_QUESTION_COUNT
            )
            self.rows = [
                (
                    acceptance.CURRENT_RELEASE_ID,
                    "1.0",
                    acceptance.CURRENT_MANIFEST_SHA256,
                    acceptance.CURRENT_REGISTRY_ID,
                    acceptance.CURRENT_QUESTION_COUNT,
                    acceptance.CURRENT_CHOICE_COUNT,
                    question_count,
                    acceptance.CURRENT_CHOICE_COUNT,
                    "registry_v1",
                    True,
                    1,
                    acceptance.CURRENT_QUESTION_COUNT,
                    acceptance.CURRENT_CHOICE_COUNT,
                    "detail_v1",
                    True,
                )
            ]
            return
        if "architecture_iq_acceptance:registry_content" in normalized:
            rows = _registry_content_rows()
            if self.connection.fault == "registry_content":
                rows[0] = (*rows[0][:-1], "c_tampered")
            self.rows = rows
            return
        if "architecture_iq_acceptance:probe_residue" in normalized:
            self.rows = [(0,)]
            return
        if "architecture_iq_acceptance:probe:registry_check" in normalized:
            raise FakeDatabaseError("23514")
        if "architecture_iq_acceptance:probe:registry_deferred_force" in normalized:
            if self.connection.fault != "registry_counterexamples":
                raise FakeDatabaseError("23514")
            return
        if "architecture_iq_acceptance:probe:registry_deferred" in normalized:
            return
        if "architecture_iq_acceptance:probe:" in normalized:
            if (
                self.connection.fault == "append_only"
                and not self.connection.probe_fault_used
            ):
                self.connection.probe_fault_used = True
                return
            raise FakeDatabaseError("55000")
        raise AssertionError(f"unexpected SQL in acceptance test: {normalized[:120]}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)


class FakeConnection:
    def __init__(self, *, fault: str | None = None) -> None:
        self.fault = fault
        self.probe_fault_used = False
        self.queries: list[tuple[str, Any]] = []
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def rollback(self) -> None:
        if self.fault == "rollback":
            raise FakeDatabaseError()
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _connector(
    connection: FakeConnection,
    captured: list[str] | None = None,
) -> acceptance.Connector:
    def connect(dsn: str) -> FakeConnection:
        if captured is not None:
            captured.append(dsn)
        return connection

    return connect


def _checks(evidence: acceptance.AcceptanceEvidence) -> dict[str, str]:
    return {check.code: check.status for check in evidence.checks}


def _valid_environment(**values: str) -> dict[str, str]:
    return {
        acceptance.DSN_ENV: SECRET_DSN,
        acceptance.TARGET_ENV: "architecture-iq-staging",
        **values,
    }


def test_catalog_expectations_match_the_final_migration_definitions() -> None:
    migration_root = REPO / "supabase" / "migrations"
    migration_text = {
        filename: (migration_root / filename).read_text(encoding="utf-8")
        for filename in set(FUNCTION_MIGRATIONS.values())
    }

    for spec in acceptance.EXPECTED_FUNCTIONS:
        sql = migration_text[FUNCTION_MIGRATIONS[spec.name]]
        arguments, segment = _function_declaration(sql, spec.name)
        normalized_segment = " ".join(segment.split()).lower()
        assert _canonical_argument_types(arguments) == spec.argument_types
        assert f"language {spec.language}" in normalized_segment
        assert {
            "i": "immutable",
            "s": "stable",
            "v": "volatile",
        }[spec.volatility] in normalized_segment
        assert ("security definer" in normalized_segment) is spec.security_definer
        assert "set search_path = ''" in normalized_segment
        assert ("returns table (" in normalized_segment) is spec.returns_set
        assert (
            f"grant execute on function public.{spec.name}("
            in " ".join(sql.split()).lower()
        )

    structural_files = (
        "20260711000000_feedback_events.sql",
        "20260712010000_feedback_ingest_observability.sql",
        "20260712012000_feedback_event_conflicts.sql",
        "20260712014000_feedback_question_registry.sql",
        "20260712019000_question_reactions.sql",
    )
    structural_sql = " ".join(
        (migration_root / filename).read_text(encoding="utf-8")
        for filename in structural_files
    ).lower()
    normalized_structural = " ".join(structural_sql.split())
    for (_table_name, trigger_name), (
        trigger_type,
        deferrable,
        deferred,
    ) in acceptance.EXPECTED_TRIGGERS.items():
        assert f"create trigger {trigger_name}" in structural_sql or (
            f"create constraint trigger {trigger_name}" in structural_sql
        )
        if trigger_type == 58:
            trigger_sql = structural_sql.split(
                f"create trigger {trigger_name}", maxsplit=1
            )[1].split(";", maxsplit=1)[0]
            assert "before update or delete or truncate" in trigger_sql
            assert "for each statement" in trigger_sql
        elif trigger_type == 7:
            trigger_sql = structural_sql.split(
                f"create trigger {trigger_name}", maxsplit=1
            )[1].split(";", maxsplit=1)[0]
            assert "before insert" in trigger_sql
            assert "for each row" in trigger_sql
        else:
            assert trigger_type == 5
            trigger_sql = structural_sql.split(
                f"create constraint trigger {trigger_name}", maxsplit=1
            )[1].split(";", maxsplit=1)[0]
            assert "after insert" in trigger_sql
            assert "for each row" in trigger_sql
            assert ("deferrable" in trigger_sql) is deferrable
            assert ("initially deferred" in trigger_sql) is deferred

    implicit_constraint_names = {
        "feedback_events_pkey",
        "feedback_ingest_request_outcomes_pkey",
        "feedback_quiz_releases_pkey",
        "feedback_quiz_releases_registry_id_key",
    }
    constraint_keywords = {
        "c": "check",
        "f": "foreign key",
        "p": "primary key",
        "u": "unique",
    }
    for (_table_name, constraint_name), (
        constraint_type,
        _deferrable,
        _deferred,
    ) in acceptance.EXPECTED_CONSTRAINTS.items():
        if constraint_name not in implicit_constraint_names:
            marker = f"constraint {constraint_name}"
            assert marker in normalized_structural
            declaration = normalized_structural.split(marker, maxsplit=1)[1][:500]
            assert constraint_keywords[constraint_type] in declaration
            assert "not valid" not in declaration.split(",", maxsplit=1)[0]
    assert "event_id text primary key" in structural_sql
    assert "request_id uuid primary key" in structural_sql
    assert "release_id text primary key" in structural_sql
    assert "registry_id text not null unique" in structural_sql

    for table_name in acceptance.EXPECTED_TABLE_GRANTS:
        assert f"alter table public.{table_name} enable row level security" in (
            structural_sql
        )
        assert f"alter table public.{table_name} force row level security" in (
            structural_sql
        )
    lockdown_sql = (
        migration_root / "20260712012500_feedback_event_writer_lockdown.sql"
    ).read_text(encoding="utf-8")
    normalized_lockdown = " ".join(lockdown_sql.split()).lower()
    assert "revoke insert on table public.feedback_events from service_role" in (
        normalized_lockdown
    )
    assert "grant select on table public.feedback_events to service_role" in (
        normalized_lockdown
    )
    assert (
        "grant select, insert on table public.feedback_ingest_request_outcomes "
        "to service_role"
    ) in " ".join(
        (migration_root / "20260712010000_feedback_ingest_observability.sql")
        .read_text(encoding="utf-8")
        .split()
    ).lower()
    assert "grant select on table public.feedback_event_conflicts to service_role" in (
        normalized_structural
    )
    assert (
        "grant insert on table public.feedback_event_conflicts"
        not in normalized_structural
    )
    assert (
        "grant select on table public.feedback_quiz_releases, "
        "public.feedback_quiz_questions, public.feedback_quiz_choices to service_role"
    ) in normalized_structural

    assert "pg_catalog.pg_options_to_table" not in acceptance.FUNCTIONS_SQL
    assert "pg_catalog.unnest(" in acceptance.FUNCTIONS_SQL
    assert "'search_path=', 'search_path=\"\"'" in acceptance.FUNCTIONS_SQL
    # COALESCE is PostgreSQL grammar, not a pg_catalog function.  A qualified
    # spelling parses as an ordinary function call and fails at execution.
    assert "pg_catalog.coalesce(" not in acceptance.FUNCTIONS_SQL
    assert "pg_catalog.coalesce(" not in acceptance.TRIGGERS_SQL
    for migration in migration_root.glob("*.sql"):
        assert "pg_catalog.coalesce(" not in migration.read_text(encoding="utf-8")


def test_full_acceptance_uses_injected_boundary_and_always_rolls_back() -> None:
    connection = FakeConnection()
    captured: list[str] = []

    evidence = acceptance.run_acceptance(
        dsn=SECRET_DSN,
        target_label="architecture-iq-staging",
        connector=_connector(connection, captured),
    )

    assert evidence.accepted is True
    assert evidence.database_contacted is True
    assert evidence.transaction_rolled_back is True
    assert all(status == acceptance.PASS for status in _checks(evidence).values())
    assert captured == [SECRET_DSN]
    assert connection.rolled_back is True
    assert connection.closed is True
    assert not any(
        "commit" in " ".join(query.split()).lower() for query, _ in connection.queries
    )
    assert (
        sum(
            query.lower().startswith("savepoint architecture_iq_acceptance_")
            for query, _ in connection.queries
        )
        == 16
    )
    assert evidence.registry is not None
    assert evidence.registry["question_count"] == 60
    assert evidence.registry["choice_count"] == 180


def test_main_emits_strict_versioned_json_without_dsn_or_diagnostics() -> None:
    connection = FakeConnection()
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = acceptance.main(
        ["--confirm-staging"],
        environ=_valid_environment(),
        connector=_connector(connection),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == acceptance.EXIT_OK
    payload = json.loads(stdout.getvalue())
    assert set(payload) == {
        "schema_version",
        "evidence_type",
        "accepted",
        "target_label",
        "database_contacted",
        "transaction_rolled_back",
        "server",
        "registry",
        "checks",
        "summary",
    }
    assert payload["schema_version"] == "1.0"
    assert payload["accepted"] is True
    assert payload["database_contacted"] is True
    assert payload["server"]["observed_at"].endswith("Z")
    assert payload["summary"] == {"fail": 0, "pass": len(payload["checks"])}
    combined = stdout.getvalue() + stderr.getvalue()
    assert SECRET_DSN not in combined
    assert "super-secret" not in combined
    assert "diagnostic may contain" not in combined


@pytest.mark.parametrize(
    ("argv", "environ"),
    [
        ([], _valid_environment()),
        (["--confirm-staging"], {}),
        (
            ["--confirm-staging"],
            {acceptance.DSN_ENV: SECRET_DSN},
        ),
        (
            ["--confirm-staging"],
            {
                acceptance.DSN_ENV: 123,  # type: ignore[dict-item]
                acceptance.TARGET_ENV: "architecture-iq-staging",
            },
        ),
        (
            ["--confirm-staging", "--target-label", "architecture-iq-test"],
            _valid_environment(),
        ),
        (
            ["--confirm-staging", f"--dsn={SECRET_DSN}"],
            _valid_environment(),
        ),
    ],
)
def test_configuration_fails_before_connecting_without_leaking_secrets(
    argv: list[str],
    environ: dict[str, str],
) -> None:
    calls: list[str] = []
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = acceptance.main(
        argv,
        environ=environ,
        connector=lambda dsn: calls.append(dsn),  # type: ignore[arg-type,return-value]
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == acceptance.EXIT_CONFIGURATION
    assert calls == []
    payload = json.loads(stdout.getvalue())
    assert payload["accepted"] is False
    assert payload["database_contacted"] is False
    assert payload["checks"][0]["code"] == "configuration"
    combined = stdout.getvalue() + stderr.getvalue()
    assert SECRET_DSN not in combined
    assert "super-secret" not in combined


@pytest.mark.parametrize(
    "label",
    [
        "production",
        "architecture-iq-prod",
        "architecture-iq-live",
        "architecture-iq-main",
        "architecture-iq-staging-prod2",
        "architecture-iq-staging-production2",
        "architecture-iq-internal",
        "Staging",
        "staging secret",
    ],
)
def test_target_label_must_unambiguously_name_non_production(label: str) -> None:
    with pytest.raises(acceptance.AcceptanceConfigurationError):
        acceptance._resolve_configuration(
            ["--confirm-staging", "--target-label", label],
            {acceptance.DSN_ENV: SECRET_DSN},
        )


def test_lazy_psycopg_import_has_actionable_install_guidance() -> None:
    def missing(name: str) -> Any:
        assert name == "psycopg"
        raise ModuleNotFoundError(name)

    with pytest.raises(
        acceptance.AcceptanceConfigurationError,
        match=r"psycopg\[binary\]",
    ):
        acceptance._load_psycopg_connector(missing)


def test_default_connector_forces_non_autocommit_and_bounded_connect() -> None:
    connection = FakeConnection()
    captured: dict[str, Any] = {}

    class FakePsycopg:
        @staticmethod
        def connect(dsn: str, **kwargs: Any) -> FakeConnection:
            captured.update({"dsn": dsn, **kwargs})
            return connection

    connector = acceptance._load_psycopg_connector(
        lambda name: FakePsycopg if name == "psycopg" else None
    )

    assert connector(SECRET_DSN) is connection
    assert captured == {
        "dsn": SECRET_DSN,
        "connect_timeout": 10,
        "application_name": "architecture_iq_postgres_acceptance",
        "autocommit": False,
    }


def test_target_label_may_come_only_from_the_safe_argument() -> None:
    dsn, target = acceptance._resolve_configuration(
        ["--confirm-staging", "--target-label", "architecture-iq-test"],
        {acceptance.DSN_ENV: SECRET_DSN},
    )

    assert dsn == SECRET_DSN
    assert target == "architecture-iq-test"


def test_injected_acceptance_boundary_also_rejects_production_label() -> None:
    calls: list[str] = []

    with pytest.raises(acceptance.AcceptanceConfigurationError):
        acceptance.run_acceptance(
            dsn=SECRET_DSN,
            target_label="architecture-iq-production",
            connector=lambda dsn: calls.append(dsn),  # type: ignore[arg-type,return-value]
        )

    assert calls == []


@pytest.mark.parametrize(
    ("fault", "failed_code"),
    [
        ("identity", "postgres.identity"),
        ("migrations", "postgres.migrations"),
        ("functions", "postgres.functions"),
        ("function_overload", "postgres.functions"),
        ("function_grants", "postgres.function_grants"),
        ("rls", "postgres.rls"),
        ("table_grants", "postgres.table_grants"),
        ("triggers", "postgres.triggers"),
        ("constraints", "postgres.constraints"),
        ("registry", "postgres.registry_authority"),
        ("registry_content", "postgres.registry_authority"),
        ("append_only", "postgres.append_only_probes"),
        ("registry_counterexamples", "postgres.registry_counterexamples"),
    ],
)
def test_each_acceptance_boundary_fails_closed(
    fault: str,
    failed_code: str,
) -> None:
    connection = FakeConnection(fault=fault)

    evidence = acceptance.run_acceptance(
        dsn=SECRET_DSN,
        target_label="architecture-iq-staging",
        connector=_connector(connection),
    )

    assert evidence.accepted is False
    assert _checks(evidence)[failed_code] == acceptance.FAIL
    assert evidence.transaction_rolled_back is True
    assert connection.closed is True


def test_unexpected_database_error_and_rollback_failure_are_sanitized() -> None:
    connection = FakeConnection(fault="execution")
    evidence = acceptance.run_acceptance(
        dsn=SECRET_DSN,
        target_label="architecture-iq-staging",
        connector=_connector(connection),
    )
    serialized = json.dumps(evidence.to_dict(), sort_keys=True)

    assert evidence.accepted is False
    assert _checks(evidence)["postgres.execution"] == acceptance.FAIL
    assert "postgresql://" not in serialized
    assert "leaked" not in serialized
    assert connection.rolled_back is True

    rollback_connection = FakeConnection(fault="rollback")
    rollback_evidence = acceptance.run_acceptance(
        dsn=SECRET_DSN,
        target_label="architecture-iq-staging",
        connector=_connector(rollback_connection),
    )
    assert rollback_evidence.accepted is False
    assert rollback_evidence.transaction_rolled_back is False
    assert _checks(rollback_evidence)["postgres.rollback"] == acceptance.FAIL


def test_connection_failure_has_distinct_acceptance_exit_and_no_secret() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    def fail_connect(dsn: str) -> FakeConnection:
        assert dsn == SECRET_DSN
        raise FakeDatabaseError()

    exit_code = acceptance.main(
        ["--confirm-staging"],
        environ=_valid_environment(),
        connector=fail_connect,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == acceptance.EXIT_ACCEPTANCE_FAILED
    payload = json.loads(stdout.getvalue())
    assert payload["checks"] == [
        {
            "code": "postgres.connection",
            "status": "FAIL",
            "summary": "Database connection failed before acceptance evidence was captured.",
        }
    ]
    assert SECRET_DSN not in stdout.getvalue() + stderr.getvalue()


def test_source_surface_has_no_cli_dsn_commit_or_hosted_claim() -> None:
    source = acceptance.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()  # noqa: PTH123 - inspect module source.

    assert 'parser.add_argument("--dsn"' not in text
    assert "connection.commit(" not in text
    assert "hosted_verified" not in text
    assert "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN" in text
    assert "connection.rollback()" in text
