"""Tests for the offline, fail-closed feedback rollout preflight."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from tools import feedback_rollout_preflight as preflight


REPO = Path(__file__).resolve().parents[1]
GIT_SHA = "a" * 40


@lru_cache(maxsize=1)
def _registry_attestation() -> preflight.RegistryAttestationEvidence:
    return preflight.collect_registry_attestation(REPO)


def _sources(phase: str) -> dict[str, bytes]:
    paths = preflight.deployment_paths_for_phase(phase)
    sources, missing = preflight.load_sources(REPO, paths)
    assert not missing
    return sources


def _clean_git(phase: str) -> preflight.GitEvidence:
    sources = _sources(phase)
    paths = preflight.deployment_paths_for_phase(phase)
    return preflight.GitEvidence(
        sha=GIT_SHA,
        tracked_paths=frozenset(paths),
        dirty_paths=frozenset(),
        head_blob_sha256=tuple(
            (path, hashlib.sha256(sources[path]).hexdigest()) for path in paths
        ),
        migration_inventory=preflight.EXPECTED_MIGRATION_INVENTORY,
        registry_attestation=(
            _registry_attestation()
            if preflight.QUESTION_REGISTRY_JSON in paths
            else None
        ),
    )


def _checks(result: preflight.PreflightResult) -> dict[str, preflight.Check]:
    return {check.code: check for check in result.checks}


def _fake_git_runner(
    *,
    omit_tracked: str | None = None,
    dirty: bool = False,
) -> tuple[preflight.GitRunner, list[tuple[str, ...]]]:
    commands: list[tuple[str, ...]] = []

    def run(_repo_root: Path, args: tuple[str, ...]) -> preflight.GitCommandResult:
        commands.append(args)
        if args[0] == "rev-parse":
            return preflight.GitCommandResult(0, f"{GIT_SHA}\n".encode())
        if args[0] == "cat-file":
            _sha, path = args[2].split(":", 1)
            return preflight.GitCommandResult(0, (REPO / path).read_bytes())
        marker = args.index("--")
        paths = tuple(args[marker + 1 :])
        if args[0] == "ls-files":
            tracked = tuple(path for path in paths if path != omit_tracked)
            return preflight.GitCommandResult(
                0,
                b"\0".join(path.encode() for path in tracked) + b"\0",
            )
        if args[0] == "status":
            output = f"?? {paths[0]}\0".encode() if dirty else b""
            return preflight.GitCommandResult(0, output)
        raise AssertionError(f"unexpected Git query: {args!r}")

    return run, commands


def test_phase_inputs_are_ordered_and_cumulative() -> None:
    previous: tuple[str, ...] = ()
    for phase in preflight.PHASES:
        current = preflight.deployment_paths_for_phase(phase)
        assert current[: len(previous)] == previous
        assert len(current) == len(set(current))
        assert preflight.PREFLIGHT_TOOL in current
        previous = current

    with pytest.raises(ValueError, match="unsupported rollout phase"):
        preflight.deployment_paths_for_phase("deploy-everything")

    lockdown_paths = preflight.deployment_paths_for_phase("lockdown-report")
    assert preflight.RAW_VIEW_HARDENING_MIGRATION not in (
        preflight.deployment_paths_for_phase("ingest-cutover")
    )
    assert (
        lockdown_paths.index(preflight.CONFLICT_REPORT_MIGRATION)
        < (lockdown_paths.index(preflight.RAW_VIEW_HARDENING_MIGRATION))
        < lockdown_paths.index(preflight.QUESTION_REGISTRY_MIGRATION)
        < lockdown_paths.index(preflight.QUESTION_REGISTRY_DATA_MIGRATION)
        < lockdown_paths.index(preflight.AUTHORITATIVE_REPORT_MIGRATION)
        < lockdown_paths.index(preflight.DETAIL_REPORT_MIGRATION)
        < lockdown_paths.index(preflight.BUSINESS_SNAPSHOT_MIGRATION)
        < lockdown_paths.index(preflight.SESSION_ATTEMPT_FILTER_MIGRATION)
        < lockdown_paths.index(preflight.QUESTION_REACTION_MIGRATION)
        < lockdown_paths.index(preflight.SURPRISE_REPORT_MIGRATION)
        < lockdown_paths.index(preflight.REPORT_CLIENT)
    )
    ingest_paths = preflight.deployment_paths_for_phase("ingest-cutover")
    assert (
        ingest_paths.index(preflight.INGEST_EDGE)
        < ingest_paths.index(preflight.INSPECTOR_FEEDBACK)
        < ingest_paths.index(preflight.HOSTED_ROUNDTRIP_VERIFIER)
    )
    report_app_paths = preflight.deployment_paths_for_phase("report-app")
    assert preflight.POSTGRES_ACCEPTANCE_VERIFIER in report_app_paths
    assert preflight.DEPLOYMENT_LEDGER_TOOL in report_app_paths
    assert preflight.DEPLOYMENT_LEDGER_README in report_app_paths
    assert preflight.DEPLOYMENT_LEDGER_JOURNAL not in report_app_paths
    assert not any(
        path.startswith(preflight.DEPLOYMENT_LEDGER_EVIDENCE_PREFIX)
        for path in report_app_paths
    )
    assert preflight.POSTGRES_ACCEPTANCE_VERIFIER not in lockdown_paths
    assert preflight.DEPLOYMENT_LEDGER_TOOL not in lockdown_paths
    assert preflight.DEPLOYMENT_LEDGER_README not in lockdown_paths
    for inspector_path in (
        preflight.INSPECTOR_APP,
        preflight.INSPECTOR_OUTBOX,
        preflight.INSPECTOR_RECOVERY,
        preflight.INSPECTOR_RELEASE_MANIFEST,
        preflight.INSPECTOR_RECOMMENDER,
        preflight.INSPECTOR_SURPRISE_CATALOG,
    ):
        assert inspector_path in report_app_paths


def test_postgres_acceptance_is_an_optional_report_app_fingerprint_input() -> None:
    phase = "report-app"
    paths = preflight.deployment_paths_for_phase(phase)
    sources: dict[str, bytes | str] = _sources(phase)
    original = preflight.deployment_fingerprint(sources, paths)

    assert preflight.POSTGRES_ACCEPTANCE_VERIFIER in paths
    assert (
        'postgres-acceptance = ["psycopg[binary]>=3.1"]'
        in sources[preflight.PROJECT_METADATA].decode()
    )

    changed = dict(sources)
    changed[preflight.POSTGRES_ACCEPTANCE_VERIFIER] += b"\n# fingerprint probe\n"
    assert preflight.deployment_fingerprint(changed, paths) != original


def test_deployment_ledger_contract_sources_are_fingerprinted_but_records_are_not() -> (
    None
):
    phase = "report-app"
    paths = preflight.deployment_paths_for_phase(phase)
    sources: dict[str, bytes | str] = _sources(phase)
    original = preflight.deployment_fingerprint(sources, paths)

    for path in (
        preflight.DEPLOYMENT_LEDGER_TOOL,
        preflight.DEPLOYMENT_LEDGER_README,
    ):
        changed = dict(sources)
        value = changed[path]
        assert isinstance(value, bytes)
        changed[path] = value + b"\n"
        assert preflight.deployment_fingerprint(changed, paths) != original

    assert preflight.DEPLOYMENT_LEDGER_JOURNAL not in paths
    assert not any(
        path.startswith(preflight.DEPLOYMENT_LEDGER_EVIDENCE_PREFIX) for path in paths
    )
    with_post_deploy_outputs = dict(sources)
    with_post_deploy_outputs[preflight.DEPLOYMENT_LEDGER_JOURNAL] = b'{"record":1}\n'
    with_post_deploy_outputs[
        f"{preflight.DEPLOYMENT_LEDGER_EVIDENCE_PREFIX}deploy-1/hosted-roundtrip.json"
    ] = b'{"ok":true}\n'
    assert preflight.deployment_fingerprint(with_post_deploy_outputs, paths) == original


@pytest.mark.parametrize("phase", preflight.PHASES)
def test_current_contracts_pass_locally_but_hosted_stays_unverified(
    phase: str,
) -> None:
    result = preflight.evaluate_preflight(
        _sources(phase),
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    assert result.static_overall == preflight.PASS
    assert result.overall == preflight.UNVERIFIED
    assert result.exit_code == 0
    assert result.checked_rollout_input_sha256 is not None
    assert len(result.checked_rollout_input_sha256) == 64
    assert all(
        check.status == preflight.PASS
        for check in result.checks
        if check.code != "hosted.acceptance"
    )
    assert _checks(result)["hosted.acceptance"].status == preflight.UNVERIFIED

    envelope = result.to_dict()
    assert envelope["scope"] == "local_static"
    assert envelope["hosted_verified"] is False
    assert envelope["deploy_ready"] is False
    assert envelope["checked_rollout_input_paths"] == list(
        preflight.deployment_paths_for_phase(phase)
    )
    assert "deployment_input_paths" not in envelope


def test_require_hosted_changes_exit_policy_without_fabricating_evidence() -> None:
    phase = "report-app"
    result = preflight.evaluate_preflight(
        _sources(phase),
        phase=phase,
        git_evidence=_clean_git(phase),
        require_hosted=True,
    )

    assert result.static_overall == preflight.PASS
    assert result.overall == preflight.UNVERIFIED
    assert result.exit_code == 2
    assert _checks(result)["hosted.acceptance"].status == preflight.UNVERIFIED


@pytest.mark.parametrize(
    ("phase", "path", "before", "after", "failed_code"),
    [
        (
            "expand",
            preflight.EXPAND_MIGRATION,
            "commit;",
            "revoke insert on table public.feedback_events from service_role;\ncommit;",
            "contract.expand.direct_insert_preserved",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REACTION_MIGRATION,
            "feedback_events_question_reaction_payload_check",
            "feedback_events_unchecked_reaction_payload",
            "contract.report.question_reaction_store",
        ),
        (
            "lockdown-report",
            preflight.SURPRISE_REPORT_MIGRATION,
            "where reactions.reaction_rank = 1",
            "where reactions.reaction_rank = 2",
            "contract.report.surprise_aggregate",
        ),
        (
            "lockdown-report",
            preflight.SURPRISE_REPORT_MIGRATION,
            "p_attempt_id text default null\n)\nreturns table (\n    question_id text,",
            (
                "p_attempt_id text default null,\n"
                "    p_limit integer default 1000\n"
                ")\nreturns table (\n    question_id text,"
            ),
            "contract.report.surprise_aggregate",
        ),
        (
            "ingest-cutover",
            preflight.INGEST_EDGE,
            "/rest/v1/rpc/feedback_ingest_events",
            "/rest/v1/feedback_events",
            "contract.ingest.rpc_only",
        ),
        (
            "ingest-cutover",
            preflight.INSPECTOR_FEEDBACK,
            "MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991",
            "MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_990",
            "contract.ingest.interoperable_client",
        ),
        (
            "ingest-cutover",
            preflight.INGEST_EDGE,
            'value.timing !== "after_reveal"',
            'value.timing !== "before_reveal"',
            "contract.ingest.question_reaction_wire",
        ),
        (
            "lockdown-report",
            preflight.LOCKDOWN_MIGRATION,
            "commit;",
            "delete from public.feedback_events;\ncommit;",
            "contract.lockdown.only_writer_lockdown",
        ),
        (
            "lockdown-report",
            preflight.CONFLICT_REPORT_MIGRATION,
            "conflict_audit_event_count bigint,",
            "conflict_audit_row_count bigint,",
            "contract.report.exact_columns",
        ),
        (
            "lockdown-report",
            preflight.RAW_VIEW_HARDENING_MIGRATION,
            "count(distinct (question_id, question_version)) as question_count",
            "count(distinct question_id) as question_count",
            "contract.report.raw_views_hardened",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REGISTRY_MIGRATION,
            "grant select on table",
            "grant insert on table",
            "contract.report.authoritative_registry_schema",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REGISTRY_MIGRATION,
            "feedback_quiz_choice_release_inventory_complete",
            "feedback_quiz_choice_release_inventory_removed",
            "contract.report.authoritative_registry_schema",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REGISTRY_MIGRATION,
            "pg_catalog.pg_advisory_xact_lock",
            "pg_catalog.pg_advisory_lock_removed",
            "contract.report.authoritative_registry_schema",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REGISTRY_MIGRATION,
            "    coalesce(",
            "    pg_catalog.coalesce(",
            "contract.report.authoritative_registry_schema",
        ),
        (
            "lockdown-report",
            preflight.QUESTION_REGISTRY_JSON,
            "registry_db3f1a166af0b526e08d4eff49539c6a2150653d1940b0fcccbdbfbe0b525131",
            "registry_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "contract.report.authoritative_registry_data",
        ),
        (
            "lockdown-report",
            preflight.AUTHORITATIVE_REPORT_MIGRATION,
            "and authoritative_is_correct",
            "and payload -> 'is_correct' = 'true'::jsonb",
            "contract.report.authoritative_business_cutover",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "events.authoritative_is_correct as is_correct",
            "events.payload -> 'is_correct' as is_correct",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.BUSINESS_SNAPSHOT_MIGRATION,
            "'business_snapshot_v1'::text as snapshot_revision",
            "'business_snapshot_v0'::text as snapshot_revision",
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.HOSTED_ROUNDTRIP_VERIFIER,
            (
                "reports_client.fetch_business_snapshot(\n"
                '                filters={"question_id": snapshot_missing_question_id},'
            ),
            (
                "reports_client.fetch_page(\n"
                '                filters={"question_id": snapshot_missing_question_id},'
            ),
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.BUSINESS_SNAPSHOT_MIGRATION,
            "security invoker",
            "security definer",
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.BUSINESS_SNAPSHOT_MIGRATION,
            "cross join lateral public.feedback_report_answers(",
            "cross join lateral public.feedback_report_comments(",
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.BUSINESS_SNAPSHOT_MIGRATION,
            "4194304::bigint as snapshot_pages_bytes",
            "41943040::bigint as snapshot_pages_bytes",
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.BUSINESS_SNAPSHOT_MIGRATION,
            "grant execute on function public.feedback_report_business_snapshot(",
            "grant execute on function public.feedback_report_summary(",
            "contract.report.atomic_business_snapshot",
        ),
        (
            "lockdown-report",
            preflight.SESSION_ATTEMPT_FILTER_MIGRATION,
            "or events.session_id = p_session_id",
            "or events.session_id <> p_session_id",
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.SESSION_ATTEMPT_FILTER_MIGRATION,
            "or events.report_attempt_id = p_attempt_id",
            "or events.report_attempt_id <> p_attempt_id",
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.SESSION_ATTEMPT_FILTER_MIGRATION,
            ("    p_limit integer default 200,\n    p_session_id text default null,"),
            ("    p_session_id text default null,\n    p_limit integer default 200,"),
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.SESSION_ATTEMPT_FILTER_MIGRATION,
            "            parameters.session_id,\n",
            "",
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.SESSION_ATTEMPT_FILTER_MIGRATION,
            (
                "grant execute on function public.feedback_report_summary(\n"
                "    text, text, text, text, timestamptz, timestamptz, text, text\n"
                ") to service_role;"
            ),
            "-- summary grant removed",
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.HOSTED_ROUNDTRIP_VERIFIER,
            "session_attempt_filters_verified = True",
            "session_attempt_filters_verified = False",
            "contract.report.session_attempt_filters",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "revoke all on function public.feedback_report_answers(",
            "revoke all on function public.feedback_report_proposals(",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "grant execute on function public.feedback_report_answers(",
            "grant execute on function public.feedback_report_proposals(",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "events.authoritative_release_id as release_id",
            "events.payload #>> '{release_id}' as release_id",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "events.registry_status = 'matched'",
            "events.registry_status = 'client_claimed'",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            (
                "revoke all on function public.feedback_report_answers(\n"
                "    text, text, text, text, timestamptz, timestamptz\n"
                ") from public, anon, authenticated, service_role;"
            ),
            "-- answer detail revoke removed",
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.DETAIL_REPORT_MIGRATION,
            "        'custom_setting_rejected'\n    )",
            (
                "        'custom_setting_rejected',\n"
                "        'custom_run_completed'\n"
                "    )"
            ),
            "contract.report.authoritative_details",
        ),
        (
            "lockdown-report",
            preflight.AUTHORITATIVE_REPORT_MIGRATION,
            "'registry_v1'::text as authority_revision",
            "'registry_v0'::text as authority_revision",
            "contract.report.authoritative_business_cutover",
        ),
        (
            "lockdown-report",
            preflight.RAW_VIEW_HARDENING_MIGRATION,
            "coalesce(nullif(payload ->> 'attempt_id', ''), '')",
            "nullif(payload ->> 'attempt_id', '')",
            "contract.report.raw_views_hardened",
        ),
        (
            "lockdown-report",
            preflight.RAW_VIEW_HARDENING_MIGRATION,
            "jsonb_typeof(payload -> 'is_correct') = 'boolean'",
            "jsonb_typeof(payload -> 'is_correct') = 'string'",
            "contract.report.raw_views_hardened",
        ),
        (
            "lockdown-report",
            preflight.RAW_VIEW_HARDENING_MIGRATION,
            "between -2147483648 and 2147483647",
            "between -2147483648 and 2147483648",
            "contract.report.raw_views_hardened",
        ),
        (
            "report-app",
            preflight.REPORT_QUERY,
            '  "feedback_report_comments",',
            '  "feedback_report_comments_v2",',
            "contract.report_app.view_inventory",
        ),
        (
            "report-app",
            preflight.REPORT_QUERY,
            '  "feedback_report_surprise_quality",',
            '  "feedback_report_surprise_quality_v2",',
            "contract.report_app.view_inventory",
        ),
        (
            "report-app",
            preflight.REPORT_QUERY,
            "const codePointLength = Array.from(value).length",
            "value.length",
            "contract.report_app.read_only_edge",
        ),
        (
            "report-app",
            preflight.REPORT_QUERY,
            "p_limit: query.limit",
            "p_limit: 1",
            "contract.report_app.read_only_edge",
        ),
        (
            "report-app",
            preflight.REPORT_EDGE,
            "rawRowsJson = await response.text()",
            "rawRowsJson = JSON.stringify(await response.json())",
            "contract.report_app.read_only_edge",
        ),
        (
            "report-app",
            preflight.REPORT_APP,
            "business_snapshot = client.fetch_business_snapshot(",
            "business_snapshot = client.fetch_page(",
            "contract.report_app.strict_client_app",
        ),
        (
            "report-app",
            preflight.REPORT_APP,
            "            next_pages = {",
            (
                "            client.fetch_page(\n"
                "                SUMMARY_VIEW, filters=requested_filters,\n"
                "                limit=requested_limit, offset=0,\n"
                "            )\n"
                "            next_pages = {"
            ),
            "contract.report_app.strict_client_app",
        ),
        (
            "report-app",
            preflight.PROJECT_METADATA,
            'postgres-acceptance = ["psycopg[binary]>=3.1"]',
            'postgres-acceptance = ["psycopg>=3.1"]',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            'parser.add_argument("--confirm-staging", action="store_true")',
            'parser.add_argument("--confirm-live", action="store_true")',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            'DSN_ENV: Final = "ARCHITECTURE_IQ_POSTGRES_ACCEPTANCE_DSN"',
            'DSN_ENV: Final = "DATABASE_URL"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            'token.startswith(("live", "main", "prod"))',
            'token.startswith(("production-safe",))',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "connection.rollback()",
            "connection.commit()",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "connection.close()",
            "connection.cursor()",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.function_grants"',
            '"postgres.function_grants_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "tuple(row[8:11]) == (False, False, True)",
            "tuple(row[8:11]) == (True, True, True)",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.rls"',
            '"postgres.rls_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "observed_rls[name] == (True, True, 0)",
            "observed_rls[name] == (True, False, 0)",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.triggers"',
            '"postgres.triggers_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "feedback_quiz_choice_inventory_complete",
            "feedback_quiz_choice_inventory_removed",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.constraints"',
            '"postgres.constraints_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "feedback_quiz_questions_correct_choice_fkey",
            "feedback_quiz_questions_correct_choice_removed",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.registry_authority"',
            '"postgres.registry_authority_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            "architecture_iq_acceptance:registry_content",
            "architecture_iq_acceptance:registry_rows",
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.append_only_probes"',
            '"postgres.append_only_probes_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.POSTGRES_ACCEPTANCE_VERIFIER,
            '"postgres.registry_counterexamples"',
            '"postgres.registry_counterexamples_removed"',
            "contract.postgres.staging_acceptance",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            '"source_mapping_attested",',
            '"source_mapping_unverified",',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'document.get("mapping_authority")\n'
            '        != "reviewed_provider_control_plane_capture"',
            'document.get("mapping_authority")\n        != "operator_claim"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'return "ACTIVATED_REVIEWED"',
            'return "ACTIVE"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'return "READY_FOR_REVIEWED_ACTIVATION"',
            'return "PROVIDER_VERIFIED_READY"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            '        "backend_project_id",\n',
            '        "backend_project_label",\n',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'identifier = f"deployment_context_{hashlib.sha256(_canonical_json_bytes(binding)).hexdigest()}"',
            'identifier = f"deployment_context_{hashlib.sha256(b"unbound").hexdigest()}"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'frozenset({"path", "sha256", "media_type"})',
            'frozenset({"path", "media_type"})',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            "if hashlib.sha256(raw).hexdigest() != expected_hash:",
            "if False and hashlib.sha256(raw).hexdigest() != expected_hash:",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            "_rollout_fingerprint_from_commit(repo_root, commit, preflight_paths)",
            "fingerprint",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'if event_type in reviewed_events and reviewed_by == record.get("recorded_by"):',
            'if event_type in reviewed_events and reviewed_by != record.get("recorded_by"):',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            '"reviewed decision requires a distinct reviewer"',
            '"reviewer may equal recorder"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            '"provider export SHA-256 is reused by another deployment"',
            '"provider export may be reused"',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            '_git_blob(repo_root, commit, entrypoint, label="candidate.source.entrypoint")',
            "Path(entrypoint).read_bytes()",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            "env=_git_environment()",
            "env=os.environ.copy()",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'if document["previous_record_sha256"] != previous_hash:',
            'if False and document["previous_record_sha256"] != previous_hash:',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            "    if not confirm:\n",
            "    if confirm:\n",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_TOOL,
            'raise ConfirmationRequired("append requires --confirm-append")',
            'raise ConfirmationRequired("append is optional")',
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_README,
            "successful operational state is named `ACTIVATED_REVIEWED`",
            "successful operational state is provider-verified",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.DEPLOYMENT_LEDGER_README,
            "external head pin needed to",
            "hash chain alone detects every possible truncation",
            "contract.deployment_ledger",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            '["git", "rev-parse", "--verify", "HEAD^{commit}"]',
            '["git", "rev-parse", "HEAD"]',
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            '    "SOURCE_VERSION",\n',
            '    "UNTRUSTED_GIT_SHA",\n',
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            '"GIT_NO_REPLACE_OBJECTS": "1"',
            '"GIT_NO_REPLACE_OBJECTS": "0"',
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            "env=_checkout_git_environment()",
            'env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}',
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            "if checkout_sha is None or GIT_SHA_PATTERN.fullmatch(checkout_sha) is None:",
            "if checkout_sha is not None and GIT_SHA_PATTERN.fullmatch(checkout_sha) is None:",
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            "if len(candidates) != 1:",
            "if not candidates:",
            "contract.inspector.runtime_git_identity",
        ),
        (
            "report-app",
            preflight.INSPECTOR_APP,
            '"Upload pending session events"',
            '"Upload every event without receipts"',
            "contract.inspector.feedback_ui",
        ),
    ],
)
def test_key_contract_tampering_fails_closed(
    phase: str,
    path: str,
    before: str,
    after: str,
    failed_code: str,
) -> None:
    sources: dict[str, bytes | str] = _sources(phase)
    original = sources[path].decode()
    assert before in original
    sources[path] = original.replace(before, after, 1)

    result = preflight.evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    assert result.static_overall == preflight.FAIL
    assert result.overall == preflight.FAIL
    assert result.exit_code == 1
    assert _checks(result)[failed_code].status == preflight.FAIL
    assert _checks(result)["hosted.acceptance"].status == preflight.UNVERIFIED


def test_registry_json_and_sql_must_match_byte_for_byte_exporter_render() -> None:
    phase = "lockdown-report"
    sources: dict[str, bytes | str] = _sources(phase)
    registry = json.loads(sources[preflight.QUESTION_REGISTRY_JSON])
    original_registry_id = registry["registry_id"]
    question = registry["questions"][0]
    replacement_letter = next(
        letter
        for letter in sorted(question["choices"])
        if letter != question["correct_letter"]
    )
    question["correct_letter"] = replacement_letter
    question["correct_candidate_id"] = question["choices"][replacement_letter]
    identity_core = {
        key: registry[key]
        for key in (
            "schema_version",
            "release_id",
            "question_count",
            "choice_count",
            "questions",
        )
    }
    registry["registry_id"] = (
        "registry_"
        + hashlib.sha256(
            json.dumps(
                identity_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    sources[preflight.QUESTION_REGISTRY_JSON] = (
        json.dumps(
            registry,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    registry_sql = sources[preflight.QUESTION_REGISTRY_DATA_MIGRATION].decode()
    sources[preflight.QUESTION_REGISTRY_DATA_MIGRATION] = registry_sql.replace(
        original_registry_id,
        registry["registry_id"],
    )

    result = preflight.evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    assert _checks(result)["contract.report.authoritative_registry_data"].status == (
        preflight.FAIL
    )


def test_registry_pair_must_match_fresh_full_bundle_attestation() -> None:
    phase = "lockdown-report"
    sources: dict[str, bytes | str] = _sources(phase)
    registry = json.loads(sources[preflight.QUESTION_REGISTRY_JSON])
    question = registry["questions"][0]
    replacement_letter = next(
        letter
        for letter in sorted(question["choices"])
        if letter != question["correct_letter"]
    )
    question["correct_letter"] = replacement_letter
    question["correct_candidate_id"] = question["choices"][replacement_letter]
    identity_core = {
        key: registry[key]
        for key in (
            "schema_version",
            "release_id",
            "question_count",
            "choice_count",
            "questions",
        )
    }
    registry["registry_id"] = (
        "registry_"
        + hashlib.sha256(
            json.dumps(
                identity_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    sources[preflight.QUESTION_REGISTRY_JSON] = preflight.serialize_feedback_registry(
        registry
    )
    sources[preflight.QUESTION_REGISTRY_DATA_MIGRATION] = (
        preflight.render_feedback_registry_sql(registry)
    )

    result = preflight.evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    assert _checks(result)["contract.report.authoritative_registry_data"].status == (
        preflight.FAIL
    )


def test_report_columns_are_extracted_dynamically_and_match_strict_client() -> None:
    sources = _sources("lockdown-report")
    sql_columns = preflight.extract_sql_return_columns(
        sources[preflight.CONFLICT_REPORT_MIGRATION].decode(),
        "feedback_report_ingestion_summary",
    )
    client_columns = preflight.extract_python_string_constant(
        sources[preflight.REPORT_CLIENT].decode(),
        "_INGESTION_SUMMARY_COLUMNS",
    )

    assert sql_columns == client_columns
    assert sql_columns is not None
    assert "conflict_audit_event_count" in sql_columns

    detail_sql = sources[preflight.DETAIL_REPORT_MIGRATION].decode()
    assert preflight.extract_sql_return_columns(
        detail_sql,
        "feedback_report_answers",
    ) == preflight.extract_python_string_constant(
        sources[preflight.REPORT_CLIENT].decode(),
        "_ANSWER_COLUMNS",
    )
    assert preflight.extract_sql_return_columns(
        detail_sql,
        "feedback_report_proposals",
    ) == preflight.extract_python_string_constant(
        sources[preflight.REPORT_CLIENT].decode(),
        "_PROPOSAL_COLUMNS",
    )


def test_raw_view_columns_preserve_baseline_prefixes_and_append_quality() -> None:
    sources = _sources("lockdown-report")
    baseline = sources[preflight.BASELINE_EVENT_MIGRATION].decode()
    hardened = sources[preflight.RAW_VIEW_HARDENING_MIGRATION].decode()

    baseline_session = preflight.extract_sql_view_columns(
        baseline, "feedback_session_summary"
    )
    baseline_question = preflight.extract_sql_view_columns(
        baseline, "feedback_question_stats"
    )
    baseline_proposals = preflight.extract_sql_view_columns(
        baseline, "feedback_proposals"
    )
    hardened_session = preflight.extract_sql_view_columns(
        hardened, "feedback_session_summary"
    )
    hardened_question = preflight.extract_sql_view_columns(
        hardened, "feedback_question_stats"
    )
    hardened_proposals = preflight.extract_sql_view_columns(
        hardened, "feedback_proposals"
    )

    assert hardened_session == preflight._RAW_SESSION_VIEW_COLUMNS
    assert hardened_question == preflight._RAW_QUESTION_VIEW_COLUMNS
    assert hardened_proposals == preflight._RAW_PROPOSAL_VIEW_COLUMNS
    assert baseline_session is not None
    assert baseline_question is not None
    assert hardened_session[: len(baseline_session)] == baseline_session
    assert hardened_question[: len(baseline_question)] == baseline_question
    assert hardened_proposals == baseline_proposals


def test_ordered_column_extractor_rejects_unordered_sets_by_default() -> None:
    source = '_COLUMNS = frozenset({"b", "a"})\n'
    assert preflight.extract_python_string_constant(source, "_COLUMNS") is None
    assert frozenset(
        preflight.extract_python_string_constant(
            source,
            "_COLUMNS",
            allow_unordered=True,
        )
        or ()
    ) == frozenset({"a", "b"})


def test_sql_column_extractor_rejects_comment_and_later_function_shadowing() -> None:
    shadowed = """
create function public.feedback_report_ingestion_summary()
returns record
-- returns table (trusted_column bigint)
language sql
as $function$ select null; $function$;

create function public.feedback_report_ingestion_summary(p_other text)
returns table (trusted_column bigint)
language sql
as $function$ select 1::bigint; $function$;
"""
    assert (
        preflight.extract_sql_return_columns(
            shadowed,
            "feedback_report_ingestion_summary",
        )
        is None
    )


def test_sql_view_extractor_rejects_spoofs_duplicates_and_missing_aliases() -> None:
    valid = """
-- create or replace view public.feedback_probe as
-- select forged as trusted from public.feedback_events;
create view public.feedback_probe
with (security_invoker = true, security_barrier = true)
as
select event_id, count(*) as event_count
from public.feedback_events;
"""
    assert preflight.extract_sql_view_columns(valid, "feedback_probe") == (
        "event_id",
        "event_count",
    )
    assert (
        preflight.extract_sql_view_columns(f"{valid}\n{valid}", "feedback_probe")
        is None
    )
    assert (
        preflight.extract_sql_view_columns(
            valid.replace("count(*) as event_count", "session_id as event_id"),
            "feedback_probe",
        )
        is None
    )
    assert (
        preflight.extract_sql_view_columns(
            valid.replace("count(*) as event_count", "count(*)"),
            "feedback_probe",
        )
        is None
    )


def test_first_call_position_is_scoped_to_the_named_function() -> None:
    source = """
def legacy(client):
    client.post_event("legacy")

def authoritative(reports, client):
    reports.fetch_business_snapshot()
    client.post_event("authoritative")
"""

    snapshot = preflight._function_first_call_position(
        source,
        "authoritative",
        "fetch_business_snapshot",
    )
    post = preflight._function_first_call_position(
        source,
        "authoritative",
        "post_event",
    )

    assert snapshot is not None
    assert post is not None
    assert snapshot < post
    assert (
        preflight._function_first_call_position(
            source,
            "missing",
            "fetch_business_snapshot",
        )
        is None
    )


def test_migration_inventory_rejects_unclassified_sql_between_phases() -> None:
    phase = "expand"
    clean = _clean_git(phase)
    injected = "20260712012300_unclassified_writer_change.sql"
    evidence = replace(
        clean,
        migration_inventory=tuple(
            sorted((*preflight.EXPECTED_MIGRATION_INVENTORY, injected))
        ),
    )
    result = preflight.evaluate_preflight(
        _sources(phase),
        phase=phase,
        git_evidence=evidence,
    )

    assert _checks(result)["migrations.inventory_order"].status == preflight.FAIL
    assert result.exit_code == 1


def test_fingerprint_covers_exact_ordered_path_and_content_bytes() -> None:
    phase = "expand"
    paths = preflight.deployment_paths_for_phase(phase)
    sources: dict[str, bytes | str] = _sources(phase)
    original = preflight.deployment_fingerprint(sources, paths)

    sources[paths[0]] = sources[paths[0]] + b"\n-- harmless fingerprint change\n"
    changed = preflight.deployment_fingerprint(sources, paths)

    assert original is not None
    assert changed is not None
    assert changed != original
    assert preflight.deployment_fingerprint({}, paths) is None


def test_report_runtime_metadata_is_inside_the_checked_fingerprint() -> None:
    phase = "report-app"
    paths = preflight.deployment_paths_for_phase(phase)
    sources: dict[str, bytes | str] = _sources(phase)
    original = preflight.deployment_fingerprint(sources, paths)

    for path in (
        preflight.REQUIREMENTS,
        preflight.PROJECT_METADATA,
        preflight.PROJECT_README,
    ):
        changed_sources = dict(sources)
        value = changed_sources[path]
        assert isinstance(value, bytes)
        changed_sources[path] = value + b"\n"
        assert preflight.deployment_fingerprint(changed_sources, paths) != original


def test_fake_git_results_fail_untracked_and_dirty_inputs() -> None:
    phase = "ingest-cutover"
    paths = preflight.deployment_paths_for_phase(phase)
    runner, commands = _fake_git_runner(omit_tracked=paths[-1], dirty=True)
    evidence = preflight.collect_git_evidence(REPO, paths, runner=runner)
    result = preflight.evaluate_preflight(
        _sources(phase),
        phase=phase,
        git_evidence=evidence,
    )

    assert _checks(result)["git.inputs_tracked"].status == preflight.FAIL
    assert _checks(result)["git.inputs_clean"].status == preflight.FAIL
    assert result.exit_code == 1
    command_names = tuple(command[0] for command in commands)
    assert command_names[:2] == ("rev-parse", "ls-files")
    assert command_names[-1] == "status"
    assert command_names.count("cat-file") == len(paths)


def test_commit_blob_comparison_catches_status_hidden_byte_drift() -> None:
    phase = "expand"
    sources: dict[str, bytes | str] = _sources(phase)
    sources[preflight.EXPAND_MIGRATION] += b"\n-- status-hidden drift\n"
    result = preflight.evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    by_code = _checks(result)
    assert by_code["git.inputs_clean"].status == preflight.PASS
    assert by_code["git.inputs_match_head"].status == preflight.FAIL
    assert result.exit_code == 1


def test_source_loader_rejects_symlinks_and_non_regular_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "target.sql").write_text("select 1;", encoding="utf-8")
    (tmp_path / "linked.sql").symlink_to(tmp_path / "target.sql")
    (tmp_path / "directory.sql").mkdir()

    sources, missing = preflight.load_sources(
        tmp_path,
        ("linked.sql", "directory.sql"),
    )

    assert sources == {}
    assert missing == frozenset({"linked.sql", "directory.sql"})


@pytest.mark.parametrize(
    "failed_query",
    ["sha", "tracked", "clean", "head_blobs", "migration_inventory"],
)
def test_unknown_git_evidence_fails_closed(failed_query: str) -> None:
    phase = "expand"
    evidence = preflight.GitEvidence(
        sha=None if failed_query == "sha" else GIT_SHA,
        tracked_paths=frozenset(preflight.deployment_paths_for_phase(phase)),
        dirty_paths=frozenset(),
        head_blob_sha256=tuple(
            (
                path,
                hashlib.sha256(_sources(phase)[path]).hexdigest(),
            )
            for path in preflight.deployment_paths_for_phase(phase)
        ),
        migration_inventory=preflight.EXPECTED_MIGRATION_INVENTORY,
        failed_queries=frozenset({failed_query}),
    )
    result = preflight.evaluate_preflight(
        _sources(phase),
        phase=phase,
        git_evidence=evidence,
    )

    assert result.static_overall == preflight.FAIL
    assert result.exit_code == 1


def test_json_and_human_output_never_include_source_content_or_claim_readiness() -> (
    None
):
    phase = "expand"
    secret = "TOP-SECRET-ROLLBACK-TOKEN"
    sources: dict[str, bytes | str] = _sources(phase)
    sources[preflight.EXPAND_MIGRATION] += f"\n-- {secret}\n".encode()
    result = preflight.evaluate_preflight(
        sources,
        phase=phase,
        git_evidence=_clean_git(phase),
    )

    rendered_json = preflight.render_json(result)
    parsed = json.loads(rendered_json)
    rendered_text = preflight.render_text(result)
    assert secret not in rendered_json
    assert secret not in rendered_text
    assert parsed["deploy_ready"] is False
    assert parsed["hosted_verified"] is False
    assert "HOSTED UNVERIFIED — NOT DEPLOY-READY" in rendered_text


def test_cli_surface_has_no_endpoint_or_credential_options() -> None:
    parser = preflight.build_parser()
    option_strings = {
        option
        for action in parser._actions  # noqa: SLF001 - verifying the public CLI.
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--phase",
        "--json",
        "--require-hosted",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "expand", "--url", "https://example.test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "expand", "--token", "do-not-accept"])


def test_module_imports_no_network_or_deployment_library() -> None:
    tree = ast.parse((REPO / "tools/feedback_rollout_preflight.py").read_text())
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(
        {"http", "requests", "socket", "supabase", "urllib"}
    )


def test_default_git_runner_disables_side_effects_lazy_fetch_and_secret_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = b""

    def fake_run(command: tuple[str, ...], **kwargs: object) -> _Completed:
        captured["command"] = command
        captured.update(kwargs)
        return _Completed()

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "database-secret")
    monkeypatch.setenv("FEEDBACK_INGEST_TOKEN", "ingest-secret")
    monkeypatch.setenv("GIT_DIR", "/tmp/redirected-secret-repo")
    monkeypatch.setenv("GIT_WORK_TREE", "/tmp/redirected-work-tree")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/redirected-object-store")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/hidden-replacements/")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    preflight._default_git_runner(REPO, ("status", "--porcelain=v1"))

    command = captured["command"]
    environment = captured["env"]
    assert isinstance(command, tuple)
    assert isinstance(environment, dict)
    assert command[:5] == (
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={preflight.os.devnull}",
    )
    assert environment == preflight._git_subprocess_environment()
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    serialized = repr(captured)
    assert "database-secret" not in serialized
    assert "ingest-secret" not in serialized
    assert "redirected-secret-repo" not in serialized
    assert "redirected-work-tree" not in serialized
    assert "redirected-object-store" not in serialized
    assert "hidden-replacements" not in serialized


def test_main_exit_codes_and_git_queries_are_read_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, commands = _fake_git_runner()
    assert (
        preflight.main(
            ["--phase", "report-app", "--json"],
            repo_root=REPO,
            git_runner=runner,
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["static_overall"] == preflight.PASS
    assert output["overall"] == preflight.UNVERIFIED
    assert output["deploy_ready"] is False

    assert (
        preflight.main(
            ["--phase", "report-app", "--json", "--require-hosted"],
            repo_root=REPO,
            git_runner=runner,
        )
        == 2
    )
    capsys.readouterr()

    dirty_runner, _dirty_commands = _fake_git_runner(dirty=True)
    assert (
        preflight.main(
            ["--phase", "expand", "--json"],
            repo_root=REPO,
            git_runner=dirty_runner,
        )
        == 1
    )
    capsys.readouterr()

    assert commands
    assert {command[0] for command in commands} == {
        "cat-file",
        "rev-parse",
        "ls-files",
        "status",
    }
    assert not any(
        forbidden in command
        for command in commands
        for forbidden in {"add", "apply", "commit", "deploy", "push", "write"}
    )
