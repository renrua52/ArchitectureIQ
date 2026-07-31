"""Focused tests for the retrospective deployment evidence ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import uuid

import pytest

from tools import deployment_ledger as ledger


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rollout_fingerprint(path: str, content: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"ArchitectureIQ feedback rollout inputs v1\0")
    path_bytes = path.encode("utf-8")
    digest.update(len(path_bytes).to_bytes(8, "big"))
    digest.update(path_bytes)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _uuid(number: int) -> str:
    return str(uuid.UUID(int=number))


def _base_event(
    event_type: str,
    deployment_key: str,
    facts: dict[str, Any],
    *,
    reviewed_by: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "architecture_iq_deployment_event",
        "event_type": event_type,
        "deployment_key": deployment_key,
        "recorded_at": "2026-07-12T12:00:00Z",
        "recorded_by": "github:test-maintainer",
        "reviewed_by": reviewed_by,
        "facts": facts,
    }


def _context_summary(
    *,
    deployment_key: str,
    release_id: str,
    manifest_sha256: str,
    registry_id: str,
    source_commit: str,
    declaration: dict[str, Any],
) -> dict[str, str]:
    binding = {
        "deployment_key": deployment_key,
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
        "registry_id": registry_id,
        "source_commit": source_commit,
        **declaration,
    }
    context_id = f"deployment_context_{hashlib.sha256(_canonical(binding)).hexdigest()}"
    return {
        "deployment_context_id": context_id,
        "environment": declaration["environment"],
        "target_label": declaration["target_label"],
        "provider": declaration["provider"],
        "project_id": declaration["project_id"],
        "deploy_id": declaration["deploy_id"],
        "site_url": declaration["site_url"],
        "backend_project_id": declaration["backend_project_id"],
        "ingest_origin_sha256": declaration["ingest_origin_sha256"],
        "report_origin_sha256": declaration["report_origin_sha256"],
    }


def _append(
    repo: Path,
    draft_path: Path,
    event: dict[str, Any],
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    _write_json(draft_path, event)
    result = ledger.append_event(
        repo_root=repo,
        ledger_path=ledger_path or Path("deployments/ledger.jsonl"),
        event_json_path=draft_path,
        confirm=True,
    )
    return dict(result)


@pytest.fixture
def evidence_repo(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Ledger Test")

    question = {
        "question_id": "q_test",
        "question_version": f"qv1_{'1' * 64}",
        "family": "test_family",
        "dataset_id": "dataset_test",
        "question_type": "mixed",
        "correct_letter": "A",
        "correct_candidate_id": "c_a",
        "choices": {"A": "c_a", "B": "c_b"},
    }
    manifest_core = {
        "schema_version": "1.0",
        "source_runs": [],
        "questions": [{"question_id": "q_test"}],
        "counts": {"questions": 1, "artifact_files": 0},
        "artifacts": [],
    }
    release_id = f"release_{hashlib.sha256(_canonical(manifest_core)).hexdigest()}"
    manifest = {
        **manifest_core,
        "release_id": release_id,
        "generated_at": "2026-07-12T11:00:00Z",
    }
    manifest_path = repo / "bundle" / "quiz_manifest.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    registry_core = {
        "schema_version": "1.0",
        "release_id": release_id,
        "question_count": 1,
        "choice_count": 2,
        "questions": [question],
    }
    registry_id = f"registry_{hashlib.sha256(_canonical(registry_core)).hexdigest()}"
    registry = {
        **registry_core,
        "registry_id": registry_id,
        "manifest_sha256": manifest_sha256,
    }
    registry_path = repo / "registries" / f"{release_id}.json"
    registry_sha256 = _write_json(registry_path, registry)
    app_path = repo / "app.py"
    app_content = b"print('ArchitectureIQ')\n"
    app_path.write_bytes(app_content)
    _git(
        repo,
        "add",
        "app.py",
        "bundle/quiz_manifest.json",
        f"registries/{release_id}.json",
    )
    _git(repo, "commit", "-qm", "source candidate")
    source_commit = _git(repo, "rev-parse", "HEAD")

    fingerprint = _rollout_fingerprint("app.py", app_content)
    preflight = {
        "schema_version": "1.0",
        "scope": "local_static",
        "rollout_contract": "staged_upgrade",
        "fingerprint_scope": "enumerated_repository_rollout_inputs",
        "baseline_migrations_are_compatibility_inputs": True,
        "phase": "report-app",
        "static_overall": "PASS",
        "overall": "UNVERIFIED",
        "hosted_verified": False,
        "deploy_ready": False,
        "require_hosted": False,
        "git_sha": source_commit,
        "checked_rollout_input_paths": ["app.py"],
        "checked_rollout_input_sha256": fingerprint,
        "checks": [
            {"code": "git.inputs_tracked", "status": "PASS", "summary": "tracked"},
            {"code": "git.inputs_clean", "status": "PASS", "summary": "clean"},
            {
                "code": "git.inputs_match_head",
                "status": "PASS",
                "summary": "matches",
            },
            {
                "code": "hosted.acceptance",
                "status": "UNVERIFIED",
                "summary": "not contacted",
            },
        ],
    }
    preflight_path = repo / "evidence" / "preflight.json"
    preflight_sha256 = _write_json(preflight_path, preflight)

    deployment_key = "quiz-staging-001"
    declaration = {
        "environment": "staging",
        "target_label": "architecture-iq-staging",
        "provider": "streamlit-community-cloud",
        "project_id": "architecture-iq",
        "deploy_id": "deploy-001",
        "site_url": "https://architecture-iq-staging.streamlit.app/",
        "backend_project_id": "supabase-staging",
        "ingest_origin_sha256": "3" * 64,
        "report_origin_sha256": "4" * 64,
    }
    candidate_facts = {
        "release_id": release_id,
        "manifest": {
            "path": "bundle/quiz_manifest.json",
            "sha256": manifest_sha256,
            "question_count": 1,
        },
        "registry": {
            "path": f"registries/{release_id}.json",
            "sha256": registry_sha256,
            "registry_id": registry_id,
            "manifest_sha256": manifest_sha256,
            "question_count": 1,
            "choice_count": 2,
        },
        "source": {
            "repo_url": "https://github.com/example/ArchitectureIQ.git",
            "branch": "main",
            "commit": source_commit,
            "entrypoint": "app.py",
        },
        "rollout": {
            "phase": "report-app",
            "fingerprint": fingerprint,
            "preflight": {
                "path": "evidence/preflight.json",
                "sha256": preflight_sha256,
            },
        },
    }
    context_summary = _context_summary(
        deployment_key=deployment_key,
        release_id=release_id,
        manifest_sha256=manifest_sha256,
        registry_id=registry_id,
        source_commit=source_commit,
        declaration=declaration,
    )

    postgres = {
        "schema_version": "1.0",
        "evidence_type": "architecture_iq_postgres_staging_acceptance",
        "accepted": True,
        "target_label": declaration["target_label"],
        "database_contacted": True,
        "transaction_rolled_back": True,
        "server": {
            "observed_at": "2026-07-12T11:30:00.000000Z",
            "database": "postgres",
            "role": "acceptance_owner",
            "server_version_num": 150002,
            "in_recovery": False,
        },
        "registry": {
            "release_id": release_id,
            "registry_id": registry_id,
            "question_count": 1,
            "choice_count": 2,
            "registered_release_count": 1,
            "registered_question_count": 1,
            "registered_choice_count": 2,
            "authority_revision": "registry_v1",
            "detail_revision": "detail_v1",
        },
        "checks": [
            {"code": "postgres.identity", "status": "PASS", "summary": "captured"},
            {"code": "postgres.registry", "status": "PASS", "summary": "matched"},
        ],
        "summary": {"pass": 2, "fail": 0},
    }
    postgres_path = repo / "evidence" / "postgres.json"
    postgres_sha256 = _write_json(postgres_path, postgres)
    postgres_facts = {
        "evidence": {"path": "evidence/postgres.json", "sha256": postgres_sha256},
        "summary": {
            **context_summary,
            "observed_at": postgres["server"]["observed_at"],
            "database": "postgres",
            "server_version_num": 150002,
            "release_id": release_id,
            "registry_id": registry_id,
            "question_count": 1,
            "choice_count": 2,
            "authority_revision": "registry_v1",
            "detail_revision": "detail_v1",
        },
    }

    roundtrip = {
        "schema_version": "1.0",
        "evidence_type": "architecture_iq_hosted_feedback_roundtrip",
        "verified_at": "2026-07-12T11:40:00Z",
        "manifest_sha256": manifest_sha256,
        "registry_question_count": 1,
        "registry_choice_count": 2,
        "authority_mode": "authoritative",
        "ok": True,
        "run_id": "ledger-test",
        "release_id": release_id,
        "question_id": "q_test",
        "event_id": "evt_e2e_ledger-test",
        "request_id": _uuid(2),
        "conflict_request_id": _uuid(3),
        "conflict_verified": True,
        "mixed_batch_request_id": None,
        "mixed_batch_verified": False,
        "successful_batch_first_request_id": _uuid(4),
        "successful_batch_replay_request_id": _uuid(5),
        "successful_batch_verified": True,
        "successful_batch_first_write_verified": True,
        "registry_id": registry_id,
        "authority_status_verified": True,
        "detail_reports_verified": True,
        "business_snapshot_verified": True,
        "session_attempt_filters_verified": True,
        "polls": 1,
        "receipt": {
            "accepted": 1,
            "duplicate": 0,
            "conflict": 0,
            "rejected": 0,
            "request_id": _uuid(2),
        },
    }
    roundtrip_path = repo / "evidence" / "roundtrip.json"
    roundtrip_sha256 = _write_json(roundtrip_path, roundtrip)
    roundtrip_facts = {
        "evidence": {"path": "evidence/roundtrip.json", "sha256": roundtrip_sha256},
        "summary": {
            **context_summary,
            "verified_at": roundtrip["verified_at"],
            "run_id": roundtrip["run_id"],
            "release_id": release_id,
            "registry_id": registry_id,
            "request_id": roundtrip["request_id"],
            "conflict_request_id": roundtrip["conflict_request_id"],
            "successful_batch_first_request_id": roundtrip[
                "successful_batch_first_request_id"
            ],
            "successful_batch_replay_request_id": roundtrip[
                "successful_batch_replay_request_id"
            ],
            "authority_mode": "authoritative",
        },
    }

    provider_export = {
        "captured_from": "provider deployment history",
        "provider": declaration["provider"],
        "project_id": declaration["project_id"],
        "deploy_id": declaration["deploy_id"],
        "source_commit": source_commit,
        "entrypoint": "app.py",
    }
    provider_export_path = repo / "evidence" / "provider-export.json"
    provider_export_sha256 = _write_json(provider_export_path, provider_export)
    mapping = {
        "schema_version": "1.0",
        "evidence_type": "architecture_iq_provider_deployment_mapping",
        "captured_at": "2026-07-12T11:50:00Z",
        "mapping_authority": "reviewed_provider_control_plane_capture",
        "provider_export": {
            "path": "evidence/provider-export.json",
            "sha256": provider_export_sha256,
            "media_type": "application/json",
        },
        "provider": declaration["provider"],
        "project_id": declaration["project_id"],
        "deploy_id": declaration["deploy_id"],
        "site_url": declaration["site_url"],
        "environment": declaration["environment"],
        "target_label": declaration["target_label"],
        "backend_project_id": declaration["backend_project_id"],
        "ingest_origin_sha256": declaration["ingest_origin_sha256"],
        "report_origin_sha256": declaration["report_origin_sha256"],
        "deployment_context_id": context_summary["deployment_context_id"],
        "deployed_at": "2026-07-12T11:35:00Z",
        "deployment_status": "ready",
        "repo_url": candidate_facts["source"]["repo_url"],
        "branch": "main",
        "source_commit": source_commit,
        "entrypoint": "app.py",
        "release_id": release_id,
        "manifest_sha256": manifest_sha256,
        "registry_id": registry_id,
        "rollout_input_fingerprint": fingerprint,
    }
    mapping_path = repo / "evidence" / "provider.json"
    mapping_sha256 = _write_json(mapping_path, mapping)
    mapping_facts = {
        "evidence": {"path": "evidence/provider.json", "sha256": mapping_sha256},
        "summary": {
            **context_summary,
            "captured_at": mapping["captured_at"],
            "deployed_at": mapping["deployed_at"],
            "deployment_status": "ready",
            "mapping_authority": "reviewed_provider_control_plane_capture",
            "provider_export_path": "evidence/provider-export.json",
            "provider_export_sha256": provider_export_sha256,
            "provider_export_media_type": "application/json",
            "source_commit": source_commit,
            "release_id": release_id,
        },
    }
    return {
        "repo": repo,
        "draft": repo / "draft.json",
        "ledger": Path("deployments/ledger.jsonl"),
        "key": deployment_key,
        "release_id": release_id,
        "candidate": candidate_facts,
        "declaration": declaration,
        "postgres": postgres,
        "postgres_facts": postgres_facts,
        "roundtrip": roundtrip,
        "roundtrip_facts": roundtrip_facts,
        "mapping": mapping,
        "mapping_facts": mapping_facts,
        "context_summary": context_summary,
        "provider_export_sha256": provider_export_sha256,
    }


def _candidate_event(context: dict[str, Any], key: str | None = None) -> dict[str, Any]:
    return _base_event(
        "candidate_attested", key or context["key"], context["candidate"]
    )


def _declared_event(
    context: dict[str, Any], key: str | None = None, *, deploy_id: str | None = None
) -> dict[str, Any]:
    facts = dict(context["declaration"])
    if deploy_id is not None:
        facts["deploy_id"] = deploy_id
    return _base_event("deployment_declared", key or context["key"], facts)


def _append_candidate_and_declaration(context: dict[str, Any]) -> None:
    _append(context["repo"], context["draft"], _candidate_event(context))
    _append(context["repo"], context["draft"], _declared_event(context))


def _append_complete_evidence(context: dict[str, Any]) -> None:
    for event_type, facts in (
        ("source_mapping_attested", context["mapping_facts"]),
        ("postgres_accepted", context["postgres_facts"]),
        ("roundtrip_accepted", context["roundtrip_facts"]),
    ):
        _append(
            context["repo"],
            context["draft"],
            _base_event(
                event_type,
                context["key"],
                facts,
                reviewed_by=(
                    "github:mapping-reviewer"
                    if event_type == "source_mapping_attested"
                    else None
                ),
            ),
        )


def test_full_chain_accepts_evidence_in_any_order_and_activates(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    _append_complete_evidence(context)
    _append(
        context["repo"],
        context["draft"],
        _base_event("activated", context["key"], {}, reviewed_by="github:reviewer"),
    )

    snapshot = ledger.verify_ledger(
        repo_root=context["repo"], ledger_path=context["ledger"]
    )

    assert len(snapshot.records) == 6
    assert snapshot.deployments[0].status == "ACTIVATED_REVIEWED"
    assert set(snapshot.deployments[0].evidence) == ledger.EVIDENCE_EVENTS


@pytest.mark.parametrize("mutation", ["hash", "reorder", "truncate_middle"])
def test_hash_chain_rejects_tamper_reorder_and_middle_removal(
    evidence_repo: dict[str, Any], mutation: str
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    _append(
        context["repo"],
        context["draft"],
        _base_event("postgres_accepted", context["key"], context["postgres_facts"]),
    )
    path = context["repo"] / context["ledger"]
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "hash":
        record = json.loads(lines[0])
        record["recorded_by"] = "github:tampered"
        lines[0] = _canonical(record) + b"\n"
    elif mutation == "reorder":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        del lines[1]
    path.write_bytes(b"".join(lines))

    with pytest.raises(ledger.LedgerFormatError):
        ledger.verify_ledger(repo_root=context["repo"], ledger_path=context["ledger"])


def test_duplicate_json_keys_and_noncanonical_lines_are_rejected(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    path = context["repo"] / context["ledger"]
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}\n', encoding="utf-8"
    )
    with pytest.raises(ledger.LedgerFormatError, match="duplicate"):
        ledger.verify_ledger(repo_root=context["repo"], ledger_path=context["ledger"])

    event = _candidate_event(context)
    event["previous_record_sha256"] = None
    event["record_sha256"] = ledger._record_hash(event)
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerFormatError, match="canonical"):
        ledger.verify_ledger(repo_root=context["repo"], ledger_path=context["ledger"])


def test_git_inspection_ignores_external_redirection_and_replace_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class Completed:
        returncode = 0
        stdout = b"commit-bytes"

    def fake_run(command: list[str], **kwargs: Any) -> Completed:
        captured["command"] = command
        captured.update(kwargs)
        return Completed()

    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_REPLACE_REF_BASE",
    ):
        monkeypatch.setenv(name, f"/redirected/{name.lower()}")
    monkeypatch.setattr(ledger.subprocess, "run", fake_run)

    assert ledger._git(tmp_path, ("rev-parse", "HEAD")) == b"commit-bytes"
    assert captured["command"] == [
        "git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "HEAD",
    ]
    assert captured["env"] == {
        "PATH": ledger.os.defpath,
        "HOME": ledger.os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": ledger.os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": ledger.os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def test_state_machine_rejects_repeated_and_out_of_order_events(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append(context["repo"], context["draft"], _candidate_event(context))
    with pytest.raises(ledger.StateTransitionError, match="already has"):
        _append(context["repo"], context["draft"], _candidate_event(context))
    with pytest.raises(ledger.StateTransitionError, match="before deployment_declared"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("postgres_accepted", context["key"], context["postgres_facts"]),
        )
    _append(context["repo"], context["draft"], _declared_event(context))
    event = _base_event("postgres_accepted", context["key"], context["postgres_facts"])
    _append(context["repo"], context["draft"], event)
    with pytest.raises(ledger.StateTransitionError, match="only once"):
        _append(context["repo"], context["draft"], event)


def test_activation_requires_all_three_evidence_types_and_reviewer(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    with pytest.raises(ledger.StateTransitionError, match="requires PostgreSQL"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("activated", context["key"], {}, reviewed_by="github:reviewer"),
        )
    _append_complete_evidence(context)
    with pytest.raises(ledger.StateTransitionError, match="reviewed_by"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("activated", context["key"], {}),
        )
    _append(
        context["repo"],
        context["draft"],
        _base_event("activated", context["key"], {}, reviewed_by="github:reviewer"),
    )
    with pytest.raises(ledger.StateTransitionError, match="require reviewed_by"):
        _append(
            context["repo"],
            context["draft"],
            _base_event(
                "rolled_back",
                context["key"],
                {
                    "reason": "unreviewed rollback",
                    "replacement_deployment_key": None,
                },
            ),
        )
    _append(
        context["repo"],
        context["draft"],
        _base_event(
            "rolled_back",
            context["key"],
            {"reason": "staging exercise complete", "replacement_deployment_key": None},
            reviewed_by="github:reviewer",
        ),
    )
    with pytest.raises(ledger.StateTransitionError, match="terminal"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("activated", context["key"], {}, reviewed_by="github:reviewer"),
        )


def test_status_stays_source_mapping_unverified_until_mapping_exists(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    for event_type, facts in (
        ("postgres_accepted", context["postgres_facts"]),
        ("roundtrip_accepted", context["roundtrip_facts"]),
    ):
        _append(
            context["repo"],
            context["draft"],
            _base_event(
                event_type,
                context["key"],
                facts,
                reviewed_by=(
                    "github:mapping-reviewer"
                    if event_type == "source_mapping_attested"
                    else None
                ),
            ),
        )
    snapshot = ledger.verify_ledger(
        repo_root=context["repo"], ledger_path=context["ledger"]
    )
    assert (
        snapshot.deployments[0].status
        == "DEPLOYMENT_DECLARED_SOURCE_MAPPING_UNVERIFIED"
    )

    _append(
        context["repo"],
        context["draft"],
        _base_event(
            "source_mapping_attested",
            context["key"],
            context["mapping_facts"],
            reviewed_by="github:mapping-reviewer",
        ),
    )
    snapshot = ledger.verify_ledger(
        repo_root=context["repo"], ledger_path=context["ledger"]
    )
    assert snapshot.deployments[0].status == "READY_FOR_REVIEWED_ACTIVATION"


@pytest.mark.parametrize(
    ("event_type", "document_key", "wrong_value"),
    [
        ("postgres_accepted", "release_id", f"release_{'a' * 64}"),
        ("roundtrip_accepted", "authority_mode", "legacy"),
        ("roundtrip_accepted", "event_id", " evt_e2e_ledger-test "),
        ("source_mapping_attested", "source_commit", "b" * 40),
    ],
)
def test_evidence_mismatch_fails_closed(
    evidence_repo: dict[str, Any],
    event_type: str,
    document_key: str,
    wrong_value: str,
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    if event_type == "postgres_accepted":
        document = json.loads(json.dumps(context["postgres"]))
        document["registry"][document_key] = wrong_value
        facts = json.loads(json.dumps(context["postgres_facts"]))
        path = context["repo"] / "evidence" / "postgres-bad.json"
    elif event_type == "roundtrip_accepted":
        document = dict(context["roundtrip"])
        document[document_key] = wrong_value
        facts = json.loads(json.dumps(context["roundtrip_facts"]))
        path = context["repo"] / "evidence" / "roundtrip-bad.json"
    else:
        document = dict(context["mapping"])
        document[document_key] = wrong_value
        facts = json.loads(json.dumps(context["mapping_facts"]))
        path = context["repo"] / "evidence" / "provider-bad.json"
    evidence_hash = _write_json(path, document)
    facts["evidence"] = {
        "path": path.relative_to(context["repo"]).as_posix(),
        "sha256": evidence_hash,
    }
    with pytest.raises(ledger.EvidenceValidationError):
        _append(
            context["repo"],
            context["draft"],
            _base_event(
                event_type,
                context["key"],
                facts,
                reviewed_by=(
                    "github:mapping-reviewer"
                    if event_type == "source_mapping_attested"
                    else None
                ),
            ),
        )


def test_same_release_can_have_multiple_deployment_keys(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    first = context["key"]
    second = "quiz-staging-002"
    _append(context["repo"], context["draft"], _candidate_event(context, first))
    _append(context["repo"], context["draft"], _declared_event(context, first))
    _append(context["repo"], context["draft"], _candidate_event(context, second))
    _append(
        context["repo"],
        context["draft"],
        _declared_event(context, second, deploy_id="deploy-002"),
    )

    snapshot = ledger.verify_ledger(
        repo_root=context["repo"], ledger_path=context["ledger"]
    )

    assert len(snapshot.deployments) == 2
    assert {item.release_id for item in snapshot.deployments} == {context["release_id"]}


def test_preflight_fingerprint_is_recomputed_from_source_commit(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    preflight_path = context["repo"] / "evidence" / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    forged_fingerprint = "f" * 64
    preflight["checked_rollout_input_sha256"] = forged_fingerprint
    preflight_sha256 = _write_json(preflight_path, preflight)
    candidate = json.loads(json.dumps(context["candidate"]))
    candidate["rollout"]["fingerprint"] = forged_fingerprint
    candidate["rollout"]["preflight"]["sha256"] = preflight_sha256

    with pytest.raises(ledger.EvidenceValidationError, match="source commit blobs"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("candidate_attested", context["key"], candidate),
        )


def test_evidence_summary_must_share_exact_deployment_context(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    facts = json.loads(json.dumps(context["postgres_facts"]))
    facts["summary"]["deployment_context_id"] = f"deployment_context_{'0' * 64}"

    with pytest.raises(ledger.EvidenceValidationError, match="safe summary"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("postgres_accepted", context["key"], facts),
        )


def test_hosted_evidence_sha_cannot_be_reused_across_deployments(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    first = context["key"]
    second = "quiz-staging-002"
    _append(context["repo"], context["draft"], _candidate_event(context, first))
    _append(context["repo"], context["draft"], _declared_event(context, first))
    _append(
        context["repo"],
        context["draft"],
        _base_event("postgres_accepted", first, context["postgres_facts"]),
    )
    second_declaration = dict(context["declaration"])
    second_declaration["deploy_id"] = "deploy-002"
    _append(context["repo"], context["draft"], _candidate_event(context, second))
    _append(
        context["repo"],
        context["draft"],
        _base_event("deployment_declared", second, second_declaration),
    )
    second_summary = _context_summary(
        deployment_key=second,
        release_id=context["release_id"],
        manifest_sha256=context["candidate"]["manifest"]["sha256"],
        registry_id=context["candidate"]["registry"]["registry_id"],
        source_commit=context["candidate"]["source"]["commit"],
        declaration=second_declaration,
    )
    reused_facts = json.loads(json.dumps(context["postgres_facts"]))
    reused_facts["summary"].update(second_summary)

    with pytest.raises(ledger.StateTransitionError, match="reused"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("postgres_accepted", second, reused_facts),
        )


def test_source_mapping_requires_distinct_reviewer_and_raw_provider_export(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    event = _base_event(
        "source_mapping_attested", context["key"], context["mapping_facts"]
    )
    with pytest.raises(ledger.StateTransitionError, match="require reviewed_by"):
        _append(context["repo"], context["draft"], event)
    event["reviewed_by"] = event["recorded_by"]
    with pytest.raises(ledger.StateTransitionError, match="distinct reviewer"):
        _append(context["repo"], context["draft"], event)

    export_path = context["repo"] / "evidence" / "provider-export.json"
    _write_json(export_path, {"forged_after_review": True})
    event["reviewed_by"] = "github:mapping-reviewer"
    with pytest.raises(ledger.EvidenceValidationError, match="raw SHA-256"):
        _append(context["repo"], context["draft"], event)


def test_activation_rejects_roundtrip_that_predates_provider_deployment(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    _append_candidate_and_declaration(context)
    late_mapping = json.loads(json.dumps(context["mapping"]))
    late_mapping["deployed_at"] = "2026-07-12T11:45:00Z"
    mapping_path = context["repo"] / "evidence" / "provider-late.json"
    mapping_sha256 = _write_json(mapping_path, late_mapping)
    mapping_facts = json.loads(json.dumps(context["mapping_facts"]))
    mapping_facts["evidence"] = {
        "path": "evidence/provider-late.json",
        "sha256": mapping_sha256,
    }
    mapping_facts["summary"]["deployed_at"] = late_mapping["deployed_at"]
    for event_type, facts, reviewer in (
        ("postgres_accepted", context["postgres_facts"], None),
        ("roundtrip_accepted", context["roundtrip_facts"], None),
        ("source_mapping_attested", mapping_facts, "github:mapping-reviewer"),
    ):
        _append(
            context["repo"],
            context["draft"],
            _base_event(event_type, context["key"], facts, reviewed_by=reviewer),
        )

    with pytest.raises(ledger.StateTransitionError, match="predates"):
        _append(
            context["repo"],
            context["draft"],
            _base_event("activated", context["key"], {}, reviewed_by="github:reviewer"),
        )


def test_cli_append_without_confirmation_never_creates_ledger(
    evidence_repo: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    context = evidence_repo
    _write_json(context["draft"], _candidate_event(context))
    ledger_path = context["repo"] / context["ledger"]

    exit_code = ledger.main(
        [
            "append",
            "--repo",
            str(context["repo"]),
            "--ledger",
            str(context["ledger"]),
            "--event-json",
            str(context["draft"]),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert not ledger_path.exists()
    assert "confirm-append" in captured.err


def test_missing_ledger_verifies_as_empty_without_creating_a_record(
    evidence_repo: dict[str, Any],
) -> None:
    context = evidence_repo
    path = context["repo"] / context["ledger"]

    snapshot = ledger.verify_ledger(
        repo_root=context["repo"], ledger_path=context["ledger"]
    )

    assert snapshot.records == ()
    assert snapshot.deployments == ()
    assert snapshot.head_record_sha256 is None
    assert not path.exists()
