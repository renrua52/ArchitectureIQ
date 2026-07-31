"""Tests for the attested, insert-only feedback registry exporter."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import export_feedback_registry as registry_cli  # noqa: E402
import quiz_bundle.feedback_registry as registry_module  # noqa: E402
from quiz_bundle import publish_quiz_bundle  # noqa: E402
from quiz_bundle.feedback_registry import (  # noqa: E402
    FeedbackRegistryError,
    build_feedback_registry,
    export_feedback_registry,
    render_feedback_registry_sql,
    serialize_feedback_registry,
)
from quiz_bundle.versioning import compute_question_version  # noqa: E402


DEMO = REPO / "examples" / "quiz_demo" / "bundle"
DEMO_REGISTRY = (
    REPO
    / "supabase/registries/release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f.json"
)
DEMO_REGISTRY_SQL = (
    REPO
    / "supabase/migrations/20260712014500_feedback_question_registry_release_4e752a.sql"
)
DEMO_QUESTION = sorted(DEMO.glob("datasets/*/*/questions/*/q_*/question.json"))[0]
RELEASE_CORE_KEYS = (
    "schema_version",
    "source_runs",
    "questions",
    "counts",
    "artifacts",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any], *, ensure_ascii: bool = False) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n",
        encoding="utf-8",
    )


def _release_id(document: dict[str, Any]) -> str:
    core = {key: document[key] for key in RELEASE_CORE_KEYS}
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"release_{hashlib.sha256(canonical).hexdigest()}"


def _resign_bundle(
    root: Path,
    *,
    changed_artifact: str | None = None,
    update_question_version: bool = False,
) -> None:
    manifest_path = root / "quiz_manifest.json"
    document = _read(manifest_path)
    if changed_artifact is not None:
        artifact_path = root / PurePosixPath(changed_artifact)
        artifact = next(
            item for item in document["artifacts"] if item["path"] == changed_artifact
        )
        artifact["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact["size"] = artifact_path.stat().st_size
        document["counts"]["artifact_bytes"] = sum(
            item["size"] for item in document["artifacts"]
        )
    if update_question_version:
        question_entry = document["questions"][0]
        question = _read(root / question_entry["path"] / "question.json")
        question_entry["version"] = compute_question_version(question)
    document["release_id"] = _release_id(document)
    _write(manifest_path, document)


@pytest.fixture(scope="session")
def feedback_registry_bundle_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("feedback_registry") / "bundle"
    source = DEMO_QUESTION.parent.relative_to(DEMO)
    publish_quiz_bundle(
        DEMO,
        [source],
        root,
        generated_at="2026-07-12T00:00:00Z",
    )
    return root


@pytest.fixture
def feedback_registry_bundle(
    tmp_path: Path,
    feedback_registry_bundle_template: Path,
) -> Path:
    root = tmp_path / "bundle"
    shutil.copytree(feedback_registry_bundle_template, root)
    return root


def test_registry_shape_content_hash_and_answer_mapping_are_deterministic(
    feedback_registry_bundle: Path,
) -> None:
    first = build_feedback_registry(feedback_registry_bundle)
    second = build_feedback_registry(feedback_registry_bundle)

    assert first == second
    assert set(first) == {
        "schema_version",
        "release_id",
        "manifest_sha256",
        "registry_id",
        "question_count",
        "choice_count",
        "questions",
    }
    assert first["schema_version"] == "1.0"
    assert first["question_count"] == 1
    assert first["choice_count"] == 3
    assert (
        first["manifest_sha256"]
        == hashlib.sha256(
            (feedback_registry_bundle / "quiz_manifest.json").read_bytes()
        ).hexdigest()
    )

    question = first["questions"][0]
    source_question = _read(
        next(
            feedback_registry_bundle.glob("datasets/*/*/questions/*/q_*/question.json")
        )
    )
    expected_choices = {
        choice["letter"]: choice["candidate_id"]
        for choice in source_question["choices"]
    }
    assert question == {
        "question_id": source_question["question_id"],
        "question_version": compute_question_version(source_question),
        "family": source_question["family"],
        "dataset_id": source_question["dataset_id"],
        "question_type": source_question["type"],
        "correct_letter": source_question["correct_letter"],
        "correct_candidate_id": expected_choices[source_question["correct_letter"]],
        "choices": dict(sorted(expected_choices.items())),
    }

    core = {
        key: value
        for key, value in first.items()
        if key not in {"registry_id", "manifest_sha256"}
    }
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert first["registry_id"] == (f"registry_{hashlib.sha256(canonical).hexdigest()}")
    assert serialize_feedback_registry(first).endswith("\n")


def test_checked_in_demo_registry_exactly_matches_attested_60_question_bundle() -> None:
    registry = export_feedback_registry(
        DEMO,
        json_output=DEMO_REGISTRY,
        sql_output=DEMO_REGISTRY_SQL,
        check=True,
    )

    assert registry["question_count"] == 60
    assert registry["choice_count"] == 180
    assert registry["release_id"] == (
        "release_4e752ad75ce29cebe0252cb5705880b6e346baf66c8c25fc49cb536de711084f"
    )
    assert registry["registry_id"] == (
        "registry_db3f1a166af0b526e08d4eff49539c6a2150653d1940b0fcccbdbfbe0b525131"
    )


def test_manifest_provenance_changes_do_not_change_registry_identity(
    feedback_registry_bundle: Path,
) -> None:
    first = build_feedback_registry(feedback_registry_bundle)
    manifest_path = feedback_registry_bundle / "quiz_manifest.json"
    manifest = _read(manifest_path)
    manifest["generated_at"] = "2026-07-12T00:00:01Z"
    _write(manifest_path, manifest)

    second = build_feedback_registry(feedback_registry_bundle)

    assert second["release_id"] == first["release_id"]
    assert second["questions"] == first["questions"]
    assert second["registry_id"] == first["registry_id"]
    assert second["manifest_sha256"] != first["manifest_sha256"]


def test_sql_is_deterministic_insert_only_and_uses_explicit_columns(
    feedback_registry_bundle: Path,
) -> None:
    registry = build_feedback_registry(feedback_registry_bundle)
    sql = render_feedback_registry_sql(registry)

    assert sql == render_feedback_registry_sql(registry)
    assert sql.startswith("begin;\n\n")
    assert sql.endswith("\n\ncommit;\n")
    assert sql.count("insert into ") == 3
    assert "insert into public.feedback_quiz_releases (" in sql
    assert "insert into public.feedback_quiz_questions (" in sql
    assert "insert into public.feedback_quiz_choices (" in sql
    for column in (
        "release_id",
        "registry_schema_version",
        "manifest_sha256",
        "registry_id",
        "question_count",
        "choice_count",
        "question_id",
        "question_version",
        "family",
        "dataset_id",
        "question_type",
        "correct_letter",
        "correct_candidate_id",
        "letter",
        "candidate_id",
    ):
        assert column in sql
    assert "registered_at" not in sql
    lowered = sql.lower()
    for forbidden in (
        "update ",
        "delete ",
        "upsert",
        "on conflict",
        "merge ",
        "truncate ",
        "copy ",
        "http",
        "service_role",
    ):
        assert forbidden not in lowered


def test_sql_text_literals_escape_quotes_and_reject_nul() -> None:
    assert registry_module._sql_literal("x'); delete from t; --") == (
        "'x''); delete from t; --'"
    )
    with pytest.raises(FeedbackRegistryError, match="NUL"):
        registry_module._sql_literal("bad\x00value")


def test_runtime_attestation_rejects_tampered_artifact(
    feedback_registry_bundle: Path,
) -> None:
    prompt = next(
        feedback_registry_bundle.glob("datasets/*/*/questions/*/q_*/prompt.txt")
    )
    prompt.write_text(prompt.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")

    with pytest.raises(FeedbackRegistryError, match="runtime release attestation"):
        build_feedback_registry(feedback_registry_bundle)


def test_publisher_gt_validation_runs_after_valid_runtime_attestation(
    feedback_registry_bundle: Path,
) -> None:
    question = _read(
        next(
            feedback_registry_bundle.glob("datasets/*/*/questions/*/q_*/question.json")
        )
    )
    summary = (
        feedback_registry_bundle
        / question["choices"][0]["candidate_path"]
        / "results"
        / "summary.json"
    )
    summary_value = _read(summary)
    summary_value["excluded"] = True
    _write(summary, summary_value)
    _resign_bundle(
        feedback_registry_bundle,
        changed_artifact=summary.relative_to(feedback_registry_bundle).as_posix(),
    )

    with pytest.raises(
        FeedbackRegistryError,
        match="publisher ground-truth validation.*excluded by ground truth",
    ):
        build_feedback_registry(feedback_registry_bundle)


def test_duplicate_choices_are_not_exported_even_when_manifest_is_resigned(
    feedback_registry_bundle: Path,
) -> None:
    question_path = next(
        feedback_registry_bundle.glob("datasets/*/*/questions/*/q_*/question.json")
    )
    question = _read(question_path)
    question["choices"][1]["letter"] = question["choices"][0]["letter"]
    _write(question_path, question)
    _resign_bundle(
        feedback_registry_bundle,
        changed_artifact=question_path.relative_to(feedback_registry_bundle).as_posix(),
        update_question_version=True,
    )

    with pytest.raises(
        FeedbackRegistryError,
        match="publisher ground-truth validation.*duplicate answer letter",
    ):
        build_feedback_registry(feedback_registry_bundle)


def test_manifest_must_exactly_match_publisher_rebuild(
    feedback_registry_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = registry_module.build_bundle_manifest

    def disagree(*args: Any, **kwargs: Any) -> dict[str, Any]:
        rebuilt = original(*args, **kwargs)
        rebuilt["generated_at"] = "different"
        return rebuilt

    monkeypatch.setattr(registry_module, "build_bundle_manifest", disagree)
    with pytest.raises(FeedbackRegistryError, match="does not exactly match"):
        build_feedback_registry(feedback_registry_bundle)


def test_strict_manifest_unicode_contract_is_applied(
    feedback_registry_bundle: Path,
) -> None:
    manifest_path = feedback_registry_bundle / "quiz_manifest.json"
    manifest = _read(manifest_path)
    manifest["generated_at"] = "\ud800"
    _write(manifest_path, manifest, ensure_ascii=True)

    with pytest.raises(FeedbackRegistryError, match="Unicode surrogate"):
        build_feedback_registry(feedback_registry_bundle)


def test_unsafe_question_number_is_reported_as_attestation_failure(
    feedback_registry_bundle: Path,
) -> None:
    question_path = next(
        feedback_registry_bundle.glob("datasets/*/*/questions/*/q_*/question.json")
    )
    question = _read(question_path)
    question["unsafe_number"] = 1 << 53
    _write(question_path, question)
    _resign_bundle(
        feedback_registry_bundle,
        changed_artifact=question_path.relative_to(feedback_registry_bundle).as_posix(),
    )

    with pytest.raises(
        FeedbackRegistryError,
        match="runtime release attestation.*9007199254740991",
    ):
        build_feedback_registry(feedback_registry_bundle)


def test_mutated_registry_correct_answer_and_content_hash_are_rejected(
    feedback_registry_bundle: Path,
) -> None:
    registry = build_feedback_registry(feedback_registry_bundle)
    wrong = deepcopy(registry)
    question = wrong["questions"][0]
    question["correct_candidate_id"] = next(
        candidate_id
        for letter, candidate_id in question["choices"].items()
        if letter != question["correct_letter"]
    )

    with pytest.raises(FeedbackRegistryError, match="correct answer"):
        render_feedback_registry_sql(wrong)

    wrong = deepcopy(registry)
    wrong["choice_count"] += 1
    with pytest.raises(FeedbackRegistryError, match="counts do not match"):
        serialize_feedback_registry(wrong)


def test_atomic_export_check_mode_and_external_output_rule(
    feedback_registry_bundle: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "release_outputs"
    json_output = output_dir / "feedback_registry.json"
    sql_output = output_dir / "feedback_registry.sql"

    registry = export_feedback_registry(
        feedback_registry_bundle,
        json_output=json_output,
        sql_output=sql_output,
    )
    assert json.loads(json_output.read_text(encoding="utf-8")) == registry
    assert sql_output.read_text(encoding="utf-8") == render_feedback_registry_sql(
        registry
    )
    assert not list(output_dir.glob(".*.tmp"))
    assert (
        export_feedback_registry(
            feedback_registry_bundle,
            json_output=json_output,
            sql_output=sql_output,
            check=True,
        )
        == registry
    )

    json_output.write_text("stale\n", encoding="utf-8")
    before_json = json_output.read_bytes()
    before_sql = sql_output.read_bytes()
    with pytest.raises(FeedbackRegistryError, match="does not exactly match"):
        export_feedback_registry(
            feedback_registry_bundle,
            json_output=json_output,
            sql_output=sql_output,
            check=True,
        )
    assert json_output.read_bytes() == before_json
    assert sql_output.read_bytes() == before_sql

    with pytest.raises(FeedbackRegistryError, match="outside"):
        export_feedback_registry(
            feedback_registry_bundle,
            json_output=feedback_registry_bundle / "registry.json",
            sql_output=tmp_path / "unused.sql",
        )
    assert not (feedback_registry_bundle / "registry.json").exists()


def test_cli_writes_and_checks_both_outputs(
    feedback_registry_bundle: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_output = tmp_path / "registry.json"
    sql_output = tmp_path / "registry.sql"
    arguments = [
        "--bundle",
        str(feedback_registry_bundle),
        "--json-output",
        str(json_output),
        "--sql-output",
        str(sql_output),
    ]

    assert registry_cli.main(arguments) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["checked"] is False
    assert json_output.is_file()
    assert sql_output.is_file()

    assert registry_cli.main([*arguments, "--check"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["checked"] is True

    sql_output.write_text("stale\n", encoding="utf-8")
    assert registry_cli.main([*arguments, "--check"]) == 1
    assert "does not exactly match" in capsys.readouterr().err
