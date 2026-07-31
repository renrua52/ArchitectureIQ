"""Tests for runtime release attestation and feedback attribution."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
INSPECTOR = REPO / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

from feedback import compute_question_version  # noqa: E402
from release_manifest import (  # noqa: E402
    ReleaseManifestError,
    load_quiz_manifest,
)


class _SessionState(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "quiz_manifest.json"
    ]


def _release_id(document: dict[str, Any]) -> str:
    core = {
        key: document[key]
        for key in (
            "schema_version",
            "source_runs",
            "questions",
            "counts",
            "artifacts",
        )
    }
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"release_{hashlib.sha256(canonical).hexdigest()}"


def _write_manifest(root: Path, document: dict[str, Any]) -> None:
    document["release_id"] = _release_id(document)
    _write_json(root / "quiz_manifest.json", document)


def _candidate_counts(artifacts: list[dict[str, Any]]) -> tuple[int, int]:
    sets: set[tuple[str, ...]] = set()
    candidates: set[tuple[str, ...]] = set()
    for artifact in artifacts:
        parts = PurePosixPath(artifact["path"]).parts
        if len(parts) >= 6 and parts[0] == "datasets" and parts[3] == "candidates":
            sets.add(parts[:5])
            if len(parts) >= 7:
                candidates.add(parts[:6])
    return len(sets), len(candidates)


def _build_bundle(
    root: Path,
    *,
    question_ids: tuple[str, ...] = ("q_test",),
) -> dict[str, Any]:
    family = "family"
    dataset_id = "dataset"
    run_id = "run_test"
    run_path = f"datasets/{family}/{dataset_id}/questions/{run_id}"
    questions: list[dict[str, Any]] = []
    for question_id in sorted(question_ids):
        question_path = f"{run_path}/{question_id}"
        question = {
            "question_id": question_id,
            "family": family,
            "dataset_id": dataset_id,
            "question_run_id": run_id,
            "type": "mixed",
            "choices": [],
        }
        _write_json(root / question_path / "question.json", question)
        (root / question_path / "prompt.txt").write_text(
            f"prompt for {question_id}\n", encoding="utf-8"
        )
        questions.append(
            {
                "question_id": question_id,
                "version": compute_question_version(question),
                "family": family,
                "dataset_id": dataset_id,
                "path": question_path,
                "source_run": run_id,
                "source_run_path": run_path,
            }
        )

    _write_json(
        root / run_path / "run.json",
        {
            "run_id": run_id,
            "family": family,
            "dataset_id": dataset_id,
            "question_ids": list(question_ids),
            "num_questions": len(question_ids),
        },
    )
    candidate = (
        root / "datasets" / family / dataset_id / "candidates" / "set_demo" / "c_demo"
    )
    _write_json(candidate / "results" / "summary.json", {"final_metric": 1.0})
    (candidate / "results" / "curves.npz").write_bytes(b"curve-bytes-0001")

    artifacts = _artifact_inventory(root)
    set_count, candidate_count = _candidate_counts(artifacts)
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "release_id": "",
        "generated_at": "2026-07-11T00:00:00Z",
        "source_runs": [
            {
                "run_id": run_id,
                "family": family,
                "dataset_id": dataset_id,
                "path": run_path,
                "selected_question_ids": sorted(question_ids),
                "declared_question_count": len(question_ids),
                "partial": False,
            }
        ],
        "questions": questions,
        "counts": {
            "questions": len(questions),
            "source_runs": 1,
            "datasets": 1,
            "candidate_sets": set_count,
            "candidates": candidate_count,
            "artifact_files": len(artifacts),
            "artifact_bytes": sum(item["size"] for item in artifacts),
        },
        "artifacts": artifacts,
    }
    _write_manifest(root, document)
    return document


def _rewrite_artifact_claim(
    root: Path,
    document: dict[str, Any],
    relative: str,
) -> None:
    path = root / PurePosixPath(relative)
    item = next(
        artifact for artifact in document["artifacts"] if artifact["path"] == relative
    )
    item["sha256"] = _sha256(path)
    item["size"] = path.stat().st_size
    document["counts"]["artifact_bytes"] = sum(
        artifact["size"] for artifact in document["artifacts"]
    )
    _write_manifest(root, document)


def test_manifest_order_exact_identity_and_manifest_digest(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root, question_ids=("q_second", "q_first"))

    manifest = load_quiz_manifest(root)

    assert manifest is not None
    assert [path.name for path in manifest.question_dirs()] == ["q_first", "q_second"]
    assert manifest.question_count == 2
    assert manifest.source_run_count == 1
    assert manifest.artifact_count == document["counts"]["artifact_files"]
    assert manifest.artifact_bytes == document["counts"]["artifact_bytes"]
    assert manifest.manifest_sha256 == _sha256(root / "quiz_manifest.json")
    first = manifest.questions[0]
    assert (
        manifest.release_id_for(
            first.question_id,
            first.version,
            question_path=root / first.path,
        )
        == document["release_id"]
    )
    assert manifest.release_id_for(first.question_id, f"qv1_{'3' * 64}") is None


def test_missing_manifest_means_unversioned_data_root(tmp_path: Path) -> None:
    assert load_quiz_manifest(tmp_path) is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.pop("artifacts"), "top-level schema"),
        (lambda document: document.__setitem__("unexpected", True), "top-level schema"),
        (
            lambda document: document["counts"].__setitem__("questions", 2),
            "counts.questions",
        ),
        (
            lambda document: document["questions"][0].__setitem__("path", "../outside"),
            "normalized relative path",
        ),
        (
            lambda document: document["questions"][0].__setitem__("extra", True),
            "exact schema",
        ),
        (
            lambda document: document["source_runs"][0].__setitem__("partial", True),
            "partial",
        ),
    ],
)
def test_invalid_manifest_schema_or_cross_reference_is_rejected(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    mutation(document)
    _write_json(root / "quiz_manifest.json", document)

    with pytest.raises(ReleaseManifestError, match=message):
        load_quiz_manifest(root)


def test_forged_well_formed_release_id_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    document["release_id"] = f"release_{'a' * 64}"
    _write_json(root / "quiz_manifest.json", document)

    with pytest.raises(ReleaseManifestError, match="release_id does not match"):
        load_quiz_manifest(root)


def test_duplicate_manifest_object_key_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _build_bundle(root)
    manifest_path = root / "quiz_manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace("{", '{"schema_version":"1.0",', 1),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseManifestError, match="duplicate JSON object key"):
        load_quiz_manifest(root)


@pytest.mark.parametrize(
    "relative",
    [
        "datasets/family/dataset/questions/run_test/q_test/prompt.txt",
        "datasets/family/dataset/candidates/set_demo/c_demo/results/summary.json",
        "datasets/family/dataset/candidates/set_demo/c_demo/results/curves.npz",
    ],
)
def test_tampered_prompt_summary_or_curve_is_rejected(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "bundle"
    _build_bundle(root)
    path = root / PurePosixPath(relative)
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ReleaseManifestError, match="size is"):
        load_quiz_manifest(root)


def test_same_size_and_mtime_tamper_is_still_hashed(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _build_bundle(root)
    assert load_quiz_manifest(root) is not None
    prompt = root / "datasets/family/dataset/questions/run_test/q_test/prompt.txt"
    before = prompt.stat()
    original = prompt.read_bytes()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    prompt.write_bytes(replacement)
    os.utime(prompt, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = prompt.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns

    with pytest.raises(ReleaseManifestError, match="SHA-256"):
        load_quiz_manifest(root)


def test_missing_and_extra_physical_artifacts_are_rejected(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    _build_bundle(missing_root)
    (
        missing_root / "datasets/family/dataset/questions/run_test/q_test/prompt.txt"
    ).unlink()
    with pytest.raises(ReleaseManifestError, match="inventory.*missing"):
        load_quiz_manifest(missing_root)

    extra_root = tmp_path / "extra"
    _build_bundle(extra_root)
    (extra_root / "extra.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(ReleaseManifestError, match="inventory.*extra"):
        load_quiz_manifest(extra_root)


def test_any_symbolic_link_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _build_bundle(root)
    try:
        (root / "linked-prompt").symlink_to(
            root / "datasets/family/dataset/questions/run_test/q_test/prompt.txt"
        )
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ReleaseManifestError, match="symbolic links"):
        load_quiz_manifest(root)


def test_artifact_paths_must_be_unique_normalized_and_sorted(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    document["artifacts"][0], document["artifacts"][1] = (
        document["artifacts"][1],
        document["artifacts"][0],
    )
    _write_json(root / "quiz_manifest.json", document)
    with pytest.raises(ReleaseManifestError, match="sorted"):
        load_quiz_manifest(root)

    document = _build_bundle(root)
    document["artifacts"][0]["path"] = "../outside"
    _write_json(root / "quiz_manifest.json", document)
    with pytest.raises(ReleaseManifestError, match="normalized relative path"):
        load_quiz_manifest(root)


def test_question_json_identity_and_version_are_attested(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    document["questions"][0]["version"] = f"qv1_{'0' * 64}"
    _write_manifest(root, document)

    with pytest.raises(ReleaseManifestError, match="version does not match"):
        load_quiz_manifest(root)


def test_optional_question_run_id_and_run_count_match_publisher_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    question_relative = (
        "datasets/family/dataset/questions/run_test/q_test/question.json"
    )
    question = json.loads((root / question_relative).read_text(encoding="utf-8"))
    question.pop("question_run_id")
    _write_json(root / question_relative, question)
    document["questions"][0]["version"] = compute_question_version(question)
    _rewrite_artifact_claim(root, document, question_relative)

    run_relative = "datasets/family/dataset/questions/run_test/run.json"
    run = json.loads((root / run_relative).read_text(encoding="utf-8"))
    run.pop("num_questions")
    _write_json(root / run_relative, run)
    _rewrite_artifact_claim(root, document, run_relative)

    manifest = load_quiz_manifest(root)
    assert manifest is not None
    assert manifest.question_count == 1


def test_question_json_is_strict_even_when_artifact_claim_is_refreshed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    document = _build_bundle(root)
    relative = "datasets/family/dataset/questions/run_test/q_test/question.json"
    (root / relative).write_text(
        '{"question_id":"q_test","score":NaN}', encoding="utf-8"
    )
    _rewrite_artifact_claim(root, document, relative)

    with pytest.raises(ReleaseManifestError, match="non-standard numeric constant"):
        load_quiz_manifest(root)


def test_demo_release_attests_all_60_questions_and_1483_artifacts() -> None:
    root = REPO / "examples" / "quiz_demo" / "bundle"
    manifest = load_quiz_manifest(root)

    assert manifest is not None
    assert manifest.question_count == 60
    assert manifest.source_run_count == 3
    assert manifest.artifact_count == 1483
    assert manifest.artifact_bytes == 4_039_296
    assert manifest.manifest_sha256 == _sha256(root / "quiz_manifest.json")


def test_event_context_attaches_only_the_attested_active_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as inspector_app

    root = tmp_path / "bundle"
    document = _build_bundle(root)
    manifest = load_quiz_manifest(root)
    assert manifest is not None
    question_entry = document["questions"][0]
    question = json.loads(
        (root / question_entry["path"] / "question.json").read_text(encoding="utf-8")
    )
    bundle = inspector_app.QuestionBundle(
        question_root=(root / question_entry["path"]).resolve(),
        data_root=root.resolve(),
        question=question,
        prompt_text="",
        dataset_dir=root,
        choices=[],
    )
    state = _SessionState(
        quiz_attempt_id="attempt_test",
        quiz_manifest=manifest,
        bundle=bundle,
        data_root=str(root),
    )
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    assert (
        inspector_app._event_context(question)["release_id"] == document["release_id"]
    )
    question["type"] = "optimizer_only"
    assert "release_id" not in inspector_app._event_context(question)


def test_default_bundled_root_manifest_failure_stops_question_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as inspector_app

    root = tmp_path / "bundle"
    _write_json(root / "quiz_manifest.json", {})
    question = root / "datasets/family/dataset/questions/run/q/question.json"
    _write_json(question, {"question_id": "q_stale"})
    state = _SessionState(quiz_manifest=None, quiz_manifest_error=None)
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    monkeypatch.setattr(
        inspector_app, "_bundled_data_root", lambda **kwargs: root.resolve()
    )

    manifest, pool = inspector_app._load_question_pool(str(root))

    assert manifest is None
    assert pool == []
    assert "top-level schema" in state.quiz_manifest_error


def test_missing_default_manifest_fails_but_local_unversioned_root_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as inspector_app

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    local = tmp_path / "local"
    question = local / "datasets/family/dataset/questions/run/q/question.json"
    _write_json(question, {"question_id": "q_local"})
    state = _SessionState(quiz_manifest=None, quiz_manifest_error=None)
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    monkeypatch.setattr(
        inspector_app, "_bundled_data_root", lambda **kwargs: bundled.resolve()
    )

    manifest, pool = inspector_app._load_question_pool(str(bundled))
    assert manifest is None
    assert pool == []
    assert "required quiz_manifest.json is missing" in state.quiz_manifest_error

    manifest, pool = inspector_app._load_question_pool(str(local))
    assert manifest is None
    assert pool == [question.parent.resolve()]
    assert state.quiz_manifest_error is None


def test_runtime_git_sha_requires_checkout_and_rejects_bad_env_values() -> None:
    import app as inspector_app

    sha = "A" * 40

    def no_checkout(_root: Path) -> None:
        return None

    assert (
        inspector_app._runtime_git_sha({"GIT_COMMIT": sha}, git_reader=no_checkout)
        is None
    )
    assert (
        inspector_app._runtime_git_sha({"GIT_COMMIT": "abc123"}, git_reader=no_checkout)
        is None
    )
    assert (
        inspector_app._runtime_git_sha(
            {"GIT_COMMIT": "a" * 40, "COMMIT_SHA": "b" * 40},
            git_reader=no_checkout,
        )
        is None
    )

    def malformed_checkout(_root: Path) -> str:
        return "abc123"

    assert (
        inspector_app._runtime_git_sha(
            {"GIT_COMMIT": sha},
            git_reader=malformed_checkout,
        )
        is None
    )


def test_runtime_git_sha_uses_checkout_and_rejects_declaration_mismatch(
    tmp_path: Path,
) -> None:
    import app as inspector_app

    checkout_sha = "c" * 40
    calls: list[Path] = []

    def checkout(root: Path) -> str:
        calls.append(root)
        return checkout_sha

    assert (
        inspector_app._runtime_git_sha({}, repo_root=tmp_path, git_reader=checkout)
        == checkout_sha
    )
    assert calls == [tmp_path]
    assert (
        inspector_app._runtime_git_sha(
            {"GIT_COMMIT": checkout_sha.upper()},
            repo_root=tmp_path,
            git_reader=checkout,
        )
        == checkout_sha
    )
    assert (
        inspector_app._runtime_git_sha(
            {"GIT_COMMIT": "d" * 40},
            repo_root=tmp_path,
            git_reader=checkout,
        )
        is None
    )


def test_checkout_git_reader_uses_minimal_non_redirectable_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as inspector_app

    sha = "e" * 40
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = f"{sha}\n"

    def fake_run(command: list[str], **kwargs: object) -> _Completed:
        captured["command"] = command
        captured.update(kwargs)
        return _Completed()

    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_REPLACE_REF_BASE",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_COUNT",
    ):
        monkeypatch.setenv(name, f"/redirected/{name.lower()}")
    monkeypatch.setattr(inspector_app.subprocess, "run", fake_run)

    assert inspector_app._checkout_git_sha(tmp_path) == sha
    assert captured["command"] == [
        "git",
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]
    assert captured["env"] == inspector_app._checkout_git_environment()
    assert captured["env"] == {
        "PATH": inspector_app.os.defpath,
        "HOME": inspector_app.os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def test_default_root_ignores_stale_preexisting_local_data(tmp_path: Path) -> None:
    import app as inspector_app

    stale = tmp_path / "data/datasets/family/old/questions/run_old/q_old"
    _write_json(stale / "question.json", {"question_id": "q_old"})
    bundled = tmp_path / "examples/quiz_demo/bundle"
    bundled.mkdir(parents=True)

    selected = inspector_app._initial_data_root(
        environ={},
        argv=["app.py"],
        root=tmp_path,
    )

    assert Path(selected) == bundled.resolve()
    assert Path(selected) != (tmp_path / "data").resolve()


def test_explicit_question_argument_selects_its_local_data_root(tmp_path: Path) -> None:
    import app as inspector_app

    question = tmp_path / "data/datasets/family/dataset/questions/run/q"
    _write_json(question / "question.json", {"question_id": "q_local"})

    selected = inspector_app._initial_data_root(
        environ={inspector_app.DATA_ROOT_ENV: str(tmp_path / "ignored")},
        argv=["app.py", str(question)],
        root=tmp_path,
    )

    assert Path(selected) == (tmp_path / "data").resolve()


def test_default_streamlit_app_displays_runtime_release_attestation() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(INSPECTOR / "app.py")).run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("Attested release `release_4e752" in value for value in captions)
    assert any("Manifest SHA-256 `" in value for value in captions)
    assert any("1483 verified artifacts" in value for value in captions)
    assert any(
        "Entry: `tools/question_inspector/app.py`" in value for value in captions
    )
