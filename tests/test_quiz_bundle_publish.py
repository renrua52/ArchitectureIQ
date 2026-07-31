"""Tests for the maintainer-facing canonical quiz bundle publisher."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
INSPECTOR = TOOLS / "question_inspector"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(INSPECTOR))

from artifact_loader import list_question_dirs, load_question_bundle  # noqa: E402
from feedback import (  # noqa: E402
    FeedbackValidationError,
    compute_question_version as feedback_question_version,
)
import quiz_bundle.publisher as publisher_module  # noqa: E402
from quiz_bundle import (  # noqa: E402
    BundlePublishError,
    QuestionVersionError,
    build_bundle_manifest,
    compute_question_version,
    publish_quiz_bundle,
)
from quiz_bundle.publisher import (  # noqa: E402
    _question_evaluation,
    _validate_question_budget,
)


DEMO = REPO / "examples" / "quiz_demo" / "bundle"
DEMO_QUESTIONS = sorted(DEMO.glob("datasets/*/*/questions/*/q_*/question.json"))
DEMO_QUESTION = DEMO_QUESTIONS[0]
_TEMPLATE_TEMP: tempfile.TemporaryDirectory[str] | None = None
_TEMPLATE_SOURCE: Path | None = None


def _question_source() -> Path:
    return DEMO_QUESTION.parent.relative_to(DEMO)


def _other_question_source() -> Path:
    return DEMO_QUESTIONS[1].parent.relative_to(DEMO)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _byte_tree(root: Path) -> tuple[tuple[str, str, bytes | None], ...] | None:
    if not root.exists():
        return None
    entries: list[tuple[str, str, bytes | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append((relative, "directory", None))
        elif path.is_file():
            entries.append((relative, "file", path.read_bytes()))
        else:
            entries.append((relative, "other", None))
    return tuple(entries)


def _isolated_source(tmp_path: Path) -> tuple[Path, Path]:
    global _TEMPLATE_SOURCE, _TEMPLATE_TEMP
    if _TEMPLATE_SOURCE is None:
        _TEMPLATE_TEMP = tempfile.TemporaryDirectory(
            prefix="architecture_iq_publisher_tests_"
        )
        _TEMPLATE_SOURCE = Path(_TEMPLATE_TEMP.name) / "source"
        publish_quiz_bundle(DEMO, [_question_source()], _TEMPLATE_SOURCE)
        (_TEMPLATE_SOURCE / "quiz_manifest.json").unlink()

    source = tmp_path / "source"
    shutil.copytree(_TEMPLATE_SOURCE, source)
    question_dir = list_question_dirs(source)[0]
    return source, question_dir


def _choice_candidate(
    source: Path,
    question: dict[str, Any],
    index: int = 0,
) -> Path:
    return source / question["choices"][index]["candidate_path"]


def test_publish_demo_question_preserves_and_loads_canonical_artifacts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bundle"
    manifest = publish_quiz_bundle(DEMO, [_question_source()], target)

    assert manifest == _read(target / "quiz_manifest.json")
    assert manifest["counts"]["questions"] == 1
    assert manifest["source_runs"][0]["partial"] is True
    assert manifest["source_runs"][0]["selected_question_ids"] == [
        _read(DEMO_QUESTION)["question_id"]
    ]

    copied_questions = list_question_dirs(target)
    assert len(copied_questions) == 1
    bundle = load_question_bundle(copied_questions[0], target)
    assert bundle.question == _read(DEMO_QUESTION)
    assert (bundle.dataset_dir / "dataset_spec.json").is_file()
    assert (bundle.dataset_dir / "synthesize.py").is_file()
    assert (bundle.dataset_dir / "train.pt").is_file()
    assert (bundle.dataset_dir / "test.pt").is_file()

    source_run = DEMO_QUESTION.parents[1] / "run.json"
    copied_run = copied_questions[0].parent / "run.json"
    assert copied_run.read_bytes() == source_run.read_bytes()
    for choice in bundle.question["choices"]:
        candidate = target / choice["candidate_path"]
        assert (candidate / "candidate_spec.json").is_file()
        assert (candidate / "results" / "summary.json").is_file()
        assert (target / choice["candidate_set_path"] / "set.json").is_file()

    # The whole set is present, including candidates not selected by this q.
    set_dir = target / bundle.question["choices"][0]["candidate_set_path"]
    set_spec = _read(set_dir / "set.json")
    assert len(list(set_dir.glob("c_*/candidate_spec.json"))) == set_spec["count"]


def test_publish_ignores_noncanonical_runtime_and_user_files(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    custom_file = question_dir / "custom_settings" / "setting_1" / "comment.txt"
    custom_file.parent.mkdir(parents=True)
    custom_file.write_text("must not ship", encoding="utf-8")
    candidate = source / question["choices"][0]["candidate_path"]
    cache_file = candidate / "__pycache__" / "train.pyc"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"runtime cache")

    target = tmp_path / "target"
    manifest = publish_quiz_bundle(source, [question_dir], target)

    assert not (target / custom_file.relative_to(source)).exists()
    assert not (target / cache_file.relative_to(source)).exists()
    assert all("custom_settings" not in item["path"] for item in manifest["artifacts"])
    assert all("__pycache__" not in item["path"] for item in manifest["artifacts"])


@pytest.mark.parametrize("filename", ["notes.txt", "quiz_manifest.json"])
def test_manifest_build_rejects_noncanonical_extra_file(
    tmp_path: Path,
    filename: str,
) -> None:
    target = tmp_path / "target"
    publish_quiz_bundle(DEMO, [_question_source()], target)
    extra = next(target.glob("datasets/*/*/questions/*/q_*")) / filename
    extra.write_text("not canonical", encoding="utf-8")

    with pytest.raises(BundlePublishError, match="non-canonical artifact"):
        build_bundle_manifest(target)


def test_question_run_directory_is_accepted(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    run_path = question_dir.parent
    run = _read(run_path / "run.json")
    run["question_ids"] = [_read(question_dir / "question.json")["question_id"]]
    run["num_questions"] = 1
    _write(run_path / "run.json", run)

    target = tmp_path / "from_run"
    manifest = publish_quiz_bundle(source, [run_path], target)
    assert manifest["counts"]["questions"] == 1
    assert manifest["source_runs"][0]["partial"] is False


def test_release_and_question_versions_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = publish_quiz_bundle(
        DEMO, [_question_source()], first, generated_at="2026-07-12T00:00:00Z"
    )
    second_manifest = publish_quiz_bundle(
        DEMO, [_question_source()], second, generated_at="later"
    )

    assert first_manifest["release_id"] == second_manifest["release_id"]
    assert first_manifest["questions"] == second_manifest["questions"]
    assert first_manifest["questions"][0]["version"] == compute_question_version(
        _read(DEMO_QUESTION)
    )
    assert first_manifest["questions"][0]["version"] == feedback_question_version(
        _read(DEMO_QUESTION)
    )
    rebuilt = build_bundle_manifest(first, generated_at="another time")
    assert rebuilt["release_id"] == first_manifest["release_id"]


@pytest.mark.parametrize(
    "value",
    [
        (1 << 53),
        -(1 << 53),
        float(1 << 53),
        "\ud800",
    ],
)
def test_publisher_and_feedback_question_versions_reject_noninteroperable_json(
    value: Any,
) -> None:
    question = {"question_id": "q_noninteroperable", "value": value}

    with pytest.raises(QuestionVersionError):
        compute_question_version(question)
    with pytest.raises(FeedbackValidationError):
        feedback_question_version(question)


def test_complete_demo_bundle_passes_release_smoke_validation() -> None:
    manifest = build_bundle_manifest(DEMO)

    assert manifest["counts"]["questions"] == 60
    assert manifest["release_id"] == _read(DEMO / "quiz_manifest.json")["release_id"]


def test_complete_demo_bundle_has_strict_json_and_complete_choice_gt() -> None:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard constant {constant}")

    json_files = list(DEMO.rglob("*.json"))
    assert json_files
    for path in json_files:
        json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )

    choice_count = 0
    for question_path in DEMO.glob("datasets/*/*/questions/*/q_*/question.json"):
        question = _read(question_path)
        for choice in question["choices"]:
            choice_count += 1
            summary = _read(
                DEMO / choice["candidate_path"] / "results" / "summary.json"
            )
            assert summary["excluded"] is False
            assert summary["failed_seeds"] == 0
    assert choice_count == 180


def test_complete_demo_non_mse_prompts_do_not_claim_best_test_mse() -> None:
    checked = 0
    for question_path in DEMO.glob("datasets/*/*/questions/*/q_*/question.json"):
        question = _read(question_path)
        if question["evaluation"]["selection_metric"] == "test_mse":
            continue
        checked += 1
        prompt = (question_path.parent / "prompt.txt").read_text(encoding="utf-8")
        assert "best test MSE" not in prompt
    assert checked == 20


def test_dry_run_returns_projection_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "dry"
    projected = publish_quiz_bundle(DEMO, [_question_source()], target, dry_run=True)
    assert projected["counts"]["questions"] == 1
    assert not target.exists()

    existing = tmp_path / "existing"
    publish_quiz_bundle(DEMO, [_question_source()], existing)
    before = _byte_tree(existing)
    projected = publish_quiz_bundle(
        DEMO, [_other_question_source()], existing, dry_run=True
    )
    assert projected["counts"]["questions"] == 2
    assert _byte_tree(existing) == before


def test_copy_failure_restores_existing_target_byte_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "existing"
    publish_quiz_bundle(DEMO, [_question_source()], target)
    before = _byte_tree(target)
    original = publisher_module._install_staged_file
    calls = 0

    def fail_during_copy(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        original(*args, **kwargs)

    monkeypatch.setattr(publisher_module, "_install_staged_file", fail_during_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        publish_quiz_bundle(DEMO, [_other_question_source()], target)

    assert calls == 2
    assert _byte_tree(target) == before


def test_copy_failure_removes_new_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "new"
    original = publisher_module._install_staged_file
    calls = 0

    def fail_during_copy(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        original(*args, **kwargs)

    monkeypatch.setattr(publisher_module, "_install_staged_file", fail_during_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        publish_quiz_bundle(DEMO, [_question_source()], target)

    assert calls == 2
    assert not target.exists()


def test_final_validation_failure_restores_existing_target_byte_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "existing"
    publish_quiz_bundle(DEMO, [_question_source()], target)
    before = _byte_tree(target)
    original = publisher_module.build_bundle_manifest
    builds = 0

    def fail_final_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal builds
        builds += 1
        if builds == 2:
            raise OSError("injected final validation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        publisher_module, "build_bundle_manifest", fail_final_validation
    )

    with pytest.raises(OSError, match="injected final validation failure"):
        publish_quiz_bundle(DEMO, [_other_question_source()], target)

    assert builds == 2
    assert _byte_tree(target) == before


def test_manifest_failure_restores_existing_target_byte_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "existing"
    publish_quiz_bundle(DEMO, [_question_source()], target)
    before = _byte_tree(target)
    original = publisher_module._write_manifest_value

    def fail_after_manifest_write(*args: Any, **kwargs: Any) -> Path:
        original(*args, **kwargs)
        raise OSError("injected manifest failure")

    monkeypatch.setattr(
        publisher_module, "_write_manifest_value", fail_after_manifest_write
    )

    with pytest.raises(OSError, match="injected manifest failure"):
        publish_quiz_bundle(DEMO, [_other_question_source()], target)

    assert _byte_tree(target) == before


def test_manifest_failure_removes_new_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "new"
    original = publisher_module._write_manifest_value

    def fail_after_manifest_write(*args: Any, **kwargs: Any) -> Path:
        original(*args, **kwargs)
        raise OSError("injected manifest failure")

    monkeypatch.setattr(
        publisher_module, "_write_manifest_value", fail_after_manifest_write
    )

    with pytest.raises(OSError, match="injected manifest failure"):
        publish_quiz_bundle(DEMO, [_question_source()], target)

    assert not target.exists()


def test_formal_publish_uses_validated_staged_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    prompt = question_dir / "prompt.txt"
    validated_bytes = prompt.read_bytes()
    original = publisher_module.build_bundle_manifest
    builds = 0

    def mutate_source_after_projection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal builds
        manifest = original(*args, **kwargs)
        builds += 1
        if builds == 1:
            prompt.write_bytes(validated_bytes + b"\nsource changed after projection\n")
        return manifest

    monkeypatch.setattr(
        publisher_module, "build_bundle_manifest", mutate_source_after_projection
    )
    target = tmp_path / "target"

    publish_quiz_bundle(source, [question_dir], target)

    assert builds == 2
    assert (target / prompt.relative_to(source)).read_bytes() == validated_bytes
    assert prompt.read_bytes() != validated_bytes


def test_missing_candidate_artifact_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    candidate = source / question["choices"][0]["candidate_path"]
    (candidate / "results" / "summary.json").unlink()

    with pytest.raises(BundlePublishError, match="missing candidate artifact"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_ground_truth_excluded_choice_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    candidate = source / question["choices"][0]["candidate_path"]
    summary_path = candidate / "results" / "summary.json"
    summary = _read(summary_path)
    summary["excluded"] = True
    _write(summary_path, summary)

    with pytest.raises(BundlePublishError, match="excluded by ground truth"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_ground_truth_partial_seed_failure_choice_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    candidate = source / question["choices"][0]["candidate_path"]
    summary_path = candidate / "results" / "summary.json"
    summary = _read(summary_path)
    summary["failed_seeds"] = 1
    summary["seed_results"][0]["failed"] = True
    _write(summary_path, summary)

    with pytest.raises(BundlePublishError, match="partial seed failures"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_non_standard_numeric_constant_in_summary_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    summary_path = (
        source / question["choices"][0]["candidate_path"] / "results" / "summary.json"
    )
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8").replace(
            "{",
            '{"legacy_non_finite": Infinity,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(BundlePublishError, match="non-standard numeric constant"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_duplicate_json_object_key_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    raw = question_path.read_text(encoding="utf-8")
    question_path.write_text(
        raw.replace("{", '{"question_id":"q_ambiguous",', 1),
        encoding="utf-8",
    )

    with pytest.raises(BundlePublishError, match="duplicate JSON object key"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_non_mse_question_with_stale_mse_prompt_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    assert question["evaluation"]["selection_metric"] != "test_mse"
    prompt_path = question_dir / "prompt.txt"
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8")
        + "\nChoose the setup with the best test MSE.\n",
        encoding="utf-8",
    )

    with pytest.raises(BundlePublishError, match="non-test_mse"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_wrong_but_valid_correct_letter_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    question["correct_letter"] = next(
        choice["letter"]
        for choice in question["choices"]
        if choice["letter"] != question["correct_letter"]
    )
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match="ground-truth winner"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_metric_and_significance_gap_mismatches_are_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    question["significance"]["gap"] += 0.25
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match="significance gap"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "gap-target")

    question = _read(question_path)
    question["significance"]["gap"] -= 0.25
    question["evaluation"]["selection_metric"] = "different_metric"
    question["significance"]["metric"] = "different_metric"
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match="does not match dataset_spec"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "metric-target")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution", "parallel_logic", "candidate_py_files"),
        ("candidate_id", "c_wrong", "summary candidate_id"),
        ("n_seeds", 9, "seed configuration"),
        ("n_seeds", True, "positive integer"),
        ("base_seed", False, "summary.base_seed"),
        ("selection_metric", "wrong_metric", "selection_metric"),
    ],
)
def test_invalid_choice_summary_identity_is_rejected(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    summary_path = _choice_candidate(source, question) / "results" / "summary.json"
    summary = _read(summary_path)
    summary[field] = value
    _write(summary_path, summary)

    with pytest.raises(BundlePublishError, match=message):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_summary_seed_values_must_be_integers(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    summary_path = _choice_candidate(source, question) / "results" / "summary.json"
    summary = _read(summary_path)
    summary["seed_results"][0]["seed"] = False
    _write(summary_path, summary)

    with pytest.raises(BundlePublishError, match=r"seed_results\[0\].seed"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_candidate_set_and_question_budget_mismatches_are_rejected(
    tmp_path: Path,
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question = _read(question_dir / "question.json")
    candidate_spec_path = _choice_candidate(source, question) / "candidate_spec.json"
    candidate_spec = _read(candidate_spec_path)
    candidate_spec["budget"]["total_samples_seen"] += 1
    _write(candidate_spec_path, candidate_spec)

    with pytest.raises(BundlePublishError, match="training_steps × batch_size"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "candidate-target")

    candidate_spec["budget"]["total_samples_seen"] -= 1
    _write(candidate_spec_path, candidate_spec)
    set_path = source / question["choices"][0]["candidate_set_path"] / "set.json"
    set_spec = _read(set_path)
    set_spec["budget"]["total_samples_seen"] += 1
    _write(set_path, set_spec)

    with pytest.raises(BundlePublishError, match="budget does not match"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "set-target")

    set_spec["budget"]["total_samples_seen"] -= 1
    _write(set_path, set_spec)
    question["budget"]["total_samples_seen"] += 1
    _write(question_dir / "question.json", question)

    with pytest.raises(BundlePublishError, match="question budget"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "question-target")


@pytest.mark.parametrize(
    "leak",
    [
        '\n{"correct_letter": "B"}\n',
        "\nSee results/summary.json for details.\n",
        "\n**Correct answer:** Choice B\n",
        "\nChoice B is the winner.\n",
        "\n## Training Results\n",
        "\nMean test MSE: 0.43\n",
    ],
)
def test_prompt_ground_truth_leakage_is_rejected(
    tmp_path: Path,
    leak: str,
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    prompt_path = question_dir / "prompt.txt"
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + leak, encoding="utf-8"
    )

    with pytest.raises(BundlePublishError, match="prompt contains"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_unknown_metric_requires_explicit_direction(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    dataset_spec_path = (
        source
        / "datasets"
        / question["family"]
        / question["dataset_id"]
        / "dataset_spec.json"
    )
    dataset_spec = _read(dataset_spec_path)
    dataset_spec["selection_metric"] = "test_accuracy"
    question["evaluation"]["selection_metric"] = "test_accuracy"
    question["significance"]["metric"] = "test_accuracy"
    _write(dataset_spec_path, dataset_spec)
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match="must declare higher_is_better"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_metric_direction_contract_supports_legacy_and_explicit_metrics() -> None:
    question = {
        "evaluation": {
            "selection_metric": "test_mse",
            "n_seeds": 3,
            "base_seed": 10,
        }
    }
    dataset = {"selection_metric": "test_mse"}
    assert _question_evaluation(question, dataset) == ("test_mse", 3, 10, False)

    question["evaluation"]["selection_metric"] = "test_accuracy"
    dataset["selection_metric"] = "test_accuracy"
    question["evaluation"]["higher_is_better"] = True
    dataset["higher_is_better"] = True
    assert _question_evaluation(question, dataset) == (
        "test_accuracy",
        3,
        10,
        True,
    )

    dataset["higher_is_better"] = False
    with pytest.raises(BundlePublishError, match="disagree"):
        _question_evaluation(question, dataset)


def test_explicit_higher_is_better_publishes_maximum_summary_winner(
    tmp_path: Path,
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    old_metric = question["evaluation"]["selection_metric"]
    metric = "test_accuracy"

    dataset_spec_path = (
        source
        / "datasets"
        / question["family"]
        / question["dataset_id"]
        / "dataset_spec.json"
    )
    dataset_spec = _read(dataset_spec_path)
    dataset_spec["selection_metric"] = metric
    dataset_spec["higher_is_better"] = True
    _write(dataset_spec_path, dataset_spec)

    run = _read(question_dir.parent / "run.json")
    for set_reference in run["candidate_sets"]:
        for summary_path in sorted(
            (source / set_reference).glob("c_*/results/summary.json")
        ):
            summary = _read(summary_path)
            summary["selection_metric"] = metric
            summary[f"mean_{metric}"] = summary.pop(f"mean_{old_metric}")
            summary[f"std_{metric}"] = summary.pop(f"std_{old_metric}")
            for seed_result in summary["seed_results"]:
                seed_result[f"final_{metric}"] = seed_result.pop(f"final_{old_metric}")
            _write(summary_path, summary)

    question["evaluation"]["selection_metric"] = metric
    question["evaluation"]["higher_is_better"] = True
    question["significance"]["metric"] = metric
    choice_means = {
        choice["letter"]: _read(
            source / choice["candidate_path"] / "results" / "summary.json"
        )[f"mean_{metric}"]
        for choice in question["choices"]
    }
    ordered = sorted(choice_means.items(), key=lambda item: item[1], reverse=True)
    maximum_letter, maximum_mean = ordered[0]
    runner_up_mean = ordered[1][1]
    assert maximum_letter != question["correct_letter"]

    _write(question_path, question)
    with pytest.raises(BundlePublishError, match="ground-truth winner"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "wrong-target")

    question["correct_letter"] = maximum_letter
    question["significance"]["gap"] = maximum_mean - runner_up_mean
    _write(question_path, question)

    manifest = publish_quiz_bundle(source, [question_dir], tmp_path / "target")
    assert manifest["counts"]["questions"] == 1


def test_uniform_and_cross_budget_question_contract() -> None:
    _validate_question_budget(
        {"budget": {"total_samples_seen": 1024}},
        [1024, 1024],
    )
    _validate_question_budget(
        {"budget": {"total_samples_seen": [1024, 2048], "mixed": True}},
        [2048, 1024],
    )

    with pytest.raises(BundlePublishError, match="mixed must be true"):
        _validate_question_budget(
            {"budget": {"total_samples_seen": [1024, 2048]}},
            [1024, 2048],
        )
    with pytest.raises(BundlePublishError, match="sorted unique"):
        _validate_question_budget(
            {"budget": {"total_samples_seen": [2048, 1024], "mixed": True}},
            [1024, 2048],
        )


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    question["choices"][0]["candidate_path"] = "../outside/c_bad"
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match="path traversal"):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")


def test_duplicate_question_id_in_target_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    publish_quiz_bundle(DEMO, [_question_source()], target)

    with pytest.raises(BundlePublishError, match="duplicate question_id"):
        publish_quiz_bundle(DEMO, [_question_source()], target)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda question: question.__setitem__("correct_letter", "Z"),
            "correct_letter",
        ),
        (
            lambda question: question["choices"][0].__setitem__("excluded", True),
            "excluded",
        ),
        (
            lambda question: question["choices"][0].__setitem__(
                "candidate_id", "c_does_not_match"
            ),
            "ID/path mismatch",
        ),
    ],
)
def test_invalid_answer_excluded_choice_and_candidate_mismatch_are_rejected(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    source, question_dir = _isolated_source(tmp_path)
    question_path = question_dir / "question.json"
    question = _read(question_path)
    mutation(question)
    _write(question_path, question)

    with pytest.raises(BundlePublishError, match=message):
        publish_quiz_bundle(source, [question_dir], tmp_path / "target")
