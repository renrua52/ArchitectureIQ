from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import architecture_iq.manifest as manifest
from architecture_iq.candidates.axes import choices_compatible
from architecture_iq.candidates.sets import (
    list_candidate_sets,
    list_candidates_in_set,
)
from architecture_iq.paths import (
    candidate_dir,
    candidate_in_set_dir,
    candidate_set_dir,
    dataset_dir,
    question_dir,
)
from architecture_iq.util import read_json, short_hash, write_json

_AGGREGATOR_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "quiz_attempt_aggregator.py"
)
_AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "quiz_attempt_aggregator", _AGGREGATOR_PATH
)
assert _AGGREGATOR_SPEC is not None
assert _AGGREGATOR_SPEC.loader is not None
_AGGREGATOR = importlib.util.module_from_spec(_AGGREGATOR_SPEC)
_AGGREGATOR_SPEC.loader.exec_module(_AGGREGATOR)

_SEQ_AGGREGATOR_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "sequential_summary_aggregator.py"
)
_SEQ_AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "sequential_summary_aggregator", _SEQ_AGGREGATOR_PATH
)
assert _SEQ_AGGREGATOR_SPEC is not None
assert _SEQ_AGGREGATOR_SPEC.loader is not None
_SEQ_AGGREGATOR = importlib.util.module_from_spec(_SEQ_AGGREGATOR_SPEC)
_SEQ_AGGREGATOR_SPEC.loader.exec_module(_SEQ_AGGREGATOR)

_SESSION_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "sequential_feedback_session.py"
)
_SESSION_SPEC = importlib.util.spec_from_file_location(
    "sequential_feedback_session", _SESSION_PATH
)
assert _SESSION_SPEC is not None
assert _SESSION_SPEC.loader is not None
_SESSION = importlib.util.module_from_spec(_SESSION_SPEC)
_SESSION_SPEC.loader.exec_module(_SESSION)

_START_QUIZ_PATH = Path(__file__).resolve().parents[1] / "tools" / "start_quiz.py"
_START_QUIZ_SPEC = importlib.util.spec_from_file_location("start_quiz", _START_QUIZ_PATH)
assert _START_QUIZ_SPEC is not None
assert _START_QUIZ_SPEC.loader is not None
_START_QUIZ = importlib.util.module_from_spec(_START_QUIZ_SPEC)
_START_QUIZ_SPEC.loader.exec_module(_START_QUIZ)


def test_short_hash_is_stable_for_dict_order() -> None:
    left = {"b": 2, "a": [1, Path("x")]}
    right = {"a": [1, Path("x")], "b": 2}
    assert short_hash(left) == short_hash(right)
    assert len(short_hash(left)) == 6


def test_write_json_round_trip_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    payload = {"schema_version": "1.0", "items": [1, 2, 3]}

    write_json(target, payload)

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert read_json(target) == payload


def test_start_quiz_materializes_bundled_demo(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo / _START_QUIZ.BUNDLED_DEMO_DATA,
        tmp_path / _START_QUIZ.BUNDLED_DEMO_DATA,
    )

    question_run, installed = _START_QUIZ.resolve_question_run(
        tmp_path,
        _START_QUIZ.DEFAULT_RUN,
    )

    assert installed is True
    assert (question_run / "question.json").is_file()
    assert (tmp_path / "data" / "datasets").is_dir()


def test_start_quiz_treats_keyboard_interrupt_as_clean_shutdown() -> None:
    class InterruptingProcess:
        terminated = False

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None and not self.terminated:
                raise KeyboardInterrupt
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("a responsive process should not be killed")

    process = InterruptingProcess()

    assert _START_QUIZ.wait_for_process(process) == 0
    assert process.terminated is True


def test_path_helpers_encode_current_and_legacy_layouts() -> None:
    dataset_path = dataset_dir("univariate_regression", "sym_test")
    set_path = candidate_set_dir(dataset_path, "set_1024_var_fix_fix_abc123")

    assert dataset_path.parts[-3:] == ("datasets", "univariate_regression", "sym_test")
    assert set_path == dataset_path / "candidates" / "set_1024_var_fix_fix_abc123"
    assert candidate_in_set_dir(set_path, "c_001") == set_path / "c_001"
    assert candidate_dir("univariate_regression", "sym_test", 1024, "c_001").parts[-2:] == (
        "budget_1024",
        "c_001",
    )
    assert question_dir("q_001").parts[-2:] == ("questions", "q_001")


def test_write_benchmark_manifest_includes_runtime_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    monkeypatch.setattr(manifest, "git_commit_hash", lambda root: "abc123")

    manifest.write_benchmark_manifest("v1", {"run_id": "smoke"})

    payload = read_json(tmp_path / "benchmark_manifest.json")
    assert payload["profile"] == "v1"
    assert payload["git_commit"] == "abc123"
    assert payload["run_id"] == "smoke"
    assert payload["python"]
    assert payload["platform"]
    assert payload["torch"]


def test_list_candidate_sets_and_complete_candidates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "datasets" / "univariate_regression" / "sym_test"
    complete_set = dataset_path / "candidates" / "set_complete"
    incomplete_set = dataset_path / "candidates" / "set_incomplete"
    complete_candidate = complete_set / "c_good"
    missing_results_candidate = complete_set / "c_missing_results"

    write_json(complete_set / "set.json", {"set_id": "set_complete"})
    write_json(incomplete_set / "set.json", {"set_id": "set_incomplete"})
    write_json(complete_candidate / "candidate_spec.json", {"candidate_id": "c_good"})
    write_json(complete_candidate / "results" / "summary.json", {"excluded": False})
    write_json(
        missing_results_candidate / "candidate_spec.json",
        {"candidate_id": "c_missing_results"},
    )
    (dataset_path / "candidates" / "not_a_set").mkdir(parents=True)

    assert list_candidate_sets(dataset_path) == [
        complete_set.resolve(),
        incomplete_set.resolve(),
    ]
    assert list_candidates_in_set(complete_set) == [complete_candidate.resolve()]
    assert list_candidates_in_set(tmp_path / "missing") == []


def test_choices_compatible_rejects_empty_identical_and_unknown_type() -> None:
    spec = {
        "budget": {"batch_size": 16},
        "model": {"depth": 2},
        "optimizer": {"type": "Adam"},
        "loss": {"loss_id": "mse"},
    }
    varied_model = {
        **spec,
        "model": {"depth": 3},
    }

    assert not choices_compatible([])
    assert not choices_compatible([spec, spec])
    assert choices_compatible([spec, varied_model], "architecture_only")

    try:
        choices_compatible([spec, varied_model], "unknown")
    except ValueError as exc:
        assert "Unknown question type" in str(exc)
    else:
        raise AssertionError("choices_compatible should reject unknown question types")


def test_quiz_attempt_aggregator_scores_prediction_wrappers(tmp_path: Path) -> None:
    answer_key = _AGGREGATOR.build_key(
        [
            {
                "question_id": "q1",
                "family": "demo",
                "correct_letter": "B",
                "choices": [
                    {"letter": "A", "candidate_id": "c_a"},
                    {"letter": "B", "candidate_id": "c_b"},
                ],
            },
            {
                "question_id": "q2",
                "family": "demo",
                "correct_letter": "A",
                "choices": [
                    {"letter": "A", "candidate_id": "c_x"},
                    {"letter": "B", "candidate_id": "c_y"},
                ],
            },
        ]
    )
    attempt_path = tmp_path / "attempt.json"
    write_json(
        attempt_path,
        {
            "predictions": [
                {"question_id": "q1", "predicted_candidate_id": "c_b"},
                {"question_id": "q2", "predicted_letter": "B"},
            ]
        },
    )

    scored = _AGGREGATOR.score_attempt(attempt_path, answer_key)

    assert scored["correct"] == 1
    assert scored["total"] == 2
    assert scored["missing"] == []
    assert scored["invalid"] == []


def test_sequential_summary_aggregator_accepts_agent_summary_variants(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(
        first,
        {
            "experiment_name": "first",
            "correct_count": 8,
            "total_questions": 10,
            "overall_accuracy": 0.8,
            "block_accuracy": [
                {"block": "1-10", "correct": 8, "total": 10, "accuracy": 0.8}
            ],
        },
    )
    write_json(
        second,
        {
            "experiment_name": "second",
            "total_correct": 6,
            "total_questions": 10,
            "overall_accuracy": 0.6,
            "block_accuracy": [
                {"block": "1-10", "correct": 6, "total": 10, "accuracy": 0.6}
            ],
        },
    )

    report = _SEQ_AGGREGATOR.aggregate([first, second])

    assert report["overall"]["n"] == 2
    assert report["overall"]["mean_accuracy"] == 0.7
    assert report["by_block"][0]["block"] == "1-10"
    assert report["by_block"][0]["mean_accuracy"] == 0.7


def test_sequential_feedback_session_records_prediction_before_feedback(
    tmp_path: Path,
) -> None:
    questions_path = tmp_path / "questions.json"
    feedback_path = tmp_path / "feedback.json"
    session_path = tmp_path / "session.json"
    summary_path = tmp_path / "summary.json"
    write_json(
        questions_path,
        [
            {
                "question_id": "q1",
                "family": "demo",
                "choices": [
                    {"letter": "A", "candidate_id": "c_a"},
                    {"letter": "B", "candidate_id": "c_b"},
                ],
            }
        ],
    )
    write_json(
        feedback_path,
        [
            {
                "n": 1,
                "question_id": "q1",
                "family": "demo",
                "correct_letter": "B",
                "metric": "loss",
                "choice_mean_metrics": {"A": 2.0, "B": 1.0},
            }
        ],
    )

    _SESSION.init_session(
        session_path,
        questions_path,
        feedback_path,
        "smoke",
    )
    current = _SESSION.current_question(session_path)
    answer_result = _SESSION.submit_answer(
        session_path,
        "A",
        "c_a",
        0.4,
        "A looked simpler.",
    )
    lesson_result = _SESSION.record_lesson(session_path, "B had lower loss.")
    summary = _SESSION.write_summary(session_path, summary_path)

    assert current["done"] is False
    assert current["question"]["question_id"] == "q1"
    assert "feedback" not in current
    assert answer_result["recorded_prediction"]["predicted_letter"] == "A"
    assert answer_result["feedback"]["correct_letter"] == "B"
    assert lesson_result["lesson"] == "B had lower loss."
    assert summary["correct_count"] == 0
    assert summary["answered_questions"] == 1
    assert read_json(summary_path)["final_lessons"] == ["B had lower loss."]
