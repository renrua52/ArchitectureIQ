"""Integration contracts for recommendation-backed Inspector navigation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = ROOT / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import app as inspector_app  # noqa: E402
import release_manifest  # noqa: E402


class _SessionState(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _manifest() -> release_manifest.QuizManifest:
    manifest = release_manifest.load_quiz_manifest(
        ROOT / "examples" / "quiz_demo" / "bundle"
    )
    assert manifest is not None
    return manifest


def _state() -> _SessionState:
    return _SessionState(
        quiz_answers={},
        quiz_attempt_id="attempt_navigation",
        feedback_trace=inspector_app.feedback.SessionTrace("anon_navigation"),
    )


def test_recommended_next_is_deterministic_attested_and_not_current(
    monkeypatch: Any,
) -> None:
    manifest = _manifest()
    current = manifest.question_dirs()[0]
    state = _state()
    monkeypatch.setattr(inspector_app.st, "session_state", state)

    first_path, first = inspector_app._recommended_next_path(
        manifest,
        current,
        seed=0,
    )
    second_path, second = inspector_app._recommended_next_path(
        manifest,
        current,
        seed=0,
    )

    assert first == second
    assert first_path == second_path
    assert first_path.resolve() in {path.resolve() for path in manifest.question_dirs()}
    assert first_path.resolve() != current.resolve()
    assert first.question.release_id == manifest.release_id


def test_recommended_next_excludes_answered_question_version(monkeypatch: Any) -> None:
    manifest = _manifest()
    current = manifest.question_dirs()[0]
    state = _state()
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    first_path, first = inspector_app._recommended_next_path(
        manifest,
        current,
        seed=0,
    )

    state.quiz_answers[first.question.question_version] = {
        "question_id": first.question.question_id,
        "selected_letter": "A",
        "is_correct": False,
    }
    next_path, next_recommendation = inspector_app._recommended_next_path(
        manifest,
        current,
        seed=0,
    )

    assert next_recommendation.question != first.question
    assert next_path.resolve() != first_path.resolve()


def test_recommended_next_avoids_current_family_when_others_exist(
    monkeypatch: Any,
) -> None:
    manifest = _manifest()
    current_record = manifest.questions[0]
    current = manifest.question_dirs()[0]
    monkeypatch.setattr(
        inspector_app.st,
        "session_state",
        _state(),
    )

    _path, recommendation = inspector_app._recommended_next_path(
        manifest,
        current,
        seed=0,
    )
    selected = next(
        record
        for record in manifest.questions
        if record.question_id == recommendation.question.question_id
        and record.version == recommendation.question.question_version
    )

    assert len({record.family for record in manifest.questions}) > 1
    assert selected.family != current_record.family
