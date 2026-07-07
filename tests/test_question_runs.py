from __future__ import annotations

import json
from pathlib import Path

import pytest

from architecture_iq.questions.runs import make_run_name


def test_make_run_name_format() -> None:
    name = make_run_name(
        num_questions=5,
        num_choices=2,
        candidate_set_names=["set_1024_var_fix_fix_abc"],
        salt=0,
    )
    assert name.startswith("run_5q_2c_")


def test_list_questions_from_run_dir(tmp_path: Path) -> None:
    import sys

    tools = Path(__file__).resolve().parents[1] / "tools" / "llm_eval"
    sys.path.insert(0, str(tools))
    from question_loader import list_questions  # noqa: E402

    run_dir = tmp_path / "run_1q_2c_test"
    qdir = run_dir / "q_abc123"
    qdir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    (qdir / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q_abc123",
                "correct_letter": "A",
                "choices": [{"letter": "A"}, {"letter": "B"}],
                "prompt": {"rendered_path": "prompt.txt"},
            }
        ),
        encoding="utf-8",
    )
    (qdir / "prompt.txt").write_text("Pick A or B", encoding="utf-8")

    items = list_questions(run_dir)
    assert len(items) == 1
    assert items[0].question_id == "q_abc123"


def test_list_question_dirs_dataset_scoped(tmp_path: Path) -> None:
    import sys

    tools = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
    sys.path.insert(0, str(tools))
    from artifact_loader import list_question_dirs, resolve_data_root  # noqa: E402

    data_root = tmp_path / "data"
    qdir = (
        data_root
        / "datasets"
        / "univariate_regression"
        / "sym_test"
        / "questions"
        / "run_1q_2c_x"
        / "q_test"
    )
    qdir.mkdir(parents=True)
    (qdir / "question.json").write_text("{}", encoding="utf-8")

    found = list_question_dirs(data_root)
    assert found == [qdir.resolve()]
    assert resolve_data_root(qdir) == data_root.resolve()
