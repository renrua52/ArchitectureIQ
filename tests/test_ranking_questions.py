from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "ranking_questions"
sys.path.insert(0, str(TOOLS))

import generate  # noqa: E402
from common import count_inversions, max_inversions, write_json  # noqa: E402
from make_blind_bundle import make_blind_bundle  # noqa: E402
from score_answers import score_answers  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEMO_SET = (
    REPO
    / "examples"
    / "quiz_demo"
    / "bundle"
    / "datasets"
    / "univariate_regression"
    / "sym_62678b"
    / "candidates"
    / "set_2048_var_var_fix_696edb"
)


def test_count_inversions_perfect_order() -> None:
    assert count_inversions(["T1", "T2", "T3"], ["T1", "T2", "T3"]) == 0


def test_count_inversions_reversed_order() -> None:
    assert count_inversions(["T3", "T2", "T1"], ["T1", "T2", "T3"]) == 3
    assert max_inversions(3) == 3


def test_count_inversions_partial_order() -> None:
    assert count_inversions(["T2", "T1", "T3", "T4"], ["T1", "T2", "T3", "T4"]) == 1


def test_count_inversions_rejects_duplicate_or_missing_labels() -> None:
    with pytest.raises(ValueError, match="exactly the true labels"):
        count_inversions(["T1", "T1", "T3"], ["T1", "T2", "T3"])


def test_ranking_generation_blind_bundle_and_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "ranking"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            str(DEMO_SET),
            "--output",
            str(output_root),
            "--run-name",
            "smoke",
            "--num-questions",
            "1",
            "--calibration-size",
            "1",
            "--target-size",
            "2",
            "--max-candidates",
            "3",
        ],
    )

    assert generate.main() == 0
    run_dir = output_root / "smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_questions"] == 1
    assert (run_dir / "index.html").is_file()
    assert (run_dir / "llm_eval" / "README.json").is_file()

    blind_dir = tmp_path / "blind"
    private_key = make_blind_bundle(
        run_dir,
        blind_dir,
        seed=0,
        omit_calibration=False,
    )
    assert (blind_dir / "manifest.json").is_file()
    assert not (blind_dir / "answer_key.json").exists()

    private_key_path = tmp_path / "private_key.json"
    answers_path = tmp_path / "answers.json"
    write_json(private_key_path, private_key)
    write_json(answers_path, private_key)
    result = score_answers(private_key_path, answers_path)
    assert result["normalized_score"] == 1.0


def test_blind_bundle_rejects_output_inside_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"question_ids": []})
    write_json(run_dir / "answer_key.json", {"questions": {}})

    with pytest.raises(ValueError, match="outside"):
        make_blind_bundle(
            run_dir,
            run_dir / "blind",
            seed=0,
            omit_calibration=True,
        )
