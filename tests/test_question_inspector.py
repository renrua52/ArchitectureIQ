"""Tests for the standalone question inspector loader."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import from tools directory without installing a package.
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))

from artifact_loader import (  # noqa: E402
    format_metrics,
    list_question_dirs,
    load_question_bundle,
    load_dataset_tensors,
    question_label,
)
from candidate_curves import all_step_samples, load_candidate_curves, reconstruct_eval_samples  # noqa: E402
from expression_latex import expression_to_latex  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


@pytest.fixture
def question_path() -> Path | None:
    questions = sorted((DATA / "questions").glob("q_*/question.json"))
    if not questions:
        return None
    return questions[0].parent


def test_expression_to_latex() -> None:
    latex = expression_to_latex("cos(6.283185307179586*(0.8333 + x)) + (x - x + x / -2)")
    assert r"\cos" in latex
    assert "x" in latex


def test_expression_to_latex_powers() -> None:
    latex = expression_to_latex("sin(6.283185307179586*x) + (x)**2")
    assert r"\sin" in latex
    assert "2" in latex


def test_expression_to_latex_multivariate_subscripts() -> None:
    latex = expression_to_latex("x0 + sin(6.283185307179586*x1) * x2")
    assert "x_{0}" in latex or r"x_0" in latex
    assert "x_{1}" in latex or r"x_1" in latex
    assert "x_{2}" in latex or r"x_2" in latex
    assert "3 x" not in latex


@pytest.mark.skipif(not (DATA / "questions").is_dir(), reason="no generated questions")
def test_load_question_bundle(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    assert bundle.question["question_id"].startswith("q_")
    assert len(bundle.choices) == bundle.question["num_choices"]
    for choice in bundle.choices:
        assert choice["candidate_dir"].is_dir()
        assert (choice["candidate_dir"] / "candidate_spec.json").is_file()


@pytest.mark.skipif(not (DATA / "questions").is_dir(), reason="no generated questions")
def test_load_dataset_tensors(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    tx, ty, vx, vy = load_dataset_tensors(bundle.dataset_dir)
    assert tx.ndim == 2 and ty.shape == tx.shape
    assert vx.shape == tx.shape and vy.shape == ty.shape


def test_format_metrics() -> None:
    text = format_metrics({"selection_metric": "test_mse", "mean_test_mse": 0.1, "std_test_mse": 0.02})
    assert "0.100000" in text
    assert "0.020000" in text


def test_all_step_samples() -> None:
    assert all_step_samples(2048, 32) == [32 * step for step in range(1, 65)]
    assert all_step_samples(1024, 64) == [64 * step for step in range(1, 17)]


def test_reconstruct_eval_samples_sparse() -> None:
    assert reconstruct_eval_samples(2048, 32, 64) == [
        32,
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        576,
        640,
        704,
        768,
        832,
        896,
        960,
        1024,
        1088,
        1152,
        1216,
        1280,
        1344,
        1408,
        1472,
        1536,
        1600,
        1664,
        1728,
        1792,
        1856,
        1920,
        1984,
        2048,
    ]
    assert reconstruct_eval_samples(1024, 64, 64) == [
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        576,
        640,
        704,
        768,
        832,
        896,
        960,
        1024,
    ]


@pytest.mark.skipif(not (DATA / "questions").is_dir(), reason="no generated questions")
def test_load_candidate_curves(question_path: Path) -> None:
    bundle = load_question_bundle(question_path, DATA)
    choice = bundle.choices[0]
    curves_path = choice["candidate_dir"] / "results" / "curves.npz"
    if not curves_path.is_file():
        pytest.skip("no curves.npz for candidate")
    spec_path = choice["candidate_dir"] / "candidate_spec.json"
    import json

    spec = json.loads(spec_path.read_text())
    budget = spec["budget"]
    loaded = load_candidate_curves(
        curves_path,
        total_samples_seen=int(budget["total_samples_seen"]),
        batch_size=int(budget["batch_size"]),
    )
    assert "error" not in loaded
    assert loaded["curves"].ndim == 2
    assert len(loaded["eval_samples"]) == loaded["curves"].shape[1]


@pytest.mark.skipif(not (DATA / "questions").is_dir(), reason="no generated questions")
def test_list_question_dirs() -> None:
    pool = list_question_dirs(DATA)
    assert pool
    assert all(p.name.startswith("q_") for p in pool)
    assert all((p / "question.json").is_file() for p in pool)


@pytest.mark.skipif(not (DATA / "questions").is_dir(), reason="no generated questions")
def test_question_label(question_path: Path) -> None:
    label = question_label(question_path)
    assert question_path.name in label
    assert " · " in label
