from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tools.meta_model_study.external import (
    FAMILY_TO_EXPERIMENT,
    choice_to_example,
    load_prediction_inputs,
    score_predictions,
    write_unscored_predictions,
)


def test_external_module_does_not_eagerly_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import tools.meta_model_study.external; "
                "print('torch' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def _model(width: int = 8) -> dict:
    return {
        "type": "mlp",
        "input_dim": 3,
        "depth": 2,
        "width": width,
        "residual": True,
        "activations": ["relu", "gelu"],
        "layer_norm": [True, False],
    }


def _choice(letter: str, candidate_id: str, *, width: int = 8) -> dict:
    return {
        "letter": letter,
        "candidate_id": candidate_id,
        "budget": {
            "training_steps": 32,
            "batch_size": 32,
            "total_samples_seen": 1024,
        },
        "model": _model(width),
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
        "loss": {"loss_id": "mse"},
    }


def _questions() -> list[dict]:
    return [
        {
            "question_id": "q_one",
            "question_run_id": "run_test",
            "family": "multivariate_regression",
            "dataset_id": "mvar_c59a30",
            "selection_metric": "test_mse",
            "choices": [
                _choice("A", "c_a"),
                _choice("B", "c_b", width=16),
            ],
        }
    ]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_choice_example_uses_registry_parameter_count_and_preserves_rng() -> None:
    torch.manual_seed(1234)
    expected_next = torch.rand(4)
    torch.manual_seed(1234)

    example = choice_to_example(_choice("A", "c_a"))
    actual_next = torch.rand(4)

    assert example["setting"] == {
        key: _choice("A", "c_a")[key]
        for key in ("budget", "model", "optimizer", "loss")
    }
    assert example["derived"]["total_params"] == 201
    assert example["derived"]["trainable_params"] == 201
    assert example["derived"]["log_total_params"] == pytest.approx(
        math.log(201)
    )
    assert torch.equal(actual_next, expected_next)


def test_choice_example_can_exclude_parameter_count_without_touching_rng() -> None:
    torch.manual_seed(4321)
    expected_next = torch.rand(4)
    torch.manual_seed(4321)

    example = choice_to_example(
        _choice("A", "c_a"),
        include_parameter_count=False,
    )

    assert example["derived"] == {}
    assert torch.equal(torch.rand(4), expected_next)


def test_load_prediction_inputs_routes_family_and_never_resolves_candidates(
    tmp_path: Path,
) -> None:
    questions_path = tmp_path / "questions_sanitized.json"
    _write(questions_path, _questions())

    loaded = load_prediction_inputs(questions_path)

    assert len(loaded) == 1
    assert loaded[0]["experiment_id"] == FAMILY_TO_EXPERIMENT[
        "multivariate_regression"
    ]
    assert loaded[0]["question_id"] == "q_one"
    assert [choice["letter"] for choice in loaded[0]["choices"]] == ["A", "B"]
    assert set(loaded[0]["choices"][0]["example"]) == {"setting", "derived"}
    assert "target" not in loaded[0]["choices"][0]["example"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda questions: questions[0].update(correct_letter="A"), "non-blind"),
        (
            lambda questions: questions[0]["choices"][0].update(
                candidate_path="candidate/results/summary.json"
            ),
            "non-blind",
        ),
        (
            lambda questions: questions[0]["choices"][0]["budget"].update(
                total_samples_seen=999
            ),
            "violates",
        ),
    ],
)
def test_load_prediction_inputs_rejects_leaks_and_invalid_budget(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    questions = _questions()
    mutation(questions)
    questions_path = tmp_path / "questions_sanitized.json"
    _write(questions_path, questions)

    with pytest.raises(ValueError, match=message):
        load_prediction_inputs(questions_path)


def test_write_unscored_predictions_is_hashed_and_rejects_scored_fields(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "predictions.json"
    rows = [
        {
            "question_id": "q_one",
            "family": "multivariate_regression",
            "experiment_id": FAMILY_TO_EXPERIMENT["multivariate_regression"],
            "predicted_letter": "B",
            "predicted_candidate_id": "c_b",
            "choices": [
                {"letter": "A", "predicted_loss": 0.3},
                {"letter": "B", "predicted_loss": 0.2},
            ],
        }
    ]

    digest = write_unscored_predictions(
        output,
        rows,
        metadata={"method_id": "ridge"},
    )

    raw = output.read_bytes()
    payload = json.loads(raw)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert payload["num_questions"] == 1
    assert payload["metadata"] == {"method_id": "ridge"}
    assert payload["predictions"] == rows
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    contaminated = deepcopy(rows)
    contaminated[0]["correct_letter"] = "B"
    with pytest.raises(ValueError, match="scored/answer"):
        write_unscored_predictions(output, contaminated)
    assert output.read_bytes() == raw


def test_score_predictions_validates_set_and_reports_total_family_and_rows(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions = [
        {
            "question_id": "q_b",
            "family": "bigram_lm",
            "predicted_letter": "A",
            "predicted_candidate_id": "c_ba",
        },
        {
            "question_id": "q_m",
            "family": "multivariate_regression",
            "predicted_letter": "B",
            "predicted_candidate_id": "c_mb",
        },
    ]
    predictions_sha = write_unscored_predictions(predictions_path, predictions)
    answer_key_path = tmp_path / "answer_key.json"
    answer_key = [
        {
            "question_id": "q_b",
            "family": "bigram_lm",
            "correct_letter": "A",
            "choices": [
                {"letter": "A", "candidate_id": "c_ba"},
                {"letter": "B", "candidate_id": "c_bb"},
            ],
        },
        {
            "question_id": "q_m",
            "family": "multivariate_regression",
            "correct_letter": "A",
            "choices": [
                {"letter": "A", "candidate_id": "c_ma"},
                {"letter": "B", "candidate_id": "c_mb"},
            ],
        },
    ]
    _write(answer_key_path, answer_key)

    result = score_predictions(predictions_path, answer_key_path)

    assert result["predictions_sha256"] == predictions_sha
    assert result["total"] == {
        "num_questions": 2,
        "num_correct": 1,
        "accuracy": 0.5,
    }
    assert result["by_family"]["bigram_lm"]["accuracy"] == 1.0
    assert result["by_family"]["multivariate_regression"]["accuracy"] == 0.0
    assert [row["is_correct"] for row in result["questions"]] == [True, False]

    incomplete = [predictions[0]]
    write_unscored_predictions(predictions_path, incomplete)
    with pytest.raises(ValueError, match="question sets differ"):
        score_predictions(predictions_path, answer_key_path)


def test_scoring_rejects_letter_candidate_disagreement(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    write_unscored_predictions(
        predictions_path,
        [
            {
                "question_id": "q_one",
                "family": "multivariate_regression",
                "predicted_letter": "A",
                "predicted_candidate_id": "c_b",
            }
        ],
    )
    answer_key_path = tmp_path / "answer_key.json"
    _write(
        answer_key_path,
        [
            {
                "question_id": "q_one",
                "family": "multivariate_regression",
                "correct_letter": "A",
                "choices": [
                    {"letter": "A", "candidate_id": "c_a"},
                    {"letter": "B", "candidate_id": "c_b"},
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="letter/candidate mismatch"):
        score_predictions(predictions_path, answer_key_path)
