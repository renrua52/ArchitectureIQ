from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = REPO / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORDER_ANALYSIS = load_tool("analyze_order_parameters")
ARITHMETIC_RULES = load_tool("evaluate_arithmetic_rules")
SINGLE_BLIND = load_tool("make_single_question_blind_quiz")


def test_spearman_uses_average_ranks_for_ties() -> None:
    x = np.asarray([1.0, 1.0, 2.0, 3.0])
    y = np.asarray([1.0, 2.0, 2.0, 3.0])
    expected = float(np.corrcoef([0.5, 0.5, 2.0, 3.0], [0.0, 1.5, 1.5, 3.0])[0, 1])

    assert ORDER_ANALYSIS.spearman(x, y) == pytest.approx(expected)


def test_choice_features_accepts_empty_layer_norm() -> None:
    choice = {
        "candidate_id": "c_test",
        "model": {
            "type": "mlp",
            "depth": 1,
            "width": 8,
            "input_dim": 1,
            "layer_norm": [],
        },
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
        "loss": {"loss_id": "mse"},
    }

    features = ARITHMETIC_RULES.choice_features(choice)

    assert features["layer_norm_frac"] == 0.0
    assert features["params"] > 0


def test_single_question_bundle_is_safe_and_portable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    questions = [
        {
            "question_id": "q_test",
            "family": "univariate_regression",
            "question_type": "mixed",
            "selection_metric": "test_mse",
            "dataset_params": {"expression": "x"},
            "choices": [
                {"letter": "A", "candidate_id": "c_a", "model": {"depth": 1}},
                {"letter": "B", "candidate_id": "c_b", "model": {"depth": 2}},
            ],
        }
    ]
    answers = [{"question_id": "q_test", "correct_letter": "B"}]
    (source / "questions_sanitized.json").write_text(
        json.dumps(questions), encoding="utf-8"
    )
    (source / "answer_key.json").write_text(json.dumps(answers), encoding="utf-8")
    output = tmp_path / "public"
    private_key = tmp_path / "private_key.json"

    SINGLE_BLIND.make_bundle(source, output, private_key)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "source"
    assert not (output / "answer_key.json").exists()
    assert json.loads(private_key.read_text(encoding="utf-8"))["answers"] == {"q01": "B"}

    with pytest.raises(FileExistsError, match="--force"):
        SINGLE_BLIND.make_bundle(source, output, private_key)
    with pytest.raises(ValueError, match="outside the public blind bundle"):
        SINGLE_BLIND.make_bundle(source, tmp_path / "other", tmp_path / "other" / "key.json")
