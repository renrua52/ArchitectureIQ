from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tools.meta_model_study.heuristics import (
    COMPONENT_NAMES,
    CalibratedFormula,
    component_matrix,
    cross_validate_formulas,
    fit_study,
    fixed_formula,
    formula_components,
    predict_external,
)


FAMILY = "multivariate_regression"
EXPERIMENT_ID = "multivariate_test"


def _example(
    *,
    total_params: int = 10_000,
    optimizer: str = "Adam",
    lr: float = 0.003,
    width: int = 64,
    depth: int = 3,
    residual: bool = False,
) -> dict:
    return {
        "setting": {
            "model": {
                "type": "mlp",
                "input_dim": 4,
                "width": width,
                "depth": depth,
                "residual": residual,
                "activations": ["relu"] * depth,
                "layer_norm": [False] * depth,
            },
            "optimizer": {
                "type": optimizer,
                "lr": lr,
                "weight_decay": 0.0001,
            },
            "loss": {"loss_id": "mse"},
            "budget": {
                "batch_size": 32,
                "training_steps": 160,
                "total_samples_seen": 5120,
            },
        },
        "derived": {
            "total_params": total_params,
            "trainable_params": total_params,
            "log_total_params": math.log(total_params),
        },
    }


def _row(index: int, *, split: str) -> dict:
    optimizer = "Adam" if index % 2 == 0 else "SGD"
    lr = (0.0003, 0.001, 0.003, 0.01)[index % 4]
    width = (16, 32, 64, 128)[index % 4]
    depth = 1 + index % 5
    example = _example(
        total_params=500 + width * width * depth,
        optimizer=optimizer,
        lr=lr,
        width=width,
        depth=depth,
        residual=index % 3 == 0,
    )
    score = fixed_formula(FAMILY).predict([example])[0]
    log_loss = 0.2 + 1.7 * score + 0.01 * (index % 3)
    return {
        "experiment_id": EXPERIMENT_ID,
        "family": FAMILY,
        "dataset_id": "mvar_test",
        "split": split,
        "stratum": optimizer,
        "usable_for_regression": True,
        "example_fingerprint_sha256": f"{index:064x}",
        **example,
        "target": {
            "selection_metric": "test_mse",
            "mean_loss": math.exp(log_loss),
            "log_mean_loss": log_loss,
            "benchmark_eligible": True,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_fixed_formula_is_target_free_and_has_declared_directions() -> None:
    base = _example(total_params=10_000, lr=0.003, residual=False)
    with_fake_target = {**base, "target": {"mean_loss": 10**100}}
    assert formula_components(base, FAMILY) == formula_components(
        with_fake_target, FAMILY
    )

    larger = _example(total_params=100_000, lr=0.003, residual=False)
    bad_lr = _example(total_params=10_000, lr=0.00001, residual=False)
    residual = _example(total_params=10_000, lr=0.003, residual=True)
    model = fixed_formula(FAMILY)
    assert model.predict([larger])[0] < model.predict([base])[0]
    assert model.predict([bad_lr])[0] > model.predict([base])[0]
    assert model.predict([residual])[0] < model.predict([base])[0]
    assert tuple(formula_components(base, FAMILY)) == COMPONENT_NAMES


def test_transformer_components_include_shape_and_optimizer_lr_quadratic() -> None:
    example = {
        "setting": {
            "model": {
                "type": "transformer_lm",
                "context_length": 16,
                "vocab_size": 32,
                "d_model": 64,
                "d_ff": 256,
                "num_heads": 2,
                "num_layers": 2,
            },
            "optimizer": {
                "type": "Adam",
                "lr": 0.001,
                "weight_decay": 0.0001,
            },
        },
        "derived": {"total_params": 50_000},
    }
    components = formula_components(example, "bigram_lm")
    assert components["optimizer_lr_quadratic"] == 0.0
    assert components["architecture_shape"] == 0.0

    example["setting"]["model"]["num_heads"] = 8
    example["setting"]["optimizer"]["lr"] = 0.01
    components = formula_components(example, "bigram_lm")
    assert components["optimizer_lr_quadratic"] > 0.0
    assert components["architecture_shape"] > 0.0


def test_cv_calibrations_preserve_constraints_and_select_ridge_on_train_only() -> None:
    rows = [_row(index, split="train") for index in range(18)]
    models, results = cross_validate_formulas(
        rows,
        FAMILY,
        n_splits=3,
        seed=7,
        alpha_grid=(0.01, 1.0),
    )

    assert set(models) == {
        "fixed_zero_shot",
        "positive_affine",
        "component_nnls",
        "component_ridge",
    }
    assert models["positive_affine"].affine_slope is not None
    assert models["positive_affine"].affine_slope >= 0.0
    assert min(models["component_nnls"].coefficients) >= -1e-12
    assert models["component_ridge"].alpha in {0.01, 1.0}
    assert results["component_ridge"]["selected_alpha"] in {0.01, 1.0}
    assert len(results["component_ridge"]["oof_predictions"]) == len(rows)

    matrix = component_matrix(rows, FAMILY)
    zero_shot = models["fixed_zero_shot"].predict_components(matrix)
    affine = models["positive_affine"].predict_components(matrix)
    assert np.array_equal(np.argsort(zero_shot), np.argsort(affine))


def test_fit_and_blind_external_prediction_write_auditable_artifacts(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    experiment_dir = dataset_root / EXPERIMENT_ID
    train_rows = [_row(index, split="train") for index in range(18)]
    validation_rows = [
        _row(100 + index, split="validation") for index in range(6)
    ]
    _write_jsonl(experiment_dir / "train.jsonl", train_rows)
    _write_jsonl(experiment_dir / "validation.jsonl", validation_rows)
    (experiment_dir / "manifest.json").write_text(
        json.dumps({"experiment_id": EXPERIMENT_ID}) + "\n", encoding="utf-8"
    )

    artifact_dir = tmp_path / "artifacts"
    summary = fit_study(
        dataset_root,
        artifact_dir,
        n_splits=3,
        seed=11,
        alpha_grid=(0.01, 1.0),
    )
    assert summary["num_experiments"] == 1
    assert summary["experiments"][0]["selected_method"] in {
        "fixed_zero_shot",
        "positive_affine",
        "component_nnls",
        "component_ridge",
    }
    experiment_artifact = json.loads(
        (artifact_dir / "experiments" / f"{EXPERIMENT_ID}.json").read_text()
    )
    assert experiment_artifact["selection_protocol"][
        "validation_used_for_selection"
    ] is False
    for method in experiment_artifact["models"].values():
        restored = CalibratedFormula.from_json(method)
        assert np.all(np.isfinite(restored.predict(validation_rows)))

    choices = []
    for letter, row in zip("ABC", validation_rows[:3]):
        choices.append(
            {
                "letter": letter,
                "candidate_id": f"candidate_{letter}",
                "example": {
                    "setting": row["setting"],
                    "derived": row["derived"],
                },
            }
        )
    prepared_path = tmp_path / "prepared_inputs.json"
    prepared_path.write_text(
        json.dumps(
            {
                "schema_version": "meta_model_prepared_external_inputs_v1",
                "num_questions": 1,
                "questions": [
                    {
                        "question_id": "q_test",
                        "family": FAMILY,
                        "experiment_id": EXPERIMENT_ID,
                        "choices": choices,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "heuristic_formula_v2.json"
    result = predict_external(prepared_path, artifact_dir, output_path)
    payload = json.loads(output_path.read_text())
    assert result["num_questions"] == 1
    assert result["answer_key_opened"] is False
    assert payload["metadata"]["answer_key_opened"] is False
    assert payload["predictions"][0]["predicted_letter"] in "ABC"
    assert "correct_letter" not in output_path.read_text()
