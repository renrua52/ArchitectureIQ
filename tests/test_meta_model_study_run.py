from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.meta_model_study import run as study_run
from tools.meta_model_study.external import FAMILY_TO_EXPERIMENT
from tools.meta_model_study.models import SearchDefinition, search_definitions


EXPERIMENT_ID = FAMILY_TO_EXPERIMENT["multivariate_regression"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _model(width: int) -> dict:
    return {
        "type": "mlp",
        "input_dim": 3,
        "depth": 2,
        "width": width,
        "residual": True,
        "activations": ["relu", "gelu"],
        "layer_norm": [True, False],
    }


def _row(
    experiment_id: str,
    index: int,
    *,
    split: str,
    stratum: str,
    fingerprint_index: int | None = None,
) -> dict:
    total_params = 100 + 20 * index
    mean_loss = 1.0 / total_params
    fingerprint_value = index if fingerprint_index is None else fingerprint_index
    return {
        "experiment_id": experiment_id,
        "family": "multivariate_regression",
        "dataset_id": "mvar_test",
        "split": split,
        "stratum": stratum,
        "usable_for_regression": True,
        "example_fingerprint_sha256": f"{fingerprint_value:064x}",
        "setting": {
            "model": _model(8 + index),
            "optimizer": {
                "type": "Adam" if stratum == "adam" else "SGD",
                "lr": 0.001 if stratum == "adam" else 0.01,
                "weight_decay": 0.0,
            },
            "loss": {"loss_id": "mse"},
            "budget": {
                "batch_size": 32,
                "training_steps": 32,
                "total_samples_seen": 1024,
            },
        },
        "derived": {
            "total_params": total_params,
            "trainable_params": total_params,
            "log_total_params": math.log(total_params),
        },
        "target": {
            "selection_metric": "test_mse",
            "mean_loss": mean_loss,
            "log_mean_loss": math.log(mean_loss),
            "benchmark_eligible": index % 3 != 0,
        },
        "provenance": {"candidate_path": f"candidate_{index}"},
    }


def _experiment_rows(experiment_id: str) -> tuple[list[dict], list[dict]]:
    train = [
        _row(
            experiment_id,
            index,
            split="train",
            stratum="adam" if index % 2 == 0 else "sgd",
        )
        for index in range(8)
    ]
    validation = [
        _row(
            experiment_id,
            20 + index,
            split="validation",
            stratum="adam" if index % 2 == 0 else "sgd",
        )
        for index in range(4)
    ]
    return train, validation


def _write_experiment(root: Path, experiment_id: str) -> Path:
    experiment_dir = root / experiment_id
    train, validation = _experiment_rows(experiment_id)
    _write_jsonl(experiment_dir / "train.jsonl", train)
    _write_jsonl(experiment_dir / "validation.jsonl", validation)
    _write_json(
        experiment_dir / "manifest.json",
        {"experiment_id": experiment_id, "config_sha256": "a" * 64},
    )
    return experiment_dir


def test_discover_and_load_experiments_audit_completed_split_contract(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    second = _write_experiment(dataset_root, "z_experiment")
    first = _write_experiment(dataset_root, "a_experiment")
    incomplete = dataset_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "train.jsonl").write_text("", encoding="utf-8")

    assert study_run.discover_experiments(dataset_root) == [first, second]
    train, validation, manifest = study_run.load_experiment_rows(first)
    assert len(train) == 8
    assert len(validation) == 4
    assert manifest["config_sha256"] == "a" * 64
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in validation} == {"validation"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda train, validation: validation[0].update(
                example_fingerprint_sha256=train[0]["example_fingerprint_sha256"]
            ),
            "overlap",
        ),
        (
            lambda train, validation: train[1].update(
                example_fingerprint_sha256=train[0]["example_fingerprint_sha256"]
            ),
            "duplicate fingerprint",
        ),
        (
            lambda train, validation: train[0].update(
                example_fingerprint_sha256="z" * 64
            ),
            "invalid/duplicate fingerprint",
        ),
        (
            lambda train, validation: validation[0].update(split="train"),
            "has split",
        ),
        (
            lambda train, validation: train[0]["target"].update(
                log_mean_loss=123.0
            ),
            "disagree",
        ),
    ],
)
def test_load_experiment_rows_rejects_split_leakage_and_corruption(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    experiment_dir = _write_experiment(tmp_path, "experiment")
    train, validation = _experiment_rows("experiment")
    mutation(train, validation)
    _write_jsonl(experiment_dir / "train.jsonl", train)
    _write_jsonl(experiment_dir / "validation.jsonl", validation)

    with pytest.raises(ValueError, match=message):
        study_run.load_experiment_rows(experiment_dir)


def test_discover_requires_at_least_one_completed_experiment(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        study_run.discover_experiments(tmp_path / "missing")

    incomplete_root = tmp_path / "incomplete_root"
    (incomplete_root / "partial").mkdir(parents=True)
    with pytest.raises(ValueError, match="No completed experiments"):
        study_run.discover_experiments(incomplete_root)


def test_make_stratified_folds_is_deterministic_balanced_and_exhaustive() -> None:
    rows = [
        {"stratum": stratum}
        for stratum in ("adam", "sgd", "rmsprop")
        for _ in range(6)
    ]

    first = study_run.make_stratified_folds(rows, n_splits=3, seed=17)
    second = study_run.make_stratified_folds(rows, n_splits=3, seed=17)

    assert len(first) == 3
    assert all(
        np.array_equal(left_train, right_train)
        and np.array_equal(left_test, right_test)
        for (left_train, left_test), (right_train, right_test) in zip(first, second)
    )
    all_test = np.concatenate([test for _, test in first])
    assert sorted(all_test.tolist()) == list(range(len(rows)))
    for train, test in first:
        assert set(train).isdisjoint(test)
        assert len(train) == 12
        assert len(test) == 6
        assert {
            stratum: sum(rows[index]["stratum"] == stratum for index in test)
            for stratum in ("adam", "sgd", "rmsprop")
        } == {"adam": 2, "sgd": 2, "rmsprop": 2}

    with pytest.raises(ValueError, match="at least two"):
        study_run.make_stratified_folds(rows, n_splits=1, seed=17)
    with pytest.raises(ValueError, match="empty"):
        study_run.make_stratified_folds([], n_splits=2, seed=17)
    with pytest.raises(ValueError, match="at least 4 rows"):
        study_run.make_stratified_folds(
            [{"stratum": "small"}] * 3 + [{"stratum": "large"}] * 6,
            n_splits=4,
            seed=17,
        )


def _minimal_definitions(seed: int) -> list[SearchDefinition]:
    return search_definitions(seed)[:2]


def _patch_lightweight_study(monkeypatch, fit_calls: list[str]) -> None:
    real_fit = study_run.fit_definition

    def counting_fit(definition, *args, **kwargs):
        fit_calls.append(definition.name)
        return real_fit(definition, *args, **kwargs)

    monkeypatch.setattr(study_run, "search_definitions", _minimal_definitions)
    monkeypatch.setattr(study_run, "fit_definition", counting_fit)
    monkeypatch.setattr(
        study_run,
        "split_half_noise_ceiling",
        lambda rows, base_dir: {"status": "stub", "n_rows": len(rows)},
    )
    monkeypatch.setattr(
        "tools.meta_model_study.interpretation.build_interpretation",
        lambda rows, models: {"status": "stub", "n_rows": len(rows)},
    )


def _fit_minimal(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path, list[str], dict]:
    dataset_root = tmp_path / "dataset"
    experiment_dir = _write_experiment(dataset_root, EXPERIMENT_ID)
    output_root = tmp_path / "study"
    calls: list[str] = []
    _patch_lightweight_study(monkeypatch, calls)
    summary = study_run.fit_experiment(
        experiment_dir,
        output_root,
        jobs=1,
        seed=123,
        n_splits=2,
    )
    return experiment_dir, output_root, calls, summary


def test_fit_experiment_checkpoints_two_models_and_resumes_without_refit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_dir, output_root, calls, first = _fit_minimal(tmp_path, monkeypatch)

    assert calls == ["constant_mean", "max_params_heuristic"]
    assert first["cv_champion"] == "max_params_heuristic"
    assert first["best_interpretable"] == "max_params_heuristic"
    model_root = output_root / "experiments" / EXPERIMENT_ID / "models"
    for method in calls:
        assert (model_root / f"{method}.joblib").is_file()
        sidecar = json.loads((model_root / f"{method}.json").read_text("utf-8"))
        assert sidecar["method"] == method
        assert len(sidecar["input_digest"]) == 64
    progress = json.loads(
        (output_root / "experiments" / EXPERIMENT_ID / "progress.json").read_text(
            "utf-8"
        )
    )
    assert progress["current"] is None
    assert progress["completed"] == calls

    calls.clear()
    second = study_run.fit_experiment(
        experiment_dir,
        output_root,
        jobs=1,
        seed=123,
        n_splits=2,
    )
    assert calls == []
    assert second["cv_champion"] == first["cv_champion"]


def _public_choice(letter: str, candidate_id: str, width: int) -> dict:
    return {
        "letter": letter,
        "candidate_id": candidate_id,
        "model": _model(width),
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
        "loss": {"loss_id": "mse"},
        "budget": {
            "batch_size": 32,
            "training_steps": 32,
            "total_samples_seen": 1024,
        },
    }


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *( (_walk_keys(child) for child in value.values()) ),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_walk_keys(child) for child in value), set())
    return set()


def test_external_prediction_then_independent_scoring_small_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, output_root, _, _ = _fit_minimal(tmp_path, monkeypatch)
    questions_path = tmp_path / "questions_sanitized.json"
    _write_json(
        questions_path,
        [
            {
                "question_id": "q_external",
                "question_run_id": "run_external",
                "family": "multivariate_regression",
                "dataset_id": "mvar_test",
                "selection_metric": "test_mse",
                "choices": [
                    _public_choice("A", "candidate_a", width=8),
                    _public_choice("B", "candidate_b", width=32),
                ],
            }
        ],
    )
    answer_key = tmp_path / "answer_key.json"
    assert not answer_key.exists()

    unscored = study_run.predict_external(output_root, questions_path)

    assert unscored["answer_key_opened"] is False
    assert unscored["primary_method"] == "cv_champion"
    assert {artifact["method"] for artifact in unscored["artifacts"]} == {
        "best_interpretable",
        "constant_mean",
        "cv_champion",
        "max_params_heuristic",
    }
    for artifact in unscored["artifacts"]:
        payload = json.loads(Path(artifact["path"]).read_text("utf-8"))
        assert payload["metadata"]["answer_key_opened"] is False
        assert not {
            "answer_key",
            "correct_letter",
            "is_correct",
            "target",
        }.intersection(_walk_keys(payload))

    _write_json(
        answer_key,
        [
            {
                "question_id": "q_external",
                "family": "multivariate_regression",
                "correct_letter": "B",
                "choices": [
                    {"letter": "A", "candidate_id": "candidate_a"},
                    {"letter": "B", "candidate_id": "candidate_b"},
                ],
            }
        ],
    )
    scored = study_run.score_external(output_root, answer_key)

    by_method = {row["method"]: row for row in scored["methods"]}
    assert scored["phase"] == "scored_external_predictions"
    assert by_method["cv_champion"]["total"] == {
        "num_questions": 1,
        "num_correct": 1,
        "accuracy": 1.0,
    }
    assert by_method["constant_mean"]["total"]["accuracy"] == 0.0
    assert Path(by_method["cv_champion"]["score_path"]).is_file()
