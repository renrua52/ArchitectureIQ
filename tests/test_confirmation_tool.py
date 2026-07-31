from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_tool():
    path = REPO / "tools" / "batch_generate" / "confirmation.py"
    spec = importlib.util.spec_from_file_location("confirmation_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONFIRMATION = _load_tool()


def _summary(
    candidate_id: str,
    mean: float,
    *,
    base_seed: int,
    n_seeds: int = 2,
    failed_seeds: int = 0,
) -> dict[str, Any]:
    values = [mean - 0.005, mean + 0.005]
    assert n_seeds == len(values)
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "selection_metric": "test_mse",
        "execution": "candidate_py_files",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "failed_seeds": failed_seeds,
        "excluded": False,
        "mean_test_mse": mean,
        "std_test_mse": 0.005,
        "seed_results": [
            {
                "seed": base_seed + index,
                "failed": False,
                "final_test_mse": value,
            }
            for index, value in enumerate(values)
        ],
        "environment": {"device": "cpu"},
    }


def _question_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    data_root = tmp_path / "data"
    dataset = data_root / "datasets" / "univariate_regression" / "sym_test"
    (dataset / "candidates").mkdir(parents=True)
    (dataset / "dataset_spec.json").write_text(
        json.dumps(
            {
                "dataset_id": "sym_test",
                "family": "univariate_regression",
                "selection_metric": "test_mse",
            }
        ),
        encoding="utf-8",
    )

    candidates: dict[str, Path] = {}
    for candidate_id, mean in (("c_a", 0.10), ("c_b", 0.90)):
        candidate = dataset / "candidates" / "set_test" / candidate_id
        candidate.mkdir(parents=True)
        spec = {
            "schema_version": "1.0",
            "candidate_id": candidate_id,
            "dataset_id": "sym_test",
            "family": "univariate_regression",
            "budget": {
                "training_steps": 1,
                "batch_size": 1,
                "total_samples_seen": 1,
            },
            "model": {"type": "mlp"},
            "optimizer": {"type": "Adam", "lr": 0.001},
            "loss": {"loss_id": "mse"},
        }
        (candidate / "candidate_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        results = candidate / "results"
        results.mkdir()
        (results / "summary.json").write_text(
            json.dumps(_summary(candidate_id, mean, base_seed=0)),
            encoding="utf-8",
        )
        candidates[candidate_id] = candidate

    run = dataset / "questions" / "run_test"
    question_dir = run / "q_test"
    question_dir.mkdir(parents=True)
    question = {
        "schema_version": "1.0",
        "question_id": "q_test",
        "profile": "v1",
        "family": "univariate_regression",
        "dataset_id": "sym_test",
        "correct_letter": "A",
        "choices": [
            {
                "letter": "A",
                "candidate_id": "c_a",
                "candidate_path": str(candidates["c_a"].relative_to(data_root)),
            },
            {
                "letter": "B",
                "candidate_id": "c_b",
                "candidate_path": str(candidates["c_b"].relative_to(data_root)),
            },
        ],
        "evaluation": {
            "selection_metric": "test_mse",
            "base_seed": 0,
            "n_seeds": 2,
        },
    }
    question_path = question_dir / "question.json"
    question_path.write_text(json.dumps(question), encoding="utf-8")
    (run / "run.json").write_text(
        json.dumps({"run_id": "run_test", "question_ids": ["q_test"]}),
        encoding="utf-8",
    )
    return data_root, question_path, candidates


def test_worker_uses_temp_candidate_and_preserves_original_results(tmp_path: Path) -> None:
    candidate = tmp_path / "source" / "c_test"
    candidate.mkdir(parents=True)
    candidate_spec = {
        "candidate_id": "c_test",
        "family": "univariate_regression",
        "model": {"type": "mlp"},
    }
    (candidate / "candidate_spec.json").write_text(
        json.dumps(candidate_spec), encoding="utf-8"
    )
    original_results = candidate / "results"
    original_results.mkdir()
    sentinel = original_results / "summary.json"
    sentinel.write_text('{"sentinel": true}\n', encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    temp_root = tmp_path / "temporary"
    temp_root.mkdir()
    observed: dict[str, Any] = {}

    def fake_write_candidate(spec: dict[str, Any], out: Path, model: object) -> Path:
        assert spec == candidate_spec
        assert out != candidate
        out.mkdir(parents=True)
        (out / "candidate_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        observed["temp_candidate"] = out
        observed["model"] = model
        return out

    def fake_run_ground_truth(temp: Path, profile: Any, dataset_path: Path) -> dict[str, Any]:
        assert temp == observed["temp_candidate"]
        assert dataset_path == dataset.resolve()
        assert profile.base_seed == 100
        assert profile.n_seeds == 2
        assert profile.raw["ground_truth"]["base_seed"] == 100
        (temp / "results").mkdir()
        summary = _summary("c_test", 0.2, base_seed=100)
        (temp / "results" / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary

    model_family = object()
    with (
        patch("architecture_iq.registry.ensure_registries"),
        patch("architecture_iq.registry.get_model_type", return_value=model_family),
        patch(
            "architecture_iq.candidates.generator.write_candidate",
            side_effect=fake_write_candidate,
        ),
        patch(
            "architecture_iq.ground_truth.runner.run_ground_truth",
            side_effect=fake_run_ground_truth,
        ),
    ):
        result = CONFIRMATION._confirm_candidate_worker(
            str(candidate),
            str(dataset),
            "v1",
            100,
            2,
            str(temp_root),
        )

    assert result["summary"]["base_seed"] == 100
    assert result["summary"]["n_seeds"] == 2
    assert sentinel.read_text(encoding="utf-8") == '{"sentinel": true}\n'
    assert not observed["temp_candidate"].exists()


def test_confirmation_deduplicates_candidates_and_records_changed_winner(
    tmp_path: Path,
) -> None:
    data_root, question_path, candidates = _question_fixture(tmp_path)
    calls: list[str] = []

    def fake_worker(
        candidate_path: str,
        dataset_path: str,
        profile_name: str,
        base_seed: int,
        n_seeds: int,
        temp_root: str | None,
    ) -> dict[str, Any]:
        del temp_root
        calls.append(candidate_path)
        candidate_id = Path(candidate_path).name
        # Fresh seeds reverse the original winner.
        mean = 0.80 if candidate_id == "c_a" else 0.10
        return {
            "candidate_path": candidate_path,
            "candidate_id": candidate_id,
            "dataset_path": dataset_path,
            "profile": profile_name,
            "elapsed_seconds": 0.01,
            "summary": _summary(
                candidate_id,
                mean,
                base_seed=base_seed,
                n_seeds=n_seeds,
            ),
        }

    index = CONFIRMATION.run_confirmation(
        [question_path, question_path],
        data_root=data_root,
        base_seed=100,
        n_seeds=2,
        workers=1,
        worker_fn=fake_worker,
    )

    assert len(calls) == 2
    expected_keys = {str(path.resolve()) for path in candidates.values()}
    assert set(index["candidates"]) == expected_keys
    question = index["questions"][str(question_path.resolve())]
    assert question["original_winner"]["candidate_id"] == "c_a"
    assert question["confirmation_winner"]["candidate_id"] == "c_b"
    assert not question["winner_matches"]
    assert question["validate_significance"]["original"]["passed"]
    assert question["validate_significance"]["confirmation"]["passed"]
    assert index["summary"]["changed_winners"] == 1
    assert index["summary"]["unique_candidates"] == 2


def test_worker_error_is_atomic_and_returns_nonzero(tmp_path: Path) -> None:
    data_root, question_path, _ = _question_fixture(tmp_path)
    output = tmp_path / "confirmation.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    def failing_worker(*args: Any) -> dict[str, Any]:
        candidate_id = Path(str(args[0])).name
        if candidate_id == "c_b":
            raise RuntimeError("synthetic worker failure")
        return {
            "candidate_path": str(args[0]),
            "candidate_id": candidate_id,
            "dataset_path": str(args[1]),
            "profile": str(args[2]),
            "elapsed_seconds": 0.01,
            "summary": _summary(candidate_id, 0.1, base_seed=int(args[3])),
        }

    exit_code = CONFIRMATION.run_confirmation_to_file(
        [question_path],
        output_path=output,
        data_root=data_root,
        base_seed=100,
        n_seeds=2,
        workers=1,
        worker_fn=failing_worker,
    )

    assert exit_code == 1
    index = json.loads(output.read_text(encoding="utf-8"))
    assert index["status"] == "error"
    assert len(index["worker_errors"]) == 1
    error = next(iter(index["worker_errors"].values()))
    assert error == {"type": "RuntimeError", "message": "synthetic worker failure"}
    assert not list(tmp_path.glob(".confirmation.json.*.tmp"))


def test_rejects_confirmation_seed_overlap(tmp_path: Path) -> None:
    data_root, question_path, _ = _question_fixture(tmp_path)

    with pytest.raises(ValueError, match="overlap original GT seeds"):
        CONFIRMATION.run_confirmation(
            [question_path],
            data_root=data_root,
            base_seed=1,
            n_seeds=2,
            workers=1,
            worker_fn=lambda *args: {},
        )


def test_collect_question_paths_accepts_run_and_relative_list(tmp_path: Path) -> None:
    _, question_path, _ = _question_fixture(tmp_path)
    run_path = question_path.parents[1]
    path_list = tmp_path / "questions.json"
    path_list.write_text(
        json.dumps({"questions": [str(run_path.relative_to(tmp_path))]}),
        encoding="utf-8",
    )

    assert CONFIRMATION.collect_question_paths([path_list]) == [question_path.resolve()]
