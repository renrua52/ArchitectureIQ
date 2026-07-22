"""Regression tests for custom-setting training progress callbacks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from architecture_iq.ground_truth import runner
from architecture_iq.profile import load_profile


def _candidate_spec() -> dict[str, Any]:
    return {
        "candidate_id": "cand_progress_test",
        "family": "univariate_regression",
        "budget": {"batch_size": 2, "training_steps": 2, "total_samples_seen": 4},
        "model": {"type": "mlp"},
        "execution": {"device": "cpu"},
    }


def test_run_ground_truth_emits_seed_boundaries_and_evaluation_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner exposes one bounded sequence of progress events per seed."""
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "candidate_spec.json").write_text(
        json.dumps(_candidate_spec()), encoding="utf-8"
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_spec.json").write_text(
        json.dumps({"selection_metric": "test_mse"}), encoding="utf-8"
    )

    profile = load_profile("v1")
    profile.ground_truth["n_seeds"] = 2
    profile.ground_truth["base_seed"] = 17
    events: list[dict[str, Any]] = []

    monkeypatch.setattr(runner, "_resolve_execution_device", lambda *_: torch.device("cpu"))
    monkeypatch.setattr(runner, "_sync_candidate_files", lambda *_: None)
    monkeypatch.setattr(
        runner,
        "get_dataset_family",
        lambda _name: SimpleNamespace(
            load_tensors=lambda _path: (
                torch.zeros((4, 1)),
                torch.zeros(4),
                torch.zeros((2, 1)),
                torch.zeros(2),
            )
        ),
    )

    def fake_single_seed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seed = int(kwargs.get("seed", args[6]))
        callback = kwargs["progress_callback"]
        for step, (samples_seen, metric) in enumerate(zip((2, 4), (0.4, 0.2)), start=1):
            callback(
                {
                    "step": step,
                    "samples_seen": samples_seen,
                    "total_samples_seen": samples_seen,
                    "metric": metric,
                }
            )
        return {
            "seed": seed,
            "failed": False,
            "final_test_mse": 0.1,
            "eval_samples": [2, 4],
            "step_metrics": [0.4, 0.2],
        }

    monkeypatch.setattr(runner, "run_single_seed", fake_single_seed)
    summary = runner.run_ground_truth(
        candidate_dir,
        profile,
        dataset_path=dataset_dir,
        sync_files=False,
        progress_callback=events.append,
    )

    assert summary["n_seeds"] == 2
    assert [event["phase"] for event in events] == [
        "seed_started",
        "evaluation",
        "evaluation",
        "seed_finished",
        "seed_started",
        "evaluation",
        "evaluation",
        "seed_finished",
    ]
    evaluations = [event for event in events if event["phase"] == "evaluation"]
    assert [event["seed_index"] for event in evaluations] == [1, 1, 2, 2]
    assert [event["step"] for event in evaluations] == [1, 2, 1, 2]
    for event in evaluations:
        assert event["n_seeds"] == 2
        assert event["training_steps"] == 2
        assert event["selection_metric"] == "test_mse"
        assert isinstance(event["metric"], float)
        assert isinstance(event["samples_seen"], int)
        assert isinstance(event["total_samples_seen"], int)


def test_run_custom_setting_forwards_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspector runs pass the caller's callback through to ground truth."""
    import sys

    tools_dir = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
    sys.path.insert(0, str(tools_dir))
    import custom_settings

    spec = _candidate_spec()
    profile = load_profile("v1")
    storage_root = tmp_path / "settings"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    def fake_write_candidate(candidate_spec: dict[str, Any], output_dir: Path, _family: Any) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "candidate_spec.json").write_text(
            json.dumps(candidate_spec), encoding="utf-8"
        )

    monkeypatch.setattr(custom_settings, "write_candidate", fake_write_candidate)
    monkeypatch.setattr(custom_settings, "ensure_registries", lambda: None)
    monkeypatch.setattr(custom_settings, "get_model_type", lambda _name: object())

    seen_callbacks: list[Any] = []

    def fake_ground_truth(*args: Any, **kwargs: Any) -> dict[str, Any]:
        callback = kwargs["progress_callback"]
        seen_callbacks.append(callback)
        callback({"phase": "seed_started", "seed_index": 0})
        output_dir = Path(args[0])
        results = output_dir / "results"
        results.mkdir(parents=True, exist_ok=True)
        np.savez(results / "curves.npz", curves=np.asarray([[1.0]]), samples=np.asarray([2]))
        return {"selection_metric": "test_mse", "mean_test_mse": 1.0}

    monkeypatch.setattr(custom_settings, "run_ground_truth", fake_ground_truth)
    events: list[dict[str, Any]] = []
    custom_settings.run_custom_setting(
        storage_root,
        dataset_dir,
        profile,
        spec,
        label="Progress",
        n_seeds=1,
        base_seed=3,
        progress_callback=events.append,
    )

    assert seen_callbacks == [events.append]
    assert events == [{"phase": "seed_started", "seed_index": 0}]
