from __future__ import annotations

import json

import numpy as np

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.datasets import create_dataset
from architecture_iq.ground_truth import runner
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.paths import candidate_dir
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type
from architecture_iq.runtime.loader import load_candidate_train, load_synthesize_module
from architecture_iq.util import write_json


def test_synthesize_py_is_executed() -> None:
    ensure_registries()
    profile = load_profile("v1")
    _, ds_path = create_dataset(profile, 99, family_name="univariate_regression")
    synth_path = ds_path / "synthesize.py"
    assert synth_path.exists()
    module = load_synthesize_module(synth_path)
    tx, ty, vx, vy = module.synthesize()
    assert tx.shape == (256, 1)
    assert vy.shape == (256, 1)


def test_ground_truth_runs_candidate_py_files() -> None:
    ensure_registries()
    profile = load_profile("v1")
    dataset_spec, ds_path = create_dataset(
        profile, 7, family_name="univariate_regression"
    )
    model = {
        "type": "mlp",
        "depth": 2,
        "width": 16,
        "residual": False,
        "layer_norm": [False, True],
        "activations": ["relu", "gelu"],
    }
    spec = build_candidate_spec(
        profile,
        dataset_id=dataset_spec["dataset_id"],
        family="univariate_regression",
        budget=1024,
        batch_size=16,
        model=model,
        optimizer={"type": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        loss={"loss_id": "mse"},
    )
    out = candidate_dir(
        spec["family"],
        spec["dataset_id"],
        spec["budget"]["total_samples_seen"],
        spec["candidate_id"],
    )
    write_candidate(spec, out, get_model_type("mlp"))

    train_mod = load_candidate_train(out)
    assert hasattr(train_mod, "train_and_eval")

    summary = run_ground_truth(out, profile, ds_path, sync_files=False)
    assert summary["execution"] == "candidate_py_files"
    assert summary["n_seeds"] == profile.n_seeds
    assert "mean_test_mse" in summary

    curves_path = out / "results" / "curves.npz"
    curves_data = np.load(curves_path)
    assert "samples" in curves_data
    assert len(curves_data["samples"]) == 64
    assert curves_data["curves"].shape[1] == len(curves_data["samples"])
    assert int(curves_data["samples"][-1]) == 1024
    assert int(curves_data["batch_size"]) == 16


def test_fully_failed_ground_truth_writes_strict_null_aggregates(
    tmp_path,
    monkeypatch,
) -> None:
    profile = load_profile("v1")
    dataset_path = tmp_path / "dataset"
    candidate_path = tmp_path / "candidate"
    write_json(
        dataset_path / "dataset_spec.json",
        {"selection_metric": "test_mse"},
    )
    write_json(
        candidate_path / "candidate_spec.json",
        {
            "candidate_id": "c_failed",
            "family": "fake_family",
            "budget": {"batch_size": 16, "training_steps": 64, "total_samples_seen": 1024},
        },
    )

    class FakeFamily:
        @staticmethod
        def load_tensors(path):
            assert path == dataset_path.resolve()
            return (object(), object(), object(), object())

    def fake_single_seed(
        candidate,
        candidate_spec,
        train_x,
        train_y,
        test_x,
        test_y,
        seed,
        fail_threshold,
        *,
        selection_metric,
        device,
        progress_callback=None,
    ):
        del (
            candidate,
            candidate_spec,
            train_x,
            train_y,
            test_x,
            test_y,
            fail_threshold,
        )
        assert selection_metric == "test_mse"
        return {
            "seed": seed,
            "failed": True,
            "final_test_mse": 3.0,
            "eval_samples": [16],
            "step_metrics": [3.0],
        }

    monkeypatch.setattr(runner, "get_dataset_family", lambda family: FakeFamily())
    monkeypatch.setattr(runner, "run_single_seed", fake_single_seed)
    summary = runner.run_ground_truth(
        candidate_path,
        profile,
        dataset_path,
        sync_files=False,
    )

    assert summary["failed_seeds"] == profile.n_seeds
    assert summary["excluded"] is True
    assert summary["mean_test_mse"] is None
    assert summary["std_test_mse"] is None
    persisted = json.loads(
        (candidate_path / "results" / "summary.json").read_text(encoding="utf-8"),
        parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)),
    )
    assert persisted["mean_test_mse"] is None
    assert persisted["std_test_mse"] is None
