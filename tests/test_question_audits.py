from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from question_audit_lib import audit_question_inputs, audit_question_run  # noqa: E402
from architecture_iq.profile import load_profile  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _spec(profile_hash: str, candidate_id: str, *, depth: int, mean: float) -> tuple[dict, dict]:
    spec = {
        "schema_version": "1.0",
        "profile": "v1",
        "profile_hash": profile_hash,
        "candidate_id": candidate_id,
        "dataset_id": "sym_audit",
        "family": "univariate_regression",
        "budget": {"training_steps": 64, "batch_size": 16, "total_samples_seen": 1024},
        "model": {
            "type": "mlp",
            "depth": depth,
            "width": 16,
            "residual": False,
            "layer_norm": [False] * depth,
            "activations": ["relu"] * depth,
            "input_dim": 1,
        },
        "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "loss": {"loss_id": "mse"},
        "execution": {"device": "cpu"},
    }
    seed_results = [
        {"seed": seed, "failed": False, "final_test_mse": mean}
        for seed in range(10)
    ]
    summary = {
        "candidate_id": candidate_id,
        "selection_metric": "test_mse",
        "n_seeds": 10,
        "base_seed": 0,
        "failed_seeds": 0,
        "excluded": False,
        "mean_test_mse": mean,
        "std_test_mse": 0.01,
        "seed_results": seed_results,
        "environment": {"device": "cpu"},
    }
    return spec, summary


def _write_candidate(path: Path, spec: dict, summary: dict) -> None:
    _write_json(path / "candidate_spec.json", spec)
    _write_json(path / "results" / "summary.json", summary)
    np.savez(path / "results" / "curves.npz", test_mse=np.array([summary["mean_test_mse"]]))


def test_input_audit_accepts_complete_provenance_and_marks_legacy_review(tmp_path: Path) -> None:
    profile = load_profile("v1")
    dataset = tmp_path / "dataset"
    _write_json(
        dataset / "dataset_spec.json",
        {"dataset_id": "sym_audit", "family": "univariate_regression", "selection_metric": "test_mse"},
    )
    candidate_set = dataset / "candidates" / "set_audit"
    _write_json(
        candidate_set / "set.json",
        {
            "dataset_id": "sym_audit", "family": "univariate_regression", "profile": "v1",
            "profile_hash": profile.profile_hash, "count": 2,
            "varying_axes": ["model"], "invariant_axes": ["optimizer", "loss"],
        },
    )
    for candidate_id, depth, mean in (("c_one", 1, 0.1), ("c_two", 2, 0.3)):
        spec, summary = _spec(profile.profile_hash, candidate_id, depth=depth, mean=mean)
        _write_candidate(candidate_set / candidate_id, spec, summary)

    report = audit_question_inputs(dataset, [candidate_set], profile)
    assert report["valid"] is True
    assert report["summary"]["pass"] == 2

    manifest = json.loads((candidate_set / "set.json").read_text(encoding="utf-8"))
    manifest.pop("profile_hash")
    _write_json(candidate_set / "set.json", manifest)
    for candidate_id in ("c_one", "c_two"):
        path = candidate_set / candidate_id / "candidate_spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec.pop("profile_hash")
        _write_json(path, spec)
    legacy = audit_question_inputs(dataset, [candidate_set], profile)
    assert legacy["valid"] is False
    assert legacy["summary"]["review"] == 2


def test_run_audit_recomputes_winner_without_rendering_or_mutating_candidates(tmp_path: Path) -> None:
    profile = load_profile("v1")
    data_root = tmp_path / "data"
    dataset = data_root / "datasets" / "univariate_regression" / "sym_audit"
    _write_json(dataset / "dataset_spec.json", {"dataset_id": "sym_audit", "family": "univariate_regression", "selection_metric": "test_mse"})
    candidates = dataset / "candidates" / "set_audit"
    records = []
    for candidate_id, depth, mean in (("c_one", 1, 0.1), ("c_two", 2, 0.3)):
        spec, summary = _spec(profile.profile_hash, candidate_id, depth=depth, mean=mean)
        path = candidates / candidate_id
        _write_candidate(path, spec, summary)
        records.append((candidate_id, path))
    run = dataset / "questions" / "run_audit"
    _write_json(run / "run.json", {"num_questions": 1, "question_ids": ["q_audit"], "profile": "v1", "profile_hash": profile.profile_hash})
    choices = [
        {"letter": "A", "candidate_id": records[0][0], "candidate_path": "datasets/univariate_regression/sym_audit/candidates/set_audit/c_one"},
        {"letter": "B", "candidate_id": records[1][0], "candidate_path": "datasets/univariate_regression/sym_audit/candidates/set_audit/c_two"},
    ]
    _write_json(
        run / "q_audit" / "question.json",
        {
            "question_id": "q_audit", "family": "univariate_regression", "dataset_id": "sym_audit",
            "num_choices": 2, "choices": choices, "type": "architecture_only",
            "invariant_axes": ["optimizer", "loss", "batch_size"], "varying_axes": ["model"],
            "budget": {"total_samples_seen": 1024}, "correct_letter": "A",
            "significance": {"passed": True, "gap": 0.2, "win_rate": 1.0, "metric": "test_mse"},
            "evaluation": {"selection_metric": "test_mse"}, "prompt": {"rendered_path": "prompt.txt"},
        },
    )
    prompt_path = run / "q_audit" / "prompt.txt"
    prompt_path.write_text("Choose the better architecture.", encoding="utf-8")
    before = (candidates / "c_one" / "candidate_spec.json").read_bytes()
    report = audit_question_run(run, profile, data_root=data_root)
    assert report["valid"] is True
    assert (candidates / "c_one" / "candidate_spec.json").read_bytes() == before
