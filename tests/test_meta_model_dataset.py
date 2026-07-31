from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_model_type
from architecture_iq.util import read_json, write_json
from tools.meta_model_dataset.core import (
    assign_pre_execution_splits,
    build_attempt_row,
    full_candidate_fingerprint,
    generated_parameter_counts,
    select_usable_rows,
    sha256_file,
    sha256_json,
)
from tools.meta_model_dataset import build as dataset_builder


def _mlp_model() -> dict:
    return {
        "type": "mlp",
        "input_dim": 3,
        "depth": 2,
        "width": 8,
        "residual": True,
        "activations": ["relu", "gelu"],
        "layer_norm": [True, False],
    }


def _candidate_spec(model: dict, *, family: str = "multivariate_regression") -> dict:
    profile = load_profile("v1")
    return build_candidate_spec(
        profile,
        dataset_id="dataset_test",
        family=family,
        budget=1024,
        batch_size=16,
        model=model,
        optimizer={"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
        loss={"loss_id": "cross_entropy" if family == "bigram_lm" else "mse"},
    )


def _write_candidate(tmp_path: Path, spec: dict) -> Path:
    ensure_registries()
    candidate_dir = tmp_path / spec["candidate_id"]
    write_candidate(spec, candidate_dir, get_model_type(spec["model"]["type"]))
    return candidate_dir


def test_full_fingerprint_does_not_trust_short_candidate_id() -> None:
    spec = _candidate_spec(_mlp_model())
    same_setting = deepcopy(spec)
    same_setting["candidate_id"] = "c_collision"
    changed_setting = deepcopy(same_setting)
    changed_setting["optimizer"]["lr"] = 0.003

    assert full_candidate_fingerprint(spec) == full_candidate_fingerprint(same_setting)
    assert full_candidate_fingerprint(spec) != full_candidate_fingerprint(changed_setting)
    assert len(full_candidate_fingerprint(spec)) == 64


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_mlp_model(), 201),
        (
            {
                "type": "transformer_lm",
                "vocab_size": 32,
                "context_length": 16,
                "d_model": 32,
                "num_layers": 1,
                "num_heads": 2,
                "d_ff": 64,
            },
            11136,
        ),
    ],
)
def test_parameter_count_comes_from_generated_model_and_matches_registry(
    tmp_path: Path,
    model: dict,
    expected: int,
) -> None:
    family = "bigram_lm" if model["type"] == "transformer_lm" else "multivariate_regression"
    spec = _candidate_spec(model, family=family)
    candidate_dir = _write_candidate(tmp_path, spec)

    counts = generated_parameter_counts(candidate_dir, spec)

    assert counts == {"total_params": expected, "trainable_params": expected}


def test_split_is_exact_deterministic_stratified_and_pre_gt() -> None:
    records = [
        {
            "fingerprint": f"{index:064x}",
            "stratum": f"optimizer={index % 5}",
            "spec": {"setting": index},
        }
        for index in range(120)
    ]

    first = assign_pre_execution_splits(
        records,
        num_rows=100,
        train_rows=90,
        seed=7,
    )
    second = assign_pre_execution_splits(
        records,
        num_rows=100,
        train_rows=90,
        seed=7,
    )

    assert first == second
    primary = [row for row in first if row["selection_role"] == "primary"]
    reserve = [row for row in first if row["selection_role"] == "reserve"]
    assert len(primary) == 100
    assert len(reserve) == 20
    assert sum(row["split"] == "train" for row in primary) == 90
    assert sum(row["split"] == "validation" for row in primary) == 10
    for stratum in {row["stratum"] for row in primary}:
        group = [row for row in primary if row["stratum"] == stratum]
        assert sum(row["split"] == "validation" for row in group) == 2
    assert all("target" not in row for row in first)


def test_attempt_row_uses_stored_metric_and_keeps_target_out_of_features(
    tmp_path: Path,
) -> None:
    spec = _candidate_spec(_mlp_model())
    candidate_dir = _write_candidate(tmp_path, spec)
    summary = {
        "schema_version": "1.0",
        "candidate_id": spec["candidate_id"],
        "selection_metric": "test_mse",
        "execution": "candidate_py_files",
        "n_seeds": 2,
        "base_seed": 0,
        "failed_seeds": 0,
        "excluded": False,
        "mean_test_mse": 0.25,
        "std_test_mse": 0.05,
        "seed_results": [
            {"seed": 0, "failed": False, "final_test_mse": 0.2},
            {"seed": 1, "failed": False, "final_test_mse": 0.3},
        ],
    }
    write_json(candidate_dir / "results" / "summary.json", summary)
    dataset_spec = {
        "family": "multivariate_regression",
        "dataset_id": "dataset_test",
        "selection_metric": "test_mse",
        "significance": {"fail_threshold": 2.0},
    }

    row = build_attempt_row(
        experiment_id="experiment_test",
        profile_name="v1",
        dataset_spec=dataset_spec,
        candidate_dir=candidate_dir,
        split="validation",
        selection_role="primary",
        stratum="Adam|8",
        relative_to=tmp_path,
    )

    assert row["target"]["mean_loss"] == pytest.approx(0.25)
    assert row["target"]["std_loss"] == pytest.approx(0.05)
    assert row["target"]["benchmark_eligible"] is True
    assert row["usable_for_regression"] is True
    assert row["features"]["derived.total_params"] == 201
    assert all("target" not in name for name in row["features"])
    assert all("seed" not in name for name in row["features"])
    assert all("candidate" not in name for name in row["features"])
    assert row["provenance"]["execution"] == "candidate_py_files"


def _attempt(
    fingerprint: str,
    *,
    split: str,
    role: str,
    usable: bool,
    stratum: str = "s",
) -> dict:
    return {
        "example_fingerprint_sha256": fingerprint,
        "split": split,
        "selection_role": role,
        "stratum": stratum,
        "usable_for_regression": usable,
    }


def test_unusable_primary_is_replaced_without_crossing_splits() -> None:
    attempts = [
        _attempt("train-good", split="train", role="primary", usable=True),
        _attempt("train-bad", split="train", role="primary", usable=False),
        _attempt("val-good", split="validation", role="primary", usable=True),
        _attempt("train-reserve", split="train", role="reserve", usable=True),
        _attempt("val-reserve", split="validation", role="reserve", usable=True),
    ]

    selected, replacements = select_usable_rows(
        attempts,
        train_rows=2,
        validation_rows=1,
    )

    assert {row["example_fingerprint_sha256"] for row in selected} == {
        "train-good",
        "train-reserve",
        "val-good",
    }
    assert replacements == [
        {
            "split": "train",
            "replaced": "train-bad",
            "replacement": "train-reserve",
        }
    ]


def test_gt_completion_rejects_changed_code_and_tampered_summary(
    tmp_path: Path,
) -> None:
    spec = _candidate_spec(_mlp_model())
    candidate_dir = _write_candidate(tmp_path, spec)
    summary = {
        "schema_version": "1.0",
        "candidate_id": spec["candidate_id"],
        "selection_metric": "test_mse",
        "execution": "candidate_py_files",
        "n_seeds": 2,
        "base_seed": 0,
        "failed_seeds": 0,
        "excluded": False,
        "mean_test_mse": 0.25,
        "std_test_mse": 0.05,
        "seed_results": [],
    }
    summary_path = candidate_dir / "results" / "summary.json"
    write_json(summary_path, summary)
    curves_path = candidate_dir / "results" / "curves.npz"
    np.savez(
        curves_path,
        curves=np.asarray([[0.3, 0.2]], dtype=float),
        samples=np.asarray([16, 32], dtype=int),
        batch_size=16,
    )
    config = {
        "profile": "v1",
        "profile_config": load_profile("v1").raw,
        "dataset_spec": {
            "family": "multivariate_regression",
            "dataset_id": "dataset_test",
            "selection_metric": "test_mse",
        },
        "ground_truth": {
            "n_seeds": 2,
            "base_seed": 0,
            "fail_threshold_mode": "finite_only",
        },
    }
    gt_hash = dataset_builder._gt_config_hash(config)
    context_hash = "execution-context"
    inputs_hash = dataset_builder._candidate_execution_inputs_sha256(
        candidate_dir,
        context_hash,
    )
    marker = {
        "schema_version": "1.0",
        "status": "ok",
        "gt_config_sha256": gt_hash,
        "execution_context_sha256": context_hash,
        "execution_inputs_sha256": inputs_hash,
        "summary_sha256": sha256_file(summary_path),
        "curves_sha256": sha256_file(curves_path),
    }
    write_json(candidate_dir / "results" / "meta_model_gt.json", marker)

    def complete(expected_inputs: str) -> bool:
        return dataset_builder._gt_complete(
            candidate_dir,
            config=config,
            gt_config_sha256=gt_hash,
            execution_context_sha256=context_hash,
            execution_inputs_sha256=expected_inputs,
        )

    assert complete(inputs_hash)

    model_path = candidate_dir / "model.py"
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed_inputs = dataset_builder._candidate_execution_inputs_sha256(
        candidate_dir,
        context_hash,
    )
    assert changed_inputs != inputs_hash
    assert not complete(changed_inputs)

    write_candidate(spec, candidate_dir, get_model_type(spec["model"]["type"]))
    restored_inputs = dataset_builder._candidate_execution_inputs_sha256(
        candidate_dir,
        context_hash,
    )
    marker["execution_inputs_sha256"] = restored_inputs
    write_json(candidate_dir / "results" / "meta_model_gt.json", marker)
    assert complete(restored_inputs)

    summary["mean_test_mse"] = 0.5
    write_json(summary_path, summary)
    assert not complete(restored_inputs)

    write_json(summary_path, {**summary, "mean_test_mse": 0.25})
    marker["summary_sha256"] = sha256_file(summary_path)
    write_json(candidate_dir / "results" / "meta_model_gt.json", marker)
    curves_path.write_bytes(curves_path.read_bytes() + b"tampered")
    assert not complete(restored_inputs)


def test_error_attempt_ignores_partial_summary(tmp_path: Path) -> None:
    spec = _candidate_spec(_mlp_model())
    candidate_dir = _write_candidate(tmp_path, spec)
    partial_summary = candidate_dir / "results" / "summary.json"
    partial_summary.parent.mkdir(parents=True)
    partial_summary.write_text("{partial", encoding="utf-8")

    row = build_attempt_row(
        experiment_id="experiment_test",
        profile_name="v1",
        dataset_spec={
            "family": "multivariate_regression",
            "dataset_id": "dataset_test",
            "selection_metric": "test_mse",
        },
        candidate_dir=candidate_dir,
        split="train",
        selection_role="reserve",
        stratum="s",
        include_summary=False,
    )

    assert row["usable_for_regression"] is False
    assert row["target"]["mean_loss"] is None
    assert row["provenance"]["summary_sha256"] is None


def test_wide_plan_exclusions_groups_and_phase_selection(tmp_path: Path) -> None:
    profile = load_profile("v1")
    dataset_spec = {
        "family": "multivariate_regression",
        "dataset_id": "dataset_test",
        "params": {"input_dim": 3},
        "selection_metric": "test_mse",
    }
    relevant = _candidate_spec(_mlp_model())
    irrelevant = _candidate_spec(_mlp_model())
    irrelevant["dataset_id"] = "another_dataset"

    candidate_set = tmp_path / "candidate_set"
    write_json(candidate_set / "a" / "candidate_spec.json", relevant)
    write_json(candidate_set / "b" / "candidate_spec.json", irrelevant)
    old_sampling = tmp_path / "sampling_manifest.json"
    write_json(
        old_sampling,
        {"records": [{"spec": relevant}, {"spec": irrelevant}]},
    )
    experiment = {
        "experiment_id": "wide_test",
        "phase": "b1_pilot",
        "dataset_path": str(tmp_path),
        "budget": 1024,
        "batch_size": 16,
        "vary": ["model", "optimizer", "loss"],
        "group_labels": {"distribution_role": "development"},
        "num_rows": 3,
        "train_rows": 2,
        "reserve_rows": 1,
    }
    plan = {
        "schema_version": "2.0",
        "exclusions": {
            "candidate_sets": [str(candidate_set)],
            "sampling_manifests": [str(old_sampling)],
        },
        "experiments": [experiment, {**experiment, "experiment_id": "wide_b2", "phase": "b2_scale"}],
    }

    config = dataset_builder._experiment_config(
        plan,
        experiment,
        profile,
        dataset_spec,
    )

    assert config["phase"] == "b1_pilot"
    assert config["group_labels"] == {
        "phase": "b1_pilot",
        "family": "multivariate_regression",
        "dataset": "dataset_test",
        "environment": "wide_test",
        "distribution_role": "development",
    }
    assert config["external_evaluation"]["excluded_fingerprints_sha256"] == [
        full_candidate_fingerprint(relevant)
    ]
    assert [source["relevant_candidate_specs"] for source in config["external_evaluation"]["exclusion_sources"]] == [1, 1]
    records = dataset_builder._sample_records(config, profile)
    assert len(records) == 4
    assert all(record["group_labels"] == config["group_labels"] for record in records)
    assert all(record["fingerprint"] != full_candidate_fingerprint(relevant) for record in records)
    assert sha256_json(config) != sha256_json({**config, "phase": "changed"})

    selected = dataset_builder._selected_experiments(
        plan,
        requested=None,
        requested_phases={"b2_scale"},
    )
    assert [item["experiment_id"] for item in selected] == ["wide_b2"]

    legacy_experiment = {
        **experiment,
        "exclude_candidate_sets": [str(candidate_set)],
    }
    legacy_config = dataset_builder._experiment_config(
        {"schema_version": "1.0", "experiments": [legacy_experiment]},
        legacy_experiment,
        profile,
        dataset_spec,
    )
    assert "phase" not in legacy_config
    assert "group_labels" not in legacy_config
    assert "exclusion_sources" not in legacy_config["external_evaluation"]


def test_wide_v2_plan_has_frozen_scale_and_axis_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = read_json(root / "tools/meta_model_dataset/plan_wide_v2.json")
    defaults = plan["defaults"]
    experiments = plan["experiments"]

    assert plan["profile"] == "meta_wide_v2"
    assert defaults["vary"] == ["model", "optimizer", "loss"]
    assert len(experiments) == 30
    assert sum(item.get("num_rows", defaults["num_rows"]) for item in experiments) == 10_000
    assert sum(item.get("train_rows", defaults["train_rows"]) for item in experiments) == 9_000
    assert sum(item.get("reserve_rows", defaults["reserve_rows"]) for item in experiments) == 510
    assert {item.get("phase", defaults["phase"]) for item in experiments} == {
        "b1_pilot",
        "b2_scale",
    }
    assert len({item["dataset_path"] for item in experiments}) == 15
    assert {item["budget"] for item in experiments} == {
        1024,
        2048,
        5120,
        10240,
        20480,
    }
    assert {item["batch_size"] for item in experiments} == {8, 16, 32, 64, 128}

    profile = load_profile("meta_wide_v2")
    assert min(profile.mlp["width"]) == 8
    assert max(profile.mlp["width"]) == 384
    assert min(profile.optimizer_grids["lr"]) == pytest.approx(1e-5)
    assert max(profile.optimizer_grids["lr"]) == pytest.approx(1e-1)
