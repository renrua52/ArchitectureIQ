from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.meta_model_study.metrics import winner_metrics_3choice
from tools.meta_model_study.wide import freeze_snapshot_manifest, load_environment
from tools.meta_model_study.wide_run import (
    MAX_JOBS,
    _regret_counts,
    build_parser,
    dynamic_split_noise_ceiling,
    fit_wide_grouped,
    fit_wide_id,
    main,
    noise_ceiling_for_environment,
    score_wide_predictions,
    validate_jobs,
    wide_search_definitions,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    experiment_id: str,
    family: str,
    dataset_id: str,
    index: int,
    *,
    split: str,
    cohort: str,
    n_seeds: int = 5,
) -> dict:
    total_params = 100 + 3 * index
    mean_loss = 0.4 + index / 100.0
    optimizer = "Adam" if index % 2 else "SGD"
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "family": family,
        "dataset_id": dataset_id,
        "split": split,
        "stratum": f"optimizer.type={optimizer}|loss.loss_id=mse",
        "usable_for_regression": True,
        "example_fingerprint_sha256": f"{index:064x}",
        "group_labels": {
            "environment": experiment_id,
            "dataset": dataset_id,
            "family": family,
            "phase": "test",
            "dataset_cohort": cohort,
        },
        "setting": {
            "model": {
                "type": "mlp",
                "input_dim": 2,
                "depth": 1,
                "width": 8 + index % 3,
                "residual": False,
                "activations": ["relu"],
                "layer_norm": [False],
            },
            "optimizer": {
                "type": optimizer,
                "lr": 0.001 if index % 3 else 0.01,
                "weight_decay": 0.0,
                **({"momentum": 0.9} if optimizer == "SGD" else {}),
            },
            "loss": {"loss_id": "mse"},
            "budget": {
                "batch_size": 8,
                "training_steps": 16,
                "total_samples_seen": 128,
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
            "std_loss": 0.01,
            "n_seeds": n_seeds,
            "failed_seeds": 0,
            "benchmark_eligible": index % 5 != 0,
        },
        "provenance": {
            "candidate_path": f"{experiment_id}/candidates/c_{index}",
        },
    }


def _write_environment(
    root: Path,
    experiment_id: str,
    dataset_id: str,
    *,
    first_index: int,
    cohort: str,
    family: str = "family_a",
    write_curves: bool = False,
) -> Path:
    train = [
        _row(
            experiment_id,
            family,
            dataset_id,
            first_index + index,
            split="train",
            cohort=cohort,
        )
        for index in range(6)
    ]
    validation = [
        _row(
            experiment_id,
            family,
            dataset_id,
            first_index + 6 + index,
            split="validation",
            cohort=cohort,
        )
        for index in range(3)
    ]
    path = root / experiment_id
    _write_jsonl(path / "all.jsonl", [*train, *validation])
    _write_jsonl(path / "train.jsonl", train)
    _write_jsonl(path / "validation.jsonl", validation)
    _write_json(
        path / "manifest.json",
        {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "config": {
                "experiment_id": experiment_id,
                "num_rows": len(train) + len(validation),
                "train_rows": len(train),
                "budget": 128,
                "batch_size": 8,
                "phase": "test",
                "dataset_path": "fixture_dataset",
                "ground_truth": {"n_seeds": 5},
                "dataset_spec": {"selection_metric": "test_mse"},
                "group_labels": {
                    "environment": experiment_id,
                    "dataset": dataset_id,
                    "family": family,
                    "phase": "test",
                    "dataset_cohort": cohort,
                },
            },
            "ground_truth": {"n_seeds": 5},
            "split_policy": {
                "assigned_before_ground_truth": True,
                "group_labels_frozen_before_ground_truth": True,
                "train": len(train),
                "validation": len(validation),
                "group_labels": {
                    "environment": experiment_id,
                    "dataset": dataset_id,
                    "family": family,
                    "phase": "test",
                    "dataset_cohort": cohort,
                },
            },
            "selected": {"total": len(train) + len(validation)},
        },
    )
    if write_curves:
        factors = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
        for row in validation:
            results = root / row["provenance"]["candidate_path"] / "results"
            results.mkdir(parents=True, exist_ok=True)
            final = float(row["target"]["mean_loss"]) * factors
            np.savez(results / "curves.npz", curves=np.column_stack([final, final]))
    return path


def _write_plan(root: Path, experiment_ids: list[str]) -> Path:
    path = root / "plan.json"
    _write_json(
        path,
        {
            "schema_version": "2.0",
            "defaults": {
                "num_rows": 9,
                "train_rows": 6,
                "budget": 128,
                "batch_size": 8,
                "phase": "test",
                "dataset_path": "fixture_dataset",
                "ground_truth": {"n_seeds": 5},
            },
            "experiments": [
                {"experiment_id": experiment_id} for experiment_id in experiment_ids
            ],
        },
    )
    return path


def _study_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wide"
    specifications = [
        ("env_a", "dataset_a", 100, "development"),
        ("env_b", "dataset_b", 200, "development"),
        ("env_c", "dataset_c", 300, "holdout_candidate"),
    ]
    for experiment, dataset, index, cohort in specifications:
        _write_environment(
            root,
            experiment,
            dataset,
            first_index=index,
            cohort=cohort,
        )
    return root, _write_plan(tmp_path, [item[0] for item in specifications])


def test_dataset_pooled_id_trains_and_scores_within_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    specifications = [
        ("env_a", "dataset_a", 100, "development"),
        ("env_b", "dataset_a", 200, "development"),
        ("env_c", "dataset_b", 300, "development"),
    ]
    for experiment, dataset, index, cohort in specifications:
        _write_environment(
            root,
            experiment,
            dataset,
            first_index=index,
            cohort=cohort,
        )
    plan = _write_plan(tmp_path, [item[0] for item in specifications])

    manifest = fit_wide_id(
        root,
        tmp_path / "dataset_id",
        plan,
        jobs=1,
        seed=11,
        n_splits=2,
        method_names={"constant_mean"},
        include_xgboost=False,
        scopes={"dataset"},
        force=False,
        noise_ceiling="skip",
        require_complete=True,
    )

    assert manifest["scopes"] == ["dataset"]
    assert manifest["results"]["dataset"] == {"n_tasks": 2, "n_test": 9}
    aggregate = json.loads(
        (
            tmp_path / "dataset_id" / "id" / "dataset" / "aggregate.json"
        ).read_text("utf-8")
    )
    assert aggregate["protocol"]["name"] == "dataset_pooled_id"
    assert aggregate["protocol"]["train"] == "all dataset train_rows"
    assert aggregate["protocol"]["test"] == "all dataset validation_rows"
    assert aggregate["task_ids"] == ["dataset=dataset_a", "dataset=dataset_b"]


def test_global_pooled_id_uses_one_shared_task(
    tmp_path: Path,
) -> None:
    dataset_root, plan = _study_fixture(tmp_path)

    manifest = fit_wide_id(
        dataset_root,
        tmp_path / "global_id",
        plan,
        jobs=1,
        seed=11,
        n_splits=2,
        method_names={"constant_mean"},
        include_xgboost=False,
        scopes={"global"},
        force=False,
        noise_ceiling="skip",
        require_complete=True,
    )

    assert manifest["scopes"] == ["global"]
    assert manifest["results"]["global"] == {"n_tasks": 1, "n_test": 9}
    aggregate = json.loads(
        (tmp_path / "global_id" / "id" / "global" / "aggregate.json").read_text(
            "utf-8"
        )
    )
    assert aggregate["protocol"]["name"] == "global_pooled_id"
    assert aggregate["protocol"]["shared_model_count"] == 1
    assert aggregate["task_ids"] == ["global"]


def test_wide_method_ladder_and_resource_cap() -> None:
    names = {definition.name for definition in wide_search_definitions(7)}
    assert {
        "constant_mean",
        "max_params_heuristic",
        "params_ols",
        "params_ridge",
        "params_polynomial_ridge",
        "optimizer_lr_lookup",
        "optimizer_lr_ridge",
        "compact_ols",
        "compact_ridge",
        "compact_polynomial_ridge",
        "compact_elastic_net",
        "full_ols",
        "full_ridge",
        "full_elastic_net",
        "shallow_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "gradient_boosting",
        "rbf_svr",
        "mlp",
        "compact_ridge_fixed",
        "full_ridge_fixed",
        "random_forest_fixed",
        "extra_trees_fixed",
        "mlp_fixed",
    }.issubset(names)
    assert "xgboost" not in names
    no_params = wide_search_definitions(7, include_parameter_count=False)
    no_param_names = {definition.name for definition in no_params}
    assert not {
        "max_params_heuristic",
        "params_ols",
        "params_ridge",
        "params_polynomial_ridge",
    }.intersection(no_param_names)
    for definition in no_params:
        if hasattr(definition.estimator, "named_steps") and "features" in definition.estimator.named_steps:
            assert definition.estimator.named_steps["features"].include_parameter_count is False
    assert validate_jobs(MAX_JOBS) == MAX_JOBS
    with pytest.raises(ValueError, match="between 1 and 4"):
        validate_jobs(5)


def test_exact_regret_counter_matches_brute_force_metrics() -> None:
    truth = np.asarray([1.0, 4.0, 2.0, 3.0, 2.5])
    prediction = np.asarray([3.0, 1.0, 4.0, 2.0, 2.0])
    counts, n_groups = _regret_counts(truth, prediction)
    brute = winner_metrics_3choice(truth, prediction)

    assert n_groups == brute["n_groups"]
    assert counts[0.0] / n_groups == pytest.approx(brute["accuracy"])
    assert sum(
        value * count for value, count in counts.items()
    ) / n_groups == pytest.approx(brute["mean_regret"])


def test_wide_metrics_keep_choices_within_environment_and_report_eligible() -> None:
    rows = [
        _row(
            "env_a",
            "family_a",
            "dataset_a",
            10 + index,
            split="validation",
            cohort="development",
        )
        for index in range(4)
    ] + [
        _row(
            "env_b",
            "family_a",
            "dataset_b",
            20 + index,
            split="validation",
            cohort="development",
        )
        for index in range(4)
    ]
    for offset, row in enumerate(rows):
        mean_loss = 0.4 + 0.1 * (offset % 4)
        row["target"]["mean_loss"] = mean_loss
        row["target"]["log_mean_loss"] = math.log(mean_loss)
    truth = np.asarray([row["target"]["log_mean_loss"] for row in rows])
    result = score_wide_predictions(rows, truth)

    assert result["all"]["log"]["rmse"] == pytest.approx(0.0)
    choice = result["all"]["within_environment"]["three_choice"]
    assert choice["n_groups"] == 2 * math.comb(4, 3)
    assert choice["accuracy"] == pytest.approx(1.0)
    assert choice["log_regret"] == {"mean": 0.0, "median": 0.0}
    assert choice["gap_ge_0_05"]["accuracy"] == pytest.approx(1.0)
    assert result["all"]["within_environment"]["macro"][
        "three_choice_accuracy"
    ] == pytest.approx(1.0)
    assert set(result["all"]["per_environment"]) == {"env_a", "env_b"}
    assert result["all"]["per_environment"]["env_a"]["raw"]["rmse"] == pytest.approx(
        0.0
    )
    assert result["benchmark_eligible"] is not None


def test_five_seed_ceiling_uses_all_two_three_splits_and_can_skip(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root,
        "env_a",
        "dataset_a",
        first_index=100,
        cohort="development",
        write_curves=True,
    )
    environment = load_environment(path)

    ceiling = dynamic_split_noise_ceiling(environment)
    skipped = noise_ceiling_for_environment(environment, mode="skip")

    assert ceiling["n_seeds"] == 5
    assert ceiling["split_sizes"] == [2, 3]
    assert ceiling["n_complementary_partitions"] == math.comb(5, 2)
    assert ceiling["n_directed_comparisons"] == 20
    assert skipped == {
        "status": "skipped",
        "reason": "disabled_by_user",
        "n_seeds": 5,
    }


def test_minimal_id_runner_writes_digest_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, plan = _study_fixture(tmp_path)
    output = tmp_path / "study"
    kwargs = {
        "jobs": 1,
        "seed": 11,
        "n_splits": 2,
        "method_names": {"constant_mean", "params_ols"},
        "include_xgboost": False,
        "scopes": {"environment", "family"},
        "force": False,
        "noise_ceiling": "skip",
        "require_complete": True,
    }

    manifest = fit_wide_id(dataset_root, output, plan, **kwargs)
    assert manifest["n_seeds"] == 5
    assert manifest["n_environments"] == 3
    assert manifest["jobs"] == 1
    assert manifest["thread_limit"] == 1
    assert manifest["results"]["environment"] == {"n_tasks": 3, "n_test": 9}
    assert manifest["results"]["family"] == {"n_tasks": 1, "n_test": 9}
    sidecar = json.loads(
        (output / "id/environment/env_a/models/params_ols.json").read_text("utf-8")
    )
    assert sidecar["plan_sha256"] == manifest["plan_sha256"]
    assert set(sidecar["source_hashes"]) == {
        "wide_loader",
        "wide_runner",
        "feature_encoder",
        "model_estimators",
    }
    assert sidecar["method_config"]["name"] == "params_ols"
    assert len(sidecar["input_digest"]) == 64

    def fail_if_refit(*args: object, **kwargs: object) -> None:
        raise AssertionError("valid checkpoints should have resumed")

    monkeypatch.setattr("tools.meta_model_study.wide_run.fit_definition", fail_if_refit)
    resumed = fit_wide_id(dataset_root, output, plan, **kwargs)
    assert resumed["results"] == manifest["results"]


def test_minimal_id_runner_filters_complete_phase_and_excludes_params(
    tmp_path: Path,
) -> None:
    dataset_root, plan = _study_fixture(tmp_path)
    output = tmp_path / "study_no_params"

    manifest = fit_wide_id(
        dataset_root,
        output,
        plan,
        jobs=1,
        seed=11,
        n_splits=2,
        method_names={"constant_mean", "compact_ols"},
        include_xgboost=False,
        scopes={"environment", "family"},
        force=False,
        noise_ceiling="skip",
        require_complete=False,
        phases={"test"},
        include_parameter_count=False,
    )

    assert manifest["phases"] == ["test"]
    assert manifest["include_parameter_count"] is False
    assert manifest["n_environments"] == 3
    assert manifest["methods"] == ["constant_mean", "compact_ols"]


def test_id_runner_consumes_snapshot_and_records_snapshot_provenance(
    tmp_path: Path,
) -> None:
    dataset_root, _ = _study_fixture(tmp_path)
    snapshot_path = tmp_path / "completed_snapshot.json"
    freeze_snapshot_manifest(
        [dataset_root / name for name in ("env_a", "env_b", "env_c")],
        snapshot_path,
    )
    output = tmp_path / "snapshot_study"

    manifest = fit_wide_id(
        None,
        output,
        None,
        jobs=1,
        seed=11,
        n_splits=2,
        method_names={"constant_mean"},
        include_xgboost=False,
        scopes={"environment", "family"},
        force=False,
        noise_ceiling="skip",
        require_complete=True,
        snapshot_manifest=snapshot_path,
    )

    assert manifest["dataset_root"] is None
    assert manifest["plan_path"] is None
    assert manifest["plan_sha256"] is None
    assert manifest["snapshot_manifest_path"] == str(snapshot_path.resolve())
    assert len(manifest["snapshot_manifest_sha256"]) == 64
    assert manifest["dataset_validation"]["source"] == "snapshot_manifest"
    assert manifest["dataset_validation"]["counts"]["all"] == 27
    assert manifest["n_environments"] == 3
    sidecar = json.loads(
        (output / "id/environment/env_a/models/constant_mean.json").read_text("utf-8")
    )
    assert sidecar["plan_sha256"] == manifest["snapshot_manifest_sha256"]


def test_snapshot_cli_is_mutually_exclusive_with_dataset_root_plan_and_phases(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fit-id",
                "--dataset-root",
                str(tmp_path / "wide"),
                "--snapshot-manifest",
                str(tmp_path / "snapshot.json"),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "fit-id",
                "--snapshot-manifest",
                str(tmp_path / "snapshot.json"),
                "--plan",
                str(tmp_path / "plan.json"),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "fit-id",
                "--snapshot-manifest",
                str(tmp_path / "snapshot.json"),
                "--phases",
                "test",
            ]
        )


def test_grouped_runner_holds_out_complete_groups_and_candidate_dataset(
    tmp_path: Path,
) -> None:
    dataset_root, plan = _study_fixture(tmp_path)
    output = tmp_path / "study"
    manifest = fit_wide_grouped(
        dataset_root,
        output,
        plan,
        jobs=1,
        seed=13,
        n_splits=2,
        method_names={"constant_mean"},
        include_xgboost=False,
        protocols={"environment", "dataset", "holdout_candidate"},
        force=False,
        require_complete=True,
    )

    assert manifest["results"]["environment"] == {"n_tasks": 3, "n_test": 27}
    assert manifest["results"]["dataset"] == {"n_tasks": 3, "n_test": 27}
    assert manifest["results"]["holdout_candidate"] == {
        "n_tasks": 1,
        "n_test": 9,
    }
    leaderboard = json.loads(
        (output / "ood/holdout_candidate/family_a/leaderboard.json").read_text("utf-8")
    )
    assert leaderboard["protocol"]["held_out_datasets"] == ["dataset_c"]
    assert leaderboard["test_used_for_selection"] is False
    assert leaderboard["n_train"] == 12
    assert leaderboard["n_test"] == 9


def test_grouped_runner_holds_out_complete_family_and_writes_validation_view(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    specifications = [
        ("env_a", "family_a", "dataset_a", 100),
        ("env_b", "family_b", "dataset_b", 200),
        ("env_c", "family_c", "dataset_c", 300),
    ]
    for experiment, family, dataset, index in specifications:
        _write_environment(
            root,
            experiment,
            dataset,
            first_index=index,
            cohort="development",
            family=family,
        )
    plan = _write_plan(tmp_path, [item[0] for item in specifications])

    output = tmp_path / "family_logo"
    manifest = fit_wide_grouped(
        root,
        output,
        plan,
        jobs=1,
        seed=13,
        n_splits=5,
        method_names={"constant_mean"},
        include_xgboost=False,
        protocols={"family"},
        force=False,
        require_complete=True,
    )

    assert manifest["results"]["family"] == {"n_tasks": 3, "n_test": 27}
    aggregate = json.loads(
        (output / "ood/family_logo/aggregate.json").read_text("utf-8")
    )
    assert aggregate["protocol"]["name"] == "leave_one_family_out"
    validation = json.loads(
        (
            output / "ood/family_logo/locked_validation_aggregate.json"
        ).read_text("utf-8")
    )
    assert validation["protocol"]["name"] == (
        "leave_one_family_out_locked_validation"
    )
    assert validation["n_test"] == 9

    leaderboard = json.loads(
        (output / "ood/family_logo/family_a/leaderboard.json").read_text("utf-8")
    )
    assert leaderboard["n_train"] == 12
    assert leaderboard["n_test"] == 9
    assert leaderboard["protocol"]["held_out_family"] == "family_a"
    assert leaderboard["protocol"]["training_families"] == ["family_b", "family_c"]
    assert leaderboard["inner_cv"]["kind"] == "group_kfold"
    assert leaderboard["inner_cv"]["axis"] == "family"
    assert leaderboard["inner_cv"]["effective_splits"] == 2
    assert leaderboard["inner_cv"]["groups"] == ["family_b", "family_c"]


def test_grouped_runner_rejects_family_logo_with_only_two_families(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    specifications = [
        ("env_a", "family_a", "dataset_a", 100),
        ("env_b", "family_b", "dataset_b", 200),
    ]
    for experiment, family, dataset, index in specifications:
        _write_environment(
            root,
            experiment,
            dataset,
            first_index=index,
            cohort="development",
            family=family,
        )
    plan = _write_plan(tmp_path, [item[0] for item in specifications])

    with pytest.raises(ValueError, match="at least three families"):
        fit_wide_grouped(
            root,
            tmp_path / "family_logo",
            plan,
            jobs=1,
            seed=13,
            n_splits=5,
            method_names={"constant_mean"},
            include_xgboost=False,
            protocols={"family"},
            force=False,
            require_complete=True,
        )


def test_dataset_conditioning_is_in_manifest_and_checkpoint_digest(tmp_path: Path) -> None:
    dataset_root, plan = _study_fixture(tmp_path)
    kwargs = dict(
        jobs=1,
        seed=7,
        n_splits=2,
        method_names={"compact_ols"},
        include_xgboost=False,
        scopes={"environment"},
        force=False,
        noise_ceiling="skip",
        require_complete=True,
    )
    unaware = fit_wide_id(dataset_root, tmp_path / "unaware", plan, **kwargs)
    described = fit_wide_id(
        dataset_root,
        tmp_path / "described",
        plan,
        dataset_conditioning="description",
        **kwargs,
    )
    assert unaware["dataset_conditioning"] == "unaware"
    assert described["dataset_conditioning"] == "description"
    unaware_sidecar = next((tmp_path / "unaware").glob("id/environment/*/models/compact_ols.json"))
    described_sidecar = next((tmp_path / "described").glob("id/environment/*/models/compact_ols.json"))
    first = json.loads(unaware_sidecar.read_text("utf-8"))
    second = json.loads(described_sidecar.read_text("utf-8"))
    assert first["input_digest"] != second["input_digest"]
    assert second["method_config"]["dataset_conditioning"] == "description"
