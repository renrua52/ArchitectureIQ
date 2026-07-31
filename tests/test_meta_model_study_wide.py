from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.meta_model_study.features import FeatureEncoder
from tools.meta_model_study.freeze_wide_snapshot import main as freeze_snapshot_main
from tools.meta_model_study.wide import (
    GroupFold,
    WideGroupFold,
    freeze_snapshot_manifest,
    group_protocol_manifest,
    load_corpus,
    load_environment,
    load_seed_losses,
    load_snapshot,
    main,
    make_group_folds,
    validate_root,
    within_environment_metrics,
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
    n_seeds: int = 5,
) -> dict:
    total_params = 100 + index
    mean_loss = 0.5 + index / 100.0
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "family": family,
        "dataset_id": dataset_id,
        "split": split,
        "stratum": "optimizer.type=Adam|loss.loss_id=mse",
        "usable_for_regression": True,
        "example_fingerprint_sha256": f"{index:064x}",
        "group_labels": {
            "environment": experiment_id,
            "dataset": dataset_id,
            "family": family,
            "phase": "test",
            "dataset_cohort": "development",
        },
        "setting": {
            "model": {
                "type": "mlp",
                "input_dim": 2,
                "depth": 1,
                "width": 8,
                "residual": False,
                "activations": ["relu"],
                "layer_norm": [False],
            },
            "optimizer": {"type": "Adam", "lr": 0.001, "weight_decay": 0.0},
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
            "std_loss": 0.0,
            "n_seeds": n_seeds,
            "failed_seeds": 0,
            "benchmark_eligible": True,
        },
        "provenance": {
            "candidate_path": f"{experiment_id}/candidates/c_{index}",
        },
    }


def _write_environment(
    root: Path,
    experiment_id: str,
    family: str,
    dataset_id: str,
    *,
    first_index: int,
    n_seeds: int = 5,
) -> Path:
    path = root / experiment_id
    group_labels = {
        "environment": experiment_id,
        "dataset": dataset_id,
        "family": family,
        "phase": "test",
        "dataset_cohort": "development",
    }
    train = [
        _row(
            experiment_id,
            family,
            dataset_id,
            first_index + index,
            split="train",
            n_seeds=n_seeds,
        )
        for index in range(3)
    ]
    validation = [
        _row(
            experiment_id,
            family,
            dataset_id,
            first_index + 3,
            split="validation",
            n_seeds=n_seeds,
        )
    ]
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
                "ground_truth": {"n_seeds": n_seeds},
                "dataset_spec": {"selection_metric": "test_mse"},
                "dataset_path": f"data/datasets/{family}/{dataset_id}",
                "budget": 128,
                "batch_size": 8,
                "num_rows": len(train) + len(validation),
                "train_rows": len(train),
                "phase": "test",
                "group_labels": group_labels,
            },
            "ground_truth": {"n_seeds": n_seeds},
            "split_policy": {
                "assigned_before_ground_truth": True,
                "group_labels_frozen_before_ground_truth": True,
                "train": len(train),
                "validation": len(validation),
                "group_labels": group_labels,
            },
            "selected": {"total": len(train) + len(validation)},
        },
    )
    return path


def _write_plan(path: Path, experiment_ids: list[str], *, n_seeds: int = 5) -> None:
    _write_json(
        path,
        {
            "schema_version": "2.0",
            "defaults": {
                "phase": "test",
                "num_rows": 4,
                "train_rows": 3,
                "ground_truth": {"n_seeds": n_seeds},
            },
            "experiments": [
                {
                    "experiment_id": experiment_id,
                    "dataset_path": "data/datasets/family_a/dataset_a",
                    "budget": 128,
                    "batch_size": 8,
                }
                for experiment_id in experiment_ids
            ],
        },
    )


def test_load_environment_and_corpus_support_manifest_declared_five_seeds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    first = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    _write_environment(root, "env_b", "family_a", "dataset_a", first_index=20)

    environment = load_environment(first)
    corpus = load_corpus(root, expected_environment_count=2, expected_n_seeds=5)

    assert environment.n_seeds == corpus.n_seeds == 5
    assert len(environment.all_rows) == 4
    assert len(corpus.all_rows) == 8
    assert len(corpus.train_rows) == 6
    assert len(corpus.validation_rows) == 2
    matrix = FeatureEncoder("compact").fit_transform(corpus.train_rows)
    assert matrix.shape[0] == 6
    assert np.all(np.isfinite(matrix))


def test_frozen_snapshot_loads_cross_root_environments_and_checks_contracts(
    tmp_path: Path,
) -> None:
    first = _write_environment(
        tmp_path / "base",
        "env_a",
        "family_a",
        "dataset_a",
        first_index=10,
    )
    second = _write_environment(
        tmp_path / "extended",
        "env_b",
        "family_b",
        "dataset_b",
        first_index=20,
    )
    snapshot_path = tmp_path / "snapshots" / "completed.json"

    frozen = freeze_snapshot_manifest([first, second], snapshot_path)
    snapshot = load_snapshot(snapshot_path)

    assert frozen["counts"] == {
        "environments": 2,
        "all": 8,
        "train": 6,
        "validation": 2,
    }
    assert [item.experiment_id for item in snapshot.corpus.environments] == [
        "env_a",
        "env_b",
    ]
    assert snapshot.corpus.root == snapshot_path.parent.resolve()
    assert len(snapshot.sha256) == 64
    assert all(not Path(item["path"]).is_absolute() for item in frozen["environments"])

    (first / "all.jsonl").write_text(
        (first / "all.jsonl").read_text("utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="all.jsonl SHA-256 mismatch"):
        load_snapshot(snapshot_path)


def test_frozen_snapshot_rejects_row_count_and_cross_environment_collision(
    tmp_path: Path,
) -> None:
    first = _write_environment(
        tmp_path / "base",
        "env_a",
        "family_a",
        "dataset_a",
        first_index=10,
    )
    second = _write_environment(
        tmp_path / "extended",
        "env_b",
        "family_b",
        "dataset_b",
        first_index=20,
    )
    snapshot_path = tmp_path / "completed.json"
    manifest = freeze_snapshot_manifest([first, second], snapshot_path)
    manifest["environments"][0]["files"]["train.jsonl"]["rows"] += 1
    _write_json(snapshot_path, manifest)
    with pytest.raises(ValueError, match="train.jsonl row count mismatch"):
        load_snapshot(snapshot_path)

    duplicate = _write_environment(
        tmp_path / "duplicate",
        "env_c",
        "family_c",
        "dataset_c",
        first_index=10,
    )
    with pytest.raises(ValueError, match="fingerprints collide across environments"):
        freeze_snapshot_manifest([first, duplicate], snapshot_path)


def test_freeze_snapshot_cli_reports_path_hash_and_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = _write_environment(
        tmp_path / "base",
        "env_a",
        "family_a",
        "dataset_a",
        first_index=10,
    )
    snapshot_path = tmp_path / "completed.json"

    assert (
        freeze_snapshot_main(
            [
                "--output",
                str(snapshot_path),
                "--environment",
                str(environment),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["snapshot_manifest_path"] == str(snapshot_path.resolve())
    assert len(output["snapshot_manifest_sha256"]) == 64
    assert output["counts"]["environments"] == 1


def test_load_environment_rejects_seed_split_and_parameter_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    rows = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
    rows[0]["target"]["n_seeds"] = 10
    _write_jsonl(path / "train.jsonl", rows)
    with pytest.raises(ValueError, match="target.n_seeds=10"):
        load_environment(path)

    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    rows = [json.loads(line) for line in (path / "all.jsonl").read_text().splitlines()]
    rows[0]["derived"]["log_total_params"] += 1.0
    _write_jsonl(path / "all.jsonl", rows)
    with pytest.raises(ValueError, match="inconsistent total_params"):
        load_environment(path)


def test_all_five_group_labels_match_rows_config_and_split_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    manifest["split_policy"]["group_labels"]["dataset_cohort"] = "other"
    _write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="config/split_policy group_labels disagree"):
        load_environment(path)

    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    rows = [json.loads(line) for line in (path / "all.jsonl").read_text().splitlines()]
    rows[0]["group_labels"]["phase"] = "other"
    _write_jsonl(path / "all.jsonl", rows)
    with pytest.raises(ValueError, match="group label 'phase' disagrees"):
        load_environment(path)

    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    del manifest["config"]["group_labels"]["dataset_cohort"]
    _write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dataset_cohort must be a non-empty string"):
        load_environment(path)


def test_non_mapping_manifest_config_is_reported_invalid(tmp_path: Path) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    manifest["config"] = []
    _write_json(path / "manifest.json", manifest)

    report = validate_root(root)

    assert report["status"] == "invalid"
    assert "manifest.config must be a mapping" in report["validation_errors"]["env_a"]


def test_seed_curve_loader_uses_environment_seed_count(tmp_path: Path) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    environment = load_environment(path)
    row = environment.validation_rows[0]
    mean_loss = row["target"]["mean_loss"]
    results = root / row["provenance"]["candidate_path"] / "results"
    results.mkdir(parents=True)
    np.savez(results / "curves.npz", curves=np.full((5, 2), mean_loss))

    losses = load_seed_losses(environment)

    assert losses.shape == (1, 5)
    assert losses == pytest.approx(mean_loss)


def test_seed_curve_loader_rejects_partial_failed_seed_coverage(tmp_path: Path) -> None:
    root = tmp_path / "wide"
    path = _write_environment(
        root, "env_a", "family_a", "dataset_a", first_index=10
    )
    for filename in ("all.jsonl", "validation.jsonl"):
        rows = [json.loads(line) for line in (path / filename).read_text().splitlines()]
        rows[-1]["target"]["failed_seeds"] = 1
        _write_jsonl(path / filename, rows)
    environment = load_environment(path)

    with pytest.raises(ValueError, match="requires failed_seeds == 0"):
        load_seed_losses(environment)


def test_group_protocols_are_exhaustive_for_environment_dataset_and_family(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    specifications = [
        ("env_a1", "family_a", "dataset_a", 10),
        ("env_a2", "family_a", "dataset_a", 20),
        ("env_b1", "family_b", "dataset_b", 30),
        ("env_b2", "family_b", "dataset_c", 40),
    ]
    for experiment, family, dataset, index in specifications:
        _write_environment(root, experiment, family, dataset, first_index=index)
    corpus = load_corpus(root, expected_environment_count=4)

    expected_folds = {"environment": 4, "dataset": 3, "family": 2}
    assert GroupFold is WideGroupFold
    for axis, count in expected_folds.items():
        folds = make_group_folds(corpus.train_rows, axis)
        assert all(isinstance(fold, WideGroupFold) for fold in folds)
        assert len(folds) == count
        assert sorted(np.concatenate([fold.test_indices for fold in folds]).tolist()) == list(
            range(len(corpus.train_rows))
        )
        for fold in folds:
            train_groups = {
                corpus.train_rows[index]["group_labels"][axis]
                for index in fold.train_indices
            }
            test_groups = {
                corpus.train_rows[index]["group_labels"][axis]
                for index in fold.test_indices
            }
            assert train_groups.isdisjoint(test_groups)

    protocol = group_protocol_manifest(corpus.train_rows)
    assert {axis: protocol[axis]["n_folds"] for axis in protocol} == expected_folds


def test_choice_and_ranking_metrics_never_form_cross_environment_groups() -> None:
    rows = [
        _row("env_a", "family", "dataset", 10 + index, split="validation")
        for index in range(3)
    ] + [
        _row("env_b", "family", "dataset", 20 + index, split="validation")
        for index in range(3)
    ]
    rows[0]["target"]["benchmark_eligible"] = False
    truth = np.asarray([row["target"]["log_mean_loss"] for row in rows])
    predicted = np.concatenate([truth[:3], truth[3:][::-1]])

    result = within_environment_metrics(rows, predicted)

    assert result["scope"] == "within_environment_only"
    assert result["all"]["three_choice"] == result["three_choice"]
    assert result["n_environments"] == 2
    assert result["three_choice"] == {
        "n_groups": 2,
        "n_correct": 1,
        "accuracy": 0.5,
        "scope": "all_triples_within_each_environment",
    }
    assert result["within_environment_ranking"]["pair_concordance"] == pytest.approx(
        0.5
    )
    assert result["within_environment_ranking"]["macro_spearman"] == pytest.approx(0.0)
    assert result["benchmark_eligible"]["n_rows"] == 5
    assert result["benchmark_eligible"]["three_choice"] == {
        "n_groups": 1,
        "n_correct": 0,
        "accuracy": 0.0,
        "scope": "all_triples_within_each_environment",
    }


def test_validate_cli_reports_partial_without_loading_it_as_complete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "wide"
    _write_environment(root, "env_a", "family_a", "dataset_a", first_index=10)
    partial = root / "env_b"
    _write_json(partial / "sampling_manifest.json", {"experiment_id": "env_b"})
    plan = tmp_path / "plan.json"
    _write_plan(plan, ["env_a", "env_b"])

    report = validate_root(root, plan_path=plan)
    assert report["status"] == "partial"
    assert report["counts"] == {
        "environments": 1,
        "all": 4,
        "train": 3,
        "validation": 1,
    }
    assert report["completed_environments"] == ["env_a"]
    assert report["partial_environments"] == ["env_b"]
    assert report["validation_errors"] == {}
    assert len(report["plan_sha256"]) == 64

    assert main(["validate", "--dataset-root", str(root), "--plan", str(plan)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "partial"
    assert (
        main(
            [
                "validate",
                "--dataset-root",
                str(root),
                "--plan",
                str(plan),
                "--require-complete",
            ]
        )
        == 1
    )


def test_plan_validation_is_per_environment_and_cannot_offset_counts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wide"
    _write_environment(root, "env_a", "family_a", "dataset_a", first_index=10)
    _write_environment(root, "env_b", "family_a", "dataset_a", first_index=20)
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "defaults": {
                "phase": "test",
                "num_rows": 4,
                "train_rows": 3,
                "ground_truth": {"n_seeds": 5},
            },
            "experiments": [
                {
                    "experiment_id": "env_a",
                    "num_rows": 5,
                    "train_rows": 4,
                    "budget": 256,
                    "batch_size": 8,
                    "phase": "test",
                    "dataset_path": "data/datasets/family_a/dataset_a",
                },
                {
                    "experiment_id": "env_b",
                    "num_rows": 3,
                    "train_rows": 2,
                    "budget": 128,
                    "batch_size": 8,
                    "phase": "test",
                    "dataset_path": "data/datasets/family_a/dataset_a",
                },
            ],
        },
    )

    report = validate_root(root, plan_path=plan)

    # Aggregate plan and data counts both equal 8/6/2, but both environments
    # are still rejected independently; env_a also catches its budget mismatch.
    assert report["expected"]["all_rows"] == 8
    assert report["expected"]["train_rows"] == 6
    assert report["expected"]["validation_rows"] == 2
    assert report["status"] == "invalid"
    assert set(report["environment_mismatches"]) == {"env_a", "env_b"}
    assert set(report["environment_mismatches"]["env_a"]) == {
        "budget",
        "num_rows",
        "train_rows",
    }
    assert set(report["environment_mismatches"]["env_b"]) == {
        "num_rows",
        "train_rows",
    }
    expected_env = report["expected"]["environments"]["env_a"]
    assert expected_env == {
        "num_rows": 5,
        "train_rows": 4,
        "validation_rows": 1,
        "budget": 256,
        "batch_size": 8,
        "phase": "test",
        "dataset_path": "data/datasets/family_a/dataset_a",
    }


def test_cli_no_plan_and_group_protocol_errors_are_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "wide"
    _write_environment(root, "env_a", "family_a", "dataset_a", first_index=10)

    assert main(["validate", "--dataset-root", str(root), "--no-plan"]) == 0
    no_plan = json.loads(capsys.readouterr().out)
    assert no_plan["status"] == "complete"
    assert no_plan["plan_path"] is None

    assert (
        main(
            [
                "validate",
                "--dataset-root",
                str(root),
                "--no-plan",
                "--include-group-protocols",
            ]
        )
        == 1
    )
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "invalid"
    assert "group_protocol_error" in failed
    assert "group_protocols" in failed["validation_errors"]
def test_wide_loader_attaches_sanitized_stable_dataset_context(tmp_path: Path) -> None:
    root = tmp_path / "wide"
    path = _write_environment(root, "env_a", "family_a", "dataset_a", first_index=10)
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    manifest["config"]["dataset_spec"].update(
        {
            "dataset_id": "secret_identity",
            "seed": 123,
            "params": {"degree": 4, "split_seed": 999},
            "files": ["private.pt"],
            "significance": {"gap": 0.1},
        }
    )
    _write_json(path / "manifest.json", manifest)

    environment = load_environment(path)
    contexts = {json.dumps(row["dataset_context"], sort_keys=True) for row in environment.all_rows}
    assert len(contexts) == 1
    assert environment.all_rows[0]["dataset_context"] == {
        "dataset_id": "dataset_a",
        "description": {
            "family": "family_a",
            "params": {"degree": 4},
            "selection_metric": "test_mse",
        },
    }
