from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.meta_model_study import ood


def _row(
    index: int,
    *,
    family: str = "bigram_lm",
    optimizer: str = "Adam",
    lr: float = 0.001,
    size: int = 32,
) -> dict:
    if family == "bigram_lm":
        model = {
            "type": "transformer_lm",
            "d_model": size,
            "d_ff": 2 * size,
            "num_layers": 2,
            "num_heads": 2,
            "context_length": 16,
            "vocab_size": 32,
        }
        loss_id = "cross_entropy"
    else:
        model = {
            "type": "mlp",
            "input_dim": 3,
            "depth": 2,
            "width": size,
            "residual": False,
            "activations": ["relu", "gelu"],
            "layer_norm": [False, True],
        }
        loss_id = "mse"
    total_params = 1000 + index + size
    mean_loss = 1.0 + index / 100.0
    return {
        "experiment_id": "experiment",
        "family": family,
        "split": "train" if index % 2 else "validation",
        "usable_for_regression": True,
        "example_fingerprint_sha256": f"{index:064x}",
        "setting": {
            "model": model,
            "optimizer": {
                "type": optimizer,
                "lr": lr,
                "weight_decay": 0.0,
            },
            "loss": {"loss_id": loss_id},
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
            "mean_loss": mean_loss,
            "log_mean_loss": math.log(mean_loss),
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _brute_three_choice(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int]:
    correct = 0
    total = 0
    for indices in itertools.combinations(range(truth.size), 3):
        selected = indices[int(np.argmin(prediction[list(indices)]))]
        correct += truth[selected] == np.min(truth[list(indices)])
        total += 1
    return total, correct


def test_group_folds_are_exhaustive_disjoint_and_declared() -> None:
    rows = [
        _row(
            index,
            optimizer=optimizer,
            lr=lr,
            size=32 if index % 2 else 64,
        )
        for index, (optimizer, lr) in enumerate(
            itertools.product(("Adam", "SGD"), (0.001, 0.01))
        )
        for _ in range(3)
    ]
    # Restore unique IDs after repeating each grid cell three times.
    for index, row in enumerate(rows):
        row["example_fingerprint_sha256"] = f"{index:064x}"

    declared = (
        "optimizer=Adam|lr=0.001",
        "optimizer=Adam|lr=0.01",
        "optimizer=SGD|lr=0.001",
        "optimizer=SGD|lr=0.01",
    )
    folds = ood.make_group_folds(
        rows,
        "leave_one_optimizer_lr_cell_out",
        declared_groups=declared,
    )

    assert [fold.group for fold in folds] == list(declared)
    assert sorted(np.concatenate([fold.test_indices for fold in folds])) == list(
        range(len(rows))
    )
    for fold in folds:
        train_groups = {
            ood.group_value(rows[index], "leave_one_optimizer_lr_cell_out")
            for index in fold.train_indices
        }
        test_groups = {
            ood.group_value(rows[index], "leave_one_optimizer_lr_cell_out")
            for index in fold.test_indices
        }
        assert test_groups == {fold.group}
        assert fold.group not in train_groups

    with pytest.raises(ValueError, match="group grid mismatch"):
        ood.make_group_folds(
            rows,
            "leave_one_optimizer_lr_cell_out",
            declared_groups=declared[:-1],
        )


@pytest.mark.parametrize("seed", range(5))
def test_exact_three_choice_matches_brute_force_with_ties(seed: int) -> None:
    generator = np.random.default_rng(seed)
    truth = generator.integers(0, 4, size=11).astype(float)
    prediction = generator.integers(0, 5, size=11).astype(float)
    expected_total, expected_correct = _brute_three_choice(truth, prediction)

    actual = ood.exact_three_choice_accuracy(truth, prediction)

    assert actual["n_groups"] == expected_total
    assert actual["n_correct"] == expected_correct
    assert actual["accuracy"] == pytest.approx(expected_correct / expected_total)


def test_three_choice_requires_three_rows_and_metrics_are_log_space() -> None:
    assert ood.exact_three_choice_accuracy([1.0, 2.0], [1.0, 2.0]) == {
        "n_groups": 0,
        "n_correct": 0,
        "accuracy": None,
    }
    metrics = ood.prediction_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert metrics["log_rmse"] == 0.0
    assert metrics["log_r2"] == 1.0
    assert metrics["spearman"] == 1.0
    assert metrics["three_choice"]["accuracy"] == 1.0


def test_run_study_smoke_writes_resumable_json_and_markdown(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    experiment = dataset_root / "experiment"
    rows = [
        _row(index, size=size)
        for index, size in enumerate((32, 32, 32, 64, 64, 64, 128, 128, 128))
    ]
    _write_jsonl(experiment / "all.jsonl", rows)
    (experiment / "manifest.json").write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"

    result = ood.run_study(
        dataset_root=dataset_root,
        output_root=output_root,
        protocols=("leave_one_size_out",),
        methods=("params_ridge",),
        expected_row_count=None,
        seed=17,
    )

    assert (output_root / "results.json").is_file()
    assert (output_root / "report.md").is_file()
    experiment_result = result["experiments"]["experiment"]
    method = experiment_result["protocols"]["leave_one_size_out"]["methods"][
        "params_ridge"
    ]
    assert method["micro"]["n"] == 9
    assert method["micro"]["three_choice"]["n_groups"] == 3
    assert method["micro"]["three_choice"]["scope"] == "within_held_group_only"
    assert len(method["predictions"]) == 9
    checkpoints = list((output_root / "checkpoints").rglob("*.json"))
    assert len(checkpoints) == 3

    # A second run must accept and reproduce the fold checkpoints.
    rerun = ood.run_study(
        dataset_root=dataset_root,
        output_root=output_root,
        protocols=("leave_one_size_out",),
        methods=("params_ridge",),
        expected_row_count=None,
        seed=17,
    )
    rerun_method = rerun["experiments"]["experiment"]["protocols"][
        "leave_one_size_out"
    ]["methods"]["params_ridge"]
    assert rerun_method["predictions"] == method["predictions"]


def test_expected_groups_are_profile_predeclared() -> None:
    assert ood.expected_groups(
        "bigram_lm", "leave_one_size_out"
    ) == ("d_model=32", "d_model=64", "d_model=128")
    assert len(
        ood.expected_groups(
            "univariate_regression", "leave_one_optimizer_lr_cell_out"
        )
    ) == 25
