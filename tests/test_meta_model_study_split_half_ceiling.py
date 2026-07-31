from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.meta_model_study import split_half_ceiling
from tools.meta_model_study.split_half_ceiling import main, summarize_split_half
from tools.meta_model_study.wide import WideEnvironment


def _environment(
    tmp_path: Path,
    experiment_id: str,
    family: str,
    *,
    write_curves: bool,
) -> WideEnvironment:
    root = tmp_path / "wide"
    environment_path = root / experiment_id
    factors = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
    rows: list[dict] = []
    for index, mean_loss in enumerate((0.4, 0.6, 0.8, 1.0)):
        candidate_path = f"{experiment_id}/candidates/c_{index}"
        rows.append(
            {
                "example_fingerprint_sha256": f"{index + 1:064x}",
                "provenance": {"candidate_path": candidate_path},
                "target": {
                    "mean_loss": mean_loss,
                    "failed_seeds": 0,
                    "benchmark_eligible": index != 0,
                },
            }
        )
        if write_curves:
            results = root / candidate_path / "results"
            results.mkdir(parents=True, exist_ok=True)
            final_losses = mean_loss * factors
            np.savez(
                results / "curves.npz",
                curves=np.column_stack([final_losses * 1.1, final_losses]),
            )
    return WideEnvironment(
        path=environment_path,
        experiment_id=experiment_id,
        family=family,
        dataset_id=f"dataset_{experiment_id}",
        n_seeds=5,
        manifest={},
        all_rows=tuple(rows),
        train_rows=(),
        validation_rows=tuple(rows),
    )


def _snapshot(tmp_path: Path, environments: tuple[WideEnvironment, ...]):
    manifest = (tmp_path / "snapshot.json").resolve()
    return SimpleNamespace(
        path=manifest,
        sha256="a" * 64,
        corpus=SimpleNamespace(environments=environments),
    )


def _metric_tree(
    *,
    three_choice: float,
    gap: float,
    pair: float,
    spearman: float | None,
    log_rmse: float,
) -> dict:
    return {
        "log": {"rmse": log_rmse},
        "ranking": {
            "pair_concordance": pair,
            "spearman": spearman,
        },
        "three_choice": {
            "accuracy": three_choice,
            "gap_ge_0_05": {"accuracy": gap},
        },
    }


def _ceiling(all_metrics: dict, eligible_metrics: dict | None) -> dict:
    return {
        "status": "computed",
        "n_seeds": 5,
        "split_sizes": [2, 3],
        "n_complementary_partitions": 10,
        "n_directed_comparisons": 20,
        "all": {"n_rows": 4, "median_metrics": all_metrics},
        "benchmark_eligible": (
            {"n_rows": 3, "median_metrics": eligible_metrics}
            if eligible_metrics is not None
            else None
        ),
    }


def test_reads_stored_curves_and_labels_estimate_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, "env_curves", "family_a", write_curves=True)
    monkeypatch.setattr(
        split_half_ceiling,
        "load_snapshot",
        lambda _path: _snapshot(tmp_path, (environment,)),
    )

    result = summarize_split_half(tmp_path / "snapshot.json")

    assert result["snapshot_sha256"] == "a" * 64
    assert result["coverage"]["overall"]["computed_environments"] == 1
    assert result["aggregation"]["stored_seed_curves_only"] is True
    assert result["estimate"]["is_mathematical_upper_bound"] is False
    assert "empirical label-reproducibility estimate" in result["estimate"]["caveat"]
    all_macro = result["all"]["overall"]["environment_equal_macro"]
    assert all_macro["three_choice_accuracy"] == pytest.approx(1.0)
    assert all_macro["gap_ge_0_05_three_choice_accuracy"] == pytest.approx(1.0)
    assert all_macro["pair_concordance"] == pytest.approx(1.0)
    assert all_macro["spearman"] == pytest.approx(1.0)
    assert all_macro["log_rmse"] is not None
    assert result["environments"][0]["n_directed_comparisons"] == 20


def test_environment_equal_macros_family_groups_and_skip_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environments = (
        _environment(tmp_path, "env_a1", "family_a", write_curves=False),
        _environment(tmp_path, "env_a2", "family_a", write_curves=False),
        _environment(tmp_path, "env_b1", "family_b", write_curves=False),
    )
    monkeypatch.setattr(
        split_half_ceiling,
        "load_snapshot",
        lambda _path: _snapshot(tmp_path, environments),
    )
    all_a1 = _metric_tree(
        three_choice=0.8,
        gap=0.9,
        pair=0.7,
        spearman=0.6,
        log_rmse=0.2,
    )
    all_a2 = _metric_tree(
        three_choice=0.6,
        gap=0.7,
        pair=0.5,
        spearman=None,
        log_rmse=0.4,
    )
    eligible_a1 = _metric_tree(
        three_choice=0.75,
        gap=0.85,
        pair=0.65,
        spearman=0.55,
        log_rmse=0.25,
    )
    modes: list[str] = []

    def fake_ceiling(environment: WideEnvironment, *, mode: str) -> dict:
        modes.append(mode)
        if environment.experiment_id == "env_a1":
            return _ceiling(all_a1, eligible_a1)
        if environment.experiment_id == "env_a2":
            return _ceiling(all_a2, None)
        return {
            "status": "skipped",
            "reason": "stored_seed_curves_unavailable: missing fixture",
            "n_seeds": 5,
        }

    monkeypatch.setattr(
        split_half_ceiling, "noise_ceiling_for_environment", fake_ceiling
    )

    result = summarize_split_half(tmp_path / "snapshot.json", missing_curves="skip")

    assert modes == ["auto", "auto", "auto"]
    assert result["coverage"]["overall"] == {
        "total_environments": 3,
        "computed_environments": 2,
        "skipped_environments": 1,
        "computed_fraction": pytest.approx(2 / 3),
        "status_counts": {"computed": 2, "skipped": 1},
    }
    all_overall = result["all"]["overall"]
    assert all_overall["environment_equal_macro"] == {
        "three_choice_accuracy": pytest.approx(0.7),
        "gap_ge_0_05_three_choice_accuracy": pytest.approx(0.8),
        "pair_concordance": pytest.approx(0.6),
        "spearman": pytest.approx(0.6),
        "log_rmse": pytest.approx(0.3),
    }
    assert all_overall["metric_coverage"]["spearman"] == {
        "computed_environments": 1,
        "skipped_environments": 2,
        "computed_fraction": pytest.approx(1 / 3),
    }
    assert result["all"]["by_family"]["family_a"]["environment_equal_macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.7)
    assert result["all"]["by_family"]["family_b"]["coverage"] == {
        "total_environments": 1,
        "computed_environments": 0,
        "skipped_environments": 1,
        "computed_fraction": 0.0,
    }
    eligible = result["benchmark_eligible"]["overall"]
    assert eligible["coverage"]["computed_environments"] == 1
    assert eligible["environment_equal_macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.75)


def test_require_fails_clearly_but_skip_records_missing_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, "env_missing", "family_a", write_curves=False)
    monkeypatch.setattr(
        split_half_ceiling,
        "load_snapshot",
        lambda _path: _snapshot(tmp_path, (environment,)),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Stored seed curves are required for env_missing: Missing stored GT curves",
    ):
        summarize_split_half(tmp_path / "snapshot.json", missing_curves="require")

    result = summarize_split_half(tmp_path / "snapshot.json", missing_curves="skip")
    assert result["coverage"]["overall"]["skipped_environments"] == 1
    assert result["environments"][0]["status"] == "skipped"
    assert result["environments"][0]["reason"].startswith(
        "stored_seed_curves_unavailable: Missing stored GT curves"
    )


def test_cli_writes_only_the_requested_summary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "reports" / "split_half.json"
    calls: list[tuple[Path, str]] = []

    def fake_summary(path: Path, *, missing_curves: str) -> dict:
        calls.append((path, missing_curves))
        return {"snapshot_sha256": "b" * 64}

    monkeypatch.setattr(split_half_ceiling, "summarize_split_half", fake_summary)

    assert (
        main(
            [
                "--snapshot-manifest",
                str(tmp_path / "snapshot.json"),
                "--missing-curves",
                "skip",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert calls == [(tmp_path / "snapshot.json", "skip")]
    assert json.loads(output.read_text("utf-8")) == {"snapshot_sha256": "b" * 64}
