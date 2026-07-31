from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.meta_model_study.report import generate_report, load_study_artifacts


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _metrics(log_rmse: float, spearman: float, accuracy: float) -> dict:
    return {
        "all": {
            "log": {"rmse": log_rmse},
            "raw": {"rmse": log_rmse * 2},
            "ranking": {"spearman": spearman},
            "three_choice": {"accuracy": accuracy},
        },
        "benchmark_eligible": {
            "three_choice": {"accuracy": min(1.0, accuracy + 0.05)}
        },
    }


def _experiment(
    root: Path,
    experiment_id: str,
    family: str,
    *,
    interpretation: bool,
) -> None:
    directory = root / "experiments" / experiment_id
    _write(
        directory / "summary.json",
        {
            "schema_version": "meta_model_experiment_summary_v1",
            "experiment_id": experiment_id,
            "family": family,
            "train_rows": 900,
            "validation_rows": 100,
        },
    )
    _write(
        directory / "leaderboard.json",
        {
            "experiment_id": experiment_id,
            "methods": [
                {
                    "rank": 1,
                    "method": "compact_ridge",
                    "cv_rmse_log": 0.21,
                    "validation": _metrics(0.25, 0.8, 0.75),
                },
                {
                    "rank": 2,
                    "method": "constant_mean",
                    "cv_rmse_log": 0.5,
                    "validation": _metrics(0.6, 0.0, 0.34),
                },
            ],
        },
    )
    _write(
        directory / "noise_ceiling.json",
        {
            "experiment_id": experiment_id,
            "n_seeds": 10,
            "all": {
                "median_metrics": {
                    "log": {"rmse": 0.1},
                    "ranking": {"spearman": 0.95},
                    "three_choice": {"accuracy": 0.9},
                }
            },
        },
    )
    if interpretation:
        _write(
            directory / "interpretation.json",
            {
                "experiment_id": experiment_id,
                "simple_rules": [
                    {
                        "rule": "Larger models tend to win after controlling for LR."
                    }
                ],
                "findings": ["Optimizer and learning rate interact."],
            },
        )


def _study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    _experiment(
        study,
        "bigram_bg_0021c1_b5120_bs64_cross_entropy",
        "bigram_lm",
        interpretation=True,
    )
    _experiment(
        study,
        "multivariate_mvar_c59a30_b5120_bs32_mse",
        "multivariate_regression",
        interpretation=False,
    )
    _write(
        study / "external" / "score.json",
        {
            "schema_version": "meta_model_scored_predictions_v1",
            "total": {"num_questions": 40, "num_correct": 30, "accuracy": 0.75},
            "by_family": {
                "bigram_lm": {
                    "num_questions": 20,
                    "num_correct": 16,
                    "accuracy": 0.8,
                },
                "multivariate_regression": {
                    "num_questions": 20,
                    "num_correct": 14,
                    "accuracy": 0.7,
                },
            },
            "questions": [],
        },
    )
    return study


def test_generate_report_consolidates_json_and_tolerates_missing_interpretation(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)

    report = generate_report(study)

    assert len(report["experiments"]) == 2
    assert report["experiments"][0]["interpretation"] is not None
    assert report["experiments"][1]["interpretation"] is None
    assert report["highlights"]["external"] == [
        {
            "method": "selected_per_family",
            "num_questions": 40,
            "num_correct": 30,
            "accuracy": 0.75,
        }
    ]
    assert all(
        winner["winner"]["method"] == "compact_ridge"
        for winner in report["highlights"]["experiment_winners"]
    )

    written_json = json.loads((study / "report.json").read_text(encoding="utf-8"))
    assert written_json == report
    markdown = (study / "report.md").read_text(encoding="utf-8")
    assert "# ArchitectureIQ setting-to-loss meta-model study" in markdown
    assert "30 | 40 | 75.0%" in markdown
    assert "compact_ridge" in markdown
    assert "Noise ceiling 3-choice" in markdown
    assert "Larger models tend to win" in markdown
    assert "No optional interpretation artifact was produced." in markdown
    assert not list(study.glob(".report.*.tmp"))


def test_missing_required_noise_ceiling_is_reported_clearly(tmp_path: Path) -> None:
    study = _study(tmp_path)
    missing = (
        study
        / "experiments"
        / "multivariate_mvar_c59a30_b5120_bs32_mse"
        / "noise_ceiling.json"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="noise_ceiling"):
        load_study_artifacts(study)


def test_external_family_must_match_experiment_families(tmp_path: Path) -> None:
    study = _study(tmp_path)
    external_path = study / "external" / "score.json"
    external = json.loads(external_path.read_text(encoding="utf-8"))
    external["by_family"]["unknown_family"] = {
        "num_questions": 1,
        "num_correct": 1,
        "accuracy": 1.0,
    }
    _write(external_path, external)

    with pytest.raises(ValueError, match="unknown families"):
        load_study_artifacts(study)
