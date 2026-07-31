from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.meta_model_study import summarize_initial_matrix


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _command(
    track_root: Path,
    *,
    command: str,
    selector_flag: str,
    selector: str,
    conditioning: str,
    include_params: bool,
) -> list[str]:
    value = [
        "python",
        "-m",
        "tools.meta_model_study.wide_run",
        command,
        "--output-root",
        str(track_root),
        "--dataset-conditioning",
        conditioning,
        selector_flag,
        selector,
    ]
    if not include_params:
        value.append("--exclude-parameter-count")
    return value


def _method(
    name: str,
    *,
    n_tasks: int,
    all_view: dict | None = None,
    benchmark_eligible_view: dict | None = None,
) -> dict:
    return {
        "method": name,
        "n_tasks": n_tasks,
        "test": {
            "all": all_view,
            "benchmark_eligible": benchmark_eligible_view,
        },
    }


def _view(
    *,
    n: int = 100,
    micro_three: float | None = None,
    micro_gap: float | None = None,
    micro_pair: float | None = None,
    micro_log_rmse: float | None = None,
    macro_three: float | None = None,
    macro_gap: float | None = None,
    macro_log_rmse: float | None = None,
    macro_spearman: float | None = None,
) -> dict:
    return {
        "n": n,
        "log": {"rmse": micro_log_rmse},
        "within_environment": {
            "pair_concordance": micro_pair,
            "three_choice": {
                "accuracy": micro_three,
                "gap_ge_0_05": {"accuracy": micro_gap},
            },
            "macro": {
                "three_choice_accuracy": macro_three,
                "gap_ge_0_05_three_choice_accuracy": macro_gap,
                "log_rmse": macro_log_rmse,
                "spearman": macro_spearman,
            },
        },
    }


def test_summarizes_available_and_running_tracks(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix"
    completed_name = "family_pooled_id__description__with_params"
    running_name = "grouped_dataset_logo__id__no_params"
    manifest = {
        "status": "running",
        "tracks": [
            {
                "name": completed_name,
                "status": "success",
                "output_root": f"/remote/matrix/{completed_name}",
                "command": _command(
                    Path("/remote") / completed_name,
                    command="fit-id",
                    selector_flag="--scopes",
                    selector="family",
                    conditioning="description",
                    include_params=True,
                ),
            },
            {
                "name": running_name,
                "status": "running",
                "output_root": f"/remote/matrix/{running_name}",
                "command": _command(
                    Path("/remote") / running_name,
                    command="fit-grouped",
                    selector_flag="--protocols",
                    selector="dataset",
                    conditioning="id",
                    include_params=False,
                ),
            },
        ],
    }
    _write_json(matrix_root / "matrix_manifest.json", manifest)
    _write_json(
        matrix_root / completed_name / "id" / "family" / "aggregate.json",
        {
            "protocol": {"name": "family_pooled_id"},
            "n_tasks": 3,
            "methods": [
                _method(
                    "compact_ridge",
                    n_tasks=3,
                    all_view=_view(micro_three=0.80, macro_three=0.72),
                    benchmark_eligible_view=_view(macro_three=0.6753),
                ),
                _method(
                    "extra_trees",
                    n_tasks=3,
                    all_view=_view(
                        n=1000,
                        micro_three=0.7331,
                        micro_gap=0.78,
                        micro_pair=0.80,
                        micro_log_rmse=1.4,
                        macro_three=0.71,
                        macro_gap=0.75,
                        macro_log_rmse=1.1,
                        macro_spearman=0.70,
                    ),
                    benchmark_eligible_view=_view(
                        n=900,
                        micro_three=0.70,
                        micro_gap=0.76,
                        micro_pair=0.79,
                        micro_log_rmse=1.2,
                        macro_three=0.6808,
                        macro_gap=0.74,
                        macro_log_rmse=0.9,
                        macro_spearman=0.69,
                    ),
                ),
            ],
        },
    )

    summary = summarize_initial_matrix.summarize_matrix(matrix_root)

    assert summary["matrix_status"] == "running"
    assert summary["track_status_counts"] == {"running": 1, "success": 1}
    assert summary["aggregate_status_counts"] == {"available": 1, "missing": 1}
    completed, running = summary["tracks"]
    assert completed["conditioning"] == "description"
    assert completed["parameter_count"] == "with_params"
    assert completed["include_parameter_count"] is True
    assert completed["scope"] == "family"
    assert completed["protocol"] == "family_pooled_id"
    assert completed["task_count"] == 3
    assert completed["best_method"] == "extra_trees"
    extra_trees = completed["methods"][1]
    assert extra_trees["views"]["all"] == {
        "n": 1000,
        "micro": {
            "three_choice_accuracy": pytest.approx(0.7331),
            "gap_ge_0_05_three_choice_accuracy": pytest.approx(0.78),
            "pair_concordance": pytest.approx(0.80),
            "log_rmse": pytest.approx(1.4),
        },
        "macro": {
            "three_choice_accuracy": pytest.approx(0.71),
            "gap_ge_0_05_three_choice_accuracy": pytest.approx(0.75),
            "log_rmse": pytest.approx(1.1),
            "spearman": pytest.approx(0.70),
        },
    }
    assert extra_trees["views"]["benchmark_eligible"]["macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.6808)
    assert extra_trees["primary_metric"] == {
        "metric": "three_choice_accuracy",
        "view": "benchmark_eligible",
        "aggregation": "macro",
        "path": "benchmark_eligible.macro.three_choice_accuracy",
        "value": pytest.approx(0.6808),
        "preference_rank": 0,
        "fallback_used": False,
    }
    assert extra_trees["is_best"] is True
    assert "overall_three_choice_accuracy" not in extra_trees
    assert completed["methods"][0]["is_best"] is False
    assert completed["best_method_primary_metric"] == extra_trees["primary_metric"]
    assert running["status"] == "running"
    assert running["conditioning"] == "id"
    assert running["parameter_count"] == "no_params"
    assert running["scope"] == "dataset"
    assert running["protocol"] == "leave_one_dataset_out"
    assert running["aggregate_status"] == "missing"
    assert running["methods"] == []
    assert running["best_method"] is None
    assert running["best_method_primary_metric"] is None


def test_primary_metric_fallback_order_and_null_safe_views(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix"
    name = "environment_id__unaware__with_params"
    _write_json(
        matrix_root / "matrix_manifest.json",
        {
            "status": "success",
            "tracks": [
                {
                    "name": name,
                    "status": "success",
                    "command": _command(
                        matrix_root / name,
                        command="fit-id",
                        selector_flag="--scopes",
                        selector="environment",
                        conditioning="unaware",
                        include_params=True,
                    ),
                }
            ],
        },
    )
    _write_json(
        matrix_root / name / "id" / "environment" / "aggregate.json",
        {
            "n_tasks": 2,
            "methods": [
                _method(
                    "benchmark_micro",
                    n_tasks=2,
                    all_view=_view(micro_three=0.99, macro_three=0.98),
                    benchmark_eligible_view=_view(micro_three=0.64),
                ),
                _method(
                    "all_macro",
                    n_tasks=2,
                    all_view=_view(micro_three=0.90, macro_three=0.65),
                ),
                _method(
                    "all_micro",
                    n_tasks=2,
                    all_view=_view(micro_three=0.66),
                ),
                _method("no_metric", n_tasks=2),
            ],
        },
    )

    track = summarize_initial_matrix.summarize_matrix(matrix_root)["tracks"][0]
    by_name = {method["method"]: method for method in track["methods"]}

    assert by_name["benchmark_micro"]["primary_metric"]["path"] == (
        "benchmark_eligible.micro.three_choice_accuracy"
    )
    assert by_name["benchmark_micro"]["primary_metric"]["value"] == pytest.approx(0.64)
    assert by_name["all_macro"]["primary_metric"]["path"] == (
        "benchmark_eligible.micro.three_choice_accuracy"
    )
    assert by_name["all_macro"]["primary_metric"]["value"] is None
    assert by_name["all_micro"]["primary_metric"]["path"] == (
        "benchmark_eligible.micro.three_choice_accuracy"
    )
    assert by_name["all_micro"]["primary_metric"]["value"] is None
    assert by_name["no_metric"]["primary_metric"]["value"] is None
    assert by_name["no_metric"]["views"]["benchmark_eligible"] == {
        "n": None,
        "micro": {
            "three_choice_accuracy": None,
            "gap_ge_0_05_three_choice_accuracy": None,
            "pair_concordance": None,
            "log_rmse": None,
        },
        "macro": {
            "three_choice_accuracy": None,
            "gap_ge_0_05_three_choice_accuracy": None,
            "log_rmse": None,
            "spearman": None,
        },
    }
    assert track["best_method"] == "benchmark_micro"


def test_invalid_aggregate_is_reported_without_failing(tmp_path: Path) -> None:
    matrix_root = tmp_path / "matrix"
    name = "grouped_holdout_candidate__unaware__no_params"
    _write_json(
        matrix_root / "matrix_manifest.json",
        {
            "status": "running",
            "tracks": [
                {
                    "name": name,
                    "status": "running",
                    "command": _command(
                        matrix_root / name,
                        command="fit-grouped",
                        selector_flag="--protocols",
                        selector="holdout_candidate",
                        conditioning="unaware",
                        include_params=False,
                    ),
                }
            ],
        },
    )
    aggregate = matrix_root / name / "ood" / "holdout_candidate" / "aggregate.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text("{in progress", encoding="utf-8")

    track = summarize_initial_matrix.summarize_matrix(matrix_root)["tracks"][0]

    assert track["aggregate_status"] == "invalid"
    assert track["protocol"] == "holdout_candidate_dataset_ood"
    assert track["methods"] == []
    assert track["best_method"] is None
    assert track["aggregate_error"].startswith("JSONDecodeError:")


def test_cli_prints_json_or_writes_requested_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matrix_root = tmp_path / "matrix"
    _write_json(
        matrix_root / "matrix_manifest.json",
        {"status": "success", "tracks": []},
    )

    assert summarize_initial_matrix.main(["--matrix-root", str(matrix_root)]) == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["track_count"] == 0

    output = tmp_path / "reports" / "summary.json"
    assert (
        summarize_initial_matrix.main(
            ["--matrix-root", str(matrix_root), "--output", str(output)]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8")) == stdout_summary
