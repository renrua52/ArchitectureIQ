from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from tools.meta_model_study import initial_matrix, upper_sweep


_LAYOUT = {
    "environment": ("id", "environment", "aggregate.json"),
    "family": ("id", "family", "aggregate.json"),
    "dataset": ("ood", "dataset_logo", "aggregate.json"),
    "holdout_candidate": ("ood", "holdout_candidate", "aggregate.json"),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot.json"
    path.write_text('{"completed": true}\n', encoding="utf-8")
    return path


def _view(value: float) -> dict:
    return {
        "n": 100,
        "log": {"rmse": 1.0},
        "within_environment": {
            "pair_concordance": 0.7,
            "three_choice": {"accuracy": value},
            "macro": {"three_choice_accuracy": value},
        },
    }


def _track_command(track: initial_matrix.Track, output_root: Path) -> list[str]:
    command = [
        "python",
        "-m",
        "tools.meta_model_study.wide_run",
        track.command,
        "--output-root",
        str(output_root / track.name),
        "--dataset-conditioning",
        track.conditioning,
        track.selector_flag,
        track.selector,
    ]
    if not track.include_parameter_count:
        command.append("--exclude-parameter-count")
    return command


def _metric_for(track: initial_matrix.Track) -> float:
    values = {
        "family": {"unaware": 0.61, "id": 0.65, "description": 0.63},
        "holdout_candidate": {
            "unaware": 0.60,
            "id": 0.59,
            "description": 0.62,
        },
        "dataset": {"unaware": 0.45, "id": 0.48, "description": 0.47},
    }
    if track.include_parameter_count and track.selector in values:
        return values[track.selector][track.conditioning]
    return 0.4


def _completed_initial_matrix(
    tmp_path: Path, snapshot: Path
) -> tuple[Path, list[initial_matrix.Track]]:
    root = tmp_path / "initial"
    tracks = list(initial_matrix.build_matrix())
    manifest_tracks = []
    for track in tracks:
        manifest_tracks.append(
            {
                "name": track.name,
                "status": "success",
                "command": _track_command(track, root),
            }
        )
        aggregate = root / track.name / Path(*_LAYOUT[track.selector])
        metric = _metric_for(track)
        _write_json(
            aggregate,
            {
                "n_tasks": 3,
                "methods": [
                    {
                        "method": "extra_trees",
                        "n_tasks": 3,
                        "test": {
                            "all": _view(metric + 0.01),
                            "benchmark_eligible": _view(metric),
                        },
                    }
                ],
            },
        )
    _write_json(
        root / "matrix_manifest.json",
        {
            "status": "success",
            "snapshot_manifest_sha256": _sha256(snapshot),
            "tracks": manifest_tracks,
        },
    )
    return root, tracks


def _manifest(output: Path) -> dict:
    return json.loads((output / upper_sweep.MANIFEST_NAME).read_text(encoding="utf-8"))


def test_dry_run_selects_best_conditioning_and_builds_strong_model_plan(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    output = tmp_path / "upper"

    def unexpected_run(command: list[str]) -> int:
        raise AssertionError(f"dry-run attempted training: {command}")

    monkeypatch.setattr(upper_sweep, "_run_command", unexpected_run)
    assert (
        upper_sweep.run_upper_sweep(
            snapshot, initial_root, output, jobs=32, dry_run=True
        )
        == 0
    )

    manifest = _manifest(output)
    assert manifest["status"] == "planned"
    assert manifest["snapshot_manifest_sha256"] == _sha256(snapshot)
    assert manifest["initial_matrix_validation"] == {
        "status": "success",
        "track_count": 20,
        "success_count": 20,
        "aggregate_count": 20,
    }
    assert manifest["selection_basis"]["method"] == "extra_trees"
    assert manifest["selection_basis"]["include_parameter_count"] is True
    assert manifest["selection_basis"]["primary_metric_path"] == (
        "benchmark_eligible.macro.three_choice_accuracy"
    )
    tracks = manifest["tracks"]
    assert [track["name"] for track in tracks] == [
        "family_pooled_id",
        "holdout_candidate",
        "dataset_logo",
    ]
    assert [track["selection"]["selected_conditioning"] for track in tracks] == [
        "id",
        "description",
        "id",
    ]
    assert all(track["status"] == "planned" for track in tracks)
    assert all(track["command"][0] == sys.executable for track in tracks)
    assert all(
        track["command"][track["command"].index("--jobs") + 1] == "32"
        for track in tracks
    )
    assert all(
        track["command"][track["command"].index("--methods") + 1]
        == ",".join(upper_sweep.DEFAULT_METHODS)
        for track in tracks
    )
    assert "extra_trees" not in manifest["methods"]
    assert "xgboost" not in manifest["methods"]


def test_tracks_execute_strictly_serial_in_required_order(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    output = tmp_path / "upper"
    calls: list[str] = []

    def succeed(command: list[str]) -> int:
        name = Path(command[command.index("--output-root") + 1]).name
        running = _manifest(output)["tracks"]
        index = len(calls)
        assert [track["status"] for track in running[:index]] == ["success"] * index
        assert running[index]["status"] == "running"
        assert [track["status"] for track in running[index + 1 :]] == ["pending"] * (
            2 - index
        )
        calls.append(name)
        return 0

    monkeypatch.setattr(upper_sweep, "_run_command", succeed)
    assert upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4) == 0
    assert calls == ["family_pooled_id", "holdout_candidate", "dataset_logo"]
    manifest = _manifest(output)
    assert manifest["status"] == "success"
    assert [track["status"] for track in manifest["tracks"]] == [
        "success",
        "success",
        "success",
    ]


def test_resume_skips_successful_tracks(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    output = tmp_path / "upper"
    calls: list[list[str]] = []

    def succeed(command: list[str]) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(upper_sweep, "_run_command", succeed)
    assert upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4) == 0
    assert len(calls) == 3
    calls.clear()

    assert upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4) == 0
    assert calls == []
    manifest = _manifest(output)
    assert all(track["resume_skipped"] is True for track in manifest["tracks"])


def test_failure_stops_later_protocol_and_resume_continues(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    output = tmp_path / "upper"
    calls: list[str] = []

    def fail_holdout(command: list[str]) -> int:
        name = Path(command[command.index("--output-root") + 1]).name
        calls.append(name)
        return 7 if name == "holdout_candidate" else 0

    monkeypatch.setattr(upper_sweep, "_run_command", fail_holdout)
    assert upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4) == 1
    assert calls == ["family_pooled_id", "holdout_candidate"]
    manifest = _manifest(output)
    assert manifest["status"] == "failed"
    assert [track["status"] for track in manifest["tracks"]] == [
        "success",
        "failed",
        "pending",
    ]

    calls.clear()
    monkeypatch.setattr(
        upper_sweep,
        "_run_command",
        lambda command: (
            calls.append(Path(command[command.index("--output-root") + 1]).name) or 0
        ),
    )
    assert upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4) == 0
    assert calls == ["holdout_candidate", "dataset_logo"]


@pytest.mark.parametrize("fault", ["incomplete", "missing_aggregate"])
def test_incomplete_initial_matrix_is_rejected_before_any_output(
    tmp_path: Path, monkeypatch, fault: str
) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, tracks = _completed_initial_matrix(tmp_path, snapshot)
    if fault == "incomplete":
        manifest_path = initial_root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest["tracks"][-1]["status"] = "running"
        _write_json(manifest_path, manifest)
    else:
        last = tracks[-1]
        (initial_root / last.name / Path(*_LAYOUT[last.selector])).unlink()

    def unexpected_run(command: list[str]) -> int:
        raise AssertionError(f"invalid matrix attempted training: {command}")

    monkeypatch.setattr(upper_sweep, "_run_command", unexpected_run)
    output = tmp_path / "upper"
    with pytest.raises(ValueError, match="initial matrix"):
        upper_sweep.run_upper_sweep(snapshot, initial_root, output, jobs=4)
    assert not output.exists()


def test_primary_metric_must_be_uniform(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    name = "grouped_dataset_logo__description__with_params"
    aggregate_path = initial_root / name / "ood" / "dataset_logo" / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    method = aggregate["methods"][0]
    method["test"]["benchmark_eligible"]["within_environment"]["macro"][
        "three_choice_accuracy"
    ] = None
    _write_json(aggregate_path, aggregate)

    with pytest.raises(ValueError, match="do not share one primary metric"):
        upper_sweep.run_upper_sweep(
            snapshot,
            initial_root,
            tmp_path / "upper",
            jobs=4,
            dry_run=True,
        )


def test_cli_accepts_method_override_without_training(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    initial_root, _ = _completed_initial_matrix(tmp_path, snapshot)
    output = tmp_path / "upper"
    monkeypatch.setattr(
        upper_sweep.subprocess,
        "run",
        lambda command, check: CompletedProcess(command, 0),
    )

    assert (
        upper_sweep.main(
            [
                "--snapshot-manifest",
                str(snapshot),
                "--initial-matrix-root",
                str(initial_root),
                "--output-root",
                str(output),
                "--jobs",
                "3",
                "--methods",
                "random_forest,mlp",
                "--dry-run",
            ]
        )
        == 0
    )
    assert _manifest(output)["methods"] == ["random_forest", "mlp"]
