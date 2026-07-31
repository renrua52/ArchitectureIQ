from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from tools.meta_model_study import initial_matrix


def _snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot.json"
    path.write_text('{"completed": true}\n', encoding="utf-8")
    return path


def test_dry_run_writes_complete_named_matrix(tmp_path: Path, monkeypatch) -> None:
    def unexpected_executor(*args, **kwargs):
        raise AssertionError("dry-run must not create a thread pool")

    monkeypatch.setattr(initial_matrix, "ThreadPoolExecutor", unexpected_executor)
    output = tmp_path / "matrix"
    assert initial_matrix.main(
        [
            "--snapshot-manifest",
            str(_snapshot(tmp_path)),
            "--output-root",
            str(output),
            "--jobs",
            "2",
            "--dry-run",
        ]
    ) == 0

    manifest = json.loads((output / "matrix_manifest.json").read_text())
    tracks = manifest["tracks"]
    expected_names = [track.name for track in initial_matrix.build_matrix()]
    assert len(tracks) == 20
    assert len({track["name"] for track in tracks}) == 20
    assert [track["name"] for track in tracks] == expected_names
    names = [track["name"] for track in tracks]
    assert sum(name.startswith("family_pooled_id__") for name in names) == 6
    assert sum(name.startswith("grouped_dataset_logo__") for name in names) == 6
    assert sum(name.startswith("grouped_holdout_candidate__") for name in names) == 6
    assert all(track["status"] == "planned" for track in tracks)
    assert all(
        "constant_mean,compact_ridge,extra_trees" in track["command"]
        for track in tracks
    )
    assert manifest["track_jobs"] == 1


def test_tracks_run_concurrently_and_manifest_order_is_fixed(
    tmp_path: Path, monkeypatch
) -> None:
    expected_names = [track.name for track in initial_matrix.build_matrix()]
    output = tmp_path / "matrix"
    first_tracks_ready = threading.Barrier(3)
    lock = threading.Lock()
    active = 0
    max_active = 0
    completion_order: list[str] = []
    running_names: list[str] = []
    running_statuses: list[str] = []
    writer_threads: list[int] = []
    main_thread = threading.get_ident()
    atomic_write_json = initial_matrix._atomic_write_json

    def track_atomic_write(path: Path, value: object) -> None:
        writer_threads.append(threading.get_ident())
        atomic_write_json(path, value)

    def succeed(command: list[str], *, check: bool) -> CompletedProcess:
        nonlocal active, max_active
        output_index = command.index("--output-root") + 1
        name = Path(command[output_index]).name
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if name == expected_names[0]:
                running_manifest = json.loads(
                    (output / "matrix_manifest.json").read_text()
                )
                with lock:
                    running_names.extend(
                        track["name"] for track in running_manifest["tracks"]
                    )
                    running_statuses.extend(
                        track["status"] for track in running_manifest["tracks"]
                    )
            if name in expected_names[:3]:
                first_tracks_ready.wait(timeout=2)
            if name == expected_names[0]:
                time.sleep(0.03)
            with lock:
                completion_order.append(name)
            return CompletedProcess(command, 0)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(initial_matrix.subprocess, "run", succeed)
    monkeypatch.setattr(initial_matrix, "_atomic_write_json", track_atomic_write)
    assert initial_matrix.run_matrix(
        _snapshot(tmp_path), output, jobs=2, track_jobs=3
    ) == 0

    assert max_active > 1
    assert completion_order[0] != expected_names[0]
    assert running_names == expected_names
    assert running_statuses == ["running"] * 20
    assert set(writer_threads) == {main_thread}
    manifest = json.loads((output / "matrix_manifest.json").read_text())
    assert manifest["track_jobs"] == 3
    assert [track["name"] for track in manifest["tracks"]] == expected_names
    assert all(track["status"] == "success" for track in manifest["tracks"])
    assert all(
        track["command"][track["command"].index("--jobs") + 1] == "2"
        for track in manifest["tracks"]
    )


def test_resume_skips_successful_tracks(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "matrix"
    calls: list[list[str]] = []
    lock = threading.Lock()

    def succeed(command: list[str], *, check: bool) -> CompletedProcess:
        with lock:
            calls.append(command)
        return CompletedProcess(command, 0)

    monkeypatch.setattr(initial_matrix.subprocess, "run", succeed)
    snapshot = _snapshot(tmp_path)
    assert initial_matrix.run_matrix(snapshot, output, jobs=1, track_jobs=4) == 0
    assert len(calls) == 20
    calls.clear()
    assert initial_matrix.run_matrix(snapshot, output, jobs=1, track_jobs=2) == 0
    assert calls == []
    manifest = json.loads((output / "matrix_manifest.json").read_text())
    assert manifest["track_jobs"] == 2
    assert all(track["status"] == "success" for track in manifest["tracks"])
    assert all(track["resume_skipped"] is True for track in manifest["tracks"])


def test_failure_is_recorded_and_later_tracks_continue(
    tmp_path: Path, monkeypatch
) -> None:
    expected_names = [track.name for track in initial_matrix.build_matrix()]
    failed_name = expected_names[1]
    calls: set[str] = set()
    lock = threading.Lock()

    def fail_once(command: list[str], *, check: bool) -> CompletedProcess:
        output_index = command.index("--output-root") + 1
        name = Path(command[output_index]).name
        with lock:
            calls.add(name)
        return CompletedProcess(command, 7 if name == failed_name else 0)

    monkeypatch.setattr(initial_matrix.subprocess, "run", fail_once)
    output = tmp_path / "matrix"
    assert initial_matrix.run_matrix(
        _snapshot(tmp_path), output, jobs=1, track_jobs=4
    ) == 1
    assert calls == set(expected_names)
    manifest = json.loads((output / "matrix_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [track["name"] for track in manifest["tracks"]] == expected_names
    assert manifest["tracks"][1]["status"] == "failed"
    assert manifest["tracks"][1]["returncode"] == 7
    assert manifest["tracks"][-1]["status"] == "success"


def test_track_jobs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--track-jobs must be at least 1"):
        initial_matrix.main(
            [
                "--snapshot-manifest",
                str(_snapshot(tmp_path)),
                "--output-root",
                str(tmp_path / "matrix"),
                "--jobs",
                "1",
                "--track-jobs",
                "0",
            ]
        )
