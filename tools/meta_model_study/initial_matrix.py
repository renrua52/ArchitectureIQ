"""Run the reproducible initial meta-model matrix from a frozen snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_METHODS = ("constant_mean", "compact_ridge", "extra_trees")
CONDITIONING = ("unaware", "id", "description")
SCHEMA_VERSION = "meta_model_initial_matrix_v1"


@dataclass(frozen=True)
class Track:
    name: str
    command: str
    selector_flag: str
    selector: str
    conditioning: str
    include_parameter_count: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_matrix() -> tuple[Track, ...]:
    """Return tracks in their required execution order."""

    tracks: list[Track] = []

    def add(
        prefix: str,
        command: str,
        selector_flag: str,
        selector: str,
        conditioning: str,
        include_params: bool,
    ) -> None:
        parameter_label = "with_params" if include_params else "no_params"
        tracks.append(
            Track(
                name=f"{prefix}__{conditioning}__{parameter_label}",
                command=command,
                selector_flag=selector_flag,
                selector=selector,
                conditioning=conditioning,
                include_parameter_count=include_params,
            )
        )

    for include_params in (True, False):
        add(
            "environment_id",
            "fit-id",
            "--scopes",
            "environment",
            "unaware",
            include_params,
        )
    for conditioning in CONDITIONING:
        for include_params in (True, False):
            add(
                "family_pooled_id",
                "fit-id",
                "--scopes",
                "family",
                conditioning,
                include_params,
            )
    for prefix, protocol in (
        ("grouped_dataset_logo", "dataset"),
        ("grouped_holdout_candidate", "holdout_candidate"),
    ):
        for conditioning in CONDITIONING:
            for include_params in (True, False):
                add(
                    prefix,
                    "fit-grouped",
                    "--protocols",
                    protocol,
                    conditioning,
                    include_params,
                )
    return tuple(tracks)


def _track_command(
    track: Track,
    *,
    snapshot_manifest: Path,
    output_root: Path,
    jobs: int,
    methods: tuple[str, ...],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.meta_model_study.wide_run",
        track.command,
        "--snapshot-manifest",
        str(snapshot_manifest),
        "--output-root",
        str(output_root / track.name),
        "--jobs",
        str(jobs),
        "--methods",
        ",".join(methods),
        "--dataset-conditioning",
        track.conditioning,
        track.selector_flag,
        track.selector,
    ]
    if not track.include_parameter_count:
        command.append("--exclude-parameter-count")
    return command


def _run_track(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def run_matrix(
    snapshot_manifest: Path,
    output_root: Path,
    *,
    jobs: int,
    track_jobs: int = 1,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    dry_run: bool = False,
) -> int:
    if track_jobs < 1:
        raise ValueError("track_jobs must be at least 1")
    snapshot_manifest = snapshot_manifest.resolve()
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            f"snapshot manifest does not exist: {snapshot_manifest}"
        )
    if not methods:
        raise ValueError("methods must not be empty")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "matrix_manifest.json"
    snapshot_sha256 = _sha256_file(snapshot_manifest)
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_tracks = {
        item.get("name"): item
        for item in previous.get("tracks", [])
        if item.get("name")
    }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_manifest_path": str(snapshot_manifest),
        "snapshot_manifest_sha256": snapshot_sha256,
        "output_root": str(output_root),
        "jobs": jobs,
        "track_jobs": track_jobs,
        "methods": list(methods),
        "dry_run": dry_run,
        "started_at": _utc_now(),
        "completed_at": None,
        "tracks": [],
    }
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, list[str]]] = []
    for index, track in enumerate(build_matrix()):
        command = _track_command(
            track,
            snapshot_manifest=snapshot_manifest,
            output_root=output_root,
            jobs=jobs,
            methods=methods,
        )
        old = previous_tracks.get(track.name, {})
        resumable = (
            not dry_run
            and previous.get("snapshot_manifest_sha256") == snapshot_sha256
            and old.get("status") == "success"
            and old.get("command") == command
        )
        record: dict[str, Any] = {
            "name": track.name,
            "output_root": str(output_root / track.name),
            "command": command,
            "status": "planned" if dry_run else "running",
            "started_at": None if dry_run else _utc_now(),
            "completed_at": None,
            "returncode": None,
        }
        if resumable:
            record = dict(old)
            record["resume_skipped"] = True
            record["resume_skipped_at"] = _utc_now()
        elif not dry_run:
            pending.append((index, command))
        records.append(record)
    manifest["tracks"] = records

    if dry_run:
        manifest["completed_at"] = _utc_now()
        manifest["status"] = "planned"
        _atomic_write_json(manifest_path, manifest)
        return 0

    _atomic_write_json(manifest_path, manifest)
    if pending:
        with ThreadPoolExecutor(max_workers=track_jobs) as executor:
            futures = {
                executor.submit(_run_track, command): index
                for index, command in pending
            }
            for future in as_completed(futures):
                record = records[futures[future]]
                try:
                    returncode = future.result()
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                else:
                    record["returncode"] = returncode
                    record["status"] = "success" if returncode == 0 else "failed"
                record["completed_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)

    failed = any(record["status"] == "failed" for record in records)
    manifest["completed_at"] = _utc_now()
    manifest["status"] = "failed" if failed else "success"
    _atomic_write_json(manifest_path, manifest)
    return 1 if failed else 0


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods:
        raise argparse.ArgumentTypeError("--methods must not be empty")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--track-jobs", type=int, default=1)
    parser.add_argument("--methods", type=_parse_methods, default=DEFAULT_METHODS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.jobs <= 4:
        raise ValueError("--jobs must be between 1 and 4")
    if args.track_jobs < 1:
        raise ValueError("--track-jobs must be at least 1")
    return run_matrix(
        args.snapshot_manifest,
        args.output_root,
        jobs=args.jobs,
        track_jobs=args.track_jobs,
        methods=args.methods,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
