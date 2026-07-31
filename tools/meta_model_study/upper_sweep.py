"""Run the strong-model upper sweep after a completed initial matrix.

This module only orchestrates existing :mod:`tools.meta_model_study.wide_run`
commands. Importing it never trains a model; training starts only when the CLI
or :func:`run_upper_sweep` is called explicitly without ``dry_run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.meta_model_study.summarize_initial_matrix import summarize_matrix


DEFAULT_METHODS = (
    "compact_polynomial_ridge",
    "random_forest",
    "hist_gradient_boosting",
    "gradient_boosting",
    "rbf_svr",
    "mlp",
)
CONDITIONING_ORDER = ("unaware", "id", "description")
EXPECTED_INITIAL_TRACKS = 20
SELECTION_METHOD = "extra_trees"
SCHEMA_VERSION = "meta_model_upper_sweep_v1"
MANIFEST_NAME = "upper_sweep_manifest.json"


@dataclass(frozen=True)
class SweepProtocol:
    name: str
    scope: str
    command: str
    selector_flag: str
    selector: str


SWEEP_PROTOCOLS = (
    SweepProtocol(
        name="family_pooled_id",
        scope="family",
        command="fit-id",
        selector_flag="--scopes",
        selector="family",
    ),
    SweepProtocol(
        name="holdout_candidate",
        scope="holdout_candidate",
        command="fit-grouped",
        selector_flag="--protocols",
        selector="holdout_candidate",
    ),
    SweepProtocol(
        name="dataset_logo",
        scope="dataset",
        command="fit-grouped",
        selector_flag="--protocols",
        selector="dataset",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _metric_value(method: Mapping[str, Any]) -> tuple[str, float]:
    metric = method.get("primary_metric")
    if not isinstance(metric, Mapping):
        raise ValueError(f"{SELECTION_METHOD} is missing its primary metric")
    path = metric.get("path")
    value = metric.get("value")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{SELECTION_METHOD} primary metric path is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{SELECTION_METHOD} primary metric value is missing")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{SELECTION_METHOD} primary metric value is not finite")
    return path, numeric_value


def _validate_completed_matrix(
    initial_matrix_root: Path,
    *,
    snapshot_sha256: str,
) -> tuple[dict[str, Any], Path, str]:
    summary = summarize_matrix(initial_matrix_root)
    tracks = summary.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("initial matrix summary tracks must be a list")
    if summary.get("matrix_status") != "success":
        raise ValueError("initial matrix status must be success")
    if summary.get("track_count") != EXPECTED_INITIAL_TRACKS or len(tracks) != 20:
        raise ValueError(
            f"initial matrix must contain {EXPECTED_INITIAL_TRACKS} tracks"
        )

    failed_tracks = [
        str(track.get("name"))
        for track in tracks
        if not isinstance(track, Mapping) or track.get("status") != "success"
    ]
    if failed_tracks:
        raise ValueError(
            "initial matrix must be 20/20 success; incomplete tracks: "
            + ", ".join(failed_tracks)
        )
    missing_aggregates = [
        str(track.get("name"))
        for track in tracks
        if not isinstance(track, Mapping)
        or track.get("aggregate_status") != "available"
    ]
    if missing_aggregates:
        raise ValueError(
            "initial matrix must have 20/20 aggregates; unavailable tracks: "
            + ", ".join(missing_aggregates)
        )

    manifest_path = initial_matrix_root / "matrix_manifest.json"
    initial_manifest = _load_object(manifest_path)
    initial_snapshot_sha256 = initial_manifest.get("snapshot_manifest_sha256")
    if initial_snapshot_sha256 != snapshot_sha256:
        raise ValueError(
            "initial matrix snapshot SHA does not match --snapshot-manifest"
        )
    return summary, manifest_path, _sha256_file(manifest_path)


def _select_conditioning(
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    raw_tracks = summary.get("tracks")
    if not isinstance(raw_tracks, list):
        raise ValueError("initial matrix summary tracks must be a list")

    selected_tracks: dict[str, list[dict[str, Any]]] = {
        protocol.scope: [] for protocol in SWEEP_PROTOCOLS
    }
    metric_paths: set[str] = set()
    for raw_track in raw_tracks:
        if not isinstance(raw_track, Mapping):
            continue
        scope = raw_track.get("scope")
        if (
            scope not in selected_tracks
            or raw_track.get("include_parameter_count") is not True
        ):
            continue
        conditioning = raw_track.get("conditioning")
        if conditioning not in CONDITIONING_ORDER:
            raise ValueError(f"unsupported conditioning in {raw_track.get('name')}")
        methods = raw_track.get("methods")
        if not isinstance(methods, list):
            raise ValueError(f"methods are missing for {raw_track.get('name')}")
        matches = [
            method
            for method in methods
            if isinstance(method, Mapping) and method.get("method") == SELECTION_METHOD
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{raw_track.get('name')} must contain exactly one {SELECTION_METHOD}"
            )
        metric_path, metric_value = _metric_value(matches[0])
        metric_paths.add(metric_path)
        selected_tracks[str(scope)].append(
            {
                "track_name": raw_track.get("name"),
                "conditioning": conditioning,
                "aggregate_path": raw_track.get("aggregate_path"),
                "metric_path": metric_path,
                "metric_value": metric_value,
            }
        )

    selections: list[dict[str, Any]] = []
    for protocol in SWEEP_PROTOCOLS:
        candidates = selected_tracks[protocol.scope]
        conditionings = [candidate["conditioning"] for candidate in candidates]
        if len(candidates) != len(CONDITIONING_ORDER) or set(conditionings) != set(
            CONDITIONING_ORDER
        ):
            raise ValueError(
                f"{protocol.name} requires one with_params track for each conditioning"
            )
        candidates.sort(
            key=lambda candidate: CONDITIONING_ORDER.index(candidate["conditioning"])
        )
        best = min(
            candidates,
            key=lambda candidate: (
                -float(candidate["metric_value"]),
                CONDITIONING_ORDER.index(candidate["conditioning"]),
            ),
        )
        selections.append(
            {
                "protocol": protocol.name,
                "scope": protocol.scope,
                "selected_conditioning": best["conditioning"],
                "selected_track": best["track_name"],
                "metric_path": best["metric_path"],
                "metric_value": best["metric_value"],
                "candidates": candidates,
            }
        )

    if len(metric_paths) != 1:
        raise ValueError(
            f"{SELECTION_METHOD} tracks do not share one primary metric: "
            + ", ".join(sorted(metric_paths))
        )
    return selections, next(iter(metric_paths))


def _track_command(
    protocol: SweepProtocol,
    selection: Mapping[str, Any],
    *,
    snapshot_manifest: Path,
    output_root: Path,
    jobs: int,
    methods: tuple[str, ...],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.meta_model_study.wide_run",
        protocol.command,
        "--snapshot-manifest",
        str(snapshot_manifest),
        "--output-root",
        str(output_root / protocol.name),
        "--jobs",
        str(jobs),
        "--methods",
        ",".join(methods),
        "--dataset-conditioning",
        str(selection["selected_conditioning"]),
        protocol.selector_flag,
        protocol.selector,
    ]


def _run_command(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def run_upper_sweep(
    snapshot_manifest: Path,
    initial_matrix_root: Path,
    output_root: Path,
    *,
    jobs: int,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    dry_run: bool = False,
) -> int:
    """Validate, plan, and sequentially run the three upper-sweep tracks."""

    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if not methods or any(
        not isinstance(method, str) or not method for method in methods
    ):
        raise ValueError("methods must not be empty")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")

    snapshot_manifest = snapshot_manifest.resolve()
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            f"snapshot manifest does not exist: {snapshot_manifest}"
        )
    initial_matrix_root = initial_matrix_root.resolve()
    if not initial_matrix_root.is_dir():
        raise FileNotFoundError(
            f"initial matrix root does not exist: {initial_matrix_root}"
        )
    snapshot_sha256 = _sha256_file(snapshot_manifest)
    summary, initial_manifest_path, initial_manifest_sha256 = (
        _validate_completed_matrix(
            initial_matrix_root,
            snapshot_sha256=snapshot_sha256,
        )
    )
    selections, metric_path = _select_conditioning(summary)

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    previous = _load_object(manifest_path) if manifest_path.is_file() else {}
    previous_tracks = {
        track.get("name"): track
        for track in previous.get("tracks", [])
        if isinstance(track, Mapping) and track.get("name")
    }
    previous_compatible = (
        previous.get("schema_version") == SCHEMA_VERSION
        and previous.get("snapshot_manifest_sha256") == snapshot_sha256
        and previous.get("initial_matrix_manifest_sha256") == initial_manifest_sha256
    )

    records: list[dict[str, Any]] = []
    for index, (protocol, selection) in enumerate(
        zip(SWEEP_PROTOCOLS, selections, strict=True), start=1
    ):
        command = _track_command(
            protocol,
            selection,
            snapshot_manifest=snapshot_manifest,
            output_root=output_root,
            jobs=jobs,
            methods=methods,
        )
        base: dict[str, Any] = {
            "name": protocol.name,
            "order": index,
            "scope": protocol.scope,
            "output_root": str(output_root / protocol.name),
            "selection": selection,
            "command": command,
            "status": "planned" if dry_run else "pending",
            "started_at": None,
            "completed_at": None,
            "returncode": None,
        }
        old = previous_tracks.get(protocol.name)
        resumable = (
            previous_compatible
            and isinstance(old, Mapping)
            and old.get("status") == "success"
            and old.get("command") == command
            and old.get("selection") == selection
        )
        if resumable:
            base = dict(old)
            if dry_run:
                base["would_resume_skip"] = True
            else:
                base["resume_skipped"] = True
                base["resume_skipped_at"] = _utc_now()
        records.append(base)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if dry_run else "running",
        "dry_run": dry_run,
        "snapshot_manifest_path": str(snapshot_manifest),
        "snapshot_manifest_sha256": snapshot_sha256,
        "initial_matrix_root": str(initial_matrix_root),
        "initial_matrix_manifest_path": str(initial_manifest_path),
        "initial_matrix_manifest_sha256": initial_manifest_sha256,
        "initial_matrix_validation": {
            "status": summary["matrix_status"],
            "track_count": summary["track_count"],
            "success_count": sum(
                track.get("status") == "success" for track in summary["tracks"]
            ),
            "aggregate_count": sum(
                track.get("aggregate_status") == "available"
                for track in summary["tracks"]
            ),
        },
        "selection_basis": {
            "method": SELECTION_METHOD,
            "include_parameter_count": True,
            "primary_metric_path": metric_path,
            "primary_metric_policy": summary.get("primary_metric_policy"),
            "conditioning_tie_break_order": list(CONDITIONING_ORDER),
            "protocol_order": [protocol.name for protocol in SWEEP_PROTOCOLS],
        },
        "jobs": jobs,
        "methods": list(methods),
        "output_root": str(output_root),
        "started_at": _utc_now(),
        "completed_at": None,
        "tracks": records,
    }

    if dry_run:
        manifest["completed_at"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        return 0

    _atomic_write_json(manifest_path, manifest)
    for record in records:
        if record["status"] == "success":
            continue
        record["status"] = "running"
        record["started_at"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        try:
            returncode = _run_command(record["command"])
        except Exception as error:
            returncode = None
            record["error"] = f"{type(error).__name__}: {error}"
        record["returncode"] = returncode
        record["completed_at"] = _utc_now()
        record["status"] = "success" if returncode == 0 else "failed"
        _atomic_write_json(manifest_path, manifest)
        if record["status"] == "failed":
            manifest["status"] = "failed"
            manifest["completed_at"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)
            return 1

    manifest["status"] = "success"
    manifest["completed_at"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    return 0


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods:
        raise argparse.ArgumentTypeError("--methods must not be empty")
    if len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("--methods must not contain duplicates")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--initial-matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--methods", type=_parse_methods, default=DEFAULT_METHODS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_upper_sweep(
        args.snapshot_manifest,
        args.initial_matrix_root,
        args.output_root,
        jobs=args.jobs,
        methods=args.methods,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
