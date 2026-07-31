"""Summarize completed outputs from an initial meta-model matrix without training."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "meta_model_initial_matrix_summary_v2"

PRIMARY_METRIC_PATHS = (
    "benchmark_eligible.macro.three_choice_accuracy",
    "benchmark_eligible.micro.three_choice_accuracy",
    "all.macro.three_choice_accuracy",
    "all.micro.three_choice_accuracy",
)

_SCOPE_LAYOUT = {
    "environment": ("id", "environment", "aggregate.json"),
    "family": ("id", "family", "aggregate.json"),
    "dataset": ("ood", "dataset_logo", "aggregate.json"),
    "holdout_candidate": ("ood", "holdout_candidate", "aggregate.json"),
}

_EXPECTED_PROTOCOL = {
    "environment": "per_environment_id",
    "family": "family_pooled_id",
    "dataset": "leave_one_dataset_out",
    "holdout_candidate": "holdout_candidate_dataset_ood",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _command(track: Mapping[str, Any]) -> list[str]:
    command = track.get("command")
    if not isinstance(command, list):
        return []
    return [str(item) for item in command]


def _option(command: Sequence[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _scope(track: Mapping[str, Any]) -> str | None:
    command = _command(track)
    selected = _option(command, "--scopes") or _option(command, "--protocols")
    if selected in _SCOPE_LAYOUT:
        return selected

    name = str(track.get("name", ""))
    for prefix, fallback in (
        ("environment_id__", "environment"),
        ("family_pooled_id__", "family"),
        ("grouped_dataset_logo__", "dataset"),
        ("grouped_holdout_candidate__", "holdout_candidate"),
    ):
        if name.startswith(prefix):
            return fallback
    return None


def _conditioning(track: Mapping[str, Any]) -> str | None:
    command_value = _option(_command(track), "--dataset-conditioning")
    if command_value is not None:
        return command_value
    parts = str(track.get("name", "")).split("__")
    return parts[1] if len(parts) >= 3 and parts[1] else None


def _includes_parameter_count(track: Mapping[str, Any]) -> bool | None:
    command = _command(track)
    if command:
        return "--exclude-parameter-count" not in command
    name = str(track.get("name", ""))
    if name.endswith("__with_params"):
        return True
    if name.endswith("__no_params"):
        return False
    return None


def _dig(value: object, *keys: str) -> object | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _task_count(value: object, fallback: int | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _summarize_view(value: object) -> dict[str, Any]:
    return {
        "n": _task_count(_dig(value, "n"), None),
        "micro": {
            "three_choice_accuracy": _number(
                _dig(value, "within_environment", "three_choice", "accuracy")
            ),
            "gap_ge_0_05_three_choice_accuracy": _number(
                _dig(
                    value,
                    "within_environment",
                    "three_choice",
                    "gap_ge_0_05",
                    "accuracy",
                )
            ),
            "pair_concordance": _number(
                _dig(value, "within_environment", "pair_concordance")
            ),
            "log_rmse": _number(_dig(value, "log", "rmse")),
        },
        "macro": {
            "three_choice_accuracy": _number(
                _dig(
                    value,
                    "within_environment",
                    "macro",
                    "three_choice_accuracy",
                )
            ),
            "gap_ge_0_05_three_choice_accuracy": _number(
                _dig(
                    value,
                    "within_environment",
                    "macro",
                    "gap_ge_0_05_three_choice_accuracy",
                )
            ),
            "log_rmse": _number(_dig(value, "within_environment", "macro", "log_rmse")),
            "spearman": _number(_dig(value, "within_environment", "macro", "spearman")),
        },
    }


def _metric_value(views: Mapping[str, Any], path: str) -> float | None:
    view, aggregation, metric = path.split(".")
    return _number(_dig(views, view, aggregation, metric))


def _primary_metric(views: Mapping[str, Any], path: str | None) -> dict[str, Any]:
    if path is not None:
        view, aggregation, metric = path.split(".")
        preference_rank = PRIMARY_METRIC_PATHS.index(path)
        return {
            "metric": metric,
            "view": view,
            "aggregation": aggregation,
            "path": path,
            "value": _metric_value(views, path),
            "preference_rank": preference_rank,
            "fallback_used": preference_rank > 0,
        }
    return {
        "metric": "three_choice_accuracy",
        "view": None,
        "aggregation": None,
        "path": None,
        "value": None,
        "preference_rank": None,
        "fallback_used": None,
    }


def _summarize_methods(
    aggregate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    aggregate_tasks = _task_count(aggregate.get("n_tasks"), None)
    raw_methods = aggregate.get("methods")
    if not isinstance(raw_methods, list):
        raise ValueError("aggregate.methods must be a list")

    methods: list[dict[str, Any]] = []
    for raw_method in raw_methods:
        if not isinstance(raw_method, Mapping):
            continue
        name = raw_method.get("method")
        if not isinstance(name, str) or not name:
            continue
        views = {
            view: _summarize_view(_dig(raw_method, "test", view))
            for view in ("all", "benchmark_eligible")
        }
        methods.append(
            {
                "method": name,
                "task_count": _task_count(raw_method.get("n_tasks"), aggregate_tasks),
                "views": views,
                "is_best": False,
            }
        )

    # Every method in a track must be compared on the same metric view.
    primary_path = next(
        (
            path
            for path in PRIMARY_METRIC_PATHS
            if any(
                _metric_value(method["views"], path) is not None for method in methods
            )
        ),
        None,
    )
    for method in methods:
        method["primary_metric"] = _primary_metric(method["views"], primary_path)

    eligible = [
        method for method in methods if method["primary_metric"]["value"] is not None
    ]
    if not eligible:
        return methods, None

    def best_key(method: Mapping[str, Any]) -> tuple[float, str]:
        return (
            -float(method["primary_metric"]["value"]),
            str(method["method"]),
        )

    best = min(eligible, key=best_key)
    best["is_best"] = True
    return methods, str(best["method"])


def _summarize_track(matrix_root: Path, track: Mapping[str, Any]) -> dict[str, Any]:
    name_value = track.get("name")
    name = name_value if isinstance(name_value, str) else str(name_value or "")
    scope = _scope(track)
    include_parameter_count = _includes_parameter_count(track)
    result: dict[str, Any] = {
        "name": name,
        "status": track.get("status"),
        "conditioning": _conditioning(track),
        "include_parameter_count": include_parameter_count,
        "parameter_count": (
            "with_params"
            if include_parameter_count is True
            else "no_params"
            if include_parameter_count is False
            else None
        ),
        "scope": scope,
        "protocol": _EXPECTED_PROTOCOL.get(scope),
        "aggregate_path": None,
        "aggregate_status": "unsupported_scope",
        "aggregate_error": None,
        "task_count": None,
        "methods": [],
        "best_method": None,
        "best_method_primary_metric": None,
    }
    if scope is None:
        return result

    aggregate_path = matrix_root / name / Path(*_SCOPE_LAYOUT[scope])
    result["aggregate_path"] = str(aggregate_path)
    if not aggregate_path.is_file():
        result["aggregate_status"] = "missing"
        return result

    try:
        aggregate = _load_object(aggregate_path)
        methods, best_method = _summarize_methods(aggregate)
    except (OSError, ValueError) as error:
        result["aggregate_status"] = "invalid"
        result["aggregate_error"] = f"{type(error).__name__}: {error}"
        return result

    protocol_name = _dig(aggregate, "protocol", "name")
    if isinstance(protocol_name, str) and protocol_name:
        result["protocol"] = protocol_name
    result["aggregate_status"] = "available"
    result["task_count"] = _task_count(aggregate.get("n_tasks"), None)
    result["methods"] = methods
    result["best_method"] = best_method
    if best_method is not None:
        best = next(method for method in methods if method["method"] == best_method)
        result["best_method_primary_metric"] = best["primary_metric"]
    return result


def summarize_matrix(matrix_root: Path) -> dict[str, Any]:
    """Return a read-only summary of every track in an initial matrix."""

    matrix_root = matrix_root.resolve()
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = _load_object(manifest_path)
    raw_tracks = manifest.get("tracks")
    if not isinstance(raw_tracks, list):
        raise ValueError("matrix_manifest.json tracks must be a list")

    tracks = [
        _summarize_track(matrix_root, track)
        for track in raw_tracks
        if isinstance(track, Mapping)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_root": str(matrix_root),
        "matrix_manifest_path": str(manifest_path),
        "matrix_status": manifest.get("status"),
        "primary_metric_policy": {
            "metric": "three_choice_accuracy",
            "preference_order": list(PRIMARY_METRIC_PATHS),
        },
        "track_count": len(tracks),
        "track_status_counts": dict(
            sorted(Counter(str(track["status"]) for track in tracks).items())
        ),
        "aggregate_status_counts": dict(
            sorted(Counter(str(track["aggregate_status"]) for track in tracks).items())
        ),
        "tracks": tracks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_matrix(args.matrix_root)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
