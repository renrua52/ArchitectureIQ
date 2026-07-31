"""Consolidate meta-model study JSON artifacts into human-readable reports.

The reporter is intentionally independent from fitting code.  A study output
directory has the following minimal contract::

    study_dir/
      experiments/
        <experiment_id>/
          summary.json          # object; requires experiment_id and family
          leaderboard.json      # {"methods": [{"method": ..., ...}, ...]}
          noise_ceiling.json     # split_half_noise_ceiling result object
          interpretation.json   # optional object
      external/
        score.json               # score_predictions result object

``external_score.json`` at the study root is also accepted.  For compatibility
with simple runners, experiment directories may live directly below
``study_dir``.  ``leaderboard.json`` may use ``rows`` or ``leaderboard`` in
place of ``methods``, or be a bare list.  Each leaderboard row minimally needs
``method`` (``name`` is accepted); the recommended validation metric tree is
the output of ``metrics.evaluate_predictions`` under ``validation``.

Only ``interpretation.json`` is optional.  Unknown fields are retained in
``report.json`` so the consolidated artifact is lossless, while ``report.md``
selects a compact set of stable headline metrics.  This module reads JSON
artifacts only and never imports fitted estimators or candidate code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA_VERSION = "meta_model_study_report_v1"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, root: Path, kind: str) -> dict[str, str]:
    try:
        portable_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        portable_path = str(path.resolve())
    return {
        "kind": kind,
        "path": portable_path,
        "sha256": _sha256_file(path),
    }


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _leaderboard_rows(value: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("methods", "rows", "leaderboard"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(
                f"leaderboard.json must contain methods, rows, or leaderboard: {path}"
            )
    if not isinstance(value, list) or not value:
        raise ValueError(f"leaderboard must be a non-empty list: {path}")
    rows: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    for index, raw_row in enumerate(value):
        if not isinstance(raw_row, dict):
            raise TypeError(f"leaderboard row {index} must be an object: {path}")
        row = deepcopy(raw_row)
        method = row.get("method", row.get("name"))
        method = _nonempty_string(method, label=f"leaderboard row {index} method")
        if method in seen_methods:
            raise ValueError(f"Duplicate leaderboard method {method!r}: {path}")
        seen_methods.add(method)
        row.setdefault("method", method)
        row.setdefault("rank", index + 1)
        rows.append(row)
    return rows


def _experiment_dirs(study_dir: Path) -> list[Path]:
    experiments_root = study_dir / "experiments"
    search_root = experiments_root if experiments_root.is_dir() else study_dir
    directories = sorted(
        path.parent
        for path in search_root.glob("*/leaderboard.json")
        if path.parent.name not in {"external"}
    )
    if not directories:
        location = experiments_root if experiments_root.is_dir() else study_dir
        raise FileNotFoundError(f"No experiment leaderboard.json files under {location}")
    return directories


def _cross_check_artifact_id(
    artifact: Mapping[str, Any],
    experiment_id: str,
    *,
    label: str,
    path: Path,
) -> None:
    artifact_id = artifact.get("experiment_id")
    if artifact_id is not None and artifact_id != experiment_id:
        raise ValueError(
            f"{label} experiment_id mismatch in {path}: "
            f"expected {experiment_id!r}, got {artifact_id!r}"
        )


def _load_experiment(directory: Path, study_dir: Path) -> dict[str, Any]:
    required = {
        "summary": directory / "summary.json",
        "leaderboard": directory / "leaderboard.json",
        "noise_ceiling": directory / "noise_ceiling.json",
    }
    missing = [f"{kind}: {path}" for kind, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required experiment artifact(s): " + "; ".join(missing))

    summary = _read_json_object(required["summary"], label="experiment summary")
    experiment_id = _nonempty_string(
        summary.get("experiment_id"), label=f"{required['summary']} experiment_id"
    )
    family = _nonempty_string(
        summary.get("family"), label=f"{required['summary']} family"
    )
    if directory.name != experiment_id:
        raise ValueError(
            f"Experiment directory {directory.name!r} does not match "
            f"summary experiment_id {experiment_id!r}"
        )

    try:
        raw_leaderboard = json.loads(required["leaderboard"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in leaderboard: {required['leaderboard']}") from exc
    if isinstance(raw_leaderboard, dict):
        _cross_check_artifact_id(
            raw_leaderboard,
            experiment_id,
            label="leaderboard",
            path=required["leaderboard"],
        )
    leaderboard = _leaderboard_rows(raw_leaderboard, path=required["leaderboard"])
    noise_ceiling = _read_json_object(
        required["noise_ceiling"], label="noise ceiling"
    )
    _cross_check_artifact_id(
        noise_ceiling,
        experiment_id,
        label="noise ceiling",
        path=required["noise_ceiling"],
    )

    interpretation_path = directory / "interpretation.json"
    interpretation = None
    source_files = [
        _artifact_record(path, study_dir, kind) for kind, path in required.items()
    ]
    if interpretation_path.is_file():
        interpretation = _read_json_object(
            interpretation_path, label="interpretation"
        )
        _cross_check_artifact_id(
            interpretation,
            experiment_id,
            label="interpretation",
            path=interpretation_path,
        )
        source_files.append(
            _artifact_record(interpretation_path, study_dir, "interpretation")
        )

    return {
        "experiment_id": experiment_id,
        "family": family,
        "summary": summary,
        "leaderboard": leaderboard,
        "noise_ceiling": noise_ceiling,
        "interpretation": interpretation,
        "source_files": source_files,
    }


def _external_score_path(study_dir: Path) -> Path:
    candidates = [
        study_dir / "external" / "score.json",
        study_dir / "external_score.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "Missing external score; expected external/score.json or "
            f"external_score.json under {study_dir}"
        )
    if len(existing) > 1:
        raise ValueError(
            "Ambiguous external score artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0]


def _external_score_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    if "methods" in value:
        raw_rows = value["methods"]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("external score methods must be a non-empty list")
        rows = []
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                raise TypeError(f"external score method row {index} must be an object")
            row = deepcopy(raw_row)
            row["method"] = _nonempty_string(
                row.get("method", row.get("name")),
                label=f"external score method row {index} method",
            )
            rows.append(row)
        return rows
    if not isinstance(value.get("total"), dict):
        raise ValueError("external score must contain a total object")
    row = deepcopy(value)
    row.setdefault("method", "selected_per_family")
    return [row]


def load_study_artifacts(study_dir: Path) -> dict[str, Any]:
    """Load and validate the JSON inputs used by both report formats."""

    study_dir = study_dir.resolve()
    if not study_dir.is_dir():
        raise FileNotFoundError(f"Study output directory does not exist: {study_dir}")
    experiments = [
        _load_experiment(directory, study_dir)
        for directory in _experiment_dirs(study_dir)
    ]
    experiment_ids = [experiment["experiment_id"] for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Duplicate experiment_id across study artifacts")

    external_path = _external_score_path(study_dir)
    external_score = _read_json_object(external_path, label="external score")
    external_rows = _external_score_rows(external_score)
    families = {experiment["family"] for experiment in experiments}
    for row in external_rows:
        by_family = row.get("by_family")
        if isinstance(by_family, dict):
            unexpected = sorted(set(by_family) - families)
            if unexpected:
                raise ValueError(
                    f"External score contains unknown families: {unexpected}"
                )

    source_files = [
        artifact
        for experiment in experiments
        for artifact in experiment["source_files"]
    ]
    source_files.append(_artifact_record(external_path, study_dir, "external_score"))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study_dir": str(study_dir),
        "experiments": experiments,
        "external_score": external_score,
        "source_files": sorted(source_files, key=lambda row: row["path"]),
    }


def _path_get(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_path(value: Any, *paths: Sequence[str]) -> Any:
    for path in paths:
        found = _path_get(value, path)
        if found is not None:
            return found
    return None


def _method_headlines(row: Mapping[str, Any]) -> dict[str, Any]:
    validation = row.get("validation", row.get("validation_metrics", {}))
    return {
        "rank": row.get("rank"),
        "method": row.get("method", row.get("name")),
        "cv_rmse_log": _first_path(
            row,
            ("cv", "rmse_log"),
            ("cv_rmse_log",),
        ),
        "validation_log_rmse": _first_path(
            validation,
            ("all", "log", "rmse"),
            ("log", "rmse"),
        ),
        "validation_raw_rmse": _first_path(
            validation,
            ("all", "raw", "rmse"),
            ("raw", "rmse"),
        ),
        "spearman": _first_path(
            validation,
            ("all", "ranking", "spearman"),
            ("ranking", "spearman"),
        ),
        "three_choice_accuracy": _first_path(
            validation,
            ("all", "three_choice", "accuracy"),
            ("three_choice", "accuracy"),
        ),
        "eligible_three_choice_accuracy": _first_path(
            validation,
            ("benchmark_eligible", "three_choice", "accuracy"),
        ),
    }


def _rank_value(row: Mapping[str, Any], fallback: int) -> tuple[float, int]:
    rank = row.get("rank")
    try:
        numeric = float(rank)
    except (TypeError, ValueError):
        numeric = float(fallback)
    if not math.isfinite(numeric):
        numeric = float(fallback)
    return numeric, fallback


def _build_highlights(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    winners = []
    for experiment in artifacts["experiments"]:
        leaderboard = experiment["leaderboard"]
        winner = min(
            enumerate(leaderboard, start=1),
            key=lambda pair: _rank_value(pair[1], pair[0]),
        )[1]
        ceiling = experiment["noise_ceiling"]
        winners.append(
            {
                "experiment_id": experiment["experiment_id"],
                "family": experiment["family"],
                "winner": _method_headlines(winner),
                "noise_ceiling_three_choice_accuracy": _first_path(
                    ceiling,
                    ("all", "median_metrics", "three_choice", "accuracy"),
                    ("median_metrics", "three_choice", "accuracy"),
                ),
            }
        )

    external_rows = _external_score_rows(artifacts["external_score"])
    external_headlines = []
    for row in external_rows:
        total = row.get("total", {})
        external_headlines.append(
            {
                "method": row["method"],
                "num_questions": total.get("num_questions"),
                "num_correct": total.get("num_correct"),
                "accuracy": total.get("accuracy"),
            }
        )
    return {
        "experiment_winners": winners,
        "external": external_headlines,
    }


def _format_number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    if percent:
        return f"{100.0 * number:.1f}%"
    return f"{number:.4g}"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_cell(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _summary_count(summary: Mapping[str, Any], name: str) -> Any:
    return _first_path(
        summary,
        (name,),
        ("data", name),
        ("dataset", name),
    )


def _interpretation_lines(interpretation: Mapping[str, Any] | None) -> list[str]:
    if interpretation is None:
        return ["No optional interpretation artifact was produced."]
    lines: list[str] = []
    for key, heading in (
        ("simple_rules", "Simple rules"),
        ("findings", "Findings"),
        ("notes", "Notes"),
    ):
        values = interpretation.get(key)
        if not isinstance(values, list) or not values:
            continue
        lines.extend([f"#### {heading}", ""])
        for value in values:
            if isinstance(value, Mapping):
                text = value.get("rule", value.get("finding", value.get("text")))
                if text is None:
                    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
            else:
                text = value
            lines.append(f"- {text}")
        lines.append("")
    if not lines:
        lines.append(
            "An interpretation artifact was retained in `report.json`; it has no "
            "recognized simple_rules/findings/notes list for Markdown rendering."
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a consolidated report payload as Markdown."""

    highlights = report["highlights"]
    lines = [
        "# ArchitectureIQ setting-to-loss meta-model study",
        "",
        "This report compares independently fitted meta-models within each dataset "
        "family. Lower loss-prediction error is better; higher ranking, three-choice, "
        "and external-question accuracy is better.",
        "",
        "## Executive summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "Family",
                "Selected method",
                "Validation log RMSE",
                "Validation Spearman",
                "3-choice accuracy",
                "Noise ceiling 3-choice",
            ],
            [
                [
                    winner["family"],
                    winner["winner"]["method"],
                    _format_number(winner["winner"]["validation_log_rmse"]),
                    _format_number(winner["winner"]["spearman"]),
                    _format_number(
                        winner["winner"]["three_choice_accuracy"], percent=True
                    ),
                    _format_number(
                        winner["noise_ceiling_three_choice_accuracy"], percent=True
                    ),
                ]
                for winner in highlights["experiment_winners"]
            ],
        )
    )

    lines.extend(["", "## Frozen external evaluation", ""])
    external_rows = _external_score_rows(report["external_score"])
    lines.extend(
        _markdown_table(
            ["Method", "Correct", "Questions", "Accuracy"],
            [
                [
                    row["method"],
                    row.get("total", {}).get("num_correct", "—"),
                    row.get("total", {}).get("num_questions", "—"),
                    _format_number(row.get("total", {}).get("accuracy"), percent=True),
                ]
                for row in external_rows
            ],
        )
    )
    for row in external_rows:
        by_family = row.get("by_family")
        if not isinstance(by_family, Mapping) or not by_family:
            continue
        lines.extend(["", f"### {_escape_cell(row['method'])}: by family", ""])
        lines.extend(
            _markdown_table(
                ["Family", "Correct", "Questions", "Accuracy"],
                [
                    [
                        family,
                        counts.get("num_correct", "—"),
                        counts.get("num_questions", "—"),
                        _format_number(counts.get("accuracy"), percent=True),
                    ]
                    for family, counts in sorted(by_family.items())
                    if isinstance(counts, Mapping)
                ],
            )
        )

    for experiment in report["experiments"]:
        summary = experiment["summary"]
        lines.extend(
            [
                "",
                f"## {experiment['family']}",
                "",
                f"Experiment: `{experiment['experiment_id']}`",
                "",
            ]
        )
        train_rows = _summary_count(summary, "train_rows")
        validation_rows = _summary_count(summary, "validation_rows")
        if train_rows is not None or validation_rows is not None:
            lines.append(
                "Data split: "
                f"{_format_number(train_rows)} train / "
                f"{_format_number(validation_rows)} validation rows."
            )
            lines.append("")
        lines.extend(["### Validation leaderboard", ""])
        headline_rows = [_method_headlines(row) for row in experiment["leaderboard"]]
        lines.extend(
            _markdown_table(
                [
                    "Rank",
                    "Method",
                    "CV log RMSE",
                    "Val log RMSE",
                    "Val raw RMSE",
                    "Spearman",
                    "3-choice",
                    "Eligible 3-choice",
                ],
                [
                    [
                        row["rank"],
                        row["method"],
                        _format_number(row["cv_rmse_log"]),
                        _format_number(row["validation_log_rmse"]),
                        _format_number(row["validation_raw_rmse"]),
                        _format_number(row["spearman"]),
                        _format_number(row["three_choice_accuracy"], percent=True),
                        _format_number(
                            row["eligible_three_choice_accuracy"], percent=True
                        ),
                    ]
                    for row in headline_rows
                ],
            )
        )

        ceiling = experiment["noise_ceiling"]
        lines.extend(["", "### Split-half noise ceiling", ""])
        ceiling_metrics = _first_path(
            ceiling,
            ("all", "median_metrics"),
            ("median_metrics",),
        )
        lines.extend(
            _markdown_table(
                ["Log RMSE", "Spearman", "3-choice accuracy"],
                [
                    [
                        _format_number(
                            _first_path(ceiling_metrics, ("log", "rmse"))
                        ),
                        _format_number(
                            _first_path(
                                ceiling_metrics, ("ranking", "spearman")
                            )
                        ),
                        _format_number(
                            _first_path(
                                ceiling_metrics,
                                ("three_choice", "accuracy"),
                            ),
                            percent=True,
                        ),
                    ]
                ],
            )
        )
        lines.extend(["", "### Interpretation", ""])
        lines.extend(_interpretation_lines(experiment["interpretation"]))

    lines.extend(
        [
            "",
            "## Evaluation contract",
            "",
            "Validation labels are used for final reporting, not hyperparameter "
            "selection. External predictions must be frozen before the answer key "
            "is opened. Noise ceilings come from complementary 5/5 splits of the "
            "stored ten-seed ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_report(
    study_dir: Path,
    *,
    markdown_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, Any]:
    """Load a completed study and atomically write ``report.md``/``report.json``."""

    study_dir = study_dir.resolve()
    markdown_path = markdown_path or study_dir / "report.md"
    json_path = json_path or study_dir / "report.json"
    if markdown_path.resolve() == json_path.resolve():
        raise ValueError("markdown_path and json_path must differ")

    report = load_study_artifacts(study_dir)
    report["highlights"] = _build_highlights(report)
    markdown = render_markdown(report)
    json_bytes = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(json_path, json_bytes)
    _atomic_write(markdown_path, markdown.encode("utf-8"))
    return report
