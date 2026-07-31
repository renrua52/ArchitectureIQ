"""Summarize full-wide split-half label reproducibility without training.

This command reads a frozen wide snapshot and the stored per-seed GT curves
referenced by its validation rows.  It never regenerates GT or fits a model.
The result is an empirical label-reproducibility estimate, not a mathematical
upper bound on accuracy for a fixed benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.meta_model_study.wide import WideEnvironment, load_snapshot
from tools.meta_model_study.wide_run import noise_ceiling_for_environment


SCHEMA_VERSION = "meta_model_full_wide_split_half_reproducibility_v1"
VIEWS = ("all", "benchmark_eligible")
METRIC_PATHS = {
    "three_choice_accuracy": ("three_choice", "accuracy"),
    "gap_ge_0_05_three_choice_accuracy": (
        "three_choice",
        "gap_ge_0_05",
        "accuracy",
    ),
    "pair_concordance": ("ranking", "pair_concordance"),
    "spearman": ("ranking", "spearman"),
    "log_rmse": ("log", "rmse"),
}


def _dig(value: object, path: Sequence[str]) -> object:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _optional_finite_number(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _non_negative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return value


def _view_metrics(value: object, *, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping or null")
    n_rows = _non_negative_int(value.get("n_rows"), context=f"{context}.n_rows")
    median = value.get("median_metrics")
    if not isinstance(median, Mapping):
        raise TypeError(f"{context}.median_metrics must be a mapping")
    metrics = {
        name: _optional_finite_number(
            _dig(median, path),
            context=f"{context}.median_metrics.{'.'.join(path)}",
        )
        for name, path in METRIC_PATHS.items()
    }
    return {
        "n_rows": n_rows,
        "median_split_half": metrics,
    }


def _computed_environment(
    environment: WideEnvironment,
    ceiling: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": environment.experiment_id,
        "family": environment.family,
        "dataset_id": environment.dataset_id,
        "status": "computed",
        "reason": None,
        "n_seeds": _non_negative_int(ceiling.get("n_seeds"), context="ceiling.n_seeds"),
        "split_sizes": list(ceiling.get("split_sizes", [])),
        "n_complementary_partitions": _non_negative_int(
            ceiling.get("n_complementary_partitions"),
            context="ceiling.n_complementary_partitions",
        ),
        "n_directed_comparisons": _non_negative_int(
            ceiling.get("n_directed_comparisons"),
            context="ceiling.n_directed_comparisons",
        ),
        "views": {
            view: _view_metrics(
                ceiling.get(view),
                context=f"{environment.experiment_id}.{view}",
            )
            for view in VIEWS
        },
    }


def _skipped_environment(
    environment: WideEnvironment,
    ceiling: Mapping[str, Any],
) -> dict[str, Any]:
    reason = ceiling.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError(
            f"Skipped environment {environment.experiment_id} has no reason"
        )
    return {
        "experiment_id": environment.experiment_id,
        "family": environment.family,
        "dataset_id": environment.dataset_id,
        "status": "skipped",
        "reason": reason,
        "n_seeds": environment.n_seeds,
        "split_sizes": None,
        "n_complementary_partitions": None,
        "n_directed_comparisons": None,
        "views": {view: None for view in VIEWS},
    }


def _status_coverage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts = Counter(str(record["status"]) for record in records)
    computed = counts.get("computed", 0)
    skipped = counts.get("skipped", 0)
    return {
        "total_environments": total,
        "computed_environments": computed,
        "skipped_environments": skipped,
        "computed_fraction": computed / total if total else None,
        "status_counts": dict(sorted(counts.items())),
    }


def _view_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    available = [
        record
        for record in records
        if isinstance(_dig(record, ("views", view)), Mapping)
    ]
    total = len(records)
    computed = len(available)
    macro: dict[str, float | None] = {}
    metric_coverage: dict[str, dict[str, Any]] = {}
    for metric in METRIC_PATHS:
        values = [
            _optional_finite_number(
                _dig(record, ("views", view, "median_split_half", metric)),
                context=f"{record['experiment_id']}.{view}.{metric}",
            )
            for record in available
        ]
        finite_values = [value for value in values if value is not None]
        macro[metric] = (
            float(sum(finite_values) / len(finite_values)) if finite_values else None
        )
        metric_coverage[metric] = {
            "computed_environments": len(finite_values),
            "skipped_environments": total - len(finite_values),
            "computed_fraction": len(finite_values) / total if total else None,
        }
    return {
        "coverage": {
            "total_environments": total,
            "computed_environments": computed,
            "skipped_environments": total - computed,
            "computed_fraction": computed / total if total else None,
        },
        "environment_equal_macro": macro,
        "metric_coverage": metric_coverage,
    }


def _summarize_view(
    records: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    families = sorted({str(record["family"]) for record in records})
    return {
        "overall": _view_summary(records, view=view),
        "by_family": {
            family: _view_summary(
                [record for record in records if record["family"] == family],
                view=view,
            )
            for family in families
        },
    }


def summarize_split_half(
    snapshot_manifest: str | Path,
    *,
    missing_curves: str = "require",
) -> dict[str, Any]:
    """Read stored seed curves and return environment-equal reproducibility macros."""

    if missing_curves not in {"require", "skip"}:
        raise ValueError("missing_curves must be 'require' or 'skip'")
    snapshot = load_snapshot(snapshot_manifest)
    mode = "require" if missing_curves == "require" else "auto"
    records: list[dict[str, Any]] = []
    for environment in snapshot.corpus.environments:
        try:
            ceiling = noise_ceiling_for_environment(environment, mode=mode)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Stored seed curves are required for "
                f"{environment.experiment_id}: {error}"
            ) from error
        status = ceiling.get("status")
        if status == "computed":
            records.append(_computed_environment(environment, ceiling))
        elif status == "skipped" and missing_curves == "skip":
            records.append(_skipped_environment(environment, ceiling))
        else:
            raise ValueError(
                f"Unexpected ceiling status for {environment.experiment_id}: {status!r}"
            )

    families = sorted({record["family"] for record in records})
    return {
        "schema_version": SCHEMA_VERSION,
        "estimate": {
            "name": "empirical_split_half_label_reproducibility",
            "description": (
                "Agreement between complementary averages of stored per-seed "
                "validation labels, summarized by the median directed split-half "
                "comparison within each environment."
            ),
            "is_mathematical_upper_bound": False,
            "caveat": (
                "This is an empirical label-reproducibility estimate. It is not a "
                "mathematical upper bound on agent or meta-model accuracy for a "
                "fixed benchmark."
            ),
        },
        "snapshot_manifest": str(snapshot.path),
        "snapshot_sha256": snapshot.sha256,
        "split": "validation",
        "missing_curves": missing_curves,
        "aggregation": {
            "within_environment": "median_over_directed_complementary_seed_splits",
            "across_environments": "equal_weight_mean_of_environment_medians",
            "stored_seed_curves_only": True,
        },
        "coverage": {
            "overall": _status_coverage(records),
            "by_family": {
                family: _status_coverage(
                    [record for record in records if record["family"] == family]
                )
                for family in families
            },
        },
        "all": _summarize_view(records, view="all"),
        "benchmark_eligible": _summarize_view(records, view="benchmark_eligible"),
        "environments": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--missing-curves",
        choices=("require", "skip"),
        default="require",
        help=(
            "require all stored seed curves (default), or skip environments whose "
            "stored curves are unavailable"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_split_half(
        args.snapshot_manifest,
        missing_curves=args.missing_curves,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
