"""Read-only oracle, random-baseline, and winner-stability diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.meta_model_study.wide import load_corpus, load_snapshot, validate_root


SCHEMA_VERSION = "meta_model_bounds_v1"


def _loss(row: Mapping[str, Any]) -> float:
    value = float(row["target"]["mean_loss"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("target.mean_loss must be finite and positive")
    return value


def _stability_interval(row: Mapping[str, Any], z: float) -> tuple[float, float] | None:
    target = row.get("target")
    if not isinstance(target, Mapping):
        return None
    try:
        mean = float(target["mean_loss"])
        std = float(target["std_loss"])
        n_seeds = int(target["n_seeds"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(mean) or not math.isfinite(std) or std < 0 or n_seeds <= 0:
        return None
    radius = z * std / math.sqrt(n_seeds)
    return mean - radius, mean + radius


def analyze_environment(
    rows: Sequence[Mapping[str, Any]], *, z: float = 1.96
) -> dict[str, Any]:
    """Compute exact finite-row diagnostics for one environment."""

    if not math.isfinite(z) or z < 0:
        raise ValueError("z must be finite and non-negative")
    losses = [_loss(row) for row in rows]
    triples = list(combinations(range(len(rows)), 3))
    pairs = [
        pair
        for pair in combinations(range(len(rows)), 2)
        if losses[pair[0]] != losses[pair[1]]
    ]

    random_correct = 0.0
    random_regret = 0.0
    stable = 0
    proxy_eligible = 0
    for triple in triples:
        values = [losses[index] for index in triple]
        best = min(values)
        random_correct += sum(value == best for value in values) / 3.0
        random_regret += sum(value - best for value in values) / 3.0

        intervals = [_stability_interval(rows[index], z) for index in triple]
        if all(interval is not None for interval in intervals):
            proxy_eligible += 1
            winners = [
                position for position, value in enumerate(values) if value == best
            ]
            if len(winners) == 1:
                winner = winners[0]
                assert intervals[winner] is not None
                if all(
                    intervals[winner][1] < intervals[position][0]  # type: ignore[index]
                    for position in range(3)
                    if position != winner
                ):
                    stable += 1

    n_triples = len(triples)
    observed_range = max(losses) - min(losses) if losses else None
    return {
        "n_rows": len(rows),
        "observed_gt_oracle": {
            "top1_accuracy": 1.0 if n_triples else None,
            "mean_regret": 0.0 if n_triples else None,
        },
        "uniform_random_three_choice": {
            "n_triples": n_triples,
            "top1_accuracy": random_correct / n_triples if n_triples else None,
            "mean_regret": random_regret / n_triples if n_triples else None,
            "tie_policy": "a random choice is correct when it selects any exact GT minimizer",
        },
        "random_pair": {
            "n_comparable_pairs": len(pairs),
            "pair_concordance": 0.5 if pairs else None,
            "true_ties_excluded": math.comb(len(rows), 2) - len(pairs),
        },
        "winner_stability_empirical_proxy": {
            "label": "empirical proxy, not a theoretical ceiling",
            "method": "unique observed winner upper interval is below every rival lower interval",
            "interval": f"mean_loss +/- {z:g} * std_loss / sqrt(n_seeds)",
            "n_eligible_triples": proxy_eligible,
            "n_stable_triples": stable,
            "stable_fraction": stable / proxy_eligible if proxy_eligible else None,
        },
        "metric_ranges": {
            "top1_accuracy": [0.0, 1.0],
            "pair_concordance": [0.0, 1.0],
            "mean_regret": [0.0, None],
            "observed_environment_regret_upper": observed_range,
        },
        "finite_sample": {
            "effective_n_rows": len(rows),
            "n_triples": n_triples,
            "n_comparable_pairs": len(pairs),
            "warning": "overlapping pairs and triples are not independent; combinatorial counts are not effective sample sizes",
        },
    }


def analyze_corpus(
    corpus: Any, *, split: str = "validation", z: float = 1.96
) -> dict[str, Any]:
    environments: dict[str, Any] = {}
    for environment in corpus.environments:
        environments[environment.experiment_id] = analyze_environment(
            environment.rows(split), z=z
        )

    def weighted(section: str, metric: str, count: str) -> float | None:
        items = [value[section] for value in environments.values()]
        denominator = sum(int(item[count]) for item in items)
        if denominator == 0:
            return None
        return (
            sum(float(item[metric]) * int(item[count]) for item in items) / denominator
        )

    total_rows = sum(value["n_rows"] for value in environments.values())
    total_triples = sum(
        value["uniform_random_three_choice"]["n_triples"]
        for value in environments.values()
    )
    total_pairs = sum(
        value["random_pair"]["n_comparable_pairs"] for value in environments.values()
    )
    eligible = sum(
        value["winner_stability_empirical_proxy"]["n_eligible_triples"]
        for value in environments.values()
    )
    stable = sum(
        value["winner_stability_empirical_proxy"]["n_stable_triples"]
        for value in environments.values()
    )
    aggregate = {
        "n_environments": len(environments),
        "n_rows": total_rows,
        "observed_gt_oracle": {
            "top1_accuracy": 1.0 if total_triples else None,
            "mean_regret": 0.0 if total_triples else None,
        },
        "uniform_random_three_choice": {
            "n_triples": total_triples,
            "top1_accuracy": weighted(
                "uniform_random_three_choice", "top1_accuracy", "n_triples"
            ),
            "mean_regret": None,
            "mean_regret_aggregation": (
                "reported_per_environment_only; raw CE/MSE scales are not pooled"
            ),
        },
        "random_pair": {
            "n_comparable_pairs": total_pairs,
            "pair_concordance": 0.5 if total_pairs else None,
        },
        "winner_stability_empirical_proxy": {
            "label": "empirical proxy, not a theoretical ceiling",
            "n_eligible_triples": eligible,
            "n_stable_triples": stable,
            "stable_fraction": stable / eligible if eligible else None,
        },
        "metric_ranges": {
            "top1_accuracy": [0.0, 1.0],
            "pair_concordance": [0.0, 1.0],
            "mean_regret": [0.0, None],
        },
        "finite_sample": {
            "effective_n_environments": len(environments),
            "effective_n_rows": total_rows,
            "n_triples": total_triples,
            "n_comparable_pairs": total_pairs,
            "warning": "environment clusters and rows, not overlapping combinations, govern independent information",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "environments": environments,
        "aggregate": aggregate,
    }


def _load_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.snapshot_manifest is not None:
        snapshot = load_snapshot(args.snapshot_manifest)
        return snapshot.corpus, {
            "kind": "snapshot_manifest",
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
        }
    report = validate_root(args.dataset_root, plan_path=args.plan)
    if report["status"] != "complete":
        raise ValueError(f"wide dataset must be complete, got {report['status']!r}")
    corpus = load_corpus(
        args.dataset_root,
        expected_environment_count=report["expected"]["n_environments"],
        expected_n_seeds=report["expected"]["n_seeds"],
        require_no_partial=True,
    )
    return corpus, {
        "kind": "dataset_root_and_plan",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "plan": str(Path(args.plan).resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-manifest", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--split", choices=("all", "train", "validation"), default="validation"
    )
    parser.add_argument("--stability-z", type=float, default=1.96)
    args = parser.parse_args(argv)
    if args.dataset_root is not None and args.plan is None:
        parser.error("--dataset-root requires --plan")
    if args.snapshot_manifest is not None and args.plan is not None:
        parser.error("--plan cannot be used with --snapshot-manifest")
    corpus, source_info = _load_inputs(args)
    result = analyze_corpus(corpus, split=args.split, z=args.stability_z)
    result["source"] = source_info
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
