#!/usr/bin/env python3
"""Build report-facing diagnostics for completed wide setting-to-loss runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import joblib

from tools.meta_model_study.wide import load_snapshot


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path("data/meta_model/setting_to_loss_wide_v2")
DEFAULT_WITH = Path(
    "data/meta_model_studies/setting_to_loss_wide_v2_b1_with_params/id"
)
DEFAULT_WITHOUT = Path(
    "data/meta_model_studies/setting_to_loss_wide_v2_b1_no_params/id"
)
DEFAULT_OUTPUT = Path(
    "artifacts/high_budget_gpt54_eval/wide_b1_setting_to_loss_analysis.json"
)


def load_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_view(method: dict[str, Any]) -> dict[str, Any]:
    test = method["test"]["all"]
    return {
        "n": test["n"],
        "raw": test.get("raw"),
        "log": test["log"],
        "ranking": test["ranking"],
        "per_environment": test.get("per_environment", {}),
        "within_environment": test["within_environment"],
        "three_choice": test["within_environment"]["three_choice"],
    }


def method_map(leaderboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["method"]: item for item in leaderboard["methods"]}


def champion_view(leaderboard: dict[str, Any]) -> dict[str, Any]:
    champion = leaderboard["cv_champion"]
    return {
        "method": champion,
        **metric_view(method_map(leaderboard)[champion]),
    }


def macro_three_choice(row: dict[str, Any]) -> float:
    within = row["within_environment"]
    macro = within.get("macro")
    if isinstance(macro, dict) and macro.get("three_choice_accuracy") is not None:
        return float(macro["three_choice_accuracy"])
    return float(within["three_choice"]["accuracy"])


def heldout_oracle_view(leaderboard: dict[str, Any]) -> dict[str, Any]:
    selected = max(
        leaderboard["methods"],
        key=lambda item: item["test"]["all"]["within_environment"][
            "three_choice"
        ]["accuracy"],
    )
    return {"method": selected["method"], **metric_view(selected)}


def prediction_map(path: Path, method: str) -> dict[str, float]:
    data = load_json(path)
    selected = next(item for item in data["methods"] if item["method"] == method)
    return {
        item["example_fingerprint_sha256"]: item["predicted_log_loss"]
        for item in selected["predictions"]
    }


def setting_summary(row: dict[str, Any]) -> dict[str, Any]:
    setting = row["setting"]
    model = setting["model"]
    optimizer = setting["optimizer"]
    loss = setting["loss"]
    model_summary = {
        key: model[key]
        for key in (
            "type",
            "depth",
            "width",
            "residual",
            "d_model",
            "d_ff",
            "num_layers",
            "num_heads",
        )
        if key in model
    }
    return {
        "fingerprint": row["example_fingerprint_sha256"],
        "environment": row["experiment_id"],
        "family": row["family"],
        "candidate_id_short": row["candidate_id_short"],
        "optimizer": optimizer["type"],
        "learning_rate": optimizer["lr"],
        "weight_decay": optimizer.get("weight_decay"),
        "loss_id": loss["loss_id"],
        "loss_lambda": loss.get("lambda"),
        "model": model_summary,
        "total_params": row.get("derived", {}).get("total_params"),
        "actual_log_loss": row["target"]["log_mean_loss"],
        "actual_loss": row["target"]["mean_loss"],
        "benchmark_eligible": row["target"].get("benchmark_eligible"),
    }


def top_feature_importance(model_path: Path, limit: int = 12) -> list[dict[str, Any]]:
    fitted = joblib.load(ROOT / model_path)
    pipeline = fitted.estimator
    encoder = pipeline.named_steps["features"]
    estimator = pipeline.named_steps["model"]
    names = encoder.feature_names
    values = estimator.feature_importances_.tolist()
    ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": importance}
        for feature, importance in ranked[:limit]
    ]


def lr_band(value: float) -> str:
    if value <= 1e-4:
        return "<=1e-4"
    if value <= 3e-3:
        return "3e-4..3e-3"
    if value <= 3e-2:
        return "1e-2..3e-2"
    return ">=1e-1"


def grouped_error(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row["abs_log_error_with_params"])
    return [
        {"group": group, "n": len(values), "log_mae": mean(values)}
        for group, values in sorted(groups.items())
    ]


def optional_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None


def environment_bootstrap_ci(
    per_family: dict[str, dict[str, Any]],
    *,
    seed: int = 20260715,
    draws: int = 5000,
) -> dict[str, Any] | None:
    values_by_family: dict[str, list[float]] = {}
    for family, metrics in per_family.items():
        per_environment = metrics.get("per_environment")
        if not isinstance(per_environment, dict) or not per_environment:
            return None
        values_by_family[family] = [
            float(item["three_choice"]["all"]["accuracy"])
            for item in per_environment.values()
        ]
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        family_means = []
        for values in values_by_family.values():
            family_means.append(
                mean(generator.choice(values) for _ in range(len(values)))
            )
        samples.append(mean(family_means))
    samples.sort()
    return {
        "unit": "environment bootstrap within family, then equal-family macro",
        "draws": draws,
        "seed": seed,
        "lower": samples[int(0.025 * draws)],
        "upper": samples[int(0.975 * draws) - 1],
    }


def analyze(
    *,
    corpus_root: Path | None,
    snapshot_manifest: Path | None,
    with_root: Path,
    without_root: Path,
) -> dict[str, Any]:
    with_manifest = load_json(with_root / "manifest.json")
    without_manifest = load_json(without_root / "manifest.json")
    environment_rows: list[dict[str, Any]] = []
    if snapshot_manifest is not None:
        snapshot = load_snapshot(ROOT / snapshot_manifest)
        environments_by_id = {
            environment.experiment_id: environment
            for environment in snapshot.corpus.environments
        }
        corpus_counts = {
            "n_environments": len(snapshot.corpus.environments),
            "n_selected_settings": len(snapshot.corpus.all_rows),
            "n_train": len(snapshot.corpus.train_rows),
            "n_validation": len(snapshot.corpus.validation_rows),
            "n_benchmark_eligible": sum(
                row["target"].get("benchmark_eligible") is True
                for row in snapshot.corpus.all_rows
            ),
            "n_validation_benchmark_eligible": sum(
                row["target"].get("benchmark_eligible") is True
                for row in snapshot.corpus.validation_rows
            ),
        }
        snapshot_sha256 = snapshot.sha256
    else:
        if corpus_root is None:
            raise ValueError("corpus_root is required without snapshot_manifest")
        environments_by_id = {}
        corpus_counts = {}
        snapshot_sha256 = None
    validation_rows: list[dict[str, Any]] = []
    train_count = 0
    benchmark_eligible_count = 0
    validation_benchmark_eligible_count = 0

    for with_path_abs in sorted((ROOT / with_root / "environment").glob("*/leaderboard.json")):
        task_id = with_path_abs.parent.name
        with_path = with_path_abs.relative_to(ROOT)
        without_path = without_root / "environment" / task_id / "leaderboard.json"
        with_board = load_json(with_path)
        without_board = load_json(without_path)
        with_champion = champion_view(with_board)
        without_champion = champion_view(without_board)
        with_oracle = heldout_oracle_view(with_board)
        if snapshot_manifest is not None:
            phase = str(environments_by_id[task_id].manifest["config"]["phase"])
        else:
            assert corpus_root is not None
            phase = str(
                load_json(corpus_root / task_id / "manifest.json")["config"][
                    "phase"
                ]
            )
        environment_rows.append(
            {
                "task_id": task_id,
                "phase": phase,
                "family": with_board["protocol"]["family"],
                "dataset": with_board["protocol"]["dataset"],
                "with_parameter_count": with_champion,
                "without_parameter_count": without_champion,
                "heldout_oracle_diagnostic": with_oracle,
                "three_choice_delta_without_minus_with": (
                    without_champion["three_choice"]["accuracy"]
                    - with_champion["three_choice"]["accuracy"]
                ),
                "selection_gap_to_heldout_oracle": (
                    with_oracle["three_choice"]["accuracy"]
                    - with_champion["three_choice"]["accuracy"]
                ),
                "source_paths": {
                    "with_parameter_count": str(with_path),
                    "without_parameter_count": str(without_path),
                },
            }
        )
        if snapshot_manifest is not None:
            environment = environments_by_id[task_id]
            validation_rows.extend(environment.validation_rows)
            train_count += len(environment.train_rows)
            benchmark_eligible_count += sum(
                row["target"].get("benchmark_eligible") is True
                for row in environment.all_rows
            )
            validation_benchmark_eligible_count += sum(
                row["target"].get("benchmark_eligible") is True
                for row in environment.validation_rows
            )
        else:
            assert corpus_root is not None
            environment_validation = load_jsonl(
                corpus_root / task_id / "validation.jsonl"
            )
            environment_train = load_jsonl(corpus_root / task_id / "train.jsonl")
            validation_rows.extend(environment_validation)
            train_count += len(environment_train)
            benchmark_eligible_count += sum(
                row["target"].get("benchmark_eligible") is True
                for row in (*environment_train, *environment_validation)
            )
            validation_benchmark_eligible_count += sum(
                row["target"].get("benchmark_eligible") is True
                for row in environment_validation
            )

    family_rows: list[dict[str, Any]] = []
    family_method_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "with_parameter_count": {},
        "without_parameter_count": {},
    }
    case_rows: list[dict[str, Any]] = []
    feature_importance: dict[str, Any] = {}
    for family_path_abs in sorted((ROOT / with_root / "family").glob("*/leaderboard.json")):
        family = family_path_abs.parent.name
        with_path = family_path_abs.relative_to(ROOT)
        without_path = without_root / "family" / family / "leaderboard.json"
        with_board = load_json(with_path)
        without_board = load_json(without_path)
        family_method_metrics["with_parameter_count"][family] = {
            item["method"]: {
                **metric_view(item),
                "selection_cv_log_rmse": item["cv_rmse_log"],
            }
            for item in with_board["methods"]
        }
        family_method_metrics["without_parameter_count"][family] = {
            item["method"]: {
                **metric_view(item),
                "selection_cv_log_rmse": item["cv_rmse_log"],
            }
            for item in without_board["methods"]
        }
        family_rows.append(
            {
                "family": family,
                "with_parameter_count": champion_view(with_board),
                "without_parameter_count": champion_view(without_board),
                "source_paths": {
                    "with_parameter_count": str(with_path),
                    "without_parameter_count": str(without_path),
                },
            }
        )
        with_predictions = prediction_map(
            with_root / "family" / family / "predictions.json", "xgboost"
        )
        without_predictions = prediction_map(
            without_root / "family" / family / "predictions.json", "xgboost"
        )
        for row in validation_rows:
            if row["family"] != family:
                continue
            fingerprint = row["example_fingerprint_sha256"]
            if fingerprint not in with_predictions or fingerprint not in without_predictions:
                continue
            actual = row["target"]["log_mean_loss"]
            with_prediction = with_predictions[fingerprint]
            without_prediction = without_predictions[fingerprint]
            summary = setting_summary(row)
            summary.update(
                {
                    "predicted_log_loss_with_params": with_prediction,
                    "predicted_log_loss_without_params": without_prediction,
                    "abs_log_error_with_params": abs(with_prediction - actual),
                    "abs_log_error_without_params": abs(without_prediction - actual),
                    "parameter_count_error_benefit": (
                        abs(without_prediction - actual)
                        - abs(with_prediction - actual)
                    ),
                    "lr_band": lr_band(float(row["setting"]["optimizer"]["lr"])),
                }
            )
            case_rows.append(summary)

        all_with_importance = top_feature_importance(
            with_root / "family" / family / "models" / "xgboost.joblib",
            limit=10_000,
        )
        parameter_feature = next(
            (
                {**item, "rank": index}
                for index, item in enumerate(all_with_importance, start=1)
                if item["feature"] == "derived.log_total_params"
            ),
            None,
        )
        feature_importance[family] = {
            "with_parameter_count": all_with_importance[:12],
            "without_parameter_count": top_feature_importance(
                without_root / "family" / family / "models" / "xgboost.joblib"
            ),
            "parameter_count_feature": parameter_feature,
        }

    with_family_aggregate = load_json(with_root / "family" / "aggregate.json")
    without_family_aggregate = load_json(without_root / "family" / "aggregate.json")

    def aggregate_methods(data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"method": item["method"], **metric_view(item)}
            for item in data["methods"]
        ]

    by_optimizer_rows = [dict(row, optimizer_type=row["optimizer"]) for row in case_rows]
    phase_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"environments": 0, "settings": 0, "train": 0, "validation": 0}
    )
    for task_id in [row["task_id"] for row in environment_rows]:
        if snapshot_manifest is not None:
            environment = environments_by_id[task_id]
            phase = str(environment.manifest["config"]["phase"])
            n_all = len(environment.all_rows)
            n_train = len(environment.train_rows)
            n_validation = len(environment.validation_rows)
            family = environment.family
        else:
            assert corpus_root is not None
            manifest = load_json(corpus_root / task_id / "manifest.json")
            phase = str(manifest["config"]["phase"])
            n_all = int(manifest["selected"]["total"])
            n_train = int(manifest["split_policy"]["train"])
            n_validation = int(manifest["split_policy"]["validation"])
            family = str(manifest["config"]["group_labels"]["family"])
        phase_counts[phase] += 1
        family_counts[family]["environments"] += 1
        family_counts[family]["settings"] += n_all
        family_counts[family]["train"] += n_train
        family_counts[family]["validation"] += n_validation

    family_balanced = {
        condition: mean(
            macro_three_choice(row[condition])
            for row in family_rows
        )
        for condition in ("with_parameter_count", "without_parameter_count")
    }

    def phase_family_macro(phase: str, condition: str) -> float | None:
        by_family: dict[str, list[float]] = defaultdict(list)
        for row in environment_rows:
            if row["phase"] != phase:
                continue
            by_family[row["family"]].append(
                float(row[condition]["three_choice"]["accuracy"])
            )
        if not by_family:
            return None
        return mean(mean(values) for values in by_family.values())

    family_balanced_methods: dict[str, list[dict[str, Any]]] = {}
    for condition, by_family in family_method_metrics.items():
        shared_methods = set.intersection(
            *(set(methods) for methods in by_family.values())
        )
        method_rows: list[dict[str, Any]] = []
        for method in sorted(shared_methods):
            per_family = {
                family: methods[method] for family, methods in sorted(by_family.items())
            }
            method_rows.append(
                {
                    "method": method,
                    "selection_cv_log_rmse": mean(
                        metrics["selection_cv_log_rmse"]
                        for metrics in per_family.values()
                    ),
                    "family_macro": {
                        "log_rmse": mean(
                            metrics["within_environment"].get("macro", {}).get(
                                "log_rmse", metrics["log"]["rmse"]
                            )
                            for metrics in per_family.values()
                        ),
                        "spearman": optional_mean(
                            metrics["within_environment"].get("macro", {}).get(
                                "spearman",
                                metrics["within_environment"].get(
                                    "spearman_macro"
                                ),
                            )
                            for metrics in per_family.values()
                        ),
                        "three_choice_accuracy": mean(
                            macro_three_choice(metrics)
                            for metrics in per_family.values()
                        ),
                    },
                    "per_family": per_family,
                    "bootstrap_95_ci": environment_bootstrap_ci(per_family),
                }
            )
        method_rows.sort(
            key=lambda row: (
                row["selection_cv_log_rmse"], row["method"]
            )
        )
        family_balanced_methods[condition] = method_rows

    family_champion_metrics = {
        condition: {
            row["family"]: row[condition]
            for row in family_rows
        }
        for condition in ("with_parameter_count", "without_parameter_count")
    }
    return {
        "schema_version": "architecture_iq.wide_setting_to_loss_report.v1",
        "status": {
            "included": (
                "frozen completed-GT snapshot"
                if snapshot_manifest is not None
                else "wide-v2 b1_pilot only"
            ),
            "decision": "COMPLETED_ENVIRONMENTS_ONLY",
            "excluded": "environments without complete validated exports",
            "excluded_environment_count": 30 - len(environment_rows),
            "target": "log(mean_loss) from stored executed-candidate GT",
            "remote_audit": {
                "ref": "origin/codex/complete-local-features@11ccaff",
                "finding": "wide_v2 setting-to-loss is the current priority; the remote ref adds routing documentation but no completed wide-v2 score artifact",
            },
        },
        "corpus": {
            "phases": dict(sorted(phase_counts.items())),
            "n_environments": corpus_counts.get(
                "n_environments", len(environment_rows)
            ),
            "n_selected_settings": corpus_counts.get(
                "n_selected_settings", train_count + len(validation_rows)
            ),
            "n_train": corpus_counts.get("n_train", train_count),
            "n_validation": corpus_counts.get(
                "n_validation", len(validation_rows)
            ),
            "n_benchmark_eligible": corpus_counts.get(
                "n_benchmark_eligible",
                benchmark_eligible_count,
            ),
            "n_validation_benchmark_eligible": corpus_counts.get(
                "n_validation_benchmark_eligible",
                validation_benchmark_eligible_count,
            ),
            "n_validation_triples": sum(
                math.comb(item["with_parameter_count"]["n"], 3)
                for item in environment_rows
            ),
            "family_distribution": dict(sorted(family_counts.items())),
            "snapshot_manifest_sha256": snapshot_sha256,
            "include_parameter_count_runs": [
                with_manifest["include_parameter_count"],
                without_manifest["include_parameter_count"],
            ],
        },
        "per_environment": environment_rows,
        "family_pooled": family_rows,
        "overall_methods": {
            "with_parameter_count": aggregate_methods(with_family_aggregate),
            "without_parameter_count": aggregate_methods(without_family_aggregate),
        },
        "family_balanced_methods": family_balanced_methods,
        "phenomena": {
            "family_balanced_macro_three_choice": family_balanced,
            "family_balanced_champion_bootstrap_95_ci": {
                condition: environment_bootstrap_ci(metrics)
                for condition, metrics in family_champion_metrics.items()
            },
            "balanced_b1_anchor": {
                "with_parameter_count": phase_family_macro(
                    "b1_pilot", "with_parameter_count"
                ),
                "without_parameter_count": phase_family_macro(
                    "b1_pilot", "without_parameter_count"
                ),
                "definition": "per-environment CV champions; environment mean within family, then equal-family macro",
            },
            "per_environment_champion_macro_three_choice": {
                "with_parameter_count": mean(
                    row["with_parameter_count"]["three_choice"]["accuracy"]
                    for row in environment_rows
                ),
                "without_parameter_count": mean(
                    row["without_parameter_count"]["three_choice"]["accuracy"]
                    for row in environment_rows
                ),
            },
            "parameter_count_ablation": {
                "environments_improved_without_parameter_count": sum(
                    row["three_choice_delta_without_minus_with"] > 0
                    for row in environment_rows
                ),
                "environments_hurt_without_parameter_count": sum(
                    row["three_choice_delta_without_minus_with"] < 0
                    for row in environment_rows
                ),
                "environments_tied_without_parameter_count": sum(
                    row["three_choice_delta_without_minus_with"] == 0
                    for row in environment_rows
                ),
                "largest_improvements_without_parameter_count": sorted(
                    environment_rows,
                    key=lambda row: row["three_choice_delta_without_minus_with"],
                    reverse=True,
                )[:3],
                "largest_drops_without_parameter_count": sorted(
                    environment_rows,
                    key=lambda row: row["three_choice_delta_without_minus_with"],
                )[:3],
                "by_family": [
                    {
                        "family": row["family"],
                        "with_parameter_count_method": row[
                            "with_parameter_count"
                        ]["method"],
                        "with_parameter_count_accuracy": macro_three_choice(
                            row["with_parameter_count"]
                        ),
                        "without_parameter_count_method": row[
                            "without_parameter_count"
                        ]["method"],
                        "without_parameter_count_accuracy": macro_three_choice(
                            row["without_parameter_count"]
                        ),
                        "delta_without_minus_with": (
                            macro_three_choice(row["without_parameter_count"])
                            - macro_three_choice(row["with_parameter_count"])
                        ),
                    }
                    for row in family_rows
                ],
            },
            "cv_selection": {
                "champion_matches_heldout_oracle": sum(
                    row["with_parameter_count"]["method"]
                    == row["heldout_oracle_diagnostic"]["method"]
                    for row in environment_rows
                ),
                "n_environments": len(environment_rows),
                "mean_three_choice_gap_to_heldout_oracle": mean(
                    row["selection_gap_to_heldout_oracle"]
                    for row in environment_rows
                ),
                "warning": "Heldout oracle is a post-hoc diagnostic, never a selectable method.",
            },
            "hardest_environments": sorted(
                environment_rows,
                key=lambda row: row["with_parameter_count"]["three_choice"][
                    "accuracy"
                ],
            )[:5],
            "largest_cv_selection_gaps": sorted(
                environment_rows,
                key=lambda row: row["selection_gap_to_heldout_oracle"],
                reverse=True,
            )[:5],
            "worst_xgboost_log_errors": sorted(
                case_rows,
                key=lambda row: row["abs_log_error_with_params"],
                reverse=True,
            )[:12],
            "parameter_count_helps_cases": sorted(
                case_rows,
                key=lambda row: row["parameter_count_error_benefit"],
                reverse=True,
            )[:8],
            "parameter_count_hurts_cases": sorted(
                case_rows,
                key=lambda row: row["parameter_count_error_benefit"],
            )[:8],
            "xgboost_error_by_optimizer": grouped_error(
                by_optimizer_rows, "optimizer_type"
            ),
            "xgboost_error_by_lr_band": grouped_error(case_rows, "lr_band"),
        },
        "xgboost_feature_importance": feature_importance,
        "source_paths": {
            "corpus": str(corpus_root) if corpus_root is not None else None,
            "snapshot_manifest": (
                str(snapshot_manifest) if snapshot_manifest is not None else None
            ),
            "with_parameter_count": str(with_root),
            "without_parameter_count": str(without_root),
            "b1_audit": "data/meta_model/setting_to_loss_wide_v2/audit_b1_pilot.json",
            "b2_audit": "data/meta_model/setting_to_loss_wide_v2/audit_b2_partial.json",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--snapshot-manifest", type=Path, default=None)
    parser.add_argument("--with-root", type=Path, default=DEFAULT_WITH)
    parser.add_argument("--without-root", type=Path, default=DEFAULT_WITHOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(
        corpus_root=None if args.snapshot_manifest is not None else args.corpus_root,
        snapshot_manifest=args.snapshot_manifest,
        with_root=args.with_root,
        without_root=args.without_root,
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
