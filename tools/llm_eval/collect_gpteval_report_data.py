#!/usr/bin/env python3
"""Collect GPT, heuristic, and meta-model report data with provenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rel(path: str | Path) -> str:
    return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 1)


def fair_sequential_rows() -> list[dict[str, Any]]:
    path = Path("artifacts/quiz_attempt_60/fair_sequential_v2/aggregate.json")
    data = load_json(path)
    labels = {"gpt-5.4": "GPT-5.4", "gpt-5.5": "GPT-5.5", "gpt-5.6-sol": "GPT-5.6-SOL"}
    rows: list[dict[str, Any]] = []
    for key, value in data["models"].items():
        run_correct = [run["correct"] for run in value["runs"]]
        rows.append(
            {
                "method": labels[key],
                "exact_model_id": key,
                "provider": "OpenAI/Codex subagent",
                "question_set": "old_60_clean",
                "protocol": "fair_sequential_v2",
                "category": "primary",
                "correct": sum(run_correct) / len(run_correct),
                "total": 60,
                "accuracy": value["mean_accuracy"],
                "accuracy_pct": pct(value["mean_accuracy"]),
                "runs": run_correct,
                "source_path": rel(path),
            }
        )
    return rows


def old_llm_rows() -> list[dict[str, Any]]:
    rows = fair_sequential_rows()

    perq_path = Path("artifacts/_archive_default_unused/quiz_attempt_65/per_question_blind_gpt54_gpt55/summary.json")
    perq = load_json(perq_path)
    for item in perq["results"].values():
        rows.append(
            {
                "method": "GPT-5.4" if item["model"] == "gpt-5.4" else "GPT-5.5",
                "exact_model_id": item["model"],
                "provider": "OpenAI/Codex subagent",
                "question_set": "old_60_clean",
                "protocol": "per_question_blind",
                "category": "historical",
                "correct": item["correct"],
                "total": item["total"],
                "accuracy": item["accuracy"],
                "accuracy_pct": pct(item["accuracy"]),
                "source_path": rel(perq_path),
            }
        )

    gpt56_single_path = Path("artifacts/quiz_attempt_60/gpt56_sol/single_question_blind/score.json")
    gpt56_single = load_json(gpt56_single_path)
    rows.append(
        {
            "method": "GPT-5.6-SOL",
            "exact_model_id": "gpt-5.6-sol",
            "provider": "OpenAI/Codex subagent",
            "question_set": "old_60_clean",
            "protocol": "per_question_blind",
            "category": "historical",
            "correct": gpt56_single["correct"],
            "total": gpt56_single["total"],
            "accuracy": gpt56_single["accuracy"],
            "accuracy_pct": pct(gpt56_single["accuracy"]),
            "source_path": rel(gpt56_single_path),
        }
    )

    gpt54_full_path = Path("artifacts/_archive_default_unused/quiz_attempt_65/gpt54_blind_10/summary.json")
    gpt54_full = load_json(gpt54_full_path)
    rows.append(
        {
            "method": "GPT-5.4",
            "exact_model_id": "gpt-5.4",
            "provider": "OpenAI/Codex subagent",
            "question_set": "old_60_clean",
            "protocol": "full_set_blind",
            "category": "historical",
            "correct": gpt54_full["aggregate"]["mean_correct"],
            "total": 60,
            "accuracy": gpt54_full["aggregate"]["mean_accuracy"],
            "accuracy_pct": pct(gpt54_full["aggregate"]["mean_accuracy"]),
            "source_path": rel(gpt54_full_path),
        }
    )

    gpt55_full_path = Path("artifacts/_archive_default_unused/quiz_attempt_65/experiment_summary.md")
    rows.append(
        {
            "method": "GPT-5.5",
            "exact_model_id": "gpt-5.5",
            "provider": "OpenAI/Codex subagent",
            "question_set": "old_60_clean",
            "protocol": "full_set_blind",
            "category": "historical",
            "correct": 34.7,
            "total": 60,
            "accuracy": 34.7 / 60,
            "accuracy_pct": pct(34.7 / 60),
            "note": "Clean-60 filtered result from archive experiment summary; individual clean runs 32, 37, 35.",
            "source_path": rel(gpt55_full_path),
        }
    )

    gpt56_full_path = Path("artifacts/quiz_attempt_60/gpt56_sol/blind/summary.json")
    gpt56_full = load_json(gpt56_full_path)
    rows.append(
        {
            "method": "GPT-5.6-SOL",
            "exact_model_id": "gpt-5.6-sol",
            "provider": "OpenAI/Codex subagent",
            "question_set": "old_60_clean",
            "protocol": "full_set_blind",
            "category": "historical",
            "correct": gpt56_full["mean_correct"],
            "total": 60,
            "accuracy": gpt56_full["mean_accuracy"],
            "accuracy_pct": pct(gpt56_full["mean_accuracy"]),
            "source_path": rel(gpt56_full_path),
        }
    )

    grouped_sources = [
        ("GPT-5.4", "gpt-5.4", Path("artifacts/quiz_attempt_60/gpt54/grouped_10/summary.json")),
        ("GPT-5.6-SOL", "gpt-5.6-sol", Path("artifacts/quiz_attempt_60/gpt56_sol/grouped_10/summary.json")),
    ]
    for display, exact, path in grouped_sources:
        data = load_json(path)
        overall = data["overall"]
        rows.append(
            {
                "method": display,
                "exact_model_id": exact,
                "provider": "OpenAI/Codex subagent",
                "question_set": "old_60_clean",
                "protocol": "grouped_10_feedback",
                "category": "historical",
                "correct": overall["correct"],
                "total": overall["total"],
                "accuracy": overall["accuracy"],
                "accuracy_pct": pct(overall["accuracy"]),
                "source_path": rel(path),
            }
        )
    rows.append(
        {
            "method": "GPT-5.5",
            "exact_model_id": "gpt-5.5",
            "provider": "OpenAI/Codex subagent",
            "question_set": "old_60_clean",
            "protocol": "grouped_10_feedback",
            "category": "historical",
            "correct": 39,
            "total": 60,
            "accuracy": 39 / 60,
            "accuracy_pct": pct(39 / 60),
            "note": "Clean-60 filtered result from archive experiment summary; raw archive also contains a 65-question row.",
            "source_path": rel(Path("artifacts/_archive_default_unused/quiz_attempt_65/experiment_summary.md")),
        }
    )
    return rows


def meta_model_rows() -> list[dict[str, Any]]:
    path = Path("data/meta_model_studies/setting_to_loss_60q_id_v1/external_score.json")
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    primary = data.get("primary_method")
    for item in data["methods"]:
        method = item["method"]
        category = "primary" if method == primary else "historical"
        family_specific = len(set(item.get("method_by_experiment", {}).values())) > 1
        rows.append(
            {
                "method": method,
                "question_set": "old_60_clean_external",
                "protocol": "external_score",
                "category": category,
                "correct": item["total"]["num_correct"],
                "total": item["total"]["num_questions"],
                "accuracy": item["total"]["accuracy"],
                "accuracy_pct": pct(item["total"]["accuracy"]),
                "by_family": item["by_family"],
                "predictions_sha256": item["predictions_sha256"],
                "score_path": item["score_path"],
                "source_path": rel(path),
                "family_specific_training": family_specific,
                "method_by_experiment": item.get("method_by_experiment", {}),
            }
        )

    posthoc_path = Path("data/meta_model_studies/setting_to_loss_60q_id_v1/posthoc_external_scores.json")
    posthoc = load_json(posthoc_path)
    for item in posthoc["methods"]:
        rows.append(
            {
                "method": item["method"],
                "question_set": "old_60_clean_external",
                "protocol": "posthoc_external_score",
                "category": "post-hoc",
                "correct": item["total"]["num_correct"],
                "total": item["total"]["num_questions"],
                "accuracy": item["total"]["accuracy"],
                "accuracy_pct": pct(item["total"]["accuracy"]),
                "by_family": item["by_family"],
                "predictions_sha256": item["predictions_sha256"],
                "score_path": item["score_path"],
                "source_path": rel(posthoc_path),
            }
        )
    diagnostic_path = Path(
        "data/meta_model_studies/setting_to_loss_60q_id_v1/heuristics/heuristic_formula_v2/external_posthoc_diagnostics.json"
    )
    diagnostics = load_json(diagnostic_path)
    for item in diagnostics["methods"]:
        if item["method"] != "fixed_zero_shot":
            continue
        questions = item.get("questions", [])
        correct = sum(1 for row in questions if row.get("is_correct"))
        total = len(questions)
        rows.append(
            {
                "method": "fixed_zero_shot_formula",
                "question_set": "old_60_clean_external",
                "protocol": "posthoc_diagnostic_after_answer_key",
                "category": "post-hoc",
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else None,
                "accuracy_pct": pct(correct / total) if total else None,
                "by_family": item.get("by_family"),
                "posthoc_warning": "Computed after answer key was opened; not a pre-registered blind result.",
                "source_path": rel(diagnostic_path),
            }
        )
    return rows


def no_parameter_count_rows() -> list[dict[str, Any]]:
    path = Path(
        "data/meta_model_studies/setting_to_loss_60q_id_v1_no_params/external_score.json"
    )
    if not (ROOT / path).is_file():
        return []
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    for item in data["methods"]:
        rows.append(
            {
                "method": item["method"],
                "question_set": "old_60_clean_external",
                "protocol": "external_score_no_parameter_count",
                "category": "controlled_ablation",
                "include_parameter_count": False,
                "correct": item["total"]["num_correct"],
                "total": item["total"]["num_questions"],
                "accuracy": item["total"]["accuracy"],
                "accuracy_pct": pct(item["total"]["accuracy"]),
                "by_family": item["by_family"],
                "predictions_sha256": item["predictions_sha256"],
                "score_path": item["score_path"],
                "source_path": rel(path),
                "method_by_experiment": item.get("method_by_experiment", {}),
            }
        )
    return rows


def family_validation_rows(
    output_root: Path,
    *,
    condition: str,
) -> list[dict[str, Any]]:
    manifest_path = output_root / "study_manifest.json"
    if not (ROOT / manifest_path).is_file():
        return []
    manifest = load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for experiment in manifest["experiments"]:
        leaderboard_path = (
            output_root
            / "experiments"
            / experiment["experiment_id"]
            / "leaderboard.json"
        )
        leaderboard = load_json(leaderboard_path)
        method = experiment["cv_champion"]
        selected = next(
            item for item in leaderboard["methods"] if item["method"] == method
        )
        rows.append(
            {
                "condition": condition,
                "include_parameter_count": bool(
                    manifest.get("include_parameter_count", True)
                ),
                "experiment_id": experiment["experiment_id"],
                "family": experiment["family"],
                "selection_metric": experiment["selection_metric"],
                "method": method,
                "num_train": experiment["num_train"],
                "num_validation": experiment["num_validation"],
                "selection_protocol": experiment["selection_protocol"],
                "validation_all": selected["validation"]["all"],
                "validation_benchmark_eligible": selected["validation"].get(
                    "benchmark_eligible"
                ),
                "source_path": rel(leaderboard_path),
            }
        )
    return rows


def _wide_task_row(path: Path, condition: str) -> dict[str, Any]:
    data = load_json(path)
    champion = data["cv_champion"]
    selected = next(item for item in data["methods"] if item["method"] == champion)
    test = selected["test"]["all"]
    return {
        "condition": condition,
        "include_parameter_count": condition == "with_parameter_count",
        "task_id": data.get("task_id", path.parent.name),
        "family": data.get("protocol", {}).get("family"),
        "dataset": data.get("protocol", {}).get("dataset"),
        "method": champion,
        "inner_cv": data["inner_cv"],
        "n_test": test["n"],
        "raw": test.get("raw"),
        "log": test["log"],
        "ranking": test["ranking"],
        "per_environment": test.get("per_environment", {}),
        "within_environment": test["within_environment"],
        "source_path": rel(path),
    }


def wide_study_rows(output_root: Path, *, condition: str) -> dict[str, Any] | None:
    manifest_path = output_root / "id" / "manifest.json"
    if not (ROOT / manifest_path).is_file():
        return None
    manifest = load_json(manifest_path)
    environments = [
        _wide_task_row(path.relative_to(ROOT), condition)
        for path in sorted((ROOT / output_root / "id" / "environment").glob("*/leaderboard.json"))
    ]
    families = [
        _wide_task_row(path.relative_to(ROOT), condition)
        for path in sorted((ROOT / output_root / "id" / "family").glob("*/leaderboard.json"))
    ]
    environment_aggregate_path = output_root / "id" / "environment" / "aggregate.json"
    family_aggregate_path = output_root / "id" / "family" / "aggregate.json"
    return {
        "condition": condition,
        "include_parameter_count": bool(manifest["include_parameter_count"]),
        "phases": manifest.get("phases"),
        "snapshot_manifest_path": manifest.get("snapshot_manifest_path"),
        "snapshot_manifest_sha256": manifest.get("snapshot_manifest_sha256"),
        "n_environments": manifest["n_environments"],
        "n_seeds": manifest["n_seeds"],
        "methods": manifest["methods"],
        "per_environment": environments,
        "family_pooled": families,
        "environment_aggregate": load_json(environment_aggregate_path),
        "family_aggregate": load_json(family_aggregate_path),
        "source_paths": {
            "manifest": rel(manifest_path),
            "environment_aggregate": rel(environment_aggregate_path),
            "family_aggregate": rel(family_aggregate_path),
        },
    }


def evaluation_contract() -> dict[str, Any]:
    wide_audit_path = Path(
        "data/meta_model/setting_to_loss_wide_v2/audit_b2_partial.json"
    )
    wide = load_json(wide_audit_path) if (ROOT / wide_audit_path).is_file() else None
    b1_audit_path = Path(
        "data/meta_model/setting_to_loss_wide_v2/audit_b1_pilot.json"
    )
    b1 = load_json(b1_audit_path) if (ROOT / b1_audit_path).is_file() else None
    b2_observed = None
    if wide is not None:
        terminal = [
            item for item in wide["experiments"] if item.get("terminal_and_exported")
        ]
        b2_observed = {
            "terminal_and_exported_environments": len(terminal),
            "attempts": sum(item["export"]["attempts"].get("total", 0) for item in terminal),
            "selected": sum(item["export"]["selected"].get("total", 0) for item in terminal),
        }
    return {
        "primary_metric": {
            "name": "locked_setting_to_loss_evaluation",
            "target": "log(mean_loss) from executed generated-candidate GT",
            "selection": "minimum train-only inner-CV log RMSE",
            "reported": [
                "raw/log MAE, RMSE, R2 per environment",
                "Spearman, Kendall tau-b, pair concordance per environment",
                "all-triple and raw-gap>=0.05 three-choice accuracy",
                "raw absolute regret and log loss-ratio regret",
            ],
            "aggregation": "environment first, then dataset/family macro, then equal-family macro; raw CE/MSE is never pooled across datasets or families",
            "lower_is_better": ["MAE", "RMSE", "regret"],
            "higher_is_better": ["R2", "rank correlation", "three-choice accuracy"],
        },
        "setting_to_loss_protocol": {
            "target": "log(mean_loss) from stored results/summary.json",
            "selection": "minimum train-only 5-fold selection-CV log RMSE separately for each environment task and each family-pooled task",
            "locked_validation": [
                "raw/log MAE",
                "raw/log RMSE",
                "raw/log R2",
                "Spearman",
                "Kendall tau-b",
                "pairwise concordance",
                "exact three-choice accuracy over C(n_validation,3) within each environment",
                "gap-filtered three-choice accuracy",
                "mean/median regret",
            ],
            "cv_warning": "OOF metrics reuse folds after hyperparameter selection and are selection diagnostics, not nested-CV generalization estimates.",
        },
        "feature_encoding": {
            "training_unit": "one candidate setting",
            "target_free_inputs": [
                "optimizer type, log10 learning rate, optimizer details",
                "model type and numeric architecture conditions",
                "activation and LayerNorm counts/fractions/transitions",
                "loss id and loss-specific numeric scalars, log budget, log batch size",
                "explicit optimizer x log10(lr) interaction",
                "exact generated-model log(total parameters) when enabled",
            ],
            "categorical_encoding": "DictVectorizer one-hot encoding learned inside each training fold",
            "numeric_encoding": "constant columns removed inside each fold, then all retained columns standardized",
            "mixing": "numeric and one-hot columns share one dense matrix; nonlinear learners model interactions, while compact polynomial Ridge adds degree-2 interactions explicitly",
            "parameter_count_provenance": "count Model.parameters() from generated code used by GT and cross-check against the registry-built model; external questions rebuild from public model spec",
            "no_parameter_count_caveat": "removing the exact parameter-count column does not remove width/depth/d_model/d_ff, so architecture can still act as a size proxy",
        },
        "scope_status": {
            "old_60": "external transfer appendix only: 60/60 questions have stored GT and a complete answer key",
            "phase_a_setting_corpus": "historical auxiliary study only: 3 families x (900 train + 100 locked validation), plus predeclared reserve attempts",
            "wide_v2_b1": None
            if b1 is None
            else {
                "included_in_setting_metrics": True,
                "decision": b1["decision"],
                "expected": b1["expected"],
                "observed": b1["observed"],
                "source_path": rel(b1_audit_path),
            },
            "high_budget_confirmed_15": "supplemental only",
            "wide_v2_b2": None
            if wide is None
            else {
                "included_in_full_phase_score": False,
                "completed_environments_may_enter_frozen_snapshot": True,
                "decision": wide["decision"],
                "expected": wide["expected"],
                "observed": b2_observed,
                "reason": "the 21-environment phase is incomplete; only individually complete, validated exports named in the frozen snapshot are used, while unfinished environments are excluded",
                "source_path": rel(wide_audit_path),
            },
            "remote_audit": {
                "fetched_at": "2026-07-15 Asia/Shanghai",
                "ref": "origin/codex/complete-local-features@11ccaff",
                "finding": "remote adds GPT Eval routing/status documentation; it does not add a complete wide-v2 score artifact",
            },
        },
    }


def arithmetic_rows() -> list[dict[str, Any]]:
    path = Path("artifacts/order_parameter_analysis/arithmetic_rule_results.json")
    rows = []
    for item in load_json(path):
        rows.append(
            {
                "method": item["rule"],
                "question_set": "old_60_clean",
                "protocol": "arithmetic_rule",
                "category": "historical" if not item["rule"].startswith("fit_") else "post-hoc",
                "correct": item["correct"],
                "total": item["n"],
                "accuracy": item["accuracy"],
                "accuracy_pct": pct(item["accuracy"]),
                "by_family_accuracy": item.get("by_family"),
                "source_path": rel(path),
            }
        )
    return rows


def load_new_score(path: Path, protocol: str) -> dict[str, Any] | None:
    if not (ROOT / path).is_file():
        return None
    data = load_json(path)
    return {
        "method": "GPT-5.4",
        "exact_model_id": "gpt-5.4",
        "provider": "OpenAI/Codex subagent",
        "question_set": "high_budget_confirmed_15",
        "protocol": protocol,
        "category": "new-question evaluation",
        "correct": data["overall"]["correct"],
        "total": data["overall"]["total"],
        "accuracy": data["overall"]["accuracy"],
        "accuracy_pct": pct(data["overall"]["accuracy"]),
        "by_family": data.get("by_family"),
        "by_question_type": data.get("by_question_type"),
        "prediction_sha256": data.get("prediction_sha256"),
        "bundle_sha256": data.get("bundle_sha256"),
        "source_path": rel(path),
    }


def build_report_data(*, generated_at: str | None = None) -> dict[str, Any]:
    new_rows = [
        row
        for row in [
            load_new_score(Path("artifacts/high_budget_gpt54_eval/blind_fullset/score.json"), "full_set_blind"),
            load_new_score(Path("artifacts/high_budget_gpt54_eval/sequential/score.json"), "fixed_order_sequential_feedback"),
        ]
        if row is not None
    ]
    return {
        "schema_version": "architecture_iq.gpteval_report_data.v1",
        "generated_at": generated_at or utc_now_iso(),
        "old_60_llm": old_llm_rows(),
        "heuristic_formula_and_arithmetic": arithmetic_rows(),
        "meta_model_external": meta_model_rows(),
        "meta_model_no_parameter_count": no_parameter_count_rows(),
        "family_validation": [
            *family_validation_rows(
                Path("data/meta_model_studies/setting_to_loss_60q_id_v1"),
                condition="with_parameter_count",
            ),
            *family_validation_rows(
                Path("data/meta_model_studies/setting_to_loss_60q_id_v1_no_params"),
                condition="without_parameter_count",
            ),
        ],
        "wide_v2_b1": [
            row
            for row in [
                wide_study_rows(
                    Path("data/meta_model_studies/setting_to_loss_wide_v2_b1_with_params"),
                    condition="with_parameter_count",
                ),
                wide_study_rows(
                    Path("data/meta_model_studies/setting_to_loss_wide_v2_b1_no_params"),
                    condition="without_parameter_count",
                ),
            ]
            if row is not None
        ],
        "wide_v2_completed_snapshot": [
            row
            for row in [
                wide_study_rows(
                    Path("data/meta_model_studies/setting_to_loss_wide_v2_completed_with_params"),
                    condition="with_parameter_count",
                ),
                wide_study_rows(
                    Path("data/meta_model_studies/setting_to_loss_wide_v2_completed_no_params"),
                    condition="without_parameter_count",
                ),
            ]
            if row is not None
        ],
        "wide_v2_completed_analysis": (
            load_json(
                Path(
                    "artifacts/high_budget_gpt54_eval/wide_completed_setting_to_loss_analysis.json"
                )
            )
            if (
                ROOT
                / "artifacts/high_budget_gpt54_eval/wide_completed_setting_to_loss_analysis.json"
            ).is_file()
            else None
        ),
        "wide_v2_b1_analysis": (
            load_json(
                Path(
                    "artifacts/high_budget_gpt54_eval/wide_b1_setting_to_loss_analysis.json"
                )
            )
            if (
                ROOT
                / "artifacts/high_budget_gpt54_eval/wide_b1_setting_to_loss_analysis.json"
            ).is_file()
            else None
        ),
        "evaluation_contract": evaluation_contract(),
        "new_high_budget_gpt54": new_rows,
        "source_notes": {
            "setting_to_loss_primary_html": "docs/0715_setting_to_loss.html",
            "old60_behavioral_appendix_html": "docs/0710_gpteval.html",
            "html_generation": "0715 is rendered from the completed analysis JSON; 0710 remains the static old-60 behavioral appendix.",
            "new_question_manifest": "artifacts/high_budget_public_manifest.json",
            "new_question_release_manifest": "data/releases/high_budget_confirmed_v1/quiz_manifest.json",
            "new_question_bundle": "artifacts/high_budget_gpt54_eval/sanitized_bundle.json",
            "completed_gt_snapshot": "artifacts/wide_v2_completed_gt_snapshot.json",
            "api_transport_check": "artifacts/high_budget_gpt54_eval/api_transport_check.json",
        },
    }


def main() -> None:
    out = ROOT / "artifacts/high_budget_gpt54_eval/gpteval_report_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_report_data()
    if out.is_file():
        previous = json.loads(out.read_text(encoding="utf-8"))
        previous_without_timestamp = dict(previous)
        previous_without_timestamp.pop("generated_at", None)
        payload_without_timestamp = dict(payload)
        payload_without_timestamp.pop("generated_at", None)
        if previous_without_timestamp == payload_without_timestamp:
            payload["generated_at"] = previous.get("generated_at", payload["generated_at"])
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
