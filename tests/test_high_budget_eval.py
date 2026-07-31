from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL = ROOT / "tools" / "llm_eval"
sys.path.insert(0, str(LLM_EVAL))

import high_budget_eval as hbe  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    qdir = root / "data/datasets/univariate_regression/sym_test/questions/run_1/q_a"
    c_a = root / "data/datasets/univariate_regression/sym_test/candidates/set_1/c_a"
    c_b = root / "data/datasets/univariate_regression/sym_test/candidates/set_1/c_b"
    write_json(c_a / "results/summary.json", {"mean_test_mse": 0.4})
    write_json(c_b / "results/summary.json", {"mean_test_mse": 0.2})
    (qdir / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
    (qdir / "prompt.txt").write_text("Pick the best choice.\nA or B?", encoding="utf-8")
    write_json(
        qdir / "question.json",
        {
            "question_id": "q_a",
            "family": "univariate_regression",
            "dataset_id": "sym_test",
            "type": "architecture_only",
            "evaluation": {"selection_metric": "test_mse"},
            "budget": {"total_samples_seen": 128},
            "correct_letter": "B",
            "significance": {"gap": 0.2},
            "choices": [
                {"letter": "A", "candidate_id": "c_a", "candidate_path": str(c_a)},
                {"letter": "B", "candidate_id": "c_b", "candidate_path": str(c_b)},
            ],
        },
    )
    public = root / "artifacts/high_budget_public_manifest.json"
    private = root / "artifacts/high_budget_private_answer_key.json"
    release = root / "data/releases/high_budget_confirmed_v1/quiz_manifest.json"
    write_json(
        public,
        {
            "schema_version": "architecture_iq.shortlist.v1",
            "question_count": 1,
            "questions": [
                {
                    "question_id": "q_a",
                    "family": "univariate_regression",
                    "dataset_id": "sym_test",
                    "type": "architecture_only",
                    "budget": 128,
                    "prompt_path": str((qdir / "prompt.txt").relative_to(root)),
                }
            ],
        },
    )
    write_json(
        private,
        {
            "schema_version": "architecture_iq.private_answer_key.v1",
            "answers": [
                {
                    "question_id": "q_a",
                    "correct_letter": "B",
                    "candidate_id": "c_b",
                    "gap": 0.2,
                    "win_rate": 1.0,
                }
            ],
        },
    )
    write_json(
        release,
        {
            "schema_version": "architecture_iq.release.v1",
            "questions": [{"question_id": "q_a"}],
            "artifacts": [],
        },
    )
    return public, private, release


def test_build_bundle_excludes_answer_keys_and_gt_paths(tmp_path: Path) -> None:
    public, private, release = make_fixture(tmp_path)
    bundle, feedback = hbe.build_bundle(
        repo_root=tmp_path,
        public_manifest_path=public,
        private_answer_key_path=private,
        release_manifest_path=release,
    )
    assert bundle["question_count"] == 1
    assert bundle["canonical_order"] == ["q_a"]
    assert bundle["questions"][0]["choices"] == [
        {"letter": "A", "candidate_id": "c_a"},
        {"letter": "B", "candidate_id": "c_b"},
    ]
    assert "correct_letter" not in json.dumps(bundle)
    assert "results/summary.json" not in json.dumps(bundle)
    assert hbe.scan_for_leakage(bundle) == []
    assert feedback[0]["correct_letter"] == "B"
    assert feedback[0]["choice_mean_metrics"] == {"A": 0.4, "B": 0.2}


def test_leakage_scan_rejects_forbidden_fields() -> None:
    findings = hbe.scan_for_leakage({"question": {"correct_letter": "A"}})
    assert findings
    assert findings[0]["reason"] == "forbidden key"


def test_score_predictions_validates_hash_and_counts_missing_duplicate(tmp_path: Path) -> None:
    public, private, release = make_fixture(tmp_path)
    bundle, _feedback = hbe.build_bundle(
        repo_root=tmp_path,
        public_manifest_path=public,
        private_answer_key_path=private,
        release_manifest_path=release,
    )
    bundle_path = tmp_path / "bundle.json"
    write_json(bundle_path, bundle)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "question_id": "q_a",
                "predicted_letter": "B",
                "predicted_candidate_id": "c_b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score = hbe.score_predictions(
        prediction_path=predictions,
        bundle_path=bundle_path,
        answer_key_path=private,
        expected_bundle_sha256=bundle["bundle_sha256"],
    )
    assert score["overall"]["correct"] == 1
    assert score["overall"]["total"] == 1
    assert score["missing_question_ids"] == []
    assert score["duplicate_question_ids"] == []
    assert score["parse_failures"] == []


def test_score_predictions_can_parse_answer_tag(tmp_path: Path) -> None:
    public, private, release = make_fixture(tmp_path)
    bundle, _feedback = hbe.build_bundle(
        repo_root=tmp_path,
        public_manifest_path=public,
        private_answer_key_path=private,
        release_manifest_path=release,
    )
    bundle_path = tmp_path / "bundle.json"
    write_json(bundle_path, bundle)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"question_id": "q_a", "model_response": "Reasoning\n<answer>B</answer>"}) + "\n",
        encoding="utf-8",
    )
    score = hbe.score_predictions(
        prediction_path=predictions,
        bundle_path=bundle_path,
        answer_key_path=private,
    )
    assert score["overall"]["correct"] == 1
    assert score["rows"][0]["predicted_letter"] == "B"


def test_high_budget_artifacts_match_report_claims() -> None:
    artifact_dir = ROOT / "artifacts" / "high_budget_gpt54_eval"
    bundle = json.loads((artifact_dir / "sanitized_bundle.json").read_text(encoding="utf-8"))
    questions_only = json.loads((artifact_dir / "questions_sanitized.json").read_text(encoding="utf-8"))
    leakage = json.loads((artifact_dir / "leakage_scan.json").read_text(encoding="utf-8"))
    blind_score = json.loads((artifact_dir / "blind_fullset" / "score.json").read_text(encoding="utf-8"))
    sequential_score = json.loads((artifact_dir / "sequential" / "score.json").read_text(encoding="utf-8"))
    report_data_path = artifact_dir / "gpteval_report_data.json"
    report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
    primary_html = (ROOT / "docs" / "0715_setting_to_loss.html").read_text(
        encoding="utf-8"
    )
    old60_html = (ROOT / "docs" / "0710_gpteval.html").read_text(encoding="utf-8")

    assert bundle["question_count"] == 15
    assert len(bundle["canonical_order"]) == 15
    assert Counter(q["family"] for q in bundle["questions"]) == {
        "multivariate_regression": 11,
        "univariate_regression": 2,
        "bigram_lm": 2,
    }
    assert Counter(q["question_type"] for q in bundle["questions"]) == {
        "architecture_only": 10,
        "optimizer_only": 3,
        "mixed": 2,
    }
    assert leakage["passed"] is True
    assert leakage["findings"] == []
    assert hbe.scan_for_leakage(bundle) == []
    assert hbe.scan_for_leakage(questions_only) == []
    assert "correct_letter" not in json.dumps(questions_only)
    assert "results/summary.json" not in json.dumps(questions_only)
    assert "private_answer_key" not in json.dumps(bundle)

    assert blind_score["overall"]["correct"] == 3
    assert blind_score["overall"]["total"] == 15
    assert sequential_score["overall"]["correct"] == 7
    assert sequential_score["overall"]["total"] == 15
    for score in (blind_score, sequential_score):
        assert score["missing_question_ids"] == []
        assert score["duplicate_question_ids"] == []
        assert score["unexpected_question_ids"] == []
        assert score["parse_failures"] == []
        assert score["candidate_id_mismatches"] == []
        assert score["bundle_sha256"] == bundle["bundle_sha256"]

    report_hash = hashlib.sha256(report_data_path.read_bytes()).hexdigest()
    assert report_hash in old60_html
    assert "旧 60 题 GPT 行为附录" in old60_html
    assert "0715_setting_to_loss.html" in old60_html
    assert "主评测对象仍是老 60 题" not in old60_html

    analysis_path = artifact_dir / "wide_completed_setting_to_loss_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis_hash = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    snapshot_path = ROOT / "artifacts" / "wide_v2_completed_gt_snapshot.json"
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    assert analysis_hash in primary_html
    assert snapshot_hash in primary_html
    assert "5,671 executed settings" in primary_html
    assert "各 family CV champion · 含参数量" in primary_html
    assert "75.0%" in primary_html
    assert "74.5%" in primary_html
    assert "41.0%" in primary_html
    assert "Capacity shortcut 没有迁移" in primary_html
    assert "13 个尚无完整 export" in primary_html
    assert "old-60 仅外部附录" in primary_html

    new_rows = report_data["new_high_budget_gpt54"]
    assert {row["protocol"] for row in new_rows} == {
        "full_set_blind",
        "fixed_order_sequential_feedback",
    }
    assert {row["exact_model_id"] for row in new_rows} == {"gpt-5.4"}
    assert all(row["question_set"] == "high_budget_confirmed_15" for row in new_rows)
    fixed_zero_rows = [
        row for row in report_data["meta_model_external"] if row["method"] == "fixed_zero_shot_formula"
    ]
    assert fixed_zero_rows
    assert fixed_zero_rows[0]["category"] == "post-hoc"
    assert "posthoc_warning" in fixed_zero_rows[0]

    no_param_rows = report_data["meta_model_no_parameter_count"]
    assert len(no_param_rows) == 18
    assert all(row["include_parameter_count"] is False for row in no_param_rows)
    assert next(row for row in no_param_rows if row["method"] == "cv_champion")[
        "correct"
    ] == 51
    assert next(row for row in no_param_rows if row["method"] == "xgboost")[
        "correct"
    ] == 53

    wide_rows = report_data["wide_v2_completed_snapshot"]
    assert {row["condition"] for row in wide_rows} == {
        "with_parameter_count",
        "without_parameter_count",
    }
    assert all(row["n_environments"] == 17 for row in wide_rows)
    assert all(len(row["per_environment"]) == 17 for row in wide_rows)
    assert all(len(row["family_pooled"]) == 3 for row in wide_rows)
    with_params = next(
        row for row in wide_rows if row["condition"] == "with_parameter_count"
    )
    without = next(
        row for row in wide_rows if row["condition"] == "without_parameter_count"
    )
    assert len(with_params["methods"]) == 22
    assert len(without["methods"]) == 18
    assert without["include_parameter_count"] is False
    assert not {"max_params_heuristic", "params_ols", "params_ridge"}.intersection(
        without["methods"]
    )

    assert analysis["corpus"]["n_environments"] == 17
    assert analysis["corpus"]["n_selected_settings"] == 5671
    assert analysis["corpus"]["n_train"] == 5100
    assert analysis["corpus"]["n_validation"] == 571
    assert analysis["status"]["excluded_environment_count"] == 13
    for condition in ("with_parameter_count", "without_parameter_count"):
        methods = analysis["family_balanced_methods"][condition]
        assert methods[0]["selection_cv_log_rmse"] == min(
            row["selection_cv_log_rmse"] for row in methods
        )
        assert methods[0]["method"] == "xgboost"
    with_methods = {
        row["method"]: row
        for row in analysis["family_balanced_methods"]["with_parameter_count"]
    }
    without_methods = {
        row["method"]: row
        for row in analysis["family_balanced_methods"]["without_parameter_count"]
    }
    assert with_methods["xgboost"]["family_macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.7497325304755333)
    assert without_methods["xgboost"]["family_macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.7454280111360283)
    assert with_methods["max_params_heuristic"]["family_macro"][
        "three_choice_accuracy"
    ] == pytest.approx(0.4100700733617005)
    b2 = report_data["evaluation_contract"]["scope_status"]["wide_v2_b2"]
    assert b2["included_in_full_phase_score"] is False
    assert b2["completed_environments_may_enter_frozen_snapshot"] is True
    assert b2["decision"] == "INCOMPLETE"
    assert b2["observed"]["terminal_and_exported_environments"] == 7


def test_high_budget_run_specs_are_versioned_and_gpt54_only() -> None:
    artifact_dir = ROOT / "artifacts" / "high_budget_gpt54_eval"
    bundle = json.loads((artifact_dir / "sanitized_bundle.json").read_text(encoding="utf-8"))
    for rel_path, condition_id in (
        ("blind_fullset/run_spec.json", "gpt54_new_questions_blind_fullset_v1"),
        ("sequential/run_spec.json", "gpt54_new_questions_fixed_order_sequential_v1"),
    ):
        spec = json.loads((artifact_dir / rel_path).read_text(encoding="utf-8"))
        assert spec["condition_id"] == condition_id
        assert spec["exact_model_id"] == "gpt-5.4"
        assert spec["model_display_name"] == "GPT-5.4"
        assert spec["question_manifest_path"] == "artifacts/high_budget_public_manifest.json"
        assert spec["release_manifest_path"] == "data/releases/high_budget_confirmed_v1/quiz_manifest.json"
        assert spec["question_count"] == 15
        assert spec["canonical_order"] == bundle["canonical_order"]
        assert spec["bundle_sha256"] == bundle["bundle_sha256"]
        assert spec["prompt_template_sha256"]
        assert spec["api_transport"]["external_openai_api_completed_requests"] == 0
        assert spec["api_transport"]["usage_tokens_available"] is False
        assert spec["invalid_or_missing_scoring"] == "incorrect"
        assert spec["git"]["commit"]
        if condition_id == "gpt54_new_questions_blind_fullset_v1":
            assert (
                spec["model_input_bundle_sha256"]
                == "a4d5e445473ce7bb01b9d1af45112c5acd15c9ccbc6ebd178d1202317878aec7"
            )
