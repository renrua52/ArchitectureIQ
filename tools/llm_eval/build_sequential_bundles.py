#!/usr/bin/env python
"""Build per-setting questions.json/feedback.json bundles for the
sequential_feedback_session.py tool, reusing the same 9 settings x 10
questions selected in setting3x3_eval/questions_sanitized.json.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
BUNDLE_DIR = REPO_ROOT / "artifacts" / "setting3x3_eval"
SEQ_DIR = BUNDLE_DIR / "sequential"


def load_candidate_spec(candidate_path_rel: str) -> dict:
    return json.loads((DATA_ROOT / candidate_path_rel / "candidate_spec.json").read_text())


def load_summary(candidate_path_rel: str) -> dict:
    return json.loads((DATA_ROOT / candidate_path_rel / "results" / "summary.json").read_text())


def main():
    sanitized = json.loads((BUNDLE_DIR / "questions_sanitized.json").read_text())

    # Need original question.json (with correct_letter + candidate_path) to build feedback.
    # Re-derive by scanning the same question dirs referenced when the bundle was built.
    groups: dict[str, list[dict]] = {}
    for q in sanitized:
        key = f"{q['family']}/{q['dataset_id']}"
        groups.setdefault(key, []).append(q)

    for setting_key, qs in sorted(groups.items()):
        family, dataset_id = setting_key.split("/")
        dataset_dir = DATA_ROOT / "datasets" / family / dataset_id
        safe_key = setting_key.replace("/", "__")
        out_dir = SEQ_DIR / safe_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find full question.json for each question_id (has correct_letter + candidate_path).
        q_ids = {q["question_id"] for q in qs}
        full_by_id = {}
        for qp in dataset_dir.glob("questions/*/q_*/question.json"):
            full = json.loads(qp.read_text())
            if full["question_id"] in q_ids:
                full_by_id[full["question_id"]] = full

        questions_out = []
        feedback_out = []
        for q in qs:
            full = full_by_id[q["question_id"]]
            metric_name = None
            choice_mean_metrics = {}
            for ch in full["choices"]:
                summary = load_summary(ch["candidate_path"])
                metric_name = metric_name or summary["selection_metric"]
                mean_key = f"mean_{summary['selection_metric']}"
                choice_mean_metrics[ch["letter"]] = summary.get(mean_key)

            questions_out.append(q)
            feedback_out.append({
                "question_id": q["question_id"],
                "correct_letter": full["correct_letter"],
                "metric": metric_name,
                "choice_mean_metrics": choice_mean_metrics,
            })

        (out_dir / "questions.json").write_text(json.dumps(questions_out, indent=2))
        (out_dir / "feedback.json").write_text(json.dumps(feedback_out, indent=2))
        print(f"{setting_key}: {len(questions_out)} questions -> {out_dir}")


if __name__ == "__main__":
    main()
