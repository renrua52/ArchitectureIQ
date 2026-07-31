#!/usr/bin/env python
"""Build a fair 'few-shot' benchmark bundle for the 3x3 setting.

For each of the 9 (family, dataset_id) settings, this:
1. Keeps the SAME 10 held-out test questions used in the existing
   isolated/sequential experiments (artifacts/setting3x3_eval/questions_sanitized.json).
2. Loads a small number of few-shot DEMO questions drawn from a FRESH
   candidate set (generated with a different seed, `set_*_98c129`/`*_424242`
   style dirs) so their candidate_ids are guaranteed disjoint from the test
   questions' candidate_ids (verified programmatically, not assumed).
3. For each demo question, looks up GT mean metric per choice from
   results/summary.json (never recomputed) to build a short, honest
   metric-grounded rationale.
4. Emits a per-setting bundle: few-shot preamble (worked examples with
   correct answer + GT-based rationale) + the 10 sanitized test questions,
   for a SINGLE batch call per setting (no per-question feedback loop,
   isolating "in-context learning from disjoint demos" from "sequential
   answer-memorization via candidate reuse").
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "artifacts" / "setting3x3_eval"
OUT_DIR = EVAL_DIR / "fewshot"

SETTINGS = [
    ("bigram_lm", "bg_0021c1"),
    ("bigram_lm", "bg_0fff4b"),
    ("bigram_lm", "bg_a58df7"),
    ("multivariate_regression", "mvar_c59a30"),
    ("multivariate_regression", "mvar_9faf9d"),
    ("multivariate_regression", "mvar_978f4c"),
    ("univariate_regression", "sym_62678b"),
    ("univariate_regression", "sym_c804cc"),
    ("univariate_regression", "sym_411dbf"),
]

N_DEMO = 3
FRESH_SEED_TAG = "424242"  # the --seed used when generating the disjoint candidate/question pool


def load_candidate_spec(candidate_path_rel: str) -> dict:
    return json.loads((DATA_ROOT / candidate_path_rel / "candidate_spec.json").read_text())


def load_candidate_summary(candidate_path_rel: str) -> dict | None:
    p = DATA_ROOT / candidate_path_rel / "results" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_setting(family: str, dataset_id: str, test_questions: list[dict]) -> dict:
    dataset_dir = DATA_ROOT / "datasets" / family / dataset_id
    test_cand_ids = {c["candidate_id"] for q in test_questions for c in q["choices"]}

    # Find question runs generated from the fresh (disjoint) candidate set.
    # A question run qualifies if ALL its choices' candidate_ids are disjoint
    # from test_cand_ids AND it wasn't part of the original test selection.
    q_paths = sorted(glob.glob(str(dataset_dir / "questions" / "*" / "q_*" / "question.json")))
    test_qids = {q["question_id"] for q in test_questions}

    demo_candidates = []
    for qp in q_paths:
        dq = json.loads(Path(qp).read_text())
        if dq["question_id"] in test_qids:
            continue
        cset = {c["candidate_id"] for c in dq["choices"]}
        if cset.isdisjoint(test_cand_ids):
            demo_candidates.append(dq)

    if len(demo_candidates) < N_DEMO:
        raise RuntimeError(
            f"{family}/{dataset_id}: only found {len(demo_candidates)} disjoint-candidate demo "
            f"questions (need {N_DEMO}). Generate more fresh candidates/questions first."
        )

    demo_candidates = demo_candidates[:N_DEMO]

    demos = []
    for dq in demo_candidates:
        choices_out = []
        metric_name = dq["evaluation"]["selection_metric"]
        for ch in dq["choices"]:
            spec = load_candidate_spec(ch["candidate_path"])
            summary = load_candidate_summary(ch["candidate_path"])
            mean_metric = summary.get(f"mean_{metric_name}") if summary else None
            choices_out.append({
                "letter": ch["letter"],
                "candidate_id": ch["candidate_id"],
                "model": spec["model"],
                "optimizer": spec["optimizer"],
                "loss": spec["loss"],
                "budget": spec["budget"],
                "gt_mean_metric": {metric_name: mean_metric},
            })
        demos.append({
            "question_id": dq["question_id"],
            "question_type": dq["type"],
            "varying_axes": dq["varying_axes"],
            "invariant_axes": dq["invariant_axes"],
            "selection_metric": metric_name,
            "choices": choices_out,
            "correct_letter": dq["correct_letter"],
        })

    # sanity: verify true disjointness at candidate-id granularity
    demo_cand_ids = {c["candidate_id"] for d in demos for c in d["choices"]}
    overlap = demo_cand_ids & test_cand_ids
    assert not overlap, f"{family}/{dataset_id}: demo/test candidate overlap detected: {overlap}"

    return {
        "family": family,
        "dataset_id": dataset_id,
        "demos": demos,
        "test_questions": test_questions,
        "n_demo_candidates": len(demo_cand_ids),
        "n_test_candidates": len(test_cand_ids),
    }


def render_prompt(bundle: dict) -> str:
    family = bundle["family"]
    dataset_id = bundle["dataset_id"]
    lines = []
    lines.append(f"# Few-shot benchmark: {family} / {dataset_id}")
    lines.append("")
    lines.append(
        "You are a blind-answer benchmark agent. Below are worked EXAMPLE questions "
        "(from a DIFFERENT, disjoint set of candidates than the test questions) showing "
        "the correct answer and the ground-truth metric that decided it. These examples "
        "teach you the reasoning style and evaluation criteria for this dataset family. "
        "The candidates in these examples will NEVER reappear in the test questions below, "
        "so you cannot simply recall a memorized answer -- you must generalize the pattern."
    )
    lines.append("")
    lines.append("## Worked examples (with ground truth revealed)")
    for i, d in enumerate(bundle["demos"], 1):
        lines.append(f"\n### Example {i} ({d['question_type']}, varies: {d['varying_axes']})")
        lines.append(f"Selection metric: {d['selection_metric']} (lower is better)")
        lines.append(json.dumps({
            "choices": [
                {k: v for k, v in c.items() if k != "gt_mean_metric"}
                for c in d["choices"]
            ]
        }, indent=2))
        lines.append("Ground truth mean metrics per choice:")
        for c in d["choices"]:
            lines.append(f"  {c['letter']}: {c['gt_mean_metric']}")
        lines.append(f"Correct answer: {d['correct_letter']}")
    lines.append("\n## Now answer these 10 TEST questions (no ground truth shown, choices are from a disjoint candidate pool)")
    lines.append(
        "Return strict JSON only: an array of exactly 10 objects with keys "
        "`question_id`, `predicted_letter`, `reason`."
    )
    lines.append("\nTest questions:")
    lines.append(json.dumps(bundle["test_questions"], indent=2))
    return "\n".join(lines)


def main():
    sanitized = json.loads((EVAL_DIR / "questions_sanitized.json").read_text())
    by_setting: dict[tuple, list] = {}
    for q in sanitized:
        by_setting.setdefault((q["family"], q["dataset_id"]), []).append(q)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family, dataset_id in SETTINGS:
        key = (family, dataset_id)
        test_qs = by_setting[key]
        bundle = build_setting(family, dataset_id, test_qs)
        prompt = render_prompt(bundle)

        setting_dir = OUT_DIR / f"{family}__{dataset_id}"
        setting_dir.mkdir(parents=True, exist_ok=True)
        (setting_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
        (setting_dir / "prompt.txt").write_text(prompt)

        print(
            f"{family}/{dataset_id}: {len(bundle['demos'])} demos "
            f"({bundle['n_demo_candidates']} cands), {len(test_qs)} test qs "
            f"({bundle['n_test_candidates']} cands), overlap=0 (verified)"
        )
        manifest.append({"family": family, "dataset_id": dataset_id,
                          "n_demo_candidates": bundle["n_demo_candidates"],
                          "n_test_candidates": bundle["n_test_candidates"]})

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nBuilt few-shot bundles for {len(SETTINGS)} settings -> {OUT_DIR}")


if __name__ == "__main__":
    main()
