#!/usr/bin/env python3
"""Extract binary-classification questions from the v1_llm bundle into a single JSON.

Output: binary_questions.json with per-question: choices (with full candidate
summary fields incl. momentum, KAN grid, layer_norm), prompt.txt content, GT
summary, llm answers, significance.
"""
import json, os, sys, hashlib
from pathlib import Path
from collections import defaultdict

BUNDLE = Path(os.environ.get("BUNDLE", os.environ.get("V1_BUNDLE", "/tmp/v1bundle")))
HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
OUT_DIR = WORKTREE / "data" / "v1_review"   # gitignored runtime data
OUT = OUT_DIR / "binary_questions.json"
ANSWERS_DIR = OUT_DIR / "answers"
CURVES_DIR = OUT_DIR / "curves"


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def candidate_summary(spec):
    """Full-field summary from candidate_spec.json."""
    m = spec.get("model", {})
    opt = spec.get("optimizer", {})
    loss = spec.get("loss", {})
    budget = spec.get("budget", {})
    ex = spec.get("execution", {})
    s = {
        "model_type": m.get("type"),
        "depth": m.get("depth"),
        "width": m.get("width"),
        "residual": m.get("residual"),
        "layer_norm": m.get("layer_norm"),
        "activations": m.get("activations"),
        # KAN fields
        "grid_size": m.get("grid_size"),
        "spline_order": m.get("spline_order"),
        "grid_range": m.get("grid_range"),
        "base_activation": m.get("base_activation"),
        # optimizer (incl momentum)
        "optimizer": opt.get("type"),
        "lr": opt.get("lr"),
        "momentum": opt.get("momentum"),
        "weight_decay": opt.get("weight_decay"),
        "loss": loss.get("loss_id"),
        "batch_size": budget.get("batch_size"),
        "training_steps": budget.get("training_steps"),
        "total_samples_seen": budget.get("total_samples_seen"),
        "device": ex.get("device"),
        "params": None,
    }
    return s


def count_params(spec):
    """Best-effort param count from model spec."""
    m = spec.get("model", {})
    t = m.get("type")
    depth = m.get("depth", 1)
    width = m.get("width", 0)
    in_dim = m.get("input_dim", 2)
    n_cls = m.get("num_classes", 2)
    if t == "mlp":
        dims = [in_dim] + [width] * depth + [n_cls]
        p = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
        return p
    if t == "kan":
        dims = [in_dim] + [width] * depth + [n_cls]
        grid = m.get("grid_size", 5)
        order = m.get("spline_order", 3)
        # KAN layer params: in * out * (grid + order) + in * out (residual)
        p = sum(dims[i] * dims[i + 1] * (grid + order + 1) for i in range(len(dims) - 1))
        return p
    return None


def extract():
    qdir = BUNDLE / "benchmarks" / "v1_llm" / "questions"
    questions = []
    for qid in sorted(os.listdir(qdir)):
        qpath = qdir / qid / "question.json"
        if not qpath.exists():
            continue
        q = read_json(qpath)
        if not q:
            continue
        # only synthetic_tabular_classification
        cp0 = q["choices"][0]["candidate_path"]
        if "synthetic_tabular_classification" not in cp0:
            continue
        # load prompt
        prompt_p = qdir / qid / "prompt.txt"
        prompt = prompt_p.read_text() if prompt_p.exists() else ""

        # load dataset spec for rule_family
        ds_id = [p for p in cp0.split("/") if p.startswith("stabcls_")][0]
        ds_spec = read_json(BUNDLE / "data" / "datasets" / "synthetic_tabular_classification" / ds_id / "dataset_spec.json")
        rule = (ds_spec or {}).get("params", {}).get("rule_family", "?")
        params = (ds_spec or {}).get("params", {})

        choices = []
        for ch in q["choices"]:
            cp = ch["candidate_path"]
            cand_dir = BUNDLE / "data" / cp
            spec = read_json(cand_dir / "candidate_spec.json") or {}
            summ = read_json(cand_dir / "results" / "summary.json") or {}
            s = candidate_summary(spec)
            s["letter"] = ch["letter"]
            s["candidate_id"] = cand_dir.name
            s["candidate_path"] = cp
            s["params"] = count_params(spec)
            s["mean_test_ce"] = summ.get("mean_test_ce")
            s["std_test_ce"] = summ.get("std_test_ce")
            s["n_seeds"] = summ.get("n_seeds")
            choices.append(s)

        correct = q.get("correct_letter")
        winner = next((c for c in choices if c["letter"] == correct), None)
        runner = min((c for c in choices if c["letter"] != correct), key=lambda c: c.get("mean_test_ce") or 9) if len(choices) > 1 else None
        gap = None
        if winner and runner and winner.get("mean_test_ce") is not None and runner.get("mean_test_ce") is not None:
            gap = runner["mean_test_ce"] - winner["mean_test_ce"]
        param_ratio = None
        params_list = [c["params"] for c in choices if c.get("params")]
        if params_list:
            param_ratio = max(params_list) / min(params_list) if min(params_list) > 0 else None

        sig = q.get("significance", {}) or {}
        item = {
            "question_id": qid,
            "source": "v1_llm",
            "family": "synthetic_tabular_classification",
            "dataset_id": ds_id,
            "rule_family": rule,
            "bucket": rule,  # alias for viewer compatibility
            "type": q.get("type"),
            "correct_letter": correct,
            "gap": round(gap, 6) if gap is not None else None,
            "gap_cap": round(sig.get("gap", gap or 0), 6) if gap is not None else None,
            "win_rate": sig.get("win_rate"),
            "param_ratio": round(param_ratio, 3) if param_ratio else None,
            "metric": "test_ce",
            "budget_tier": q.get("budget", {}).get("total_samples_seen"),
            "budget": q.get("budget", {}),
            "choices": choices,
            "prompt": prompt,
            "selection_metric": (ds_spec or {}).get("selection_metric", "test_ce"),
            "significance": sig,
            "llm_difficulty": None,
            "llm_consensus_acc": None,
            "llm_n_correct": None,
            "llm_n_models": None,
            "llm_frontier_miss": None,
            "answers": [],
        }
        # save answers + curves per question
        # (LLM answers are in benchmarks/v1_llm/answers if present)
        questions.append(item)

    # load LLM answers from llm_runs/{model}/results/{qid}.json
    runs_dir = BUNDLE / "benchmarks" / "v1_llm" / "llm_runs"
    if runs_dir.exists():
        models = sorted(os.listdir(runs_dir))
        for q in questions:
            qid = q["question_id"]
            answers = []
            for model in models:
                rp = runs_dir / model / "results" / f"{qid}.json"
                if not rp.exists():
                    continue
                rd = read_json(rp)
                if not rd:
                    continue
                letter = rd.get("parsed_letter")
                gt = rd.get("ground_truth_letter", q["correct_letter"])
                is_c = rd.get("correct", letter == gt)
                answers.append({
                    "model": model,
                    "letter": letter,
                    "correct": bool(is_c),
                    "ground_truth": gt,
                })
            q["answers"] = answers
            q["llm_n_models"] = len(answers)
            q["llm_n_correct"] = sum(1 for a in answers if a["correct"])
            q["llm_consensus_acc"] = round(q["llm_n_correct"] / len(answers), 3) if answers else None
            q["llm_frontier_miss"] = len(answers) > 0 and q["llm_n_correct"] == 0

    # difficulty heuristic from gap + consensus
    for q in questions:
        g = q["gap"] or 0
        c = q["llm_consensus_acc"] or 0.5
        if g > 0.3 and c > 0.7:
            q["llm_difficulty"] = "easy"
        elif g > 0.15:
            q["llm_difficulty"] = "medium"
        elif g > 0.05:
            q["llm_difficulty"] = "hard"
        else:
            q["llm_difficulty"] = "very_hard"

    OUT.write_text(json.dumps({"schema": "v1_review_binary_v1", "questions": questions}, ensure_ascii=False))
    print(f"wrote {OUT}: {len(questions)} questions", flush=True)

    # write per-question answers + curves stubs
    ANSWERS_DIR.mkdir(exist_ok=True)
    CURVES_DIR.mkdir(exist_ok=True)
    for q in questions:
        (ANSWERS_DIR / f"{q['question_id']}.json").write_text(json.dumps({"answers": q["answers"]}, ensure_ascii=False))
        # curves: read from bundle
        curves = {"curves": []}
        for ch in q["choices"]:
            cp = BUNDLE / "data" / q["choices"][0]["candidate_path"]
            # need to find the actual candidate path — stored in question.json originally
        # curves extraction is heavy; let viewer fetch from train API on demand instead
    print("done.", flush=True)


if __name__ == "__main__":
    extract()
