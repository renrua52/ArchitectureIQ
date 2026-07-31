#!/usr/bin/env python
"""Build the '3 families x 3 settings x 10 questions' blind eval bundle.

Selects 3 dataset instances ("settings") per family (bigram_lm,
multivariate_regression, univariate_regression), samples 10 already-generated
questions from each (seeded, deterministic), and assembles a sanitized
90-question blind prompt + answer key, mirroring the format used for the
original 60-question llm_baseline_eval bundle.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "artifacts" / "setting3x3_eval"

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
N_PER_SETTING = 10
SEED = 20260724


def load_candidate_spec(dataset_dir: Path, candidate_path_rel: str) -> dict:
    # candidate_path_rel looks like "datasets/{family}/{id}/candidates/set_.../c_xxx"
    # stored relative to data/
    p = DATA_ROOT / candidate_path_rel / "candidate_spec.json"
    return json.loads(p.read_text())


def build():
    rng = random.Random(SEED)
    sanitized = []
    answer_key = {}
    question_meta = {}  # question_id -> {family, dataset_id}
    n = 0

    for family, dataset_id in SETTINGS:
        dataset_dir = DATA_ROOT / "datasets" / family / dataset_id
        q_paths = sorted(glob.glob(str(dataset_dir / "questions" / "*" / "q_*" / "question.json")))
        chosen = rng.sample(q_paths, N_PER_SETTING) if len(q_paths) > N_PER_SETTING else list(q_paths)
        chosen = sorted(chosen)
        assert len(chosen) == N_PER_SETTING, f"{family}/{dataset_id} only has {len(chosen)} questions"

        for qp in chosen:
            q = json.loads(Path(qp).read_text())
            n += 1
            choices = []
            for ch in q["choices"]:
                spec = load_candidate_spec(dataset_dir, ch["candidate_path"])
                choices.append({
                    "letter": ch["letter"],
                    "candidate_id": ch["candidate_id"],
                    "model": spec["model"],
                    "optimizer": spec["optimizer"],
                    "loss": spec["loss"],
                    "budget": spec["budget"],
                })
            entry = {
                "n": n,
                "question_id": q["question_id"],
                "family": q["family"],
                "dataset_id": q["dataset_id"],
                "question_type": q["type"],
                "varying_axes": q["varying_axes"],
                "invariant_axes": q["invariant_axes"],
                "choices": choices,
            }
            sanitized.append(entry)
            answer_key[q["question_id"]] = q["correct_letter"]
            question_meta[q["question_id"]] = {"family": q["family"], "dataset_id": q["dataset_id"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "questions_sanitized.json").write_text(json.dumps(sanitized, indent=2))
    (OUT_DIR / "answer_key.json").write_text(json.dumps(answer_key, indent=2))
    (OUT_DIR / "question_meta.json").write_text(json.dumps(question_meta, indent=2))

    header = f"""# 3-family x 3-setting x 10-question blind baseline prompt

You are an independent blind-answer agent for ArchitectureIQ.

STRICT PROTOCOL:

- You will receive a full {n}-question sanitized set at once (3 dataset families: bigram_lm, multivariate_regression, univariate_regression; 3 dataset instances/"settings" per family; 10 questions per setting).
- Answer only from the visible sanitized questions and qualitative reasoning.
- Do not read answer keys, feedback files, scoring files, result summaries, curves, previous attempts, repository files, or any hidden ground-truth artifacts.
- Do not run shell commands, Python, Node, jq, scripts, training, local simulations, approximate experiments, or data reconstruction.
- You may compare across the visible questions, repeated candidates, model families, optimizers, learning rates, budgets, and architecture patterns, including across settings within the same family.
- Return strict JSON only with keys: `agent`, `model`, `reasoning_effort`, `source_used`, `forbidden_files_viewed`, and `predictions`.
- `predictions` must be an array of exactly {n} records.
- Each prediction must contain: `n`, `question_id`, `predicted_letter`, `predicted_candidate_id`, `confidence`, and `reason`.
- `predicted_candidate_id` must match the selected letter in that question.

Sanitized {n}-question JSON:

"""
    prompt_text = header + json.dumps(sanitized, indent=2) + "\n"
    (OUT_DIR / "prompt.txt").write_text(prompt_text)

    print(f"Built {n} questions across {len(SETTINGS)} settings -> {OUT_DIR}")
    by_fam = {}
    for e in sanitized:
        by_fam[e["family"]] = by_fam.get(e["family"], 0) + 1
    print("Per family:", by_fam)


if __name__ == "__main__":
    build()
