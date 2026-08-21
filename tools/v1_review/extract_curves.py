#!/usr/bin/env python3
"""Extract per-question learning curves from the bundle into curves/{qid}.json.

Each output file: {"curves": [{letter, mean, std, samples}], "metric": "test_ce"}
"""
import json, os, sys
from pathlib import Path
import numpy as np

BUNDLE = Path(os.environ.get("BUNDLE", "/tmp/v1bundle"))
HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(WORKTREE / "data" / "v1_review")))
OUT = DATA_DIR / "curves"
QDIR = BUNDLE / "benchmarks" / "v1_llm" / "questions"


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main():
    OUT.mkdir(exist_ok=True)
    n = 0
    for qid in os.listdir(QDIR):
        qpath = QDIR / qid / "question.json"
        if not qpath.exists():
            continue
        q = read_json(qpath)
        if not q or "synthetic_tabular_classification" not in q["choices"][0]["candidate_path"]:
            continue
        series = []
        for ch in q["choices"]:
            cp = ch["candidate_path"]
            curves_p = BUNDLE / "data" / cp / "results" / "curves.npz"
            if not curves_p.exists():
                continue
            try:
                npz = np.load(curves_p, allow_pickle=True)
                per_seed = npz["curves"]  # (n_seeds, n_steps)
                samples = npz["samples"]
                mean = per_seed.mean(axis=0)
                std = per_seed.std(axis=0)
                series.append({
                    "letter": ch["letter"],
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "samples": samples.tolist(),
                })
            except Exception as e:
                print(f"  {qid}/{ch['letter']}: curves error: {e}", file=sys.stderr)
        (OUT / f"{qid}.json").write_text(json.dumps({"curves": series, "metric": "test_ce"}, ensure_ascii=False))
        n += 1
    print(f"wrote {n} curve files to {OUT}", flush=True)


if __name__ == "__main__":
    main()
