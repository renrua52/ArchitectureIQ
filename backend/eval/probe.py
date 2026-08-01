"""Reproducible probing harness for eval question sets.

Samples a stratified mini-batch of questions from a generated 题集
(``backend/eval/sets/{set}/questions.jsonl``) for LLM probing with
sub-agents:

  * Stratifies by winner-vs-runner-up ratio: tight (<1.15), medium
    (1.15-2.0), loose (>=2.0).
  * Never puts two questions of the same problem in one batch (no
    cross-question calibration leakage inside a batch).
  * Writes ``questions.jsonl`` (full items) + ``prompts.txt`` (rendered
    prompts, numbered) per batch under
    ``artifacts/eval_probe/{set}/batches/``.

Scoring: each probe agent writes ``batch_{i}_answers.jsonl`` with one JSON
per line (question_id / answer / reason); ``--score`` checks them against
the set's ground truth and prints accuracy.

Usage:
    .venv/bin/python -m backend.eval.probe --set select_best_v1.1 \\
        --num-batches 2 --batch-size 6 --seed 20260802
    .venv/bin/python -m backend.eval.probe --set select_best_v1.1 --score
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

SETS_ROOT = Path("backend/eval/sets")
ARTIFACTS_ROOT = Path("artifacts/eval_probe")

TIGHT = 1.15
MEDIUM = 2.0


def load_set(set_name: str) -> list[dict]:
    path = SETS_ROOT / set_name / "questions.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def sample_batches(
    items: list[dict],
    rng: random.Random,
    num_batches: int,
    batch_size: int,
) -> list[list[dict]]:
    """Stratified sampling; each batch has distinct problem_ids."""
    strata = {"tight": [], "medium": [], "loose": []}
    for it in items:
        r = it.get("statistics", {}).get("ratio", 0)
        if r < TIGHT:
            strata["tight"].append(it)
        elif r < MEDIUM:
            strata["medium"].append(it)
        else:
            strata["loose"].append(it)

    per_batch = batch_size // 3
    counts = {"tight": per_batch, "medium": per_batch, "loose": per_batch}
    remainder = batch_size - 3 * per_batch
    for i, name in enumerate(["tight", "medium", "loose"]):
        if i < remainder:
            counts[name] += 1

    batches: list[list[dict]] = [[] for _ in range(num_batches)]
    batch_problems: list[set[str]] = [set() for _ in range(num_batches)]
    for name in ("tight", "medium", "loose"):
        n = counts[name]
        pool = list(strata[name])
        rng.shuffle(pool)
        idx = 0
        for b in range(num_batches):
            placed = 0
            while placed < n and idx < len(pool):
                it = pool[idx]
                idx += 1
                if it["problem_id"] in batch_problems[b]:
                    continue
                batches[b].append(it)
                batch_problems[b].add(it["problem_id"])
                placed += 1
    return batches


def write_batches(set_name: str, batches: list[list[dict]], seed: int) -> Path:
    out = ARTIFACTS_ROOT / set_name / "batches"
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batches, start=1):
        with (out / f"batch_{i}_questions.jsonl").open("w", encoding="utf-8") as f:
            for it in batch:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with (out / f"batch_{i}_prompts.txt").open("w", encoding="utf-8") as f:
            for j, it in enumerate(batch, start=1):
                f.write(f"===== QUESTION {j} =====\n")
                f.write(it["prompt"])
                f.write("\n\n")
        print(f"batch {i}: {len(batch)} questions, "
              f"ratios={[round(x['statistics']['ratio'], 2) for x in batch]}")
    (out / "sampling.json").write_text(json.dumps({
        "set": set_name, "seed": seed,
        "batches": [[it["question_id"] for it in b] for b in batches],
    }, indent=2), encoding="utf-8")
    return out


def score(set_name: str) -> int:
    items = {it["question_id"]: it for it in load_set(set_name)}
    out = ARTIFACTS_ROOT / set_name / "batches"
    batches = sorted(out.glob("batch_*_questions.jsonl"))
    total = correct = 0
    for qfile in batches:
        tag = qfile.stem.replace("_questions", "")
        items = [json.loads(line) for line in qfile.open(encoding="utf-8")]
        answers = {}
        for line in (out / f"{tag}_answers.jsonl").open(encoding="utf-8"):
            a = json.loads(line)
            if "question_id" in a:
                answers[a["question_id"]] = a
            elif "q" in a:
                answers[items[int(a["q"]) - 1]["question_id"]] = a
        n = c = 0
        for it in items:
            n += 1
            a = answers.get(it["question_id"])
            if a is None:
                continue
            if a["answer"] == it["correct_letter"]:
                c += 1
            else:
                print(f"  WRONG {it['question_id']} "
                      f"(model={a['answer']} correct={it['correct_letter']} "
                      f"ratio={it['statistics']['ratio']})")
        total += n
        correct += c
        print(f"{tag}: {c}/{n}")
    print(f"TOTAL: {correct}/{total} ({100 * correct / total:.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--num-batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.score:
        return score(args.set)

    rng = random.Random(args.seed)
    items = load_set(args.set)
    batches = sample_batches(items, rng, args.num_batches, args.batch_size)
    write_batches(args.set, batches, args.seed)
    print(f"wrote {sum(len(b) for b in batches)} questions to "
          f"{ARTIFACTS_ROOT / args.set / 'batches'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
