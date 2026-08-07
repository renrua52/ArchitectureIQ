#!/usr/bin/env python3
"""Curate a human-facing Stage-3 subset from the V1 LLM benchmark and bake it.

Human-readable dataset buckets only:
  univariate, xor, spiral, bigram

Target ~250 three-choice questions with gradients over question type,
llm_difficulty, and training budget.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_ROOT = REPO / "benchmarks" / "v1_llm" / "questions"
TAGS_PATH = REPO / "benchmarks" / "v1_llm" / "llm_difficulty.json"
HUMAN_ROOT = REPO / "benchmarks" / "v1_human"
MANIFEST_PATH = HUMAN_ROOT / "manifest.json"
BAKE_ROOT = HUMAN_ROOT / "bake_root"
BAKE_OUT = HUMAN_ROOT / "questions.json"
DATA_DATASETS = REPO / "data" / "datasets"

HUMAN_BUCKETS = ("univariate", "xor", "spiral", "bigram")
TYPE_ORDER = ("architecture_only", "optimizer_only", "mixed")
DIFF_ORDER = ("easy", "medium", "hard", "very_hard")

# Soft quotas inside each bucket (normalized if inventory is short).
TYPE_WEIGHTS = {"architecture_only": 0.40, "optimizer_only": 0.30, "mixed": 0.30}
DIFF_WEIGHTS = {"easy": 0.20, "medium": 0.35, "hard": 0.30, "very_hard": 0.15}

COT_EXCLUDE_MODELS = {"Kimi-K3", "grok-4.2"}
COT_PREFERRED_MODELS = [
    "claude-opus-5",
    "claude-sonnet-4-6",
    "gpt-5.2",
    "gpt-5.6-sol",
    "gpt-5.4-mini",
    "gpt-4o",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "GLM-5.1",
    "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash",
]
COT_MIN_CHARS = 80
COT_MAX_CHARS = 12_000


def _cot_model_rank(name: str) -> tuple[int, str]:
    try:
        return COT_PREFERRED_MODELS.index(name), name
    except ValueError:
        return len(COT_PREFERRED_MODELS), name


def _cot_pick_text(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    parts = rec.get("message_parts") or {}
    candidates = [
        ("reasoning_content", (parts.get("reasoning_content") or parts.get("reasoning") or "").strip()),
        ("chain_of_thought", (rec.get("chain_of_thought") or "").strip()),
        ("content", (parts.get("content") or "").strip()),
        ("model_response", (rec.get("model_response") or "").strip()),
    ]
    for source, text in candidates:
        if len(text) >= COT_MIN_CHARS:
            return text, source
    for source, text in candidates:
        if text:
            return text, source
    return None, None


def _cot_truncate(text: str) -> str:
    if len(text) <= COT_MAX_CHARS:
        return text
    return text[:COT_MAX_CHARS].rstrip() + "\n\n…[truncated for quiz display]"


def inject_llm_cot(bake: dict[str, Any], runs_root: Path) -> dict[str, int]:
    """Attach multi-model llmCot entries (correct + wrong) for quiz dropdown."""
    stats = {
        "with_any_cot": 0,
        "with_correct_cot": 0,
        "no_correct": 0,
        "no_cot": 0,
        "entries": 0,
    }
    if not runs_root.is_dir():
        print(f"warn: llm_runs missing at {runs_root}; skipping llmCot")
        for item in bake.get("byId", {}).values():
            item["llmCot"] = {"available": False, "reason": "no_correct"}
            stats["no_correct"] += 1
        return stats

    models = sorted(
        [p.name for p in runs_root.iterdir() if p.is_dir() and p.name not in COT_EXCLUDE_MODELS],
        key=_cot_model_rank,
    )
    for qid, item in bake.get("byId", {}).items():
        entries: list[dict[str, Any]] = []
        saw_correct = False
        for model in models:
            path = runs_root / model / "results" / f"{qid}.json"
            if not path.is_file():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("error"):
                continue
            parsed = rec.get("parsed_letter")
            correct = bool(rec.get("correct")) and parsed is not None
            if correct:
                saw_correct = True
            text, source = _cot_pick_text(rec)
            if not text or len(text) < COT_MIN_CHARS:
                continue
            entries.append(
                {
                    "model": model,
                    "correct": correct,
                    "parsedLetter": parsed,
                    "source": source,
                    "text": _cot_truncate(text),
                }
            )

        default_model = next((e["model"] for e in entries if e["correct"]), None)
        if default_model is None and entries:
            default_model = entries[0]["model"]

        if entries:
            item["llmCot"] = {
                "available": True,
                "defaultModel": default_model,
                "entries": entries,
            }
            stats["with_any_cot"] += 1
            stats["entries"] += len(entries)
            if any(e["correct"] for e in entries):
                stats["with_correct_cot"] += 1
            elif not saw_correct:
                stats["no_correct"] += 1
        elif saw_correct:
            item["llmCot"] = {"available": False, "reason": "no_cot"}
            stats["no_cot"] += 1
        else:
            item["llmCot"] = {"available": False, "reason": "no_correct"}
            stats["no_correct"] += 1
    print("llmCot inject:", stats)
    return stats


def load_pool(tags: dict[str, Any]) -> list[dict[str, Any]]:
    pool = []
    for qdir in sorted(QUESTIONS_ROOT.glob("q_*")):
        qpath = qdir / "question.json"
        q = json.loads(qpath.read_text(encoding="utf-8"))
        bm = q.get("benchmark") or {}
        qid = q["question_id"]
        tag = tags["questions"].get(qid) or {}
        bucket = bm.get("dataset_bucket") or tag.get("dataset_bucket")
        if bucket not in HUMAN_BUCKETS:
            continue
        pool.append(
            {
                "question_id": qid,
                "dir": qdir,
                "bucket": bucket,
                "type": q.get("type"),
                "budget_tier": bm.get("budget_tier")
                or (q.get("budget") or {}).get("total_samples_seen"),
                "gap_constrained": bool(bm.get("gap_constrained")),
                "param_similar": bool(bm.get("param_similar")),
                "llm_difficulty": tag.get("llm_difficulty") or bm.get("llm_difficulty"),
                "llm_consensus_acc": tag.get("llm_consensus_acc"),
                "frontier_miss": bool(tag.get("frontier_miss") or bm.get("llm_frontier_miss")),
                "family": q.get("family"),
                "dataset_id": q.get("dataset_id"),
            }
        )
    return pool


def _allocate(total: int, weights: dict[str, float], keys: tuple[str, ...]) -> dict[str, int]:
    raw = {k: total * weights[k] for k in keys}
    base = {k: int(v) for k, v in raw.items()}
    rem = total - sum(base.values())
    order = sorted(keys, key=lambda k: -(raw[k] - base[k]))
    for k in order[:rem]:
        base[k] += 1
    return base


def stratified_sample(
    pool: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_bucket[row["bucket"]].append(row)

    per_bucket = _allocate(
        target,
        {b: 1.0 / len(HUMAN_BUCKETS) for b in HUMAN_BUCKETS},
        HUMAN_BUCKETS,
    )

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    for bucket in HUMAN_BUCKETS:
        candidates = list(by_bucket.get(bucket, []))
        rng.shuffle(candidates)
        need = per_bucket[bucket]
        type_quota = _allocate(need, TYPE_WEIGHTS, TYPE_ORDER)

        # Further split each type quota across difficulties.
        bucket_picks: list[dict[str, Any]] = []
        for qtype in TYPE_ORDER:
            type_cands = [r for r in candidates if r["type"] == qtype and r["question_id"] not in used]
            t_need = type_quota[qtype]
            diff_quota = _allocate(t_need, DIFF_WEIGHTS, DIFF_ORDER)
            for diff in DIFF_ORDER:
                d_cands = [r for r in type_cands if r["llm_difficulty"] == diff]
                rng.shuffle(d_cands)
                take = min(diff_quota[diff], len(d_cands))
                for row in d_cands[:take]:
                    bucket_picks.append(row)
                    used.add(row["question_id"])
                    type_cands = [r for r in type_cands if r["question_id"] not in used]

            # Fill remaining type slots from leftover type_cands (any difficulty).
            remaining = type_quota[qtype] - sum(
                1 for r in bucket_picks if r["type"] == qtype
            )
            if remaining > 0:
                rng.shuffle(type_cands)
                for row in type_cands[:remaining]:
                    bucket_picks.append(row)
                    used.add(row["question_id"])

        # Fill remaining bucket slots from any leftover in bucket.
        remaining = need - len(bucket_picks)
        if remaining > 0:
            leftovers = [r for r in candidates if r["question_id"] not in used]
            # Prefer budget diversity.
            leftovers.sort(
                key=lambda r: (
                    sum(1 for x in bucket_picks if x["budget_tier"] == r["budget_tier"]),
                    rng.random(),
                )
            )
            for row in leftovers[:remaining]:
                bucket_picks.append(row)
                used.add(row["question_id"])

        selected.extend(bucket_picks)

    # Global top-up if still short.
    if len(selected) < target:
        leftovers = [r for r in pool if r["question_id"] not in used]
        rng.shuffle(leftovers)
        for row in leftovers[: target - len(selected)]:
            selected.append(row)
            used.add(row["question_id"])

    # Stable order: bucket, difficulty (hard first for humans? or easy-first)
    # Interleave for a nicer quiz: cycle buckets, prefer medium/hard early mix.
    selected.sort(
        key=lambda r: (
            HUMAN_BUCKETS.index(r["bucket"]),
            DIFF_ORDER.index(r["llm_difficulty"]) if r["llm_difficulty"] in DIFF_ORDER else 99,
            TYPE_ORDER.index(r["type"]) if r["type"] in TYPE_ORDER else 99,
            str(r["budget_tier"]),
            r["question_id"],
        )
    )
    return selected[:target]


def prepare_bake_root(selected: list[dict[str, Any]]) -> Path:
    if BAKE_ROOT.exists():
        shutil.rmtree(BAKE_ROOT)
    questions_dir = BAKE_ROOT / "questions"
    questions_dir.mkdir(parents=True)
    datasets_link = BAKE_ROOT / "datasets"
    if not DATA_DATASETS.is_dir():
        raise SystemExit(f"missing datasets dir: {DATA_DATASETS}")
    datasets_link.symlink_to(DATA_DATASETS.resolve())
    for row in selected:
        dest = questions_dir / row["question_id"]
        dest.symlink_to(row["dir"].resolve())
    return BAKE_ROOT


def write_manifest(selected: list[dict[str, Any]], *, target: int, seed: int) -> None:
    HUMAN_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "buckets": dict(Counter(r["bucket"] for r in selected)),
        "types": dict(Counter(r["type"] for r in selected)),
        "llm_difficulty": dict(Counter(r["llm_difficulty"] for r in selected)),
        "budget_tier": dict(Counter(str(r["budget_tier"]) for r in selected)),
        "frontier_miss": sum(1 for r in selected if r["frontier_miss"]),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage": 3,
        "title": "ArchitectureIQ V1 human pre-review pack",
        "target": target,
        "seed": seed,
        "human_buckets": list(HUMAN_BUCKETS),
        "count": len(selected),
        "summary": summary,
        "questions": [
            {
                "question_id": r["question_id"],
                "bucket": r["bucket"],
                "type": r["type"],
                "budget_tier": r["budget_tier"],
                "llm_difficulty": r["llm_difficulty"],
                "llm_consensus_acc": r["llm_consensus_acc"],
                "frontier_miss": r["frontier_miss"],
                "family": r["family"],
                "dataset_id": r["dataset_id"],
                "gap_constrained": r["gap_constrained"],
                "param_similar": r["param_similar"],
            }
            for r in selected
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
    print("summary:", json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--skip-bake", action="store_true")
    args = parser.parse_args()

    if not TAGS_PATH.is_file():
        raise SystemExit(f"missing {TAGS_PATH}; run tools/benchmark_v1_tag_difficulty.py first")
    tags = json.loads(TAGS_PATH.read_text(encoding="utf-8"))
    pool = load_pool(tags)
    print(f"human-readable pool: {len(pool)}")
    print("pool by bucket:", dict(Counter(r["bucket"] for r in pool)))
    print("pool by difficulty:", dict(Counter(r["llm_difficulty"] for r in pool)))

    selected = stratified_sample(pool, target=args.target, seed=args.seed)
    write_manifest(selected, target=args.target, seed=args.seed)
    bake_root = prepare_bake_root(selected)
    print(f"prepared bake root {bake_root}")

    if args.skip_bake:
        return

    cmd = [
        sys.executable,
        str(REPO / "tools" / "export_quiz_static.py"),
        "--data-root",
        str(bake_root),
        "--out",
        str(BAKE_OUT),
    ]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))

    # Inject llm_difficulty into baked questions (schema allows extra keys on bakedQuestion).
    bake = json.loads(BAKE_OUT.read_text(encoding="utf-8"))
    by_id_meta = {r["question_id"]: r for r in selected}
    for qid, item in bake["byId"].items():
        meta = by_id_meta[qid]
        item["llmDifficulty"] = meta["llm_difficulty"]
        item["datasetBucket"] = meta["bucket"]
        item["llmConsensusAcc"] = meta["llm_consensus_acc"]
        item["frontierMiss"] = meta["frontier_miss"]
        if "summary" in item and isinstance(item["summary"], dict):
            # summary is mirrored into questions[]; schema forbids unknown keys there.
            pass

    inject_llm_cot(bake, REPO / "benchmarks" / "v1_llm" / "llm_runs")

    # Enrich catalog entries via track field (allowed) encoding difficulty for UI filters.
    for entry in bake["questions"]:
        meta = by_id_meta[entry["id"]]
        entry["track"] = f"human_{meta['bucket']}_{meta['llm_difficulty']}"
        bake["byId"][entry["id"]]["track"] = entry["track"]
        bake["byId"][entry["id"]]["summary"] = entry

    bake["ordered"] = True
    bake["collection"] = {
        "schema_version": "v1_human_prereview_v1",
        "collection_id": "v1_human_prereview_20260804",
        "title": "ArchitectureIQ V1 · human pre-review",
        "question_count": len(selected),
        "candidate_reuse_policy": "benchmark_v1_llm",
        "question_order_policy": "manifest_stratified",
        "profiles": ["v1"],
        "tracks": sorted({e["track"] for e in bake["questions"]}),
        "note": (
            "Stage-3 pre-human-review pack. Human-readable buckets only "
            "(univariate, xor, spiral, bigram). llm_difficulty from consensus_v1 "
            "over completed LLM eval models."
        ),
        "source": str(MANIFEST_PATH.relative_to(REPO)),
    }
    BAKE_OUT.write_text(json.dumps(bake, indent=2) + "\n", encoding="utf-8")

    validate = [
        sys.executable,
        str(REPO / "tools" / "validate_quiz_bake.py"),
        str(BAKE_OUT),
    ]
    print("validating:", " ".join(validate))
    subprocess.check_call(validate, cwd=str(REPO))
    print(f"bake ready: {BAKE_OUT}")


if __name__ == "__main__":
    main()
