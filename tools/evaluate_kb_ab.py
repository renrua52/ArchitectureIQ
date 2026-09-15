#!/usr/bin/env python3
"""Paired clean-vs-KB evaluation on a fixed ArchitectureIQ subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LLM_EVAL_DIR = ROOT / "tools" / "llm_eval"
if str(LLM_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_EVAL_DIR))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from kb_pipeline import (  # noqa: E402
    atomic_write_json,
    completion_from_payload,
    completion_payload,
    parse_solver,
    read_json,
    retrieve_claims,
    solver_prompt,
    solver_repair_prompt,
)
from llm_client import LLMClient, ModelConfig  # noqa: E402


FAMILY_QUOTAS_50 = {
    "univariate_regression": 10,
    "multivariate_regression": 10,
    "bigram_lm": 10,
    "synthetic_tabular_classification": 10,
    "xor_classification": 5,
    "spiral_classification": 5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ledger(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_records(
    records: list[dict[str, Any]],
    *,
    family_quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") == "ok":
            by_family[str(record["family"])].append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for family in sorted(family_quotas):
        available = sorted(by_family[family], key=lambda item: str(item["item_id"]))
        quota = family_quotas[family]
        if len(available) < quota:
            raise ValueError(f"Need {quota} {family} questions, found {len(available)}")
        selected.extend(rng.sample(available, quota))
    rng.shuffle(selected)
    return selected


def question_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("question.json"):
        data = read_json(path)
        question_id = str(data.get("question_id") or path.parent.name)
        if question_id in index:
            raise ValueError(f"Duplicate question id under {root}: {question_id}")
        index[question_id] = path.parent
    return index


def prepare_manifest(
    *,
    ledger_path: Path,
    questions_root: Path,
    kb_snapshot_path: Path,
    out_dir: Path,
    seed: int,
    model_config: ModelConfig,
    retrieval_limit: int,
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        expected = {
            "selection_seed": seed,
            "model": model_config.to_dict(),
            "retrieval_limit": retrieval_limit,
            "kb_snapshot_sha256": hashlib.sha256(kb_snapshot_path.read_bytes()).hexdigest(),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Existing manifest has different {key}: {path}")
        return manifest

    selected = select_records(
        load_ledger(ledger_path),
        family_quotas=FAMILY_QUOTAS_50,
        seed=seed,
    )
    index = question_index(questions_root)
    questions: list[dict[str, Any]] = []
    for record in selected:
        question_id = str(record["question_id"])
        qdir = index.get(question_id)
        if qdir is None:
            raise FileNotFoundError(f"Benchmark question not found locally: {question_id}")
        question = read_json(qdir / "question.json")
        if question.get("profile") != "v1.5":
            raise ValueError(f"Expected v1.5 question: {qdir}")
        prompt = (qdir / "prompt.txt").read_text(encoding="utf-8")
        questions.append(
            {
                "item_id": record["item_id"],
                "question_id": question_id,
                "family": record["family"],
                "question_type": record["question_type"],
                "budget": record["budget"],
                "question_dir": str(qdir),
                "prompt_sha256": sha256_text(prompt),
            }
        )
    manifest = {
        "schema_version": "architectureiq_kb_ab_v1",
        "created_at": utc_now(),
        "status": "running",
        "ledger": str(ledger_path),
        "questions_root": str(questions_root),
        "selection_seed": seed,
        "family_quotas": FAMILY_QUOTAS_50,
        "model": model_config.to_dict(),
        "retrieval_limit": retrieval_limit,
        "kb_snapshot": str(kb_snapshot_path),
        "kb_snapshot_sha256": hashlib.sha256(kb_snapshot_path.read_bytes()).hexdigest(),
        "questions": questions,
    }
    atomic_write_json(path, manifest)
    return manifest


def parse_with_repairs(
    *,
    client: LLMClient,
    config: ModelConfig,
    completion: Any,
    retrieved_ids: set[str],
    result_dir: Path,
) -> tuple[dict[str, Any], bool]:
    try:
        return parse_solver(completion, retrieved_ids), False
    except ValueError as original_error:
        last_error = original_error
        original_content = completion.content
    for attempt in range(1, 4):
        repair_path = result_dir / f"repair_response_{attempt:02d}.json"
        if repair_path.exists():
            repaired = completion_from_payload(read_json(repair_path))
        else:
            repaired = client.complete(solver_repair_prompt(original_content), config)
            atomic_write_json(repair_path, completion_payload(repaired))
        try:
            return parse_solver(repaired, retrieved_ids), True
        except ValueError as exc:
            last_error = exc
    raise last_error


def evaluate_one(
    *,
    question: dict[str, Any],
    arm: str,
    snapshot: dict[str, Any],
    out_dir: Path,
    client: LLMClient,
    config: ModelConfig,
    retrieval_limit: int,
) -> None:
    result_dir = out_dir / "results" / arm / str(question["question_id"])
    result_path = result_dir / "result.json"
    if result_path.exists():
        return
    qdir = Path(question["question_dir"])
    question_prompt = (qdir / "prompt.txt").read_text(encoding="utf-8")
    claims = [] if arm == "clean" else retrieve_claims(snapshot, question_prompt, retrieval_limit)
    prompt = solver_prompt(question_prompt, claims)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    response_path = result_dir / "response.json"
    if response_path.exists():
        completion = completion_from_payload(read_json(response_path))
    else:
        completion = client.complete(prompt, config)
        atomic_write_json(response_path, completion_payload(completion))
    solver, repaired = parse_with_repairs(
        client=client,
        config=config,
        completion=completion,
        retrieved_ids={str(claim["id"]) for claim in claims},
        result_dir=result_dir,
    )
    locked = {
        "locked_at": utc_now(),
        "solver": solver,
        "format_repaired": repaired,
        "retrieved_claim_ids": [claim["id"] for claim in claims],
    }
    atomic_write_json(result_dir / "solver_locked.json", locked)

    # Read GT only after the response and parsed prediction are durable.
    question_data = read_json(qdir / "question.json")
    correct_letter = str(question_data["correct_letter"]).upper()
    result = {
        "question_id": question["question_id"],
        "item_id": question["item_id"],
        "family": question["family"],
        "question_type": question["question_type"],
        "budget": question["budget"],
        "arm": arm,
        "answer": solver["answer"],
        "correct_letter": correct_letter,
        "is_correct": solver["answer"] == correct_letter,
        "primary_claim": solver["primary_claim"],
        "retrieved_claim_ids": locked["retrieved_claim_ids"],
        "format_repaired": repaired,
    }
    atomic_write_json(result_path, result)


def exact_mcnemar_p(clean_only: int, kb_only: int) -> float:
    discordant = clean_only + kb_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(clean_only, kb_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for question in manifest["questions"]:
        question_id = str(question["question_id"])
        clean = read_json(out_dir / "results" / "clean" / question_id / "result.json")
        kb = read_json(out_dir / "results" / "kb" / question_id / "result.json")
        pairs.append(
            {
                "question_id": question_id,
                "item_id": question["item_id"],
                "family": question["family"],
                "question_type": question["question_type"],
                "budget": question["budget"],
                "clean_correct": clean["is_correct"],
                "kb_correct": kb["is_correct"],
                "clean_answer": clean["answer"],
                "kb_answer": kb["answer"],
                "correct_letter": clean["correct_letter"],
                "kb_primary_claim": kb["primary_claim"],
            }
        )
    clean_correct = sum(pair["clean_correct"] for pair in pairs)
    kb_correct = sum(pair["kb_correct"] for pair in pairs)
    clean_only = sum(pair["clean_correct"] and not pair["kb_correct"] for pair in pairs)
    kb_only = sum(pair["kb_correct"] and not pair["clean_correct"] for pair in pairs)
    both_correct = sum(pair["clean_correct"] and pair["kb_correct"] for pair in pairs)
    both_wrong = len(pairs) - clean_only - kb_only - both_correct
    summary = {
        "schema_version": "architectureiq_kb_ab_result_v1",
        "completed_at": utc_now(),
        "question_count": len(pairs),
        "clean": {"correct": clean_correct, "accuracy": clean_correct / len(pairs)},
        "kb": {"correct": kb_correct, "accuracy": kb_correct / len(pairs)},
        "accuracy_delta": (kb_correct - clean_correct) / len(pairs),
        "paired": {
            "both_correct": both_correct,
            "clean_only_correct": clean_only,
            "kb_only_correct": kb_only,
            "both_wrong": both_wrong,
            "exact_mcnemar_p": exact_mcnemar_p(clean_only, kb_only),
        },
        "kb_citations": sum(pair["kb_primary_claim"]["type"] == "kb" for pair in pairs),
        "pairs": pairs,
    }
    atomic_write_json(out_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--questions-root", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--seed", type=int, default=70136)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retrieval-limit", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    base_url = os.environ.get("VAPI_BASE", "").strip()
    api_key = os.environ.get("VAPI_KEY", "").strip()
    if not base_url or not api_key:
        raise RuntimeError("VAPI_BASE and VAPI_KEY must both be set")
    model_extra: dict[str, Any] = {}
    if args.model.lower().startswith("claude"):
        model_extra = {"thinking": {"type": "enabled", "budget_tokens": 8192}}
    config = ModelConfig(
        name=args.model,
        temperature=0.0,
        max_tokens=16384,
        extra=model_extra,
    )
    manifest = prepare_manifest(
        ledger_path=args.ledger,
        questions_root=args.questions_root,
        kb_snapshot_path=args.kb_snapshot,
        out_dir=args.out_dir,
        seed=args.seed,
        model_config=config,
        retrieval_limit=args.retrieval_limit,
    )
    if args.prepare_only:
        print(f"Prepared {len(manifest['questions'])} questions at {args.out_dir}")
        return 0
    snapshot = read_json(args.kb_snapshot)
    client = LLMClient(base_url=base_url, api_key=api_key, timeout_s=args.timeout)
    tasks = [
        (question, arm)
        for question in manifest["questions"]
        for arm in ("clean", "kb")
        if not (
            args.out_dir / "results" / arm / str(question["question_id"]) / "result.json"
        ).exists()
    ]
    completed = 2 * len(manifest["questions"]) - len(tasks)
    failures: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                question=question,
                arm=arm,
                snapshot=snapshot,
                out_dir=args.out_dir,
                client=client,
                config=config,
                retrieval_limit=args.retrieval_limit,
            ): (str(question["question_id"]), arm)
            for question, arm in tasks
        }
        for future in as_completed(futures):
            question_id, arm = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((question_id, arm, f"{type(exc).__name__}: {exc}"))
                print(f"FAILED {arm} {question_id}: {exc}", flush=True)
            else:
                completed += 1
                print(f"[{completed}/{2 * len(manifest['questions'])}] {arm} {question_id}", flush=True)
    if failures:
        atomic_write_json(
            args.out_dir / "failures.json",
            [{"question_id": qid, "arm": arm, "error": error} for qid, arm, error in failures],
        )
        return 1

    summary = summarize(args.out_dir, manifest)
    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["summary"] = str(args.out_dir / "summary.json")
    atomic_write_json(args.out_dir / "manifest.json", manifest)
    print(
        f"clean={summary['clean']['correct']}/{summary['question_count']} "
        f"kb={summary['kb']['correct']}/{summary['question_count']} "
        f"delta={summary['accuracy_delta']:+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
