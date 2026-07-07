"""Run an LLM over ArchitectureIQ question prompts and score against ground truth."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from completion import ModelExchange, fetch_model_response
from prompt_wrapper import format_eval_prompt
from question_loader import QuestionItem, list_questions
from response_parser import parse_choice_letter, split_chain_of_thought


class LLMBackend(Protocol):
    def complete(self, prompt: str, config: Any) -> Any: ...


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    prompt_hash: str
    ground_truth_letter: str
    parsed_letter: str | None
    correct: bool
    model_response: str
    chain_of_thought: str | None
    question_type: str
    family: str
    eval_prompt: str
    finish_reason: str | None = None
    truncated: bool = False
    continuation_count: int = 0
    usage: dict[str, Any] | None = None
    message_parts: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "prompt_hash": self.prompt_hash,
            "ground_truth_letter": self.ground_truth_letter,
            "parsed_letter": self.parsed_letter,
            "correct": self.correct,
            "model_response": self.model_response,
            "chain_of_thought": self.chain_of_thought,
            "question_type": self.question_type,
            "family": self.family,
            "eval_prompt": self.eval_prompt,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "continuation_count": self.continuation_count,
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        if self.message_parts:
            payload["message_parts"] = self.message_parts
        return payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_model_slug(model_name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name)
    return slug.strip("_") or "model"


def default_run_dir(runs_root: Path, model_name: str, run_id: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = run_id or _safe_model_slug(model_name)
    return runs_root / f"{stamp}_{suffix}"


def evaluate_question(item: QuestionItem, exchange: ModelExchange, eval_prompt: str) -> QuestionResult:
    parsed = parse_choice_letter(exchange.model_response, item.valid_letters)
    gt = item.correct_letter
    return QuestionResult(
        question_id=item.question_id,
        prompt_hash=item.prompt_hash,
        ground_truth_letter=gt,
        parsed_letter=parsed,
        correct=parsed == gt if parsed is not None else False,
        model_response=exchange.model_response,
        chain_of_thought=split_chain_of_thought(exchange.model_response, parsed),
        question_type=str(item.question.get("type", "?")),
        family=str(item.question.get("family", "?")),
        eval_prompt=eval_prompt,
        finish_reason=exchange.finish_reason,
        truncated=exchange.truncated,
        continuation_count=exchange.continuation_count,
        usage=exchange.usage,
        message_parts=exchange.message_parts or None,
    )


def summarize_results(results: list[QuestionResult]) -> dict[str, Any]:
    scored = [r for r in results if r.parsed_letter is not None]
    correct = sum(1 for r in scored if r.correct)
    total = len(scored)
    accuracy = (correct / total) if total else None

    by_type: dict[str, dict[str, int | float | None]] = {}
    buckets: dict[str, list[QuestionResult]] = {}
    for result in results:
        buckets.setdefault(result.question_type, []).append(result)
    for qtype, rows in buckets.items():
        parsed_rows = [r for r in rows if r.parsed_letter is not None]
        n = len(parsed_rows)
        c = sum(1 for r in parsed_rows if r.correct)
        by_type[qtype] = {
            "total": n,
            "correct": c,
            "accuracy": (c / n) if n else None,
        }

    return {
        "total_questions": len(results),
        "parsed": total,
        "unparsed": len(results) - total,
        "correct": correct,
        "accuracy": accuracy,
        "by_type": by_type,
    }


def write_question_result(run_dir: Path, result: QuestionResult) -> Path:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{result.question_id}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def _load_cached_result(cached: dict[str, Any]) -> QuestionResult:
    return QuestionResult(
        question_id=cached["question_id"],
        prompt_hash=cached["prompt_hash"],
        ground_truth_letter=cached["ground_truth_letter"],
        parsed_letter=cached.get("parsed_letter"),
        correct=bool(cached["correct"]),
        model_response=cached["model_response"],
        chain_of_thought=cached.get("chain_of_thought"),
        question_type=cached.get("question_type", "?"),
        family=cached.get("family", "?"),
        eval_prompt=cached.get("eval_prompt", ""),
        finish_reason=cached.get("finish_reason"),
        truncated=bool(cached.get("truncated", False)),
        continuation_count=int(cached.get("continuation_count", 0)),
        usage=cached.get("usage"),
        message_parts=cached.get("message_parts"),
    )


def _evaluate_one(
    item: QuestionItem,
    run_dir: Path,
    model_config: Any,
    client: LLMBackend,
) -> QuestionResult:
    eval_prompt = format_eval_prompt(item.prompt_text, item.valid_letters)
    exchange = fetch_model_response(client, eval_prompt, model_config, item.valid_letters)
    result = evaluate_question(item, exchange, eval_prompt)
    write_question_result(run_dir, result)
    return result


def run_evaluation(
    *,
    questions_root: Path,
    run_dir: Path,
    model_config: Any,
    client: LLMBackend,
    limit: int | None = None,
    skip_existing: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    items = list_questions(questions_root)
    if limit is not None:
        items = items[:limit]

    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[QuestionResult] = []
    pending: list[QuestionItem] = []

    for item in items:
        result_path = run_dir / "results" / f"{item.question_id}.json"
        if skip_existing and result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(_load_cached_result(cached))
        else:
            pending.append(item)

    if pending:
        if workers <= 1:
            for item in pending:
                results.append(_evaluate_one(item, run_dir, model_config, client))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_evaluate_one, item, run_dir, model_config, client)
                    for item in pending
                ]
                for future in as_completed(futures):
                    results.append(future.result())

    summary = summarize_results(results)
    manifest = {
        "run_id": run_dir.name,
        "created_at": _utc_now_iso(),
        "questions_root": str(questions_root.resolve()),
        "model": model_config.to_dict() if hasattr(model_config, "to_dict") else model_config,
        "workers": workers,
        "summary": summary,
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
