#!/usr/bin/env python3
"""Build and score high-budget GPT evaluation bundles.

This module deliberately does not import ``architecture_iq``. It reads frozen
question artifacts, emits a prompt-only blind bundle, and scores frozen
prediction files only after their SHA-256 is known.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_BUNDLE_KEYS = {
    "correct_letter",
    "significance",
    "candidate_path",
    "candidate_set_path",
    "candidate_sets",
    "answer_key",
    "private_answer_key_path",
    "private_answer_key_sha256",
}

FORBIDDEN_TEXT_PATTERNS = (
    "results/summary.json",
    "curves.npz",
    "choice_mean_metrics",
    "correct_candidate_id",
    "mean_test_mse\":",
    "mean_test_ce\":",
    "final_test_mse\":",
    "final_test_ce\":",
)

ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short_sha256": sha256_text(status or ""),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def question_path_from_prompt(prompt_path: Path) -> Path:
    if prompt_path.name != "prompt.txt":
        raise ValueError(f"Expected prompt.txt path, got {prompt_path}")
    return prompt_path.with_name("question.json")


def resolve_artifact_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    data_relative = repo_root / "data" / path
    if data_relative.exists():
        return data_relative
    return repo_relative


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _choice_metric(summary: dict[str, Any], metric: str) -> float | None:
    key = f"mean_{metric}"
    value = summary.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    answers = payload.get("answers") if isinstance(payload, dict) else payload
    if not isinstance(answers, list):
        raise ValueError("Answer key must contain an answers list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in answers:
        qid = str(row["question_id"])
        if qid in by_id:
            raise ValueError(f"Duplicate answer key question_id: {qid}")
        by_id[qid] = row
    return by_id


def sanitize_question(repo_root: Path, row: dict[str, Any], n: int) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_path = (repo_root / row["prompt_path"]).resolve()
    question_path = question_path_from_prompt(prompt_path)
    question = load_json(question_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    qid = str(row["question_id"])
    if question.get("question_id") != qid:
        raise ValueError(f"Manifest/question mismatch for {qid}: {question_path}")

    choices = [
        {
            "letter": str(choice["letter"]).upper(),
            "candidate_id": str(choice["candidate_id"]),
        }
        for choice in question["choices"]
    ]
    sanitized = {
        "n": n,
        "question_id": qid,
        "family": row.get("family", question.get("family")),
        "dataset_id": row.get("dataset_id", question.get("dataset_id")),
        "question_type": row.get("type", question.get("type")),
        "budget": row.get("budget", question.get("budget", {}).get("total_samples_seen")),
        "prompt_path": _repo_relative(repo_root, prompt_path),
        "prompt_sha256": sha256_text(prompt_text),
        "valid_letters": [choice["letter"] for choice in choices],
        "choices": choices,
        "prompt_text": prompt_text,
    }

    metric = question.get("evaluation", {}).get("selection_metric")
    feedback_choices: dict[str, float | None] = {}
    for choice in question["choices"]:
        candidate_path = resolve_artifact_path(repo_root, choice["candidate_path"])
        summary_path = candidate_path / "results" / "summary.json"
        summary = load_json(summary_path)
        feedback_choices[str(choice["letter"]).upper()] = _choice_metric(summary, str(metric))

    private_feedback = {
        "n": n,
        "question_id": qid,
        "question_path": _repo_relative(repo_root, question_path),
        "metric": metric,
        "choice_mean_metrics": feedback_choices,
    }
    return sanitized, private_feedback


def build_bundle(
    *,
    repo_root: Path,
    public_manifest_path: Path,
    private_answer_key_path: Path,
    release_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public_manifest = load_json(public_manifest_path)
    answer_key = load_answer_key(private_answer_key_path)
    release_manifest = load_json(release_manifest_path) if release_manifest_path else None
    release_ids = {q["question_id"] for q in release_manifest.get("questions", [])} if release_manifest else None

    questions: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n, row in enumerate(public_manifest.get("questions", []), start=1):
        qid = str(row["question_id"])
        if qid in seen:
            raise ValueError(f"Duplicate question_id in public manifest: {qid}")
        if qid not in answer_key:
            raise ValueError(f"Missing answer key row for {qid}")
        if release_ids is not None and qid not in release_ids:
            raise ValueError(f"Question {qid} missing from release manifest")
        sanitized, private_feedback = sanitize_question(repo_root, row, n)
        private_feedback.update(
            {
                "correct_letter": str(answer_key[qid]["correct_letter"]).upper(),
                "correct_candidate_id": str(answer_key[qid]["candidate_id"]),
                "gap": answer_key[qid].get("gap"),
                "win_rate": answer_key[qid].get("win_rate"),
            }
        )
        questions.append(sanitized)
        feedback.append(private_feedback)
        seen.add(qid)

    bundle = {
        "schema_version": "architecture_iq.high_budget_blind_bundle.v1",
        "created_at": utc_now_iso(),
        "source_public_manifest": _repo_relative(repo_root, public_manifest_path),
        "source_public_manifest_sha256": sha256_file(public_manifest_path),
        "source_release_manifest": _repo_relative(repo_root, release_manifest_path) if release_manifest_path else None,
        "source_release_manifest_sha256": sha256_file(release_manifest_path) if release_manifest_path else None,
        "question_count": len(questions),
        "canonical_order": [q["question_id"] for q in questions],
        "questions": questions,
    }
    bundle["bundle_sha256"] = sha256_text(
        canonical_json({k: v for k, v in bundle.items() if k not in {"bundle_sha256", "created_at"}})
    )
    return bundle, feedback


def scan_for_leakage(payload: Any) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_s = str(key)
                if key_s in FORBIDDEN_BUNDLE_KEYS:
                    findings.append({"path": f"{path}.{key_s}", "reason": "forbidden key"})
                walk(item, f"{path}.{key_s}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                walk(item, f"{path}[{idx}]")
        elif isinstance(value, str):
            for pattern in FORBIDDEN_TEXT_PATTERNS:
                if pattern in value:
                    findings.append({"path": path, "reason": f"forbidden text pattern {pattern!r}"})

    walk(payload, "$")
    return findings


def fullset_prompt(bundle: dict[str, Any]) -> str:
    questions = bundle["questions"]
    compact_questions = [
        {
            "n": q["n"],
            "question_id": q["question_id"],
            "family": q["family"],
            "dataset_id": q["dataset_id"],
            "question_type": q["question_type"],
            "budget": q["budget"],
            "valid_letters": q["valid_letters"],
            "choices": q["choices"],
            "prompt_text": q["prompt_text"],
        }
        for q in questions
    ]
    return (
        "# ArchitectureIQ GPT-5.4 full-set blind evaluation\n\n"
        "STRICT PROTOCOL:\n"
        "- You receive the complete high-budget question set at once.\n"
        "- Use only the visible prompt text below and qualitative architecture/training reasoning.\n"
        "- Do not read answer keys, feedback files, scoring files, result summaries, curves, previous attempts, repository files, or hidden ground-truth artifacts.\n"
        "- Do not run shell commands, scripts, training, simulation, approximate experiments, or data reconstruction.\n"
        "- Return strict JSON only with keys: run_label, model, source_used, forbidden_files_viewed, predictions.\n"
        "- predictions must contain exactly one record per question in canonical order.\n"
        "- Each prediction must contain: n, question_id, predicted_letter, predicted_candidate_id, confidence, reason.\n"
        "- predicted_candidate_id must match the selected letter's candidate_id.\n\n"
        f"Bundle SHA-256: {bundle['bundle_sha256']}\n"
        f"Question count: {bundle['question_count']}\n\n"
        "Sanitized questions JSON:\n"
        "```json\n"
        f"{json.dumps(compact_questions, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )


def write_bundle_artifacts(out_dir: Path, bundle: dict[str, Any], feedback: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    leakage_findings = scan_for_leakage(bundle)
    write_json(out_dir / "sanitized_bundle.json", bundle)
    write_json(out_dir / "private_feedback_key.json", feedback)
    write_json(
        out_dir / "questions_sanitized.json",
        [
            {
                "n": q["n"],
                "question_id": q["question_id"],
                "family": q["family"],
                "dataset_id": q["dataset_id"],
                "question_type": q["question_type"],
                "budget": q["budget"],
                "prompt_sha256": q["prompt_sha256"],
                "choices": q["choices"],
                "prompt_text": q["prompt_text"],
            }
            for q in bundle["questions"]
        ],
    )
    prompt = fullset_prompt(bundle)
    (out_dir / "blind_fullset_prompt.md").write_text(prompt, encoding="utf-8")
    write_json(
        out_dir / "leakage_scan.json",
        {
            "schema_version": "architecture_iq.leakage_scan.v1",
            "created_at": utc_now_iso(),
            "bundle_sha256": bundle["bundle_sha256"],
            "passed": not leakage_findings,
            "findings": leakage_findings,
            "blind_fullset_prompt_sha256": sha256_text(prompt),
        },
    )


def parse_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
    return records


def parse_answer_tag(text: str, valid_letters: set[str]) -> str | None:
    matches = ANSWER_TAG_RE.findall(text or "")
    if not matches:
        return None
    letter = matches[-1].upper()
    return letter if letter in valid_letters else None


def normalize_prediction_record(record: dict[str, Any], valid_letters: set[str]) -> dict[str, Any]:
    predicted = record.get("predicted_letter")
    if predicted is None and isinstance(record.get("model_response"), str):
        predicted = parse_answer_tag(record["model_response"], valid_letters)
    if predicted is not None:
        predicted = str(predicted).strip().upper()
        if predicted not in valid_letters:
            predicted = None
    normalized = dict(record)
    normalized["predicted_letter"] = predicted
    return normalized


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ci(correct_flags: list[bool], *, samples: int = 10000, seed: int = 0) -> dict[str, float] | None:
    if not correct_flags:
        return None
    rng = random.Random(seed)
    n = len(correct_flags)
    accs = []
    ints = [1 if flag else 0 for flag in correct_flags]
    for _ in range(samples):
        accs.append(sum(rng.choice(ints) for _ in range(n)) / n)
    return {"low": percentile(accs, 0.025), "high": percentile(accs, 0.975), "samples": samples}


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def score_predictions(
    *,
    prediction_path: Path,
    bundle_path: Path,
    answer_key_path: Path,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    bundle_sha = bundle.get("bundle_sha256")
    if expected_bundle_sha256 and bundle_sha != expected_bundle_sha256:
        raise ValueError(f"Bundle hash mismatch: expected {expected_bundle_sha256}, got {bundle_sha}")
    answer_key = load_answer_key(answer_key_path)
    raw_records = parse_prediction_jsonl(prediction_path)
    by_qid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in raw_records:
        qid = str(record.get("question_id"))
        if qid in by_qid:
            duplicates.append(qid)
        by_qid[qid] = record

    scored_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    unexpected = sorted(set(by_qid) - set(bundle["canonical_order"]))
    for q in bundle["questions"]:
        qid = q["question_id"]
        record = by_qid.get(qid)
        if record is None:
            missing.append(qid)
            continue
        valid_letters = set(q["valid_letters"])
        normalized = normalize_prediction_record(record, valid_letters)
        predicted_letter = normalized.get("predicted_letter")
        correct_letter = str(answer_key[qid]["correct_letter"]).upper()
        letter_to_candidate = {choice["letter"]: choice["candidate_id"] for choice in q["choices"]}
        predicted_candidate_id = normalized.get("predicted_candidate_id")
        candidate_matches = (
            predicted_letter is not None
            and (not predicted_candidate_id or predicted_candidate_id == letter_to_candidate.get(predicted_letter))
        )
        scored_rows.append(
            {
                "n": q["n"],
                "question_id": qid,
                "family": q["family"],
                "question_type": q["question_type"],
                "predicted_letter": predicted_letter,
                "predicted_candidate_id": predicted_candidate_id,
                "correct_letter": correct_letter,
                "correct_candidate_id": answer_key[qid]["candidate_id"],
                "parse_failed": predicted_letter is None,
                "candidate_id_matches_letter": candidate_matches,
                "correct": predicted_letter == correct_letter if predicted_letter is not None else False,
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_family[row["family"]].append(row)
        by_type[row["question_type"]].append(row)

    correct_flags = [row["correct"] for row in scored_rows]
    summary = summarize_bucket(scored_rows)
    return {
        "schema_version": "architecture_iq.high_budget_score.v1",
        "created_at": utc_now_iso(),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "answer_key_path": str(answer_key_path),
        "answer_key_sha256": sha256_file(answer_key_path),
        "question_count_expected": bundle["question_count"],
        "question_count_scored": len(scored_rows),
        "missing_question_ids": missing,
        "duplicate_question_ids": duplicates,
        "unexpected_question_ids": unexpected,
        "parse_failures": [row["question_id"] for row in scored_rows if row["parse_failed"]],
        "candidate_id_mismatches": [
            row["question_id"] for row in scored_rows if not row["candidate_id_matches_letter"]
        ],
        "overall": {
            **summary,
            "bootstrap_95_ci": bootstrap_ci(correct_flags),
        },
        "by_family": {key: summarize_bucket(rows) for key, rows in sorted(by_family.items())},
        "by_question_type": {key: summarize_bucket(rows) for key, rows in sorted(by_type.items())},
        "rows": scored_rows,
    }


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    condition_id: str
    protocol_id: str
    bundle_sha256: str
    bundle_path: str
    question_manifest_path: str | None
    question_manifest_sha256: str | None
    release_manifest_path: str | None
    release_manifest_sha256: str | None
    question_count: int
    canonical_order: list[str]
    prompt_template_sha256: str | None
    model_input_bundle_sha256: str | None
    git: dict[str, Any]
    model_display_name: str = "GPT-5.4"
    exact_model_id: str = "gpt-5.4"
    provider: str = "OpenAI/Codex subagent"
    reasoning_effort: str = "high"
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    retry_policy: str = "No best-of reruns; invalid/missing answers score incorrect."
    parsing_policy: str = "predicted_letter field; fallback to final <answer>LETTER</answer> tag."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "architecture_iq.high_budget_run_spec.v1",
            "created_at": utc_now_iso(),
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "protocol_id": self.protocol_id,
            "question_manifest_path": self.question_manifest_path,
            "question_manifest_sha256": self.question_manifest_sha256,
            "release_manifest_path": self.release_manifest_path,
            "release_manifest_sha256": self.release_manifest_sha256,
            "question_count": self.question_count,
            "canonical_order": self.canonical_order,
            "question_ids": self.canonical_order,
            "bundle_path": self.bundle_path,
            "bundle_sha256": self.bundle_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
            "model_input_bundle_sha256": self.model_input_bundle_sha256,
            "model_display_name": self.model_display_name,
            "exact_model_id": self.exact_model_id,
            "provider": self.provider,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "retry_policy": self.retry_policy,
            "parsing_policy": self.parsing_policy,
            "invalid_or_missing_scoring": "incorrect",
            "seed": None,
            "api_transport": {
                "external_openai_api_completed_requests": 0,
                "usage_tokens_available": False,
                "cost_available": False,
                "request_ids_available": False,
                "fallback_reason": "External OpenAI API was unavailable or exact gpt-5.4 API model id could not be confirmed; GPT-5.4 Codex subagent provider used.",
                "transport_check_path": "artifacts/high_budget_gpt54_eval/api_transport_check.json",
            },
            "start_timestamp_utc": None,
            "end_timestamp_utc": None,
            "git": self.git,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-bundle")
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument("--public-manifest", type=Path, default=Path("artifacts/high_budget_public_manifest.json"))
    build.add_argument("--private-answer-key", type=Path, default=Path("artifacts/high_budget_private_answer_key.json"))
    build.add_argument("--release-manifest", type=Path, default=Path("data/releases/high_budget_confirmed_v1/quiz_manifest.json"))
    build.add_argument("--out-dir", type=Path, required=True)

    spec = sub.add_parser("write-run-spec")
    spec.add_argument("--bundle", type=Path, required=True)
    spec.add_argument("--out", type=Path, required=True)
    spec.add_argument("--run-id", required=True)
    spec.add_argument("--condition-id", required=True)
    spec.add_argument("--protocol-id", required=True)
    spec.add_argument("--provider", default="OpenAI/Codex subagent")
    spec.add_argument("--prompt-template", type=Path)
    spec.add_argument("--repo-root", type=Path, default=Path.cwd())

    score = sub.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--answer-key", type=Path, default=Path("artifacts/high_budget_private_answer_key.json"))
    score.add_argument("--expected-bundle-sha256")
    score.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-bundle":
        repo_root = args.repo_root.resolve()
        bundle, feedback = build_bundle(
            repo_root=repo_root,
            public_manifest_path=(repo_root / args.public_manifest).resolve(),
            private_answer_key_path=(repo_root / args.private_answer_key).resolve(),
            release_manifest_path=(repo_root / args.release_manifest).resolve() if args.release_manifest else None,
        )
        write_bundle_artifacts((repo_root / args.out_dir).resolve(), bundle, feedback)
    elif args.command == "write-run-spec":
        bundle = load_json(args.bundle)
        repo_root = args.repo_root.resolve()
        prompt_template_sha256 = sha256_file(args.prompt_template) if args.prompt_template else None
        model_input_bundle_sha256 = None
        if args.prompt_template and args.prompt_template.is_file():
            match = re.search(
                r"Bundle SHA-256:\s*([0-9a-f]{64})",
                args.prompt_template.read_text(encoding="utf-8"),
            )
            if match:
                model_input_bundle_sha256 = match.group(1)
        run_spec = RunSpec(
            run_id=args.run_id,
            condition_id=args.condition_id,
            protocol_id=args.protocol_id,
            bundle_path=_repo_relative(repo_root, args.bundle.resolve()),
            bundle_sha256=bundle["bundle_sha256"],
            question_manifest_path=bundle.get("source_public_manifest"),
            question_manifest_sha256=bundle.get("source_public_manifest_sha256"),
            release_manifest_path=bundle.get("source_release_manifest"),
            release_manifest_sha256=bundle.get("source_release_manifest_sha256"),
            question_count=int(bundle["question_count"]),
            canonical_order=[str(qid) for qid in bundle["canonical_order"]],
            prompt_template_sha256=prompt_template_sha256,
            model_input_bundle_sha256=model_input_bundle_sha256,
            git=git_metadata(repo_root),
            provider=args.provider,
        )
        write_json(args.out, run_spec.to_dict())
    elif args.command == "score":
        scored = score_predictions(
            prediction_path=args.predictions,
            bundle_path=args.bundle,
            answer_key_path=args.answer_key,
            expected_bundle_sha256=args.expected_bundle_sha256,
        )
        write_json(args.out, scored)


if __name__ == "__main__":
    main()
