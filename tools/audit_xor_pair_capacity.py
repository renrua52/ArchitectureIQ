#!/usr/bin/env python3
"""Audit the usable MLP/KAN pair capacity of XOR candidate sets.

This is an inventory-only audit: candidate specs and stored summaries are
read, candidate ids are de-duplicated, and every cross-family pair is checked
with the canonical significance validator.  The only write is the requested
JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from architecture_iq.profile import Profile, load_profile
from architecture_iq.significance.validator import validate_significance


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _candidate_id(spec: dict[str, Any], candidate_dir: Path) -> str:
    value = spec.get("candidate_id")
    return str(value) if value else candidate_dir.name


def _model_type(spec: dict[str, Any]) -> str | None:
    model = spec.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("type")
    return str(value).lower() if value is not None else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _strict_summary_check(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("failed_seeds") != 0:
        reasons.append("failed_seeds must equal 0")
    for key in ("mean_test_ce", "std_test_ce"):
        if not _finite(summary.get(key)):
            reasons.append(f"{key} must be finite")
    seed_results = summary.get("seed_results")
    if not isinstance(seed_results, list) or not seed_results:
        reasons.append("seed_results must be a non-empty list")
    else:
        for index, result in enumerate(seed_results):
            if not isinstance(result, dict):
                reasons.append(f"seed_results[{index}] must be an object")
                continue
            if result.get("failed") is not False:
                reasons.append(f"seed_results[{index}].failed must be false")
            if not _finite(result.get("final_test_ce")):
                reasons.append(f"seed_results[{index}].final_test_ce must be finite")
    return reasons


def _discover_candidates(candidate_sets: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all candidate artifacts, de-duplicating by candidate id.

    The first path in deterministic sorted set/path order wins when an id is
    repeated across sets.  Duplicate locations are retained in the report so
    callers can detect accidental overlap without making the audit mutate or
    reject an otherwise usable pool.
    """

    records: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for root in sorted((Path(path).resolve() for path in candidate_sets), key=str):
        if not root.is_dir():
            raise FileNotFoundError(f"candidate set is not a directory: {root}")
        for spec_path in sorted(root.rglob("candidate_spec.json"), key=str):
            candidate_dir = spec_path.parent
            candidate_id = _candidate_id(_read_json(spec_path), candidate_dir)
            summary_path = candidate_dir / "results" / "summary.json"
            if not summary_path.is_file():
                record = {
                    "candidate_id": candidate_id,
                    "candidate_path": str(candidate_dir),
                    "model_type": None,
                    "spec": _read_json(spec_path),
                    "summary": None,
                    "valid": False,
                    "invalid_reasons": ["missing results/summary.json"],
                }
            else:
                spec = _read_json(spec_path)
                summary = _read_json(summary_path)
                reasons = _strict_summary_check(summary)
                record = {
                    "candidate_id": candidate_id,
                    "candidate_path": str(candidate_dir),
                    "model_type": _model_type(spec),
                    "spec": spec,
                    "summary": summary,
                    "valid": not reasons,
                    "invalid_reasons": reasons,
                }
            if candidate_id in records:
                duplicates.append(
                    {
                        "candidate_id": candidate_id,
                        "kept_path": records[candidate_id]["candidate_path"],
                        "duplicate_path": str(candidate_dir),
                    }
                )
                continue
            records[candidate_id] = record
    return list(records.values()), duplicates


def _can_select_question_count(
    question_count: int, mlp_wins: int, kan_wins: int, max_winner_fraction: float
) -> tuple[bool, str]:
    if question_count < 0:
        return False, "question_count must be non-negative"
    if not 0.0 <= max_winner_fraction <= 1.0:
        return False, "max_winner_fraction must be between 0 and 1"
    total = mlp_wins + kan_wins
    if question_count > total:
        return False, f"only {total} significant pairs available"
    if question_count == 0:
        return True, ""
    # Enumerate the possible number of selected MLP winners.  This avoids
    # rounding surprises at fractions such as 0.70 and handles either family
    # being the majority.
    for selected_mlp in range(max(0, question_count - kan_wins), min(question_count, mlp_wins) + 1):
        selected_kan = question_count - selected_mlp
        if (
            selected_mlp <= max_winner_fraction * question_count + 1e-12
            and selected_kan <= max_winner_fraction * question_count + 1e-12
        ):
            return True, ""
    return False, "winner-family fraction constraint cannot be satisfied"


def audit_xor_pair_capacity(
    candidate_sets: list[Path],
    profile: Profile,
    *,
    target_significant: int = 130,
    question_count: int = 100,
    max_winner_fraction: float = 0.70,
) -> dict[str, Any]:
    """Return a JSON-serializable MLP/KAN capacity audit report."""

    records, duplicates = _discover_candidates(candidate_sets)
    valid = [row for row in records if row["valid"]]
    mlps = [row for row in valid if row["model_type"] == "mlp"]
    kans = [row for row in valid if row["model_type"] == "kan"]
    pairs: list[dict[str, Any]] = []
    mlp_wins = 0
    kan_wins = 0
    for mlp in sorted(mlps, key=lambda row: row["candidate_id"]):
        for kan in sorted(kans, key=lambda row: row["candidate_id"]):
            result = validate_significance(
                [mlp["summary"], kan["summary"]], profile, metric="test_ce"
            )
            winner = None
            if result.passed:
                winner = "mlp" if result.winner_index == 0 else "kan"
                if winner == "mlp":
                    mlp_wins += 1
                else:
                    kan_wins += 1
            pairs.append(
                {
                    "candidate_ids": [mlp["candidate_id"], kan["candidate_id"]],
                    "mlp_candidate_id": mlp["candidate_id"],
                    "kan_candidate_id": kan["candidate_id"],
                    "gap": float(result.gap),
                    "win_rate": float(result.win_rate),
                    "significant": bool(result.passed),
                    "winner": winner,
                    "reason": result.reason,
                }
            )
    significant = mlp_wins + kan_wins
    can_select, capacity_reason = _can_select_question_count(
        question_count, mlp_wins, kan_wins, max_winner_fraction
    )
    invalid = [
        {
            "candidate_id": row["candidate_id"],
            "model_type": row["model_type"],
            "candidate_path": row["candidate_path"],
            "reasons": row["invalid_reasons"],
        }
        for row in records
        if not row["valid"]
    ]
    report = {
        "schema_version": "xor_pair_capacity_audit_v1",
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "candidate_sets": [str(Path(path).resolve()) for path in candidate_sets],
        "candidate_count": len(records),
        "valid_candidate_count": len(valid),
        "invalid_candidate_count": len(invalid),
        "invalid_candidates": invalid,
        "duplicate_candidate_ids": duplicates,
        "model_counts": {"mlp": len(mlps), "kan": len(kans)},
        "total_pairs": len(pairs),
        "significant_pairs": significant,
        "mlp_wins": mlp_wins,
        "kan_wins": kan_wins,
        "pairs": pairs,
        "target_significant": target_significant,
        "target_significant_reached": significant >= target_significant,
        "question_count": question_count,
        "max_winner_fraction": max_winner_fraction,
        "can_select_question_count": can_select,
        "question_capacity": {
            "question_count": question_count,
            "max_winner_fraction": max_winner_fraction,
            "can_select": can_select,
            "reason": capacity_reason,
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="v2.3-xor-pilot")
    parser.add_argument("--candidate-set", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-significant", type=int, default=130)
    parser.add_argument("--question-count", type=int, default=100)
    parser.add_argument("--max-winner-fraction", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_xor_pair_capacity(
        list(args.candidate_set),
        load_profile(args.profile),
        target_significant=args.target_significant,
        question_count=args.question_count,
        max_winner_fraction=args.max_winner_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total_pairs", "significant_pairs", "mlp_wins", "kan_wins", "target_significant_reached", "can_select_question_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
