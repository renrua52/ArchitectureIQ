#!/usr/bin/env python3
"""Audit local demo candidate/question capacity without mutating benchmark artifacts.

The report is intentionally an inventory and an upper-bound estimate. It reads
candidate specs, GT summaries, existing Gate 1/2 audit JSON, and an optional
review collection; it does not regenerate candidates or questions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_PHASE_ROOT = ROOT / "outputs" / "question_expansion_phase"
DEFAULT_COLLECTION = DEFAULT_PHASE_ROOT / "review_collection_24.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "demo_release_integration"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_priority(status: str) -> int:
    return {"unknown": 0, "pass": 1, "review": 2, "fail": 3}.get(status, 0)


def _discover_gate12(phase_root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files: list[str] = []
    if not phase_root.exists():
        return by_candidate, files
    for path in sorted(phase_root.rglob("question_input_audit.json")):
        if not path.parent.name.startswith("gate12_"):
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        files.append(str(path))
        dataset_id = str(payload.get("dataset_id", ""))
        for candidate_set in payload.get("candidate_sets", []):
            for candidate in candidate_set.get("candidates", []):
                candidate_id = candidate.get("candidate_id")
                if not candidate_id:
                    continue
                by_candidate[candidate_id].append(
                    {
                        "status": str(candidate.get("status", "unknown")),
                        "dataset_id": dataset_id,
                        "profile": payload.get("profile"),
                        "profile_hash": payload.get("profile_hash"),
                        "audit_path": str(path),
                    }
                )
    return by_candidate, files



def _final_gate_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "unknown"
    statuses = {str(item.get("status", "unknown")) for item in records}
    # A review snapshot is more informative than an older aggregate failure;
    # keep a hard fail only when no review evidence exists for that candidate.
    if "review" in statuses:
        return "review"
    if "fail" in statuses:
        return "fail"
    if "pass" in statuses:
        return "pass"
    return "unknown"


def _load_collection(path: Path, data_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "question_count": 0,
            "candidate_ids": [],
            "candidate_overlap": {},
            "missing_questions": [],
        }
    payload = _read_json(path)
    candidate_ids: list[str] = []
    missing_questions: list[str] = []
    question_count = 0
    if isinstance(payload.get("question_paths"), list):
        for value in payload["question_paths"]:
            question_path = data_root / str(value)
            if not (question_path / "question.json").exists():
                missing_questions.append(str(question_path))
                continue
            question_count += 1
            question = _read_json(question_path / "question.json")
            candidate_ids.extend(
                str(choice["candidate_id"])
                for choice in question.get("choices", [])
                if isinstance(choice, dict) and choice.get("candidate_id")
            )
    elif isinstance(payload.get("questions"), list):
        for record in payload["questions"]:
            question_count += 1
            candidate_ids.extend(str(value) for value in record.get("candidate_ids", []) if value)
    counts = Counter(candidate_ids)
    return {
        "path": str(path),
        "exists": True,
        "schema_version": payload.get("schema_version"),
        "question_count": question_count,
        "candidate_ids": sorted(counts),
        "candidate_id_count": len(counts),
        "candidate_overlap": {key: count for key, count in counts.items() if count > 1},
        "missing_questions": missing_questions,
        "declared_candidate_reuse_policy": payload.get("candidate_reuse_policy"),
        "declared_collection_id": payload.get("collection_id"),
        "declared_source_runs": payload.get("source_runs", []),
    }


def scan(data_root: Path, phase_root: Path, collection_path: Path | None) -> dict[str, Any]:
    gate_by_candidate, gate_files = _discover_gate12(phase_root)
    candidates: list[dict[str, Any]] = []
    sets: list[dict[str, Any]] = []
    candidate_locations: dict[str, list[str]] = defaultdict(list)

    datasets_root = data_root / "datasets"
    if datasets_root.exists():
        for family_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
            for dataset_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                candidates_root = dataset_dir / "candidates"
                if not candidates_root.exists():
                    continue
                for set_dir in sorted(p for p in candidates_root.iterdir() if p.is_dir()):
                    set_manifest_path = set_dir / "set.json"
                    set_manifest = _read_json(set_manifest_path) if set_manifest_path.exists() else {}
                    set_candidates = []
                    for candidate_dir in sorted(p for p in set_dir.iterdir() if p.is_dir() and p.name.startswith("c_")):
                        spec_path = candidate_dir / "candidate_spec.json"
                        summary_path = candidate_dir / "results" / "summary.json"
                        spec = _read_json(spec_path) if spec_path.exists() else {}
                        summary = _read_json(summary_path) if summary_path.exists() else {}
                        candidate_id = str(spec.get("candidate_id") or candidate_dir.name)
                        gate_records = gate_by_candidate.get(candidate_id, [])
                        gate_status = _final_gate_status(gate_records)
                        profile_hash = spec.get("profile_hash")
                        excluded = bool(summary.get("excluded", False)) if summary else True
                        if gate_status == "pass" and profile_hash and not excluded:
                            capacity_status = "pass"
                        elif gate_status in {"fail", "review"} or excluded:
                            capacity_status = "fail" if gate_status == "fail" or excluded else "review"
                        else:
                            capacity_status = "unknown"
                        record = {
                            "candidate_id": candidate_id,
                            "family": family_dir.name,
                            "dataset_id": dataset_dir.name,
                            "set_id": set_dir.name,
                            "profile": spec.get("profile"),
                            "profile_hash": profile_hash,
                            "gate12_status": gate_status,
                            "capacity_status": capacity_status,
                            "excluded": excluded,
                            "candidate_path": str(candidate_dir),
                        }
                        candidates.append(record)
                        set_candidates.append(candidate_id)
                        candidate_locations[candidate_id].append(str(candidate_dir))
                    sets.append(
                        {
                            "family": family_dir.name,
                            "dataset_id": dataset_dir.name,
                            "set_id": set_dir.name,
                            "manifest": bool(set_manifest),
                            "declared_count": set_manifest.get("count"),
                            "actual_candidates": len(set_candidates),
                            "profile": set_manifest.get("profile"),
                            "profile_hash": set_manifest.get("profile_hash"),
                            "candidate_ids": set_candidates,
                        }
                    )

    collection = _load_collection(collection_path, data_root) if collection_path else None
    reserved = set(collection.get("candidate_ids", [])) if collection else set()
    by_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["capacity_status"] == "pass":
            by_dataset[(candidate["family"], candidate["dataset_id"])].append(candidate)
    capacity_rows = []
    for (family, dataset_id), rows in sorted(by_dataset.items()):
        available = [row for row in rows if row["candidate_id"] not in reserved]
        capacity_rows.append(
            {
                "family": family,
                "dataset_id": dataset_id,
                "formal_pass_candidates": len(rows),
                "reserved_by_collection": len(rows) - len(available),
                "remaining_candidates": len(available),
                "two_choice_question_upper_bound": len(available) // 2,
            }
        )

    profile_counts = Counter((str(row["profile"]), str(row["profile_hash"])) for row in candidates)
    gate_counts = Counter(row["gate12_status"] for row in candidates)
    capacity_counts = Counter(row["capacity_status"] for row in candidates)
    overlap = {candidate_id: locations for candidate_id, locations in candidate_locations.items() if len(locations) > 1}
    return {
        "schema_version": "demo_capacity_report_v1",
        "data_root": str(data_root),
        "phase_root": str(phase_root),
        "gate12": {
            "audit_file_count": len(gate_files),
            "audit_files": gate_files,
            "candidate_status_counts": dict(gate_counts),
        },
        "collection": collection,
        "totals": {
            "candidate_dirs": len(candidates),
            "unique_candidate_ids": len(candidate_locations),
            "set_count": len(sets),
            "dataset_count": len({(row["family"], row["dataset_id"]) for row in candidates}),
            "profile_hash_counts": {
                f"{profile}|{profile_hash}": count
                for (profile, profile_hash), count in sorted(profile_counts.items())
            },
            "gate12_status_counts": dict(gate_counts),
            "capacity_status_counts": dict(capacity_counts),
        },
        "global_candidate_overlap": {
            "overlapping_candidate_id_count": len(overlap),
            "overlaps": overlap,
        },
        "sets": sets,
        "candidates": candidates,
        "capacity": {
            "reserved_candidate_count": len(reserved),
            "formal_pass_candidate_count": sum(1 for row in candidates if row["capacity_status"] == "pass"),
            "remaining_formal_pass_candidate_count": sum(row["remaining_candidates"] for row in capacity_rows),
            "two_choice_question_upper_bound": sum(row["two_choice_question_upper_bound"] for row in capacity_rows),
            "by_dataset": capacity_rows,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    capacity = report["capacity"]
    lines = [
        "# Demo Release Integration Capacity Report",
        "",
        f"- data root: {report['data_root']}",
        f"- collection: {(report.get('collection') or {}).get('path', 'none')}",
        f"- candidate dirs / unique IDs: {totals['candidate_dirs']} / {totals['unique_candidate_ids']}",
        f"- sets / datasets: {totals['set_count']} / {totals['dataset_count']}",
        f"- formal Gate12-pass candidates: {capacity['formal_pass_candidate_count']}",
        f"- collection-reserved candidate IDs: {capacity['reserved_candidate_count']}",
        f"- remaining formal candidates: {capacity['remaining_formal_pass_candidate_count']}",
        f"- raw 2-choice question upper bound: {capacity['two_choice_question_upper_bound']}",
        "",
        "## Profile and Gate12 counts",
        "",
        "| profile/hash | candidates |",
        "|---|---:|",
    ]
    for key, count in totals["profile_hash_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "| Gate12 status | candidates |",
            "|---|---:|",
        ]
    )
    for key, count in sorted(totals["gate12_status_counts"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "## Capacity by dataset",
            "",
            "| family | dataset | formal pass | reserved | remaining | 2-choice upper bound |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in capacity["by_dataset"]:
        lines.append(
            f"| {row['family']} | {row['dataset_id']} | {row['formal_pass_candidates']} | "
            f"{row['reserved_by_collection']} | {row['remaining_candidates']} | "
            f"{row['two_choice_question_upper_bound']} |"
        )
    overlap = report["global_candidate_overlap"]
    lines.extend(
        [
            "",
            "## Global candidate-disjoint check",
            "",
            f"- overlapping candidate IDs: {overlap['overlapping_candidate_id_count']}",
            f"- collection-declared reuse policy: {(report.get('collection') or {}).get('declared_candidate_reuse_policy')}",
            f"- collection-internal repeated candidate IDs: {len((report.get('collection') or {}).get('candidate_overlap', {}))}",
            "",
            "The question upper bound is a raw two-choice pairing bound after reserving collection IDs. "
            "It does not prove significance, compatibility, or a complete question run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = scan(args.data_root.resolve(), args.phase_root.resolve(), args.collection.resolve() if args.collection else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "capacity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "capacity_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"candidate_dirs": report["totals"]["candidate_dirs"], "formal_pass": report["capacity"]["formal_pass_candidate_count"], "remaining": report["capacity"]["remaining_formal_pass_candidate_count"], "question_upper_bound": report["capacity"]["two_choice_question_upper_bound"], "overlaps": report["global_candidate_overlap"]["overlapping_candidate_id_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
