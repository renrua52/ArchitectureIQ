#!/usr/bin/env python3
"""Independent Gate 3/4 and frontend-order audit for a demo collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from architecture_iq.profile import load_profile
from question_audit_lib import audit_question_run


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection_path = args.collection.resolve()
    collection: dict[str, Any] = json.loads(collection_path.read_text(encoding="utf-8"))
    paths = collection.get("question_paths", [])
    records = collection.get("records", [])
    if len(paths) != len(records):
        raise SystemExit("collection question_paths/records length mismatch")
    candidate_owner: dict[str, str] = {}
    question_ids: list[str] = []
    failures: list[str] = []
    runs: dict[str, str] = {}
    for index, (raw_path, record) in enumerate(zip(paths, records, strict=True)):
        qdir = (DATA_ROOT / str(raw_path)).resolve()
        question = json.loads((qdir / "question.json").read_text(encoding="utf-8"))
        qid = str(question.get("question_id"))
        question_ids.append(qid)
        if record.get("order") != index or record.get("question_id") != qid:
            failures.append(f"order/id mismatch at index {index}: {qid}")
        for choice in question.get("choices", []):
            cid = str(choice.get("candidate_id"))
            owner = candidate_owner.get(cid)
            if owner is not None:
                failures.append(f"candidate reuse {cid}: {owner} and {qid}")
            candidate_owner[cid] = qid
        run_path = qdir.parent
        run_key = str(run_path)
        runs[run_key] = str(record.get("profile") or question.get("profile") or "v1")

    run_reports = []
    for run_path, profile_name in sorted(runs.items()):
        report = audit_question_run(Path(run_path), load_profile(profile_name))
        run_reports.append({
            "run": run_path,
            "profile": profile_name,
            "valid": report["valid"],
            "summary": report["summary"],
        })
        if not report["valid"]:
            failures.append(f"Gate 3/4 failed: {run_path}")

    bake_path = ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"
    bake = json.loads(bake_path.read_text(encoding="utf-8"))
    baked_ids = [str(item.get("id")) for item in bake.get("questions", [])]
    if baked_ids != question_ids:
        failures.append("React BakeFile question order differs from collection manifest")
    if not bake.get("ordered"):
        failures.append("React BakeFile is not marked ordered")

    result = {
        "schema_version": "demo_collection_audit_v1",
        "collection_id": collection.get("collection_id"),
        "question_count": len(question_ids),
        "candidate_count": len(candidate_owner),
        "candidate_reuse_count": len(failures) - sum("Gate 3/4" in f or "order" in f or "BakeFile" in f for f in failures),
        "runs": run_reports,
        "frontend_order_match": baked_ids == question_ids,
        "valid": not failures,
        "failures": failures,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("collection_id", "question_count", "candidate_count", "frontend_order_match", "valid")}, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
