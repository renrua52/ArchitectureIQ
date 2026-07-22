#!/usr/bin/env python3
"""Build a provenance-rich, globally candidate-disjoint demo collection.

Entries are supplied as ``RUN::TRACK`` in the desired order.  The tool reads
question-run manifests (rather than sorting question directories), verifies
profile provenance and rejects candidate reuse across the entire collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (ROOT / "data").resolve()


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _relative_data(path: Path) -> str:
    try:
        return path.resolve().relative_to(DATA_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside data root: {path}") from exc


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(entries: list[str], *, title: str) -> dict[str, Any]:
    if not entries:
        raise ValueError("At least one RUN::TRACK entry is required")
    question_paths: list[str] = []
    records: list[dict[str, Any]] = []
    source_runs: list[str] = []
    seen_questions: set[str] = set()
    seen_candidates: dict[str, str] = {}
    seen_runs: set[str] = set()

    for entry in entries:
        if "::" not in entry:
            raise ValueError(f"Entry must be RUN::TRACK, got {entry!r}")
        raw_run, track = entry.split("::", 1)
        if not track:
            raise ValueError(f"Track must be non-empty: {entry!r}")
        run_path = _resolve(raw_run)
        run = _read(run_path / "run.json")
        run_rel = _relative_data(run_path)
        if run_rel in seen_runs:
            raise ValueError(f"Run appears more than once: {run_rel}")
        seen_runs.add(run_rel)
        source_runs.append(run_rel)
        question_ids = run.get("question_ids")
        if not isinstance(question_ids, list) or not question_ids:
            raise ValueError(f"Run has no ordered question_ids: {run_path}")

        for question_id in question_ids:
            question_dir = run_path / str(question_id)
            question = _read(question_dir / "question.json")
            qid = str(question.get("question_id", question_id))
            q_rel = _relative_data(question_dir)
            if q_rel in seen_questions or qid in {r["question_id"] for r in records}:
                raise ValueError(f"Question appears more than once: {qid}")
            if qid != str(question_id):
                raise ValueError(f"Question directory/id mismatch: {question_dir}")
            profile = question.get("profile") or run.get("profile")
            profile_hash = question.get("profile_hash") or run.get("profile_hash")
            if not profile or not profile_hash:
                raise ValueError(f"Question lacks immutable profile provenance: {question_dir}")
            candidate_ids = [str(choice["candidate_id"]) for choice in question.get("choices", [])]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Duplicate candidate within question: {qid}")
            for candidate_id in candidate_ids:
                previous = seen_candidates.get(candidate_id)
                if previous is not None:
                    raise ValueError(
                        f"Global candidate reuse: {candidate_id} in {qid}; previously {previous}"
                    )
                seen_candidates[candidate_id] = qid
            seen_questions.add(q_rel)
            question_paths.append(q_rel)
            records.append(
                {
                    "order": len(records),
                    "question_id": qid,
                    "question_path": q_rel,
                    "family": question.get("family"),
                    "dataset_id": question.get("dataset_id"),
                    "question_type": question.get("type"),
                    "profile": profile,
                    "profile_hash": profile_hash,
                    "track": track,
                    "source_run": run_rel,
                    "candidate_ids": candidate_ids,
                }
            )

    identity = {
        "question_paths": question_paths,
        "records": records,
    }
    collection_id = "demo_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    profiles = sorted({str(record["profile"]) for record in records})
    tracks = sorted({str(record["track"]) for record in records})
    return {
        "schema_version": "demo_release_collection_v1",
        "collection_id": collection_id,
        "title": title,
        "candidate_reuse_policy": "globally_disjoint",
        "question_order_policy": "manifest_order",
        "question_count": len(records),
        "candidate_count": len(seen_candidates),
        "profiles": profiles,
        "tracks": tracks,
        "source_runs": source_runs,
        "question_paths": question_paths,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, help="RUN::TRACK, repeat in review order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="ArchitectureIQ demo release pilot")
    args = parser.parse_args()
    payload = build(args.entry, title=args.title)
    output = _resolve(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("collection_id", "question_count", "candidate_count", "profiles", "tracks")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
