#!/usr/bin/env python3
"""Bake a collection manifest into the remote React quiz BakeFile schema.

This adapter deliberately reuses the additive ``frontend/quiz`` exporter and
does not alter the existing Streamlit/static Inspector exporter.  Collection
order and provenance are treated as immutable input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_quiz_static import bake_question


ROOT = Path(__file__).resolve().parents[1]




def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def bake_collection(data_root: Path, collection_path: Path) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if not isinstance(collection, dict):
        raise ValueError("collection must be a JSON object")
    raw_paths = collection.get("question_paths")
    records = collection.get("records")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("collection must contain ordered question_paths")
    metadata = [item if isinstance(item, dict) else {} for item in (records or [])]
    if metadata and len(metadata) != len(raw_paths):
        raise ValueError("collection records must align one-to-one with question_paths")
    data_root = data_root.resolve()
    baked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_path in enumerate(raw_paths):
        question_dir = _resolve(data_root, str(raw_path))
        if not question_dir.is_relative_to(data_root) or not (question_dir / "question.json").is_file():
            raise ValueError(f"question path is outside data root or missing: {question_dir}")
        q = json.loads((question_dir / "question.json").read_text(encoding="utf-8"))
        run = json.loads((question_dir.parent / "run.json").read_text(encoding="utf-8"))
        qid = str(q.get("question_id", question_dir.name))
        if qid in seen:
            raise ValueError(f"duplicate question id: {qid}")
        seen.add(qid)
        record = metadata[index] if metadata else {}
        provenance = {
            "profile": record.get("profile") or q.get("profile") or run.get("profile"),
            "profile_hash": record.get("profile_hash") or q.get("profile_hash") or run.get("profile_hash"),
            "track": record.get("track", "default"),
            "source_run": record.get("source_run") or q.get("question_run_path"),
            "collection_id": collection.get("collection_id"),
            "order": index,
        }
        item = bake_question(question_dir, data_root)
        item["profileHash"] = provenance["profile_hash"]
        item["track"] = provenance["track"]
        item["sourceRun"] = provenance["source_run"]
        item["provenance"] = provenance
        item["summary"].update(
            {
                "profile": provenance["profile"],
                "profileHash": provenance["profile_hash"],
                "track": provenance["track"],
                "order": index,
            }
        )
        baked.append(item)
    return {
        "schema_version": 1,
        "ordered": True,
        "collection": collection,
        "questions": [item["summary"] for item in baked],
        "byId": {item["id"]: item for item in baked},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = bake_collection(args.data_root, args.collection.resolve())
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"questions": len(payload["questions"]), "ordered": payload["ordered"], "collection_id": payload["collection"].get("collection_id")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
