#!/usr/bin/env python3
"""Write a deterministic release-freeze manifest for the current demo pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    """Return a portable, repository-relative POSIX path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Release manifest path must be inside the repository: {resolved}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-spec", type=Path, help="Tracked release spec used to rebuild the collection")
    parser.add_argument("--data-bundle", type=Path, help="External data-root bundle used for the release")
    args = parser.parse_args()
    collection_path = args.collection.resolve()
    collection: dict[str, Any] = json.loads(collection_path.read_text(encoding="utf-8"))
    profiles = {}
    for name in collection.get("profiles", []):
        profile_path = ROOT / "profiles" / f"{name}.yaml"
        profiles[name] = {"path": repo_relative(profile_path), "sha256": digest(profile_path)}
    payload = {
        "schema_version": "demo_release_freeze_v2",
        "path_base": "repository_root",
        "collection_id": collection.get("collection_id"),
        "collection_path": repo_relative(collection_path),
        "collection_sha256": digest(collection_path),
        "question_count": collection.get("question_count"),
        "candidate_count": collection.get("candidate_count"),
        "candidate_reuse_policy": collection.get("candidate_reuse_policy"),
        "question_order_policy": collection.get("question_order_policy"),
        "profiles": profiles,
        "frontend_bakefile": {
            "path": repo_relative(ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"),
            "sha256": digest(ROOT / "frontend" / "quiz" / "public" / "data" / "questions.json"),
        },
        "public_private_boundary": {
            "public_prompt": "question.detail.prompt",
            "public_candidate_code": "question.detail.choices[].files",
            "private_answer": "question.reveal; API returns only after POST answer",
        },
        "performance_reports": [
            "outputs/demo_release_integration/execution_benchmark_smoke/report.json",
            "outputs/demo_release_integration/performance_cpu_threads/report.json",
        ],
        "human_audit_status": "pending_user_blind_review",
        "luna_audit_status": "pending_after_human_freeze",
    }
    if args.release_spec:
        release_spec = args.release_spec.resolve()
        payload["release_spec"] = {"path": repo_relative(release_spec), "sha256": digest(release_spec)}
    if args.data_bundle:
        data_bundle = args.data_bundle.resolve()
        payload["external_data_bundle"] = {
            "path": repo_relative(data_bundle),
            "sha256": digest(data_bundle),
            "layout": "data_root",
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("collection_id", "question_count", "candidate_count", "human_audit_status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
