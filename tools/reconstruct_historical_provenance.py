#!/usr/bin/env python3
"""Reconstruct immutable provenance for a legacy candidate set without mutating it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "historical_candidate_set_provenance_v1"
DEFAULT_FILENAME = "historical_provenance.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_hash(raw: dict[str, Any]) -> str:
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _git_show(repo: Path, revision: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=repo, text=True, encoding="utf-8"
    )


def reconstruct(candidate_set: Path, repo: Path, *, output: Path) -> dict[str, Any]:
    candidate_set = candidate_set.resolve()
    repo = repo.resolve()
    manifest = _read_json(candidate_set / "set.json")
    candidates = sorted(
        path
        for path in candidate_set.iterdir()
        if path.is_dir() and (path / "candidate_spec.json").is_file()
    )
    if not candidates:
        raise ValueError(f"No candidate specs found in {candidate_set}")
    summaries = [_read_json(path / "results" / "summary.json") for path in candidates]
    commits = {
        str(summary.get("environment", {}).get("git_commit", ""))
        for summary in summaries
    }
    if len(commits) != 1 or not next(iter(commits)):
        raise ValueError(f"Expected one non-empty GT git commit, found {sorted(commits)}")
    source_commit = next(iter(commits))
    profile_name = str(manifest["profile"])
    profile_path = f"profiles/{profile_name}.yaml"
    profile_text = _git_show(repo, source_commit, profile_path)
    raw_profile = yaml.safe_load(profile_text)
    if raw_profile.get("profile") != profile_name:
        raise ValueError(f"Historical {profile_path} does not declare profile {profile_name!r}")
    id_matches = sum(
        _read_json(path / "candidate_spec.json").get("candidate_id") == path.name
        for path in candidates
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "reconstructed_verified",
        "candidate_set": str(candidate_set),
        "set_id": manifest.get("set_id", candidate_set.name),
        "dataset_id": manifest.get("dataset_id"),
        "family": manifest.get("family"),
        "historical_profile": profile_name,
        "historical_profile_hash": _profile_hash(raw_profile),
        "historical_profile_path": profile_path,
        "historical_profile_yaml_sha256": hashlib.sha256(profile_text.encode("utf-8")).hexdigest(),
        "source_git_commit": source_commit,
        "candidate_count": len(candidates),
        "candidate_id_directory_match_count": id_matches,
        "summary_git_commits": sorted(commits),
        "verification": {
            "single_gt_commit": True,
            "historical_profile_found": True,
            "all_candidate_directory_ids_match": id_matches == len(candidates),
        },
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.candidate_set / DEFAULT_FILENAME
    try:
        payload = reconstruct(args.candidate_set, args.repo, output=output)
    except (FileExistsError, FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
