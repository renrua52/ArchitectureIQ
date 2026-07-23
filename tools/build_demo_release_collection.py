#!/usr/bin/env python3
"""Build a provenance-rich, globally candidate-disjoint demo collection.

Use either repeated ``RUN::TRACK`` entries for exploratory builds, or a tracked
release spec plus an explicit external data root for a reproducible release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DATA_ROOT = DEFAULT_DATA_ROOT


def _resolve(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _resolve_run(value: str, *, data_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (data_root / path).resolve()


def _relative_data(path: Path, *, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside data root: {path}") from exc


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_entry(value: str) -> dict[str, Any]:
    if "::" not in value:
        raise ValueError(f"Entry must be RUN::TRACK, got {value!r}")
    source_run, track = value.split("::", 1)
    if not track:
        raise ValueError(f"Track must be non-empty: {value!r}")
    return {"source_run": source_run, "track": track}


def _load_release_spec(path: Path) -> dict[str, Any]:
    spec = _read(path)
    if not isinstance(spec, dict):
        raise ValueError("Release spec must be a JSON object")
    if spec.get("schema_version") != "demo_release_spec_v1":
        raise ValueError(f"Unsupported release spec schema: {spec.get('schema_version')!r}")
    entries = spec.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Release spec must contain non-empty entries")
    if "expected" in spec and not isinstance(spec["expected"], dict):
        raise ValueError("Release spec expected must be an object")
    return spec


def _expected_question_ids(entry: dict[str, Any], available: list[str], run_path: Path) -> list[str]:
    requested = entry.get("question_ids", available)
    if not isinstance(requested, list) or not requested or not all(isinstance(item, str) for item in requested):
        raise ValueError(f"question_ids must be a non-empty string list: {run_path}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Release spec repeats a question id: {run_path}")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Release spec names questions absent from {run_path}: {unknown}")
    return requested


def build(
    entries: list[str | dict[str, Any]],
    *,
    title: str,
    data_root: Path | None = None,
    release_spec: bool = False,
) -> dict[str, Any]:
    data_root = (data_root or DEFAULT_DATA_ROOT).resolve()
    if not entries:
        raise ValueError("At least one RUN::TRACK entry is required")
    question_paths: list[str] = []
    records: list[dict[str, Any]] = []
    source_runs: list[str] = []
    seen_questions: set[str] = set()
    seen_candidates: dict[str, str] = {}
    seen_runs: set[str] = set()

    for raw_entry in entries:
        if isinstance(raw_entry, str):
            entry = _parse_entry(raw_entry)
        elif isinstance(raw_entry, dict):
            entry = raw_entry
        else:
            raise ValueError(f"Release entry must be a string or object: {raw_entry!r}")
        raw_run = entry.get("source_run")
        track = entry.get("track")
        if not isinstance(raw_run, str) or not raw_run:
            raise ValueError(f"Release entry lacks source_run: {entry!r}")
        if not isinstance(track, str) or not track:
            raise ValueError(f"Release entry lacks track: {entry!r}")
        raw_run_path = Path(raw_run).expanduser()
        if release_spec:
            if raw_run_path.is_absolute():
                raise ValueError(f"Release spec source_run must be relative to data_root: {raw_run}")
            run_path = (data_root / raw_run_path).resolve()
            try:
                run_path.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(
                    f"Release spec source_run must stay inside data_root: {raw_run}"
                ) from exc
        else:
            run_path = _resolve_run(raw_run, data_root=data_root)
        run = _read(run_path / "run.json")
        run_rel = _relative_data(run_path, data_root=data_root)
        if run_rel in seen_runs:
            raise ValueError(f"Run appears more than once: {run_rel}")
        seen_runs.add(run_rel)
        source_runs.append(run_rel)
        available_ids = run.get("question_ids")
        if not isinstance(available_ids, list) or not available_ids:
            raise ValueError(f"Run has no ordered question_ids: {run_path}")
        question_ids = _expected_question_ids(entry, [str(item) for item in available_ids], run_path)

        for question_id in question_ids:
            question_dir = run_path / question_id
            question = _read(question_dir / "question.json")
            qid = str(question.get("question_id", question_id))
            q_rel = _relative_data(question_dir, data_root=data_root)
            if q_rel in seen_questions or qid in {record["question_id"] for record in records}:
                raise ValueError(f"Question appears more than once: {qid}")
            if qid != question_id:
                raise ValueError(f"Question directory/id mismatch: {question_dir}")
            if question.get("profile") and run.get("profile") and question["profile"] != run["profile"]:
                raise ValueError(f"Question/run profile mismatch: {question_dir}")
            if question.get("profile_hash") and run.get("profile_hash") and question["profile_hash"] != run["profile_hash"]:
                raise ValueError(f"Question/run profile hash mismatch: {question_dir}")
            profile = question.get("profile") or run.get("profile")
            profile_hash = question.get("profile_hash") or run.get("profile_hash")
            if not profile or not profile_hash:
                raise ValueError(f"Question lacks immutable profile provenance: {question_dir}")
            if entry.get("profile") and entry["profile"] != profile:
                raise ValueError(f"Profile mismatch for {question_dir}: {profile!r}")
            if entry.get("profile_hash") and entry["profile_hash"] != profile_hash:
                raise ValueError(f"Profile hash mismatch for {question_dir}: {profile_hash!r}")
            candidate_ids = [str(choice["candidate_id"]) for choice in question.get("choices", [])]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"Duplicate candidate within question: {qid}")
            for candidate_id in candidate_ids:
                previous = seen_candidates.get(candidate_id)
                if previous is not None:
                    raise ValueError(f"Global candidate reuse: {candidate_id} in {qid}; previously {previous}")
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

    identity = {"question_paths": question_paths, "records": records}
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


def _validate_expected(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("collection_id", "question_count", "candidate_count", "profiles", "tracks", "candidate_reuse_policy", "question_order_policy"):
        if key in expected and payload.get(key) != expected[key]:
            raise ValueError(
                f"Release spec expected {key}={expected[key]!r}, got {payload.get(key)!r}"
            )
    if "profile_hashes" in expected:
        actual_hashes: dict[str, str] = {}
        for profile in payload["profiles"]:
            hashes = {record["profile_hash"] for record in payload["records"] if record["profile"] == profile}
            if len(hashes) != 1:
                raise ValueError(f"Release collection has mixed profile hashes for {profile!r}: {sorted(hashes)}")
            actual_hashes[profile] = hashes.pop()
        if actual_hashes != expected["profile_hashes"]:
            raise ValueError(
                f"Release spec expected profile_hashes={expected['profile_hashes']!r}, got {actual_hashes!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--entry", action="append", help="RUN::TRACK, repeat in review order")
    source.add_argument("--spec", type=Path, help="Tracked demo_release_spec_v1 JSON")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="External or local data root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="ArchitectureIQ demo release pilot")
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    if args.spec:
        spec = _load_release_spec(_resolve(args.spec))
        entries = spec["entries"]
        title = str(spec.get("title", args.title))
        payload = build(entries, title=title, data_root=data_root, release_spec=True)
        _validate_expected(payload, spec.get("expected", {}))
    else:
        payload = build([_parse_entry(value) for value in args.entry], title=args.title, data_root=data_root)

    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.spec:
        expected_hash = spec.get("expected", {}).get("collection_sha256")
        actual_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"Release spec expected collection_sha256={expected_hash!r}, got {actual_hash!r}")
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized.encode("utf-8"))
    print(json.dumps({key: payload[key] for key in ("collection_id", "question_count", "candidate_count", "profiles", "tracks")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
