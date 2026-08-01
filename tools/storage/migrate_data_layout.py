"""Migrate the legacy layout ``data/datasets/{family}/{dataset_id}/`` to the
columnar backend storage ``backend/data/{problems,trainers,candidates,results}``.

Usage:
    python tools/storage/migrate_data_layout.py                # copy into backend/data/
    python tools/storage/migrate_data_layout.py --mode dry-run # preview only
    python tools/storage/migrate_data_layout.py --mode move    # copy, then delete sources
    python tools/storage/migrate_data_layout.py --families multivariate_regression
    python tools/storage/migrate_data_layout.py --limit 3

Idempotent: reruns overwrite the same outputs safely (copy mode).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from architecture_iq.storage import repository as repo
from architecture_iq.storage import schema as sc
from architecture_iq.util import read_json, write_json

LEGACY_DEFAULT = Path("data") / "datasets"
NEW_DEFAULT = repo.data_root()


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "dry-run":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_json(path: Path, data: dict, mode: str) -> None:
    if mode == "dry-run":
        return
    write_json(path, data)


def _problem_readme(problem_id: str, spec: dict, src: Path) -> str:
    sig = spec.get("significance", {})
    sig_str = ", ".join(f"{k}={v}" for k, v in sig.items()) or "-"
    files = ", ".join(spec.get("files", {}).values()) or "-"
    return (
        f"# Problem {problem_id}\n"
        f"\n"
        f"- **family**: {spec.get('family', '-')}\n"
        f"- **dataset_id**: {spec.get('dataset_id', problem_id)}\n"
        f"- **selection_metric**: {spec.get('selection_metric', '-')}\n"
        f"- **significance**: {sig_str}\n"
        f"- **files**: {files}\n"
        f"- **source**: migrated from `{src}` (2026-07-31)\n"
    )


def _candidate_config(problem_id: str, spec: dict) -> dict:
    return {
        "schema_version": sc.CANDIDATE_SCHEMA_VERSION,
        "problem_id": problem_id,
        "candidate_id": spec["candidate_id"],
        "family": spec["family"],
        "budget": spec["budget"],
        "model": spec["model"],
        "optimizer": spec["optimizer"],
        "loss": spec["loss"],
    }


def migrate(src_root: Path, dest_root: Path, mode: str, families: list[str] | None, limit: int | None) -> dict:
    repo.DATA_ROOT = dest_root  # route all repository path helpers to the destination
    counts = {
        "problems": 0,
        "candidates": 0,
        "results": 0,
        "trainers": 0,
        "no_gt_candidates": 0,
        "missing_spec": 0,
        "skipped_questions": 0,
    }
    trainer_contents: dict[str, list[tuple[str, str]]] = {}  # family -> [(sha, content)]

    dataset_dirs = sorted(p for p in src_root.glob("*/*") if p.is_dir() and p.name != "__pycache__")
    if families:
        dataset_dirs = [p for p in dataset_dirs if p.parent.name in families]
    if limit:
        dataset_dirs = dataset_dirs[:limit]

    for dataset_dir in dataset_dirs:
        family = dataset_dir.parent.name
        spec_path = dataset_dir / sc.PROBLEM_SPEC_JSON
        if not spec_path.is_file():
            counts["missing_spec"] += 1
            continue
        spec = read_json(spec_path)
        problem_id = spec.get("dataset_id", dataset_dir.name)

        # --- problems -------------------------------------------------
        problem_spec = dict(spec)
        problem_spec["schema_version"] = sc.PROBLEM_SCHEMA_VERSION
        problem_spec["problem_id"] = problem_id
        _write_json(repo.problem_spec_path(problem_id), problem_spec, mode)
        _write_json(repo.problem_readme_path(problem_id), _problem_readme(problem_id, spec, dataset_dir), mode)
        for fname in spec.get("files", {}).values():
            fsrc = dataset_dir / fname
            if fsrc.is_file():
                _copy(fsrc, repo.problem_dir(problem_id) / fname, mode)
        counts["problems"] += 1

        # --- candidates + results -------------------------------------
        for candidate_spec_path in sorted(dataset_dir.glob("candidates/*/*/candidate_spec.json")):
            cand = read_json(candidate_spec_path)
            cand_dir = candidate_spec_path.parent
            candidate_id = cand.get("candidate_id", cand_dir.name)
            _write_json(
                repo.candidate_config_path(problem_id, candidate_id),
                _candidate_config(problem_id, cand),
                mode,
            )
            counts["candidates"] += 1

            res_dir = cand_dir / "results"
            if (res_dir / sc.SUMMARY_JSON).is_file():
                _copy(res_dir / sc.SUMMARY_JSON, repo.summary_path(problem_id, candidate_id), mode)
                if (res_dir / sc.CURVES_NPZ).is_file():
                    _copy(res_dir / sc.CURVES_NPZ, repo.curves_path(problem_id, candidate_id), mode)
                counts["results"] += 1
            else:
                counts["no_gt_candidates"] += 1

            # collect trainer templates by family
            train_py = cand_dir / "train.py"
            if train_py.is_file():
                text = train_py.read_text(encoding="utf-8")
                digest = _content_sha256(text)
                if not any(d == digest for d, _ in trainer_contents.get(family, [])):
                    trainer_contents.setdefault(family, []).append((digest, text))

        # --- questions are eval-side: not migrated --------------------
        qdir = dataset_dir / "questions"
        if qdir.is_dir():
            counts["skipped_questions"] += 1

        if mode == "move" and counts["problems"] > 0:
            shutil.rmtree(dataset_dir)

    # --- trainers ------------------------------------------------------
    for family in sorted(trainer_contents):
        for version, (digest, content) in enumerate(trainer_contents[family], start=1):
            trainer_id = f"{family}_v{version}"
            spec = {
                "schema_version": sc.TRAINER_SCHEMA_VERSION,
                "trainer_id": trainer_id,
                "family": family,
                "version": f"v{version}",
                "content_sha256": digest,
                "source": "migrated from legacy data/datasets",
            }
            if mode != "dry-run":
                repo.write_trainer(trainer_id, spec, content)
            counts["trainers"] += 1

    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=LEGACY_DEFAULT)
    ap.add_argument("--dest", type=Path, default=NEW_DEFAULT)
    ap.add_argument("--mode", choices=["copy", "move", "dry-run"], default="copy")
    ap.add_argument("--families", help="comma-separated family names")
    ap.add_argument("--limit", type=int, help="migrate only the first N datasets")
    args = ap.parse_args(argv)

    families = [f.strip() for f in args.families.split(",")] if args.families else None
    if not args.src.is_dir():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 1

    counts = migrate(args.src, args.dest, args.mode, families, args.limit)
    verb = "would" if args.mode == "dry-run" else "did"
    print(f"[{args.mode}] {verb} migrate {counts['problems']} problems / "
          f"{counts['candidates']} candidates / {counts['results']} results / "
          f"{counts['trainers']} trainers into {args.dest}")
    print(f"[{args.mode}] {counts['no_gt_candidates']} candidates without GT, "
          f"{counts['missing_spec']} datasets without spec, "
          f"{counts['skipped_questions']} question dirs left for eval side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
