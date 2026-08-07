#!/usr/bin/env python3
"""Pack / upload / download the V1 LLM+human benchmark raw bundle via GitHub Releases.

Raw payloads stay out of git. The tracked manifest under
``benchmark_releases/v1_llm_bundle/manifest.json`` records the release tag and
SHA-256 digests. After download+extract, paths match this worktree layout::

    benchmarks/v1_llm/          # questions, llm_runs, reports
    benchmarks/v1_human/        # manifest + BakeFile (+ recreated bake_root)
    data/datasets/...           # full dataset instances + candidates + GT

Colleague usage (needs ``gh`` auth to this private repo)::

    gh auth login   # once
    python tools/benchmark_v1_bundle.py download

Maintainer::

    python tools/benchmark_v1_bundle.py pack
    python tools/benchmark_v1_bundle.py upload --write-tracked-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmark_releases" / "v1_llm_bundle" / "manifest.json"
DEFAULT_STAGING = ROOT / "artifacts" / "v1_llm_bundle"
BUNDLE_SCHEMA = "architectureiq_v1_llm_bundle_v1"
DEFAULT_RELEASE_TAG = "v1-llm-bundle"
DEFAULT_REPO = "renrua52/ArchitectureIQ"

SKIP_NAME_RE = re.compile(
    r"(^|/)\.DS_Store$|(^|/)__pycache__(/|$)|(^|/)\.git(/|$)|\.pyc$"
)

PARTS = (
    {
        "id": "v1_llm",
        "archive": "v1_llm.tar.gz",
        "description": "1000-question LLM pack: questions/, llm_runs/, reports",
    },
    {
        "id": "v1_human",
        "archive": "v1_human.tar.gz",
        "description": "250-question human pre-review pack (manifest + BakeFile)",
    },
    {
        "id": "datasets",
        "archive": "datasets.tar.gz",
        "description": "data/datasets roots (full candidates + GT) referenced by V1 questions",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _should_skip(arcname: str) -> bool:
    return bool(SKIP_NAME_RE.search(arcname.replace("\\", "/")))


def _add_path(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    if _should_skip(arcname):
        return
    if src.is_symlink():
        # Skip absolute bake_root-style links; real trees are packed by value.
        return
    if src.is_dir():
        info = tarfile.TarInfo(arcname.replace("\\", "/"))
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        tar.addfile(info)
        for child in sorted(src.iterdir(), key=lambda p: p.name):
            _add_path(tar, child, f"{arcname}/{child.name}")
        return
    if src.is_file():
        tar.add(src, arcname=arcname.replace("\\", "/"), recursive=False)


def _referenced_dataset_roots() -> list[Path]:
    refs: set[str] = set()
    qroot = ROOT / "benchmarks" / "v1_llm" / "questions"
    for path in sorted(qroot.glob("q_*/question.json")):
        q = json.loads(path.read_text(encoding="utf-8"))
        for choice in q.get("choices") or []:
            cp = str(choice.get("candidate_path") or "")
            parts = cp.split("/")
            if len(parts) >= 3 and parts[0] == "datasets":
                refs.add("/".join(parts[:3]))
        for cs in q.get("candidate_sets") or []:
            parts = str(cs).split("/")
            if len(parts) >= 3 and parts[0] == "datasets":
                refs.add("/".join(parts[:3]))
    roots = []
    for rel in sorted(refs):
        p = ROOT / "data" / rel
        if not p.is_dir():
            raise FileNotFoundError(f"Referenced dataset root missing: {rel}")
        roots.append(p)
    return roots


def _write_tar(archive: Path, members: Iterable[tuple[Path, str]]) -> dict[str, Any]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        for src, arcname in members:
            _add_path(tar, src, arcname)
    digest = _sha256_file(archive)
    return {
        "path": str(archive.relative_to(ROOT)).replace("\\", "/"),
        "filename": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": digest,
    }


def _run_gh(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    gh = shutil.which("gh")
    if not gh:
        raise SystemExit(
            "GitHub CLI `gh` not found. Install: https://cli.github.com/ "
            "then run `gh auth login`."
        )
    proc = subprocess.run(
        [gh, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"gh {' '.join(args)} failed:\n{msg}")
    return proc


def _repo_slug(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("ARCHITECTUREIQ_GITHUB_REPO", "").strip()
    if env:
        return env
    proc = _run_gh(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return DEFAULT_REPO


def pack_bundle(staging: Path) -> dict[str, Any]:
    staging.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}

    v1_llm = ROOT / "benchmarks" / "v1_llm"
    if not v1_llm.is_dir():
        raise FileNotFoundError(f"Missing {v1_llm}")
    artifacts["v1_llm"] = _write_tar(
        staging / "v1_llm.tar.gz",
        [(v1_llm, "benchmarks/v1_llm")],
    )
    artifacts["v1_llm"]["description"] = PARTS[0]["description"]

    human = ROOT / "benchmarks" / "v1_human"
    if not (human / "manifest.json").is_file() or not (human / "questions.json").is_file():
        raise FileNotFoundError("Missing benchmarks/v1_human/{manifest,questions}.json")
    artifacts["v1_human"] = _write_tar(
        staging / "v1_human.tar.gz",
        [
            (human / "manifest.json", "benchmarks/v1_human/manifest.json"),
            (human / "questions.json", "benchmarks/v1_human/questions.json"),
        ],
    )
    artifacts["v1_human"]["description"] = PARTS[1]["description"]

    dataset_members = [
        (p, f"data/{p.relative_to(ROOT / 'data').as_posix()}") for p in _referenced_dataset_roots()
    ]
    artifacts["datasets"] = _write_tar(staging / "datasets.tar.gz", dataset_members)
    artifacts["datasets"]["description"] = PARTS[2]["description"]
    artifacts["datasets"]["dataset_roots"] = [m[1].removeprefix("data/") for m in dataset_members]

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": "v1_llm_bundle",
        "created_at": _utc_now(),
        "repo_paths": {
            "v1_llm": "benchmarks/v1_llm",
            "v1_human": "benchmarks/v1_human",
            "datasets": "data/datasets",
        },
        "parts": artifacts,
        "github_release": {
            "tag": DEFAULT_RELEASE_TAG,
            "repo": None,
            "assets": {},
        },
        "notes": [
            "Raw archives are not tracked in git; they live on the GitHub Release.",
            "Run: python tools/benchmark_v1_bundle.py download",
            "Extracted paths match the maintainer worktree; bake_root symlinks are recreated.",
        ],
    }
    local_manifest = staging / "manifest.local.json"
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {local_manifest}")
    for part_id, meta in artifacts.items():
        print(
            f"  {part_id}: {meta['filename']} "
            f"{meta['bytes'] / (1024**2):.1f} MiB sha256={meta['sha256'][:12]}…"
        )
    return manifest


def upload_bundle(
    staging: Path,
    *,
    tag: str,
    repo: str | None,
    write_tracked_manifest: bool,
    title: str | None,
) -> dict[str, Any]:
    local_path = staging / "manifest.local.json"
    if not local_path.is_file():
        raise SystemExit(f"Run pack first; missing {local_path}")
    local = json.loads(local_path.read_text(encoding="utf-8"))
    repo_slug = _repo_slug(repo)

    assets: dict[str, Any] = {}
    asset_paths: list[Path] = []
    for part_id, meta in local["parts"].items():
        archive = staging / meta["filename"]
        if not archive.is_file():
            raise FileNotFoundError(archive)
        digest = _sha256_file(archive)
        if digest != meta["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {archive.name}: re-run pack")
        entry: dict[str, Any] = {
            "filename": meta["filename"],
            "bytes": meta["bytes"],
            "sha256": digest,
            "description": meta.get("description"),
        }
        if part_id == "datasets":
            entry["dataset_roots"] = meta.get("dataset_roots")
        assets[part_id] = entry
        asset_paths.append(archive)

    # Create or update release, then (re)upload assets.
    view = _run_gh(["release", "view", tag, "--repo", repo_slug], check=False)
    release_title = title or "ArchitectureIQ V1 LLM + human raw bundle"
    body = (
        "Gitignored raw artifacts for the V1 LLM benchmark and human pre-review pack.\n\n"
        "Download into a clone of this repo:\n\n"
        "```bash\n"
        "gh auth login\n"
        "python tools/benchmark_v1_bundle.py download\n"
        "```\n\n"
        "Layout after extract matches `benchmarks/v1_llm`, `benchmarks/v1_human`, "
        "and referenced `data/datasets/...` (full candidates + GT).\n"
    )
    if view.returncode != 0:
        print(f"Creating release {tag} on {repo_slug}")
        _run_gh(
            [
                "release",
                "create",
                tag,
                "--repo",
                repo_slug,
                "--title",
                release_title,
                "--notes",
                body,
                *[str(p) for p in asset_paths],
            ]
        )
    else:
        print(f"Release {tag} exists; uploading/replacing assets")
        for path in asset_paths:
            # clobber replaces same-named assets
            _run_gh(
                [
                    "release",
                    "upload",
                    tag,
                    str(path),
                    "--repo",
                    repo_slug,
                    "--clobber",
                ]
            )

    tracked = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": "v1_llm_bundle",
        "uploaded_at": _utc_now(),
        "repo_paths": local["repo_paths"],
        "github_release": {
            "tag": tag,
            "repo": repo_slug,
            "assets": assets,
        },
        "notes": local.get("notes") or [],
    }
    if write_tracked_manifest:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(tracked, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote tracked manifest {MANIFEST_PATH.relative_to(ROOT)}")
    else:
        out = staging / "manifest.uploaded.json"
        out.write_text(json.dumps(tracked, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out} (pass --write-tracked-manifest to update git copy)")
    return tracked


def _verify_sha256(path: Path, expected: str) -> None:
    got = _sha256_file(path)
    if got != expected:
        raise SystemExit(f"Checksum failed for {path.name}: expected {expected}, got {got}")


def _extract_tar(archive: Path, dest_root: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(dest_root, filter="data")
        else:
            tar.extractall(dest_root)


def _recreate_human_bake_root() -> None:
    """Rebuild relative symlinks under benchmarks/v1_human/bake_root."""
    human = ROOT / "benchmarks" / "v1_human"
    bake = human / "bake_root"
    manifest_path = human / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bake.exists() or bake.is_symlink():
        if bake.is_symlink() or bake.is_file():
            bake.unlink()
        else:
            shutil.rmtree(bake)
    bake.mkdir(parents=True)
    (bake / "datasets").symlink_to(
        os.path.relpath(ROOT / "data" / "datasets", bake),
        target_is_directory=True,
    )
    questions_dir = bake / "questions"
    questions_dir.mkdir()
    qids: list[str] = []
    if isinstance(manifest.get("questions"), list):
        for item in manifest["questions"]:
            if isinstance(item, str):
                qids.append(item)
            elif isinstance(item, dict):
                qid = item.get("question_id") or item.get("id")
                if qid:
                    qids.append(str(qid))
    linked = 0
    for qid in qids:
        name = qid if str(qid).startswith("q_") else f"q_{qid}"
        src = ROOT / "benchmarks" / "v1_llm" / "questions" / name
        if not src.is_dir():
            continue
        dest = questions_dir / name
        dest.symlink_to(os.path.relpath(src, questions_dir), target_is_directory=True)
        linked += 1
    print(f"Recreated bake_root with {linked} question symlinks")


def download_bundle(
    *,
    staging: Path,
    parts: list[str] | None,
    force: bool,
    skip_extract: bool,
) -> None:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            f"Tracked manifest missing: {MANIFEST_PATH.relative_to(ROOT)}\n"
            "Someone must pack+upload and commit the manifest first."
        )
    tracked = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    release = tracked.get("github_release") or {}
    assets = release.get("assets") or {}
    if not assets:
        raise SystemExit(
            "Manifest has no GitHub Release assets yet. "
            "Maintainer must run: python tools/benchmark_v1_bundle.py upload --write-tracked-manifest"
        )
    tag = release.get("tag") or DEFAULT_RELEASE_TAG
    repo_slug = release.get("repo") or _repo_slug()
    wanted = parts or list(assets.keys())
    staging.mkdir(parents=True, exist_ok=True)

    for part_id in wanted:
        if part_id not in assets:
            raise SystemExit(f"Unknown part {part_id!r}; known: {sorted(assets)}")
        meta = assets[part_id]
        archive = staging / meta["filename"]
        if archive.is_file() and not force:
            print(f"Using cached {archive}")
            _verify_sha256(archive, meta["sha256"])
        else:
            if archive.exists():
                archive.unlink()
            print(f"Downloading {part_id}: {meta['filename']} from {repo_slug}@{tag}")
            _run_gh(
                [
                    "release",
                    "download",
                    tag,
                    "--repo",
                    repo_slug,
                    "--pattern",
                    meta["filename"],
                    "--dir",
                    str(staging),
                    "--clobber",
                ]
            )
            if not archive.is_file():
                raise SystemExit(f"Download did not produce {archive}")
            _verify_sha256(archive, meta["sha256"])

        if skip_extract:
            continue
        print(f"Extracting {meta['filename']} -> repo root")
        _extract_tar(archive, ROOT)

    if not skip_extract:
        _recreate_human_bake_root()
        print("Done. Layout matches maintainer worktree under benchmarks/ and data/datasets/.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pack = sub.add_parser("pack", help="Create local tar.gz archives")
    p_pack.add_argument("--staging", type=Path, default=DEFAULT_STAGING)

    p_up = sub.add_parser("upload", help="Upload archives to a GitHub Release")
    p_up.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    p_up.add_argument("--tag", default=DEFAULT_RELEASE_TAG)
    p_up.add_argument("--repo", default=None, help="owner/name (default: current gh repo)")
    p_up.add_argument("--title", default=None)
    p_up.add_argument(
        "--write-tracked-manifest",
        action="store_true",
        help=f"Write {MANIFEST_PATH.relative_to(ROOT)} for committing",
    )

    p_dl = sub.add_parser("download", help="Download+extract into this clone")
    p_dl.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    p_dl.add_argument(
        "--parts",
        nargs="+",
        choices=[p["id"] for p in PARTS],
        help="Subset of parts to download (default: all)",
    )
    p_dl.add_argument("--force", action="store_true", help="Re-download even if cached")
    p_dl.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download archives into staging (no extract)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "pack":
        pack_bundle(args.staging.resolve())
    elif args.cmd == "upload":
        upload_bundle(
            args.staging.resolve(),
            tag=args.tag,
            repo=args.repo,
            write_tracked_manifest=args.write_tracked_manifest,
            title=args.title,
        )
    elif args.cmd == "download":
        download_bundle(
            staging=args.staging.resolve(),
            parts=args.parts,
            force=args.force,
            skip_extract=args.skip_extract,
        )
    else:
        parser.error(f"unknown command {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
