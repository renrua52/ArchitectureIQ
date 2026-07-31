#!/usr/bin/env python3
"""Publish canonical ArchitectureIQ question artifacts into a quiz bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quiz_bundle import (
    BundlePublishError,
    build_bundle_manifest,
    publish_quiz_bundle,
    write_bundle_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_TARGET = REPO_ROOT / "examples" / "quiz_demo" / "bundle"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        help=(
            "Question or question-run directories, absolute or relative to "
            "--data-root. A single question publishes a partial run."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Source data root containing datasets/ (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Destination quiz bundle (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the projected manifest without writing files.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Scan an existing target bundle and refresh only quiz_manifest.json.",
    )
    parser.add_argument(
        "--generated-at",
        help="Optional descriptive timestamp; excluded from release_id hashing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.refresh_manifest and args.sources:
        parser.error("sources cannot be used with --refresh-manifest")
    if not args.refresh_manifest and not args.sources:
        parser.error("provide at least one question/run source")

    try:
        if args.refresh_manifest:
            if args.dry_run:
                manifest = build_bundle_manifest(
                    args.target, generated_at=args.generated_at
                )
            else:
                manifest = write_bundle_manifest(
                    args.target, generated_at=args.generated_at
                )
        else:
            manifest = publish_quiz_bundle(
                args.data_root,
                args.sources,
                args.target,
                dry_run=args.dry_run,
                generated_at=args.generated_at,
            )
    except (BundlePublishError, OSError) as exc:
        print(f"quiz bundle publish failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("dry run: target was not modified", file=sys.stderr)
    else:
        print(f"wrote {args.target / 'quiz_manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
