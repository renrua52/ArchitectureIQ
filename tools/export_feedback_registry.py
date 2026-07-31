#!/usr/bin/env python3
"""Export an attested quiz bundle as deterministic feedback registry artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quiz_bundle.feedback_registry import (
    FeedbackRegistryError,
    export_feedback_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Existing immutable quiz bundle containing quiz_manifest.json.",
    )
    parser.add_argument(
        "--json-output",
        required=True,
        type=Path,
        help="Registry JSON destination outside the bundle.",
    )
    parser.add_argument(
        "--sql-output",
        required=True,
        type=Path,
        help="PostgreSQL data-migration destination outside the bundle.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify both existing outputs byte-for-byte; never write files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = export_feedback_registry(
            args.bundle,
            json_output=args.json_output,
            sql_output=args.sql_output,
            check=args.check,
        )
    except (FeedbackRegistryError, OSError) as exc:
        print(f"feedback registry export failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "registry_id": registry["registry_id"],
                "release_id": registry["release_id"],
                "question_count": registry["question_count"],
                "choice_count": registry["choice_count"],
                "checked": args.check,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
