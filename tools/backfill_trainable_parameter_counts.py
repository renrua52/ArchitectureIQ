#!/usr/bin/env python3
"""Backfill trainable parameter counts without changing historical candidate IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from architecture_iq.candidates.generator import trainable_parameter_count  # noqa: E402
from architecture_iq.registry import ensure_registries  # noqa: E402
from architecture_iq.util import read_json, write_json  # noqa: E402


def repo_root() -> Path:
    return ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root relative to the repository root.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write missing counts. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    data_root = (root / args.data_root).resolve()
    ensure_registries()

    missing = 0
    existing = 0
    failures: list[str] = []
    for spec_path in sorted(data_root.rglob("candidate_spec.json")):
        try:
            spec: dict[str, Any] = read_json(spec_path)
            if "trainable_parameter_count" in spec:
                existing += 1
                continue
            spec["trainable_parameter_count"] = trainable_parameter_count(spec["model"])
            missing += 1
            if args.write:
                write_json(spec_path, spec)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{spec_path}: {exc}")

    action = "Updated" if args.write else "Would update"
    print(f"{action} {missing} candidate specs; {existing} already had a count.")
    if failures:
        print(f"{len(failures)} specs could not be counted:")
        print("\n".join(failures))
        return 1
    if not args.write:
        print("Dry run only; re-run with --write to persist counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())