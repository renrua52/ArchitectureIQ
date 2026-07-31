"""Freeze completed wide environments into a cross-root snapshot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tools.meta_model_study.wide import freeze_snapshot_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environment",
        type=Path,
        action="append",
        required=True,
        help="completed environment directory; repeat in frozen evaluation order",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze_snapshot_manifest(args.environment, args.output)
    snapshot_bytes = args.output.resolve().read_bytes()
    print(
        json.dumps(
            {
                "snapshot_manifest_path": str(args.output.resolve()),
                "snapshot_manifest_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "counts": manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
