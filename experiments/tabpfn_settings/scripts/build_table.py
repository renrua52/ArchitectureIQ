#!/usr/bin/env python3
"""Build a tabular CSV from a frozen question pack (candidate_spec + summary)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from src_tabpfn_settings.features import FEATURE_COLUMNS, rows_from_pack  # noqa: E402


META_COLUMNS = [
    "candidate_id",
    "family",
    "dataset_id",
    "profile",
    "selection_metric",
    "excluded",
    "source_dir",
    "mean_test_ce",
    "mean_test_mse",
    "mean_test_accuracy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "benchmark_releases" / "question_packs" / "xor-v2.5-100q-37b9da",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXP / "artifacts" / "xor_table.csv",
    )
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    pack = args.pack.resolve()
    if not pack.is_dir():
        raise SystemExit(f"Pack not found: {pack}")

    rows = rows_from_pack(pack, skip_excluded=not args.include_excluded)
    if not rows:
        raise SystemExit(f"No candidates with summary.json under {pack}")

    # union of target-like columns present
    target_cols = sorted({k for row in rows for k in row if k.startswith("mean_")})
    fieldnames = META_COLUMNS + [c for c in FEATURE_COLUMNS if c not in META_COLUMNS]
    for col in target_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {len(rows)} rows → {args.out}")
    families = sorted({str(r.get("family")) for r in rows})
    models = sorted({str(r.get("model_type")) for r in rows})
    print(f"families={families} model_types={models}")


if __name__ == "__main__":
    main()
