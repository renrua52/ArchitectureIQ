#!/usr/bin/env python3
"""Per-family pipeline: create-dataset → generate-candidates (GT) → table → TabPFN eval.

Intended P1 path: randomly sample settings from ArchitectureIQ pools; labels come from
executing generated train.py (not frozen question-pack leftovers).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

from src_tabpfn_settings.features import FEATURE_COLUMNS, rows_from_pack  # noqa: E402

FAMILY_JOBS: list[dict[str, object]] = [
    {
        "family": "univariate_regression",
        "target": "mean_test_mse",
        "budget": 2048,
        "seed": 101,
    },
    {
        "family": "multivariate_regression",
        "target": "mean_test_mse",
        "budget": 5120,
        "seed": 102,
        "input_dim": 4,
    },
    {
        "family": "bigram_lm",
        "target": "mean_test_ce",
        "budget": 5120,
        "seed": 103,
    },
    {
        "family": "synthetic_tabular_classification",
        "target": "mean_test_ce",
        "budget": 8192,
        "seed": 104,
    },
]


def _write_table(rows: list[dict], target: str, out: Path) -> list[dict]:
    finite: list[dict] = []
    for row in rows:
        try:
            value = float(row.get(target))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value == value and abs(value) != float("inf"):
            finite.append(row)

    fieldnames = [
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
        *[c for c in FEATURE_COLUMNS],
    ]
    for col in sorted({k for r in finite for k in r if k.startswith("mean_")}):
        if col not in fieldnames:
            fieldnames.append(col)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in finite:
            writer.writerow(row)
    return finite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="v2.5-tabpfn-settings")
    parser.add_argument("--count", type=int, default=60, help="Candidates per family")
    parser.add_argument(
        "--vary",
        default="model,optimizer",
        help="Axes to vary when sampling settings (comma-separated)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--families", default="", help="Comma subset of family names")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse newest dataset dir under data/datasets/{family}",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXP / "artifacts" / "generated",
    )
    args = parser.parse_args()

    from architecture_iq.candidates.sets import generate_candidate_set
    from architecture_iq.datasets import create_dataset
    from architecture_iq.paths import DATA_DIR
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    profile = load_profile(args.profile)
    vary = frozenset(a.strip() for a in args.vary.split(",") if a.strip())

    want = {x.strip() for x in args.families.split(",") if x.strip()}
    jobs = [j for j in FAMILY_JOBS if not want or str(j["family"]) in want]
    if not jobs:
        raise SystemExit(f"No jobs matched --families={args.families!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "profile": args.profile,
        "count": args.count,
        "vary": sorted(vary),
        "device": args.device,
        "jobs": [],
    }

    for job in jobs:
        family = str(job["family"])
        target = str(job["target"])
        budget = int(job["budget"])  # type: ignore[arg-type]
        seed = int(job["seed"])  # type: ignore[arg-type]
        print(
            f"\n======== family={family} budget={budget} count={args.count} "
            f"vary={sorted(vary)} ========",
            flush=True,
        )

        fam_opts = {"input_dim": job["input_dim"]} if "input_dim" in job else None
        if args.skip_generate:
            family_root = DATA_DIR / "datasets" / family
            if not family_root.is_dir():
                raise SystemExit(f"Missing {family_root}; cannot --skip-generate")
            dataset_path = sorted(
                [p for p in family_root.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[0]
            print(f"reuse dataset_path={dataset_path}", flush=True)
        else:
            print(
                f"create-dataset profile={args.profile} family={family} seed={seed} "
                f"opts={fam_opts}",
                flush=True,
            )
            if args.dry_run:
                dataset_path = DATA_DIR / "datasets" / family / f"dryrun_seed{seed}"
            else:
                spec, dataset_path = create_dataset(
                    profile,
                    seed,
                    family_name=family,
                    family_options=fam_opts,
                )
                print(f"dataset_id={spec['dataset_id']} path={dataset_path}", flush=True)

            print(
                f"generate-candidates count={args.count} budget={budget} device={args.device}",
                flush=True,
            )
            if args.dry_run:
                print("(dry-run) skip generate_candidate_set", flush=True)
            else:
                set_dir = generate_candidate_set(
                    profile,
                    dataset_path=dataset_path.resolve(),
                    budget=budget,
                    count=args.count,
                    varying_axes=vary,
                    rng=random.Random(seed),
                    seed=seed,
                    execution_device=args.device,
                )
                print(f"candidate set → {set_dir}", flush=True)

        table_path = args.out_dir / f"table_{family}.csv"
        report_path = args.out_dir / f"report_{family}.json"
        finite_rows: list[dict] = []
        if not args.dry_run:
            rows = rows_from_pack(Path(dataset_path), skip_excluded=True)
            finite_rows = _write_table(rows, target, table_path)
            print(f"table {len(finite_rows)}/{len(rows)} finite → {table_path}", flush=True)

        if not args.skip_eval and not args.dry_run:
            import subprocess

            eval_cmd = [
                sys.executable,
                str(EXP / "scripts" / "evaluate.py"),
                "--table",
                str(table_path),
                "--target",
                target,
                "--device",
                args.device,
                "--tabpfn-version",
                "v2.5",
                "--report",
                str(report_path),
            ]
            if args.skip_tabpfn:
                eval_cmd.append("--skip-tabpfn")
            print("+", " ".join(eval_cmd), flush=True)
            subprocess.run(eval_cmd, check=True, cwd=str(ROOT))

        entry: dict[str, object] = {
            "family": family,
            "dataset_path": str(dataset_path),
            "target": target,
            "budget": budget,
            "seed": seed,
            "table": str(table_path),
            "report": str(report_path),
            "n_rows_finite": len(finite_rows) if not args.dry_run else None,
        }
        if not args.dry_run and report_path.is_file():
            entry["metrics"] = json.loads(report_path.read_text(encoding="utf-8"))["models"]
        manifest["jobs"].append(entry)

    man_path = args.out_dir / "manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {man_path}", flush=True)


if __name__ == "__main__":
    main()
