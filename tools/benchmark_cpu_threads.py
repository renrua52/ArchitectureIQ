#!/usr/bin/env python3
"""Measure CPU thread settings without mutating canonical candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _fingerprint(summary: dict[str, Any]) -> str:
    payload = {key: value for key, value in summary.items() if key != "environment"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run(candidate: Path, dataset: Path, profile_name: str, threads: list[int], output: Path) -> dict[str, Any]:
    import torch
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.profile import load_profile

    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    original_threads = torch.get_num_threads()
    try:
        for count in threads:
            torch.set_num_threads(int(count))
            copy = output / f"candidate_threads_{count}"
            shutil.copytree(candidate, copy)
            started = time.perf_counter()
            summary = run_ground_truth(copy, load_profile(profile_name), dataset)
            rows.append({
                "threads": int(count),
                "elapsed_seconds": time.perf_counter() - started,
                "fingerprint": _fingerprint(summary),
                "failed_seeds": summary.get("failed_seeds"),
                "mean_metric": summary.get(f"mean_{summary.get('selection_metric', 'test_mse')}"),
            })
    finally:
        torch.set_num_threads(original_threads)
    report = {
        "schema_version": "cpu_thread_benchmark_v1",
        "candidate": str(candidate),
        "dataset": str(dataset),
        "profile": profile_name,
        "thread_counts": threads,
        "rows": rows,
        "deterministic_across_threads": len({row["fingerprint"] for row in rows}) == 1,
        "cpu_count": os.cpu_count(),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "report.md").write_text("# CPU thread benchmark\n\n" + "\n".join(
        f"- threads `{row['threads']}`: {row['elapsed_seconds']:.3f}s, failed seeds `{row['failed_seeds']}`, fingerprint `{row['fingerprint'][:12]}`"
        for row in rows
    ) + f"\n- deterministic: `{report['deterministic_across_threads']}`\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", default="v1")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.candidate.resolve(), args.dataset.resolve(), args.profile, args.threads, args.output.resolve())
    print(json.dumps({key: report[key] for key in ("deterministic_across_threads", "thread_counts", "rows")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
