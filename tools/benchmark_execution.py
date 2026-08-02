#!/usr/bin/env python3
"""Benchmark deterministic candidate execution on serial and CPU-parallel paths.

The benchmark copies candidate directories before running GT, so it never rewrites
the user's canonical artifacts. Parallel execution is candidate-level only: each
worker receives the same candidate spec, profile, dataset and seed semantics as
the serial path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_RUN_LOCK = threading.Lock()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_fingerprint(summary: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in summary.items()
        if key not in {"environment"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_one(payload: tuple[str, str, str, str]) -> dict[str, Any]:
    candidate_path, dataset_path, profile_name, output_path = payload
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from architecture_iq.ground_truth.runner import run_ground_truth
    from architecture_iq.profile import load_profile

    started = time.perf_counter()
    # Generated candidate modules use process-global import names; serialize
    # thread fallback calls to avoid module-cache races.
    with _RUN_LOCK:
        summary = run_ground_truth(
            Path(candidate_path), load_profile(profile_name), Path(dataset_path)
        )
    elapsed = time.perf_counter() - started
    result = {
        "candidate_id": summary.get("candidate_id", Path(candidate_path).name),
        "elapsed_seconds": elapsed,
        "fingerprint": _summary_fingerprint(summary),
        "summary": summary,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _copy_candidate(source: Path, destination_root: Path, index: int) -> Path:
    destination = destination_root / f"candidate_{index:03d}_{source.name}"
    shutil.copytree(source, destination)
    return destination


def benchmark(
    candidates: list[Path],
    *,
    workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    serial_root = output_dir / "serial_candidates"
    parallel_root = output_dir / "parallel_candidates"
    serial_root.mkdir()
    parallel_root.mkdir()

    payloads: list[tuple[str, str, str, str]] = []
    for index, source in enumerate(candidates):
        spec = _read_json(source / "candidate_spec.json")
        dataset_path = source.parents[2]
        profile = str(spec.get("profile", "v1"))
        serial_copy = _copy_candidate(source, serial_root, index)
        parallel_copy = _copy_candidate(source, parallel_root, index)
        serial_out = output_dir / f"serial_{index:03d}.json"
        parallel_out = output_dir / f"parallel_{index:03d}.json"
        time.perf_counter()
        serial_result = _run_one(
            (str(serial_copy), str(dataset_path), profile, str(serial_out))
        )
        serial_result["source_candidate"] = str(source)
        serial_result["execution"] = "serial"
        serial_out.write_text(json.dumps(serial_result, indent=2), encoding="utf-8")
        payloads.append((str(parallel_copy), str(dataset_path), profile, str(parallel_out)))

    parallel_started = time.perf_counter()
    parallel_backend = "process_pool"
    parallel_fallback_reason: str | None = None
    try:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            parallel_results = list(executor.map(_run_one, payloads))
    except (OSError, PermissionError) as exc:
        parallel_backend = "thread_fallback"
        parallel_fallback_reason = f"{type(exc).__name__}: {exc}"
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            parallel_results = list(executor.map(_run_one, payloads))
    parallel_elapsed = time.perf_counter() - parallel_started
    for index, result in enumerate(parallel_results):
        result["source_candidate"] = str(candidates[index])
        result["execution"] = "parallel"
        (output_dir / f"parallel_{index:03d}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    serial_results = [
        _read_json(output_dir / f"serial_{index:03d}.json")
        for index in range(len(candidates))
    ]
    matches = [
        serial_results[index]["fingerprint"] == parallel_results[index]["fingerprint"]
        for index in range(len(candidates))
    ]
    serial_elapsed = sum(float(item["elapsed_seconds"]) for item in serial_results)
    report = {
        "schema_version": "execution_benchmark_v1",
        "candidate_count": len(candidates),
        "workers": max(1, workers),
        "serial_elapsed_seconds": serial_elapsed,
        "parallel_wall_seconds": parallel_elapsed,
        "parallel_backend": parallel_backend,
        "parallel_fallback_reason": parallel_fallback_reason,
        "speedup_vs_serial_sum": serial_elapsed / parallel_elapsed
        if parallel_elapsed > 0
        else None,
        "deterministic_summary_matches": matches,
        "all_deterministic": all(matches),
        "cuda_available": False,
        "cuda_note": "This environment has CPU-only torch; GPU benchmark was not runnable.",
        "candidates": [
            {
                "source": str(candidates[index]),
                "serial_seconds": serial_results[index]["elapsed_seconds"],
                "parallel_seconds": parallel_results[index]["elapsed_seconds"],
                "fingerprint_match": matches[index],
            }
            for index in range(len(candidates))
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        "# Execution benchmark\n\n"
        f"- candidates: {report['candidate_count']}\n"
        f"- workers: {report['workers']}\n"
        f"- serial sum (s): {report['serial_elapsed_seconds']:.3f}\n"
        f"- parallel wall (s): {report['parallel_wall_seconds']:.3f}\n"
        f"- parallel backend: {report['parallel_backend']}\n"
        f"- parallel fallback: {report['parallel_fallback_reason'] or 'none'}\n"
        f"- speedup: {report['speedup_vs_serial_sum']:.3f}\n"
        f"- deterministic: `{report['all_deterministic']}`\n"
        f"- GPU: unavailable ({report['cuda_note']})\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo_release_integration/execution_benchmark"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = [path.resolve() for path in args.candidate]
    for path in candidates:
        if not (path / "candidate_spec.json").is_file():
            raise SystemExit(f"candidate_spec.json not found: {path}")
    report = benchmark(candidates, workers=args.workers, output_dir=args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
