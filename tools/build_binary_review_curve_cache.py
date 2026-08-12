#!/usr/bin/env python3
"""Build browser-readable learning-curve assets for the binary review viewer.

The V1 bundle stores one ``results/curves.npz`` per candidate. The review viewer
serves from ``data/v1_review/``, so it cannot fetch those files directly from the
bundle root. This tool converts the mean and seed standard deviation for every
question into one small JSON asset per question.

Example:
    .venv/bin/python tools/build_binary_review_curve_cache.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path("/tmp/v1bundle"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/v1_review/binary_questions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/v1_review/curves"),
    )
    return parser.parse_args()


def find_curve_file(bundle_root: Path, question: dict, candidate_id: str) -> Path:
    base = (
        bundle_root
        / "data"
        / "datasets"
        / question["family"]
        / question["dataset_id"]
        / "candidates"
    )
    matches = sorted(base.glob(f"*/{candidate_id}/results/curves.npz"))
    if not matches:
        raise FileNotFoundError(
            f"expected one curves.npz for {question['question_id']} / {candidate_id}, "
            f"found {len(matches)} under {base}"
        )
    if len(matches) > 1:
        with np.load(matches[0]) as first:
            first_curves = np.asarray(first["curves"])
            first_samples = np.asarray(first["samples"])
        for duplicate in matches[1:]:
            with np.load(duplicate) as other:
                same = np.array_equal(first_curves, other["curves"]) and np.array_equal(
                    first_samples, other["samples"]
                )
            if not same:
                raise ValueError(
                    f"candidate id is reused with different curves for "
                    f"{question['question_id']} / {candidate_id}: {matches}"
                )
    return matches[0]


def build_question_asset(bundle_root: Path, question: dict) -> dict:
    curves = []
    for choice in question.get("choices", []):
        curve_path = find_curve_file(bundle_root, question, choice["candidate_id"])
        with np.load(curve_path) as archive:
            values = np.asarray(archive["curves"], dtype=np.float64)
            samples = np.asarray(archive["samples"], dtype=np.int64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != samples.shape[0]:
            raise ValueError(f"invalid curve shape in {curve_path}: {values.shape}")
        curves.append(
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidate_id"],
                "samples": samples.tolist(),
                "mean": np.mean(values, axis=0).tolist(),
                "std": np.std(values, axis=0).tolist(),
                "n_seeds": int(values.shape[0]),
            }
        )
    return {"question_id": question["question_id"], "metric": question.get("metric"), "curves": curves}


def main() -> None:
    args = parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = data["questions"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for question in questions:
        asset = build_question_asset(args.bundle_root, question)
        (args.output_dir / f"{question['question_id']}.json").write_text(
            json.dumps(asset, separators=(",", ":")), encoding="utf-8"
        )
        written += 1
    manifest = {
        "schema": "v1_binary_review_curves_v1",
        "question_count": written,
        "source": str(args.bundle_root),
        "questions": [question["question_id"] for question in questions],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {written} question curve assets to {args.output_dir}")


if __name__ == "__main__":
    main()
