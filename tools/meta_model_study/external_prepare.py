"""Prepare target-free external inputs in a Torch-only helper process.

This process exits before the prediction process loads XGBoost.  The split is
necessary on macOS because importing PyTorch before XGBoost 3.2 can segfault
inside the two native OpenMP runtimes even though both libraries work alone.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from tools.meta_model_study.external import load_prediction_inputs, sha256_file


SCHEMA_VERSION = "meta_model_prepared_external_inputs_v1"


def _atomic_write(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare(
    questions_path: Path,
    output_path: Path,
    *,
    include_parameter_count: bool = True,
) -> dict[str, Any]:
    questions = load_prediction_inputs(
        questions_path,
        include_parameter_count=include_parameter_count,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "questions_path": str(questions_path.resolve()),
        "questions_sha256": sha256_file(questions_path),
        "num_questions": len(questions),
        "include_parameter_count": include_parameter_count,
        "questions": questions,
    }
    _atomic_write(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-parameter-count", action="store_true")
    args = parser.parse_args(argv)
    prepare(
        args.questions.resolve(),
        args.output.resolve(),
        include_parameter_count=not args.exclude_parameter_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
