"""Load candidate learning curves from ground-truth ``curves.npz`` artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def all_step_samples(total_samples_seen: int, batch_size: int) -> list[int]:
    """Samples seen after each optimizer step."""
    if batch_size <= 0:
        return []
    training_steps = total_samples_seen // batch_size
    return [step * batch_size for step in range(1, training_steps + 1)]


def reconstruct_eval_samples(
    total_samples_seen: int,
    batch_size: int,
    eval_interval_samples: int,
) -> list[int]:
    """Reconstruct sparse sample axis from older curves.npz files."""
    if batch_size <= 0:
        return []
    training_steps = total_samples_seen // batch_size
    samples: list[int] = []
    for step in range(1, training_steps + 1):
        seen = step * batch_size
        if (
            seen == batch_size
            or seen % eval_interval_samples == 0
            or step == training_steps
        ):
            samples.append(seen)
    return samples


def _legacy_eval_samples(
    total_samples_seen: int,
    batch_size: int,
    eval_interval_steps: int,
) -> list[int]:
    """Reconstruct x-axis for older curves.npz files keyed by optimizer steps."""
    training_steps = total_samples_seen // batch_size
    steps = [
        step
        for step in range(1, training_steps + 1)
        if step == 1 or step % eval_interval_steps == 0 or step == training_steps
    ]
    return [step * batch_size for step in steps]


def load_candidate_curves(
    curves_path: Path,
    *,
    total_samples_seen: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    if not curves_path.is_file():
        return {"error": f"missing {curves_path.name}"}
    data = np.load(curves_path)
    curves = np.asarray(data["curves"], dtype=np.float64)

    if "samples" in data:
        samples = np.asarray(data["samples"], dtype=np.int64).tolist()
    else:
        if total_samples_seen is None or batch_size is None:
            return {"error": "legacy curves.npz requires budget metadata"}
        if "eval_interval" in data:
            eval_interval_steps = int(data["eval_interval"])
            samples = _legacy_eval_samples(total_samples_seen, batch_size, eval_interval_steps)
        elif "eval_interval_samples" in data:
            eval_interval_samples = int(data["eval_interval_samples"])
            samples = reconstruct_eval_samples(
                total_samples_seen, batch_size, eval_interval_samples
            )
        else:
            samples = all_step_samples(total_samples_seen, batch_size)

    n_cols = min(curves.shape[1], len(samples))
    return {
        "curves": curves[:, :n_cols],
        "eval_samples": samples[:n_cols],
    }
