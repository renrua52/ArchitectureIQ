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
    if batch_size <= 0 or eval_interval_samples <= 0:
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
    if batch_size <= 0 or eval_interval_steps <= 0:
        return []
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
    try:
        with np.load(curves_path, allow_pickle=False) as data:
            if "curves" not in data:
                return {"error": "curves.npz is missing the 'curves' array"}
            curves = np.asarray(data["curves"], dtype=np.float64)
            # A few older/custom artifacts stored one seed as a flat vector.
            # Normalize that representation instead of indexing shape[1].
            if curves.ndim == 1:
                curves = curves[np.newaxis, :]
            elif curves.ndim != 2:
                return {
                    "error": (
                        "curves.npz 'curves' must be a 1D or 2D numeric array; "
                        f"got shape {curves.shape}"
                    )
                }

            if "samples" in data:
                samples = np.asarray(data["samples"], dtype=np.int64).reshape(-1).tolist()
            else:
                if total_samples_seen is None or batch_size is None:
                    return {"error": "legacy curves.npz requires budget metadata"}
                if "eval_interval" in data:
                    eval_interval_steps = int(data["eval_interval"])
                    samples = _legacy_eval_samples(
                        int(total_samples_seen), int(batch_size), eval_interval_steps
                    )
                elif "eval_interval_samples" in data:
                    eval_interval_samples = int(data["eval_interval_samples"])
                    samples = reconstruct_eval_samples(
                        int(total_samples_seen), int(batch_size), eval_interval_samples
                    )
                else:
                    samples = all_step_samples(int(total_samples_seen), int(batch_size))
    except (OSError, TypeError, ValueError) as exc:
        return {"error": f"could not read curves.npz: {exc}"}

    n_cols = min(curves.shape[1], len(samples))
    if n_cols == 0:
        return {
            "error": (
                "curves.npz has no aligned curve points "
                f"(curves shape {curves.shape}, samples length {len(samples)})"
            )
        }
    result: dict[str, Any] = {
        "curves": curves[:, :n_cols],
        "eval_samples": samples[:n_cols],
    }
    if curves.shape[1] != len(samples):
        result["warning"] = (
            f"curve/sample length mismatch ({curves.shape[1]} vs {len(samples)}); "
            f"using the first {n_cols} points"
        )
    return result
