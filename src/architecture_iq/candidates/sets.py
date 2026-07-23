"""Named candidate sets under a dataset instance."""

from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import VARYING_AXIS_CHOICES
from architecture_iq.candidates.generator import (
    _pick_batch_size,
    sample_candidate,
    sample_loss,
    sample_model,
    sample_optimizer,
    write_candidate,
)
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir
from architecture_iq.profile import Profile
from architecture_iq.registry import get_model_type
from architecture_iq.util import read_json, short_hash, write_json

CandidateProgress = Callable[[int, int, str], None]

SET_MANIFEST = "set.json"


def parse_varying_axes(values: list[str]) -> frozenset[str]:
    axes: set[str] = set()
    for raw in values:
        for part in raw.replace(",", " ").split():
            axis = part.strip().lower()
            if axis not in VARYING_AXIS_CHOICES:
                raise ValueError(
                    f"Invalid varying axis {axis!r}; choose from model, optimizer, loss"
                )
            axes.add(axis)
    if not axes:
        raise ValueError("At least one varying axis is required (model, optimizer, loss)")
    return frozenset(axes)


def make_set_name(
    budget: int,
    varying_axes: frozenset[str],
    *,
    salt: Any,
    execution_device: str = "cpu",
) -> str:
    parts = []
    for axis in ("model", "optimizer", "loss"):
        parts.append("var" if axis in varying_axes else "fix")
    suffix = short_hash({"budget": budget, "vary": sorted(varying_axes), "device": execution_device, "salt": salt})
    return f"set_{budget}_{parts[0]}_{parts[1]}_{parts[2]}_{suffix}"


def sample_candidate_set_pool(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    count: int,
    varying_axes: frozenset[str],
    rng: random.Random,
    fixed_shared: dict[str, Any] | None = None,
    dataset_params: dict[str, Any] | None = None,
    execution_device: str | None = None,
) -> list[dict[str, Any]]:
    if not varying_axes <= VARYING_AXIS_CHOICES:
        raise ValueError(f"varying_axes must be subset of {sorted(VARYING_AXIS_CHOICES)}")

    shared = deepcopy(fixed_shared) if fixed_shared is not None else {}
    if "batch_size" not in shared:
        defaults = profile.family_training_defaults(family)
        if defaults and budget == defaults["total_samples_seen"]:
            shared["batch_size"] = defaults["batch_size"]
        else:
            shared["batch_size"] = _pick_batch_size(profile, budget, rng)
    if "model" not in varying_axes and "model" not in shared:
        shared["model"] = sample_model(
            profile, rng, family=family, dataset_params=dataset_params
        )
    if "optimizer" not in varying_axes and "optimizer" not in shared:
        shared["optimizer"] = sample_optimizer(profile, rng)
    if "loss" not in varying_axes and "loss" not in shared:
        shared["loss"] = sample_loss(profile, family, rng)

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(count * 20):
        if len(specs) >= count:
            break
        fixed = deepcopy(shared)
        if "model" in varying_axes:
            fixed["model"] = sample_model(
                profile, rng, family=family, dataset_params=dataset_params
            )
        if "optimizer" in varying_axes:
            fixed["optimizer"] = sample_optimizer(profile, rng)
        if "loss" in varying_axes:
            fixed["loss"] = sample_loss(profile, family, rng)
        fixed["_dataset_params"] = dataset_params
        spec = sample_candidate(
            profile,
            dataset_id=dataset_id,
            family=family,
            budget=budget,
            rng=rng,
            fixed=fixed,
            execution_device=execution_device,
        )
        key = spec["candidate_id"]
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)

    if len(specs) < count:
        raise RuntimeError(
            f"Could not sample {count} unique candidates for varying_axes={sorted(varying_axes)}"
        )
    return specs[:count]


def write_set_manifest(
    set_path: Path,
    *,
    set_name: str,
    budget: int,
    count: int,
    varying_axes: frozenset[str],
    fixed_shared: dict[str, Any],
    seed: int,
    profile: Profile,
    dataset_id: str,
    family: str,
) -> None:
    invariant_axes = sorted(VARYING_AXIS_CHOICES - varying_axes)
    manifest = {
        "schema_version": profile.schema_version,
        "set_id": set_name,
        "dataset_id": dataset_id,
        "family": family,
        "budget": {"total_samples_seen": budget},
        "count": count,
        "varying_axes": sorted(varying_axes),
        "invariant_axes": invariant_axes,
        "fixed_shared": fixed_shared,
        "seed": seed,
        "profile": profile.name,
        "profile_hash": profile.profile_hash,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    write_json(set_path / SET_MANIFEST, manifest)


def list_candidate_sets(dataset_path: Path) -> list[Path]:
    base = dataset_path / "candidates"
    if not base.is_dir():
        return []
    sets: list[Path] = []
    for path in sorted(base.iterdir()):
        if path.is_dir() and (path / SET_MANIFEST).is_file():
            sets.append(path.resolve())
    return sets


def list_candidates_in_set(set_path: Path) -> list[Path]:
    if not set_path.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in set_path.iterdir()
        if p.is_dir()
        and (p / "candidate_spec.json").is_file()
        and (p / "results" / "summary.json").exists()
    )


def load_set_manifest(set_path: Path) -> dict[str, Any]:
    return read_json(set_path / SET_MANIFEST)


def generate_candidate_set(
    profile: Profile,
    *,
    dataset_path: Path,
    budget: int,
    count: int,
    varying_axes: frozenset[str],
    rng: random.Random,
    fixed_shared: dict[str, Any] | None = None,
    seed: int,
    on_progress: CandidateProgress | None = None,
    execution_device: str | None = None,
) -> Path:
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]
    dataset_params = dataset_spec["params"]

    pinned = deepcopy(fixed_shared) if fixed_shared is not None else {}
    specs = sample_candidate_set_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=pinned,
        dataset_params=dataset_params,
        execution_device=execution_device,
    )

    set_name = make_set_name(
        budget,
        varying_axes,
        salt=rng.randint(0, 2**31 - 1),
        execution_device=specs[0]["execution"]["device"],
    )
    set_path = candidate_set_dir(dataset_path, set_name)
    set_path.mkdir(parents=True, exist_ok=False)

    shared_record = deepcopy(pinned)
    if "batch_size" not in shared_record and specs:
        shared_record["batch_size"] = specs[0]["budget"]["batch_size"]
    if "model" not in varying_axes and "model" not in shared_record and specs:
        shared_record["model"] = specs[0]["model"]
    if "optimizer" not in varying_axes and "optimizer" not in shared_record and specs:
        shared_record["optimizer"] = specs[0]["optimizer"]
    if "loss" not in varying_axes and "loss" not in shared_record and specs:
        shared_record["loss"] = specs[0]["loss"]

    write_set_manifest(
        set_path,
        set_name=set_name,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        fixed_shared=shared_record,
        seed=seed,
        profile=profile,
        dataset_id=dataset_id,
        family=family,
    )

    total = len(specs)
    for i, spec in enumerate(specs):
        out = candidate_in_set_dir(set_path, spec["candidate_id"])
        model_family = get_model_type(spec["model"]["type"])
        write_candidate(spec, out, model_family)
        run_ground_truth(out, profile, dataset_path)
        if on_progress is not None:
            on_progress(i + 1, total, spec["candidate_id"])

    return set_path
