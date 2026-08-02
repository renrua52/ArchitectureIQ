"""Materialize pre-registered architecture matrices for XOR review.

Unlike generic candidate generation, this tool never samples model, optimizer,
loss, batch size, or candidate order. It writes normal candidate specs and runs
normal ArchitectureIQ ground truth, so prompts and GT retain one source of truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from architecture_iq.candidates.generator import build_candidate_spec, write_candidate
from architecture_iq.candidates.sets import write_set_manifest
from architecture_iq.ground_truth.runner import run_ground_truth
from architecture_iq.paths import candidate_in_set_dir, candidate_set_dir
from architecture_iq.profile import Profile, load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, get_model_type
from architecture_iq.util import read_json, short_hash, write_json


_REQUIRED_SHARED = ("total_samples_seen", "batch_size", "optimizer", "loss")


def load_matrix(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    matrix = yaml.safe_load(raw)
    if not isinstance(matrix, dict):
        raise ValueError("candidate matrix must be a mapping")
    for key in ("matrix_id", "family", "dataset_constraints", "shared_training", "comparisons"):
        if key not in matrix:
            raise ValueError(f"candidate matrix is missing {key!r}")
    if not isinstance(matrix["matrix_id"], str) or not matrix["matrix_id"]:
        raise ValueError("candidate matrix matrix_id must be a non-empty string")
    if not isinstance(matrix["dataset_constraints"], dict):
        raise ValueError("candidate matrix dataset_constraints must be a mapping")
    if not isinstance(matrix["shared_training"], dict):
        raise ValueError("candidate matrix shared_training must be a mapping")
    if not isinstance(matrix["comparisons"], list) or not matrix["comparisons"]:
        raise ValueError("candidate matrix comparisons must be a non-empty list")
    missing = [key for key in _REQUIRED_SHARED if key not in matrix["shared_training"]]
    if missing:
        raise ValueError(f"candidate matrix shared_training is missing: {', '.join(missing)}")
    return matrix, hashlib.sha256(raw).hexdigest()


def resolve_comparisons(
    matrix: dict[str, Any],
    *,
    dataset_spec: dict[str, Any],
    profile: Profile,
) -> list[tuple[str, list[dict[str, Any]]]]:
    family = str(dataset_spec.get("family"))
    if family != str(matrix["family"]):
        raise ValueError(f"matrix expects family {matrix['family']!r}, got {family!r}")
    params = dataset_spec.get("params", {})
    constraints = matrix["dataset_constraints"]
    for key in ("input_dim", "num_classes", "rule_family"):
        expected = constraints.get(key)
        if expected is not None and params.get(key) != expected:
            raise ValueError(f"matrix expects {key}={expected!r}, got {params.get(key)!r}")

    allowed_types = set(
        profile.model_types_for_family(
            family, get_dataset_family(family).compatible_model_types()
        )
    )
    result: list[tuple[str, list[dict[str, Any]]]] = []
    seen_comparisons: set[str] = set()
    for comparison in matrix["comparisons"]:
        if not isinstance(comparison, dict):
            raise ValueError("each matrix comparison must be a mapping")
        comparison_id = comparison.get("id")
        recipes = comparison.get("candidates")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise ValueError("each matrix comparison needs a non-empty id")
        if comparison_id in seen_comparisons:
            raise ValueError(f"duplicate matrix comparison id {comparison_id!r}")
        seen_comparisons.add(comparison_id)
        if not isinstance(recipes, list) or len(recipes) < 2:
            raise ValueError(f"matrix comparison {comparison_id!r} needs at least two candidates")

        models: list[dict[str, Any]] = []
        seen_models: set[str] = set()
        for recipe in recipes:
            if not isinstance(recipe, dict) or not isinstance(recipe.get("id"), str):
                raise ValueError(f"matrix comparison {comparison_id!r} has an invalid candidate recipe")
            model = recipe.get("model")
            if not isinstance(model, dict) or not isinstance(model.get("type"), str):
                raise ValueError(f"matrix candidate {recipe.get('id')!r} needs a model mapping")
            resolved = copy.deepcopy(model)
            for key in ("input_dim", "output_dim"):
                expected = int(params["input_dim"] if key == "input_dim" else params["num_classes"])
                if key in resolved and int(resolved[key]) != expected:
                    raise ValueError(f"matrix candidate {recipe['id']!r} has {key} incompatible with the dataset")
                resolved[key] = expected
            if resolved["type"] not in allowed_types:
                raise ValueError(f"matrix candidate {recipe['id']!r} uses disallowed model type {resolved['type']!r}")
            key = repr(sorted(resolved.items()))
            if key in seen_models:
                raise ValueError(f"matrix comparison {comparison_id!r} repeats a model spec")
            seen_models.add(key)
            get_model_type(resolved["type"]).validate(resolved)
            models.append(resolved)
        result.append((comparison_id, models))
    return result


def materialize_matrix(
    profile: Profile,
    *,
    dataset_path: Path,
    matrix_path: Path,
) -> list[Path]:
    ensure_registries()
    dataset_path = dataset_path.resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    matrix_path = matrix_path.resolve()
    matrix, matrix_hash = load_matrix(matrix_path)
    comparisons = resolve_comparisons(matrix, dataset_spec=dataset_spec, profile=profile)
    shared = matrix["shared_training"]
    budget = int(shared["total_samples_seen"])
    batch_size = int(shared["batch_size"])
    if budget <= 0 or batch_size <= 0 or budget % batch_size:
        raise ValueError("matrix total_samples_seen must be positive and divisible by batch_size")
    optimizer = copy.deepcopy(shared["optimizer"])
    loss = copy.deepcopy(shared["loss"])
    if not isinstance(optimizer, dict) or not isinstance(loss, dict):
        raise ValueError("matrix optimizer and loss must be mappings")

    paths: list[Path] = []
    for comparison_id, models in comparisons:
        set_name = f"matrix_{short_hash({'matrix_hash': matrix_hash, 'comparison_id': comparison_id})}"
        set_path = candidate_set_dir(dataset_path, set_name)
        if set_path.exists():
            raise FileExistsError(f"matrix candidate set already exists: {set_path}")
        set_path.mkdir(parents=True, exist_ok=False)
        write_set_manifest(
            set_path,
            set_name=set_name,
            budget=budget,
            count=len(models),
            varying_axes=frozenset({"model"}),
            fixed_shared={"batch_size": batch_size, "optimizer": optimizer, "loss": loss},
            seed=0,
            profile=profile,
            dataset_id=dataset_spec["dataset_id"],
            family=str(dataset_spec["family"]),
        )
        manifest_path = set_path / "set.json"
        manifest = read_json(manifest_path)
        manifest["source"] = {
            "mode": "matrix",
            "matrix_id": matrix["matrix_id"],
            "matrix_path": str(matrix_path),
            "matrix_sha256": matrix_hash,
            "comparison_id": comparison_id,
        }
        write_json(manifest_path, manifest)
        for model in models:
            spec = build_candidate_spec(
                profile,
                dataset_id=dataset_spec["dataset_id"],
                family=str(dataset_spec["family"]),
                budget=budget,
                batch_size=batch_size,
                model=model,
                optimizer=optimizer,
                loss=loss,
            )
            candidate_path = candidate_in_set_dir(set_path, spec["candidate_id"])
            write_candidate(spec, candidate_path, get_model_type(model["type"]))
            run_ground_truth(candidate_path, profile, dataset_path)
        paths.append(set_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    for path in materialize_matrix(profile, dataset_path=args.dataset_path, matrix_path=args.matrix):
        print(path)


if __name__ == "__main__":
    main()
