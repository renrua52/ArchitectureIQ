"""Sample, freeze, and materialize a reusable XOR MLP/KAN candidate pool.

The sampler is deliberately family-specific: it never calls the generic model
sampler, so callers receive exactly ``count`` unique MLP specs and ``count``
unique KAN specs.  Every candidate is trained once per invocation/profile;
question pairs later reference those persisted GT artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
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

_SHARED_TRAINING: dict[str, Any] = {
    "total_samples_seen": 8192,
    "batch_size": 32,
    "optimizer": {
        "type": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
    },
    "loss": {"loss_id": "cross_entropy"},
}


def _canonical_model(model: dict[str, Any]) -> str:
    return yaml.safe_dump(model, sort_keys=True)


def _sample_family_models(
    profile: Profile,
    *,
    model_type: str,
    seed: int,
    count: int,
    dataset_params: dict[str, Any],
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    sampler = get_model_type(model_type)
    sampled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(count * 1000):
        model = sampler.sample_spec(profile, rng, dataset_params=dataset_params)
        key = _canonical_model(model)
        if key in seen:
            continue
        sampler.validate(model)
        seen.add(key)
        sampled.append(model)
        if len(sampled) == count:
            return sampled
    raise RuntimeError(
        f"Could not sample {count} unique {model_type} specs from seed {seed}; "
        f"only found {len(sampled)}"
    )


def sample_matrix(
    profile: Profile,
    *,
    dataset_spec: dict[str, Any],
    mlp_seed: int,
    kan_seed: int,
    count: int,
) -> dict[str, Any]:
    params = dataset_spec.get("params", {})
    if (
        dataset_spec.get("family") != "synthetic_tabular_classification"
        or params.get("rule_family") != "xor"
        or params.get("input_dim") != 2
        or params.get("num_classes") != 2
    ):
        raise ValueError("sampled pool requires a two-dimensional binary XOR dataset")
    models: list[dict[str, Any]] = []
    for model_type, seed in (("mlp", mlp_seed), ("kan", kan_seed)):
        for index, model in enumerate(
            _sample_family_models(
                profile,
                model_type=model_type,
                seed=seed,
                count=count,
                dataset_params=params,
            ),
            start=1,
        ):
            models.append({"id": f"{model_type}_{index:02d}", "model": model})
    return {
        "schema_version": "xor_sampled_candidate_pool_v1",
        "matrix_id": f"xor-sampled-{count}x{count}-m{mlp_seed}-k{kan_seed}",
        "family": "synthetic_tabular_classification",
        "dataset_constraints": {
            "input_dim": 2,
            "num_classes": 2,
            "rule_family": "xor",
        },
        "sampling": {
            "sampler": "family_specific_model_sampler_v1",
            "mlp_seed": int(mlp_seed),
            "kan_seed": int(kan_seed),
            "models_per_family": int(count),
        },
        "shared_training": copy.deepcopy(_SHARED_TRAINING),
        "candidate_pool": models,
    }


def write_matrix(matrix: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen matrix: {path}")
    path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    matrix = yaml.safe_load(raw)
    if not isinstance(matrix, dict):
        raise ValueError("sampled matrix must be a mapping")
    if matrix.get("schema_version") != "xor_sampled_candidate_pool_v1":
        raise ValueError("not an XOR sampled candidate-pool matrix")
    pool = matrix.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("sampled matrix requires a non-empty candidate_pool")
    return matrix, hashlib.sha256(raw).hexdigest()


def ensure_xor_dataset(profile: Profile, *, dataset_path: Path, seed: int) -> dict[str, Any]:
    """Create the fixed XOR dataset directly under an explicit review root."""
    spec_path = dataset_path / "dataset_spec.json"
    if spec_path.is_file():
        return read_json(spec_path)
    ensure_registries()
    family = get_dataset_family("synthetic_tabular_classification")
    partial = family.create_instance(profile, seed, input_dim=2, rule_family="xor")
    spec = family.build_spec_with_id(partial)
    family.materialize(spec, dataset_path)
    return spec

def materialize_pool(
    profile: Profile,
    *,
    dataset_path: Path,
    matrix_path: Path,
    set_name: str | None = None,
) -> Path:
    ensure_registries()
    dataset_path = dataset_path.resolve()
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    matrix, matrix_hash = load_matrix(matrix_path.resolve())
    params = dataset_spec.get("params", {})
    for key, expected in matrix["dataset_constraints"].items():
        if params.get(key) != expected:
            raise ValueError(f"matrix expects {key}={expected!r}, got {params.get(key)!r}")

    shared = matrix["shared_training"]
    budget = int(shared["total_samples_seen"])
    batch_size = int(shared["batch_size"])
    if budget <= 0 or batch_size <= 0 or budget % batch_size:
        raise ValueError("shared training budget must be positive and divisible by batch_size")
    optimizer = copy.deepcopy(shared["optimizer"])
    loss = copy.deepcopy(shared["loss"])
    pool = matrix["candidate_pool"]
    set_name = set_name or f"sampled_pool_{short_hash({'matrix_sha256': matrix_hash, 'profile_hash': profile.profile_hash})}"
    set_path = candidate_set_dir(dataset_path, set_name)
    if set_path.exists():
        raise FileExistsError(f"candidate pool already exists: {set_path}")
    set_path.mkdir(parents=True, exist_ok=False)
    write_set_manifest(
        set_path,
        set_name=set_name,
        budget=budget,
        count=len(pool),
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
        "mode": "sampled_pool",
        "matrix_id": matrix["matrix_id"],
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": matrix_hash,
        "sampling": matrix["sampling"],
        "shared_training": shared,
    }
    write_json(manifest_path, manifest)

    for entry in pool:
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), dict):
            raise ValueError("candidate_pool entries require a model mapping")
        model = copy.deepcopy(entry["model"])
        model_type = str(model.get("type", ""))
        if model_type not in {"mlp", "kan"}:
            raise ValueError(f"sampled pool contains unsupported model type {model_type!r}")
        model["input_dim"] = int(params["input_dim"])
        model["output_dim"] = int(params["num_classes"])
        family = get_model_type(model_type)
        family.validate(model)
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
        write_candidate(spec, candidate_path, family)
        run_ground_truth(candidate_path, profile, dataset_path)
    return set_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--mlp-seed", type=int, default=20260729)
    parser.add_argument("--kan-seed", type=int, default=20260730)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--set-name")
    parser.add_argument("--create-dataset-seed", type=int)
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    if args.create_dataset_seed is not None:
        ensure_xor_dataset(
            profile,
            dataset_path=args.dataset_path.resolve(),
            seed=args.create_dataset_seed,
        )
    dataset_spec = read_json(args.dataset_path.resolve() / "dataset_spec.json")
    if args.materialize_only:
        matrix, _ = load_matrix(args.matrix)
    else:
        matrix = sample_matrix(
            profile,
            dataset_spec=dataset_spec,
            mlp_seed=args.mlp_seed,
            kan_seed=args.kan_seed,
            count=args.count,
        )
        write_matrix(matrix, args.matrix)
    set_path = materialize_pool(
        profile,
        dataset_path=args.dataset_path,
        matrix_path=args.matrix,
        set_name=args.set_name,
    )
    print(set_path)


if __name__ == "__main__":
    main()