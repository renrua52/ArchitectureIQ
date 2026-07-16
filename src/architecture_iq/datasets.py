from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from architecture_iq.families.base import DatasetFamily
from architecture_iq.paths import DATA_DIR, dataset_dir
from architecture_iq.profile import Profile
from architecture_iq.registry import get_dataset_family
from architecture_iq.util import read_json, write_json


@dataclass(frozen=True)
class DatasetInstance:
    family: str
    dataset_id: str
    path: Path


def list_dataset_instances(
    data_dir: Path | None = None,
    *,
    family: str | None = None,
) -> list[DatasetInstance]:
    """List materialized dataset instances under ``data/datasets/``."""
    root = (data_dir or DATA_DIR) / "datasets"
    if not root.is_dir():
        return []

    families = [family] if family is not None else sorted(p.name for p in root.iterdir() if p.is_dir())
    instances: list[DatasetInstance] = []
    for fam in families:
        fam_dir = root / fam
        if not fam_dir.is_dir():
            continue
        for path in sorted(fam_dir.iterdir()):
            if path.is_dir() and (path / "dataset_spec.json").is_file():
                spec = read_json(path / "dataset_spec.json")
                instances.append(
                    DatasetInstance(
                        family=fam,
                        dataset_id=spec.get("dataset_id", path.name),
                        path=path.resolve(),
                    )
                )
    return instances


def resolve_dataset_family(
    profile: Profile,
    *,
    family: str | None = None,
    random_pick: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Resolve a dataset family from the profile pool (never implicit default)."""
    families = list(profile.pools["dataset_families"])
    if not families:
        raise ValueError("Profile has no dataset_families in pool")
    if family is not None:
        if family not in families:
            raise ValueError(f"Unknown family {family!r}; choose from {families}")
        return family
    if random_pick:
        picker = rng if rng is not None else random.Random()
        return picker.choice(families)
    raise ValueError(
        "Dataset family is required: specify --family, --random-family, or use --interactive"
    )


def create_dataset(
    profile: Profile,
    seed: int,
    *,
    family_name: str,
    family_options: dict[str, Any] | None = None,
) -> tuple[dict, Path]:
    resolve_dataset_family(profile, family=family_name)
    family = get_dataset_family(family_name)
    options = family_options or {}
    if family_name in {"multivariate_regression", "synthetic_tabular_classification"}:
        partial = family.create_instance(profile, seed, **options)
    else:
        if options:
            raise ValueError(f"family_options are not supported for {family_name!r}")
        partial = family.create_instance(profile, seed)
    spec = family.build_spec_with_id(partial)
    out = dataset_dir(family.name, spec["dataset_id"])
    materialized = {**partial, **spec}
    family.materialize(materialized, out)
    return read_json(out / "dataset_spec.json"), out


def load_dataset_spec(path: Path) -> dict:
    return read_json(path / "dataset_spec.json")


def format_dataset_summary_lines(spec: dict) -> list[str]:
    """Human-readable summary lines for a dataset spec (family-specific)."""
    family = spec["family"]
    params = spec["params"]
    if family == "univariate_regression":
        return [f"Expression: {params['expression']}"]
    if family == "multivariate_regression":
        return [
            f"Input dimension: {params['input_dim']}",
            f"Expression: {params['expression']}",
        ]
    if family == "bigram_lm":
        return [
            f"Vocab size: {params['vocab_size']}",
            f"Context length: {params['context_length']}",
            f"Layout: {params['layout']}",
        ]
    if family == "synthetic_tabular_classification":
        return [
            f"Input dimension: {params['input_dim']}", f"Classes: {params['num_classes']}",
            f"Decision rule: {params['rule_family']}",
        ]
    return [f"{key}: {value}" for key, value in sorted(params.items())]
