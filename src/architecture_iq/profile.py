from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from architecture_iq.paths import PROFILES_DIR


@dataclass
class Profile:
    raw: dict[str, Any]
    name: str
    schema_version: str
    pools: dict[str, Any]
    dataset: dict[str, Any]
    mlp: dict[str, Any]
    optimizer_grids: dict[str, Any]
    loss_grids: dict[str, Any]
    budgets: dict[str, Any]
    ground_truth: dict[str, Any]
    significance: dict[str, Any]
    question_generation: dict[str, Any]
    prompts: dict[str, Any]

    @classmethod
    def load(cls, path: Path | None = None) -> Profile:
        path = path or PROFILES_DIR / "v1.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            raw=raw,
            name=raw["profile"],
            schema_version=raw["schema_version"],
            pools=raw["pools"],
            dataset=raw["dataset"],
            mlp=raw["mlp"],
            optimizer_grids=raw["optimizer_grids"],
            loss_grids=raw["loss_grids"],
            budgets=raw["budgets"],
            ground_truth=raw["ground_truth"],
            significance=raw["significance"],
            question_generation=raw["question_generation"],
            prompts=raw["prompts"],
        )

    @property
    def budget_values(self) -> list[int]:
        return list(self.budgets["total_samples_seen"])

    @property
    def num_choices(self) -> int:
        return int(self.pools["num_choices"])

    @property
    def n_seeds(self) -> int:
        return int(self.ground_truth["n_seeds"])

    @property
    def base_seed(self) -> int:
        return int(self.ground_truth["base_seed"])

    def family_config(self, family: str) -> dict[str, Any]:
        configs = self.raw.get("dataset_configs", {})
        if family in configs:
            return dict(configs[family])
        if family == self.dataset.get("family"):
            legacy = {k: v for k, v in self.dataset.items() if k != "family"}
            return legacy
        raise KeyError(f"No dataset config for family {family!r}")

    @property
    def transformer_lm(self) -> dict[str, Any]:
        return self.raw["transformer_lm"]

    def training_steps(self, total_samples_seen: int, batch_size: int) -> int:
        if total_samples_seen % batch_size != 0:
            raise ValueError(
                f"budget {total_samples_seen} not divisible by batch_size {batch_size}"
            )
        return total_samples_seen // batch_size


def load_profile(name: str = "v1") -> Profile:
    return Profile.load(PROFILES_DIR / f"{name}.yaml")
