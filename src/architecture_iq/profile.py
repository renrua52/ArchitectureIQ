from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from architecture_iq.paths import PROFILES_DIR


VALID_EXECUTION_DEVICES = frozenset({"cpu", "cuda"})


def validate_execution_device(device: str) -> str:
    """Validate the benchmark execution device stored in an artifact."""
    normalized = device.strip().lower()
    if normalized not in VALID_EXECUTION_DEVICES:
        raise ValueError(
            f"Unsupported execution device {device!r}; "
            f"choose from {sorted(VALID_EXECUTION_DEVICES)}"
        )
    return normalized


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
    training_defaults: dict[str, Any]
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
            training_defaults=raw.get("training_defaults", {}),
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

    @property
    def execution_device(self) -> str:
        """Default device for newly generated candidates in this profile."""
        return validate_execution_device(str(self.ground_truth.get("device", "cpu")))

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

    @property
    def kan(self) -> dict[str, Any]:
        return self.raw.get("kan", {})

    @property
    def model_gates(self) -> dict[str, Any]:
        """Profile-scoped overrides for family/model compatibility."""
        gates = self.raw.get("model_gates", {})
        if not isinstance(gates, dict):
            raise ValueError("model_gates must be a mapping when present")
        return gates

    def model_types_for_family(
        self,
        family: str,
        family_model_types: list[str],
    ) -> list[str]:
        """Return profile-allowed model types for a dataset family.

        Normally this is the intersection of the profile model pool and the
        family compatibility declaration. A newer profile may explicitly
        replace a family's declaration through ``model_gates`` without
        changing older profile behaviour.
        """
        gate = self.model_gates.get(family)
        compatible = list(family_model_types)
        if gate is not None:
            if not isinstance(gate, dict):
                raise ValueError(f"model gate for {family!r} must be a mapping")
            override = gate.get("compatible_model_types")
            if override is not None:
                if not isinstance(override, list) or not all(isinstance(item, str) for item in override):
                    raise ValueError(
                        f"model_gates.{family}.compatible_model_types must be a list of strings"
                    )
                compatible = list(override)
            additions = gate.get("additional_model_types", [])
            if additions:
                if not isinstance(additions, list) or not all(isinstance(item, str) for item in additions):
                    raise ValueError(
                        f"model_gates.{family}.additional_model_types must be a list of strings"
                    )
                compatible.extend(item for item in additions if item not in compatible)
            blocked = gate.get("blocked_model_types", [])
            if blocked:
                if not isinstance(blocked, list) or not all(isinstance(item, str) for item in blocked):
                    raise ValueError(
                        f"model_gates.{family}.blocked_model_types must be a list of strings"
                    )
                blocked_set = set(blocked)
                compatible = [item for item in compatible if item not in blocked_set]
        return [
            model_type
            for model_type in self.pools.get("model_types", [])
            if model_type in compatible
        ]

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def training_steps(self, total_samples_seen: int, batch_size: int) -> int:
        if total_samples_seen % batch_size != 0:
            raise ValueError(
                f"budget {total_samples_seen} not divisible by batch_size {batch_size}"
            )
        return total_samples_seen // batch_size

    def family_training_defaults(self, family: str) -> dict[str, int]:
        defaults = self.training_defaults.get(family)
        if defaults is None:
            return {}
        batch_size = int(defaults["batch_size"])
        training_steps = int(defaults["training_steps"])
        total_samples_seen = int(defaults["total_samples_seen"])
        if batch_size * training_steps != total_samples_seen:
            raise ValueError(
                f"Invalid training default for {family!r}: "
                f"{training_steps} × {batch_size} != {total_samples_seen}"
            )
        return {
            "batch_size": batch_size,
            "training_steps": training_steps,
            "total_samples_seen": total_samples_seen,
        }


def load_profile(name: str = "v1") -> Profile:
    return Profile.load(PROFILES_DIR / f"{name}.yaml")
