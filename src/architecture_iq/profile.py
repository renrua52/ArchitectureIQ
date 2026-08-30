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
            # Optional since v1.4: no sampleable loss takes a lambda, so a
            # profile that drops the L1/L2 variants has no grid to declare.
            loss_grids=raw.get("loss_grids", {}),
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
    def gru_lm(self) -> dict[str, Any]:
        return self.raw["gru_lm"]

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

    def min_training_steps(self) -> int | None:
        """Optional per-candidate floor on training_steps (None = no floor).

        Set via `budgets.min_training_steps`. When set, batch sizes that would
        push a budget below the floor are rejected during sampling, so every
        candidate trains long enough to escape systematic underfitting.
        """
        value = self.raw.get("budgets", {}).get("min_training_steps")
        return int(value) if value is not None else None

    def candidate_generation(self) -> dict[str, Any]:
        """Profile-scoped knobs for candidate sampling (absent = all defaults)."""
        raw = self.raw.get("candidate_generation", {})
        if not isinstance(raw, dict):
            raise ValueError("candidate_generation must be a mapping when present")
        return raw

    def parameter_ratio_max(self) -> float | None:
        """Largest in-set max/min trainable-parameter ratio, or None if unbounded.

        Set via `candidate_generation.parameter_ratio_max`. When set, every
        model in one candidate set is drawn from a single parameter band, so
        no ground truth is ever spent on a candidate that could not be a fair
        choice against its siblings. Absent means the old behaviour: sample
        architectures freely and let question assembly filter afterwards.
        """
        value = self.candidate_generation().get("parameter_ratio_max")
        if value is None:
            return None
        ratio = float(value)
        if ratio < 1.0:
            raise ValueError(
                f"candidate_generation.parameter_ratio_max must be >= 1.0, got {ratio}"
            )
        return ratio

    def parameter_band_probe(self) -> int:
        """How many architectures to pre-sample when locating a parameter band.

        The band has to come from the *reachable* parameter distribution: a
        profile grid can span three orders of magnitude for one model type and
        barely half of one for another, so fixed absolute edges would be empty
        for some family/model-type pairs.
        """
        value = self.candidate_generation().get("parameter_band_probe", 256)
        probe = int(value)
        if probe < 1:
            raise ValueError(
                f"candidate_generation.parameter_band_probe must be >= 1, got {probe}"
            )
        return probe

    def adam_betas_pool(self) -> list[tuple[float, float]]:
        """Adam/AdamW (beta1, beta2) options, normalised to a list of pairs.

        Two shapes are accepted. A flat ``[0.9, 0.999]`` is one fixed pair --
        every profile written before v1.4 uses that form -- and a nested
        ``[[0.9, 0.999], [0.9, 0.95]]`` is a pool to sample from. Normalising
        here keeps the two samplers (candidate generation and the inspector's
        custom settings) reading one definition instead of each re-deriving the
        shape.
        """
        raw = self.optimizer_grids["adam_betas"]
        if not isinstance(raw, list) or not raw:
            raise ValueError("optimizer_grids.adam_betas must be a non-empty list")
        if all(isinstance(item, (int, float)) for item in raw):
            if len(raw) != 2:
                raise ValueError(
                    f"a flat optimizer_grids.adam_betas must hold exactly 2 values, got {len(raw)}"
                )
            return [(float(raw[0]), float(raw[1]))]
        pairs: list[tuple[float, float]] = []
        for item in raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    f"each optimizer_grids.adam_betas entry must be a [beta1, beta2] pair, got {item!r}"
                )
            pairs.append((float(item[0]), float(item[1])))
        return pairs

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
