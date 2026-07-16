from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import torch

from architecture_iq.families.base import DatasetFamily
from architecture_iq.profile import Profile
from architecture_iq.util import short_hash, write_json


RULE_FAMILIES = ("smooth_additive", "sparse_interaction", "piecewise_boundary")


SYNTHESIZE_TEMPLATE = '''"""Synthetic tabular binary-classification dataset — source of truth."""
from __future__ import annotations

import torch


def target(
    x: torch.Tensor,
    *,
    rule_family: str = {rule_family!r},
    active_features: list[int] = {active_features!r},
    interaction_pairs: list[list[int]] = {interaction_pairs!r},
    rule_weights: list[float] = {rule_weights!r},
    piecewise_breakpoint: float = {piecewise_breakpoint!r},
) -> torch.Tensor:
    if rule_family == "smooth_additive":
        score = torch.zeros(x.shape[0], dtype=x.dtype)
        for feature, weight in zip(active_features, rule_weights):
            value = x[:, feature]
            score = score + weight * (torch.sin(value) + 0.25 * value.square())
        return score
    if rule_family == "sparse_interaction":
        score = torch.zeros(x.shape[0], dtype=x.dtype)
        for (left, right), weight in zip(interaction_pairs, rule_weights):
            score = score + weight * x[:, left] * x[:, right]
        return score
    if rule_family == "piecewise_boundary":
        primary, secondary = active_features[:2]
        below_weight, above_weight, offset_weight = rule_weights
        branch_weight = torch.where(
            x[:, primary] > piecewise_breakpoint,
            torch.full_like(x[:, primary], above_weight),
            torch.full_like(x[:, primary], below_weight),
        )
        return branch_weight * x[:, secondary] + offset_weight * x[:, primary]
    raise ValueError(f"Unknown rule family: {{rule_family}}")


def synthesize(
    *,
    train_size: int = {train_size},
    test_size: int = {test_size},
    point_seed: int = {point_seed},
    input_dim: int = {input_dim},
    noise_std: float = {noise_std!r},
    decision_threshold: float = {decision_threshold!r},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.randn(train_size, input_dim, generator=gen, dtype=torch.float32)
    test_x = torch.randn(test_size, input_dim, generator=gen, dtype=torch.float32)
    train_score = target(train_x) + noise_std * torch.randn(train_size, generator=gen)
    test_score = target(test_x) + noise_std * torch.randn(test_size, generator=gen)
    train_y = (train_score > decision_threshold).to(torch.int64)
    test_y = (test_score > decision_threshold).to(torch.int64)
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
'''


def balanced_rule_family_schedule(count: int, *, seed: int = 0) -> list[str]:
    """Return a deterministic, near-equal rule-family allocation for benchmark builds."""
    if count < 0:
        raise ValueError("count must be non-negative")
    full, remainder = divmod(count, len(RULE_FAMILIES))
    schedule = list(RULE_FAMILIES) * full
    extras = list(RULE_FAMILIES)
    random.Random(seed).shuffle(extras)
    schedule.extend(extras[:remainder])
    random.Random(seed + 1).shuffle(schedule)
    return schedule


def _raw_score_for_calibration(
    x: torch.Tensor,
    *,
    rule_family: str,
    active_features: list[int],
    interaction_pairs: list[list[int]],
    rule_weights: list[float],
    piecewise_breakpoint: float,
) -> torch.Tensor:
    """Calibration-only mirror of the generated rule; materialization executes synthesize.py."""
    if rule_family == "smooth_additive":
        score = torch.zeros(x.shape[0], dtype=x.dtype)
        for feature, weight in zip(active_features, rule_weights, strict=True):
            value = x[:, feature]
            score = score + weight * (torch.sin(value) + 0.25 * value.square())
        return score
    if rule_family == "sparse_interaction":
        score = torch.zeros(x.shape[0], dtype=x.dtype)
        for (left, right), weight in zip(interaction_pairs, rule_weights, strict=True):
            score = score + weight * x[:, left] * x[:, right]
        return score
    if rule_family == "piecewise_boundary":
        primary, secondary = active_features[:2]
        below_weight, above_weight, offset_weight = rule_weights
        branch_weight = torch.where(
            x[:, primary] > piecewise_breakpoint,
            torch.full_like(x[:, primary], above_weight),
            torch.full_like(x[:, primary], below_weight),
        )
        return branch_weight * x[:, secondary] + offset_weight * x[:, primary]
    raise ValueError(f"Unknown rule family: {rule_family}")


class SyntheticTabularClassificationFamily(DatasetFamily):
    name = "synthetic_tabular_classification"

    @staticmethod
    def _rng_streams(instance_seed: int) -> tuple[int, int, int]:
        return instance_seed, instance_seed + 1_000, instance_seed + 2_000

    def create_instance(
        self,
        profile: Profile,
        seed: int,
        *,
        input_dim: int | None = None,
        rule_family: str | None = None,
    ) -> dict[str, Any]:
        design_seed, point_seed, calibration_seed = self._rng_streams(seed)
        cfg = profile.family_config(self.name)
        rng = random.Random(design_seed)
        input_dims = [int(value) for value in cfg["input_dims"]]
        if input_dim is not None and input_dim not in input_dims:
            raise ValueError(f"input_dim must be one of {input_dims}, got {input_dim}")
        resolved_input_dim = input_dim if input_dim is not None else rng.choice(input_dims)
        allowed_rules = [str(value) for value in cfg["rule_families"]]
        if set(allowed_rules) != set(RULE_FAMILIES):
            raise ValueError(f"rule_families must contain exactly {list(RULE_FAMILIES)}")
        if rule_family is not None and rule_family not in allowed_rules:
            raise ValueError(f"rule_family must be one of {allowed_rules}, got {rule_family!r}")
        # Consecutive instance seeds cycle evenly; batch builders may instead use the schedule above.
        resolved_rule = rule_family or allowed_rules[seed % len(allowed_rules)]

        requested_active = rng.choice([int(value) for value in cfg["active_feature_counts"]])
        min_features = 2 if resolved_rule in {"sparse_interaction", "piecewise_boundary"} else 1
        active_count = max(min_features, min(requested_active, resolved_input_dim))
        active_features = sorted(rng.sample(range(resolved_input_dim), active_count))
        interaction_pairs: list[list[int]] = []
        piecewise_breakpoint = 0.0
        if resolved_rule == "smooth_additive":
            rule_weights = [rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0]) for _ in active_features]
        elif resolved_rule == "sparse_interaction":
            all_pairs = list(itertools.combinations(active_features, int(cfg["interaction_order"])))
            pair_count = min(len(all_pairs), max(1, active_count - 1))
            interaction_pairs = [list(pair) for pair in rng.sample(all_pairs, pair_count)]
            rule_weights = [rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0]) for _ in interaction_pairs]
        else:
            piecewise_breakpoint = rng.uniform(-0.5, 0.5)
            rule_weights = [rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0]) for _ in range(3)]

        noise_std = float(rng.choice(cfg["noise_std"]))
        calibration_size = int(cfg["calibration_size"])
        target_positive_rate = float(cfg["target_positive_rate"])
        calibration_gen = torch.Generator().manual_seed(calibration_seed)
        calibration_x = torch.randn(calibration_size, resolved_input_dim, generator=calibration_gen)
        calibration_score = _raw_score_for_calibration(
            calibration_x,
            rule_family=resolved_rule,
            active_features=active_features,
            interaction_pairs=interaction_pairs,
            rule_weights=rule_weights,
            piecewise_breakpoint=piecewise_breakpoint,
        ) + noise_std * torch.randn(calibration_size, generator=calibration_gen)
        decision_threshold = float(torch.quantile(calibration_score, 1.0 - target_positive_rate).item())

        params = {
            "instance_seed": seed,
            "input_dim": resolved_input_dim,
            "num_classes": 2,
            "rule_family": resolved_rule,
            "active_features": active_features,
            "interaction_order": int(cfg["interaction_order"]),
            "interaction_pairs": interaction_pairs,
            "rule_weights": rule_weights,
            "piecewise_breakpoint": piecewise_breakpoint,
            "noise_std": noise_std,
            "decision_threshold": decision_threshold,
            "train_size": int(cfg["train_size"]),
            "test_size": int(cfg["test_size"]),
            "point_sampling": {"distribution": "standard_normal", "seed": point_seed},
            "calibration": {
                "distribution": "standard_normal",
                "seed": calibration_seed,
                "size": calibration_size,
                "target_positive_rate": target_positive_rate,
            },
        }
        return {
            "schema_version": profile.schema_version,
            "family": self.name,
            "params": params,
            "selection_metric": "test_ce",
            "significance": {
                "gap_min": float(profile.significance["gap_min"]),
                "fail_threshold": float(profile.ground_truth["fail_threshold"]),
            },
            "files": {"synthesize": "synthesize.py", "train": "train.pt", "test": "test.pt"},
        }

    def build_spec_with_id(self, partial: dict[str, Any]) -> dict[str, Any]:
        spec = {key: value for key, value in partial.items() if not key.startswith("_")}
        spec["dataset_id"] = f"stabcls_{short_hash(partial['params'])}"
        return spec

    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = spec["params"]
        synth_code = SYNTHESIZE_TEMPLATE.format(**params, point_seed=params["point_sampling"]["seed"])
        (out_dir / "synthesize.py").write_text(synth_code, encoding="utf-8")
        from architecture_iq.runtime.loader import load_synthesize_module

        module = load_synthesize_module(out_dir / "synthesize.py")
        train_x, train_y, test_x, test_y = module.synthesize()
        torch.save({"x": train_x, "y": train_y}, out_dir / "train.pt")
        torch.save({"x": test_x, "y": test_y}, out_dir / "test.pt")
        write_json(out_dir / "dataset_spec.json", spec)

    def load_tensors(self, dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train = torch.load(dataset_path / "train.pt", weights_only=True)
        test = torch.load(dataset_path / "test.pt", weights_only=True)
        return train["x"], train["y"], test["x"], test["y"]

    def selection_metric_name(self) -> str:
        return "test_ce"

    def default_significance(self) -> dict[str, Any]:
        return {}

    def compatible_model_types(self) -> list[str]:
        return ["mlp"]
