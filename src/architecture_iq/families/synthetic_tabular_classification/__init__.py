from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import torch

from architecture_iq.families.base import DatasetFamily
from architecture_iq.profile import Profile
from architecture_iq.util import short_hash, write_json


# Keep this legacy tuple stable: callers that omit an explicit rule set must not
# begin sampling XOR/spiral merely because the framework learns to support them.
RULE_FAMILIES = ("smooth_additive", "sparse_interaction", "piecewise_boundary")
SUPPORTED_RULE_FAMILIES = (*RULE_FAMILIES, "xor", "spiral")


SYNTHESIZE_TEMPLATE = '''"""Synthetic tabular binary-classification dataset — source of truth."""
from __future__ import annotations

import math

import torch


def target(
    x: torch.Tensor,
    *,
    rule_family: str = {rule_family!r},
    active_features: list[int] = {active_features!r},
    interaction_pairs: list[list[int]] = {interaction_pairs!r},
    rule_weights: list[float] = {rule_weights!r},
    piecewise_breakpoint: float = {piecewise_breakpoint!r},
    spiral_turns: float = {spiral_turns!r},
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
    if rule_family == "xor":
        left, right = active_features
        return -x[:, left] * x[:, right]
    if rule_family == "spiral":
        # Soft Archimedean-arm score; synthesize() labels by generative arm, not threshold.
        left, right = active_features
        angle = torch.atan2(x[:, right], x[:, left])
        radius = torch.linalg.vector_norm(x[:, [left, right]], dim=1)
        return torch.sin(angle - radius)
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


def _two_spirals(
    n_samples: int,
    *,
    turns: float,{spiral_noise_param}
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classic interleaved Archimedean two-spirals in R^2 with balanced labels."""
    gen = torch.Generator().manual_seed(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    def arm(count: int, phase: float) -> torch.Tensor:
        t = torch.rand(count, generator=gen) * (float(turns) * 2.0 * math.pi)
        t = t + 0.5  # keep points away from the origin singularity
        radius = t
        xs = radius * torch.cos(t + phase)
        ys = radius * torch.sin(t + phase)
        points = torch.stack([xs, ys], dim=1)
        return points{spiral_jitter}

    x = torch.cat([arm(n0, 0.0), arm(n1, math.pi)], dim=0)
    y = torch.cat(
        [
            torch.zeros(n0, dtype=torch.int64),
            torch.ones(n1, dtype=torch.int64),
        ],
        dim=0,
    )
    perm = torch.randperm(n_samples, generator=gen)
    return x[perm], y[perm]


def synthesize(
    *,
    train_size: int = {train_size},
    test_size: int = {test_size},
    point_seed: int = {point_seed},
    input_dim: int = {input_dim},{noise_arg}
    decision_threshold: float = {decision_threshold!r},
    rule_family: str = {rule_family!r},
    spiral_turns: float = {spiral_turns!r},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rule_family == "spiral":
        if input_dim != 2:
            raise ValueError("spiral requires input_dim == 2")
        train_x, train_y = _two_spirals(
            train_size, turns=spiral_turns,{spiral_noise_kwarg} seed=point_seed
        )
        test_x, test_y = _two_spirals(
            test_size, turns=spiral_turns,{spiral_noise_kwarg} seed=point_seed + 1
        )
        return train_x, train_y, test_x, test_y

    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.randn(train_size, input_dim, generator=gen, dtype=torch.float32)
    test_x = torch.randn(test_size, input_dim, generator=gen, dtype=torch.float32)
    train_score = target(train_x){train_score_noise}
    test_score = target(test_x){test_score_noise}
    train_y = (train_score > decision_threshold).to(torch.int64)
    test_y = (test_score > decision_threshold).to(torch.int64)
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
'''


def balanced_rule_family_schedule(
    count: int,
    *,
    seed: int = 0,
    allowed_rules: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return a deterministic, near-equal rule-family allocation for benchmark builds."""
    if count < 0:
        raise ValueError("count must be non-negative")
    rules = tuple(RULE_FAMILIES if allowed_rules is None else allowed_rules)
    if not rules:
        raise ValueError("allowed_rules must be non-empty")
    if len(set(rules)) != len(rules):
        raise ValueError("allowed_rules must not contain duplicates")
    unknown = set(rules) - set(SUPPORTED_RULE_FAMILIES)
    if unknown:
        raise ValueError(f"allowed_rules contain unsupported values: {sorted(unknown)}")
    full, remainder = divmod(count, len(rules))
    schedule = list(rules) * full
    extras = list(rules)
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
    if rule_family == "xor":
        left, right = active_features
        return -x[:, left] * x[:, right]
    if rule_family == "spiral":
        left, right = active_features
        angle = torch.atan2(x[:, right], x[:, left])
        radius = torch.linalg.vector_norm(x[:, [left, right]], dim=1)
        return torch.sin(angle - radius)
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


def _resolve_noise_std(cfg: dict[str, Any], rng: random.Random) -> float:
    """Label/coordinate noise scale, or 0.0 when the profile declares none.

    v1.4 onwards omits ``noise_std`` entirely: labels are an exact function of
    the features and spiral points sit exactly on their arm, so nothing about
    noise reaches the spec, the generated code, or the prompt. Profiles that
    still carry the key keep drawing from it -- and keep consuming the same
    amount of randomness -- so their datasets reproduce bit-identically.
    """
    pool = cfg.get("noise_std")
    if pool is None:
        return 0.0
    return float(rng.choice(pool))

class SyntheticTabularClassificationFamily(DatasetFamily):
    """Tabular binary classification whose decision rule is a sampled axis.

    XOR and the two-spirals dataset are *not* sampled here: they are their own
    dataset families (see below), because as benchmark buckets they behave
    nothing like the smooth/sparse/piecewise rules -- spiral is 2-D by
    construction with generative labels and no threshold calibration, and XOR's
    quadrant rule is the one case where a linear score is useless. Grouping them
    under one family made a "classification" bucket whose difficulty depended
    entirely on which rule the seed happened to draw.

    The class attributes below are the whole extension mechanism: a single-rule
    subclass sets its own ``name``, ``dataset_id_prefix`` and
    ``forced_rule_family`` and inherits synthesis, materialization and metrics
    unchanged.
    """

    name = "synthetic_tabular_classification"
    train_loop_kind = "classification"
    instance_option_names = ("input_dim", "rule_family")
    #: Prefix of the content-addressed dataset_id. Distinct per family so two
    #: families that happen to produce identical params never share a folder.
    dataset_id_prefix = "stabcls"
    #: Rules the profile may list under ``dataset_configs.{name}.rule_families``.
    #: Still all five on the base family: pre-v1.4 profiles listed xor and
    #: spiral here, and those profiles are frozen records of past builds.
    supported_rule_families: tuple[str, ...] = SUPPORTED_RULE_FAMILIES
    #: When set, the rule is part of the family identity rather than a sampled
    #: axis, and the profile does not get to list alternatives.
    forced_rule_family: str | None = None

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
        if self.forced_rule_family is not None:
            # Single-rule family: rule_families in the config would be either
            # redundant or a contradiction, so it is optional and pinned.
            configured = [str(value) for value in cfg.get("rule_families", [])]
            if configured and configured != [self.forced_rule_family]:
                raise ValueError(
                    f"{self.name} always uses rule_family "
                    f"{self.forced_rule_family!r}; profile lists {configured}"
                )
            allowed_rules = [self.forced_rule_family]
        else:
            allowed_rules = [str(value) for value in cfg["rule_families"]]
        if not allowed_rules:
            raise ValueError("rule_families must be non-empty")
        if len(set(allowed_rules)) != len(allowed_rules):
            raise ValueError("rule_families must not contain duplicates")
        unknown_rules = set(allowed_rules) - set(self.supported_rule_families)
        if unknown_rules:
            raise ValueError(
                f"rule_families contain unsupported values: {sorted(unknown_rules)}"
            )
        if rule_family is not None and rule_family not in allowed_rules:
            raise ValueError(f"rule_family must be one of {allowed_rules}, got {rule_family!r}")
        # Consecutive instance seeds cycle evenly; batch builders may instead use the schedule above.
        resolved_rule = rule_family or allowed_rules[seed % len(allowed_rules)]

        spiral_turns_cfg = cfg.get("spiral_turns", 2.0)
        if isinstance(spiral_turns_cfg, (list, tuple)):
            # Diversity: sample turns per instance instead of freezing every
            # spiral dataset at the same geometry.
            spiral_turns = float(rng.choice([float(t) for t in spiral_turns_cfg]))
        else:
            spiral_turns = float(spiral_turns_cfg)
        if resolved_rule == "spiral":
            if 2 not in input_dims:
                raise ValueError("spiral requires 2 in dataset_configs.input_dims")
            if input_dim is not None and input_dim != 2:
                raise ValueError("spiral requires input_dim=2")
            resolved_input_dim = 2
            active_features = [0, 1]
            interaction_pairs: list[list[int]] = []
            rule_weights = [1.0]
            piecewise_breakpoint = 0.0
            noise_std = _resolve_noise_std(cfg, rng)
            decision_threshold = 0.0
            point_sampling: dict[str, Any] = {
                "distribution": "two_spirals",
                "seed": point_seed,
                "turns": spiral_turns,
            }
            calibration: dict[str, Any] = {
                "distribution": "none",
                "seed": calibration_seed,
                "size": 0,
                "target_positive_rate": 0.5,
                "note": "Labels are assigned by generative spiral arm (balanced).",
            }
        else:
            requested_active = rng.choice([int(value) for value in cfg["active_feature_counts"]])
            if resolved_rule == "xor":
                if resolved_input_dim < 2:
                    raise ValueError("xor requires input_dim >= 2")
                active_count = 2
            else:
                min_features = 2 if resolved_rule in {"sparse_interaction", "piecewise_boundary"} else 1
                if resolved_rule == "piecewise_boundary":
                    # The piecewise renderer consumes exactly two active
                    # features (primary/secondary); sampling more would make
                    # the prompt claim active features that never influence
                    # the decision boundary.
                    active_count = min(2, resolved_input_dim)
                else:
                    active_count = max(min_features, min(requested_active, resolved_input_dim))
            active_features = sorted(rng.sample(range(resolved_input_dim), active_count))
            interaction_pairs = []
            piecewise_breakpoint = 0.0
            if resolved_rule == "smooth_additive":
                rule_weights = [rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0]) for _ in active_features]
            elif resolved_rule == "sparse_interaction":
                all_pairs = list(itertools.combinations(active_features, int(cfg["interaction_order"])))
                pair_count = min(len(all_pairs), max(1, active_count - 1))
                interaction_pairs = [list(pair) for pair in rng.sample(all_pairs, pair_count)]
                rule_weights = [rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0]) for _ in interaction_pairs]
            elif resolved_rule == "xor":
                interaction_pairs = [[active_features[0], active_features[1]]]
                rule_weights = [-1.0]
            else:
                piecewise_breakpoint = rng.uniform(-0.5, 0.5)
                rule_weights = [rng.uniform(0.8, 1.6) * rng.choice([-1.0, 1.0]) for _ in range(3)]

            noise_std = _resolve_noise_std(cfg, rng)
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
            )
            if noise_std > 0.0:
                calibration_score = calibration_score + noise_std * torch.randn(
                    calibration_size, generator=calibration_gen
                )
            decision_threshold = float(
                torch.quantile(calibration_score, 1.0 - target_positive_rate).item()
            )
            point_sampling = {"distribution": "standard_normal", "seed": point_seed}
            calibration = {
                "distribution": "standard_normal",
                "seed": calibration_seed,
                "size": calibration_size,
                "target_positive_rate": target_positive_rate,
            }

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
            "spiral_turns": spiral_turns,
            "decision_threshold": decision_threshold,
            "train_size": int(cfg["train_size"]),
            "test_size": int(cfg["test_size"]),
            "point_sampling": point_sampling,
            "calibration": calibration,
        }
        # Only a dataset that really carries noise records it, so a noiseless
        # spec has no noise field for the renderer or the prompt to describe.
        if noise_std > 0.0:
            params["noise_std"] = noise_std

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
        spec["dataset_id"] = f"{self.dataset_id_prefix}_{short_hash(partial['params'])}"
        return spec

    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = spec["params"]
        noise_std = float(params.get("noise_std", 0.0))
        noisy = noise_std > 0.0
        synth_code = SYNTHESIZE_TEMPLATE.format(
            **params,
            point_seed=params["point_sampling"]["seed"],
            # noise_std itself is no longer a template field: when the dataset
            # carries noise it arrives inside noise_arg, and params may already
            # hold the key, which would collide as a duplicate kwarg.
            noise_arg=f"\n    noise_std: float = {noise_std!r}," if noisy else "",
            spiral_noise_param="\n    noise_std: float," if noisy else "",
            spiral_noise_kwarg=" noise_std=noise_std," if noisy else "",
            spiral_jitter=(
                " + noise_std * torch.randn(count, 2, generator=gen)" if noisy else ""
            ),
            train_score_noise=(
                " + noise_std * torch.randn(train_size, generator=gen)" if noisy else ""
            ),
            test_score_noise=(
                " + noise_std * torch.randn(test_size, generator=gen)" if noisy else ""
            ),
        )
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


class XorClassificationFamily(SyntheticTabularClassificationFamily):
    """XOR: labels from the sign product of two active coordinates.

    Its own bucket because no linear score separates the classes at all, which
    makes it the only classification family where width buys nothing and depth
    is the whole question. ``input_dim`` stays open -- the two active
    coordinates may sit inside a larger vector of distractors.
    """

    name = "xor_classification"
    instance_option_names = ("input_dim",)
    dataset_id_prefix = "xorcls"
    supported_rule_families = ("xor",)
    forced_rule_family = "xor"


class SpiralClassificationFamily(SyntheticTabularClassificationFamily):
    """Two interleaved Archimedean spirals, labelled by generative arm.

    Its own bucket because it shares almost nothing with the threshold-on-a-
    score families: the input is 2-D by construction, points come from the arms
    rather than from a normal, labels are exactly balanced by construction, and
    there is no threshold to calibrate. ``spiral_turns`` sets the difficulty.
    """

    name = "spiral_classification"
    instance_option_names = ()
    dataset_id_prefix = "spiralcls"
    supported_rule_families = ("spiral",)
    forced_rule_family = "spiral"
