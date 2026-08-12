"""Synthetic tabular binary-classification dataset — source of truth."""
from __future__ import annotations

import math

import torch


def target(
    x: torch.Tensor,
    *,
    rule_family: str = 'spiral',
    active_features: list[int] = [0, 1],
    interaction_pairs: list[list[int]] = [],
    rule_weights: list[float] = [1.0],
    piecewise_breakpoint: float = 0.0,
    spiral_turns: float = 2.0,
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
    raise ValueError(f"Unknown rule family: {rule_family}")


def _two_spirals(
    n_samples: int,
    *,
    turns: float,
    noise_std: float,
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
        return points + noise_std * torch.randn(count, 2, generator=gen)

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
    train_size: int = 1024,
    test_size: int = 2048,
    point_seed: int = 2036,
    input_dim: int = 2,
    noise_std: float = 0.05,
    decision_threshold: float = 0.0,
    rule_family: str = 'spiral',
    spiral_turns: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rule_family == "spiral":
        if input_dim != 2:
            raise ValueError("spiral requires input_dim == 2")
        train_x, train_y = _two_spirals(
            train_size, turns=spiral_turns, noise_std=noise_std, seed=point_seed
        )
        test_x, test_y = _two_spirals(
            test_size, turns=spiral_turns, noise_std=noise_std, seed=point_seed + 1
        )
        return train_x, train_y, test_x, test_y

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
