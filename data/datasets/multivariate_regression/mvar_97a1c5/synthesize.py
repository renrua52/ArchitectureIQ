"""Dataset synthesis — source of truth for this instance."""
from __future__ import annotations

import torch


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(2 * torch.pi * x[:, 0]) * x[:, 1] + (x[:, 0] - 2) ** 2 - x[:, 1]


def synthesize(
    *,
    train_size: int = 256,
    test_size: int = 256,
    point_seed: int = 5212,
    input_dim: int = 2,
    domain_low: float = 0.0,
    domain_high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.rand(train_size, input_dim, generator=gen) * (domain_high - domain_low) + domain_low
    test_x = torch.rand(test_size, input_dim, generator=gen) * (domain_high - domain_low) + domain_low
    train_y = target(train_x)
    test_y = target(test_x)
    return train_x, train_y.unsqueeze(-1), test_x, test_y.unsqueeze(-1)


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
