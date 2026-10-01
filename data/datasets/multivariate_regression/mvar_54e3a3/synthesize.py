"""Dataset synthesis — source of truth for this instance."""
from __future__ import annotations

import torch


def target(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(2 * torch.pi * x[:, 0]) * torch.sin(2 * torch.pi * x[:, 2]) + x[:, 1] ** 2 - 0.5 * torch.cos(2 * torch.pi * x[:, 2]) + torch.sin(2 * torch.pi * x[:, 0]) + torch.tanh(2 * torch.sin(2 * torch.pi * x[:, 1])) + torch.tanh(2 * x[:, 2]) - 6


def synthesize(
    *,
    train_size: int = 256,
    test_size: int = 256,
    point_seed: int = 5215,
    input_dim: int = 3,
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
