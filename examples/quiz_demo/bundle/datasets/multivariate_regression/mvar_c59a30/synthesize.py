"""Dataset synthesis — source of truth for this instance."""
from __future__ import annotations

import torch


def target(x: torch.Tensor) -> torch.Tensor:
    return (((((torch.tanh(2 * (x[:, 0])) + torch.tanh(2 * ((x[:, 1] / torch.clamp(torch.abs(torch.tensor(0.5, dtype=x.dtype, device=x.device)), min=0.1) * torch.sign(torch.tensor(0.5, dtype=x.dtype, device=x.device) + 1e-12))))) + (torch.cos(6.283185307179586 * (x[:, 2])) * torch.tensor(-0.8333, dtype=x.dtype, device=x.device))) + (x[:, 3] + torch.tensor(0.1667, dtype=x.dtype, device=x.device))) + (torch.sin(6.283185307179586 * (x[:, 1])) * torch.cos(6.283185307179586 * (x[:, 3])))) + (x[:, 3] * torch.sin(6.283185307179586 * (x[:, 0]))))


def synthesize(
    *,
    train_size: int = 256,
    test_size: int = 256,
    point_seed: int = 200532225,
    input_dim: int = 4,
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
