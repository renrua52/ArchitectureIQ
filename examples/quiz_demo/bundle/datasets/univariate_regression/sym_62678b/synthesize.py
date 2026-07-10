"""Dataset synthesis — source of truth for this instance."""
from __future__ import annotations

import torch


def target(x: torch.Tensor) -> torch.Tensor:
    return (torch.cos(6.283185307179586 * ((torch.tensor(0.8333, dtype=x.dtype, device=x.device) + x))) + ((x - x) + (x / torch.clamp(torch.abs(torch.tensor(-2, dtype=x.dtype, device=x.device)), min=0.1) * torch.sign(torch.tensor(-2, dtype=x.dtype, device=x.device) + 1e-12))))


def synthesize(
    *,
    train_size: int = 256,
    test_size: int = 256,
    point_seed: int = 1042,
    domain_low: float = 0.0,
    domain_high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.rand(train_size, generator=gen) * (domain_high - domain_low) + domain_low
    test_x = torch.rand(test_size, generator=gen) * (domain_high - domain_low) + domain_low
    train_y = target(train_x)
    test_y = target(test_x)
    return train_x.unsqueeze(-1), train_y.unsqueeze(-1), test_x.unsqueeze(-1), test_y.unsqueeze(-1)


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
