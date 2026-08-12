"""Loss function for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def loss_fn(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    base = torch.mean((pred - target) ** 2)
    l2 = torch.mean(torch.stack([torch.mean(p ** 2) for p in model.parameters()]))
    return base + 0.01 * l2
