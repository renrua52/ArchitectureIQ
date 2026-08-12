"""Loss function for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def loss_fn(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    base = torch.mean((pred - target) ** 2)
    l1 = torch.mean(torch.stack([torch.mean(torch.abs(p)) for p in model.parameters()]))
    return base + 0.01 * l1
