"""Optimizer factory for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    return torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), weight_decay=1e-05)
