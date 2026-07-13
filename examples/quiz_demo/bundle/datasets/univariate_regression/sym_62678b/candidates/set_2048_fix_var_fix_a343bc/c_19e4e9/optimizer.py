"""Optimizer factory for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    return torch.optim.Adagrad(params, lr=0.003, weight_decay=0.0001)
