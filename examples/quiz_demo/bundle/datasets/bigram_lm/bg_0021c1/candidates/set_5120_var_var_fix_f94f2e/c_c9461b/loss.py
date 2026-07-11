"""Loss function for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def loss_fn(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3:
        vocab = pred.shape[-1]
        return torch.nn.functional.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
    return torch.nn.functional.cross_entropy(pred, target.reshape(-1))
