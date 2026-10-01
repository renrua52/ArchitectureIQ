"""MLP model — matches candidate_spec model section."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """One hidden layer: leaky_relu(Linear(norm(x)))."""

    def __init__(self, width: int, use_layer_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(self.norm(x))
        return self.act(h)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.01),
            MLPBlock(width=32, use_layer_norm=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
