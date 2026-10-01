"""MLP model — matches candidate_spec model section."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """One hidden layer: silu(x + Linear(norm(x)))."""

    def __init__(self, width: int, use_layer_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(self.norm(x))
        h = h + x
        return self.act(h)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 96),
            nn.SiLU(),
            MLPBlock(width=96, use_layer_norm=False),
            MLPBlock(width=96, use_layer_norm=False),
            MLPBlock(width=96, use_layer_norm=False),
            MLPBlock(width=96, use_layer_norm=True),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
