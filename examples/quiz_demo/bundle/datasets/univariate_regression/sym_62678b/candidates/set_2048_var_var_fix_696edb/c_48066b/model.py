"""MLP model — matches candidate_spec model section."""
from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }
    return mapping[name]


class MLPBlock(nn.Module):
    def __init__(self, width: int, activation: str, use_layer_norm: bool, use_residual: bool) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(width) if use_layer_norm else None
        self.linear = nn.Linear(width, width)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.norm is not None:
            h = self.norm(h)
        h = self.linear(h)
        h = self.act(h)
        if self.use_residual:
            h = h + x
        return h


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
        MLPBlock(width=32, activation='silu', use_layer_norm=True, use_residual=False),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
