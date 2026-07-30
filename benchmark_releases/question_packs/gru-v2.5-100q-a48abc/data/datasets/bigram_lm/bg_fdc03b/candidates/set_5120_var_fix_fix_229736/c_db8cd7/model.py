"""Causal unidirectional GRU language model with per-layer residual connections."""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = 16
        self.token_embed = nn.Embedding(32, 80)
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=80,
                hidden_size=80,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(2)
        ])
        self.head = nn.Linear(80, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        for layer in self.gru_layers:
            layer_output, _ = layer(h)
            h = h + layer_output
        return self.head(h)
