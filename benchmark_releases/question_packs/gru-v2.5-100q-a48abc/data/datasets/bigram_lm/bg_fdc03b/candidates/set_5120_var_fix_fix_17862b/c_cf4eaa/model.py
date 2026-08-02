"""Causal unidirectional GRU language model."""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = 16
        self.token_embed = nn.Embedding(32, 36)
        self.gru = nn.GRU(
            input_size=36,
            hidden_size=36,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(36, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        h, _ = self.gru(h)
        return self.head(h)
