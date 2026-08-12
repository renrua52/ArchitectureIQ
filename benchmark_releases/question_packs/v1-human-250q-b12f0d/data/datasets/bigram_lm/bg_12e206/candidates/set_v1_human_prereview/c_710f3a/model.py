"""Causal transformer language model."""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = 16
        self.token_embed = nn.Embedding(32, 32)
        self.pos_embed = nn.Embedding(16, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.token_embed(x) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.head(h)
