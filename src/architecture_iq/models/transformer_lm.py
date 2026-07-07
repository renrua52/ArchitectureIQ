from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from architecture_iq.models.base import ModelFamily


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.token_embed(x) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.head(h)


class TransformerLmModelFamily(ModelFamily):
    name = "transformer_lm"

    @staticmethod
    def _dims(model_spec: dict[str, Any]) -> tuple[int, int]:
        if "d_model" in model_spec:
            d_model = int(model_spec["d_model"])
        elif "embed_dim" in model_spec:
            d_model = int(model_spec["embed_dim"])
        else:
            raise KeyError("transformer_lm model spec requires d_model")
        if "d_ff" in model_spec:
            d_ff = int(model_spec["d_ff"])
        elif "ff_dim" in model_spec:
            d_ff = int(model_spec["ff_dim"])
        else:
            raise KeyError("transformer_lm model spec requires d_ff")
        return d_model, d_ff

    def validate(self, model_spec: dict[str, Any]) -> None:
        d_model, _ = self._dims(model_spec)
        if d_model % int(model_spec["num_heads"]) != 0:
            raise ValueError("d_model must be divisible by num_heads")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        d_model, d_ff = self._dims(model_spec)
        return CausalTransformerLM(
            vocab_size=int(model_spec["vocab_size"]),
            context_length=int(model_spec["context_length"]),
            d_model=d_model,
            num_layers=int(model_spec["num_layers"]),
            num_heads=int(model_spec["num_heads"]),
            d_ff=d_ff,
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        d_model, d_ff = self._dims(model_spec)
        return f'''"""Causal transformer language model."""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = {int(model_spec["context_length"])}
        self.token_embed = nn.Embedding({int(model_spec["vocab_size"])}, {d_model})
        self.pos_embed = nn.Embedding({int(model_spec["context_length"])}, {d_model})
        layer = nn.TransformerEncoderLayer(
            d_model={d_model},
            nhead={int(model_spec["num_heads"])},
            dim_feedforward={d_ff},
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers={int(model_spec["num_layers"])})
        self.head = nn.Linear({d_model}, {int(model_spec["vocab_size"])})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.token_embed(x) + self.pos_embed(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        return self.head(h)
'''

    def sample_spec(
        self,
        profile: Any,
        rng: random.Random,
        dataset_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if dataset_params is None:
            raise ValueError("transformer_lm sampling requires dataset_params")
        cfg = profile.transformer_lm
        d_model = rng.choice(cfg["d_model"])
        num_heads = rng.choice(cfg["num_heads"])
        if d_model % num_heads != 0:
            d_model = num_heads * max(1, d_model // num_heads)
        return {
            "type": "transformer_lm",
            "vocab_size": int(dataset_params["vocab_size"]),
            "context_length": int(dataset_params["context_length"]),
            "d_model": d_model,
            "num_layers": rng.choice(cfg["num_layers"]),
            "num_heads": num_heads,
            "d_ff": rng.choice(cfg["d_ff"]),
        }
