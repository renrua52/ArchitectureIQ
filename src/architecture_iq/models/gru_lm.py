from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from architecture_iq.models.base import ModelFamily


class CausalGruLM(nn.Module):
    """A causal, unidirectional GRU language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        h, _ = self.gru(h)
        return self.head(h)


class GruLmModelFamily(ModelFamily):
    name = "gru_lm"

    _REQUIRED_KEYS = frozenset(
        {"type", "vocab_size", "context_length", "d_model", "num_layers"}
    )

    def validate(self, model_spec: dict[str, Any]) -> None:
        keys = set(model_spec)
        if keys != self._REQUIRED_KEYS:
            raise ValueError(
                "gru_lm model spec must contain exactly "
                f"{sorted(self._REQUIRED_KEYS)}; got {sorted(keys)}"
            )
        if model_spec["type"] != self.name:
            raise ValueError("gru_lm model spec type must be 'gru_lm'")
        for key, minimum in (
            ("vocab_size", 2),
            ("context_length", 1),
            ("d_model", 1),
            ("num_layers", 1),
        ):
            value = int(model_spec[key])
            if value < minimum:
                raise ValueError(f"gru_lm {key} must be >= {minimum}")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        return CausalGruLM(
            vocab_size=int(model_spec["vocab_size"]),
            context_length=int(model_spec["context_length"]),
            d_model=int(model_spec["d_model"]),
            num_layers=int(model_spec["num_layers"]),
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        return f'''"""Causal unidirectional GRU language model."""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = {int(model_spec["context_length"])}
        self.token_embed = nn.Embedding({int(model_spec["vocab_size"])}, {int(model_spec["d_model"])})
        self.gru = nn.GRU(
            input_size={int(model_spec["d_model"])},
            hidden_size={int(model_spec["d_model"])},
            num_layers={int(model_spec["num_layers"])},
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear({int(model_spec["d_model"])}, {int(model_spec["vocab_size"])})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        h, _ = self.gru(h)
        return self.head(h)
'''

    def sample_spec(
        self,
        profile: Any,
        rng: random.Random,
        dataset_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if dataset_params is None:
            raise ValueError("gru_lm sampling requires dataset_params")
        cfg = profile.gru_lm
        return {
            "type": self.name,
            "vocab_size": int(dataset_params["vocab_size"]),
            "context_length": int(dataset_params["context_length"]),
            "d_model": int(rng.choice(cfg["d_model"])),
            "num_layers": int(rng.choice(cfg["num_layers"])),
        }
