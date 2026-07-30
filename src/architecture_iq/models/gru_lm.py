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
        layer_residual: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.layer_residual = bool(layer_residual)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        if self.layer_residual:
            # One independent single-layer GRU per architectural layer makes
            # the residual path explicit without adding any parameters.
            self.gru_layers = nn.ModuleList(
                [
                    nn.GRU(
                        input_size=d_model,
                        hidden_size=d_model,
                        num_layers=1,
                        batch_first=True,
                        dropout=0.0,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
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
        if self.layer_residual:
            for layer in self.gru_layers:
                layer_output, _ = layer(h)
                h = h + layer_output
        else:
            h, _ = self.gru(h)
        return self.head(h)


class GruLmModelFamily(ModelFamily):
    name = "gru_lm"

    _REQUIRED_KEYS = frozenset(
        {"type", "vocab_size", "context_length", "d_model", "num_layers"}
    )
    _OPTIONAL_KEYS = frozenset({"layer_residual"})

    def validate(self, model_spec: dict[str, Any]) -> None:
        keys = set(model_spec)
        if not self._REQUIRED_KEYS <= keys or keys - self._REQUIRED_KEYS - self._OPTIONAL_KEYS:
            raise ValueError(
                "gru_lm model spec must contain required keys "
                f"{sorted(self._REQUIRED_KEYS)} and only optional keys "
                f"{sorted(self._OPTIONAL_KEYS)}; got {sorted(keys)}"
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
        if "layer_residual" in model_spec and not isinstance(
            model_spec["layer_residual"], bool
        ):
            raise ValueError("gru_lm layer_residual must be a boolean")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        return CausalGruLM(
            vocab_size=int(model_spec["vocab_size"]),
            context_length=int(model_spec["context_length"]),
            d_model=int(model_spec["d_model"]),
            num_layers=int(model_spec["num_layers"]),
            layer_residual=bool(model_spec.get("layer_residual", False)),
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        layer_residual = bool(model_spec.get("layer_residual", False))
        if layer_residual:
            gru_definition = f'''        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size={int(model_spec["d_model"])},
                hidden_size={int(model_spec["d_model"])},
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range({int(model_spec["num_layers"])})
        ])'''
            forward_body = '''        for layer in self.gru_layers:
            layer_output, _ = layer(h)
            h = h + layer_output'''
            description = "Causal unidirectional GRU language model with per-layer residual connections."
        else:
            gru_definition = f'''        self.gru = nn.GRU(
            input_size={int(model_spec["d_model"])},
            hidden_size={int(model_spec["d_model"])},
            num_layers={int(model_spec["num_layers"])},
            batch_first=True,
            dropout=0.0,
        )'''
            forward_body = '''        h, _ = self.gru(h)'''
            description = "Causal unidirectional GRU language model."
        return f'''"""{description}"""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_length = {int(model_spec["context_length"])}
        self.token_embed = nn.Embedding({int(model_spec["vocab_size"])}, {int(model_spec["d_model"])})
{gru_definition}
        self.head = nn.Linear({int(model_spec["d_model"])}, {int(model_spec["vocab_size"])})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
{forward_body}
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
            "layer_residual": bool(cfg.get("layer_residual", False)),
        }
