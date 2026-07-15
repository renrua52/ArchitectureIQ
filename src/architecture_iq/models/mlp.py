from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from architecture_iq.models.base import ModelFamily

LEGACY_LEAKY_RELU_SLOPE = 0.1


def _activation_module(
    name: str,
    leaky_relu_slope: float = LEGACY_LEAKY_RELU_SLOPE,
) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(leaky_relu_slope)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLPBlock(nn.Module):
    def __init__(
        self,
        width: int,
        activation: str,
        use_layer_norm: bool,
        use_residual: bool,
        leaky_relu_slope: float = LEGACY_LEAKY_RELU_SLOPE,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.norm = nn.LayerNorm(width) if use_layer_norm else None
        self.linear = nn.Linear(width, width)
        self.act = _activation_module(activation, leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.norm is not None:
            h = self.norm(h)
        h = self.linear(h)
        h = self.act(h)
        if self.use_residual:
            h = h + x
        return h


class RegressionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        activations: list[str],
        layer_norm: list[bool],
        residual: bool,
        leaky_relu_slope: float = LEGACY_LEAKY_RELU_SLOPE,
    ) -> None:
        super().__init__()
        if not (depth == len(activations) == len(layer_norm)):
            raise ValueError("depth must match activations and layer_norm length")
        layers: list[nn.Module] = [nn.Linear(input_dim, width)]
        for i in range(depth):
            layers.append(
                MLPBlock(
                    width=width,
                    activation=activations[i],
                    use_layer_norm=layer_norm[i],
                    use_residual=residual,
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MlpModelFamily(ModelFamily):
    name = "mlp"

    def validate(self, model_spec: dict[str, Any]) -> None:
        depth = int(model_spec["depth"])
        acts = model_spec["activations"]
        norms = model_spec["layer_norm"]
        if depth != len(acts) or depth != len(norms):
            raise ValueError("MLP depth mismatch with activations or layer_norm")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        return RegressionMLP(
            input_dim=int(model_spec.get("input_dim", 1)),
            depth=int(model_spec["depth"]),
            width=int(model_spec["width"]),
            activations=list(model_spec["activations"]),
            layer_norm=[bool(v) for v in model_spec["layer_norm"]],
            residual=bool(model_spec["residual"]),
            leaky_relu_slope=float(
                model_spec.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE)
            ),
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        depth = int(model_spec["depth"])
        width = int(model_spec["width"])
        input_dim = int(model_spec.get("input_dim", 1))
        residual = bool(model_spec["residual"])
        leaky_relu_slope = float(
            model_spec.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE)
        )
        acts = model_spec["activations"]
        norms = model_spec["layer_norm"]
        blocks = []
        for i in range(depth):
            blocks.append(
                f"        MLPBlock(width={width}, activation={acts[i]!r}, "
                f"use_layer_norm={norms[i]}, use_residual={residual}),"
            )
        blocks_str = "\n".join(blocks)
        return f'''"""MLP model — matches candidate_spec model section."""
from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    mapping = {{
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU({leaky_relu_slope!r}),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }}
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
            nn.Linear({input_dim}, {width}),
{blocks_str}
            nn.Linear({width}, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''

    def sample_spec(
        self,
        profile: Any,
        rng: random.Random,
        dataset_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = profile.mlp
        depth = rng.choice(cfg["depth"])
        width = rng.choice(cfg["width"])
        residual = rng.choice(cfg["residual"])
        activations = [rng.choice(cfg["activations"]) for _ in range(depth)]
        layer_norm = [rng.choice([True, False]) for _ in range(depth)]
        spec: dict[str, Any] = {
            "type": "mlp",
            "depth": depth,
            "width": width,
            "residual": residual,
            "layer_norm": layer_norm,
            "activations": activations,
            "leaky_relu_slope": float(cfg["leaky_relu_slope"]),
        }
        if dataset_params is not None and "input_dim" in dataset_params:
            spec["input_dim"] = int(dataset_params["input_dim"])
        else:
            spec["input_dim"] = 1
        return spec
