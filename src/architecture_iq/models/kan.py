from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture_iq.models.base import ModelFamily

BASE_ACTIVATIONS = {"silu", "relu", "gelu", "tanh"}


def _activation_module(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown KAN base activation: {name}")


def _make_grid(
    *,
    in_features: int,
    grid_size: int,
    spline_order: int,
    grid_range: tuple[float, float],
) -> torch.Tensor:
    low, high = grid_range
    step = (high - low) / grid_size
    core = torch.linspace(low, high, grid_size + 1)
    left = core[0] - step * torch.arange(spline_order, 0, -1, dtype=core.dtype)
    right = core[-1] + step * torch.arange(1, spline_order + 1, dtype=core.dtype)
    knots = torch.cat((left, core, right))
    return knots.unsqueeze(0).repeat(in_features, 1)


def _bspline_bases(
    x: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Evaluate uniform B-spline bases with Cox-de Boor recursion."""
    bases = (
        (x.unsqueeze(-1) >= grid[:, :-1])
        & (x.unsqueeze(-1) < grid[:, 1:])
    ).to(dtype=x.dtype)
    bases[..., -1] = ((x >= grid[:, -2]) & (x <= grid[:, -1])).to(dtype=x.dtype)

    for order in range(1, spline_order + 1):
        left_num = x.unsqueeze(-1) - grid[:, : -(order + 1)]
        left_den = grid[:, order:-1] - grid[:, : -(order + 1)]
        right_num = grid[:, order + 1 :] - x.unsqueeze(-1)
        right_den = grid[:, order + 1 :] - grid[:, 1:-order]
        bases = left_num / left_den * bases[..., :-1] + right_num / right_den * bases[..., 1:]
    return bases


class KANLinear(nn.Module):
    """A fixed-grid spline KAN layer with a base activation branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
        base_activation: str,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.base_activation_name = base_activation
        self.base_activation = _activation_module(base_activation)
        self.register_buffer(
            "grid",
            _make_grid(
                in_features=self.in_features,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                grid_range=grid_range,
            ),
        )
        basis_count = self.grid_size + self.spline_order
        self.base_weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, basis_count)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"KANLinear expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        base = F.linear(self.base_activation(x), self.base_weight)
        bases = _bspline_bases(x, self.grid, self.spline_order)
        spline = torch.einsum("...ib,oib->...o", bases, self.spline_weight)
        return base + spline


class KAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        output_dim: int,
        *,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
        base_activation: str,
    ) -> None:
        super().__init__()
        dims = [input_dim] + [width] * (depth + 1) + [output_dim]
        self.layers = nn.ModuleList(
            [
                KANLinear(
                    dims[i],
                    dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    base_activation=base_activation,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KanModelFamily(ModelFamily):
    name = "kan"

    def validate(self, model_spec: dict[str, Any]) -> None:
        if model_spec.get("type") != self.name:
            raise ValueError("KAN model spec must have type='kan'")
        for key in ("input_dim", "output_dim", "depth", "width", "grid_size", "spline_order"):
            if int(model_spec[key]) <= 0:
                raise ValueError(f"KAN {key} must be positive")
        if model_spec.get("variant", "efficient_spline_v1") != "efficient_spline_v1":
            raise ValueError("Unsupported KAN variant")
        grid_range = model_spec["grid_range"]
        if len(grid_range) != 2 or float(grid_range[0]) >= float(grid_range[1]):
            raise ValueError("KAN grid_range must be [low, high] with low < high")
        if model_spec["base_activation"] not in BASE_ACTIVATIONS:
            raise ValueError(f"Unknown KAN base activation: {model_spec['base_activation']}")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        return KAN(
            input_dim=int(model_spec["input_dim"]),
            depth=int(model_spec["depth"]),
            width=int(model_spec["width"]),
            output_dim=int(model_spec["output_dim"]),
            grid_size=int(model_spec["grid_size"]),
            spline_order=int(model_spec["spline_order"]),
            grid_range=(float(model_spec["grid_range"][0]), float(model_spec["grid_range"][1])),
            base_activation=str(model_spec["base_activation"]),
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        input_dim = int(model_spec["input_dim"])
        output_dim = int(model_spec["output_dim"])
        depth = int(model_spec["depth"])
        width = int(model_spec["width"])
        grid_size = int(model_spec["grid_size"])
        spline_order = int(model_spec["spline_order"])
        low, high = (float(v) for v in model_spec["grid_range"])
        base_activation = str(model_spec["base_activation"])
        return f'''"""Self-contained spline KAN model — matches candidate_spec."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    mapping = {{
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }}
    return mapping[name]


def _make_grid(in_features: int) -> torch.Tensor:
    low, high = {low!r}, {high!r}
    grid_size, spline_order = {grid_size}, {spline_order}
    step = (high - low) / grid_size
    core = torch.linspace(low, high, grid_size + 1)
    left = core[0] - step * torch.arange(spline_order, 0, -1, dtype=core.dtype)
    right = core[-1] + step * torch.arange(1, spline_order + 1, dtype=core.dtype)
    return torch.cat((left, core, right)).unsqueeze(0).repeat(in_features, 1)


def _bspline_bases(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    bases = ((x.unsqueeze(-1) >= grid[:, :-1]) & (x.unsqueeze(-1) < grid[:, 1:])).to(x.dtype)
    bases[..., -1] = ((x >= grid[:, -2]) & (x <= grid[:, -1])).to(x.dtype)
    for order in range(1, {spline_order} + 1):
        left_num = x.unsqueeze(-1) - grid[:, :-(order + 1)]
        left_den = grid[:, order:-1] - grid[:, :-(order + 1)]
        right_num = grid[:, order + 1:] - x.unsqueeze(-1)
        right_den = grid[:, order + 1:] - grid[:, 1:-order]
        bases = left_num / left_den * bases[..., :-1] + right_num / right_den * bases[..., 1:]
    return bases


class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_activation = _activation({base_activation!r})
        self.register_buffer("grid", _make_grid(in_features))
        basis_count = {grid_size} + {spline_order}
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, basis_count))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.base_activation(x), self.base_weight)
        bases = _bspline_bases(x, self.grid)
        spline = torch.einsum("...ib,oib->...o", bases, self.spline_weight)
        return base + spline


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dims = [{input_dim}] + [{width}] * ({depth} + 1) + [{output_dim}]
        self.layers = nn.ModuleList([KANLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
'''

    def sample_spec(
        self,
        profile: Any,
        rng: random.Random,
        dataset_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = profile.kan
        grid_range = rng.choice(cfg["grid_range"])
        input_dim = int(dataset_params.get("input_dim", 1)) if dataset_params else 1
        output_dim = int(
            dataset_params.get("num_classes", dataset_params.get("output_dim", 1))
        ) if dataset_params else 1
        archetype = None
        archetypes = cfg.get("archetypes", {})
        if isinstance(archetypes, dict):
            choices = archetypes.get(str(input_dim), archetypes.get(input_dim))
            if choices:
                archetype = rng.choice(choices)
        if archetype is None:
            architecture = {
                "depth": rng.choice(cfg["depth"]),
                "width": rng.choice(cfg["width"]),
                "grid_size": rng.choice(cfg["grid_size"]),
                "spline_order": rng.choice(cfg["spline_order"]),
                "base_activation": rng.choice(cfg["base_activation"]),
            }
        else:
            required = ("depth", "width", "grid_size", "spline_order", "base_activation")
            missing = [key for key in required if key not in archetype]
            if missing:
                raise ValueError(
                    f"KAN archetype for input_dim={input_dim} is missing: {', '.join(missing)}"
                )
            architecture = {key: archetype[key] for key in required}
        return {
            "type": self.name,
            "variant": str(cfg.get("variant", "efficient_spline_v1")),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "depth": int(architecture["depth"]),
            "width": int(architecture["width"]),
            "grid_size": int(architecture["grid_size"]),
            "spline_order": int(architecture["spline_order"]),
            "grid_range": [float(grid_range[0]), float(grid_range[1])],
            "base_activation": str(architecture["base_activation"]),
        }
