"""Self-contained spline KAN model — matches candidate_spec."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    mapping = {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    return mapping[name]


def _make_grid(in_features: int) -> torch.Tensor:
    low, high = -2.0, 2.0
    grid_size, spline_order = 3, 3
    step = (high - low) / grid_size
    core = torch.linspace(low, high, grid_size + 1)
    left = core[0] - step * torch.arange(spline_order, 0, -1, dtype=core.dtype)
    right = core[-1] + step * torch.arange(1, spline_order + 1, dtype=core.dtype)
    return torch.cat((left, core, right)).unsqueeze(0).repeat(in_features, 1)


def _bspline_bases(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    bases = ((x.unsqueeze(-1) >= grid[:, :-1]) & (x.unsqueeze(-1) < grid[:, 1:])).to(x.dtype)
    bases[..., -1] = ((x >= grid[:, -2]) & (x <= grid[:, -1])).to(x.dtype)
    for order in range(1, 3 + 1):
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
        self.base_activation = _activation('silu')
        self.register_buffer("grid", _make_grid(in_features))
        basis_count = 3 + 3
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
        dims = [2] + [8] * (1 + 1) + [2]
        self.layers = nn.ModuleList([KANLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
