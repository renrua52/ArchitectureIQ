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
    """A spline KAN layer with a base activation branch.

    Supports data-aware grid updates with spline coefficient refit:
    ``update_grid(x)`` recalculates grid knots from input quantiles AND
    refits ``spline_weight`` via least-squares so the spline function is
    preserved across the grid change (following pykan / efficient-kan).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
        base_activation: str,
        grid_eps: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_eps = float(grid_eps)
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

    def _b_splines_with_grid(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """B-spline basis evaluated on a specific grid. Returns (batch, in, coeff)."""
        return _bspline_bases(x, grid, self.spline_order)

    @torch.no_grad()
    def _curve2coeff(
        self, x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        """Refit spline coefficients on ``grid`` to match function values ``y``.

        Uses torch.linalg.lstsq (least squares), matching pykan's curve2coef.
        Args:
            x: (batch, in_features) sample points
            y: (batch, in_features, out_features) function values at x
            grid: (in_features, grid_size + 2*spline_order + 1) new grid knots
        Returns:
            (out_features, in_features, grid_size + spline_order) new spline_weight
        """
        # B-spline bases on the new grid: (batch, in, n_coef)
        bases = self._b_splines_with_grid(x, grid)  # (batch, in, n_coef)
        # We want: for each (in, out), solve bases @ w = y  =>  w = lstsq(bases, y)
        # bases: (batch, in, n_coef), y: (batch, in, out)
        # Rearrange to (in, batch, n_coef) and (in, batch, out) for batched lstsq
        A = bases.permute(1, 0, 2)   # (in, batch, n_coef)
        B = y.permute(1, 0, 2)       # (in, batch, out)
        # lstsq: A @ W = B, solve for W: (in, n_coef, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, n_coef, out)
        return solution.permute(2, 0, 1).contiguous()  # (out, in, n_coef)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.05) -> None:
        """Adapt grid to data distribution AND refit spline coefficients.

        Follows the pykan / efficient-kan approach:
        1. Evaluate current spline function at sample points (old grid + old weight)
        2. Build new grid: quantile-adaptive mixed with uniform (grid_eps)
        3. Refit spline_weight on new grid via least-squares to preserve the function

        Args:
            x: (n_samples, in_features) calibration data
            margin: padding added beyond data range (fraction of spread)
        """
        if x.dim() != 2 or x.shape[1] != self.in_features or x.shape[0] == 0:
            return
        batch = x.shape[0]

        # Step 1: compute current spline output (only the spline branch, not base)
        old_bases = _bspline_bases(x, self.grid, self.spline_order)  # (batch, in, n_coef)
        # spline output per (batch, in, out): einsum('bi c, o i c -> b i o')
        old_spline_out = torch.einsum("bic,oic->bio", old_bases, self.spline_weight)

        # Step 2: build new grid (quantile + uniform mixing)
        x_sorted = torch.sort(x, dim=0)[0]  # (batch, in)
        # adaptive grid: pick grid_size+1 evenly spaced quantile indices
        ids = torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        grid_adaptive = x_sorted[ids]  # (grid_size+1, in)

        # uniform grid spanning data range + margin
        data_low = x_sorted[0]   # (in,)
        data_high = x_sorted[-1] # (in,)
        spread = data_high - data_low
        spread = torch.where(spread.abs() < 1e-6, torch.full_like(spread, 1.0), spread)
        uniform_step = (spread + 2 * margin * spread) / self.grid_size  # (in,)
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
            * uniform_step.unsqueeze(0)
            + data_low.unsqueeze(0)
            - margin * spread.unsqueeze(0)
        )  # (grid_size+1, in)

        # mix: grid_eps=1 -> uniform, grid_eps=0 -> adaptive
        grid_core = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive  # (grid_size+1, in)

        # extend with spline_order padding knots on each side
        step_pad = (grid_core[-1] - grid_core[0]) / self.grid_size
        left = grid_core[:1] - step_pad * torch.arange(
            self.spline_order, 0, -1, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        right = grid_core[-1:] + step_pad * torch.arange(
            1, self.spline_order + 1, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        new_grid = torch.cat([left, grid_core, right], dim=0).T.contiguous()  # (in, n_knots)

        # Step 3: refit spline coefficients on new grid
        new_weight = self._curve2coeff(x, old_spline_out, new_grid)

        # Commit
        self.grid.copy_(new_grid)
        self.spline_weight.copy_(new_weight)

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
        grid_eps: float = 0.02,
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
                    grid_eps=grid_eps,
                )
                for i in range(len(dims) - 1)
            ]
        )

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.05) -> None:
        """Layer-wise data-aware grid update with coefficient refit.

        Each layer's grid is adapted to its own input distribution:
        - Layer 0 uses the raw input ``x``
        - Deeper layers use the intermediate activations from the layer below
        For each layer: evaluate old spline function -> build new quantile/uniform
        mixed grid -> refit spline_weight via least-squares (function preserved).
        """
        current = x
        for layer in self.layers:
            layer.update_grid(current, margin=margin)
            current = layer(current)

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
        variant = model_spec.get("variant", "efficient_spline_v1")
        if variant not in ("efficient_spline_v1", "dynamic_grid_spline_v1"):
            raise ValueError(f"Unsupported KAN variant: {variant}")
        grid_range = model_spec["grid_range"]
        if len(grid_range) != 2 or float(grid_range[0]) >= float(grid_range[1]):
            raise ValueError("KAN grid_range must be [low, high] with low < high")
        if model_spec["base_activation"] not in BASE_ACTIVATIONS:
            raise ValueError(f"Unknown KAN base activation: {model_spec['base_activation']}")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        grid_eps = 0.02
        grid_policy = model_spec.get("grid_policy")
        if isinstance(grid_policy, dict) and "grid_eps" in grid_policy:
            grid_eps = float(grid_policy["grid_eps"])
        return KAN(
            input_dim=int(model_spec["input_dim"]),
            depth=int(model_spec["depth"]),
            width=int(model_spec["width"]),
            output_dim=int(model_spec["output_dim"]),
            grid_size=int(model_spec["grid_size"]),
            spline_order=int(model_spec["spline_order"]),
            grid_range=(float(model_spec["grid_range"][0]), float(model_spec["grid_range"][1])),
            base_activation=str(model_spec["base_activation"]),
            grid_eps=grid_eps,
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
        grid_eps = 0.02
        grid_policy = model_spec.get("grid_policy")
        if isinstance(grid_policy, dict) and "grid_eps" in grid_policy:
            grid_eps = float(grid_policy["grid_eps"])
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
        self.grid_size = {grid_size}
        self.spline_order = {spline_order}
        self.grid_eps = {grid_eps!r}
        self.base_activation = _activation({base_activation!r})
        self.register_buffer("grid", _make_grid(in_features))
        basis_count = {grid_size} + {spline_order}
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, basis_count))
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    @torch.no_grad()
    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        bases = _bspline_bases(x, grid)
        A = bases.permute(1, 0, 2)
        B = y.permute(1, 0, 2)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.05) -> None:
        if x.dim() != 2 or x.shape[1] != self.in_features or x.shape[0] == 0:
            return
        batch = x.shape[0]
        old_bases = _bspline_bases(x, self.grid)
        old_spline_out = torch.einsum("bic,oic->bio", old_bases, self.spline_weight)
        x_sorted = torch.sort(x, dim=0)[0]
        ids = torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        grid_adaptive = x_sorted[ids]
        data_low = x_sorted[0]
        data_high = x_sorted[-1]
        spread = data_high - data_low
        spread = torch.where(spread.abs() < 1e-6, torch.full_like(spread, 1.0), spread)
        uniform_step = (spread + 2 * margin * spread) / self.grid_size
        grid_uniform = (torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
                        * uniform_step.unsqueeze(0) + data_low.unsqueeze(0) - margin * spread.unsqueeze(0))
        grid_core = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        step_pad = (grid_core[-1] - grid_core[0]) / self.grid_size
        left = grid_core[:1] - step_pad * torch.arange(self.spline_order, 0, -1, device=x.device, dtype=x.dtype).unsqueeze(1)
        right = grid_core[-1:] + step_pad * torch.arange(1, self.spline_order + 1, device=x.device, dtype=x.dtype).unsqueeze(1)
        new_grid = torch.cat([left, grid_core, right], dim=0).T.contiguous()
        new_weight = self._curve2coeff(x, old_spline_out, new_grid)
        self.grid.copy_(new_grid)
        self.spline_weight.copy_(new_weight)

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

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.05) -> None:
        current = x
        for layer in self.layers:
            layer.update_grid(current, margin=margin)
            current = layer(current)

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
        # grid_range is optional when grid_update=data_aware (grid is recalculated
        # from data quantiles before training). Use a default [-1, 1] placeholder.
        grid_ranges = cfg.get("grid_range", [[-1.0, 1.0]])
        grid_range = rng.choice(grid_ranges)
        grid_update = str(cfg.get("grid_update", "fixed"))
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
        spec = {
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
        # Add grid_policy for dynamic_grid_spline_v1 variant
        variant = str(cfg.get("variant", "efficient_spline_v1"))
        if variant == "dynamic_grid_spline_v1":
            grid_policy = cfg.get("grid_policy", {})
            if isinstance(grid_policy, dict):
                spec["grid_policy"] = {
                    "mode": grid_policy.get("mode", "train_quantile_warmup"),
                    "grid_eps": float(grid_policy.get("grid_eps", 0.02)),
                    "margin": float(grid_policy.get("margin", 0.05)),
                    "update_steps": list(grid_policy.get("update_steps", [1, 32, 64, 128])),
                    "freeze_after_warmup": True,
                }
        elif grid_update != "fixed":
            spec["grid_update"] = grid_update
        return spec
