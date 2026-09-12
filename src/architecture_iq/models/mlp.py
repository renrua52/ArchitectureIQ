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


def _activation_ctor_src(name: str, leaky_relu_slope: float) -> str:
    """Source text constructing one activation, for the generated model.py.

    Mirrors ``_activation_module``: the rendered file must build the same module
    the in-process model builds, so the two stay in one place per activation.
    """
    if name == "relu":
        return "nn.ReLU()"
    if name == "leaky_relu":
        return f"nn.LeakyReLU({leaky_relu_slope!r})"
    if name == "gelu":
        return "nn.GELU()"
    if name == "silu":
        return "nn.SiLU()"
    raise ValueError(f"Unknown activation: {name}")


def mlp_activation(model_spec: dict[str, Any]) -> str:
    """The one activation an MLP spec uses everywhere.

    The canonical field is the scalar ``activation``: the input projection and
    every hidden block share it. A legacy per-layer ``activations`` list is
    still read, but only when all entries agree -- the model has no way to
    represent a per-layer activation any more, so a mixed list is an error
    rather than something to silently collapse.
    """
    activation = model_spec.get("activation")
    if activation is not None:
        return str(activation)
    legacy = model_spec.get("activations")
    if not legacy:
        raise ValueError("MLP spec is missing 'activation'")
    distinct = sorted({str(value) for value in legacy})
    if len(distinct) > 1:
        raise ValueError(
            f"MLP activation is shared by every layer; spec mixes {distinct}"
        )
    return distinct[0]


class MLPBlock(nn.Module):
    """One hidden layer: ``act(x + Linear(norm(x)))`` residual, else ``act(Linear(norm(x)))``."""

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
        # Identity rather than None: the absent norm is a no-op in the forward
        # pass, so the pass reads as one expression instead of a None check,
        # matching the generated model.py line for line.
        self.norm = nn.LayerNorm(width) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = _activation_module(activation, leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exactly one hidden Linear per block whether or not the skip is on,
        # so `depth` counts Linear layers and the residual / non-residual
        # variants are parameter-matched. The skip is added *before* the
        # activation -- act(x + Linear(x)) -- so the branch can pull the
        # stream down as well as up.
        h = self.linear(self.norm(x))
        if self.use_residual:
            h = h + x
        return self.act(h)


class MLP(nn.Module):
    """Depth-many hidden Linear layers between an input projection and a head.

    ``activation`` and ``residual`` are network-wide: the stem and every block
    use the same activation, and the skip is either on for all blocks or off
    for all of them. ``layer_norm`` is the one per-layer switch.
    """

    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        activation: str,
        layer_norm: list[bool],
        residual: bool,
        output_dim: int = 1,
        leaky_relu_slope: float = LEGACY_LEAKY_RELU_SLOPE,
    ) -> None:
        super().__init__()
        if depth != len(layer_norm):
            raise ValueError("depth must match layer_norm length")
        # The input projection is followed by an activation; otherwise the
        # stem Linear and the first block Linear would be adjacent linear
        # maps (collapsible into one layer, inflating the parameter count
        # relative to the architecture the spec describes).
        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            _activation_module(activation, leaky_relu_slope),
        ]
        for use_layer_norm in layer_norm:
            layers.append(
                MLPBlock(
                    width=width,
                    activation=activation,
                    use_layer_norm=bool(use_layer_norm),
                    use_residual=residual,
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MlpModelFamily(ModelFamily):
    name = "mlp"

    def validate(self, model_spec: dict[str, Any]) -> None:
        depth = int(model_spec["depth"])
        norms = model_spec["layer_norm"]
        if depth != len(norms):
            raise ValueError("MLP depth mismatch with layer_norm")
        # Rejects an unknown activation name, and a legacy per-layer list
        # whose entries disagree.
        _activation_module(mlp_activation(model_spec))
        if int(model_spec.get("output_dim", 1)) < 1:
            raise ValueError("MLP output_dim must be positive")

    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        self.validate(model_spec)
        return MLP(
            input_dim=int(model_spec.get("input_dim", 1)),
            depth=int(model_spec["depth"]),
            width=int(model_spec["width"]),
            activation=mlp_activation(model_spec),
            layer_norm=[bool(v) for v in model_spec["layer_norm"]],
            residual=bool(model_spec["residual"]),
            output_dim=int(model_spec.get("output_dim", 1)),
            leaky_relu_slope=float(
                model_spec.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE)
            ),
        )

    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        self.validate(model_spec)
        width = int(model_spec["width"])
        input_dim = int(model_spec.get("input_dim", 1))
        residual = bool(model_spec["residual"])
        output_dim = int(model_spec.get("output_dim", 1))
        leaky_relu_slope = float(
            model_spec.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE)
        )
        activation = mlp_activation(model_spec)
        # The activation is written out as the constructor call it is, and the
        # skip as the line it is: a spec fixes both for the whole network, so
        # neither needs a module constant, a lookup table, or a runtime flag.
        # The generated file then reads as the one network it describes.
        activation_ctor = _activation_ctor_src(activation, leaky_relu_slope)
        residual_line = "        h = h + x\n" if residual else ""
        block_formula = (
            f"{activation}(x + Linear(norm(x)))"
            if residual
            else f"{activation}(Linear(norm(x)))"
        )
        norms = model_spec["layer_norm"]
        # layer_norm is the one per-layer switch, so it stays a block argument.
        blocks_str = "\n".join(
            f"            MLPBlock(width={width}, use_layer_norm={bool(norm)}),"
            for norm in norms
        )
        return f'''"""MLP model — matches candidate_spec model section."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """One hidden layer: {block_formula}."""

    def __init__(self, width: int, use_layer_norm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(width, width)
        self.act = {activation_ctor}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(self.norm(x))
{residual_line}        return self.act(h)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear({input_dim}, {width}),
            {activation_ctor},
{blocks_str}
            nn.Linear({width}, {output_dim}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''

    def sample_spec(
        self,
        profile: Any,
        rng: random.Random,
        dataset_params: dict[str, Any] | None = None,
        shared: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = profile.mlp
        depth = rng.choice(cfg["depth"])
        width = rng.choice(cfg["width"])
        residual = bool(rng.choice(cfg["residual"]))
        # One activation for the whole network, drawn from the profile pool.
        activation = str(rng.choice(cfg["activations"]))
        # layer_norm is the only per-layer switch, so each hidden block draws
        # its own flag.
        layer_norm = [bool(rng.choice([True, False])) for _ in range(depth)]
        spec: dict[str, Any] = {
            "type": "mlp",
            "depth": depth,
            "width": width,
            "residual": residual,
            "activation": activation,
            "layer_norm": layer_norm,
        }
        if activation == "leaky_relu" and "leaky_relu_slope" in cfg:
            spec["leaky_relu_slope"] = float(cfg["leaky_relu_slope"])
        if dataset_params is not None and "input_dim" in dataset_params:
            spec["input_dim"] = int(dataset_params["input_dim"])
        else:
            spec["input_dim"] = 1
        spec["output_dim"] = int(dataset_params.get("num_classes", 1)) if dataset_params else 1
        return spec
