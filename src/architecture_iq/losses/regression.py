from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def compute_loss(
    loss_spec: dict[str, Any],
    model: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss_id = loss_spec["loss_id"]
    base = mse_loss(pred, target)
    if loss_id == "mse":
        return base
    lam = float(loss_spec.get("lambda", 0.0))
    if loss_id == "mse_l2":
        l2 = torch.mean(torch.stack([torch.mean(p ** 2) for p in model.parameters()]))
        return base + lam * l2
    if loss_id == "mse_l1":
        l1 = torch.mean(torch.stack([torch.mean(torch.abs(p)) for p in model.parameters()]))
        return base + lam * l1
    raise ValueError(f"Unknown loss_id: {loss_id}")


def render_loss_py(loss_spec: dict[str, Any]) -> str:
    loss_id = loss_spec["loss_id"]
    lam = loss_spec.get("lambda")
    if loss_id == "mse":
        body = "    return torch.mean((pred - target) ** 2)"
    elif loss_id == "mse_l2":
        body = f"""    base = torch.mean((pred - target) ** 2)
    l2 = torch.mean(torch.stack([torch.mean(p ** 2) for p in model.parameters()]))
    return base + {lam} * l2"""
    elif loss_id == "mse_l1":
        body = f"""    base = torch.mean((pred - target) ** 2)
    l1 = torch.mean(torch.stack([torch.mean(torch.abs(p)) for p in model.parameters()]))
    return base + {lam} * l1"""
    else:
        raise ValueError(loss_id)
    return f'''"""Loss function for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def loss_fn(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
{body}
'''
