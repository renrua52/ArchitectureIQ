from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3:
        vocab = pred.shape[-1]
        return F.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
    return F.cross_entropy(pred, target.reshape(-1))


def compute_loss(
    loss_spec: dict[str, Any],
    model: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    loss_id = loss_spec["loss_id"]
    base = cross_entropy_loss(pred, target)
    if loss_id == "cross_entropy":
        return base
    lam = float(loss_spec.get("lambda", 0.0))
    if loss_id == "cross_entropy_l2":
        l2 = torch.mean(torch.stack([torch.mean(p ** 2) for p in model.parameters()]))
        return base + lam * l2
    if loss_id == "cross_entropy_l1":
        l1 = torch.mean(torch.stack([torch.mean(torch.abs(p)) for p in model.parameters()]))
        return base + lam * l1
    raise ValueError(f"Unknown loss_id: {loss_id}")


def render_loss_py(loss_spec: dict[str, Any]) -> str:
    loss_id = loss_spec["loss_id"]
    lam = loss_spec.get("lambda")
    if loss_id == "cross_entropy":
        body = """    if pred.ndim == 3:
        vocab = pred.shape[-1]
        return torch.nn.functional.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
    return torch.nn.functional.cross_entropy(pred, target.reshape(-1))"""
    elif loss_id == "cross_entropy_l2":
        body = f"""    if pred.ndim == 3:
        vocab = pred.shape[-1]
        base = torch.nn.functional.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
    else:
        base = torch.nn.functional.cross_entropy(pred, target.reshape(-1))
    l2 = torch.mean(torch.stack([torch.mean(p ** 2) for p in model.parameters()]))
    return base + {lam} * l2"""
    elif loss_id == "cross_entropy_l1":
        body = f"""    if pred.ndim == 3:
        vocab = pred.shape[-1]
        base = torch.nn.functional.cross_entropy(pred.reshape(-1, vocab), target.reshape(-1))
    else:
        base = torch.nn.functional.cross_entropy(pred, target.reshape(-1))
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
