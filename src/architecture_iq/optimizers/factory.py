from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad, Optimizer


def build_optimizer(optimizer_spec: dict[str, Any], model: nn.Module) -> Optimizer:
    opt_type = optimizer_spec["type"]
    lr = float(optimizer_spec["lr"])
    wd = float(optimizer_spec.get("weight_decay", 0.0))
    params = model.parameters()
    if opt_type == "SGD":
        return SGD(
            params,
            lr=lr,
            momentum=float(optimizer_spec.get("momentum", 0.0)),
            weight_decay=wd,
        )
    if opt_type == "Adam":
        betas = optimizer_spec.get("betas", [0.9, 0.999])
        return Adam(params, lr=lr, betas=(float(betas[0]), float(betas[1])), weight_decay=wd)
    if opt_type == "AdamW":
        betas = optimizer_spec.get("betas", [0.9, 0.999])
        return AdamW(params, lr=lr, betas=(float(betas[0]), float(betas[1])), weight_decay=wd)
    if opt_type == "RMSprop":
        return RMSprop(params, lr=lr, weight_decay=wd)
    if opt_type == "Adagrad":
        return Adagrad(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def render_optimizer_py(optimizer_spec: dict[str, Any]) -> str:
    lines = ["    params = model.parameters()"]
    opt_type = optimizer_spec["type"]
    lr = optimizer_spec["lr"]
    wd = optimizer_spec.get("weight_decay", 0.0)
    if opt_type == "SGD":
        mom = optimizer_spec.get("momentum", 0.0)
        lines.append(
            f"    return torch.optim.SGD(params, lr={lr}, momentum={mom}, weight_decay={wd})"
        )
    elif opt_type in {"Adam", "AdamW"}:
        betas = optimizer_spec.get("betas", [0.9, 0.999])
        cls = f"torch.optim.{opt_type}"
        lines.append(
            f"    return {cls}(params, lr={lr}, betas=({betas[0]}, {betas[1]}), weight_decay={wd})"
        )
    elif opt_type == "RMSprop":
        lines.append(
            f"    return torch.optim.RMSprop(params, lr={lr}, weight_decay={wd})"
        )
    elif opt_type == "Adagrad":
        lines.append(
            f"    return torch.optim.Adagrad(params, lr={lr}, weight_decay={wd})"
        )
    else:
        raise ValueError(opt_type)
    body = "\n".join(lines)
    return f'''"""Optimizer factory for this candidate."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
{body}
'''
