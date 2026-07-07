from __future__ import annotations

from typing import Any

import torch.nn as nn

from architecture_iq.losses import language, regression


def render_loss_py(loss_spec: dict[str, Any]) -> str:
    loss_id = loss_spec["loss_id"]
    if loss_id.startswith("cross_entropy"):
        return language.render_loss_py(loss_spec)
    return regression.render_loss_py(loss_spec)


def compute_loss(
    loss_spec: dict[str, Any],
    model: nn.Module,
    pred: Any,
    target: Any,
) -> Any:
    loss_id = loss_spec["loss_id"]
    if loss_id.startswith("cross_entropy"):
        return language.compute_loss(loss_spec, model, pred, target)
    return regression.compute_loss(loss_spec, model, pred, target)
