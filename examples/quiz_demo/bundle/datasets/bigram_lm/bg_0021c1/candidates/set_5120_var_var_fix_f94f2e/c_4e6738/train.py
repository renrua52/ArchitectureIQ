"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _test_ce(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        if pred.ndim == 3:
            vocab = pred.shape[-1]
            loss = F.cross_entropy(pred.reshape(-1, vocab), test_y.reshape(-1))
        else:
            loss = F.cross_entropy(pred, test_y.reshape(-1))
        return float(loss.item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
) -> dict:
    torch.manual_seed(seed)
    model = Model()
    optimizer = build_optimizer(model)
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,))
        pred = model(train_x[idx])
        loss = loss_fn(model, pred, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric = _test_ce(model, test_x, test_y)
        if not math.isfinite(metric):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(metric)

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "failed": failed,
        "final_test_ce": final_metric,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }
