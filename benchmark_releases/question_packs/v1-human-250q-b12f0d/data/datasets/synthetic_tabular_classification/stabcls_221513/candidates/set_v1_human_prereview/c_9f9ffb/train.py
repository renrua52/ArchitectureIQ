"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_metrics(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(test_x)
        ce = F.cross_entropy(logits, test_y.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == test_y.reshape(-1)).float().mean()
    return float(ce.item()), float(accuracy.item())


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
    device: str = "cpu",
    progress_callback=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)
    final_accuracy = float("nan")

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        logits = model(train_x[idx])
        loss = loss_fn(model, logits, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        ce, accuracy = _test_metrics(model, test_x, test_y)
        if not math.isfinite(ce) or not math.isfinite(accuracy):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(ce)
        final_accuracy = accuracy
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": ce,
                    "accuracy": accuracy,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "failed": failed,
        "final_test_ce": final_metric,
        "final_test_accuracy": final_accuracy,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }
