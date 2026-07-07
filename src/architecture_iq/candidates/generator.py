from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from architecture_iq.losses import render_loss_py
from architecture_iq.models.base import ModelFamily
from architecture_iq.optimizers.factory import render_optimizer_py
from architecture_iq.profile import Profile
from architecture_iq.registry import get_model_type
from architecture_iq.util import short_hash, write_json

REGRESSION_TRAIN_PY = '''"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math

import torch

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _test_mse(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        return float(torch.mean((pred - test_y) ** 2).item())


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

        metric = _test_mse(model, test_x, test_y)
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
        "final_test_mse": final_metric,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }


def train(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
) -> None:
    """Minimal training entrypoint (no evaluation)."""
    train_and_eval(
        train_x,
        train_y,
        test_x=train_x,
        test_y=train_y,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        fail_threshold=float("inf"),
    )
'''

LM_TRAIN_PY = '''"""Training loop for this candidate — executed by the ground-truth runner."""
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
'''


def _train_py_for_family(family: str) -> str:
    if family == "bigram_lm":
        return LM_TRAIN_PY
    return REGRESSION_TRAIN_PY


from architecture_iq.candidates.axes import SINGLE_AXIS_TYPES, choices_compatible


def _spec_json(spec: dict[str, Any], key: str) -> str:
    return json.dumps(spec[key], sort_keys=True)


def _varying_axes_for_question_type(question_type: str) -> frozenset[str]:
    if question_type == "architecture_only":
        return frozenset({"model"})
    if question_type == "optimizer_only":
        return frozenset({"optimizer"})
    if question_type == "loss_only":
        return frozenset({"loss"})
    if question_type == "mixed":
        return frozenset({"model", "optimizer", "loss"})
    raise ValueError(f"Unknown question type: {question_type}")


def candidate_matches_fixed(spec: dict[str, Any], fixed_shared: dict[str, Any]) -> bool:
    for key, value in fixed_shared.items():
        if key == "batch_size":
            if spec["budget"]["batch_size"] != value:
                return False
        elif key in ("model", "optimizer", "loss"):
            if _spec_json(spec, key) != json.dumps(value, sort_keys=True):
                return False
        else:
            raise ValueError(f"Unknown fixed_shared key: {key}")
    return True


def valid_batch_sizes(profile: Profile, budget: int) -> list[int]:
    return [
        b
        for b in profile.optimizer_grids["batch_size"]
        if budget % b == 0
    ]


def _pick_batch_size(profile: Profile, budget: int, rng: random.Random) -> int:
    valid = valid_batch_sizes(profile, budget)
    if not valid:
        raise ValueError(f"No batch size divides budget {budget}")
    return rng.choice(valid)


def sample_optimizer(profile: Profile, rng: random.Random) -> dict[str, Any]:
    opt_type = rng.choice(profile.pools["optimizers"])
    spec: dict[str, Any] = {
        "type": opt_type,
        "lr": rng.choice(profile.optimizer_grids["lr"]),
        "weight_decay": rng.choice(profile.optimizer_grids["weight_decay"]),
    }
    if opt_type == "SGD":
        spec["momentum"] = rng.choice(profile.optimizer_grids["sgd_momentum"])
    if opt_type in {"Adam", "AdamW"}:
        betas = profile.optimizer_grids["adam_betas"]
        spec["betas"] = [float(betas[0]), float(betas[1])]
    return spec


def sample_loss(profile: Profile, family: str, rng: random.Random) -> dict[str, Any]:
    loss_id = rng.choice(profile.pools["losses"][family])
    spec: dict[str, Any] = {"loss_id": loss_id}
    if loss_id in {"mse_l1", "mse_l2"}:
        spec["lambda"] = rng.choice(profile.loss_grids["lambda"])
    return spec


def sample_model(
    profile: Profile,
    rng: random.Random,
    *,
    family: str,
    dataset_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from architecture_iq.registry import get_dataset_family

    family_obj = get_dataset_family(family)
    allowed = set(family_obj.compatible_model_types())
    model_types = [m for m in profile.pools["model_types"] if m in allowed]
    if not model_types:
        raise ValueError(f"No compatible model types for family {family!r}")
    model_type = rng.choice(model_types)
    return get_model_type(model_type).sample_spec(profile, rng, dataset_params=dataset_params)


def build_candidate_spec(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    batch_size: int,
    model: dict[str, Any],
    optimizer: dict[str, Any],
    loss: dict[str, Any],
) -> dict[str, Any]:
    steps = profile.training_steps(budget, batch_size)
    body = {
        "schema_version": profile.schema_version,
        "dataset_id": dataset_id,
        "family": family,
        "budget": {
            "training_steps": steps,
            "batch_size": batch_size,
            "total_samples_seen": budget,
        },
        "model": model,
        "optimizer": optimizer,
        "loss": loss,
        "files": {
            "model": "model.py",
            "train": "train.py",
            "loss": "loss.py",
            "optimizer": "optimizer.py",
        },
    }
    body["candidate_id"] = f"c_{short_hash(body)}"
    return body


def write_candidate(
    spec: dict[str, Any],
    out_dir: Path,
    model_family: ModelFamily,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "candidate_spec.json", spec)
    (out_dir / "model.py").write_text(
        model_family.render_model_py(spec["model"]), encoding="utf-8"
    )
    (out_dir / "loss.py").write_text(
        render_loss_py(spec["loss"]), encoding="utf-8"
    )
    (out_dir / "optimizer.py").write_text(
        render_optimizer_py(spec["optimizer"]), encoding="utf-8"
    )
    (out_dir / "train.py").write_text(_train_py_for_family(spec["family"]), encoding="utf-8")
    return out_dir


def sample_candidate(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    rng: random.Random,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fixed = fixed or {}
    batch_size = fixed.get("batch_size") or _pick_batch_size(profile, budget, rng)
    dataset_params = fixed.get("_dataset_params")
    model = fixed.get("model") or sample_model(
        profile, rng, family=family, dataset_params=dataset_params
    )
    optimizer = fixed.get("optimizer") or sample_optimizer(profile, rng)
    loss = fixed.get("loss") or sample_loss(profile, family, rng)
    return build_candidate_spec(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        batch_size=batch_size,
        model=model,
        optimizer=optimizer,
        loss=loss,
    )


def sample_variant_pool(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    question_type: str,
    pool_size: int,
    rng: random.Random,
    fixed_shared: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from architecture_iq.candidates.sets import sample_candidate_set_pool

    varying_axes = _varying_axes_for_question_type(question_type)
    if question_type == "mixed" and fixed_shared is None:
        varying_axes = frozenset({"model", "optimizer", "loss"})
    return sample_candidate_set_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        count=pool_size,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=fixed_shared,
    )


def sample_variants_for_question(
    profile: Profile,
    *,
    dataset_id: str,
    family: str,
    budget: int,
    question_type: str,
    num_choices: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    return sample_variant_pool(
        profile,
        dataset_id=dataset_id,
        family=family,
        budget=budget,
        question_type=question_type,
        pool_size=num_choices,
        rng=rng,
    )
