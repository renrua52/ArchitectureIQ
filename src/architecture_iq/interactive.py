"""Interactive CLI prompts for dataset and question generation."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import SINGLE_AXIS_TYPES, VARYING_AXIS_CHOICES
from architecture_iq.candidates.generator import (
    _pick_batch_size,
    sample_loss,
    sample_model,
    sample_optimizer,
    valid_batch_sizes,
)
from architecture_iq.candidates.sets import list_candidate_sets, load_set_manifest
from architecture_iq.datasets import create_dataset, list_dataset_instances
from architecture_iq.profile import Profile
from architecture_iq.util import read_json

InputFn = Callable[[str], str]
WriteFn = Callable[[str], None]


def _default_input(message: str) -> str:
    return input(message)


def _default_write(text: str) -> None:
    print(text, flush=True)


def prompt_line(
    message: str,
    *,
    input_fn: InputFn = _default_input,
) -> str:
    return input_fn(message).strip()


def _format_option(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        text = f"{value:g}"
        return text
    return str(value)


def _values_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - float(b)) < 1e-12
    return a == b


def prompt_grid_value(
    label: str,
    grid: list[Any],
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
    allow_random: bool = True,
) -> Any | None:
    """Pick from grid by index, exact value, or None for random."""
    if not grid:
        raise ValueError(f"No options for {label}")
    write(f"{label}:")
    for i, value in enumerate(grid, start=1):
        write(f"  {i}) {_format_option(value)}")
    if allow_random:
        write("  Enter = random")
    raw = prompt_line("> ", input_fn=input_fn)
    if allow_random and raw == "":
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(grid):
            return grid[idx - 1]
    for value in grid:
        if raw == _format_option(value):
            return value
    try:
        parsed: Any = float(raw) if any(isinstance(v, float) for v in grid) else int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {raw!r}") from exc
    for value in grid:
        if _values_match(parsed, value):
            return value
    raise ValueError(f"{label}={raw!r} not in allowed options: {grid}")


def prompt_choice(
    label: str,
    options: list[str],
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
    allow_random: bool = True,
) -> str | None:
    """Return selected option or None for random."""
    if not options:
        raise ValueError(f"No options for {label}")
    write(f"{label}:")
    for i, option in enumerate(options, start=1):
        write(f"  {i}) {option}")
    if allow_random:
        write("  Enter = random")
    raw = prompt_line("> ", input_fn=input_fn)
    if allow_random and raw == "":
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    if raw in options:
        return raw
    raise ValueError(f"Invalid {label}: {raw!r}")


def assemble_optimizer_spec(
    profile: Profile,
    rng: random.Random,
    *,
    opt_type: str | None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
) -> dict[str, Any]:
    if opt_type is None:
        return sample_optimizer(profile, rng)
    spec: dict[str, Any] = {
        "type": opt_type,
        "lr": lr if lr is not None else rng.choice(profile.optimizer_grids["lr"]),
        "weight_decay": (
            weight_decay
            if weight_decay is not None
            else rng.choice(profile.optimizer_grids["weight_decay"])
        ),
    }
    if opt_type == "SGD":
        spec["momentum"] = (
            momentum
            if momentum is not None
            else rng.choice(profile.optimizer_grids["sgd_momentum"])
        )
    if opt_type in {"Adam", "AdamW"}:
        betas = profile.optimizer_grids["adam_betas"]
        spec["betas"] = [float(betas[0]), float(betas[1])]
    return spec


def assemble_loss_spec(
    profile: Profile,
    family: str,
    rng: random.Random,
    *,
    loss_id: str | None,
    lambda_value: float | None = None,
) -> dict[str, Any]:
    if loss_id is None:
        return sample_loss(profile, family, rng)
    spec: dict[str, Any] = {"loss_id": loss_id}
    if loss_id in {"mse_l1", "mse_l2"}:
        spec["lambda"] = (
            lambda_value
            if lambda_value is not None
            else rng.choice(profile.loss_grids["lambda"])
        )
    return spec


def assemble_model_spec(
    profile: Profile,
    rng: random.Random,
    *,
    depth: int | None = None,
    width: int | None = None,
    residual: bool | None = None,
    activation: str | None = None,
    activations: list[str] | None = None,
    layer_norm: list[bool] | None = None,
) -> dict[str, Any]:
    if all(
        v is None
        for v in (depth, width, residual, activation, activations, layer_norm)
    ):
        return sample_model(profile, rng)
    cfg = profile.mlp
    d = depth if depth is not None else rng.choice(cfg["depth"])
    w = width if width is not None else rng.choice(cfg["width"])
    r = residual if residual is not None else rng.choice(cfg["residual"])
    if activations is not None:
        if len(activations) != d:
            raise ValueError("activations length must match depth")
        acts = activations
    else:
        acts = [
            activation if activation is not None else rng.choice(cfg["activations"])
            for _ in range(d)
        ]
    if layer_norm is not None:
        if len(layer_norm) != d:
            raise ValueError("layer_norm length must match depth")
        norms = layer_norm
    else:
        norms = [rng.choice([True, False]) for _ in range(d)]
    return {
        "type": "mlp",
        "depth": d,
        "width": w,
        "residual": r,
        "layer_norm": norms,
        "activations": acts,
    }


def prompt_optimizer_spec(
    profile: Profile,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    opt_type = prompt_choice(
        "Optimizer",
        list(profile.pools["optimizers"]),
        input_fn=input_fn,
        write=write,
    )
    lr_raw = prompt_grid_value(
        "Learning rate",
        list(profile.optimizer_grids["lr"]),
        input_fn=input_fn,
        write=write,
    )
    wd_raw = prompt_grid_value(
        "Weight decay",
        list(profile.optimizer_grids["weight_decay"]),
        input_fn=input_fn,
        write=write,
    )
    momentum_raw: float | None = None
    if opt_type == "SGD":
        momentum_raw = prompt_grid_value(
            "Momentum",
            list(profile.optimizer_grids["sgd_momentum"]),
            input_fn=input_fn,
            write=write,
        )
    return assemble_optimizer_spec(
        profile,
        rng,
        opt_type=opt_type,
        lr=lr_raw,
        weight_decay=wd_raw,
        momentum=momentum_raw,
    )


def prompt_loss_spec(
    profile: Profile,
    family: str,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    loss_id = prompt_choice(
        "Loss",
        list(profile.pools["losses"][family]),
        input_fn=input_fn,
        write=write,
    )
    lambda_raw: float | None = None
    if loss_id in {"mse_l1", "mse_l2"}:
        lambda_raw = prompt_grid_value(
            "Loss lambda (for L1/L2 penalties)",
            list(profile.loss_grids["lambda"]),
            input_fn=input_fn,
            write=write,
        )
    return assemble_loss_spec(
        profile,
        family,
        rng,
        loss_id=loss_id,
        lambda_value=lambda_raw,
    )


def prompt_layer_norm_flags(
    depth: int,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> list[bool]:
    flags: list[bool] = []
    for layer in range(1, depth + 1):
        picked = prompt_grid_value(
            f"Layer {layer} layer norm",
            [True, False],
            input_fn=input_fn,
            write=write,
        )
        flags.append(bool(picked) if picked is not None else rng.choice([True, False]))
    return flags


def prompt_layer_activations(
    profile: Profile,
    depth: int,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> list[str]:
    options = list(profile.mlp["activations"])
    activations: list[str] = []
    for layer in range(1, depth + 1):
        picked = prompt_choice(
            f"Layer {layer} activation",
            options,
            input_fn=input_fn,
            write=write,
        )
        activations.append(picked if picked is not None else rng.choice(options))
    return activations


def prompt_model_spec(
    profile: Profile,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    depth_raw = prompt_grid_value(
        "Model depth",
        list(profile.mlp["depth"]),
        input_fn=input_fn,
        write=write,
    )
    width_raw = prompt_grid_value(
        "Model width",
        list(profile.mlp["width"]),
        input_fn=input_fn,
        write=write,
    )
    residual_raw = prompt_grid_value(
        "Residual connections",
        list(profile.mlp["residual"]),
        input_fn=input_fn,
        write=write,
    )

    cfg = profile.mlp
    depth = int(depth_raw) if depth_raw is not None else rng.choice(cfg["depth"])
    write(f"Per-layer settings for depth={depth} (Enter = random):")
    activations = prompt_layer_activations(
        profile, depth, rng, input_fn=input_fn, write=write
    )
    layer_norm = prompt_layer_norm_flags(depth, rng, input_fn=input_fn, write=write)

    return assemble_model_spec(
        profile,
        rng,
        depth=depth,
        width=int(width_raw) if width_raw is not None else None,
        residual=bool(residual_raw) if residual_raw is not None else None,
        activations=activations,
        layer_norm=layer_norm,
    )


def prompt_batch_size(
    profile: Profile,
    budget: int,
    rng: random.Random,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> int:
    options = valid_batch_sizes(profile, budget)
    if not options:
        raise ValueError(f"No batch size divides budget {budget}")
    picked = prompt_grid_value(
        "Batch size",
        options,
        input_fn=input_fn,
        write=write,
    )
    return picked if picked is not None else _pick_batch_size(profile, budget, rng)


def prompt_fixed_components(
    profile: Profile,
    *,
    question_type: str,
    family: str,
    budget: int,
    rng: random.Random,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    """Prompt for fixed candidate fields for single-axis question types."""
    if question_type not in SINGLE_AXIS_TYPES:
        raise ValueError(f"question_type must be one of {sorted(SINGLE_AXIS_TYPES)}")

    write(f"Fixed components for {question_type} (Enter = random sample):")

    fixed: dict[str, Any] = {
        "batch_size": prompt_batch_size(
            profile, budget, rng, input_fn=input_fn, write=write
        ),
    }

    if question_type == "architecture_only":
        fixed["loss"] = prompt_loss_spec(
            profile, family, rng, input_fn=input_fn, write=write
        )
        fixed["optimizer"] = prompt_optimizer_spec(
            profile, rng, input_fn=input_fn, write=write
        )
    elif question_type == "optimizer_only":
        fixed["model"] = prompt_model_spec(
            profile, rng, input_fn=input_fn, write=write
        )
        fixed["loss"] = prompt_loss_spec(
            profile, family, rng, input_fn=input_fn, write=write
        )
    elif question_type == "loss_only":
        fixed["model"] = prompt_model_spec(
            profile, rng, input_fn=input_fn, write=write
        )
        fixed["optimizer"] = prompt_optimizer_spec(
            profile, rng, input_fn=input_fn, write=write
        )

    return fixed


def _dataset_label(path: Path) -> str:
    spec = read_json(path / "dataset_spec.json")
    expr = spec.get("params", {}).get("expression", "?")
    if len(expr) > 60:
        expr = expr[:57] + "..."
    return f"{spec['dataset_id']} — {expr}"


def prompt_dataset_family(
    profile: Profile,
    *,
    rng: random.Random | None = None,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> str:
    families = list(profile.pools["dataset_families"])
    picker = rng if rng is not None else random.Random()
    picked = prompt_choice(
        "Dataset family",
        families,
        input_fn=input_fn,
        write=write,
        allow_random=True,
    )
    return picked if picked is not None else picker.choice(families)


def prompt_dataset_instance(
    profile: Profile,
    *,
    family: str | None = None,
    rng: random.Random | None = None,
    data_dir: Path | None = None,
    allow_create: bool = True,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> tuple[Path | None, str]:
    """Return ``(existing path, family)`` or ``(None, family)`` to create new."""
    resolved_family = family or prompt_dataset_family(
        profile, rng=rng, input_fn=input_fn, write=write
    )
    instances = list_dataset_instances(data_dir, family=resolved_family)
    write(f"\nDataset pool ({resolved_family}):")
    if instances:
        for i, entry in enumerate(instances, start=1):
            write(f"  {i}) {_dataset_label(entry.path)}")
    else:
        write("  (no existing datasets)")
    if allow_create:
        write(f"  n) Create new {resolved_family} dataset")
    elif not instances:
        raise ValueError(
            f"No datasets found for {resolved_family!r}. "
            "Run create-dataset first, then retry."
        )
    raw = prompt_line("> ", input_fn=input_fn)
    if not instances:
        if allow_create:
            return None, resolved_family
        raise ValueError(
            f"No datasets found for {resolved_family!r}. "
            "Run create-dataset first, then retry."
        )
    if raw.lower() in {"n", "new"}:
        if allow_create:
            return None, resolved_family
        raise ValueError(
            "Dataset creation is not available in this command. Run create-dataset first."
        )
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(instances):
            return instances[idx - 1].path, resolved_family
    raise ValueError(f"Invalid dataset choice: {raw!r}")


def prompt_int(
    label: str,
    *,
    default: int | None = None,
    rng: random.Random | None = None,
    random_if_empty: bool = False,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> int:
    if random_if_empty and rng is not None:
        hint = "random"
    elif default is not None:
        hint = str(default)
    else:
        hint = "required"
    raw = prompt_line(f"{label} [{hint}]: ", input_fn=input_fn)
    if raw == "":
        if random_if_empty and rng is not None:
            return rng.randint(0, 2**31 - 1)
        if default is not None:
            return default
        raise ValueError(f"{label} is required")
    return int(raw)


def prompt_fixed_components_for_axes(
    profile: Profile,
    *,
    family: str,
    budget: int,
    invariant_axes: frozenset[str],
    rng: random.Random,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    """Prompt for values pinned on invariant axes in a candidate set."""
    write(
        "Fixed components for invariant axes "
        f"({', '.join(sorted(invariant_axes))}; Enter = random sample):"
    )
    fixed: dict[str, Any] = {
        "batch_size": prompt_batch_size(
            profile, budget, rng, input_fn=input_fn, write=write
        ),
    }
    if "model" in invariant_axes:
        fixed["model"] = prompt_model_spec(
            profile, rng, input_fn=input_fn, write=write
        )
    if "optimizer" in invariant_axes:
        fixed["optimizer"] = prompt_optimizer_spec(
            profile, rng, input_fn=input_fn, write=write
        )
    if "loss" in invariant_axes:
        fixed["loss"] = prompt_loss_spec(
            profile, family, rng, input_fn=input_fn, write=write
        )
    return fixed


def prompt_varying_axes(
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> frozenset[str]:
    write("Varying axes (comma-separated subset of model, optimizer, loss; Enter = model only):")
    write("  Example: model | model,optimizer | model,optimizer,loss")
    raw = prompt_line("> ", input_fn=input_fn).strip().lower()
    if raw == "":
        return frozenset({"model"})
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    from architecture_iq.candidates.sets import parse_varying_axes

    return parse_varying_axes(parts)


def prompt_candidate_sets(
    dataset_path: Path,
    *,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> list[Path]:
    sets = list_candidate_sets(dataset_path)
    if not sets:
        raise ValueError(f"No candidate sets found under {dataset_path / 'candidates'}")
    write("Candidate sets (comma-separated indices):")
    for i, path in enumerate(sets, start=1):
        manifest = load_set_manifest(path)
        write(
            f"  {i}) {path.name} — budget={manifest['budget']['total_samples_seen']} "
            f"vary={manifest['varying_axes']}"
        )
    raw = prompt_line("> ", input_fn=input_fn)
    indices: list[int] = []
    for token in raw.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid candidate set index: {token!r}")
        indices.append(int(token))
    if not indices:
        raise ValueError("Select at least one candidate set")
    picked: list[Path] = []
    for idx in indices:
        if not 1 <= idx <= len(sets):
            raise ValueError(f"Invalid candidate set index: {idx}")
        picked.append(sets[idx - 1])
    return picked


def interactive_generate_candidate_set(
    profile: Profile,
    *,
    dataset_path: Path | None = None,
    rng: random.Random,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    resolved_dataset = dataset_path or interactive_select_dataset_path(
        profile,
        rng=rng,
        allow_create=False,
        input_fn=input_fn,
        write=write,
    )
    ds = read_json(resolved_dataset / "dataset_spec.json")

    budget_raw = prompt_grid_value(
        "Budget (total_samples_seen)",
        profile.budget_values,
        input_fn=input_fn,
        write=write,
    )
    budget = int(budget_raw) if budget_raw is not None else rng.choice(profile.budget_values)

    count = prompt_int("Number of candidates", default=32, input_fn=input_fn, write=write)
    varying_axes = prompt_varying_axes(input_fn=input_fn, write=write)
    invariant_axes = (VARYING_AXIS_CHOICES - varying_axes) | frozenset({"batch_size"})

    fixed_shared = prompt_fixed_components_for_axes(
        profile,
        family=ds["family"],
        budget=budget,
        invariant_axes=invariant_axes,
        rng=rng,
        input_fn=input_fn,
        write=write,
    )

    seed = prompt_int(
        "RNG seed",
        rng=rng,
        random_if_empty=True,
        input_fn=input_fn,
        write=write,
    )

    return {
        "dataset_path": resolved_dataset,
        "budget": budget,
        "count": count,
        "varying_axes": varying_axes,
        "fixed_shared": fixed_shared,
        "seed": seed,
    }


def interactive_generate_questions(
    profile: Profile,
    *,
    rng: random.Random,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> dict[str, Any]:
    """Prompt for generate-question parameters."""
    dataset_path = interactive_select_dataset_path(
        profile,
        rng=rng,
        allow_create=False,
        input_fn=input_fn,
        write=write,
    )
    candidate_set_paths = prompt_candidate_sets(
        dataset_path, input_fn=input_fn, write=write
    )

    num_choices = prompt_int(
        "Number of choices",
        default=profile.num_choices,
        input_fn=input_fn,
        write=write,
    )
    if num_choices < 2:
        raise ValueError("num_choices must be at least 2")

    num_questions = prompt_int(
        "Number of questions",
        default=1,
        input_fn=input_fn,
        write=write,
    )
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")

    seed = prompt_int(
        "RNG seed",
        rng=rng,
        random_if_empty=True,
        input_fn=input_fn,
        write=write,
    )

    return {
        "dataset_path": dataset_path,
        "candidate_set_paths": candidate_set_paths,
        "num_choices": num_choices,
        "num_questions": num_questions,
        "seed": seed,
    }


def _prompt_and_create_dataset(
    profile: Profile,
    family: str,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> tuple[dict[str, Any], Path]:
    picker = rng if rng is not None else random.Random()
    if seed is None:
        seed = prompt_int(
            "Instance seed",
            rng=picker,
            random_if_empty=True,
            input_fn=input_fn,
            write=write,
        )

    spec, path = create_dataset(profile, seed, family_name=family)
    write(f"Created dataset {spec['dataset_id']} at {path}")
    return spec, path


def interactive_create_dataset(
    profile: Profile,
    *,
    rng: random.Random | None = None,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> tuple[dict[str, Any], Path]:
    """Prompt for family and seed, then materialize a new dataset instance."""
    resolved_family = prompt_dataset_family(
        profile, rng=rng, input_fn=input_fn, write=write
    )
    return _prompt_and_create_dataset(
        profile,
        resolved_family,
        seed=None,
        rng=rng,
        input_fn=input_fn,
        write=write,
    )


def interactive_select_dataset_path(
    profile: Profile,
    *,
    family: str | None = None,
    rng: random.Random | None = None,
    data_dir: Path | None = None,
    allow_create: bool = True,
    input_fn: InputFn = _default_input,
    write: WriteFn = _default_write,
) -> Path:
    """Pick an existing dataset, optionally creating a new one."""
    existing, resolved_family = prompt_dataset_instance(
        profile,
        family=family,
        rng=rng,
        data_dir=data_dir,
        allow_create=allow_create,
        input_fn=input_fn,
        write=write,
    )
    if existing is not None:
        return existing
    spec, path = _prompt_and_create_dataset(
        profile,
        resolved_family,
        rng=rng,
        input_fn=input_fn,
        write=write,
    )
    write(f"Expression: {spec['params']['expression']}")
    return path
