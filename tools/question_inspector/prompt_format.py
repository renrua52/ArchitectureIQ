"""Prompt-aligned NL formatters (mirror of architecture_iq.prompts.formatters)."""

from __future__ import annotations

LEGACY_LEAKY_RELU_SLOPE = 0.1

SINGLE_AXIS_TYPES = frozenset({"architecture_only", "optimizer_only", "loss_only"})


def activation_nl(name: str) -> str:
    if name == "relu":
        return "ReLU (PyTorch defaults)"
    if name == "leaky_relu":
        return f"LeakyReLU(negative_slope={LEGACY_LEAKY_RELU_SLOPE})"
    if name == "gelu":
        return "GELU (PyTorch defaults)"
    if name == "silu":
        return "SiLU (PyTorch defaults)"
    return name


def _format_activation_line(acts: list[str]) -> str:
    return f"- Activations: [{', '.join(acts)}]"


def format_mlp_nl(model: dict) -> str:
    lines = ["- Type: MLP"]
    if "input_dim" in model and int(model["input_dim"]) > 1:
        lines.append(f"- Input dimension: {model['input_dim']}")
    if "output_dim" in model and int(model["output_dim"]) > 1:
        lines.append(f"- Output logits: {model['output_dim']}")
    lines.extend([
        f"- Depth: {model['depth']} hidden layers",
        f"- Width: {model['width']} (all hidden layers)",
        f"- Residual connections: {model['residual']}",
        f"- Layer norm per layer: {model['layer_norm']}",
        _format_activation_line(model["activations"]),
    ])
    if "leaky_relu" in model.get("activations", []):
        slope = float(model.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE))
        lines.append(f"- LeakyReLU negative slope: {slope:g}")
    lines.append("- Initialization: PyTorch Linear defaults")
    return "\n".join(lines)


def format_kan_nl(model: dict) -> str:
    lines = ["- Type: spline KAN (efficient_spline_v1)"]
    if "input_dim" in model and int(model["input_dim"]) > 1:
        lines.append(f"- Input dimension: {model['input_dim']}")
    if "output_dim" in model and int(model["output_dim"]) > 1:
        lines.append(f"- Output logits: {model['output_dim']}")
    lines.extend([
        f"- Depth: {model['depth']} hidden layers",
        f"- Width: {model['width']} (all hidden layers)",
        f"- Grid size: {model['grid_size']}",
        f"- Spline order: {model['spline_order']}",
        f"- Fixed grid range: {model['grid_range']}",
        f"- Base activation: {model['base_activation']}",
        "- Grid updates: fixed; no train/test-data adaptation",
    ])
    return "\n".join(lines)


def format_model_nl(model: dict) -> str:
    model_type = model.get("type", "mlp")
    if model_type == "mlp":
        return format_mlp_nl(model)
    if model_type == "kan":
        return format_kan_nl(model)
    if model_type == "transformer_lm":
        return format_transformer_lm_nl(model)
    return f"- Type: {model_type}"


def _transformer_dims(model: dict) -> tuple[int, int]:
    if "d_model" in model:
        d_model = int(model["d_model"])
    else:
        d_model = int(model["embed_dim"])
    if "d_ff" in model:
        d_ff = int(model["d_ff"])
    else:
        d_ff = int(model["ff_dim"])
    return d_model, d_ff


def format_transformer_lm_nl(model: dict) -> str:
    d_model, d_ff = _transformer_dims(model)
    return "\n".join(
        [
            "- Type: causal transformer LM",
            f"- Vocab size: {model['vocab_size']}",
            f"- Context length: {model['context_length']}",
            f"- d_model: {d_model}",
            f"- num_layers: {model['num_layers']}",
            f"- num_heads: {model['num_heads']}",
            f"- d_ff: {d_ff}",
        ]
    )


def format_model_spec_lines(model: dict) -> list[str]:
    """Compact lines for UI cards (strips markdown list prefixes from format_model_nl)."""
    lines: list[str] = []
    for line in format_model_nl(model).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            lines.append(stripped[2:])
        else:
            lines.append(stripped)
    return lines


def format_optimizer_nl(opt: dict) -> str:
    lines = [f"- Optimizer: {opt['type']}", f"- Learning rate: {opt['lr']}"]
    if "weight_decay" in opt:
        lines.append(f"- Weight decay: {opt['weight_decay']}")
    if opt["type"] == "SGD" and "momentum" in opt:
        lines.append(f"- Momentum: {opt['momentum']}")
    if opt["type"] in {"Adam", "AdamW"} and "betas" in opt:
        lines.append(f"- Betas: {opt['betas']}")
    return "\n".join(lines)


def format_loss_nl(loss: dict) -> str:
    if loss["loss_id"] == "mse":
        return "- Loss: mean squared error (MSE) on the minibatch"
    if loss["loss_id"] == "mse_l2":
        return (
            f"- Loss: MSE on the minibatch + L2 weight penalty "
            f"(lambda={loss['lambda']}, mean squared parameter magnitude)"
        )
    if loss["loss_id"] == "mse_l1":
        return (
            f"- Loss: MSE on the minibatch + L1 weight penalty "
            f"(lambda={loss['lambda']}, mean absolute parameter magnitude)"
        )
    if loss["loss_id"] == "cross_entropy":
        return "- Loss: cross-entropy on minibatch target labels"
    if loss["loss_id"] == "cross_entropy_l2":
        return f"- Loss: cross-entropy + L2 weight penalty (lambda={loss['lambda']})"
    if loss["loss_id"] == "cross_entropy_l1":
        return f"- Loss: cross-entropy + L1 weight penalty (lambda={loss['lambda']})"
    return f"- Loss: {loss['loss_id']}"


def format_training_schedule(budget: dict) -> str:
    steps = budget["training_steps"]
    batch_size = budget["batch_size"]
    total = budget["total_samples_seen"]
    return "\n".join(
        [
            f"- training_steps: {steps}",
            f"- batch_size: {batch_size}",
            f"- total_samples_seen: {total} (= training_steps × batch_size)",
        ]
    )


def _signed_linear_combination(terms: list[tuple[float, str]]) -> str:
    rendered: list[str] = []
    for index, (weight, expression) in enumerate(terms):
        sign = "-" if weight < 0 else "+"
        magnitude = f"{abs(float(weight)):.6g}"
        if index == 0:
            rendered.append(f"{sign if sign == '-' else ''}{magnitude}·{expression}")
        else:
            rendered.append(f"{sign} {magnitude}·{expression}")
    return " ".join(rendered)


def format_synthetic_tabular_classification_rule(params: dict) -> str:
    rule_family = params["rule_family"]
    active_features = [int(feature) for feature in params["active_features"]]
    weights = [float(weight) for weight in params["rule_weights"]]
    active = ", ".join(f"`x_{feature}`" for feature in active_features)

    if rule_family == "smooth_additive":
        terms = [
            (weight, f"[sin(x_{feature}) + 0.25·x_{feature}²]")
            for feature, weight in zip(active_features, weights, strict=True)
        ]
        score_lines = [f"  - `s(x) = {_signed_linear_combination(terms)}`"]
    elif rule_family == "sparse_interaction":
        pairs = [[int(value) for value in pair] for pair in params["interaction_pairs"]]
        terms = [
            (weight, f"x_{left}·x_{right}")
            for (left, right), weight in zip(pairs, weights, strict=True)
        ]
        score_lines = [f"  - `s(x) = {_signed_linear_combination(terms)}`"]
    elif rule_family == "piecewise_boundary":
        primary, secondary = active_features[:2]
        below_weight, above_weight, offset_weight = weights
        breakpoint = float(params["piecewise_breakpoint"])
        above = _signed_linear_combination(
            [(above_weight, f"x_{secondary}"), (offset_weight, f"x_{primary}")]
        )
        below = _signed_linear_combination(
            [(below_weight, f"x_{secondary}"), (offset_weight, f"x_{primary}")]
        )
        score_lines = [
            f"  - If `x_{primary} > {breakpoint:.6g}`: `s(x) = {above}`",
            f"  - Otherwise: `s(x) = {below}`",
        ]
    else:
        raise ValueError(f"Unknown classification rule family: {rule_family!r}")

    noise_std = float(params["noise_std"])
    threshold = float(params["decision_threshold"])
    calibration = params["calibration"]
    return "\n".join(
        [
            f"- Rule family: `{rule_family}`; active coordinates: {active}",
            "- Feature distribution: every coordinate is sampled independently from `Normal(0, 1)`.",
            "- Latent score:",
            *score_lines,
            f"- Label noise: `ε ~ Normal(0, {noise_std:.6g}²)`.",
            f"- Label rule: `y = 1` exactly when `s(x) + ε > {threshold:.6g}`; otherwise `y = 0`.",
            f"- Bayes decision boundary: without observing ε, predict class 1 when `s(x) > {threshold:.6g}`.",
            f"- Threshold calibration: `{threshold:.6g}` was estimated from {calibration['size']} independent calibration rows to target a positive-class rate of {float(calibration['target_positive_rate']):.0%}.",
            f"- Reproducibility: point/noise seed `{params['point_sampling']['seed']}`, calibration seed `{calibration['seed']}`.",
        ]
    )


def format_dataset_protocol(params: dict, *, family: str | None = None, device: str = "cpu") -> str:
    if "rule_family" in params:
        return "\n".join(
            [
                "- Task: binary classification on one fixed synthetic tabular train/test split.",
                f"- Input shape: float32 `[N, {params['input_dim']}]`; labels: int64 `[N]` in `{{0, 1}}`.",
                f"- Train rows: {params['train_size']}; held-out test rows: {params['test_size']}.",
                "- Every choice receives the same materialized split; minibatches sample train indices uniformly with replacement.",
                "- Evaluation: **test cross-entropy** is primary; test accuracy is auxiliary only.",
                f"- Reference device: {device}",
            ]
        )
    point_seed = params.get("point_sampling", {}).get("seed", "—")
    domain = params.get("domain", [0.0, 1.0])
    expression = params.get("expression", "—")
    lines = [
        f"- Target expression (canonical): `{expression}`",
        f"- Train split size: {params['train_size']} fixed `(x, y)` pairs",
        f"- Test split size: {params['test_size']} fixed `(x, y)` pairs (held out)",
        f"- Input domain: [{domain[0]}, {domain[1]}], uniform sampling",
        f"- Point-sampling seed: {point_seed} (materializes the fixed train/test splits)",
        "- Minibatch construction: each step draws `batch_size` train indices uniformly at random **with replacement**",
        "- Evaluation: **test MSE** is mean squared error on the entire fixed test split",
        "- Randomness: `torch.manual_seed(seed)` once before model init and the training loop",
        f"- Reference device: {device}",
    ]
    return "\n".join(lines)


def format_ranking_protocol(*, n_seeds: int, base_seed: int, selection_metric: str, device: str = "cpu") -> str:
    last_seed = base_seed + n_seeds - 1
    return "\n".join(
        [
            f"- Ground-truth ranking uses **{selection_metric}** on the held-out test split.",
            f"- Each choice is trained independently for **{n_seeds}** seeds "
            f"(`{base_seed}`..`{last_seed}`), one `torch.manual_seed(seed)` per run.",
            f"- Execution device: {device}.",
            f"- The correct choice has the lowest **mean** {selection_metric} across seeds.",
        ]
    )
