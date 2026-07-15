"""Natural-language formatters shared by prompt rendering."""

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
    lines = [
        "- Type: MLP",
    ]
    if "input_dim" in model and int(model["input_dim"]) > 1:
        lines.append(f"- Input dimension: {model['input_dim']}")
    lines.extend(
        [
            f"- Depth: {model['depth']} hidden layers",
            f"- Width: {model['width']} (all hidden layers)",
            f"- Residual connections: {model['residual']}",
            f"- Layer norm per layer: {model['layer_norm']}",
            _format_activation_line(model["activations"]),
        ]
    )
    if "leaky_relu" in model.get("activations", []):
        slope = float(model.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE))
        lines.append(f"- LeakyReLU negative slope: {slope:g}")
    lines.append("- Initialization: PyTorch Linear defaults")
    return "\n".join(lines)


def format_optimizer_nl(opt: dict) -> str:
    lines = [f"- Optimizer: {opt['type']}", f"- Learning rate: {opt['lr']}"]
    if "weight_decay" in opt:
        lines.append(f"- Weight decay: {opt['weight_decay']}")
    if opt["type"] == "SGD" and "momentum" in opt:
        lines.append(f"- Momentum: {opt['momentum']}")
    if opt["type"] in {"Adam", "AdamW"} and "betas" in opt:
        lines.append(f"- Betas: {opt['betas']}")
    return "\n".join(lines)


def format_model_nl(model: dict) -> str:
    model_type = model.get("type", "mlp")
    if model_type == "mlp":
        return format_mlp_nl(model)
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
        return "- Loss: cross-entropy on next-token labels"
    if loss["loss_id"] == "cross_entropy_l2":
        return (
            f"- Loss: cross-entropy + L2 weight penalty (lambda={loss['lambda']})"
        )
    if loss["loss_id"] == "cross_entropy_l1":
        return (
            f"- Loss: cross-entropy + L1 weight penalty (lambda={loss['lambda']})"
        )
    return f"- Loss: {loss['loss_id']}"


def format_regression_protocol(params: dict) -> str:
    point_seed = params.get("point_sampling", {}).get("seed", "—")
    domain = params.get("domain", [0.0, 1.0])
    expression = params.get("expression", "—")
    lines = [
        f"- Target expression (canonical): `{expression}`",
        f"- Train split size: {params['train_size']} fixed `(x, y)` pairs",
        f"- Test split size: {params['test_size']} fixed `(x, y)` pairs (held out)",
        f"- Input domain: [{domain[0]}, {domain[1]}], uniform sampling",
        f"- Point-sampling seed: {point_seed} (materializes the fixed train/test splits)",
        "- Minibatch construction: each step draws `batch_size` train indices "
        "uniformly at random **with replacement**",
        "- Evaluation: **test MSE** is mean squared error on the entire fixed test split",
        "- Randomness: `torch.manual_seed(seed)` once before model init and the training loop",
        "- Reference device: CPU",
    ]
    return "\n".join(lines)


def format_multivariate_protocol(params: dict) -> str:
    point_seed = params.get("point_sampling", {}).get("seed", "—")
    domain = params.get("domain", [0.0, 1.0])
    expression = params.get("expression", "—")
    return "\n".join(
        [
            f"- Target expression (canonical): `{expression}`",
            f"- Input dimension: {params['input_dim']}",
            f"- Train split size: {params['train_size']} fixed `(x, y)` pairs",
            f"- Test split size: {params['test_size']} fixed `(x, y)` pairs (held out)",
            f"- Input domain: [{domain[0]}, {domain[1]}] per coordinate, uniform sampling",
            f"- Point-sampling seed: {point_seed}",
            "- Evaluation: **test MSE** on the held-out split",
        ]
    )


def format_bigram_protocol(params: dict) -> str:
    return "\n".join(
        [
            f"- Vocab size: {params['vocab_size']}",
            f"- Context length: {params['context_length']}",
            "- Layout: causal LM windows (`x` shape `[N, L]`, `y` shape `[N, L]`)",
            "- One fixed bigram transition matrix `P(y|x)` shared by train and test",
            f"- Train rows: {params['train_size']}; test rows: {params['test_size']}",
            f"- Sequence seed: {params['sequence_seed']}; table seed: {params['table_seed']}",
            "- Evaluation: **test cross-entropy** on held-out windows",
        ]
    )


def format_dataset_protocol(params: dict, *, family: str | None = None) -> str:
    if family == "bigram_lm" or "vocab_size" in params:
        return format_bigram_protocol(params)
    if "input_dim" in params:
        return format_multivariate_protocol(params)
    return format_regression_protocol(params)


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


def format_ranking_protocol(*, n_seeds: int, base_seed: int, selection_metric: str) -> str:
    last_seed = base_seed + n_seeds - 1
    return "\n".join(
        [
            f"- Ground-truth ranking uses **{selection_metric}** on the held-out test split.",
            f"- Each choice is trained independently for **{n_seeds}** seeds "
            f"(`{base_seed}`..`{last_seed}`), one `torch.manual_seed(seed)` per run.",
            f"- The correct choice has the lowest **mean** {selection_metric} across seeds.",
        ]
    )
