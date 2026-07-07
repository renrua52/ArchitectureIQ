"""Natural-language formatters shared by prompt rendering."""

from __future__ import annotations

LEAKY_RELU_SLOPE = 0.1

SINGLE_AXIS_TYPES = frozenset({"architecture_only", "optimizer_only", "loss_only"})


def activation_nl(name: str) -> str:
    if name == "relu":
        return "ReLU (PyTorch defaults)"
    if name == "leaky_relu":
        return f"LeakyReLU(negative_slope={LEAKY_RELU_SLOPE})"
    if name == "gelu":
        return "GELU (PyTorch defaults)"
    if name == "silu":
        return "SiLU (PyTorch defaults)"
    return name


def format_mlp_nl(model: dict) -> str:
    acts = model["activations"]
    act_lines = [
        f"  - Layer {i + 1}: {activation_nl(name)}"
        for i, name in enumerate(acts)
    ]
    lines = [
        "- Type: MLP",
        f"- Depth: {model['depth']} hidden layers",
        f"- Width: {model['width']} (all hidden layers)",
        f"- Residual connections: {model['residual']}",
        f"- Layer norm per layer: {model['layer_norm']}",
        "- Activations per layer:",
        *act_lines,
        "- Initialization: PyTorch Linear defaults",
    ]
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


def format_dataset_protocol(params: dict) -> str:
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
