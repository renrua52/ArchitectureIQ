"""Natural-language formatters shared by prompt rendering."""

from __future__ import annotations


LEGACY_LEAKY_RELU_SLOPE = 0.1

SINGLE_AXIS_TYPES = frozenset({"architecture_only", "optimizer_only", "loss_only"})

# Dataset families that share the tabular-classification spec shape: a params
# dict carrying rule_family / active_features / rule_weights, rendered by
# format_synthetic_tabular_classification_rule. XOR and spiral are separate
# families (separate benchmark buckets) but reuse the same spec and renderer.
TABULAR_CLASSIFICATION_FAMILIES = frozenset(
    {
        "synthetic_tabular_classification",
        "xor_classification",
        "spiral_classification",
    }
)


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


def _mlp_activations(model: dict) -> list[str]:
    """Activations in layer order; a current spec has exactly one, shared.

    The canonical field is the scalar ``activation``. Pre-v1.4 artifacts on
    disk carry a per-layer ``activations`` list instead: a uniform one reads as
    the single shared value, and a genuinely mixed one is still displayed
    as-is so old bundles remain readable.
    """
    activation = model.get("activation")
    if activation is not None:
        return [str(activation)]
    legacy = [str(value) for value in (model.get("activations") or [])]
    if not legacy:
        raise ValueError("MLP spec is missing 'activation'")
    distinct = sorted(set(legacy))
    return [distinct[0]] if len(distinct) == 1 else legacy


def _format_activation_line(acts: list[str]) -> str:
    if len(acts) == 1:
        return f"- Activation: {acts[0]} (one activation, shared by every layer)"
    return f"- Activations (per layer, legacy spec): [{', '.join(acts)}]"


def _mlp_block_formula(model: dict) -> str:
    """Exact hidden-layer computation, so the NL cannot drift from model.py."""

    def form(with_norm: bool) -> str:
        inner = "Linear(LayerNorm(x))" if with_norm else "Linear(x)"
        return f"act(x + {inner})" if bool(model["residual"]) else f"act({inner})"

    norms = [bool(v) for v in model.get("layer_norm", [])]
    if norms and all(norms):
        return form(True)
    if not any(norms):
        return form(False)
    # layer_norm is per-layer, so a mixed pattern needs both forms spelled out.
    return f"{form(True)} for a layer with layer norm, {form(False)} for a layer without"


def format_mlp_nl(model: dict) -> str:
    lines = [
        "- Type: MLP",
    ]
    if "input_dim" in model and int(model["input_dim"]) > 1:
        lines.append(f"- Input dimension: {model['input_dim']}")
    if "output_dim" in model and int(model["output_dim"]) > 1:
        lines.append(f"- Output logits: {model['output_dim']}")
    depth = int(model["depth"])
    width = int(model["width"])
    plural = "layer" if depth == 1 else "layers"
    acts = _mlp_activations(model)
    lines.extend(
        [
            f"- Depth: {depth} hidden Linear {plural} of width {width}, between the "
            f"input projection and the output head "
            f"({depth + 2} nn.Linear layers in total)",
            f"- Width: {width} (all hidden layers)",
            f"- Hidden layer: {_mlp_block_formula(model)}",
            f"- Residual connections: {model['residual']}",
            f"- Layer norm per layer: {model['layer_norm']}",
            _format_activation_line(acts),
        ]
    )
    if "leaky_relu" in acts:
        slope = float(model.get("leaky_relu_slope", LEGACY_LEAKY_RELU_SLOPE))
        lines.append(f"- LeakyReLU negative slope: {slope:g}")
    lines.append("- Initialization: PyTorch Linear defaults")
    return "\n".join(lines)


def format_gru_lm_nl(model: dict) -> str:
    layer_residual = bool(model.get("layer_residual", False))
    residual_line = (
        "- Layer residual connections: enabled; after each GRU layer, "
        "h = h + GRU_layer(h)."
        if layer_residual
        else "- Layer residual connections: disabled"
    )
    return "\n".join(
        [
            "- Type: causal unidirectional GRU LM",
            f"- Vocab size: {model['vocab_size']}",
            f"- Context length: {model['context_length']}",
            f"- d_model (embedding and hidden size): {model['d_model']}",
            f"- num_layers: {model['num_layers']}",
            residual_line,
            "- No attention, position embedding, or dropout",
        ]
    )


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
    if model_type == "gru_lm":
        return format_gru_lm_nl(model)
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
            "- Positional encoding: learned embedding (nn.Embedding over positions)",
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
        return "- Loss: cross-entropy on minibatch target labels"
    if loss["loss_id"] == "cross_entropy_l2":
        return (
            f"- Loss: cross-entropy + L2 weight penalty (lambda={loss['lambda']})"
        )
    if loss["loss_id"] == "cross_entropy_l1":
        return (
            f"- Loss: cross-entropy + L1 weight penalty (lambda={loss['lambda']})"
        )
    return f"- Loss: {loss['loss_id']}"


def _format_noise_line(params: dict) -> str | None:
    """Observation-noise line, or ``None`` when the spec carries no noise.

    Datasets generated from v1.4 onwards have no noise anywhere: ``y`` is the
    exact evaluation of the target, and the prompt simply says nothing about
    noise rather than announcing its absence. Older artifacts that really do
    carry noise keep describing it -- the prompt has to match the data that was
    materialised.
    """
    noise = params.get("noise") or {}
    if not noise.get("enabled"):
        return None
    sigma = noise.get("sigma", noise.get("std", "—"))
    return f"- Noise: Gaussian observation noise with sigma={sigma} added to `y`"


def _noise_lines(params: dict) -> list[str]:
    """``_format_noise_line`` as 0 or 1 lines, for splicing into a line list."""
    line = _format_noise_line(params)
    return [line] if line else []


def format_regression_protocol(params: dict, *, device: str = "cpu") -> str:
    point_seed = params.get("point_sampling", {}).get("seed", "—")
    domain = params.get("domain", [0.0, 1.0])
    expression = params.get("expression", "—")
    lines = [
        f"- Target expression (canonical): `{expression}`",
        f"- Train split size: {params['train_size']} fixed `(x, y)` pairs",
        f"- Test split size: {params['test_size']} fixed `(x, y)` pairs (held out)",
        f"- Input domain: [{domain[0]}, {domain[1]}], uniform sampling",
        f"- Point-sampling seed: {point_seed} (materializes the fixed train/test splits)",
        *_noise_lines(params),
        "- Minibatch construction: each step draws `batch_size` train indices "
        "uniformly at random **with replacement**",
        "- Evaluation: **test MSE** is mean squared error on the entire fixed test split",
        "- Randomness: `torch.manual_seed(seed)` once before model init and the training loop",
        f"- Reference device: {device}",
    ]
    return "\n".join(lines)


def format_multivariate_protocol(params: dict, *, device: str = "cpu") -> str:
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
            *_noise_lines(params),
            "- Evaluation: **test MSE** on the held-out split",
            f"- Reference device: {device}",
        ]
    )


def format_bigram_protocol(params: dict, *, device: str = "cpu") -> str:
    return "\n".join(
        [
            f"- Vocab size: {params['vocab_size']}",
            f"- Context length: {params['context_length']}",
            "- Layout: causal LM windows (`x` shape `[N, L]`, `y` shape `[N, L]`)",
            "- One fixed bigram transition matrix `P(y|x)` shared by train and test",
            f"- Train rows: {params['train_size']}; test rows: {params['test_size']}",
            f"- Sequence seed: {params['sequence_seed']}; table seed: {params['table_seed']}",
            "- Evaluation: **test cross-entropy** on held-out windows",
            f"- Reference device: {device}",
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


def _turns_nl(turns: float) -> str:
    """`1 full turn` / `2.5 full turns` -- the grader's model reads this prose."""
    return f"{turns:.6g} full turn" + ("" if turns == 1.0 else "s")


def _spiral_t_upper_nl(turns: float) -> str:
    """`0.5 + 2π` -- the arm's parameter range, symbolically.

    Printing the product gave `[0.5, 6.78319]`, a number that hides the one
    fact the reader wants: the arm runs a whole number of half-turns. Every
    profile turns value makes `2·turns` an integer, so the multiple of π is
    exact; a non-integer multiple falls back to the decimal.
    """
    multiple = 2.0 * float(turns)
    if abs(multiple - round(multiple)) > 1e-9:
        return f"0.5 + {multiple:.6g}π"
    whole = int(round(multiple))
    return "0.5 + π" if whole == 1 else f"0.5 + {whole}π"


def _positive_rate_nl(calibration: dict) -> str:
    """What fraction of rows the stated cut-off actually labels class 1.

    The cut-off is the clean value nearest the target quantile, not the
    quantile, so it lands a couple of percent off 50%. Specs written before the
    snapping have no realized rate recorded and keep the target wording.
    """
    realized = calibration.get("realized_positive_rate")
    target = float(calibration["target_positive_rate"])
    if realized is None:
        return f"to target a positive-class rate of {target:.0%}"
    return (
        f"as the closest round value to the {target:.0%} quantile of `s(x)`; "
        f"on those {calibration['size']} rows it labels {float(realized):.1%} of them class 1"
    )


def format_synthetic_tabular_classification_rule(params: dict) -> str:
    rule_family = params["rule_family"]
    active_features = [int(feature) for feature in params["active_features"]]
    weights = [float(weight) for weight in params["rule_weights"]]
    active = ", ".join(f"`x_{feature}`" for feature in active_features)

    if rule_family == "spiral":
        turns = float(params.get("spiral_turns", 2.0))
        # Absent (v1.4 onwards) means the points sit exactly on their arm, so
        # neither the jitter clause nor the "noiseless arms" wording applies.
        noise_std = float(params.get("noise_std", 0.0) or 0.0)
        jitter_clause = (
            f"; independent Gaussian noise `ε ~ Normal(0, {noise_std:.6g}²)` is then added to each coordinate"
            if noise_std > 0.0
            else ""
        )
        arms_phrase = "the two noiseless arms" if noise_std > 0.0 else "the two arms"
        seed_label = "point/noise seed" if noise_std > 0.0 else "point seed"
        sampling = params["point_sampling"]
        left, right = active_features[:2]
        return "\n".join(
            [
                f"- Rule family: `spiral`; active coordinates: `x_{left}`, `x_{right}` (input is 2-dimensional).",
                f"- Point distribution: classic interleaved two-spirals. Each point is drawn along one of two Archimedean arms: with `t` uniform on `[0.5, {_spiral_t_upper_nl(turns)}]` ({_turns_nl(turns)}), radius `r = t`, coordinates `(r·cos(t + phase), r·sin(t + phase))` with `phase ∈ {{0, π}}`{jitter_clause}.",
                "- Label rule: `y = 0` for points drawn from the `phase = 0` arm and `y = 1` for points from the `phase = π` arm; both arms are equally likely, so classes are balanced.",
                "- Nominal soft score (intuition only): `s(x) = sin(atan2(x_1, x_0) − ‖x‖₂)`; its zero level sets trace the two arms, but labels come from the generative arm, not from thresholding `s(x)`.",
                f"- Bayes decision boundary: assign each point to the nearer of {arms_phrase} (up to label-flip symmetry); with {_turns_nl(turns)} the arms interleave, so the boundary is highly non-linear.",
                f"- Reproducibility: {seed_label} `{sampling['seed']}`, spiral turns `{turns:.6g}`.",
            ]
        )

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
    elif rule_family == "xor":
        left, right = active_features[:2]
        score_lines = [f"  - `s(x) = -x_{left}·x_{right}`"]
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

    # Absent (v1.4 onwards) means labels are an exact function of the features.
    noise_std = float(params.get("noise_std", 0.0) or 0.0)
    noisy = noise_std > 0.0
    threshold = float(params["decision_threshold"])
    calibration = params["calibration"]
    return "\n".join(
        [
            f"- Rule family: `{rule_family}`; active coordinates: {active}",
            "- Feature distribution: every coordinate is sampled independently from `Normal(0, 1)`.",
            "- Latent score:",
            *score_lines,
            *(
                [f"- Label noise: `ε ~ Normal(0, {noise_std:.6g}²)`."]
                if noisy
                else []
            ),
            f"- Label rule: `y = 1` exactly when "
            f"`{'s(x) + ε' if noisy else 's(x)'} > {threshold:.6g}`; otherwise `y = 0`.",
            *(
                # The cut-off snaps to a round value, and for XOR the calibrated
                # quantile is within a rounding step of 0 -- so the quadrant rule
                # is the label rule, exactly, and the prompt can say so instead
                # of stating it and then walking it back. A non-zero threshold
                # (or label noise) still needs the caveat.
                [
                    "- XOR interpretation: `x` is class 1 exactly when its two active coordinates have opposite signs, and class 0 when they share a sign."
                ]
                if rule_family == "xor" and threshold == 0.0 and not noisy
                else [
                    "- XOR interpretation (nominal only): with threshold `0`"
                    + (" and `ε = 0`" if noisy else "")
                    + ", opposite-sign active coordinates are class 1 and same-sign active coordinates are class 0.",
                    "- With the calibrated threshold"
                    + (" and label noise" if noisy else "")
                    + " above, individual labels (especially near either axis) need not follow that nominal quadrant interpretation.",
                ]
                if rule_family == "xor"
                else []
            ),
            (
                f"- Bayes decision boundary: without observing ε, predict class 1 when `s(x) > {threshold:.6g}`."
                if noisy
                else f"- Bayes decision boundary: `s(x) = {threshold:.6g}`; labels are an exact function of `x`, so this boundary is attainable."
            ),
            f"- Threshold calibration: `{threshold:.6g}` was chosen from {calibration['size']} independent calibration rows {_positive_rate_nl(calibration)}.",
            f"- Reproducibility: {'point/noise seed' if noisy else 'point seed'} `{params['point_sampling']['seed']}`, calibration seed `{calibration['seed']}`.",
        ]
    )


def format_synthetic_tabular_classification_protocol(params: dict, *, device: str = "cpu") -> str:
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


def format_dataset_protocol(params: dict, *, family: str | None = None, device: str = "cpu") -> str:
    if family in TABULAR_CLASSIFICATION_FAMILIES:
        return format_synthetic_tabular_classification_protocol(params, device=device)
    if family == "bigram_lm" or "vocab_size" in params:
        return format_bigram_protocol(params, device=device)
    if "input_dim" in params:
        return format_multivariate_protocol(params, device=device)
    return format_regression_protocol(params, device=device)


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
