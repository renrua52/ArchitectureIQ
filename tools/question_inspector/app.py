"""Streamlit UI for inspecting ArchitectureIQ question artifacts."""

from __future__ import annotations

import hashlib
import random
import secrets
import shutil
import sys
import tempfile
from importlib import reload
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artifact_loader  # noqa: E402
import prompt_format  # noqa: E402

reload(artifact_loader)
reload(prompt_format)
from artifact_loader import (  # noqa: E402
    QuestionBundle,
    candidate_file_paths,
    dataset_file_paths,
    format_metrics,
    list_question_dirs,
    load_dataset_tensors,
    load_question_bundle,
    question_label,
    read_json_file,
    read_text_file,
)
import candidate_curves  # noqa: E402
import custom_settings  # noqa: E402

reload(candidate_curves)
reload(custom_settings)
from candidate_curves import load_candidate_curves  # noqa: E402
from custom_settings import (  # noqa: E402
    build_custom_setting_spec,
    build_loss_spec,
    build_model_spec,
    build_optimizer_spec,
    clear_custom_settings_storage,
    clear_legacy_question_custom_settings,
    compatible_model_types,
    enforce_custom_setting_retention,
    form_values_from_candidate_spec,
    list_custom_setting_runs,
    run_custom_setting,
)
from expression_latex import expression_to_latex  # noqa: E402
from architecture_iq.profile import load_profile  # noqa: E402

st.set_page_config(
    page_title="ArchitectureIQ Question Inspector",
    page_icon="🔍",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .question-id {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        color: #0f172a;
        margin: 0 0 0.2rem 0;
    }
    .question-meta {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.1rem;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.candidate-card-marker) {
        margin-bottom: 0.5rem;
    }
    .candidate-letter {
        font-size: 2.75rem;
        font-weight: 700;
        line-height: 1;
        margin: 0;
        color: #0f172a;
    }
    .candidate-id {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.15rem;
    }
    .info-btn-slot + div[data-testid="stButton"] button {
        width: 2.1rem;
        height: 2.1rem;
        min-height: 2.1rem;
        padding: 0;
        border-radius: 50%;
        border: 1.5px solid #94a3b8 !important;
        background: #ffffff !important;
        color: #475569 !important;
        font-size: 0.95rem;
        font-weight: 700;
        font-family: Georgia, "Times New Roman", serif;
        font-style: italic;
        line-height: 1;
    }
    .info-btn-slot + div[data-testid="stButton"] button:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
        background: #eff6ff !important;
    }
    .info-btn-slot + div[data-testid="stButton"] button p {
        font-size: 0.95rem;
        line-height: 1;
    }
    .spec-block {
        font-size: 0.92rem;
        line-height: 1.55;
        margin: 0.65rem 0 0.2rem 0;
    }
    .spec-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.15rem;
    }
    .spec-line {
        margin: 0.05rem 0;
        color: #1e293b;
    }
    .metric-pill {
        display: inline-block;
        margin-top: 0.55rem;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
"""


def _init_state() -> None:
    defaults = {
        "bundle": None,
        "committed_letter": None,
        "focus_letter": None,
        "info_letter": None,
        "inspect_file": "candidate_spec.json",
        "dataset_file": "dataset_spec.json",
        "question_path": None,
        "data_root": "data",
        "question_pool": [],
        "quiz_results": {},
        "setting_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_quiz_state() -> None:
    st.session_state.committed_letter = None
    st.session_state.focus_letter = None
    st.session_state.info_letter = None
    st.session_state.inspect_file = "candidate_spec.json"
    st.session_state.setting_notice = None


def _score_stats() -> tuple[int, int]:
    results: dict[str, bool] = st.session_state.quiz_results
    total = len(results)
    correct = sum(1 for ok in results.values() if ok)
    return correct, total


def _reset_score() -> None:
    st.session_state.quiz_results = {}


def _record_answer(q: dict[str, Any], picked_letter: str) -> None:
    qid = q["question_id"]
    if qid in st.session_state.quiz_results:
        return
    st.session_state.quiz_results[qid] = picked_letter == q["correct_letter"]


def _commit_selection(q: dict[str, Any], letter: str) -> None:
    st.session_state.committed_letter = letter
    st.session_state.focus_letter = letter
    st.session_state.info_letter = None
    _record_answer(q, letter)


def _render_score_panel() -> None:
    correct, total = _score_stats()
    st.markdown(f"**Score:** {correct} / {total}")
    if st.button("Reset score", use_container_width=True):
        _reset_score()
        st.rerun()


def _switch_question(question_path: Path, data_root: str) -> None:
    previous_path = st.session_state.get("question_path")
    if previous_path:
        previous_root = Path(previous_path)
        if previous_root.is_dir():
            try:
                previous_q = read_json_file(previous_root / "question.json")
                _discard_custom_settings(previous_q["question_id"], previous_root)
            except (FileNotFoundError, KeyError, OSError):
                clear_legacy_question_custom_settings(previous_root)

    question_path = question_path.resolve()
    clear_legacy_question_custom_settings(question_path)

    st.session_state.question_path = str(question_path)
    st.session_state.bundle = _load_selected_question(question_path, data_root)
    _reset_quiz_state()


def _custom_settings_session_tag() -> str:
    return st.session_state.setdefault(
        "custom_settings_session_tag",
        secrets.token_hex(8),
    )


def _custom_settings_storage_path(question_id: str) -> Path:
    bucket: dict[str, str] = st.session_state.setdefault(
        "custom_settings_storage",
        {},
    )
    existing = bucket.get(question_id)
    if existing:
        return Path(existing)
    storage = (
        Path(tempfile.gettempdir())
        / "architectureiq_inspector"
        / _custom_settings_session_tag()
        / question_id
    )
    storage.mkdir(parents=True, exist_ok=True)
    bucket[question_id] = str(storage)
    return storage


def _discard_custom_settings(question_id: str, question_root: Path) -> None:
    bucket: dict[str, str] = st.session_state.get("custom_settings_storage", {})
    existing = bucket.pop(question_id, None)
    if existing:
        clear_custom_settings_storage(Path(existing))
    clear_legacy_question_custom_settings(question_root)


def _custom_settings_storage_for(q: dict[str, Any]) -> Path:
    return _custom_settings_storage_path(str(q["question_id"]))


def _pool_contains(pool: list[Path], path: Path) -> bool:
    target = path.resolve()
    return any(p.resolve() == target for p in pool)


def _resolve_data_root(data_root: str) -> Path:
    return Path(data_root).expanduser().resolve()


def _discover_questions(data_root: str) -> list[Path]:
    return list_question_dirs(_resolve_data_root(data_root))


def _default_question_path(pool: list[Path]) -> Path | None:
    if not pool:
        return None
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1]).resolve()
        if arg.is_file():
            arg = arg.parent
        for path in pool:
            if path.resolve() == arg:
                return path
        if arg.exists():
            return arg
    return pool[0]


def _load_selected_question(question_path: Path, data_root: str) -> QuestionBundle | None:
    try:
        bundle = load_question_bundle(question_path, data_root or None)
        _assign_question_cache_scope(bundle)
        return bundle
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None


def _assign_question_cache_scope(bundle: QuestionBundle) -> None:
    identity = str(bundle.question_root.resolve())
    bundle.question["_inspector_cache_scope"] = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]


def _render_question_picker(data_root: str) -> None:
    pool = _discover_questions(data_root)
    st.session_state.question_pool = [str(p) for p in pool]

    if not pool:
        st.warning("No questions found under the data root.")
        st.session_state.bundle = None
        return

    current = st.session_state.question_path
    if current is None or not _pool_contains(pool, Path(current)):
        default = _default_question_path(pool)
        if default is not None:
            st.session_state.question_path = str(default.resolve())

    current_path = Path(st.session_state.question_path)
    labels = [question_label(p) for p in pool]
    label_to_path = dict(zip(labels, pool, strict=True))
    try:
        current_index = pool.index(current_path.resolve())
    except ValueError:
        current_index = 0
        st.session_state.question_path = str(pool[0])

    nav_next, nav_random = st.columns(2)
    with nav_next:
        if st.button("Next", use_container_width=True):
            nxt = pool[(current_index + 1) % len(pool)]
            _switch_question(nxt, data_root)
            st.rerun()
    with nav_random:
        if st.button("Random", use_container_width=True):
            choices = [p for p in pool if p.resolve() != current_path.resolve()]
            pick = random.choice(choices or pool)
            _switch_question(pick, data_root)
            st.rerun()

    picked_label = st.selectbox(
        "Question",
        labels,
        index=current_index,
    )
    picked_path = label_to_path[picked_label]
    if picked_path.resolve() != current_path.resolve():
        _switch_question(picked_path, data_root)
        st.rerun()

    if st.session_state.bundle is None:
        clear_legacy_question_custom_settings(picked_path)
        st.session_state.bundle = _load_selected_question(picked_path, data_root)

    st.caption(f"{len(pool)} question(s) · `{picked_path.name}`")


def _selection_metric(bundle: QuestionBundle, q: dict[str, Any]) -> str:
    if "evaluation" in q:
        return str(q["evaluation"].get("selection_metric", "test_mse"))
    spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    return str(spec.get("selection_metric", "test_mse"))


def _metric_display_name(metric: str) -> str:
    if metric == "test_ce":
        return "test cross-entropy"
    if metric == "test_mse":
        return "test MSE"
    return metric


def _plot_univariate_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(
        train_x.squeeze(-1).numpy(),
        train_y.squeeze(-1).numpy(),
        s=10,
        alpha=0.55,
        label="train",
        c="#2563eb",
    )
    ax.scatter(
        test_x.squeeze(-1).numpy(),
        test_y.squeeze(-1).numpy(),
        s=10,
        alpha=0.55,
        label="test",
        c="#dc2626",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dataset points")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _plot_multivariate_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    input_dim: int,
) -> None:
    y_train = train_y.squeeze(-1).numpy()
    y_test = test_y.squeeze(-1).numpy()
    x_train = train_x.numpy()
    x_test = test_x.numpy()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    if input_dim >= 2:
        train_sc = ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            s=12,
            alpha=0.65,
            cmap="viridis",
            label="train",
        )
        ax.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test,
            s=12,
            alpha=0.65,
            cmap="viridis",
            marker="x",
            label="test",
        )
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_title("Dataset projection (color = target y)")
        fig.colorbar(train_sc, ax=ax, label="y")
    else:
        ax.scatter(x_train[:, 0], y_train, s=10, alpha=0.55, label="train", c="#2563eb")
        ax.scatter(x_test[:, 0], y_test, s=10, alpha=0.55, label="test", c="#dc2626")
        ax.set_xlabel("x0")
        ax.set_ylabel("y")
        ax.set_title("Dataset points")
        ax.legend(loc="best")

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _plot_bigram_lm(dataset_dir: Path, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    transition_path = dataset_dir / "transition.npz"
    if transition_path.is_file():
        probs = np.load(transition_path)["probs"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(probs, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("next token y")
        ax.set_ylabel("current token x")
        ax.set_title("Bigram transition P(y | x)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    sample_rows = min(3, int(train_x.shape[0]))
    context_length = int(train_x.shape[1])
    st.caption(
        f"Sample train windows ({sample_rows} of {train_x.shape[0]}, length {context_length}):"
    )
    for i in range(sample_rows):
        x_row = train_x[i].tolist()
        y_row = train_y[i].tolist()
        st.code(f"x: {x_row}\ny: {y_row}", language="text")


def _plot_dataset(bundle: QuestionBundle) -> None:
    spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    family = spec.get("family", "univariate_regression")
    params = spec.get("params", {})
    train_x, train_y, test_x, test_y = load_dataset_tensors(bundle.dataset_dir)

    if family == "multivariate_regression":
        _plot_multivariate_regression(
            train_x,
            train_y,
            test_x,
            test_y,
            input_dim=int(params.get("input_dim", train_x.shape[1])),
        )
    elif family == "bigram_lm":
        _plot_bigram_lm(bundle.dataset_dir, train_x, train_y)
    else:
        _plot_univariate_regression(train_x, train_y, test_x, test_y)


def _choice_color(index: int) -> str:
    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    return palette[index % len(palette)]


def _setting_color(index: int) -> str:
    palette = ("#0f766e", "#c026d3", "#ca8a04", "#4f46e5", "#be123c", "#0369a1")
    return palette[index % len(palette)]


def _curve_window_key(q: dict[str, Any]) -> str:
    return f"curve_x_window_{q['question_id']}"


def _curve_series_from_candidate(
    candidate_dir: Path,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> dict[str, Any] | None:
    curves_path = candidate_dir / "results" / "curves.npz"
    spec = read_json_file(candidate_dir / "candidate_spec.json")
    budget = spec.get("budget", {})
    total_samples_seen = budget.get("total_samples_seen")
    batch_size = budget.get("batch_size")
    if total_samples_seen is None or batch_size is None:
        return None

    loaded = load_candidate_curves(
        curves_path,
        total_samples_seen=int(total_samples_seen),
        batch_size=int(batch_size),
    )
    if "error" in loaded:
        return None

    curves = loaded["curves"]
    x = np.asarray(loaded["eval_samples"], dtype=np.int64)
    if curves.size == 0 or not np.isfinite(curves).any():
        return None

    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    valid = np.isfinite(mean)
    if not valid.any():
        return None
    return {
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "x": x[valid],
        "mean": mean[valid],
        "std": std[valid],
    }


def _collect_curve_series(bundle: QuestionBundle) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, choice in enumerate(bundle.choices):
        item = _curve_series_from_candidate(
            choice["candidate_dir"],
            label=f"{choice['letter']} · {choice['candidate_id']}",
            color=_choice_color(index),
        )
        if item is not None:
            series.append(item)
    return series


def _collect_custom_curve_series(bundle: QuestionBundle) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, setting in enumerate(
        list_custom_setting_runs(_custom_settings_storage_for(bundle.question))
    ):
        item = _curve_series_from_candidate(
            setting["candidate_dir"],
            label=f"Custom · {setting['label']}",
            color=_setting_color(index),
            linestyle="--",
        )
        if item is not None:
            item["setting"] = setting
            series.append(item)
    return series


def _render_curve_controls(
    series: list[dict[str, Any]],
    q: dict[str, Any],
) -> tuple[int, int, bool, bool]:
    all_x = np.concatenate([item["x"] for item in series])
    min_x = int(np.nanmin(all_x))
    max_x = int(np.nanmax(all_x))
    window_key = _curve_window_key(q)

    col_range, col_log_x, col_log_y = st.columns([5, 1, 1])
    with col_range:
        if min_x == max_x:
            x_min, x_max = min_x, max_x
            st.caption(f"Samples shown: {min_x}")
        else:
            unique_x = np.unique(all_x)
            step = (
                max(1, int(np.gcd.reduce(np.diff(unique_x))))
                if len(unique_x) > 1
                else 1
            )
            x_min, x_max = st.slider(
                "Samples shown",
                min_value=min_x,
                max_value=max_x,
                value=(min_x, max_x),
                step=step,
                key=window_key,
            )
    with col_log_x:
        use_log_x = st.checkbox("Log X", value=False, key=f"log_x_{q['question_id']}")
    with col_log_y:
        use_log_y = st.checkbox("Log Y", value=False, key=f"log_y_{q['question_id']}")

    return int(x_min), int(x_max), use_log_x, use_log_y


def _render_combined_curves(
    bundle: QuestionBundle,
    q: dict[str, Any],
    *,
    metric: str,
    include_candidates: bool,
) -> None:
    st.markdown("#### Learning curves")
    custom_series = _collect_custom_curve_series(bundle)
    series = (_collect_curve_series(bundle) if include_candidates else []) + custom_series
    if not series:
        st.info("No learning curve data available yet.")
        return

    x_min, x_max, use_log_x, use_log_y = _render_curve_controls(series, q)
    fig, ax = plt.subplots(figsize=(8, 4))
    any_plotted = False

    for item in series:
        x_valid = item["x"]
        in_window = (x_valid >= x_min) & (x_valid <= x_max)
        if use_log_x:
            in_window &= x_valid > 0
        if not in_window.any():
            continue

        x_valid = x_valid[in_window]
        mean_valid = item["mean"][in_window]
        std_valid = item["std"][in_window]
        lower = mean_valid - std_valid
        upper = mean_valid + std_valid
        line = mean_valid
        if use_log_y:
            eps = np.finfo(float).tiny
            lower = np.maximum(lower, eps)
            upper = np.maximum(upper, eps)
            line = np.maximum(line, eps)
        ax.fill_between(
            x_valid,
            lower,
            upper,
            color=item["color"],
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            x_valid,
            line,
            color=item["color"],
            linewidth=2.2,
            linestyle=item.get("linestyle", "-"),
            label=item["label"],
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        st.info("No learning curve points in the selected range.")
        return

    ax.set_xlabel("Samples seen")
    ax.set_ylabel(_metric_display_name(metric))
    ax.set_xlim(x_min, x_max)
    if use_log_x:
        ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")
    title = "Learning curves (mean ± std across seeds)"
    if custom_series and not include_candidates:
        title = "Custom setting curves (mean ± std across seeds)"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_file_panel(
    paths: dict[str, Path],
    selected: str,
    key_prefix: str,
) -> None:
    names = list(paths.keys())
    idx = names.index(selected) if selected in names else 0
    choice = st.radio("File", names, index=idx, horizontal=True, key=f"{key_prefix}_file_radio")
    st.session_state[f"{key_prefix}_file"] = choice
    path = paths[choice]

    if choice.endswith(".json"):
        st.json(read_json_file(path))
    else:
        st.code(read_text_file(path), language="python" if choice.endswith(".py") else None)


def _format_model_lines(model: dict[str, Any]) -> list[str]:
    return prompt_format.format_model_spec_lines(model)


def _format_optimizer_lines(opt: dict[str, Any]) -> list[str]:
    lines = [
        f"Type: {opt.get('type', '?')}",
        f"Learning rate: {opt.get('lr')}",
        f"Weight decay: {opt.get('weight_decay', 0)}",
    ]
    if opt.get("type") == "SGD" and "momentum" in opt:
        lines.append(f"Momentum: {opt['momentum']}")
    if opt.get("type") in {"Adam", "AdamW"} and "betas" in opt:
        lines.append(f"Betas: {opt['betas']}")
    return lines


def _format_loss_lines(loss: dict[str, Any]) -> list[str]:
    lines = [f"Loss: {loss.get('loss_id', '?')}"]
    if "lambda" in loss:
        lines.append(f"Lambda: {loss['lambda']}")
    return lines


def _format_training_lines(budget: dict[str, Any]) -> list[str]:
    return [
        f"Training steps: {budget.get('training_steps')}",
        f"Batch size: {budget.get('batch_size')}",
        f"Total samples seen: {budget.get('total_samples_seen')}",
    ]


def _spec_block(label: str, lines: list[str]) -> str:
    body = "".join(f'<div class="spec-line">{line}</div>' for line in lines)
    return f'<div class="spec-block"><div class="spec-label">{label}</div>{body}</div>'


def _render_candidate_spec_html(spec: dict[str, Any]) -> str:
    blocks = [
        _spec_block("Training", _format_training_lines(spec.get("budget", {}))),
        _spec_block("Model", _format_model_lines(spec.get("model", {}))),
        _spec_block("Optimizer", _format_optimizer_lines(spec.get("optimizer", {}))),
        _spec_block("Loss", _format_loss_lines(spec.get("loss", {}))),
    ]
    return "".join(blocks)


def _question_budget(q: dict[str, Any]) -> int:
    budget = q["budget"]
    if isinstance(budget, dict):
        return int(budget["total_samples_seen"])
    return int(budget)


def _setting_key(q: dict[str, Any], name: str) -> str:
    scope = q.get("_inspector_cache_scope")
    if not scope:
        identity = "|".join(
            str(q.get(field, ""))
            for field in ("family", "dataset_id", "question_run_id", "question_id")
        )
        scope = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"custom_setting_{scope}_{name}"


def _ensure_setting_value(q: dict[str, Any], name: str, default: Any) -> str:
    key = _setting_key(q, name)
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def _default_batch_size(bundle: QuestionBundle) -> int:
    if bundle.choices:
        spec = read_json_file(bundle.choices[0]["candidate_dir"] / "candidate_spec.json")
        value = spec.get("budget", {}).get("batch_size")
        if value is not None:
            return int(value)
    return 32


def _render_mlp_setting_fields(profile: Any, q: dict[str, Any]) -> dict[str, Any]:
    st.markdown("**Architecture parameters**")
    depth_col, width_col, residual_col = st.columns(3)
    with depth_col:
        depth = int(
            st.number_input(
                "Depth",
                min_value=1,
                max_value=12,
                step=1,
                key=_ensure_setting_value(
                    q,
                    "mlp_depth",
                    int(profile.mlp["depth"][1]),
                ),
            )
        )
    with width_col:
        width = int(
            st.number_input(
                "Width",
                min_value=4,
                max_value=2048,
                step=4,
                key=_ensure_setting_value(
                    q,
                    "mlp_width",
                    int(profile.mlp["width"][1]),
                ),
            )
        )
    with residual_col:
        residual = st.checkbox(
            "Residual connections",
            key=_ensure_setting_value(q, "mlp_residual", False),
        )

    st.caption("Choose the activation and layer norm independently for each hidden block.")
    activations: list[str] = []
    layer_norm: list[bool] = []
    layer_columns = st.columns(min(depth, 4))
    for index in range(depth):
        with layer_columns[index % len(layer_columns)]:
            st.markdown(f"Layer {index + 1}")
            activations.append(
                st.selectbox(
                    "Activation",
                    list(profile.mlp["activations"]),
                    key=_ensure_setting_value(
                        q,
                        f"mlp_activation_{index}",
                        profile.mlp["activations"][0],
                    ),
                    label_visibility="collapsed",
                )
            )
            layer_norm.append(
                st.checkbox(
                    "Layer norm",
                    key=_ensure_setting_value(q, f"mlp_norm_{index}", False),
                )
            )
    return {
        "depth": depth,
        "width": width,
        "residual": residual,
        "activations": activations,
        "layer_norm": layer_norm,
    }


def _render_transformer_setting_fields(profile: Any, q: dict[str, Any]) -> dict[str, Any]:
    st.markdown("**Architecture parameters**")
    d_model_col, layers_col, heads_col, d_ff_col = st.columns(4)
    with d_model_col:
        d_model = int(
            st.number_input(
                "Model width",
                min_value=8,
                max_value=1024,
                step=8,
                key=_ensure_setting_value(
                    q,
                    "transformer_d_model",
                    int(profile.transformer_lm["d_model"][0]),
                ),
            )
        )
    with layers_col:
        num_layers = int(
            st.number_input(
                "Layers",
                min_value=1,
                max_value=12,
                step=1,
                key=_ensure_setting_value(
                    q,
                    "transformer_layers",
                    int(profile.transformer_lm["num_layers"][0]),
                ),
            )
        )
    with heads_col:
        num_heads = int(
            st.number_input(
                "Attention heads",
                min_value=1,
                max_value=32,
                step=1,
                key=_ensure_setting_value(
                    q,
                    "transformer_heads",
                    int(profile.transformer_lm["num_heads"][0]),
                ),
            )
        )
    with d_ff_col:
        d_ff = int(
            st.number_input(
                "Feed-forward width",
                min_value=8,
                max_value=4096,
                step=8,
                key=_ensure_setting_value(
                    q,
                    "transformer_d_ff",
                    int(profile.transformer_lm["d_ff"][0]),
                ),
            )
        )
    return {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
    }


def _render_optimizer_setting_fields(profile: Any, q: dict[str, Any]) -> dict[str, Any]:
    st.markdown("**Optimizer parameters**")
    opt_col, lr_col, wd_col = st.columns(3)
    optimizers = list(profile.pools["optimizers"])
    default_index = optimizers.index("Adam") if "Adam" in optimizers else 0
    with opt_col:
        optimizer_type = st.selectbox(
            "Optimizer",
            optimizers,
            key=_ensure_setting_value(
                q,
                "optimizer_type",
                optimizers[default_index],
            ),
        )
    with lr_col:
        lr = float(
            st.number_input(
                "Learning rate",
                min_value=1e-8,
                max_value=10.0,
                step=1e-4,
                format="%.6g",
                key=_ensure_setting_value(q, "learning_rate", 1e-3),
            )
        )
    with wd_col:
        weight_decay = float(
            st.number_input(
                "Weight decay",
                min_value=0.0,
                max_value=10.0,
                step=1e-5,
                format="%.6g",
                key=_ensure_setting_value(q, "weight_decay", 0.0),
            )
        )

    momentum: float | None = None
    betas: tuple[float, float] | None = None
    if optimizer_type == "SGD":
        momentum = float(
            st.number_input(
                "Momentum",
                min_value=0.0,
                max_value=0.999999,
                step=0.05,
                format="%.6g",
                key=_ensure_setting_value(q, "momentum", 0.0),
            )
        )
    elif optimizer_type in {"Adam", "AdamW"}:
        beta1_col, beta2_col = st.columns(2)
        with beta1_col:
            beta1 = float(
                st.number_input(
                    "Beta 1",
                    min_value=0.0,
                    max_value=0.999999,
                    step=0.01,
                    format="%.6g",
                    key=_ensure_setting_value(q, "beta1", 0.9),
                )
            )
        with beta2_col:
            beta2 = float(
                st.number_input(
                    "Beta 2",
                    min_value=0.0,
                    max_value=0.999999,
                    step=0.001,
                    format="%.6g",
                    key=_ensure_setting_value(q, "beta2", 0.999),
                )
            )
        betas = (beta1, beta2)
    return {
        "optimizer_type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "betas": betas,
    }


def _apply_inherited_setting(
    bundle: QuestionBundle,
    q: dict[str, Any],
    source_letter: str,
) -> None:
    choice = next(choice for choice in bundle.choices if choice["letter"] == source_letter)
    spec = read_json_file(choice["candidate_dir"] / "candidate_spec.json")
    values = form_values_from_candidate_spec(
        spec,
        source_letter=source_letter,
        evaluation=q.get("evaluation"),
    )

    for name, value in values.items():
        st.session_state[_setting_key(q, name)] = value
    st.session_state[_setting_key(q, "inherited_letter")] = source_letter
    st.session_state[_setting_key(q, "inherited_candidate_id")] = choice["candidate_id"]


def _inherit_source_changed(bundle: QuestionBundle, q: dict[str, Any]) -> None:
    source_label = st.session_state[_setting_key(q, "inherit_source")]
    if source_label.startswith("Choice "):
        source_letter = source_label.split()[1]
        _apply_inherited_setting(bundle, q, source_letter)
        st.session_state.setting_notice = (
            f"Loaded every editable parameter from Choice {source_letter}."
        )
        return
    st.session_state.pop(_setting_key(q, "inherited_letter"), None)
    st.session_state.pop(_setting_key(q, "inherited_candidate_id"), None)


def _render_custom_setting_builder(bundle: QuestionBundle, q: dict[str, Any]) -> None:
    profile = load_profile(str(q.get("profile", "v1")))
    dataset_spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    family = str(dataset_spec["family"])
    runs = enforce_custom_setting_retention(_custom_settings_storage_for(q))

    notice = st.session_state.setting_notice
    if notice:
        st.success(notice)
        st.session_state.setting_notice = None

    with st.expander("＋ Add custom setting", expanded=not runs):
        st.caption(
            "Train a setting on this question's dataset. Its curve is added without "
            "changing the original choices or score."
        )
        no_inheritance = "No inheritance (keep editor values)"
        source_labels = [no_inheritance] + [
            f"Choice {choice['letter']} · {choice['candidate_id']}"
            for choice in bundle.choices
        ]
        st.selectbox(
            "Initialize from",
            source_labels,
            key=_setting_key(q, "inherit_source"),
            on_change=_inherit_source_changed,
            args=(bundle, q),
            help="Selecting A/B/C immediately replaces every editable field below.",
        )
        inherited_letter = st.session_state.get(_setting_key(q, "inherited_letter"))
        inherited_candidate_id = st.session_state.get(
            _setting_key(q, "inherited_candidate_id")
        )
        if inherited_letter and inherited_candidate_id:
            st.caption(
                f"Loaded Choice {inherited_letter} · `{inherited_candidate_id}`. "
                "Any field you change below will create a derived spec."
            )

        name_col, budget_col, batch_col = st.columns([2, 1, 1])
        with name_col:
            label = st.text_input(
                "Name prefix",
                key=_ensure_setting_value(q, "label", "Setting"),
                help="A unique sequence suffix is added automatically.",
            )
        with budget_col:
            budget = int(
                st.number_input(
                    "Total samples",
                    min_value=1,
                    max_value=10_000_000,
                    step=1,
                    key=_ensure_setting_value(q, "budget", _question_budget(q)),
                )
            )
        with batch_col:
            batch_size = int(
                st.number_input(
                    "Batch size",
                    min_value=1,
                    max_value=65_536,
                    step=1,
                    key=_ensure_setting_value(
                        q,
                        "batch_size",
                        _default_batch_size(bundle),
                    ),
                )
            )

        model_types = compatible_model_types(profile, family)
        model_type = st.selectbox(
            "Architecture",
            model_types,
            key=_ensure_setting_value(q, "model_type", model_types[0]),
        )
        if model_type == "mlp":
            model_params = _render_mlp_setting_fields(profile, q)
        else:
            model_params = _render_transformer_setting_fields(profile, q)

        optimizer_params = _render_optimizer_setting_fields(profile, q)

        loss_col, lambda_col, seeds_col, base_seed_col = st.columns(4)
        losses = list(profile.pools["losses"][family])
        with loss_col:
            loss_id = st.selectbox(
                "Loss",
                losses,
                key=_ensure_setting_value(q, "loss", losses[0]),
            )
        lambda_value: float | None = None
        with lambda_col:
            if loss_id.endswith("_l1") or loss_id.endswith("_l2"):
                lambda_value = float(
                    st.number_input(
                        "Loss lambda",
                        min_value=0.0,
                        max_value=10.0,
                        step=1e-4,
                        format="%.6g",
                        key=_ensure_setting_value(q, "loss_lambda", 1e-3),
                    )
                )
            else:
                st.text_input("Loss lambda", value="—", disabled=True)
        with seeds_col:
            n_seeds = int(
                st.number_input(
                    "Runs",
                    min_value=1,
                    max_value=20,
                    step=1,
                    key=_ensure_setting_value(q, "n_seeds", 3),
                    help="Independent random seeds used for the mean and standard deviation.",
                )
            )
        with base_seed_col:
            base_seed = int(
                st.number_input(
                    "Base seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    step=1,
                    key=_ensure_setting_value(q, "base_seed", 0),
                )
            )

        st.caption(
            f"This run performs {budget // batch_size if batch_size else 0} optimizer "
            f"steps × {n_seeds} seed(s)."
        )
        if st.button(
            "Confirm and generate curve",
            type="primary",
            key=_setting_key(q, "confirm"),
        ):
            try:
                model = build_model_spec(model_type, model_params, dataset_spec.get("params", {}))
                optimizer = build_optimizer_spec(
                    optimizer_params["optimizer_type"],
                    lr=optimizer_params["lr"],
                    weight_decay=optimizer_params["weight_decay"],
                    momentum=optimizer_params["momentum"],
                    betas=optimizer_params["betas"],
                )
                loss = build_loss_spec(loss_id, lambda_value=lambda_value)
                spec = build_custom_setting_spec(
                    profile,
                    dataset_spec,
                    budget=budget,
                    batch_size=batch_size,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                )
                inherited_candidate_id = st.session_state.get(
                    _setting_key(q, "inherited_candidate_id")
                )
                inherited_letter = st.session_state.get(
                    _setting_key(q, "inherited_letter")
                )
                inherited_from = None
                if inherited_candidate_id and inherited_letter:
                    inherited_from = {
                        "letter": inherited_letter,
                        "candidate_id": inherited_candidate_id,
                        "exact_spec_match": spec["candidate_id"]
                        == inherited_candidate_id,
                    }
                with st.spinner(f"Training {label or 'custom setting'}…"):
                    result = run_custom_setting(
                        _custom_settings_storage_for(q),
                        bundle.dataset_dir,
                        profile,
                        spec,
                        label=label,
                        n_seeds=n_seeds,
                        base_seed=base_seed,
                        inherited_from=inherited_from,
                    )
            except Exception as exc:
                st.error(f"Could not generate the curve: {exc}")
            else:
                inheritance_note = ""
                if result.get("inherited_from"):
                    inheritance_note = (
                        " Exact spec match."
                        if result["inherited_from"]["exact_spec_match"]
                        else " Inherited values were modified."
                    )
                st.session_state.setting_notice = (
                    f"Generated curve for {result['label']} ({result['candidate_id']})."
                    f"{inheritance_note}"
                )
                st.rerun()

        if runs:
            st.divider()
            st.markdown("**Retained settings (maximum 2)**")
            for index, run in enumerate(runs):
                role = "latest" if index == 0 else "best historical loss"
                st.caption(
                    f"{run['label']} · {role} · {run['candidate_id']} · "
                    f"loss {run['final_metric']:.6g} · "
                    f"{run['n_seeds']} seed(s), base {run['base_seed']}"
                )


def _inspect_paths(candidate_dir: Path, *, show_summary: bool) -> dict[str, Path]:
    return candidate_file_paths(candidate_dir, include_summary=show_summary)


def _render_metadata(q: dict[str, Any]) -> None:
    st.markdown(f'<div class="question-id">{q["question_id"]}</div>', unsafe_allow_html=True)
    st.markdown(
        (
            f'<div class="question-meta">'
            f"Type: {q.get('type', '—')} · "
            f"Budget: {_question_budget(q)} samples · "
            f"Metric: {q.get('significance', {}).get('metric', 'test_mse')} · "
            f"Choices: {q.get('num_choices', len(q['choices']))}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _card_border_style(
    letter: str,
    *,
    committed: bool,
    committed_letter: str | None,
    correct_letter: str,
    focused: bool,
) -> str:
    if not committed:
        return "2px solid #2563eb" if focused else "2px solid #e2e8f0"
    if letter == correct_letter:
        return "2px solid #16a34a"
    if letter == committed_letter and letter != correct_letter:
        return "2px solid #dc2626"
    return "2px solid #e2e8f0"


def _render_candidate_card(
    choice: dict[str, Any],
    q: dict[str, Any],
    *,
    committed: bool,
    committed_letter: str | None,
    correct_letter: str,
    focus_letter: str | None,
) -> None:
    letter = choice["letter"]
    spec = read_json_file(choice["candidate_dir"] / "candidate_spec.json")
    summary_path = choice["candidate_dir"] / "results" / "summary.json"
    summary = read_json_file(summary_path) if summary_path.is_file() else {}

    border = _card_border_style(
        letter,
        committed=committed,
        committed_letter=committed_letter,
        correct_letter=correct_letter,
        focused=focus_letter == letter,
    )
    bg = "#f8fafc" if focus_letter == letter else "#ffffff"

    st.markdown('<div class="candidate-card-marker"></div>', unsafe_allow_html=True)

    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.markdown(
            f'<p class="candidate-letter">{letter}</p>'
            f'<p class="candidate-id">{choice["candidate_id"]}</p>',
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown('<div class="info-btn-slot"></div>', unsafe_allow_html=True)
        if st.button("i", key=f"info_{letter}", help="View candidate files"):
            st.session_state.info_letter = letter
            st.session_state.focus_letter = letter
            st.rerun()

    metric_pill = ""
    if committed and summary and "error" not in summary:
        metric_pill = f'<span class="metric-pill">{format_metrics(summary)}</span>'

    st.markdown(
        f'<div style="border: {border}; border-radius: 12px; padding: 0.85rem 1rem; '
        f'background: {bg}; min-height: 18rem;">'
        f"{_render_candidate_spec_html(spec)}"
        f"{metric_pill}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if committed and letter == committed_letter:
        button_label = "Your pick"
        button_type = "primary"
        disabled = True
    elif committed:
        button_label = "View"
        button_type = "secondary"
        disabled = False
    else:
        button_label = "Select"
        button_type = "primary" if focus_letter == letter else "secondary"
        disabled = False

    if st.button(
        button_label,
        key=f"select_{letter}",
        use_container_width=True,
        type=button_type,
        disabled=disabled,
    ):
        if not committed:
            _commit_selection(q, letter)
        else:
            st.session_state.focus_letter = letter
            st.session_state.info_letter = None
        st.rerun()


def _render_answer_banner(q: dict[str, Any], committed_letter: str, *, metric: str) -> None:
    correct = q["correct_letter"]
    metric_label = _metric_display_name(metric)
    if committed_letter == correct:
        st.success(f"Correct — **{committed_letter}** achieves the best {metric_label}.")
    else:
        st.error(
            f"Incorrect — you picked **{committed_letter}**, "
            f"correct answer is **{correct}**."
        )


def _render_ranked_metrics(bundle: QuestionBundle, q: dict[str, Any]) -> None:
    st.markdown("#### Ranked results")
    rows = []
    for choice in bundle.choices:
        summary_path = choice["candidate_dir"] / "results" / "summary.json"
        summary = read_json_file(summary_path) if summary_path.is_file() else {}
        metric = summary.get("selection_metric", "test_mse")
        mean_key = f"mean_{metric}"
        rows.append(
            {
                "letter": choice["letter"],
                "candidate_id": choice["candidate_id"],
                "mean": summary.get(mean_key),
                "std": summary.get(f"std_{metric}"),
                "correct": choice["letter"] == q["correct_letter"],
            }
        )
    rows.sort(key=lambda r: (r["mean"] is None, r["mean"] or float("inf")))
    for row in rows:
        tag = " ✓ best" if row["correct"] else ""
        if row["mean"] is not None:
            st.write(
                f"**{row['letter']}** `{row['candidate_id']}` — "
                f"{row['mean']:.6f} ± {row['std']:.6f}{tag}"
            )
        else:
            st.write(f"**{row['letter']}** `{row['candidate_id']}` — no metrics")


def _render_dataset_info(spec: dict[str, Any], dataset_id: str) -> None:
    family = spec.get("family", "univariate_regression")
    params = spec.get("params", {})
    st.markdown(f"**ID:** `{dataset_id}`")
    st.markdown(f"**Family:** `{family}`")

    if family == "bigram_lm":
        st.markdown(f"**Vocab size:** {params.get('vocab_size', '—')}")
        st.markdown(f"**Context length:** {params.get('context_length', '—')}")
        st.markdown(
            "Fixed bigram law **P(y|x)** shared by train and test; "
            "only sampled windows differ between splits."
        )
        return

    expression = params.get("expression", "—")
    domain = params.get("domain", [0, 1])
    st.markdown("**Expression:**")
    st.latex(expression_to_latex(expression))
    if family == "multivariate_regression":
        st.markdown(f"**Input dimension:** {params.get('input_dim', '—')}")
        st.markdown(f"**Domain:** `[{domain[0]}, {domain[1]}]` per coordinate")
    else:
        st.markdown(f"**Domain:** `[{domain[0]}, {domain[1]}]`")


def _render_dataset_panel(bundle: QuestionBundle) -> None:
    st.markdown("#### Dataset")
    spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    family = spec.get("family", "univariate_regression")

    if family == "multivariate_regression":
        _render_dataset_info(spec, bundle.dataset_dir.name)
    else:
        info_col, plot_col = st.columns([1, 1.4])
        with info_col:
            _render_dataset_info(spec, bundle.dataset_dir.name)
        with plot_col:
            _plot_dataset(bundle)

    with st.expander("Dataset files", expanded=False):
        ds_paths = dataset_file_paths(bundle.dataset_dir)
        _render_file_panel(ds_paths, st.session_state.dataset_file, "dataset")


def _render_prompt_page(bundle: QuestionBundle) -> None:
    st.markdown("#### Prompt")
    st.code(bundle.prompt_text, language="markdown")


def _render_question_page(
    bundle: QuestionBundle,
    q: dict[str, Any],
    *,
    committed: bool,
    focus_letter: str | None,
) -> None:
    metric = _selection_metric(bundle, q)
    _render_metadata(q)
    _render_dataset_panel(bundle)
    st.markdown("#### Choices")

    cols = st.columns(len(bundle.choices))
    for col, choice in zip(cols, bundle.choices, strict=True):
        with col:
            _render_candidate_card(
                choice,
                q,
                committed=committed,
                committed_letter=st.session_state.committed_letter,
                correct_letter=q["correct_letter"],
                focus_letter=focus_letter,
            )

    _render_custom_setting_builder(bundle, q)
    custom_runs = list_custom_setting_runs(_custom_settings_storage_for(q))

    if committed:
        _render_answer_banner(q, st.session_state.committed_letter, metric=metric)
    if committed or custom_runs:
        _render_combined_curves(
            bundle,
            q,
            metric=metric,
            include_candidates=committed,
        )
    if committed:
        _render_ranked_metrics(bundle, q)

    inspect_letter = st.session_state.info_letter or st.session_state.focus_letter
    if inspect_letter:
        selected_choice = next(c for c in bundle.choices if c["letter"] == inspect_letter)
        st.divider()
        st.markdown(f"#### Files · Choice **{inspect_letter}** · `{selected_choice['candidate_id']}`")
        paths = _inspect_paths(selected_choice["candidate_dir"], show_summary=committed)
        _render_file_panel(
            paths,
            st.session_state.inspect_file,
            f"candidate_{inspect_letter}",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_demo_data(data_root: str) -> None:
    """Copy bundled demo questions into data/ when deploying without a local snapshot."""
    root = _repo_root()
    resolved = _resolve_data_root(data_root)
    if _discover_questions(str(resolved)):
        return
    bundled = root / "examples" / "quiz_demo" / "bundle"
    if not bundled.is_dir():
        return
    shutil.copytree(bundled, resolved, dirs_exist_ok=True)


def main() -> None:
    _init_state()
    _ensure_demo_data(st.session_state.data_root)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Questions")
        _render_score_panel()
        st.divider()
        data_root = st.text_input("Data root", key="data_root")
        _render_question_picker(data_root)

    bundle: QuestionBundle | None = st.session_state.bundle

    if bundle is None:
        st.title("ArchitectureIQ Question Inspector")
        st.info("Select a question in the sidebar to begin.")
        return

    q = bundle.question
    _assign_question_cache_scope(bundle)
    committed = st.session_state.committed_letter is not None
    focus_letter = st.session_state.focus_letter

    tab_question, tab_prompt = st.tabs(["Question", "Prompt"])
    with tab_question:
        _render_question_page(
            bundle,
            q,
            committed=committed,
            focus_letter=focus_letter,
        )
    with tab_prompt:
        _render_prompt_page(bundle)


if __name__ == "__main__":
    main()
