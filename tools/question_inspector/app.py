"""Streamlit UI for inspecting ArchitectureIQ question artifacts."""

from __future__ import annotations

import random
import sys
from importlib import reload
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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

reload(candidate_curves)
from candidate_curves import load_candidate_curves  # noqa: E402
from expression_latex import expression_to_latex  # noqa: E402

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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_quiz_state() -> None:
    st.session_state.committed_letter = None
    st.session_state.focus_letter = None
    st.session_state.info_letter = None
    st.session_state.inspect_file = "candidate_spec.json"


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
    st.session_state.question_path = str(question_path.resolve())
    st.session_state.bundle = _load_selected_question(question_path, data_root)
    _reset_quiz_state()


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
        return load_question_bundle(question_path, data_root or None)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None


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


def _render_combined_curves(bundle: QuestionBundle, q: dict[str, Any], *, metric: str) -> None:
    st.markdown("#### Learning curves")
    fig, ax = plt.subplots(figsize=(8, 4))
    any_plotted = False

    for index, choice in enumerate(bundle.choices):
        letter = choice["letter"]
        color = _choice_color(index)
        curves_path = choice["candidate_dir"] / "results" / "curves.npz"
        spec = read_json_file(choice["candidate_dir"] / "candidate_spec.json")
        budget = spec.get("budget", {})
        total_samples_seen = budget.get("total_samples_seen")
        batch_size = budget.get("batch_size")
        if total_samples_seen is None or batch_size is None:
            continue

        loaded = load_candidate_curves(
            curves_path,
            total_samples_seen=int(total_samples_seen),
            batch_size=int(batch_size),
        )
        if "error" in loaded:
            continue

        curves = loaded["curves"]
        x = np.asarray(loaded["eval_samples"], dtype=np.int64)
        if curves.size == 0 or not np.isfinite(curves).any():
            continue

        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0)
        valid = np.isfinite(mean)
        if not valid.any():
            continue

        x_valid = x[valid]
        mean_valid = mean[valid]
        std_valid = std[valid]
        ax.fill_between(
            x_valid,
            mean_valid - std_valid,
            mean_valid + std_valid,
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            x_valid,
            mean_valid,
            color=color,
            linewidth=2.2,
            label=f"{letter} · {choice['candidate_id']}",
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        st.info("No learning curve data available for these candidates.")
        return

    ax.set_xlabel("Samples seen")
    ax.set_ylabel(_metric_display_name(metric))
    ax.set_title("Learning curves (mean ± std across seeds)")
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

    st.markdown(
        f'<div style="border: {border}; border-radius: 12px; padding: 0.85rem 1rem; '
        f'background: {bg}; min-height: 18rem;">'
        f"{_render_candidate_spec_html(spec)}"
        f"{f'<span class=\"metric-pill\">{format_metrics(summary)}</span>' if committed and summary and 'error' not in summary else ''}"
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

    if committed:
        _render_answer_banner(q, st.session_state.committed_letter, metric=metric)
        _render_combined_curves(bundle, q, metric=metric)
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


def main() -> None:
    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Questions")
        _render_score_panel()
        st.divider()
        data_root = st.text_input("Data root", value=st.session_state.data_root)
        st.session_state.data_root = data_root
        _render_question_picker(data_root)

    bundle: QuestionBundle | None = st.session_state.bundle

    if bundle is None:
        st.title("ArchitectureIQ Question Inspector")
        st.info("Select a question in the sidebar to begin.")
        return

    q = bundle.question
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
