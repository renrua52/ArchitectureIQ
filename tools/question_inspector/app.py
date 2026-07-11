"""Streamlit UI for inspecting ArchitectureIQ question artifacts."""

from __future__ import annotations

import html
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

reload(artifact_loader)
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
    page_title="ArchitectureIQ",
    page_icon="AIQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    :root {
        --aiq-ink: #111217;
        --aiq-muted: #81838b;
        --aiq-line: #24262c;
        --aiq-line-soft: #d7d4cc;
        --aiq-paper: #f8f7f2;
        --aiq-panel: #fffefa;
        --aiq-accent: #734cff;
        --aiq-accent-soft: #eee8ff;
        --aiq-dark: #111216;
        --aiq-green: #20a87e;
        --aiq-orange: #f26e4f;
        --aiq-shadow: 0 20px 70px rgba(31, 29, 24, 0.10);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 8%, rgba(115, 76, 255, 0.18), transparent 22rem),
            radial-gradient(circle at 90% 92%, rgba(242, 110, 79, 0.13), transparent 30rem),
            linear-gradient(180deg, #f8f7f2 0%, #eeeae2 100%);
        color: var(--aiq-ink);
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1480px;
        padding: 1.6rem 2rem 2rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .aiq-topbar {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1rem;
        min-height: 4.7rem;
        margin-bottom: 1.8rem;
        padding: 0.7rem 1rem;
        border: 1.8px solid var(--aiq-line);
        border-radius: 28px;
        background: rgba(255, 254, 250, 0.92);
        box-shadow: var(--aiq-shadow);
    }

    .aiq-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        min-width: 0;
    }

    .aiq-logo {
        display: grid;
        width: 2.9rem;
        height: 2.9rem;
        place-items: center;
        border-radius: 0.8rem;
        background: #101116;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
    }

    .aiq-logo-mark {
        width: 1.55rem;
        height: 0.8rem;
        border-top: 3px solid #ffffff;
        border-bottom: 3px solid #734cff;
        transform: skewY(-13deg);
    }

    .aiq-brand-title {
        margin: 0;
        color: var(--aiq-ink);
        font-size: 1.25rem;
        font-weight: 850;
        line-height: 1;
    }

    .aiq-brand-subtitle {
        margin-top: 0.25rem;
        color: #777982;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .aiq-progress {
        min-width: 18rem;
        text-align: center;
    }

    .aiq-progress-label {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--aiq-ink);
        font-size: 0.95rem;
        font-weight: 850;
    }

    .aiq-progress-track {
        display: inline-block;
        width: 8rem;
        height: 0.32rem;
        overflow: hidden;
        border-radius: 999px;
        background: #dad7d0;
        vertical-align: 0.12rem;
    }

    .aiq-progress-fill {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: var(--aiq-accent);
    }

    .aiq-actions {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.65rem;
    }

    .aiq-score-pill,
    .aiq-icon-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 2.55rem;
        border: 1.7px solid var(--aiq-line);
        border-radius: 999px;
        background: #fffefa;
        color: var(--aiq-ink);
        font-size: 0.82rem;
        font-weight: 850;
    }

    .aiq-score-pill {
        min-width: 7.4rem;
        padding: 0 1rem;
    }

    .aiq-icon-pill {
        width: 2.7rem;
    }

    .aiq-welcome-wrap {
        min-height: calc(100vh - 9rem);
        display: grid;
        align-items: center;
        justify-items: center;
        padding: 4rem 1rem;
        text-align: center;
    }

    .aiq-welcome {
        width: min(920px, 100%);
    }

    .aiq-welcome h1 {
        margin: 0 0 1.1rem;
        color: var(--aiq-ink);
        font-size: clamp(3rem, 7vw, 5.7rem);
        font-weight: 850;
        line-height: 0.98;
    }

    .aiq-welcome p {
        margin: 0 auto 2rem;
        max-width: 760px;
        color: #5f616b;
        font-size: clamp(1.05rem, 2vw, 1.4rem);
        line-height: 1.35;
    }

    .aiq-subtle-science {
        position: fixed;
        inset: auto 0 0 0;
        height: 10rem;
        pointer-events: none;
        opacity: 0.36;
        background:
            linear-gradient(168deg, transparent 0 42%, rgba(115, 76, 255, 0.34) 42.5%, transparent 43%),
            linear-gradient(172deg, transparent 0 52%, rgba(115, 76, 255, 0.26) 52.5%, transparent 53%),
            linear-gradient(176deg, transparent 0 62%, rgba(115, 76, 255, 0.22) 62.5%, transparent 63%);
    }

    .aiq-eyebrow {
        margin: 0 0 0.8rem;
        color: var(--aiq-accent);
        font-size: 0.74rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .aiq-question-title {
        max-width: 760px;
        margin: 0;
        color: var(--aiq-ink);
        font-size: clamp(2.25rem, 4.1vw, 4rem);
        font-weight: 820;
        line-height: 1.05;
    }

    .aiq-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.52rem;
        margin-top: 0.9rem;
    }

    .aiq-chip {
        display: inline-flex;
        align-items: center;
        min-height: 1.95rem;
        padding: 0 0.8rem;
        border: 1.4px solid var(--aiq-line);
        border-radius: 999px;
        background: rgba(255, 254, 250, 0.9);
        color: #666871;
        font-size: 0.76rem;
        font-weight: 780;
    }

    .aiq-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(300px, 0.36fr);
        gap: 1.35rem;
        margin-top: 1.4rem;
    }

    .aiq-panel {
        border: 1.8px solid var(--aiq-line);
        border-radius: 24px;
        background: rgba(255, 254, 250, 0.90);
        box-shadow: 0 10px 40px rgba(31, 29, 24, 0.07);
    }

    .aiq-panel-inner {
        padding: 1.2rem;
    }

    .aiq-panel-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }

    .aiq-dot-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        min-width: 0;
    }

    .aiq-dot {
        width: 0.55rem;
        height: 0.55rem;
        flex: 0 0 auto;
        border-radius: 999px;
        background: var(--aiq-accent);
    }

    .aiq-panel-title {
        color: var(--aiq-ink);
        font-size: 0.94rem;
        font-weight: 880;
    }

    .aiq-panel-muted {
        margin-left: 0.35rem;
        color: #9a9ba2;
        font-size: 0.88rem;
        font-weight: 750;
    }

    .aiq-plot-frame {
        overflow: hidden;
        border: 1.5px solid var(--aiq-line);
        border-radius: 18px;
        background: #fffefa;
    }

    .aiq-side h2 {
        margin: 0 0 0.45rem;
        color: var(--aiq-ink);
        font-size: 1.35rem;
        line-height: 1.08;
    }

    .aiq-side-copy {
        margin: 0 0 1.35rem;
        color: #777982;
        font-size: 0.92rem;
        line-height: 1.35;
    }

    .aiq-rule {
        height: 1.5px;
        margin: 1rem 0 1.05rem;
        background: var(--aiq-line);
    }

    .aiq-spec-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        padding: 0.65rem 0;
        color: #8a8b92;
        font-size: 0.86rem;
        font-weight: 760;
    }

    .aiq-spec-row strong {
        color: #50525a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        text-align: right;
    }

    .aiq-notice {
        display: flex;
        gap: 0.8rem;
        margin-top: 1.25rem;
        padding: 0.95rem;
        border: 1.5px solid #a992ff;
        border-radius: 16px;
        background: #f2edff;
        color: #57515f;
        font-size: 0.8rem;
        line-height: 1.35;
    }

    .aiq-notice-badge {
        display: grid;
        width: 1.15rem;
        height: 1.15rem;
        flex: 0 0 auto;
        place-items: center;
        border: 1.4px solid var(--aiq-accent);
        border-radius: 999px;
        color: var(--aiq-accent);
        font-size: 0.72rem;
        font-weight: 900;
    }

    .aiq-candidate-grid {
        display: grid;
        grid-template-columns: repeat(var(--choice-count), minmax(0, 1fr)) minmax(310px, 0.8fr);
        gap: 0.8rem;
        margin-top: 1rem;
        padding: 0.75rem;
        border-radius: 24px;
        background: rgba(255, 254, 250, 0.9);
    }

    .aiq-candidate-card {
        position: relative;
        min-height: 12rem;
        padding: 1rem;
        border: 1.7px solid var(--card-border, var(--aiq-line));
        border-radius: 18px;
        background: var(--aiq-panel);
    }

    .aiq-candidate-card.is-focused {
        box-shadow: 0 0 0 2px rgba(115, 76, 255, 0.10);
    }

    .aiq-candidate-top {
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 0.7rem;
        align-items: start;
    }

    .aiq-letter {
        display: grid;
        width: 3rem;
        height: 3rem;
        place-items: center;
        border: 1.5px solid var(--card-accent);
        border-radius: 0.55rem;
        color: var(--aiq-ink);
        font-size: 2rem;
        font-weight: 780;
        line-height: 1;
    }

    .aiq-candidate-id {
        margin-top: 0.2rem;
        color: #85868e;
        font-size: 0.72rem;
        font-weight: 760;
    }

    .aiq-select-dot {
        width: 1.3rem;
        height: 1.3rem;
        border: 1.5px solid var(--aiq-line);
        border-radius: 999px;
        background: var(--dot-fill, transparent);
    }

    .aiq-mini-arch {
        width: 4rem;
        height: 3.6rem;
        margin-top: 0.85rem;
        background:
            radial-gradient(circle at 16% 50%, var(--card-accent) 0 4px, transparent 4.5px),
            radial-gradient(circle at 42% 20%, var(--card-accent) 0 4px, transparent 4.5px),
            radial-gradient(circle at 42% 50%, var(--card-accent) 0 4px, transparent 4.5px),
            radial-gradient(circle at 42% 80%, var(--card-accent) 0 4px, transparent 4.5px),
            radial-gradient(circle at 72% 35%, var(--card-accent) 0 4px, transparent 4.5px),
            radial-gradient(circle at 72% 65%, var(--card-accent) 0 4px, transparent 4.5px);
        opacity: 0.95;
    }

    .aiq-card-body {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.35rem 0.85rem;
        margin-top: 0.95rem;
        color: #999aa1;
        font-size: 0.78rem;
        font-weight: 760;
    }

    .aiq-card-body strong {
        color: #3e4048;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.78rem;
        text-align: right;
    }

    .aiq-commit {
        min-height: 12rem;
        padding: 1.1rem;
        border-radius: 18px;
        background:
            radial-gradient(circle at 90% 0%, rgba(115, 76, 255, 0.25), transparent 12rem),
            var(--aiq-dark);
        color: #fff;
    }

    .aiq-commit h3 {
        margin: 0 0 0.45rem;
        color: #a9aab3;
        font-size: 0.75rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .aiq-commit p {
        margin: 0 0 1.1rem;
        color: #c4c5cc;
        font-size: 0.84rem;
    }

    .aiq-answer-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }

    .aiq-status {
        margin-top: 1rem;
    }

    .aiq-file-panel {
        margin-top: 1.2rem;
    }

    div.stButton > button,
    div[data-testid="stBaseButton-primary"] button,
    div[data-testid="stBaseButton-secondary"] button {
        border-radius: 999px !important;
        font-weight: 850 !important;
        letter-spacing: 0 !important;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        border-color: var(--aiq-accent) !important;
        background: var(--aiq-accent) !important;
        color: #ffffff !important;
        box-shadow: 0 16px 34px rgba(115, 76, 255, 0.22);
    }

    .aiq-welcome .stButton button {
        min-width: min(100%, 23rem);
        min-height: 4rem;
        border-radius: 999px !important;
        font-size: 1.1rem !important;
    }

    .aiq-answer-buttons + div {
        margin-top: 0.75rem;
    }

    div[data-baseweb="select"] > div,
    div[role="radiogroup"] label {
        border-radius: 999px !important;
    }

    div[role="radiogroup"] {
        padding: 0.2rem;
        border-radius: 999px;
        background: #ebe8e0;
    }

    div[role="radiogroup"] label {
        padding: 0.25rem 0.75rem !important;
        color: #777982 !important;
        font-size: 0.82rem;
        font-weight: 800;
    }

    pre {
        border: 1px solid #dedbd2 !important;
        border-radius: 16px !important;
    }

    @media (max-width: 1000px) {
        .aiq-topbar,
        .aiq-grid,
        .aiq-candidate-grid {
            grid-template-columns: 1fr;
        }

        .aiq-actions,
        .aiq-progress {
            justify-content: flex-start;
            text-align: left;
        }
    }

    @media (max-width: 720px) {
        .block-container {
            padding: 1rem 0.75rem 1.5rem;
        }

        .aiq-topbar,
        .aiq-panel,
        .aiq-candidate-grid {
            border-radius: 18px;
        }

        .aiq-question-title {
            font-size: 2.2rem;
        }

        .aiq-candidate-grid {
            padding: 0;
            background: transparent;
        }
    }
</style>
"""

ACCENTS = ("#734cff", "#f26e4f", "#20a87e", "#2f7de1")
EVIDENCE_TABS = ("Evidence", "Protocol", "Prompt", "Source")


def _init_state() -> None:
    defaults = {
        "started": False,
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
        "evidence_tab": "Evidence",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_fmt_value(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "-"
    return str(value)


def _reset_quiz_state() -> None:
    st.session_state.committed_letter = None
    st.session_state.focus_letter = None
    st.session_state.info_letter = None
    st.session_state.inspect_file = "candidate_spec.json"
    st.session_state.evidence_tab = "Evidence"


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


def _switch_question(question_path: Path, data_root: str) -> None:
    st.session_state.question_path = str(question_path.resolve())
    st.session_state.bundle = _load_selected_question(question_path, data_root)
    _reset_quiz_state()


def _ensure_bundle(data_root: str) -> tuple[QuestionBundle | None, list[Path], int]:
    pool = _discover_questions(data_root)
    st.session_state.question_pool = [str(p) for p in pool]
    if not pool:
        st.session_state.bundle = None
        return None, pool, 0

    current = st.session_state.question_path
    if current is None or not _pool_contains(pool, Path(current)):
        default = _default_question_path(pool)
        if default is not None:
            st.session_state.question_path = str(default.resolve())

    current_path = Path(st.session_state.question_path).resolve()
    try:
        current_index = [p.resolve() for p in pool].index(current_path)
    except ValueError:
        current_index = 0
        current_path = pool[0].resolve()
        st.session_state.question_path = str(current_path)

    if st.session_state.bundle is None:
        st.session_state.bundle = _load_selected_question(current_path, data_root)
    return st.session_state.bundle, pool, current_index


def _question_budget(q: dict[str, Any]) -> int:
    budget = q["budget"]
    if isinstance(budget, dict):
        return int(budget["total_samples_seen"])
    return int(budget)


def _render_brand() -> str:
    return """
    <div class="aiq-brand">
        <div class="aiq-logo" aria-hidden="true"><div class="aiq-logo-mark"></div></div>
        <div>
            <p class="aiq-brand-title">Architecture IQ</p>
            <div class="aiq-brand-subtitle">Read the setup - predict the winner</div>
        </div>
    </div>
    """


def _render_welcome_page() -> None:
    st.markdown(
        f"""
        <div class="aiq-topbar">
            {_render_brand()}
            <div></div>
            <div class="aiq-actions">
                <span class="aiq-icon-pill">...</span>
                <span class="aiq-icon-pill">::</span>
            </div>
        </div>
        <div class="aiq-welcome-wrap">
            <div class="aiq-welcome">
                <h1>Test your Architecture IQ</h1>
                <p>Wanna know how wise you are for picking the right setup for training tasks :)?</p>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Test my architecture IQ", type="primary"):
        st.session_state.started = True
        st.rerun()
    st.markdown("</div></div><div class=\"aiq-subtle-science\"></div>", unsafe_allow_html=True)


def _render_topbar(
    pool: list[Path],
    current_index: int,
    data_root: str,
) -> None:
    correct, total = _score_stats()
    progress = 0 if not pool else int(((current_index + 1) / len(pool)) * 100)
    st.markdown(
        f"""
        <div class="aiq-topbar">
            {_render_brand()}
            <div class="aiq-progress">
                <span class="aiq-progress-label">
                    Q {current_index + 1 if pool else 0} / {len(pool)}
                    <span class="aiq-progress-track"><span class="aiq-progress-fill" style="width:{progress}%"></span></span>
                    <span>{progress}%</span>
                </span>
            </div>
            <div class="aiq-actions">
                <span class="aiq-score-pill">Score&nbsp; {correct} / {total}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pool:
        return

    labels = [question_label(p) for p in pool]
    label_to_path = dict(zip(labels, pool, strict=True))
    nav_col, pick_col, root_col, reset_col = st.columns([0.9, 2.2, 1.3, 0.9])
    with nav_col:
        left, right = st.columns(2)
        with left:
            if st.button("Next", use_container_width=True):
                _switch_question(pool[(current_index + 1) % len(pool)], data_root)
                st.rerun()
        with right:
            if st.button("Random", use_container_width=True):
                current = Path(st.session_state.question_path).resolve()
                choices = [p for p in pool if p.resolve() != current]
                _switch_question(random.choice(choices or pool), data_root)
                st.rerun()
    with pick_col:
        picked_label = st.selectbox(
            "Question",
            labels,
            index=current_index,
            label_visibility="collapsed",
        )
        picked_path = label_to_path[picked_label]
        if picked_path.resolve() != Path(st.session_state.question_path).resolve():
            _switch_question(picked_path, data_root)
            st.rerun()
    with root_col:
        new_root = st.text_input(
            "Data root",
            value=st.session_state.data_root,
            label_visibility="collapsed",
            placeholder="data",
        )
        if new_root != st.session_state.data_root:
            st.session_state.data_root = new_root
            st.session_state.bundle = None
            st.session_state.question_path = None
            st.rerun()
    with reset_col:
        if st.button("Reset score", use_container_width=True):
            _reset_score()
            st.rerun()


def _question_title(q: dict[str, Any]) -> str:
    metric = q.get("significance", {}).get("metric") or q.get("evaluation", {}).get("selection_metric")
    metric_label = str(metric or "test metric").replace("_", " ")
    return f"Which training setup will reach the lowest {metric_label}?"


def _render_question_hero(q: dict[str, Any], bundle: QuestionBundle) -> None:
    family = str(q.get("family", bundle.dataset_dir.parent.name)).replace("_", " ")
    chips = [
        family.title(),
        f"Dataset {q.get('dataset_id', bundle.dataset_dir.name)}",
        f"Budget {_question_budget(q):,} samples",
        f"{q.get('num_choices', len(bundle.choices))} choices",
    ]
    metric = q.get("significance", {}).get("metric")
    if metric:
        chips.append(str(metric))
    chip_html = "".join(f'<span class="aiq-chip">{_escape(chip)}</span>' for chip in chips)
    st.markdown(
        f"""
        <p class="aiq-eyebrow">Observe the evidence</p>
        <h1 class="aiq-question-title">{_escape(_question_title(q))}</h1>
        <div class="aiq-chip-row">{chip_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fffefa")
    ax.grid(True, alpha=0.22, color="#9c97aa")
    for spine in ax.spines.values():
        spine.set_color("#24262c")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#50525a", labelsize=9)
    ax.xaxis.label.set_color("#50525a")
    ax.yaxis.label.set_color("#50525a")
    ax.title.set_color("#111217")


def _plot_dataset(bundle: QuestionBundle) -> None:
    train_x, train_y, test_x, test_y = load_dataset_tensors(bundle.dataset_dir)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    fig.patch.set_facecolor("#fffefa")
    ax.scatter(
        train_x.squeeze().numpy(),
        train_y.squeeze().numpy(),
        s=18,
        alpha=0.62,
        label="train",
        c="#734cff",
        edgecolors="none",
    )
    ax.scatter(
        test_x.squeeze().numpy(),
        test_y.squeeze().numpy(),
        s=18,
        alpha=0.62,
        label="test",
        c="#20a87e",
        edgecolors="none",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dataset evidence")
    ax.legend(loc="best", frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _plot_candidate_curves(
    curves_path: Path,
    *,
    total_samples_seen: int,
    batch_size: int,
) -> None:
    loaded = load_candidate_curves(
        curves_path,
        total_samples_seen=total_samples_seen,
        batch_size=batch_size,
    )
    if "error" in loaded:
        st.warning(loaded["error"])
        return

    curves = loaded["curves"]
    x = np.asarray(loaded["eval_samples"], dtype=np.int64)
    if curves.size == 0 or not np.isfinite(curves).any():
        st.info("No curve data available.")
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    fig.patch.set_facecolor("#fffefa")
    for row in curves:
        valid = np.isfinite(row)
        if not valid.any():
            continue
        ax.plot(x[valid], row[valid], color="#b8b3c8", alpha=0.42, linewidth=1)

    mean_curve = np.nanmean(curves, axis=0)
    valid_mean = np.isfinite(mean_curve)
    if valid_mean.any():
        ax.plot(
            x[valid_mean],
            mean_curve[valid_mean],
            color="#734cff",
            linewidth=2.4,
            label="mean",
        )

    ax.set_xlabel("Samples seen")
    ax.set_ylabel("Test metric")
    ax.set_title("Learning curve")
    ax.legend(loc="best", frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _render_file_panel(
    paths: dict[str, Path],
    selected: str,
    key_prefix: str,
    *,
    total_samples_seen: int | None = None,
    batch_size: int | None = None,
) -> None:
    names = list(paths.keys())
    idx = names.index(selected) if selected in names else 0
    choice = st.radio("File", names, index=idx, horizontal=True, key=f"{key_prefix}_file_radio")
    st.session_state[f"{key_prefix}_file"] = choice
    path = paths[choice]

    if choice == "curves.npz":
        if total_samples_seen is None or batch_size is None:
            st.warning("Budget metadata unavailable for this candidate.")
            return
        _plot_candidate_curves(
            path,
            total_samples_seen=total_samples_seen,
            batch_size=batch_size,
        )
        return

    if choice.endswith(".json"):
        st.json(read_json_file(path))
    else:
        st.code(read_text_file(path), language="python" if choice.endswith(".py") else None)


def _flatten_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "training steps": spec.get("budget", {}).get("training_steps"),
        "batch size": spec.get("budget", {}).get("batch_size"),
        "total samples seen": spec.get("budget", {}).get("total_samples_seen"),
        "model type": spec.get("model", {}).get("type"),
        "layers": spec.get("model", {}).get("depth"),
        "width": spec.get("model", {}).get("width"),
        "residual": spec.get("model", {}).get("residual"),
        "layer norm": spec.get("model", {}).get("layer_norm"),
        "activations": spec.get("model", {}).get("activations"),
        "optimizer": spec.get("optimizer", {}).get("type"),
        "learning rate": spec.get("optimizer", {}).get("lr"),
        "weight decay": spec.get("optimizer", {}).get("weight_decay"),
        "betas": spec.get("optimizer", {}).get("betas"),
        "loss": spec.get("loss", {}).get("loss_id"),
        "lambda": spec.get("loss", {}).get("lambda"),
    }


def _candidate_specs(bundle: QuestionBundle) -> dict[str, dict[str, Any]]:
    specs = {}
    for choice in bundle.choices:
        specs[choice["letter"]] = read_json_file(choice["candidate_dir"] / "candidate_spec.json")
    return specs


def _shared_and_variant_specs(
    bundle: QuestionBundle,
) -> tuple[list[tuple[str, Any]], dict[str, list[tuple[str, Any]]]]:
    specs = _candidate_specs(bundle)
    flattened = {letter: _flatten_spec(spec) for letter, spec in specs.items()}
    labels = list(flattened.keys())
    shared: list[tuple[str, Any]] = []
    variant: dict[str, list[tuple[str, Any]]] = {letter: [] for letter in labels}

    for key in flattened[labels[0]]:
        values = [flattened[letter].get(key) for letter in labels]
        if all(value == values[0] for value in values):
            if values[0] is not None:
                shared.append((key, values[0]))
        else:
            for letter, value in zip(labels, values, strict=True):
                if value is not None:
                    variant[letter].append((key, value))

    return shared, variant


def _render_shared_panel(
    q: dict[str, Any],
    shared: list[tuple[str, Any]],
) -> None:
    preferred = [
        "training steps",
        "batch size",
        "total samples seen",
        "loss",
        "optimizer",
        "learning rate",
    ]
    shared_map = dict(shared)
    rows = [(key, shared_map[key]) for key in preferred if key in shared_map]
    if q.get("evaluation", {}).get("n_seeds") is not None:
        rows.append(("evaluation", f"{q['evaluation']['n_seeds']} seeds"))
    if not rows:
        rows = shared[:6]

    rows_html = "".join(
        f'<div class="aiq-spec-row"><span>{_escape(label.title())}</span><strong>{_escape(_fmt_value(value))}</strong></div>'
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="aiq-panel aiq-side">
            <div class="aiq-panel-inner">
                <h2>Shared training setup</h2>
                <p class="aiq-side-copy">
                    These constraints are identical for every choice. Compare only what changes across candidates.
                </p>
                <div class="aiq-rule"></div>
                {rows_html}
                <div class="aiq-notice">
                    <span class="aiq-notice-badge">i</span>
                    <div><strong>What should you notice?</strong><br>
                    Reason about capacity, optimizer dynamics, and whether each setup can use the fixed budget efficiently.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_protocol(q: dict[str, Any], shared: list[tuple[str, Any]]) -> None:
    st.markdown("#### Protocol")
    st.write("Question type:", q.get("type", "-"))
    st.write("Varying axes:", ", ".join(q.get("varying_axes", [])) or "-")
    st.write("Invariant axes:", ", ".join(q.get("invariant_axes", [])) or "-")
    st.write("Profile:", q.get("profile", "-"))
    st.write("Run:", q.get("question_run_id", "-"))
    if shared:
        st.markdown("##### Shared fields")
        st.table([{"field": key, "value": _fmt_value(value)} for key, value in shared])


def _render_source(bundle: QuestionBundle) -> None:
    st.markdown("#### Dataset files")
    ds_paths = dataset_file_paths(bundle.dataset_dir)
    _render_file_panel(ds_paths, st.session_state.dataset_file, "dataset")


def _render_evidence_panel(
    bundle: QuestionBundle,
    q: dict[str, Any],
    shared: list[tuple[str, Any]],
    *,
    committed: bool,
) -> None:
    current = st.session_state.evidence_tab
    if current not in EVIDENCE_TABS:
        current = EVIDENCE_TABS[0]
    st.markdown(
        """
        <div class="aiq-panel">
            <div class="aiq-panel-inner">
                <div class="aiq-panel-head">
                    <div class="aiq-dot-title">
                        <span class="aiq-dot"></span>
                        <span class="aiq-panel-title">Evidence 01</span>
                        <span class="aiq-panel-muted">Question artifacts</span>
                    </div>
                    <span class="aiq-panel-muted">A B C to select</span>
                </div>
        """,
        unsafe_allow_html=True,
    )
    picked = st.radio(
        "Evidence view",
        EVIDENCE_TABS,
        index=EVIDENCE_TABS.index(current),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.evidence_tab = picked

    if picked == "Evidence":
        st.markdown('<div class="aiq-plot-frame">', unsafe_allow_html=True)
        _plot_dataset(bundle)
        st.markdown("</div>", unsafe_allow_html=True)
    elif picked == "Protocol":
        _render_protocol(q, shared)
    elif picked == "Prompt":
        st.code(bundle.prompt_text, language="markdown")
    else:
        _render_source(bundle)

    if not committed:
        st.caption("Result metrics stay hidden until you lock an answer.")
    st.markdown("</div></div>", unsafe_allow_html=True)


def _candidate_display_rows(rows: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    priority = [
        "layers",
        "width",
        "activations",
        "residual",
        "layer norm",
        "optimizer",
        "learning rate",
        "loss",
        "lambda",
        "batch size",
    ]
    row_map = dict(rows)
    ordered = [(key, row_map[key]) for key in priority if key in row_map]
    for key, value in rows:
        if key not in priority:
            ordered.append((key, value))
    return ordered[:5]


def _card_state(
    letter: str,
    *,
    committed: bool,
    committed_letter: str | None,
    correct_letter: str,
    focused: bool,
) -> tuple[str, str, str]:
    if committed and letter == correct_letter:
        return "#20a87e", "#20a87e", "#20a87e"
    if committed and letter == committed_letter and letter != correct_letter:
        return "#d94b45", "#d94b45", "#d94b45"
    accent = ACCENTS[(ord(letter[0]) - ord("A")) % len(ACCENTS)]
    border = accent if focused else "#24262c"
    fill = accent if focused else "transparent"
    return accent, border, fill


def _render_candidate_visual_card(
    choice: dict[str, Any],
    q: dict[str, Any],
    rows: list[tuple[str, Any]],
    *,
    committed: bool,
    committed_letter: str | None,
    focus_letter: str | None,
) -> None:
    letter = choice["letter"]
    accent, border, fill = _card_state(
        letter,
        committed=committed,
        committed_letter=committed_letter,
        correct_letter=q["correct_letter"],
        focused=focus_letter == letter,
    )
    row_html = "".join(
        f"<span>{_escape(label)}</span><strong>{_escape(_fmt_value(value))}</strong>"
        for label, value in _candidate_display_rows(rows)
    )
    focused_class = " is-focused" if focus_letter == letter else ""
    st.markdown(
        f"""
        <div class="aiq-candidate-card{focused_class}" style="--card-accent:{accent};--card-border:{border};--dot-fill:{fill};">
            <div class="aiq-candidate-top">
                <div>
                    <div class="aiq-letter">{_escape(letter)}</div>
                    <div class="aiq-candidate-id">{_escape(choice["candidate_id"])}</div>
                    <div class="aiq-mini-arch" aria-hidden="true"></div>
                </div>
                <div class="aiq-card-body">{row_html}</div>
                <span class="aiq-select-dot" aria-hidden="true"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_answer_commit_panel(
    bundle: QuestionBundle,
    q: dict[str, Any],
    *,
    committed: bool,
) -> None:
    st.markdown(
        """
        <div class="aiq-commit">
            <h3>Choose one setup</h3>
        """,
        unsafe_allow_html=True,
    )
    if committed:
        picked = st.session_state.committed_letter
        correct = q["correct_letter"]
        if picked == correct:
            st.markdown(f"<p>Locked answer: {_escape(picked)}. Correct.</p>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p>Locked answer: {_escape(picked)}. Correct answer: {_escape(correct)}.</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<p>No result data is shown before commitment.</p>", unsafe_allow_html=True)
    st.markdown('<div class="aiq-answer-buttons">', unsafe_allow_html=True)

    cols = st.columns(max(1, len(bundle.choices)))
    for col, choice in zip(cols, bundle.choices, strict=True):
        with col:
            letter = choice["letter"]
            button_type = "primary" if st.session_state.focus_letter == letter else "secondary"
            if st.button(letter, key=f"pick_{letter}", use_container_width=True, type=button_type):
                if committed:
                    st.session_state.focus_letter = letter
                    st.session_state.info_letter = None
                else:
                    st.session_state.focus_letter = letter
                st.rerun()

    disabled = committed or st.session_state.focus_letter is None
    label = "Answer locked" if committed else "Lock answer ->"
    if st.button(label, type="primary", use_container_width=True, disabled=disabled):
        _commit_selection(q, st.session_state.focus_letter)
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)


def _render_answer_banner(q: dict[str, Any], committed_letter: str) -> None:
    correct = q["correct_letter"]
    if committed_letter == correct:
        st.success(f"Correct. {committed_letter} achieves the best test metric.")
    else:
        st.error(
            f"Incorrect. You picked {committed_letter}; the correct answer is {correct}."
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
    st.table(
        [
            {
                "choice": row["letter"],
                "candidate": row["candidate_id"],
                "mean": row["mean"],
                "std": row["std"],
                "best": "yes" if row["correct"] else "",
            }
            for row in rows
        ]
    )


def _render_candidate_files(bundle: QuestionBundle, committed: bool) -> None:
    inspect_letter = st.session_state.info_letter or st.session_state.focus_letter
    if not inspect_letter:
        return
    selected_choice = next(c for c in bundle.choices if c["letter"] == inspect_letter)
    st.markdown('<div class="aiq-file-panel">', unsafe_allow_html=True)
    st.markdown(f"#### Files - Choice **{inspect_letter}** - `{selected_choice['candidate_id']}`")
    paths = candidate_file_paths(selected_choice["candidate_dir"], include_summary=committed)
    spec = read_json_file(selected_choice["candidate_dir"] / "candidate_spec.json")
    budget = spec.get("budget", {})
    _render_file_panel(
        paths,
        st.session_state.inspect_file,
        f"candidate_{inspect_letter}",
        total_samples_seen=int(budget["total_samples_seen"])
        if budget.get("total_samples_seen") is not None
        else None,
        batch_size=int(budget["batch_size"]) if budget.get("batch_size") is not None else None,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_candidate_rail(
    bundle: QuestionBundle,
    q: dict[str, Any],
    variant: dict[str, list[tuple[str, Any]]],
    *,
    committed: bool,
) -> None:
    st.markdown(
        f'<div class="aiq-candidate-grid" style="--choice-count:{len(bundle.choices)}">',
        unsafe_allow_html=True,
    )
    candidate_cols = st.columns([1] * len(bundle.choices) + [0.95])
    for col, choice in zip(candidate_cols[:-1], bundle.choices, strict=True):
        with col:
            _render_candidate_visual_card(
                choice,
                q,
                variant.get(choice["letter"], []),
                committed=committed,
                committed_letter=st.session_state.committed_letter,
                focus_letter=st.session_state.focus_letter,
            )
            if st.button(
                "Select" if not committed else "Inspect",
                key=f"select_{choice['letter']}",
                use_container_width=True,
            ):
                st.session_state.focus_letter = choice["letter"]
                st.session_state.info_letter = None
                st.rerun()
    with candidate_cols[-1]:
        _render_answer_commit_panel(bundle, q, committed=committed)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_question_page(
    bundle: QuestionBundle,
    q: dict[str, Any],
    *,
    committed: bool,
) -> None:
    shared, variant = _shared_and_variant_specs(bundle)
    _render_question_hero(q, bundle)
    left, right = st.columns([1.9, 0.72], gap="large")
    with left:
        _render_evidence_panel(bundle, q, shared, committed=committed)
    with right:
        _render_shared_panel(q, shared)
    _render_candidate_rail(bundle, q, variant, committed=committed)

    if committed:
        st.markdown('<div class="aiq-status">', unsafe_allow_html=True)
        _render_answer_banner(q, st.session_state.committed_letter)
        _render_ranked_metrics(bundle, q)
        st.markdown("</div>", unsafe_allow_html=True)

    _render_candidate_files(bundle, committed)


def main() -> None:
    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not st.session_state.started:
        _render_welcome_page()
        return

    data_root = st.session_state.data_root
    bundle, pool, current_index = _ensure_bundle(data_root)
    _render_topbar(pool, current_index, data_root)

    if bundle is None:
        st.info("No questions found under the data root. Generate question artifacts to begin.")
        return

    q = bundle.question
    committed = st.session_state.committed_letter is not None
    _render_question_page(bundle, q, committed=committed)


if __name__ == "__main__":
    main()
