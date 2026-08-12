"""Streamlit UI for inspecting ArchitectureIQ question artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import reload
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Prefer the package from this checkout over another editable worktree.
_LOCAL_SRC = Path(__file__).resolve().parents[2] / "src"
if _LOCAL_SRC.is_dir() and str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))
import artifact_loader  # noqa: E402
import feedback  # noqa: E402
import feedback_browser_outbox  # noqa: E402
import feedback_outbox  # noqa: E402
import feedback_recovery  # noqa: E402
import prompt_format  # noqa: E402
import release_manifest  # noqa: E402
import surprise_catalog  # noqa: E402
import surprise_recommender  # noqa: E402

reload(artifact_loader)
reload(feedback_browser_outbox)
reload(feedback_outbox)
reload(feedback_recovery)
reload(prompt_format)
reload(surprise_recommender)
reload(surprise_catalog)
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
from architecture_iq.models.kan import BASE_ACTIVATIONS  # noqa: E402
from architecture_iq.profile import load_profile  # noqa: E402


QUESTION_PACKS_ROOT = (
    Path(__file__).resolve().parents[2] / "benchmark_releases" / "question_packs"
)
LOCAL_QUESTION_PACK = "local"

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

COMMENT_CATEGORY_LABELS = {
    "question_quality": "Question wording or quality",
    "answer_or_result": "Answer or ground-truth result",
    "custom_setting": "Custom / proposed setting",
    "bug": "Website bug",
    "suggestion": "Product suggestion",
    "other": "Other",
}

BUNDLED_DATA_ROOT = "examples/quiz_demo/bundle"
DATA_ROOT_ENV = "ARCHITECTURE_IQ_DATA_ROOT"
INSPECTOR_ENTRY_PATH = "tools/question_inspector/app.py"
RUNTIME_GIT_SHA_ENV_NAMES = (
    "ARCHITECTURE_IQ_GIT_SHA",
    "GIT_COMMIT",
    "COMMIT_SHA",
    "SOURCE_VERSION",
)
GIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")
BROWSER_OUTBOX_DISABLE_ENV = "ARCHITECTURE_IQ_DISABLE_BROWSER_OUTBOX"
SURPRISE_POLICY_VERSION = "surprise_policy_v1"
SEQUENTIAL_POLICY_VERSION = "sequential_fallback_v1"
RANDOM_POLICY_VERSION = "random_navigation_v1"
MANUAL_POLICY_VERSION = "manual_navigation_v1"
_STREAMLIT_TESTING_KEY = "$$STREAMLIT_INTERNAL_KEY_TESTING"
_BROWSER_OUTBOX_COMPONENT = components.declare_component(
    "architecture_iq_browser_outbox",
    path=Path(__file__).resolve().parent / "browser_outbox_component",
)


def _init_state() -> None:
    defaults = {
        "bundle": None,
        "committed_letter": None,
        "focus_letter": None,
        "info_letter": None,
        "inspect_file": "candidate_spec.json",
        "dataset_file": "dataset_spec.json",
        "question_path": None,
        "data_root": _initial_data_root(),
        "question_pool": [],
        "quiz_results": {},
        "quiz_answers": {},
        "review_collection_path": None,
        "active_question_pack": None,
        "setting_notice": None,
        "comment_notices": {},
        "feedback_uploaded_event_ids": [],
        "feedback_quarantined_events": {},
        "feedback_recovery_outboxes": {},
        "feedback_upload_notice": None,
        "question_presentation_notice": None,
        "feedback_browser_outbox_generation": 0,
        "feedback_browser_outbox_hydrated": False,
        "feedback_browser_outbox_invalid_raw": None,
        "feedback_browser_outbox_write_blocked": False,
        "feedback_browser_outbox_notice": None,
        "feedback_browser_outbox_saved_checksum": None,
        "quiz_manifest": None,
        "quiz_manifest_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "quiz_attempt_id" not in st.session_state:
        st.session_state.quiz_attempt_id = _new_quiz_attempt_id()
    if "feedback_trace" not in st.session_state:
        st.session_state.feedback_trace = feedback.SessionTrace.new()


def _new_quiz_attempt_id() -> str:
    return f"attempt_{secrets.token_hex(12)}"


def _feedback_trace() -> feedback.SessionTrace:
    return st.session_state.feedback_trace


def _browser_outbox_enabled() -> bool:
    disabled = os.environ.get(BROWSER_OUTBOX_DISABLE_ENV, "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    # Streamlit's Python AppTest cannot execute a component iframe. Keep those
    # tests deterministic while the pure snapshot and real-browser tests cover
    # this boundary separately.
    return _STREAMLIT_TESTING_KEY not in st.session_state


def _touch_browser_outbox() -> None:
    generation = st.session_state.get("feedback_browser_outbox_generation", 0)
    if isinstance(generation, bool) or not isinstance(generation, int):
        generation = 0
    st.session_state.feedback_browser_outbox_generation = min(
        generation + 1,
        feedback.MAX_SAFE_JSON_INTEGER,
    )


def _restore_quiz_progress_from_trace(
    trace: feedback.SessionTrace,
    *,
    attempt_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    answers: dict[str, dict[str, Any]] = {}
    results: dict[str, bool] = {}
    for event in trace.events:
        if event.get("event_type") != "answer_submitted":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("attempt_id") != attempt_id:
            continue
        question_version = event.get("question_version")
        question_id = event.get("question_id")
        selected_letter = payload.get("selected_letter")
        is_correct = payload.get("is_correct")
        if (
            not isinstance(question_version, str)
            or question_version in answers
            or not isinstance(question_id, str)
            or not isinstance(selected_letter, str)
            or not isinstance(is_correct, bool)
        ):
            continue
        selected_candidate_id = payload.get("selected_candidate_id")
        answers[question_version] = {
            "question_id": question_id,
            "selected_letter": selected_letter,
            "selected_candidate_id": (
                selected_candidate_id
                if isinstance(selected_candidate_id, str)
                else None
            ),
            "is_correct": is_correct,
        }
        results[question_version] = is_correct
    return answers, results


def _apply_browser_outbox_snapshot(
    snapshot: feedback_browser_outbox.BrowserOutboxSnapshot,
) -> None:
    st.session_state.feedback_trace = snapshot.trace
    st.session_state.quiz_attempt_id = snapshot.current_attempt_id
    st.session_state.feedback_uploaded_event_ids = list(snapshot.acknowledged_event_ids)
    st.session_state.feedback_quarantined_events = snapshot.quarantined_by_id()
    st.session_state.feedback_browser_outbox_generation = snapshot.generation
    st.session_state.feedback_browser_outbox_saved_checksum = snapshot.checksum
    answers, results = _restore_quiz_progress_from_trace(
        snapshot.trace,
        attempt_id=snapshot.current_attempt_id,
    )
    st.session_state.quiz_answers = answers
    st.session_state.quiz_results = results


def _browser_component_response(
    value: Any,
    *,
    request_id: str,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("request_id") != request_id:
        return None
    status = value.get("status")
    if status not in {"empty", "loaded", "saved", "cleared", "error"}:
        return None
    return value


def _hydrate_browser_outbox() -> None:
    if st.session_state.feedback_browser_outbox_hydrated:
        return
    if not _browser_outbox_enabled():
        st.session_state.feedback_browser_outbox_hydrated = True
        return

    request_id = "load_browser_outbox_v1"
    raw_response = _BROWSER_OUTBOX_COMPONENT(
        operation="load",
        request_id=request_id,
        default=None,
        key="feedback_browser_outbox_load",
    )
    response = _browser_component_response(raw_response, request_id=request_id)
    if response is None:
        st.info("Restoring the browser feedback outbox…")
        st.stop()

    status = response["status"]
    if status == "empty":
        st.session_state.feedback_browser_outbox_hydrated = True
        return
    if status == "error":
        error_name = response.get("error")
        st.session_state.feedback_browser_outbox_notice = {
            "level": "warning",
            "message": (
                "Browser persistence is unavailable for this session"
                + (f" ({error_name})." if isinstance(error_name, str) else ".")
                + " Download the session JSON before closing the page."
            ),
        }
        st.session_state.feedback_browser_outbox_hydrated = True
        st.session_state.feedback_browser_outbox_write_blocked = True
        return
    if status != "loaded" or not isinstance(response.get("value"), str):
        st.session_state.feedback_browser_outbox_notice = {
            "level": "warning",
            "message": (
                "The browser outbox returned an invalid response. Download session "
                "JSON before closing the page."
            ),
        }
        st.session_state.feedback_browser_outbox_hydrated = True
        st.session_state.feedback_browser_outbox_write_blocked = True
        return

    raw = response["value"]
    try:
        snapshot = feedback_browser_outbox.parse_browser_outbox(raw)
    except feedback_browser_outbox.BrowserOutboxError as exc:
        st.session_state.feedback_browser_outbox_invalid_raw = (
            raw
            if len(raw.encode("utf-8", errors="ignore"))
            <= feedback_browser_outbox.MAX_BROWSER_OUTBOX_BYTES
            else None
        )
        st.session_state.feedback_browser_outbox_write_blocked = True
        st.session_state.feedback_browser_outbox_notice = {
            "level": "error",
            "message": (
                "The saved browser outbox is invalid and was not restored or "
                f"overwritten: {exc}"
            ),
        }
    else:
        _apply_browser_outbox_snapshot(snapshot)
        st.session_state.feedback_browser_outbox_notice = {
            "level": "success",
            "message": (
                f"Restored {len(snapshot.trace.events)} browser-saved event(s) "
                f"for session {snapshot.trace.session_id}."
            ),
        }
    st.session_state.feedback_browser_outbox_hydrated = True


def _serialize_current_browser_outbox() -> str:
    return feedback_browser_outbox.serialize_browser_outbox(
        generation=st.session_state.feedback_browser_outbox_generation,
        current_attempt_id=st.session_state.quiz_attempt_id,
        trace=_feedback_trace(),
        acknowledged_event_ids=_uploaded_event_ids(),
        quarantined_events=_quarantined_event_records(),
    )


def _persist_browser_outbox() -> None:
    if (
        not _browser_outbox_enabled()
        or not st.session_state.feedback_browser_outbox_hydrated
        or st.session_state.feedback_browser_outbox_write_blocked
    ):
        return
    try:
        serialized = _serialize_current_browser_outbox()
        snapshot = feedback_browser_outbox.parse_browser_outbox(serialized)
    except feedback_browser_outbox.BrowserOutboxError as exc:
        st.session_state.feedback_browser_outbox_notice = {
            "level": "error",
            "message": (
                f"The live outbox could not be saved in the browser: {exc}. "
                "Download the session JSON before closing the page."
            ),
        }
        st.session_state.feedback_browser_outbox_write_blocked = True
        return

    request_id = f"save_{snapshot.checksum}"
    raw_response = _BROWSER_OUTBOX_COMPONENT(
        operation="save",
        request_id=request_id,
        value=serialized,
        default=None,
        key="feedback_browser_outbox_save",
    )
    response = _browser_component_response(raw_response, request_id=request_id)
    if response is None:
        return
    if response["status"] == "saved":
        st.session_state.feedback_browser_outbox_saved_checksum = snapshot.checksum
        return
    if response["status"] == "error":
        error_name = response.get("error")
        st.session_state.feedback_browser_outbox_notice = {
            "level": "error",
            "message": (
                "Browser outbox persistence failed"
                + (f" ({error_name})." if isinstance(error_name, str) else ".")
                + " Download the session JSON before closing the page."
            ),
        }
        st.session_state.feedback_browser_outbox_write_blocked = True


def _start_new_feedback_session() -> None:
    st.session_state.feedback_trace = feedback.SessionTrace.new()
    st.session_state.feedback_uploaded_event_ids = []
    st.session_state.feedback_quarantined_events = {}
    st.session_state.comment_notices = {}
    st.session_state.feedback_upload_notice = None
    st.session_state.question_presentation_notice = None
    st.session_state.quiz_results = {}
    st.session_state.quiz_answers = {}
    st.session_state.quiz_attempt_id = _new_quiz_attempt_id()
    st.session_state.feedback_browser_outbox_generation = 0
    st.session_state.feedback_browser_outbox_invalid_raw = None
    st.session_state.feedback_browser_outbox_write_blocked = False
    st.session_state.feedback_browser_outbox_notice = {
        "level": "info",
        "message": "Started a new browser feedback session.",
    }
    _reset_quiz_state()


def _question_session_key(q: Mapping[str, Any]) -> str:
    return feedback.question_version(q)


def _event_context(q: Mapping[str, Any]) -> dict[str, Any]:
    significance = q.get("significance", {})
    metric = significance.get("metric") if isinstance(significance, Mapping) else None
    context = {
        "attempt_id": st.session_state.quiz_attempt_id,
        "family": q.get("family"),
        "dataset_id": q.get("dataset_id"),
        "question_type": q.get("type"),
        "selection_metric": metric,
        "budget": q.get("budget"),
    }
    release_id = _matching_release_id(q)
    if release_id is not None:
        context["release_id"] = release_id
    return context


def _matching_release_id(q: Mapping[str, Any]) -> str | None:
    manifest = st.session_state.get("quiz_manifest")
    bundle = st.session_state.get("bundle")
    if not isinstance(manifest, release_manifest.QuizManifest):
        return None
    if not isinstance(bundle, QuestionBundle) or bundle.question is not q:
        return None
    active_root = _resolve_data_root(str(st.session_state.get("data_root", "data")))
    if manifest.data_root != active_root or bundle.data_root.resolve() != active_root:
        return None
    question_id = q.get("question_id")
    if not isinstance(question_id, str):
        return None
    return manifest.release_id_for(
        question_id,
        feedback.question_version(q),
        question_path=bundle.question_root,
    )


def _reset_quiz_state(q: Mapping[str, Any] | None = None) -> None:
    saved = None
    if q is not None:
        saved = st.session_state.quiz_answers.get(_question_session_key(q))
    letter = saved.get("selected_letter") if saved else None
    st.session_state.committed_letter = letter
    st.session_state.focus_letter = letter
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
    st.session_state.quiz_answers = {}
    st.session_state.quiz_attempt_id = _new_quiz_attempt_id()
    _touch_browser_outbox()
    _reset_quiz_state()


def _record_answer(q: dict[str, Any], picked_letter: str) -> None:
    question_key = _question_session_key(q)
    if question_key in st.session_state.quiz_answers:
        return
    choice = next(
        choice for choice in q["choices"] if choice["letter"] == picked_letter
    )
    is_correct = picked_letter == q["correct_letter"]
    st.session_state.quiz_answers[question_key] = {
        "question_id": q["question_id"],
        "selected_letter": picked_letter,
        "selected_candidate_id": choice["candidate_id"],
        "is_correct": is_correct,
    }
    st.session_state.quiz_results[question_key] = is_correct
    _feedback_trace().record_answer(
        q,
        selected_letter=picked_letter,
        selected_candidate_id=choice["candidate_id"],
        extra={**_event_context(q), "is_correct": is_correct},
    )
    _touch_browser_outbox()


def _question_reaction(q: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the first surprise reaction for this question and attempt."""
    question_version = _question_session_key(q)
    attempt_id = st.session_state.quiz_attempt_id
    for event in _feedback_trace().events:
        if (
            event.get("event_type") != "question_reaction_submitted"
            or event.get("question_version") != question_version
        ):
            continue
        payload = event.get("payload")
        if (
            isinstance(payload, Mapping)
            and payload.get("attempt_id") == attempt_id
            and payload.get("reaction") == "surprise"
        ):
            return event
    return None


def _record_question_reaction(
    q: Mapping[str, Any],
    *,
    value: bool,
) -> dict[str, Any]:
    context = _event_context(q)
    attempt_id = context.pop("attempt_id")
    release_id = context.pop("release_id", None)
    event = _feedback_trace().record_question_reaction(
        q,
        value=value,
        attempt_id=attempt_id,
        release_id=release_id,
        extra=context,
    )
    _touch_browser_outbox()
    return event


def _presentation_decision(
    *,
    policy_version: str,
    mode: str,
    propensity: float,
    source: str,
) -> dict[str, Any]:
    return {
        "decision_id": f"decision_{secrets.token_hex(16)}",
        "policy_version": policy_version,
        "mode": mode,
        "propensity": propensity,
        "source": source,
    }


def _record_question_presentation(
    q: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    release_id = _matching_release_id(q)
    if release_id is None:
        return None
    attempt_id = st.session_state.quiz_attempt_id
    position = 1 + sum(
        event.get("event_type") == "question_presented"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("attempt_id") == attempt_id
        for event in _feedback_trace().events
    )
    event = _feedback_trace().record_question_presented(
        q,
        attempt_id=attempt_id,
        release_id=release_id,
        decision_id=decision.get("decision_id"),
        policy_version=decision.get("policy_version"),
        mode=decision.get("mode"),
        propensity=decision.get("propensity"),
        source=decision.get("source"),
        position=position,
    )
    _touch_browser_outbox()
    return event


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


def _feedback_config() -> feedback.FeedbackConfig:
    values: Mapping[str, Any] = {}
    try:
        configured = st.secrets.get("feedback", {})
    except FileNotFoundError:
        configured = {}
    if isinstance(configured, Mapping):
        values = configured
    return feedback.FeedbackConfig.from_sources(
        endpoint=values.get("endpoint") or values.get("url"),
        bearer_token=values.get("token") or values.get("bearer_token"),
        timeout_seconds=values.get("timeout_seconds"),
    )


def _feedback_client() -> feedback.FeedbackClient:
    return feedback.FeedbackClient(_feedback_config())


def _upload_receipt_summary(
    receipt: feedback.UploadReceipt,
    *,
    sent_count: int,
) -> str:
    response = receipt.response
    if isinstance(response, Mapping):
        values = [
            response.get("accepted"),
            response.get("duplicate"),
            response.get("conflict", 0),
            response.get("rejected"),
        ]
        if all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            accepted, duplicate, conflict, rejected = values
            return (
                f"{sent_count} sent · {accepted} new · {duplicate} duplicate · "
                f"{conflict} conflict · {rejected} rejected"
            )
    return f"{sent_count} sent · HTTP {receipt.status_code}"


def _upload_conflict_message(
    exc: feedback.FeedbackUploadConflictError,
    *,
    subject: str,
) -> str:
    request_note = f" Request: {exc.request_id}." if exc.request_id else ""
    count_note = (
        f" The server identified {exc.conflict_count} conflicting event ID(s)."
        if exc.conflict_count is not None
        else ""
    )
    return (
        f"{subject} was not uploaded because an event ID already stores different "
        "logical content. The immutable local event is quarantined and excluded "
        "from future batch retries, so other pending events can still upload. "
        f"This is not a retryable network error.{count_note}"
        f"{request_note}"
    )


def _uploaded_event_ids() -> set[str]:
    raw_ids = st.session_state.get("feedback_uploaded_event_ids", [])
    if not isinstance(raw_ids, (list, tuple, set)):
        return set()
    return {event_id for event_id in raw_ids if isinstance(event_id, str)}


def _quarantined_event_records() -> dict[str, dict[str, str | None]]:
    raw = st.session_state.get("feedback_quarantined_events", {})
    if not isinstance(raw, Mapping):
        return {}
    records: dict[str, dict[str, str | None]] = {}
    for event_id, value in raw.items():
        if not isinstance(event_id, str) or not isinstance(value, Mapping):
            continue
        request_id = value.get("request_id")
        error_code = value.get("error_code")
        records[event_id] = {
            "event_id": event_id,
            "request_id": request_id if isinstance(request_id, str) else None,
            "error_code": (
                error_code if isinstance(error_code, str) else "EVENT_ID_CONFLICT"
            ),
        }
    return records


def _quarantined_event_ids() -> set[str]:
    return set(_quarantined_event_records())


def _set_comment_notice(
    question_key: str,
    *,
    level: str,
    message: str,
    event_id: str | None,
) -> None:
    notices = st.session_state.get("comment_notices", {})
    if not isinstance(notices, dict):
        notices = {}
    notices[question_key] = {
        "level": level,
        "message": message,
        "event_id": event_id,
    }
    st.session_state.comment_notices = notices


def _reconcile_comment_notices(
    *,
    acknowledged_event_ids: set[str],
    quarantined_event_ids: set[str],
) -> None:
    notices = st.session_state.get("comment_notices", {})
    if not isinstance(notices, dict):
        return
    for question_key, raw_notice in list(notices.items()):
        if not isinstance(raw_notice, Mapping):
            continue
        event_id = raw_notice.get("event_id")
        if not isinstance(event_id, str):
            continue
        if event_id in acknowledged_event_ids:
            notices[question_key] = {
                "level": "success",
                "message": "Comment upload was confirmed by the batch uploader.",
                "event_id": event_id,
            }
        elif event_id in quarantined_event_ids:
            notices[question_key] = {
                "level": "error",
                "message": (
                    "This comment is quarantined because its event ID conflicts "
                    "with different stored content. Other pending events can still upload."
                ),
                "event_id": event_id,
            }
    st.session_state.comment_notices = notices


def _mark_events_uploaded(
    events: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    receipt: feedback.UploadReceipt,
) -> bool:
    if not feedback.upload_receipt_acknowledges_all(
        receipt,
        sent_count=len(events),
    ):
        return False
    uploaded = _uploaded_event_ids()
    uploaded.update(
        event_id
        for event in events
        if isinstance((event_id := event.get("event_id")), str)
    )
    st.session_state.feedback_uploaded_event_ids = sorted(uploaded)
    quarantined = _quarantined_event_records()
    for event in events:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            quarantined.pop(event_id, None)
    st.session_state.feedback_quarantined_events = quarantined
    _reconcile_comment_notices(
        acknowledged_event_ids=uploaded,
        quarantined_event_ids=set(quarantined),
    )
    _touch_browser_outbox()
    return True


def _pending_upload_count(events: tuple[dict[str, Any], ...]) -> int:
    try:
        selected = feedback_outbox.pending_events(
            events,
            acknowledged_event_ids=_uploaded_event_ids(),
            quarantined_event_ids=_quarantined_event_ids(),
        )
    except feedback_outbox.FeedbackOutboxError:
        return len(events)
    return len(selected)


def _apply_outbox_result(result: feedback_outbox.OutboxUploadResult) -> None:
    uploaded = _uploaded_event_ids()
    uploaded.update(result.acknowledged_event_ids)
    quarantined = _quarantined_event_records()
    for event_id in result.acknowledged_event_ids:
        quarantined.pop(event_id, None)
    for item in result.quarantined_events:
        quarantined[item.event_id] = item.to_dict()
    st.session_state.feedback_uploaded_event_ids = sorted(uploaded)
    st.session_state.feedback_quarantined_events = quarantined
    _reconcile_comment_notices(
        acknowledged_event_ids=uploaded,
        quarantined_event_ids=set(quarantined),
    )
    _touch_browser_outbox()


def _recovery_outbox_state(
    recovered: feedback_recovery.RecoveredTrace,
) -> tuple[set[str], dict[str, dict[str, str | None]]]:
    """Return safe, content-scoped status for one downloaded trace."""

    event_ids = {
        str(event["event_id"])
        for event in recovered.events
        if isinstance(event.get("event_id"), str)
    }
    raw_outboxes = st.session_state.get("feedback_recovery_outboxes", {})
    raw_state = (
        raw_outboxes.get(recovered.recovery_id, {})
        if isinstance(raw_outboxes, Mapping)
        else {}
    )
    if not isinstance(raw_state, Mapping):
        raw_state = {}

    raw_acknowledged = raw_state.get("acknowledged_event_ids", [])
    acknowledged = (
        {
            event_id
            for event_id in raw_acknowledged
            if isinstance(event_id, str) and event_id in event_ids
        }
        if isinstance(raw_acknowledged, (list, tuple, set))
        else set()
    )
    raw_quarantined = raw_state.get("quarantined_events", {})
    quarantined: dict[str, dict[str, str | None]] = {}
    if isinstance(raw_quarantined, Mapping):
        for event_id, value in raw_quarantined.items():
            if (
                not isinstance(event_id, str)
                or event_id not in event_ids
                or event_id in acknowledged
                or not isinstance(value, Mapping)
            ):
                continue
            request_id = value.get("request_id")
            error_code = value.get("error_code")
            quarantined[event_id] = {
                "event_id": event_id,
                "request_id": request_id if isinstance(request_id, str) else None,
                "error_code": (
                    error_code if isinstance(error_code, str) else "EVENT_ID_CONFLICT"
                ),
            }
    return acknowledged, quarantined


def _apply_recovery_outbox_result(
    recovered: feedback_recovery.RecoveredTrace,
    result: feedback_outbox.OutboxUploadResult,
) -> None:
    """Persist only safe receipt metadata, namespaced by full trace content."""

    event_ids = {
        str(event["event_id"])
        for event in recovered.events
        if isinstance(event.get("event_id"), str)
    }
    acknowledged, quarantined = _recovery_outbox_state(recovered)
    acknowledged.update(set(result.acknowledged_event_ids) & event_ids)
    for event_id in acknowledged:
        quarantined.pop(event_id, None)
    for item in result.quarantined_events:
        if item.event_id in event_ids and item.event_id not in acknowledged:
            quarantined[item.event_id] = item.to_dict()

    raw_outboxes = st.session_state.get("feedback_recovery_outboxes", {})
    outboxes = dict(raw_outboxes) if isinstance(raw_outboxes, Mapping) else {}
    outboxes[recovered.recovery_id] = {
        "acknowledged_event_ids": sorted(acknowledged),
        "quarantined_events": quarantined,
    }
    st.session_state.feedback_recovery_outboxes = outboxes


def _recovery_pending_events(
    recovered: feedback_recovery.RecoveredTrace,
) -> tuple[dict[str, Any], ...]:
    acknowledged, quarantined = _recovery_outbox_state(recovered)
    return feedback_outbox.pending_events(
        recovered.events,
        acknowledged_event_ids=acknowledged,
        quarantined_event_ids=quarantined,
    )


def _quarantine_event(
    event: Mapping[str, Any],
    exc: feedback.FeedbackUploadConflictError,
) -> None:
    event_id = event.get("event_id")
    if not isinstance(event_id, str):
        return
    records = _quarantined_event_records()
    records[event_id] = {
        "event_id": event_id,
        "request_id": exc.request_id,
        "error_code": exc.error_code or "EVENT_ID_CONFLICT",
    }
    st.session_state.feedback_quarantined_events = records
    _reconcile_comment_notices(
        acknowledged_event_ids=_uploaded_event_ids(),
        quarantined_event_ids=set(records),
    )
    _touch_browser_outbox()


def _render_outbox_result(
    result: feedback_outbox.OutboxUploadResult,
    *,
    subject: str,
) -> None:
    """Render a payload-free result shared by live and recovery uploads."""

    if result.all_acknowledged:
        st.success(
            f"{subject} complete · {len(result.acknowledged_event_ids)} event(s) "
            f"confirmed in {result.batch_request_count} batch request(s)."
        )
        return
    if result.quarantined_events and result.issue is None:
        local_limit_count = sum(
            item.error_code == "LOCAL_BODY_LIMIT" for item in result.quarantined_events
        )
        conflict_count = len(result.quarantined_events) - local_limit_count
        reasons: list[str] = []
        if conflict_count:
            reasons.append(f"{conflict_count} server content conflict")
        if local_limit_count:
            reasons.append(f"{local_limit_count} local body-limit rejection")
        st.warning(
            f"{subject}: {len(result.acknowledged_event_ids)} event(s) confirmed; "
            f"{len(result.quarantined_events)} event(s) quarantined "
            f"({', '.join(reasons)}). "
            "Other pending events were not blocked."
        )
        return
    issue = result.issue
    if issue is None:  # Defensive: this state is not produced by the outbox.
        st.error(f"{subject} stopped without a complete receiver acknowledgement.")
        return
    request_note = f" Request: {issue.request_id}." if issue.request_id else ""
    message = (
        f"{issue.message} {len(result.acknowledged_event_ids)} confirmed · "
        f"{len(result.quarantined_events)} quarantined · "
        f"{len(result.remaining_event_ids)} still pending.{request_note}"
    )
    if issue.retryable:
        st.warning(f"{message} Retrying unchanged event IDs is safe.")
    else:
        st.error(
            f"{message} Fix the endpoint, credentials, or receipt contract "
            "before retrying."
        )


def _render_recovery_upload(client: feedback.FeedbackClient | None) -> None:
    """Validate and upload a previously downloaded trace without importing it."""

    st.divider()
    st.markdown("**Recover a downloaded session**")
    st.caption(
        "Upload an ArchitectureIQ session JSON after a refresh or device restart. "
        "The file is parsed as data only; no code is imported or executed."
    )
    uploaded_file = st.file_uploader(
        "Saved session JSON (maximum 10 MiB)",
        type=["json"],
        accept_multiple_files=False,
        key="recover_session_trace_json",
        help=(
            "Only strict ArchitectureIQ session_trace JSON is accepted. Invalid or "
            "oversized files are rejected before any feedback network request."
        ),
    )
    if uploaded_file is None:
        return

    file_size = getattr(uploaded_file, "size", None)
    if (
        isinstance(file_size, int)
        and not isinstance(file_size, bool)
        and file_size > feedback_recovery.MAX_RECOVERY_FILE_BYTES
    ):
        st.error(
            "Saved session trace exceeds the 10 MiB recovery limit and was not read "
            "or sent."
        )
        return
    try:
        raw = uploaded_file.getvalue()
        recovered = feedback_recovery.parse_recovered_trace(raw)
        pending_events = _recovery_pending_events(recovered)
    except (OSError, feedback_recovery.FeedbackRecoveryError) as exc:
        st.error(f"Saved session trace was rejected locally: {exc}")
        return
    except feedback_outbox.FeedbackOutboxError as exc:
        st.error(f"Saved session outbox is invalid: {exc}")
        return

    acknowledged, quarantined = _recovery_outbox_state(recovered)
    st.caption(
        f"Validated trace · {recovered.event_count} event(s) · "
        f"{len(pending_events)} pending · {len(acknowledged)} confirmed · "
        f"{len(quarantined)} quarantined · content …{recovered.recovery_id[-12:]}"
    )
    if quarantined:
        st.warning(
            "Conflicting IDs from this exact saved trace are quarantined locally. "
            "Changing file content creates a separate recovery outbox."
        )
    if client is None or not client.is_configured:
        st.caption(
            "Configure both the upload endpoint and Bearer token to send this "
            "validated trace."
        )
    elif not pending_events:
        if quarantined:
            st.warning(
                "Every event in this file is already confirmed or quarantined in "
                "this browser session."
            )
        else:
            st.success(
                "Every event in this file has already received a complete storage "
                "acknowledgement in this browser session."
            )

    if st.button(
        "Upload recovered pending events",
        key="upload_recovered_session_trace",
        disabled=client is None or not client.is_configured or not pending_events,
        use_container_width=True,
    ):
        assert client is not None
        try:
            with st.spinner("Uploading recovered pending events…"):
                result = feedback_recovery.upload_recovered_trace(
                    client,
                    recovered,
                    acknowledged_event_ids=acknowledged,
                    quarantined_event_ids=quarantined,
                )
        except (
            feedback_recovery.FeedbackRecoveryError,
            feedback_outbox.FeedbackOutboxError,
        ) as exc:
            st.error(f"Saved session outbox is invalid: {exc}")
        else:
            _apply_recovery_outbox_result(recovered, result)
            _render_outbox_result(result, subject="Recovery upload")


def _render_session_feedback_panel() -> None:
    trace = _feedback_trace()
    events = trace.events
    counts = {
        event_type: sum(event["event_type"] == event_type for event in events)
        for event_type in feedback.EVENT_TYPES
    }
    with st.expander("Session data & upload", expanded=False):
        pending = _pending_upload_count(events)
        quarantine_records = _quarantined_event_records()
        quarantined_ids = set(quarantine_records)
        quarantined_events = tuple(
            event for event in events if event.get("event_id") in quarantined_ids
        )
        st.caption(
            f"{len(events)} event(s) · {counts['answer_submitted']} answer(s) · "
            f"{counts['custom_setting_proposed']} proposed setting(s) · "
            f"{counts['custom_setting_rejected']} rejected setting(s) · "
            f"{counts['question_reaction_submitted']} surprise reaction(s) · "
            f"{counts['comment_submitted']} comment(s) · {pending} pending upload · "
            f"{len(quarantined_events)} quarantined"
        )
        browser_notice = st.session_state.get("feedback_browser_outbox_notice")
        if isinstance(browser_notice, Mapping):
            level = browser_notice.get("level")
            message = browser_notice.get("message")
            if level in {"info", "success", "warning", "error"} and isinstance(
                message, str
            ):
                getattr(st, level)(message)
        presentation_notice = st.session_state.get("question_presentation_notice")
        if isinstance(presentation_notice, str):
            st.warning(presentation_notice)
        if _browser_outbox_enabled() and not st.session_state.get(
            "feedback_browser_outbox_write_blocked", False
        ):
            st.caption(
                "This live outbox is mirrored to this browser's IndexedDB; "
                "endpoint URLs and tokens are never stored there."
            )
        invalid_browser_raw = st.session_state.get(
            "feedback_browser_outbox_invalid_raw"
        )
        if isinstance(invalid_browser_raw, str):
            st.download_button(
                "Download invalid browser outbox for diagnosis",
                data=invalid_browser_raw,
                file_name="architectureiq-invalid-browser-outbox.json",
                mime="application/json",
                use_container_width=True,
                key="download_invalid_browser_outbox",
            )
            if st.button(
                "Discard invalid browser copy and use this new session",
                key="discard_invalid_browser_outbox",
                use_container_width=True,
            ):
                st.session_state.feedback_browser_outbox_invalid_raw = None
                st.session_state.feedback_browser_outbox_write_blocked = False
                st.session_state.feedback_browser_outbox_notice = {
                    "level": "warning",
                    "message": (
                        "Discarded the invalid browser copy. The current new "
                        "session will replace it."
                    ),
                }
                _touch_browser_outbox()
                st.rerun()
        envelope = trace.to_envelope()
        serialized = json.dumps(envelope, ensure_ascii=False, indent=2)
        st.download_button(
            "Download my session JSON",
            data=serialized,
            file_name=f"architectureiq-{trace.session_id}.json",
            mime="application/json",
            disabled=not events,
            use_container_width=True,
            key="download_session_trace_sidebar",
        )
        if quarantined_events:
            quarantine_envelope = feedback.build_session_trace_envelope(
                trace.session_id,
                quarantined_events,
                created_at=trace.created_at,
            )
            st.download_button(
                "Download quarantined events",
                data=json.dumps(quarantine_envelope, ensure_ascii=False, indent=2),
                file_name=f"architectureiq-{trace.session_id}-quarantined.json",
                mime="application/json",
                use_container_width=True,
                key="download_quarantined_trace_sidebar",
            )
            local_limit_count = sum(
                quarantine_records[event_id].get("error_code") == "LOCAL_BODY_LIMIT"
                for event_id in quarantined_ids
            )
            conflict_count = len(quarantined_ids) - local_limit_count
            if conflict_count:
                st.warning(
                    f"{conflict_count} quarantined event ID(s) contain different "
                    "content than the server's first write. They are excluded from "
                    "retry batches so other events can continue uploading."
                )
            if local_limit_count:
                st.warning(
                    f"{local_limit_count} event(s) exceed the receiver's single-event "
                    "body limit. They were not sent and are excluded from retry "
                    "batches; download the quarantined trace for inspection."
                )

        client: feedback.FeedbackClient | None
        try:
            client = _feedback_client()
        except feedback.FeedbackConfigurationError as exc:
            client = None
            st.error(f"Feedback upload configuration is invalid: {exc}")
        if client is not None and not client.is_configured:
            st.caption(
                "Upload endpoint and Bearer token are not fully configured. Session "
                "download and event capture still work locally."
            )
        if client is not None and client.is_configured:
            if st.button(
                "Upload pending session events",
                key="upload_session_trace",
                type="primary",
                disabled=pending == 0,
                use_container_width=True,
            ):
                try:
                    with st.spinner("Uploading pending session events…"):
                        result = feedback_outbox.upload_pending_events(
                            client,
                            events,
                            acknowledged_event_ids=_uploaded_event_ids(),
                            quarantined_event_ids=_quarantined_event_ids(),
                        )
                except feedback_outbox.FeedbackOutboxError as exc:
                    st.error(f"The local feedback outbox is invalid: {exc}")
                else:
                    _apply_outbox_result(result)
                    _render_outbox_result(result, subject="Session upload")

        st.divider()
        confirm_new_session = st.checkbox(
            "I downloaded anything I still need from this session.",
            key="confirm_start_new_feedback_session",
        )
        if st.button(
            "Start a new feedback session",
            key="start_new_feedback_session",
            disabled=not confirm_new_session,
            use_container_width=True,
        ):
            _start_new_feedback_session()
            st.rerun()

        _render_recovery_upload(client)


def _switch_question(
    question_path: Path,
    data_root: str,
    *,
    presentation: Mapping[str, Any] | None = None,
) -> None:
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
    changed = (
        not isinstance(previous_path, str)
        or Path(previous_path).resolve() != question_path
    )
    clear_legacy_question_custom_settings(question_path)

    st.session_state.question_path = str(question_path)
    st.session_state.bundle = _load_selected_question(question_path, data_root)
    bundle = st.session_state.bundle
    _reset_quiz_state(bundle.question if bundle is not None else None)
    if changed and presentation is not None and bundle is not None:
        try:
            _record_question_presentation(bundle.question, presentation)
        except feedback.FeedbackValidationError as exc:
            st.session_state.question_presentation_notice = (
                "Question exposure could not be recorded; this navigation must not "
                f"be used for recommendation evaluation: {exc}"
            )


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


def _bundled_data_root(*, root: Path | None = None) -> Path:
    return ((root or _repo_root()) / BUNDLED_DATA_ROOT).resolve()


def _argument_data_root(argv: list[str] | tuple[str, ...]) -> Path | None:
    """Infer a local data root only from an existing explicit question path."""
    if len(argv) <= 1:
        return None
    candidate = Path(argv[1]).expanduser()
    if not candidate.exists():
        return None
    resolved = candidate.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name == "datasets":
            return ancestor.parent.resolve()
        if (ancestor / "datasets").is_dir():
            return ancestor.resolve()
        if (ancestor / "data" / "datasets").is_dir():
            return (ancestor / "data").resolve()
    return None


def _initial_data_root(
    *,
    environ: Mapping[str, str] | None = None,
    argv: list[str] | tuple[str, ...] | None = None,
    root: Path | None = None,
) -> str:
    """Choose explicit local data when requested, otherwise the bundled release."""
    argument_root = _argument_data_root(sys.argv if argv is None else argv)
    if argument_root is not None:
        return str(argument_root)
    values = os.environ if environ is None else environ
    configured = values.get(DATA_ROOT_ENV, "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())
    return str(_bundled_data_root(root=root))


def _checkout_git_environment() -> dict[str, str]:
    """Return a minimal environment with no inherited Git redirection."""
    return {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _checkout_git_sha(repo_root: Path) -> str | None:
    """Read the deployed checkout commit without consulting a shell."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2,
            env=_checkout_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = result.stdout.strip().lower()
    if result.returncode != 0 or GIT_SHA_PATTERN.fullmatch(candidate) is None:
        return None
    return candidate


def _runtime_git_sha(
    environ: Mapping[str, str] | None = None,
    *,
    repo_root: Path | None = None,
    git_reader: Callable[[Path], str | None] | None = None,
) -> str | None:
    """Return one commit only when runtime declarations and checkout agree."""
    values = os.environ if environ is None else environ
    configured = [
        raw.strip()
        for name in RUNTIME_GIT_SHA_ENV_NAMES
        if (raw := values.get(name)) is not None and raw.strip()
    ]
    if any(GIT_SHA_PATTERN.fullmatch(raw) is None for raw in configured):
        return None
    candidates = {raw.lower() for raw in configured}
    root = repo_root or Path(__file__).resolve().parents[2]
    checkout_sha = (git_reader or _checkout_git_sha)(root)
    if checkout_sha is None or GIT_SHA_PATTERN.fullmatch(checkout_sha) is None:
        return None
    candidates.add(checkout_sha.lower())
    if len(candidates) != 1:
        return None
    return next(iter(candidates))


def _discover_questions(data_root: str) -> list[Path]:
    return list_question_dirs(_resolve_data_root(data_root))


def _load_active_manifest(data_root: str) -> release_manifest.QuizManifest | None:
    root = _resolve_data_root(data_root)
    error: str | None = None
    try:
        manifest = release_manifest.load_quiz_manifest(root)
    except release_manifest.ReleaseManifestError as exc:
        error = str(exc)
        manifest = None
    if manifest is None and error is None and root == _bundled_data_root():
        error = (
            f"required {release_manifest.MANIFEST_FILENAME} is missing from the "
            "version-controlled bundled release"
        )
    st.session_state.quiz_manifest_error = error
    st.session_state.quiz_manifest = manifest
    return manifest


def _ordered_question_pool(
    data_root: str,
    manifest: release_manifest.QuizManifest | None,
) -> list[Path]:
    discovered = _discover_questions(data_root)
    if manifest is None:
        return discovered
    published = manifest.question_dirs()
    published_paths = {path.resolve() for path in published}
    unpublished = [path for path in discovered if path.resolve() not in published_paths]
    return [*published, *unpublished]


def _load_question_pool(
    data_root: str,
) -> tuple[release_manifest.QuizManifest | None, list[Path]]:
    manifest = _load_active_manifest(data_root)
    if st.session_state.quiz_manifest_error:
        return manifest, []
    return manifest, _ordered_question_pool(data_root, manifest)


def _question_pack_registry(
    packs_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return valid tracked question packs keyed by their stable pack ID."""
    root = (packs_root or QUESTION_PACKS_ROOT).resolve()
    if not root.is_dir():
        return {}

    packs: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(root.glob("*/pack.json")):
        try:
            manifest = read_json_file(manifest_path)
        except (OSError, ValueError):
            continue
        pack_root = manifest_path.parent.resolve()
        pack_id = manifest.get("pack_id")
        display_name = manifest.get("display_name")
        collection_value = manifest.get("collection_path")
        data_root_value = manifest.get("data_root")
        if not all(
            isinstance(value, str) and value
            for value in (pack_id, display_name, collection_value, data_root_value)
        ):
            continue
        if pack_id != pack_root.name or pack_id in packs:
            continue
        collection_path = (pack_root / collection_value).resolve()
        pack_data_root = (pack_root / data_root_value).resolve()
        if (
            not collection_path.is_relative_to(pack_root)
            or not pack_data_root.is_relative_to(pack_root)
            or not collection_path.is_file()
            or not pack_data_root.is_dir()
        ):
            continue
        packs[pack_id] = {
            **manifest,
            "manifest_path": manifest_path.resolve(),
            "pack_root": pack_root,
            "collection_path": collection_path,
            "data_root": pack_data_root,
        }
    return packs


def _reset_for_question_pack(pack_id: str) -> None:
    """Clear per-question state when the URL-selected pack changes."""
    if st.session_state.active_question_pack == pack_id:
        return
    st.session_state.active_question_pack = pack_id
    st.session_state.bundle = None
    st.session_state.question_path = None
    st.session_state.question_pool = []
    st.session_state.review_collection_path = None
    _reset_score()
    _reset_quiz_state()


def _render_question_pack_selector(
    packs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Render a URL-addressable pack selector and return the active pack."""
    options = [LOCAL_QUESTION_PACK, *packs]

    def format_pack(pack_id: str) -> str:
        if pack_id == LOCAL_QUESTION_PACK:
            return "Local data root"
        return str(packs[pack_id]["display_name"])

    query_value = st.query_params.get("question_pack", LOCAL_QUESTION_PACK)
    query_selected = (
        query_value if query_value in options else LOCAL_QUESTION_PACK
    )
    widget_key = "question_pack_widget"
    widget_selected = st.session_state.get(widget_key)
    active = st.session_state.active_question_pack
    if widget_selected not in options:
        st.session_state[widget_key] = query_selected
    elif query_selected != active and query_selected != widget_selected:
        # The address bar changed outside the widget; make the URL authoritative.
        st.session_state[widget_key] = query_selected

    selected = st.selectbox(
        "Question pack",
        options,
        format_func=format_pack,
        key=widget_key,
    )
    if selected == LOCAL_QUESTION_PACK:
        if "question_pack" in st.query_params:
            del st.query_params["question_pack"]
    elif st.query_params.get("question_pack") != selected:
        st.query_params["question_pack"] = selected

    _reset_for_question_pack(selected)
    if selected == LOCAL_QUESTION_PACK:
        explicit_root = _argument_data_root(sys.argv)
        if (
            explicit_root is None
            and st.session_state.data_root.startswith(str(QUESTION_PACKS_ROOT.resolve()))
        ):
            st.session_state.data_root = "data"
        return None

    pack = packs[selected]
    st.session_state.data_root = str(pack["data_root"])
    return pack


def _collection_manifest_path(
    collection_path: Path | str | None = None,
) -> Path | None:
    if collection_path is not None:
        candidate = Path(collection_path).resolve()
        if candidate.suffix.lower() == ".json" and candidate.is_file():
            return candidate
        return None
    if len(sys.argv) <= 1:
        return None

    candidate = Path(sys.argv[1]).resolve()
    if candidate.suffix.lower() == ".json" and candidate.is_file():
        return candidate
    return None


def _startup_question_collection(
    data_root: str,
    collection_path: Path | str | None = None,
) -> list[Path] | None:
    """Load a review collection from a tracked pack or Streamlit CLI argument."""
    manifest_path = _collection_manifest_path(collection_path)
    if manifest_path is None:
        return None

    try:
        manifest = read_json_file(manifest_path)
    except (OSError, ValueError):
        return []
    values = manifest.get("question_paths")
    if not isinstance(values, list):
        return []

    root = _resolve_data_root(data_root)
    questions: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if (
            resolved in seen
            or not resolved.is_relative_to(root)
            or not (resolved / "question.json").is_file()
        ):
            continue
        seen.add(resolved)
        questions.append(resolved)
    return questions


def _discover_questions(
    data_root: str,
    collection_path: Path | str | None = None,
) -> list[Path]:
    collection = _startup_question_collection(data_root, collection_path)
    if collection is not None:
        return collection
    return [Path(path) for path in _cached_question_dirs(str(_resolve_data_root(data_root)))]


@st.cache_data(ttl=10, show_spinner=False)
def _cached_question_dirs(data_root: str) -> tuple[str, ...]:
    """Avoid recursively scanning every question twice on each UI rerun."""
    return tuple(str(path) for path in list_question_dirs(Path(data_root)))


def _default_question_path(
    pool: list[Path],
    data_root: str,
    collection_path: Path | str | None = None,
) -> Path | None:
    if not pool:
        return None
    if _startup_question_collection(data_root, collection_path) is not None:
        return pool[0]
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1]).resolve()
        if arg.is_file():
            arg = arg.parent
        for path in pool:
            if path.resolve() == arg:
                return path
    return pool[0]


def _load_selected_question(
    question_path: Path, data_root: str
) -> QuestionBundle | None:
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


def _completed_release_questions(
    manifest: release_manifest.QuizManifest,
) -> set[surprise_recommender.QuestionIdentity]:
    completed: set[surprise_recommender.QuestionIdentity] = set()
    answers = st.session_state.get("quiz_answers", {})
    if not isinstance(answers, Mapping):
        return completed
    for question_version, raw_answer in answers.items():
        if not isinstance(question_version, str) or not isinstance(raw_answer, Mapping):
            continue
        question_id = raw_answer.get("question_id")
        if not isinstance(question_id, str):
            continue
        if manifest.release_id_for(question_id, question_version) is None:
            continue
        completed.add(
            surprise_recommender.QuestionIdentity(
                manifest.release_id,
                question_id,
                question_version,
            )
        )
    return completed


def _release_exposure_counts(
    manifest: release_manifest.QuizManifest,
) -> dict[surprise_recommender.QuestionIdentity, int]:
    counts: dict[surprise_recommender.QuestionIdentity, int] = {}
    attempt_id = st.session_state.quiz_attempt_id
    for event in _feedback_trace().events:
        if event.get("event_type") != "question_presented":
            continue
        payload = event.get("payload")
        question_id = event.get("question_id")
        question_version = event.get("question_version")
        if (
            not isinstance(payload, Mapping)
            or payload.get("attempt_id") != attempt_id
            or payload.get("release_id") != manifest.release_id
            or not isinstance(question_id, str)
            or not isinstance(question_version, str)
            or manifest.release_id_for(question_id, question_version) is None
        ):
            continue
        identity = surprise_recommender.QuestionIdentity(
            manifest.release_id,
            question_id,
            question_version,
        )
        counts[identity] = counts.get(identity, 0) + 1
    return counts


def _recommended_next_path(
    manifest: release_manifest.QuizManifest,
    current_path: Path,
    *,
    seed: int,
) -> tuple[Path, surprise_recommender.Recommendation]:
    """Choose one answer-safe next question from the attested release only."""
    candidates = surprise_catalog.build_recommendation_candidates(
        manifest,
        exposure_counts=_release_exposure_counts(manifest),
    )
    completed = _completed_release_questions(manifest)
    current_resolved = current_path.resolve()
    last_family: str | None = None
    for record, path in zip(
        manifest.questions,
        manifest.question_dirs(),
        strict=True,
    ):
        if path.resolve() != current_resolved:
            continue
        last_family = record.family
        completed.add(
            surprise_recommender.QuestionIdentity(
                manifest.release_id,
                record.question_id,
                record.version,
            )
        )
        break

    recommendation = surprise_recommender.select_question(
        candidates,
        completed,
        epsilon=surprise_recommender.DEFAULT_EPSILON,
        seed=seed,
        last_family=last_family,
    )
    for record, path in zip(
        manifest.questions,
        manifest.question_dirs(),
        strict=True,
    ):
        if (
            record.question_id == recommendation.question.question_id
            and record.version == recommendation.question.question_version
        ):
            return path, recommendation
    raise surprise_catalog.SurpriseCatalogError(
        "recommendation returned an identity outside the attested manifest"
    )


def _render_question_picker(
    data_root: str,
    collection_path: Path | str | None = None,
) -> None:
    collection = _startup_question_collection(data_root, collection_path)
    collection_mode = collection is not None
    if collection_mode:
        pool = collection
    else:
        _, pool = _load_question_pool(data_root)
    st.session_state.question_pool = [str(p) for p in pool]

    manifest_error = st.session_state.quiz_manifest_error
    if manifest_error:
        st.error(
            "Release attestation failed. No questions from this data root will "
            f"be served: {manifest_error}"
        )
        st.session_state.bundle = None
        st.session_state.question_path = None
        return

    if not pool:
        if collection_mode:
            st.warning("This review collection contains no valid questions.")
        else:
            st.warning("No questions found under the data root.")
        st.session_state.bundle = None
        return

    if collection_mode:
        manifest_path = _collection_manifest_path(collection_path)
        assert manifest_path is not None
        collection_identity = str(manifest_path)
        if st.session_state.review_collection_path != collection_identity:
            st.session_state.review_collection_path = collection_identity
            _reset_score()
    else:
        st.session_state.review_collection_path = None

    current = st.session_state.question_path
    if current is None or not _pool_contains(pool, Path(current)):
        default = _default_question_path(pool, data_root, collection_path)
        if default is not None:
            _switch_question(
                default,
                data_root,
                presentation=_presentation_decision(
                    policy_version=SEQUENTIAL_POLICY_VERSION,
                    mode="fallback",
                    propensity=1.0,
                    source="initial",
                ),
            )

    current_path = Path(st.session_state.question_path)
    try:
        current_index = pool.index(current_path.resolve())
    except ValueError:
        current_index = 0
        st.session_state.question_path = str(pool[0])

    if collection_mode:
        picker_key = (
            "review_question_picker_"
            + hashlib.sha256(collection_identity.encode("utf-8")).hexdigest()[:12]
        )
        if current_index < len(pool) - 1:
            if st.button(
                f"Next question ({current_index + 2}/{len(pool)})",
                use_container_width=True,
                disabled=st.session_state.committed_letter is None,
            ):
                st.session_state[picker_key] = current_index + 1
                _switch_question(pool[current_index + 1], data_root)
                st.rerun()
        elif st.session_state.committed_letter is not None:
            correct, total = _score_stats()
            st.success(f"Review sequence complete · score {correct} / {total}")
            if st.button("Restart sequence", use_container_width=True):
                st.session_state[picker_key] = 0
                _reset_score()
                _switch_question(pool[0], data_root)
                st.rerun()
        else:
            st.info("Submit this final answer to complete the review sequence.")

        picked_index = st.selectbox(
            "Review question",
            options=list(range(len(pool))),
            index=current_index,
            format_func=lambda index: (
                f"{index + 1}/{len(pool)} · {question_label(pool[index])}"
            ),
            key=picker_key,
        )
        picked_path = pool[picked_index]
        if picked_index != current_index:
            _switch_question(picked_path, data_root)
            st.rerun()
    else:
        manifest = _load_active_manifest(data_root)
        nav_next, nav_random = st.columns(2)
        with nav_next:
            if st.button("Next", use_container_width=True):
                nxt = pool[(current_index + 1) % len(pool)]
                presentation = _presentation_decision(
                    policy_version=SEQUENTIAL_POLICY_VERSION,
                    mode="fallback",
                    propensity=1.0,
                    source="next",
                )
                if manifest is not None:
                    try:
                        nxt, recommendation = _recommended_next_path(
                            manifest,
                            current_path,
                            seed=secrets.randbits(63),
                        )
                        presentation = _presentation_decision(
                            policy_version=SURPRISE_POLICY_VERSION,
                            mode=recommendation.mode,
                            propensity=recommendation.propensity,
                            source="next",
                        )
                    except (
                        surprise_catalog.SurpriseCatalogError,
                        surprise_recommender.SurpriseRecommendationError,
                        OSError,
                    ):
                        # Serving remains available even when the private catalog
                        # or policy cannot be evaluated. The existing sequential
                        # Next behavior is the explicit fail-safe.
                        pass
                _switch_question(nxt, data_root, presentation=presentation)
                st.rerun()
        with nav_random:
            if st.button("Random", use_container_width=True):
                choices = [p for p in pool if p.resolve() != current_path.resolve()]
                pick = random.choice(choices or pool)
                _switch_question(pick, data_root)
                st.rerun()

        labels = [question_label(p) for p in pool]
        label_to_path = dict(zip(labels, pool, strict=True))
        picked_label = st.selectbox(
            "Question",
            labels,
            index=current_index,
        )
        picked_path = label_to_path[picked_label]
        if picked_path.resolve() != current_path.resolve():
            _switch_question(picked_path, data_root)
            st.rerun()

    loaded_bundle = st.session_state.bundle
    if (
        not isinstance(loaded_bundle, QuestionBundle)
        or loaded_bundle.question_root.resolve() != picked_path.resolve()
        or loaded_bundle.data_root.resolve() != _resolve_data_root(data_root)
    ):
        _switch_question(picked_path, data_root)
        loaded_bundle = st.session_state.bundle

    if collection_mode:
        st.caption(f"Question {current_index + 1} / {len(pool)} · `{picked_path.name}`")
    elif manifest is None:
        st.caption(f"{len(pool)} unversioned question(s) · `{picked_path.name}`")
        st.caption(
            f"Runtime Git SHA: `{_runtime_git_sha() or 'N/A'}` · "
            f"Entry: `{INSPECTOR_ENTRY_PATH}`"
        )
    else:
        unpublished_count = len(pool) - manifest.question_count
        st.caption(
            f"Attested release `{manifest.release_id}` · "
            f"{manifest.question_count} published · {len(pool)} available"
        )
        st.caption(
            f"Manifest SHA-256 `{manifest.manifest_sha256}` · "
            f"{manifest.artifact_count} verified artifacts"
        )
        st.caption(
            f"Runtime Git SHA: `{_runtime_git_sha() or 'N/A'}` · "
            f"Entry: `{INSPECTOR_ENTRY_PATH}`"
        )
        if unpublished_count:
            st.caption(
                f"{unpublished_count} additional local question(s) are not part "
                "of this release."
            )


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


def _plot_bigram_lm(
    dataset_dir: Path, train_x: torch.Tensor, train_y: torch.Tensor
) -> None:
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


def _valid_classification_feature_pair(
    first_feature: int,
    second_feature: int,
    input_dim: int,
) -> tuple[int, int]:
    if input_dim < 2:
        raise ValueError("A 2-D classification projection needs at least two features.")
    if (
        first_feature != second_feature
        and 0 <= first_feature < input_dim
        and 0 <= second_feature < input_dim
    ):
        return first_feature, second_feature
    first = 0 if first_feature < 0 or first_feature >= input_dim else first_feature
    second = next(feature for feature in range(input_dim) if feature != first)
    return first, second


def _select_rule_aware_classification_pair(
    params: dict[str, Any],
    input_dim: int,
) -> tuple[int, int, str]:
    """Choose a semantically informative pair from the frozen rule specification."""
    active = [
        int(feature)
        for feature in params.get("active_features", [])
        if isinstance(feature, int) and 0 <= int(feature) < input_dim
    ]
    rule_family = str(params.get("rule_family", ""))

    if rule_family == "sparse_interaction":
        pairs = params.get("interaction_pairs", [])
        weights = params.get("rule_weights", [])
        weighted_pairs: list[tuple[float, int, int]] = []
        for index, pair in enumerate(pairs):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(feature, int) for feature in pair)
            ):
                continue
            weight = float(weights[index]) if index < len(weights) else 0.0
            weighted_pairs.append((abs(weight), int(pair[0]), int(pair[1])))
        if weighted_pairs:
            _, first, second = max(weighted_pairs, key=lambda item: item[0])
            first, second = _valid_classification_feature_pair(first, second, input_dim)
            return first, second, "largest-magnitude interaction"
    if rule_family == "piecewise_boundary" and len(active) >= 2:
        first, second = _valid_classification_feature_pair(active[0], active[1], input_dim)
        return first, second, "piecewise-boundary coordinates"

    if rule_family == "smooth_additive" and active:
        weights = params.get("rule_weights", [])
        ranked = sorted(
            (
                (
                    abs(float(weights[index])) if index < len(weights) else 0.0,
                    feature,
                )
                for index, feature in enumerate(active)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        first = ranked[0][1]
        second = next((feature for _, feature in ranked if feature != first), -1)
        first, second = _valid_classification_feature_pair(first, second, input_dim)
        return first, second, "largest-magnitude additive effects"
    first = active[0] if active else 0
    second = active[1] if len(active) >= 2 else -1
    first, second = _valid_classification_feature_pair(first, second, input_dim)
    return first, second, "available feature coordinates"


def _classification_probability_grid(
    x: np.ndarray,
    y: np.ndarray,
    first_feature: int,
    second_feature: int,
    *,
    bins: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate empirical P(y=1) on a feature-pair grid; empty cells are NaN."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = np.asarray(x)
    labels = np.asarray(y).reshape(-1)
    if values.ndim != 2 or values.shape[0] != labels.shape[0]:
        raise ValueError("x must be [N, D] and y must have N entries")
    first_feature, second_feature = _valid_classification_feature_pair(
        first_feature, second_feature, values.shape[1]
    )
    first_values = values[:, first_feature]
    second_values = values[:, second_feature]
    finite = np.isfinite(first_values) & np.isfinite(second_values) & np.isfinite(labels)
    if not np.any(finite):
        raise ValueError("No finite points are available for the classification projection")
    first_values = first_values[finite]
    second_values = second_values[finite]
    labels = labels[finite]

    def edges(values: np.ndarray) -> np.ndarray:
        low = float(np.min(values))
        high = float(np.max(values))
        if low == high:
            low -= 0.5
            high += 0.5
        margin = 0.05 * (high - low)
        return np.linspace(low - margin, high + margin, bins + 1)

    first_edges = edges(first_values)
    second_edges = edges(second_values)
    counts, _, _ = np.histogram2d(
        first_values, second_values, bins=(first_edges, second_edges)
    )
    positive, _, _ = np.histogram2d(
        first_values[labels == 1],
        second_values[labels == 1],
        bins=(first_edges, second_edges),
    )
    probability = np.full(counts.shape, np.nan, dtype=float)
    np.divide(positive, counts, out=probability, where=counts > 0)
    return first_edges, second_edges, probability, counts


def _select_observed_classification_pair(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 12,
) -> tuple[int, int, float]:
    """Select the pair with the strongest count-weighted empirical class contrast."""
    values = np.asarray(x)
    labels = np.asarray(y).reshape(-1)
    if values.ndim != 2 or values.shape[0] != labels.shape[0] or values.shape[1] < 2:
        raise ValueError("Observed feature-pair selection needs x=[N, D], y=[N], D>=2")
    baseline = float(np.mean(labels == 1))
    best_pair = (0, 1)
    best_score = float("-inf")
    for first_feature in range(values.shape[1]):
        for second_feature in range(first_feature + 1, values.shape[1]):
            _, _, probability, counts = _classification_probability_grid(
                values, labels, first_feature, second_feature, bins=bins
            )
            occupied = counts > 0
            score = float(
                np.sum(counts[occupied] * (probability[occupied] - baseline) ** 2)
                / np.sum(counts[occupied])
            )
            if score > best_score:
                best_pair = (first_feature, second_feature)
                best_score = score
    return best_pair[0], best_pair[1], best_score


def _sample_classification_indices(
    labels: np.ndarray,
    label: int,
    *,
    maximum: int,
) -> np.ndarray:
    indices = np.flatnonzero(labels == label)
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def _plot_synthetic_tabular_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    params: dict[str, Any] | None = None,
    feature_pair: tuple[int, int] | None = None,
    selection_note: str = "available feature coordinates",
) -> None:
    """Plot empirical class probability and a sampled, low-overlap point overlay."""
    x_train = train_x.detach().cpu().numpy()
    x_test = test_x.detach().cpu().numpy()
    y_train = train_y.detach().cpu().reshape(-1).numpy()
    y_test = test_y.detach().cpu().reshape(-1).numpy()
    if (
        x_train.ndim != 2
        or x_test.ndim != 2
        or x_train.shape[1] < 2
        or x_test.shape[1] < 2
    ):
        st.info(
            "Classification dataset has fewer than two tabular features; "
            "no 2-D projection is available."
        )
        return

    if feature_pair is None:
        first_feature, second_feature, selection_note = (
            _select_rule_aware_classification_pair(params or {}, x_train.shape[1])
        )
    else:
        first_feature, second_feature = _valid_classification_feature_pair(
            feature_pair[0], feature_pair[1], x_train.shape[1]
        )
    first_edges, second_edges, probability, _ = _classification_probability_grid(
        x_train, y_train, first_feature, second_feature
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    image = ax.pcolormesh(
        first_edges,
        second_edges,
        probability.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.82,
    )
    finite_probability = probability[np.isfinite(probability)]
    if (
        finite_probability.size
        and float(np.min(finite_probability)) < 0.5
        and float(np.max(finite_probability)) > 0.5
    ):
        first_centers = (first_edges[:-1] + first_edges[1:]) / 2
        second_centers = (second_edges[:-1] + second_edges[1:]) / 2
        ax.contour(
            first_centers,
            second_centers,
            probability.T,
            levels=[0.5],
            colors="#0f172a",
            linewidths=1.0,
        )

    colors = ("#1d4ed8", "#b91c1c")
    for label, color in enumerate(colors):
        train_indices = _sample_classification_indices(y_train, label, maximum=220)
        test_indices = _sample_classification_indices(y_test, label, maximum=90)
        ax.scatter(
            x_train[train_indices, first_feature],
            x_train[train_indices, second_feature],
            s=10,
            alpha=0.32,
            c=color,
            label=f"train · class {label}",
            edgecolors="none",
        )
        ax.scatter(
            x_test[test_indices, first_feature],
            x_test[test_indices, second_feature],
            s=20,
            alpha=0.72,
            c=color,
            marker="x",
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("empirical P(class 1) in train bins")
    ax.set_xlabel(f"feature {first_feature}")
    ax.set_ylabel(f"feature {second_feature}")
    ax.set_title(f"Classification projection · {selection_note}")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_synthetic_tabular_classification_plot(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    params: dict[str, Any],
    dataset_id: str,
) -> None:
    if train_x.ndim != 2 or train_x.shape[1] < 2:
        _plot_synthetic_tabular_classification(train_x, train_y, test_x, test_y)
        return

    input_dim = int(train_x.shape[1])
    default_first, default_second, default_note = (
        _select_rule_aware_classification_pair(params, input_dim)
    )
    mode = st.radio(
        "Projection",
        ("Decision-relevant", "Observed labels", "Manual"),
        horizontal=True,
        key=f"classification_projection_mode_{dataset_id}",
    )
    if mode == "Decision-relevant":
        first_feature, second_feature, note = (
            default_first,
            default_second,
            f"rule-aware: {default_note}",
        )
    elif mode == "Observed labels":
        first_feature, second_feature, score = _select_observed_classification_pair(
            train_x.detach().cpu().numpy(),
            train_y.detach().cpu().reshape(-1).numpy(),
        )
        note = f"label-driven pair (contrast {score:.3f})"
    else:
        feature_options = list(range(input_dim))
        first_feature = st.selectbox(
            "Horizontal feature",
            feature_options,
            index=feature_options.index(default_first),
            format_func=lambda feature: f"feature {feature}",
            key=f"classification_projection_x_{dataset_id}",
        )
        second_options = [
            feature for feature in feature_options if feature != first_feature
        ]
        second_default = (
            second_options.index(default_second)
            if default_second in second_options
            else 0
        )
        second_feature = st.selectbox(
            "Vertical feature",
            second_options,
            index=second_default,
            format_func=lambda feature: f"feature {feature}",
            key=f"classification_projection_y_{dataset_id}",
        )
        note = "manual feature pair"

    st.caption(
        "Background: empirical train-set class rate per bin. "
        "Filled points: sampled train data; crosses: sampled test data."
    )
    _plot_synthetic_tabular_classification(
        train_x,
        train_y,
        test_x,
        test_y,
        params=params,
        feature_pair=(first_feature, second_feature),
        selection_note=note,
    )

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
    elif family == "synthetic_tabular_classification":
        _render_synthetic_tabular_classification_plot(
            train_x,
            train_y,
            test_x,
            test_y,
            params=params,
            dataset_id=bundle.dataset_dir.name,
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
    diagnostics: list[str] | None = None,
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
        if diagnostics is not None:
            diagnostics.append(f"{label}: {loaded['error']}")
        return None
    if loaded.get("warning") and diagnostics is not None:
        diagnostics.append(f"{label}: {loaded['warning']}")

    curves = loaded["curves"]
    x = np.asarray(loaded["eval_samples"], dtype=np.int64)
    if curves.size == 0 or not np.isfinite(curves).any():
        if diagnostics is not None:
            diagnostics.append(f"{label}: curves.npz contains no finite values")
        return None

    finite = np.isfinite(curves)
    valid = finite.any(axis=0)
    mean = np.full(curves.shape[1], np.nan, dtype=np.float64)
    std = np.full(curves.shape[1], np.nan, dtype=np.float64)
    if valid.any():
        mean[valid] = np.nanmean(curves[:, valid], axis=0)
        std[valid] = np.nanstd(curves[:, valid], axis=0)
    positive_curves = np.where(finite & (curves > 0), curves, np.nan)
    positive = np.isfinite(positive_curves).any(axis=0)
    log_q10 = np.full(curves.shape[1], np.nan, dtype=np.float64)
    log_median = np.full(curves.shape[1], np.nan, dtype=np.float64)
    log_q90 = np.full(curves.shape[1], np.nan, dtype=np.float64)
    if positive.any():
        quantiles = np.nanquantile(
            positive_curves[:, positive],
            (0.10, 0.50, 0.90),
            axis=0,
        )
        log_q10[positive], log_median[positive], log_q90[positive] = quantiles
    if not valid.any():
        if diagnostics is not None:
            diagnostics.append(f"{label}: curves.npz contains no finite columns")
        return None
    return {
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "x": x[valid],
        "mean": mean[valid],
        "std": std[valid],
        "log_q10": log_q10[valid],
        "log_median": log_median[valid],
        "log_q90": log_q90[valid],
    }

def _collect_curve_series(
    bundle: QuestionBundle,
    diagnostics: list[str] | None = None,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, choice in enumerate(bundle.choices):
        item = _curve_series_from_candidate(
            choice["candidate_dir"],
            label=f"{choice['letter']} · {choice['candidate_id']}",
            color=_choice_color(index),
            diagnostics=diagnostics,
        )
        if item is not None:
            series.append(item)
    return series


def _collect_custom_curve_series(
    bundle: QuestionBundle,
    diagnostics: list[str] | None = None,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, setting in enumerate(
        list_custom_setting_runs(_custom_settings_storage_for(bundle.question))
    ):
        item = _curve_series_from_candidate(
            setting["candidate_dir"],
            label=f"Custom · {setting['label']}",
            color=_setting_color(index),
            linestyle="--",
            diagnostics=diagnostics,
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
    diagnostics: list[str] = []
    custom_series = _collect_custom_curve_series(bundle, diagnostics)
    series = (_collect_curve_series(bundle, diagnostics) if include_candidates else []) + custom_series
    if diagnostics:
        message = "Curve diagnostics: " + " | ".join(diagnostics)
        if series:
            st.caption(message)
        else:
            st.warning(message)
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
        if use_log_y:
            lower = item["log_q10"][in_window]
            upper = item["log_q90"][in_window]
            line = item["log_median"][in_window]
            positive = (
                np.isfinite(lower)
                & np.isfinite(upper)
                & np.isfinite(line)
                & (lower > 0)
                & (upper > 0)
                & (line > 0)
            )
            if not positive.any():
                continue
            x_valid = x_valid[positive]
            lower = lower[positive]
            upper = upper[positive]
            line = line[positive]
        else:
            lower = mean_valid - std_valid
            upper = mean_valid + std_valid
            line = mean_valid
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
    uncertainty_label = (
        "median with 10–90% quantile band"
        if use_log_y
        else "mean ± std across seeds"
    )
    title = f"Learning curves ({uncertainty_label})"
    if custom_series and not include_candidates:
        title = f"Custom setting curves ({uncertainty_label})"
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
    choice = st.radio(
        "File", names, index=idx, horizontal=True, key=f"{key_prefix}_file_radio"
    )
    st.session_state[f"{key_prefix}_file"] = choice
    path = paths[choice]

    if choice.endswith(".json"):
        st.json(read_json_file(path))
    else:
        st.code(
            read_text_file(path), language="python" if choice.endswith(".py") else None
        )


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
        _spec_block(
            "Model",
            _format_model_lines(spec.get("model", {}))
            + [
                "Trainable parameters: "
                + (
                    f"{int(spec['trainable_parameter_count']):,}"
                    if spec.get("trainable_parameter_count") is not None
                    else "unavailable"
                )
            ],
        ),
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
        spec = read_json_file(
            bundle.choices[0]["candidate_dir"] / "candidate_spec.json"
        )
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

    st.caption(
        "Choose the activation and layer norm independently for each hidden block."
    )
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


def _render_transformer_setting_fields(
    profile: Any, q: dict[str, Any]
) -> dict[str, Any]:
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

def _render_gru_setting_fields(profile: Any, q: dict[str, Any]) -> dict[str, Any]:
    st.markdown("**Architecture parameters**")
    d_model_col, layers_col = st.columns(2)
    with d_model_col:
        d_model = int(
            st.number_input(
                "Model width",
                min_value=1,
                max_value=1024,
                step=8,
                key=_ensure_setting_value(
                    q,
                    "gru_d_model",
                    int(profile.gru_lm["d_model"][0]),
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
                    "gru_layers",
                    int(profile.gru_lm["num_layers"][0]),
                ),
            )
        )
    residual_default = bool(profile.gru_lm.get("layer_residual", False))
    layer_residual = st.checkbox(
        "Layer residual connections",
        key=_ensure_setting_value(q, "gru_layer_residual", residual_default),
        help=(
            "When enabled, each GRU layer adds its output to its input: "
            "h = h + GRU_layer(h)."
        ),
    )
    if layer_residual:
        st.caption("Enabled: after each GRU layer, h = h + GRU_layer(h).")
    else:
        st.caption("Disabled (legacy stacked GRU behavior).")
    return {
        "d_model": d_model,
        "num_layers": num_layers,
        "layer_residual": bool(layer_residual),
    }


def _kan_defaults(profile: Any) -> dict[str, Any]:
    """Resolve editable KAN defaults from the active profile, not a fixed pool."""
    config = profile.kan

    def pick(name: str, fallback: Any) -> Any:
        values = config.get(name)
        if not isinstance(values, list) or not values:
            return fallback
        return values[min(1, len(values) - 1)]

    grid_range = pick("grid_range", [-1.0, 1.0])
    if not isinstance(grid_range, list) or len(grid_range) != 2:
        grid_range = [-1.0, 1.0]
    # ``base_activation`` describes the legacy sampled pool. v2.2's broader
    # KAN pool is recorded as explicit, auditable archetypes, so include its
    # activations as editable choices too. Otherwise a valid inherited KAN
    # candidate such as ``relu`` could not be represented by the UI.
    activations = [str(value) for value in config.get("base_activation") or []]
    archetypes = config.get("archetypes", {})
    if isinstance(archetypes, dict):
        for family_archetypes in archetypes.values():
            if not isinstance(family_archetypes, list):
                continue
            for archetype in family_archetypes:
                if isinstance(archetype, dict) and archetype.get("base_activation"):
                    activations.append(str(archetype["base_activation"]))
    activations = list(dict.fromkeys(activations)) or ["silu"]
    return {
        "variant": str(config.get("variant", "efficient_spline_v1")),
        "depth": int(pick("depth", 1)), "width": int(pick("width", 8)),
        "grid_size": int(pick("grid_size", 5)), "spline_order": int(pick("spline_order", 3)),
        "grid_low": float(grid_range[0]), "grid_high": float(grid_range[1]),
        "base_activations": activations,
    }


def _kan_activation_options(options: list[str], current: str) -> list[str]:
    """Keep a valid inherited activation editable under a narrower profile."""
    result = list(options)
    if current in BASE_ACTIVATIONS and current not in result:
        result.append(current)
    return result


def _render_kan_setting_fields(profile: Any, q: dict[str, Any]) -> dict[str, Any]:
    """Render all KAN fields supported by ``build_model_spec``."""
    defaults = _kan_defaults(profile)
    st.markdown("**Architecture parameters**")
    variant = st.text_input("KAN variant", key=_ensure_setting_value(q, "kan_variant", defaults["variant"]))
    columns = st.columns(4)
    values: dict[str, int] = {}
    for column, label, name, maximum in zip(
        columns, ("Depth", "Width", "Grid size", "Spline order"),
        ("depth", "width", "grid_size", "spline_order"), (12, 2048, 64, 16), strict=True,
    ):
        with column:
            values[name] = int(st.number_input(label, min_value=1, max_value=maximum, step=1,
                key=_ensure_setting_value(q, f"kan_{name}", defaults[name])))
    low_col, high_col, activation_col = st.columns(3)
    with low_col:
        grid_low = float(st.number_input("Grid lower bound", step=0.1, format="%.6g",
            key=_ensure_setting_value(q, "kan_grid_low", defaults["grid_low"])))
    with high_col:
        grid_high = float(st.number_input("Grid upper bound", step=0.1, format="%.6g",
            key=_ensure_setting_value(q, "kan_grid_high", defaults["grid_high"])))
    with activation_col:
        activation_key = _ensure_setting_value(
            q, "kan_base_activation", defaults["base_activations"][0]
        )
        current_activation = str(st.session_state[activation_key])
        activation_options = _kan_activation_options(
            defaults["base_activations"], current_activation
        )
        if current_activation not in activation_options:
            st.session_state[activation_key] = activation_options[0]
        base_activation = st.selectbox(
            "Base activation", activation_options, key=activation_key
        )
    return {"variant": variant, **values, "grid_range": [grid_low, grid_high], "base_activation": base_activation}





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
    choice = next(
        choice for choice in bundle.choices if choice["letter"] == source_letter
    )
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


def _format_elapsed(seconds: float) -> str:
    whole_seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _custom_setting_progress_callback() -> Callable[[dict[str, Any]], None]:
    """Create in-place Streamlit widgets for one synchronous custom-setting run."""
    started_at = time.monotonic()
    progress_bar = st.progress(0)
    status = st.empty()
    chart = st.empty()
    histories: dict[int, tuple[list[int], list[float]]] = {}
    completed_seeds: set[int] = set()

    def render_chart(metric: str, n_seeds: int) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        for seed_index, (samples, values) in sorted(histories.items()):
            if seed_index in completed_seeds and samples:
                ax.plot(samples, values, linewidth=1.5, label=f"seed {seed_index}")
        ax.set_xlabel("Samples seen")
        ax.set_ylabel(_metric_display_name(metric))
        ax.set_title(
            "Custom-setting learning curves "
            f"(completed seeds: {len(completed_seeds)} / {n_seeds})"
        )
        ax.grid(True, alpha=0.25)
        if len(completed_seeds) <= 8:
            ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        chart.pyplot(fig, clear_figure=True)
        plt.close(fig)

    def callback(event: dict[str, Any]) -> None:
        phase = str(event.get("phase", ""))
        seed_index = int(event.get("seed_index", 1))
        n_seeds = max(1, int(event.get("n_seeds", 1)))
        training_steps = max(1, int(event.get("training_steps", 1)))
        step = min(training_steps, max(0, int(event.get("step", 0))))
        fraction = ((seed_index - 1) + step / training_steps) / n_seeds
        if phase == "seed_finished":
            fraction = seed_index / n_seeds
        fraction = min(1.0, max(0.0, fraction))
        progress_bar.progress(int(round(100 * fraction)))

        elapsed = time.monotonic() - started_at
        eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        eta_text = f" · ETA {_format_elapsed(eta)}" if eta is not None else ""
        metric = str(event.get("selection_metric", "metric"))

        if phase == "seed_started":
            status.caption(
                f"Training seed {seed_index} / {n_seeds} · "
                f"{training_steps} optimizer steps · elapsed {_format_elapsed(elapsed)}"
            )
            return

        if phase == "evaluation":
            value = float(event["metric"])
            if np.isfinite(value):
                samples, values = histories.setdefault(seed_index, ([], []))
                samples.append(int(event["samples_seen"]))
                values.append(value)
            status.caption(
                f"Seed {seed_index} / {n_seeds} · step {step} / {training_steps} · "
                f"samples {int(event.get('samples_seen', 0))} / "
                f"{int(event.get('total_samples_seen', 0))} · "
                f"latest {_metric_display_name(metric)} {value:.6g} · "
                f"elapsed {_format_elapsed(elapsed)}{eta_text}"
            )
            return

        if phase == "seed_finished":
            completed_seeds.add(seed_index)
            render_chart(metric, n_seeds)
            status.caption(
                f"Finished seed {seed_index} / {n_seeds} · "
                f"updated completed-seed curves · "
                f"elapsed {_format_elapsed(elapsed)}{eta_text}"
            )

    return callback

def _render_custom_setting_builder(bundle: QuestionBundle, q: dict[str, Any]) -> None:
    profile = load_profile(str(q.get("profile", "v1")))
    dataset_spec = read_json_file(bundle.dataset_dir / "dataset_spec.json")
    family = str(dataset_spec["family"])
    runs = enforce_custom_setting_retention(_custom_settings_storage_for(q))

    notice = st.session_state.setting_notice
    if notice:
        st.success(notice)
        st.session_state.setting_notice = None

    with st.expander("＋ Add custom setting", expanded=False):
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
        elif model_type == "kan":
            model_params = _render_kan_setting_fields(profile, q)
        elif model_type == "transformer_lm":
            model_params = _render_transformer_setting_fields(profile, q)
        elif model_type == "gru_lm":
            model_params = _render_gru_setting_fields(profile, q)
        else:
            st.error(f"Unsupported architecture in this profile: {model_type}")
            return

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
            proposal_event: dict[str, Any] | None = None
            inherited_candidate_id = st.session_state.get(
                _setting_key(q, "inherited_candidate_id")
            )
            inherited_letter = st.session_state.get(_setting_key(q, "inherited_letter"))
            raw_inherited_from = None
            if inherited_candidate_id and inherited_letter:
                raw_inherited_from = {
                    "letter": inherited_letter,
                    "candidate_id": inherited_candidate_id,
                }
            raw_setting = {
                "label": label.strip() or "Setting",
                "budget": {
                    "total_samples_seen": budget,
                    "batch_size": batch_size,
                },
                "model": {"type": model_type, **model_params},
                "optimizer": optimizer_params,
                "loss": {"loss_id": loss_id, "lambda": lambda_value},
                "evaluation": {"n_seeds": n_seeds, "base_seed": base_seed},
                "inherited_from": raw_inherited_from,
            }
            try:
                model = build_model_spec(
                    model_type, model_params, dataset_spec.get("params", {})
                )
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
                inherited_from = None
                if inherited_candidate_id and inherited_letter:
                    inherited_from = {
                        "letter": inherited_letter,
                        "candidate_id": inherited_candidate_id,
                        "exact_spec_match": spec["candidate_id"]
                        == inherited_candidate_id,
                    }
                proposal_event = _feedback_trace().record_custom_setting(
                    q,
                    setting=spec,
                    extra={
                        **_event_context(q),
                        "label": label.strip() or "Setting",
                        "n_seeds": n_seeds,
                        "base_seed": base_seed,
                        "inherited_from": inherited_from,
                    },
                )
                _touch_browser_outbox()
                progress_callback = _custom_setting_progress_callback()
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
                        progress_callback=progress_callback,
                    )

            except Exception as exc:
                if proposal_event is not None:
                    _feedback_trace().record_custom_run(
                        q,
                        run={
                            "proposal_event_id": proposal_event["event_id"],
                            "status": "failed",
                            "candidate_id": spec["candidate_id"],
                            "error_type": type(exc).__name__,
                        },
                        extra=_event_context(q),
                    )
                    _touch_browser_outbox()
                else:
                    try:
                        _feedback_trace().record_custom_setting_rejected(
                            q,
                            setting=raw_setting,
                            extra={
                                **_event_context(q),
                                "label": label.strip() or "Setting",
                                "n_seeds": n_seeds,
                                "base_seed": base_seed,
                                "inherited_from": raw_inherited_from,
                                "error_type": type(exc).__name__,
                                "phase": "spec_validation",
                            },
                        )
                        _touch_browser_outbox()
                    except feedback.FeedbackError:
                        pass
                st.error(f"Could not generate the curve: {exc}")
            else:
                final_metric = result.get("final_metric")
                if final_metric is not None and not np.isfinite(float(final_metric)):
                    final_metric = None
                _feedback_trace().record_custom_run(
                    q,
                    run={
                        "proposal_event_id": proposal_event["event_id"],
                        "status": "completed",
                        "custom_setting_id": result["custom_setting_id"],
                        "candidate_id": result["candidate_id"],
                        "label": result["label"],
                        "n_seeds": result["n_seeds"],
                        "base_seed": result["base_seed"],
                        "selection_metric": result["selection_metric"],
                        "final_metric": final_metric,
                        "inherited_from": result.get("inherited_from"),
                    },
                    extra=_event_context(q),
                )
                _touch_browser_outbox()
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



def _profile_provenance(bundle: QuestionBundle, q: dict[str, Any]) -> dict[str, str]:
    """Resolve question/run profile provenance while preserving legacy artifacts."""
    profile = str(q.get("profile") or "legacy/unknown")
    profile_hash = q.get("profile_hash")
    run_path = bundle.question_root.parent / "run.json"
    if run_path.is_file():
        try:
            run = read_json_file(run_path)
        except (OSError, ValueError, TypeError):
            run = {}
        if not q.get("profile") and run.get("profile"):
            profile = str(run["profile"])
        profile_hash = profile_hash or run.get("profile_hash")
    return {
        "profile": profile,
        "profile_hash": str(profile_hash) if profile_hash else "legacy/unknown",
    }


def _render_metadata(
    q: dict[str, Any],
    provenance: dict[str, str] | None = None,
) -> None:
    provenance = provenance or {
        "profile": str(q.get("profile") or "legacy/unknown"),
        "profile_hash": str(q.get("profile_hash") or "legacy/unknown"),
    }
    st.markdown(f'<div class="question-id">{q["question_id"]}</div>', unsafe_allow_html=True)
    st.markdown(
        (
            f'<div class="question-meta">'
            f"Type: {q.get('type', '—')} · "
            f"Budget: {_question_budget(q)} samples · "
            f"Metric: {q.get('significance', {}).get('metric', 'test_mse')} · "
            f"Choices: {q.get('num_choices', len(q['choices']))} · "
            f"Profile: {provenance['profile']} · "
            f"Profile hash: {provenance['profile_hash']}"
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


def _render_answer_banner(
    q: dict[str, Any], committed_letter: str, *, metric: str
) -> None:
    correct = q["correct_letter"]
    metric_label = _metric_display_name(metric)
    if committed_letter == correct:
        st.success(
            f"Correct — **{committed_letter}** achieves the best {metric_label}."
        )
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


def _signed_latex_sum(terms: list[tuple[float, str]]) -> str:
    rendered: list[str] = []
    for index, (weight, expression) in enumerate(terms):
        sign = "-" if weight < 0 else "+"
        magnitude = f"{abs(float(weight)):.4g}"
        if index == 0:
            rendered.append(f"{'-' if sign == '-' else ''}{magnitude}{expression}")
        else:
            rendered.append(f" {sign} {magnitude}{expression}")
    return "".join(rendered)


def _classification_score_latex(params: dict[str, Any]) -> str:
    family = str(params.get("rule_family", ""))
    features = [int(value) for value in params.get("active_features", [])]
    weights = [float(value) for value in params.get("rule_weights", [])]
    if family == "smooth_additive":
        terms = [
            (weight, rf"\left(\sin(x_{{{feature}}}) + 0.25x_{{{feature}}}^2\right)")
            for feature, weight in zip(features, weights)
        ]
        return rf"s(\mathbf{{x}}) = {_signed_latex_sum(terms)}"
    if family == "sparse_interaction":
        pairs = params.get("interaction_pairs", [])
        terms = [
            (weight, rf"x_{{{int(pair[0])}}}x_{{{int(pair[1])}}}")
            for pair, weight in zip(pairs, weights)
            if isinstance(pair, list) and len(pair) == 2
        ]
        return rf"s(\mathbf{{x}}) = {_signed_latex_sum(terms)}"
    if family == "xor" and len(features) >= 2:
        left, right = features[:2]
        return rf"s(\mathbf{{x}}) = -x_{{{left}}} \cdot x_{{{right}}}"
    if family == "piecewise_boundary" and len(features) >= 2 and len(weights) >= 3:
        primary, secondary = features[:2]
        below_weight, above_weight, offset_weight = weights[:3]
        above = _signed_latex_sum(
            [(above_weight, rf"x_{{{secondary}}}"), (offset_weight, rf"x_{{{primary}}}")]
        )
        below = _signed_latex_sum(
            [(below_weight, rf"x_{{{secondary}}}"), (offset_weight, rf"x_{{{primary}}}")]
        )
        breakpoint = float(params.get("piecewise_breakpoint", 0.0))
        return (
            rf"s(\mathbf{{x}}) = \begin{{cases}} "
            rf"{above}, & x_{{{primary}}} > {breakpoint:.4g} \\ "
            rf"{below}, & x_{{{primary}}} \le {breakpoint:.4g} "
            rf"\end{{cases}}"
        )
    return r"s(\mathbf{x}) = \text{unavailable}"


def _classification_label_latex(params: dict[str, Any]) -> str:
    threshold = float(params.get("decision_threshold", 0.0))
    noise_std = float(params.get("noise_std", 0.0))
    return (
        rf"y = \mathbf{{1}}\{{s(\mathbf{{x}}) + \varepsilon > {threshold:.4g}\}}, "
        rf"\qquad \varepsilon \sim \mathcal{{N}}(0, {noise_std:.4g}^2)"
    )

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

    if family == "synthetic_tabular_classification":
        st.markdown(f"**Input dimension:** {params.get('input_dim', '—')}")
        st.markdown(f"**Classes:** {params.get('num_classes', '—')}")
        st.markdown(f"**Decision rule:** {params.get('rule_family', '—')}")
        active = ", ".join(f"x_{value}" for value in params.get("active_features", []))
        st.markdown(f"**Active features:** {active or '—'}")
        st.markdown(f"**Noise std:** {params.get('noise_std', '—')}")
        st.markdown("**Latent classification rule:**")
        st.latex(_classification_score_latex(params))
        st.latex(_classification_label_latex(params))
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


def _render_session_page() -> None:
    trace = _feedback_trace()
    events = trace.events
    summary = feedback.summarize_session_events(events)
    answers = summary["answers"]
    settings = summary["settings"]
    runs = summary["runs"]
    reactions = summary["reactions"]

    st.markdown("#### My session")
    st.caption(
        f"Session `{trace.session_id}` · append-only activity from this browser "
        "session, including earlier score attempts."
    )

    answer_col, accuracy_col, setting_col, reaction_col, comment_col = st.columns(5)
    answer_col.metric("Answers", answers["total"])
    accuracy = answers["accuracy"]
    accuracy_col.metric(
        "Accuracy",
        "—" if accuracy is None else f"{accuracy:.0%}",
    )
    setting_col.metric(
        "Settings",
        settings["proposed"] + settings["rejected"],
    )
    reaction_col.metric(
        "Surprised",
        f"{reactions['surprised']} / {reactions['total']}",
    )
    comment_col.metric("Comments", summary["comments"])
    st.caption(
        f"{answers['unique_question_versions']} unique question version(s) · "
        f"{answers['correct']} correct / {answers['incorrect']} incorrect / "
        f"{answers['unknown']} unknown · {runs['completed']} custom run(s) completed / "
        f"{runs['failed']} failed · {summary['presentations']} question exposure(s)"
    )

    pending = _pending_upload_count(events)
    quarantined_ids = _quarantined_event_ids()
    quarantined = sum(event.get("event_id") in quarantined_ids for event in events)
    try:
        client = _feedback_client()
        config_error = None
    except feedback.FeedbackConfigurationError as exc:
        client = None
        config_error = str(exc)
    if not events:
        st.info(
            "Your answers, surprise reactions, proposed settings, custom runs, "
            "and comments will appear here."
        )
    elif quarantined:
        st.warning(
            f"{quarantined} event(s) are quarantined after an event-ID content "
            f"conflict; {pending} other event(s) remain uploadable. Use the sidebar "
            "to retry the safe pending subset or download the quarantined envelope."
        )
    elif pending == 0:
        st.success("All session events have been acknowledged by the upload endpoint.")
    elif config_error:
        st.error(f"Feedback upload configuration is invalid: {config_error}")
        st.caption(f"{pending} event(s) remain available for download and retry.")
    elif client is None or not client.is_configured:
        st.info(
            f"{pending} event(s) are saved in this session and pending upload. "
            "The endpoint and Bearer token are not fully configured."
        )
    else:
        st.warning(
            f"{pending} of {len(events)} event(s) are pending upload. "
            "Use Upload pending session events in the sidebar."
        )

    serialized = json.dumps(trace.to_envelope(), ensure_ascii=False, indent=2)
    st.download_button(
        "Download session JSON",
        data=serialized,
        file_name=f"architectureiq-{trace.session_id}.json",
        mime="application/json",
        disabled=not events,
        key="download_session_trace_page",
    )

    st.markdown("##### Answers")
    if summary["answer_rows"]:
        answer_rows = []
        for row in summary["answer_rows"]:
            correctness = row["is_correct"]
            answer_rows.append(
                {
                    "#": row["sequence"],
                    "Time": row["occurred_at"],
                    "Question": row["question_id"],
                    "Choice": row["selected_letter"],
                    "Candidate": row["selected_candidate_id"] or "—",
                    "Result": (
                        "Correct"
                        if correctness is True
                        else "Incorrect"
                        if correctness is False
                        else "Unknown"
                    ),
                }
            )
        st.dataframe(answer_rows, hide_index=True, width="stretch")
    else:
        st.caption("No answers in this session yet.")

    st.markdown("##### Question exposures")
    if summary["presentation_rows"]:
        presentation_rows = [
            {
                "#": row["sequence"],
                "Position": row["position"],
                "Time": row["occurred_at"],
                "Question": row["question_id"],
                "Policy": row["policy_version"],
                "Mode": row["mode"],
                "Source": row["source"],
                "Propensity": row["propensity"],
            }
            for row in summary["presentation_rows"]
        ]
        st.dataframe(presentation_rows, hide_index=True, width="stretch")
    else:
        st.caption("No versioned question exposures in this session yet.")

    st.markdown("##### Proposed settings")
    if summary["proposal_rows"]:
        proposal_rows = [
            {
                "#": row["sequence"],
                "Time": row["occurred_at"],
                "Question": row["question_id"],
                "Status": row["status"],
                "Label": row["label"] or "—",
                "Candidate": row["candidate_id"] or "—",
                "Model": row["model_type"] or "—",
                "Optimizer": row["optimizer_type"] or "—",
                "Loss": row["loss_id"] or "—",
                "Budget": row["total_samples_seen"] or "—",
                "Batch": row["batch_size"] or "—",
                "Error": row["error_type"] or "—",
            }
            for row in summary["proposal_rows"]
        ]
        st.dataframe(proposal_rows, hide_index=True, width="stretch")
    else:
        st.caption("No proposed or rejected settings in this session yet.")

    st.markdown("##### Surprise reactions")
    if summary["reaction_rows"]:
        reaction_rows = [
            {
                "#": row["sequence"],
                "Time": row["occurred_at"],
                "Question": row["question_id"],
                "Reaction": "Surprised" if row["value"] else "As expected",
                "Timing": "After result reveal",
            }
            for row in summary["reaction_rows"]
        ]
        st.dataframe(reaction_rows, hide_index=True, width="stretch")
    else:
        st.caption("No surprise reactions in this session yet.")

    st.markdown("##### Comments")
    if summary["comment_rows"]:
        comment_rows = [
            {
                "#": row["sequence"],
                "Time": row["occurred_at"],
                "Question": row["question_id"],
                "Category": COMMENT_CATEGORY_LABELS.get(
                    row["category"], row["category"]
                ),
                "Message": row["text"],
            }
            for row in summary["comment_rows"]
        ]
        st.dataframe(comment_rows, hide_index=True, width="stretch")
    else:
        st.caption("No comments in this session yet.")


def _render_question_reaction(q: dict[str, Any]) -> None:
    """Collect one explicit post-result surprise label for this attempt."""
    st.markdown("#### Did the revealed result surprise you?")
    st.caption(
        "This is separate from whether your answer was correct and whether you "
        "liked the question. The first response in this attempt is kept."
    )
    existing = _question_reaction(q)
    if existing is not None:
        payload = existing["payload"]
        value = payload["value"]
        event_id = existing.get("event_id")
        st.success(
            "Recorded: **😮 Surprised / 出乎意料**"
            if value
            else "Recorded: **As expected / 符合预期**"
        )
        if event_id in _uploaded_event_ids():
            st.caption("This reaction has been acknowledged by the upload endpoint.")
        elif event_id in _quarantined_event_ids():
            st.error(
                "This reaction is quarantined because its event ID conflicts with "
                "different stored content. The first local response remains unchanged."
            )
        else:
            st.caption(
                "This reaction is pending in the session outbox and can be uploaded "
                "with the rest of your trace."
            )
        return

    surprised_col, expected_col = st.columns(2)
    with surprised_col:
        surprised = st.button(
            "😮 Surprised / 出乎意料",
            key=f"reaction_surprised_{_question_session_key(q)[-12:]}",
            type="primary",
            use_container_width=True,
        )
    with expected_col:
        expected = st.button(
            "As expected / 符合预期",
            key=f"reaction_expected_{_question_session_key(q)[-12:]}",
            use_container_width=True,
        )
    if not surprised and not expected:
        return

    try:
        event = _record_question_reaction(q, value=surprised)
    except feedback.FeedbackValidationError as exc:
        st.error(str(exc))
        return
    try:
        client = _feedback_client()
    except feedback.FeedbackConfigurationError:
        # Capture is independent of transport configuration. The immutable
        # event stays pending and its durable status appears on the next run.
        st.rerun()

    if client.is_configured:
        try:
            with st.spinner("Uploading reaction…"):
                receipt = client.post_event(event)
        except feedback.FeedbackUploadConflictError as exc:
            _quarantine_event(event, exc)
        except feedback.FeedbackError:
            # The immutable event is already in the reliable outbox. The next
            # render reports it as pending without exposing transport details.
            pass
        else:
            _mark_events_uploaded([event], receipt)
    st.rerun()


def _render_question_comment(q: dict[str, Any]) -> None:
    question_key = _question_session_key(q)
    existing_comments = [
        event
        for event in _feedback_trace().events
        if event["event_type"] == "comment_submitted"
        and event["question_version"] == question_key
    ]
    try:
        client = _feedback_client()
        config_error = None
    except feedback.FeedbackConfigurationError as exc:
        client = None
        config_error = str(exc)

    with st.expander("Comment on this question", expanded=False):
        if existing_comments:
            st.caption(
                f"{len(existing_comments)} comment(s) from this session are in the trace."
            )
        if config_error:
            st.error(f"Feedback upload configuration is invalid: {config_error}")
        elif client is None or not client.is_configured:
            st.caption(
                "The upload endpoint and Bearer token are not fully configured. "
                "Your comment will be added to the downloadable session trace."
            )
        notice = st.session_state.comment_notices.get(question_key)
        if notice:
            if isinstance(notice, Mapping):
                level = notice.get("level")
                message = notice.get("message")
            elif isinstance(notice, (tuple, list)) and len(notice) == 2:
                level, message = notice
            else:
                level, message = None, None
            if level in {"info", "success", "warning", "error"} and isinstance(
                message, str
            ):
                getattr(st, level)(message)

        with st.form(
            f"comment_{q['question_id']}_{question_key[-12:]}",
            clear_on_submit=True,
        ):
            category = st.selectbox(
                "Category",
                list(COMMENT_CATEGORY_LABELS),
                format_func=COMMENT_CATEGORY_LABELS.get,
            )
            comment_text = st.text_area(
                "Message",
                max_chars=feedback.MAX_COMMENT_LENGTH,
                placeholder="What should the ArchitectureIQ team know about this question?",
            )
            upload_available = client is not None and client.is_configured
            submitted = st.form_submit_button(
                "Upload comment"
                if upload_available
                else "Add comment to session trace",
                type="primary",
            )

        if submitted:
            try:
                event = _feedback_trace().record_comment(
                    q,
                    category=category,
                    text=comment_text,
                    extra=_event_context(q),
                )
                _touch_browser_outbox()
            except feedback.FeedbackValidationError as exc:
                st.error(str(exc))
            else:
                event_id = str(event["event_id"])
                if client is None or not client.is_configured:
                    _set_comment_notice(
                        question_key,
                        level="info",
                        message=(
                            "Comment added to this session trace. Configure the endpoint "
                            "and Bearer token or use the sidebar download to send it later."
                        ),
                        event_id=event_id,
                    )
                else:
                    try:
                        with st.spinner("Uploading comment…"):
                            receipt = client.post_event(event)
                    except feedback.FeedbackUploadConflictError as exc:
                        _quarantine_event(event, exc)
                        _set_comment_notice(
                            question_key,
                            level="error",
                            message=_upload_conflict_message(exc, subject="Comment"),
                            event_id=event_id,
                        )
                    except feedback.FeedbackError as exc:
                        _set_comment_notice(
                            question_key,
                            level="error",
                            message=(
                                "Comment upload failed, but it remains in the session "
                                f"trace for batch retry. {exc}"
                            ),
                            event_id=event_id,
                        )
                    else:
                        request_note = (
                            f" · request {receipt.request_id}"
                            if receipt.request_id
                            else ""
                        )
                        if _mark_events_uploaded([event], receipt):
                            _set_comment_notice(
                                question_key,
                                level="success",
                                message=(
                                    "Comment upload complete · "
                                    f"{_upload_receipt_summary(receipt, sent_count=1)}"
                                    f"{request_note}."
                                ),
                                event_id=event_id,
                            )
                        else:
                            _set_comment_notice(
                                question_key,
                                level="warning",
                                message=(
                                    "The endpoint returned a partial or invalid receipt. "
                                    "The comment remains pending in the session trace."
                                ),
                                event_id=event_id,
                            )
                st.rerun()


def _render_question_page(
    bundle: QuestionBundle,
    q: dict[str, Any],
    *,
    committed: bool,
    focus_letter: str | None,
) -> None:
    metric = _selection_metric(bundle, q)
    _render_metadata(q, _profile_provenance(bundle, q))
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
        _render_question_reaction(q)

    _render_question_comment(q)

    inspect_letter = st.session_state.info_letter or st.session_state.focus_letter
    if inspect_letter:
        selected_choice = next(
            c for c in bundle.choices if c["letter"] == inspect_letter
        )
        st.divider()
        st.markdown(
            f"#### Files · Choice **{inspect_letter}** · `{selected_choice['candidate_id']}`"
        )
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
    _cached_question_dirs.clear()


def _render_app() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Questions")
        packs = _question_pack_registry()
        active_pack = _render_question_pack_selector(packs)
        collection_path = (
            active_pack["collection_path"] if active_pack is not None else None
        )
        if active_pack is None:
            _ensure_demo_data(st.session_state.data_root)
        _render_score_panel()
        st.divider()
        data_root = st.text_input(
            "Data root",
            key="data_root",
            value=str(st.session_state.get("data_root") or _initial_data_root()),
            disabled=active_pack is not None,
        )
        _render_question_picker(data_root, collection_path)
        st.divider()
        _render_session_feedback_panel()

    bundle: QuestionBundle | None = st.session_state.bundle

    if bundle is None:
        st.title("ArchitectureIQ Question Inspector")
        st.info("Select a question in the sidebar to begin.")
        return

    q = bundle.question
    _assign_question_cache_scope(bundle)
    committed = st.session_state.committed_letter is not None
    focus_letter = st.session_state.focus_letter

    tab_question, tab_prompt, tab_session = st.tabs(
        ["Question", "Prompt", "My session"]
    )
    with tab_question:
        _render_question_page(
            bundle,
            q,
            committed=committed,
            focus_letter=focus_letter,
        )
    with tab_prompt:
        _render_prompt_page(bundle)
    with tab_session:
        _render_session_page()


def main() -> None:
    _init_state()
    _hydrate_browser_outbox()
    try:
        _render_app()
    finally:
        _persist_browser_outbox()


if __name__ == "__main__":
    main()
