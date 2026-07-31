"""Tests for the inspector's standalone feedback event module."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))

import feedback  # noqa: E402


RECEIPT_REQUEST_ID = "72aee12d-7742-44ea-b3d9-f056ae5c8ac2"
OTHER_REQUEST_ID = "bdbb2f4b-87a0-44c9-83f1-fdc5c596c36d"


def _question() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "question_id": "q_example",
        "family": "regression",
        "choices": [
            {"letter": "A", "candidate_id": "c_a"},
            {"letter": "B", "candidate_id": "c_b"},
        ],
        "budget": {"total_samples_seen": 1_024},
    }


def test_random_anonymous_session_ids() -> None:
    first = feedback.generate_session_id()
    second = feedback.generate_session_id()

    assert first.startswith("anon_")
    assert second.startswith("anon_")
    assert first != second
    assert "/" not in first and "@" not in first


def test_question_version_is_canonical_and_ignores_inspector_keys() -> None:
    question = _question()
    reordered = {
        "budget": {"total_samples_seen": 1_024, "_inspector_plot": "temporary"},
        "choices": [
            {"candidate_id": "c_a", "letter": "A"},
            {"candidate_id": "c_b", "letter": "B"},
        ],
        "family": "regression",
        "question_id": "q_example",
        "schema_version": "2.0",
        "_inspector_cache_scope": "some-local-path-hash",
    }

    assert feedback.compute_question_version(question) == feedback.question_version(
        reordered
    )
    assert "_inspector_cache_scope" in reordered  # hashing does not mutate input

    changed = _question()
    changed["choices"][0]["candidate_id"] = "c_changed"
    assert feedback.question_version(changed) != feedback.question_version(question)


def test_feedback_json_accepts_recursive_safe_integer_boundaries() -> None:
    trace = feedback.SessionTrace("anon_safe_numbers")
    event = trace.record_custom_setting(
        _question(),
        setting={
            "nested": [
                feedback.MAX_SAFE_JSON_INTEGER,
                -feedback.MAX_SAFE_JSON_INTEGER,
                float(feedback.MAX_SAFE_JSON_INTEGER),
                float(-feedback.MAX_SAFE_JSON_INTEGER),
            ]
        },
        event_id="evt_safe_numbers",
    )

    assert event["payload"]["setting"]["nested"] == [
        feedback.MAX_SAFE_JSON_INTEGER,
        -feedback.MAX_SAFE_JSON_INTEGER,
        float(feedback.MAX_SAFE_JSON_INTEGER),
        float(-feedback.MAX_SAFE_JSON_INTEGER),
    ]


@pytest.mark.parametrize(
    "value",
    [
        feedback.MAX_SAFE_JSON_INTEGER + 1,
        -feedback.MAX_SAFE_JSON_INTEGER - 1,
        float(feedback.MAX_SAFE_JSON_INTEGER + 1),
        float(-feedback.MAX_SAFE_JSON_INTEGER - 1),
    ],
)
def test_feedback_json_rejects_recursive_unsafe_integer_values(
    value: int | float,
) -> None:
    trace = feedback.SessionTrace("anon_unsafe_numbers")

    with pytest.raises(feedback.FeedbackValidationError, match="integer-valued"):
        trace.record_custom_setting(
            _question(),
            setting={"nested": [{"value": value}]},
            event_id="evt_unsafe_numbers",
        )


def test_feedback_json_rejects_lone_surrogates_as_validation_errors() -> None:
    with pytest.raises(feedback.FeedbackValidationError, match="surrogate"):
        feedback.SessionTrace("anon_\ud800")

    trace = feedback.SessionTrace("anon_surrogate")
    with pytest.raises(feedback.FeedbackValidationError, match="surrogate"):
        trace.record_comment(
            _question(),
            category="other",
            text="bad high surrogate \ud800",
            event_id="evt_surrogate_comment",
        )
    with pytest.raises(feedback.FeedbackValidationError, match="surrogate"):
        trace.record_custom_setting(
            _question(),
            setting={"nested": ["bad low surrogate \udc00"]},
            event_id="evt_surrogate_setting",
        )
    with pytest.raises(feedback.FeedbackValidationError, match="surrogate"):
        feedback.question_version({"question_id": "q_surrogate", "value": "\ud800"})


def test_session_trace_records_all_event_types_and_builds_envelope() -> None:
    question = _question()
    trace = feedback.SessionTrace("anon_test", created_at="2026-07-11T00:00:00.000Z")

    answer = trace.record_answer(
        question,
        selected_letter="B",
        selected_candidate_id="c_b",
        event_id="evt_answer",
        occurred_at="2026-07-11T00:00:01.000Z",
    )
    setting = trace.record_custom_setting(
        question,
        setting={"model": {"type": "mlp", "width": 64}},
        event_id="evt_setting",
    )
    rejected = trace.record_custom_setting_rejected(
        question,
        setting={"budget": 100, "batch_size": 32},
        event_id="evt_rejected",
        extra={"error_type": "ValueError"},
    )
    run = trace.record_custom_run(
        question,
        run={"run_id": "run_1", "status": "completed", "metric": 0.125},
        event_id="evt_run",
    )
    failed_run = trace.record_custom_run(
        question,
        run={"run_id": "run_2", "status": "failed", "error_type": "RuntimeError"},
        event_id="evt_failed_run",
    )
    comment = trace.record_comment(
        question,
        category="suggestion",
        text="  Please expose the training seed.  ",
        event_id="evt_comment",
    )

    assert [
        answer["sequence"],
        setting["sequence"],
        rejected["sequence"],
        run["sequence"],
        failed_run["sequence"],
        comment["sequence"],
    ] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [event["event_type"] for event in trace.events] == [
        "answer_submitted",
        "custom_setting_proposed",
        "custom_setting_rejected",
        "custom_run_completed",
        "custom_run_failed",
        "comment_submitted",
    ]
    assert comment["payload"]["text"] == "Please expose the training seed."

    envelope = trace.to_envelope()
    assert envelope["schema_version"] == feedback.TRACE_SCHEMA_VERSION
    assert envelope["envelope_type"] == "session_trace"
    assert envelope["session_id"] == "anon_test"
    assert envelope["event_count"] == 6
    assert envelope["trace_id"].startswith("trace_")
    assert all(
        event["schema_version"] == feedback.EVENT_SCHEMA_VERSION
        for event in envelope["events"]
    )

    parsed = feedback.parse_session_trace_json(
        json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    )
    assert parsed == envelope
    assert feedback.validate_session_trace_envelope(envelope) == envelope


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("trace_id", "trace_forged"), "trace_id"),
        (lambda value: value.__setitem__("event_count", 2), "event_count"),
        (lambda value: value.__setitem__("unexpected", True), "exact schema"),
        (
            lambda value: value["events"][0].__setitem__("session_id", "other"),
            "same session_id",
        ),
        (
            lambda value: value["events"][0].pop("sequence"),
            "exact wire schema",
        ),
        (
            lambda value: value.__setitem__("created_at", "not-a-timestamp"),
            "RFC 3339",
        ),
        (
            lambda value: value["events"][0].__setitem__(
                "occurred_at", "2026-99-99T00:00:00Z"
            ),
            "RFC 3339",
        ),
        (
            lambda value: value["events"][0].__setitem__(
                "sequence", feedback.MAX_EVENT_SEQUENCE + 1
            ),
            "between 1",
        ),
    ],
)
def test_downloaded_trace_validation_fails_closed(
    mutation: Any,
    message: str,
) -> None:
    trace = feedback.SessionTrace(
        "anon_import",
        created_at="2026-07-12T00:00:00Z",
    )
    trace.record_comment(
        _question(),
        category="other",
        text="saved comment",
        event_id="evt_import",
        occurred_at="2026-07-12T00:00:00Z",
    )
    envelope = trace.to_envelope()
    mutation(envelope)

    with pytest.raises(feedback.FeedbackValidationError, match=message):
        feedback.validate_session_trace_envelope(envelope)


def test_downloaded_trace_accepts_rfc3339_nanosecond_timestamps() -> None:
    trace = feedback.SessionTrace(
        "anon_nanosecond_import",
        created_at="2026-07-12T00:00:00Z",
    )
    trace.record_comment(
        _question(),
        category="other",
        text="saved comment",
        event_id="evt_nanosecond_import",
        occurred_at="2026-07-12T00:00:00Z",
    )
    envelope = trace.to_envelope()
    envelope["created_at"] = "2026-07-12T00:00:00.123456789Z"
    envelope["events"][0]["occurred_at"] = "2026-07-12T00:00:00.987654321Z"

    assert feedback.validate_session_trace_envelope(envelope) == envelope


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"schema_version":"1.0","schema_version":"1.0"}', "duplicate"),
        ('{"value":NaN}', "non-standard"),
        ('{"value":9007199254740992}', "integer-valued"),
        (r'{"value":"\ud800"}', "surrogate"),
        (b"\xff", "UTF-8"),
        ("[]", "mapping"),
    ],
)
def test_saved_trace_json_parser_rejects_ambiguous_or_non_strict_input(
    raw: bytes | str,
    message: str,
) -> None:
    with pytest.raises(feedback.FeedbackValidationError, match=message):
        feedback.parse_session_trace_json(raw)


def test_empty_trace_cannot_be_recovered_or_posted() -> None:
    trace = feedback.SessionTrace(
        "anon_empty_import",
        created_at="2026-07-12T00:00:00Z",
    )
    envelope = trace.to_envelope()

    with pytest.raises(feedback.FeedbackValidationError, match="positive"):
        feedback.validate_session_trace_envelope(envelope)


def test_events_are_append_only_and_event_ids_are_idempotent() -> None:
    trace = feedback.SessionTrace("anon_test")
    original = trace.record_answer(
        _question(),
        selected_letter="A",
        event_id="evt_same",
        occurred_at="2026-07-11T00:00:01.000Z",
    )

    replay = trace.record_answer(
        _question(),
        selected_letter="A",
        event_id="evt_same",
        occurred_at="2026-07-11T00:00:10.000Z",
    )
    assert replay == original
    assert len(trace) == 1

    replay["payload"]["selected_letter"] = "B"
    assert trace.events[0]["payload"]["selected_letter"] == "A"

    with pytest.raises(feedback.EventConflictError, match="different content"):
        trace.record_answer(_question(), selected_letter="B", event_id="evt_same")
    assert len(trace) == 1


def test_event_shape_rejects_unknown_fields_but_sequences_local_events() -> None:
    event = feedback.build_event(
        "answer_submitted",
        session_id="anon_exact_shape",
        question=_question(),
        payload={"selected_letter": "A"},
        event_id="evt_exact_shape",
        occurred_at="2026-07-11T00:00:01.000Z",
    )
    assert "sequence" not in event

    trace = feedback.SessionTrace("anon_exact_shape")
    appended = trace.append_event(event)
    assert appended["sequence"] == 1
    assert set(trace.to_envelope()["events"][0]) == feedback.EVENT_KEYS

    unsupported = {**event, "trace_id": "trace_not_an_event_field"}
    with pytest.raises(feedback.FeedbackValidationError, match="unsupported fields"):
        trace.append_event(unsupported)


@pytest.mark.parametrize(
    ("original_extra", "replayed_extra"),
    [
        ({"value": 1}, {"value": 1.0}),
        ({"value": -0.0}, {"value": 0}),
        (
            {"value": {"first": 1, "second": [2, 3]}},
            {"value": {"second": [2.0, 3.0], "first": 1.0}},
        ),
    ],
)
def test_logical_event_replay_uses_json_numeric_and_object_equality(
    original_extra: dict[str, Any],
    replayed_extra: dict[str, Any],
) -> None:
    trace = feedback.SessionTrace("anon_json_equal")
    original = trace.record_answer(
        _question(),
        selected_letter="A",
        event_id="evt_json_equal",
        occurred_at="2026-07-11T00:00:01.000Z",
        extra=original_extra,
    )

    replayed = trace.record_answer(
        _question(),
        selected_letter="A",
        event_id="evt_json_equal",
        occurred_at="2026-07-11T00:00:10.000Z",
        extra=replayed_extra,
    )

    assert replayed == original
    assert len(trace) == 1


@pytest.mark.parametrize(
    ("original_extra", "conflicting_extra"),
    [
        ({"value": True}, {"value": 1}),
        ({"value": [1, 2]}, {"value": [2, 1]}),
        ({"value": None}, {}),
    ],
)
def test_logical_event_replay_preserves_json_types_order_and_presence(
    original_extra: dict[str, Any],
    conflicting_extra: dict[str, Any],
) -> None:
    trace = feedback.SessionTrace("anon_json_conflict")
    trace.record_answer(
        _question(),
        selected_letter="A",
        event_id="evt_json_conflict",
        extra=original_extra,
    )

    with pytest.raises(feedback.EventConflictError, match="different content"):
        trace.record_answer(
            _question(),
            selected_letter="A",
            event_id="evt_json_conflict",
            extra=conflicting_extra,
        )
    assert len(trace) == 1


def test_session_summary_counts_and_rows_preserve_event_order() -> None:
    secret = "must-not-appear-in-session-summary"
    first_question = _question()
    second_question = _question()
    second_question["question_id"] = "q_second"
    trace = feedback.SessionTrace("anon_summary")

    trace.record_answer(
        first_question,
        selected_letter="A",
        selected_candidate_id="c_a",
        event_id="evt_answer_correct",
        extra={"is_correct": True, "bearer_token": secret},
    )
    trace.record_custom_setting(
        first_question,
        setting={
            "candidate_id": "c_custom",
            "budget": {"total_samples_seen": 2_048, "batch_size": 32},
            "model": {"type": "mlp", "width": 64},
            "optimizer": {"type": "AdamW", "lr": 0.001},
            "loss": {"loss_id": "mse"},
            "api_key": secret,
        },
        event_id="evt_proposal",
        extra={"label": "Wide MLP", "access_token": secret},
    )
    trace.record_answer(
        second_question,
        selected_letter="B",
        selected_candidate_id="c_b",
        event_id="evt_answer_incorrect",
        extra={"is_correct": False},
    )
    trace.record_custom_setting_rejected(
        second_question,
        setting={
            "budget": {"total_samples_seen": 10, "batch_size": 3},
            "model": {"type": "linear"},
            "optimizer": {"optimizer_type": "SGD"},
            "private_note": secret,
        },
        event_id="evt_rejected",
        extra={"label": "Invalid budget", "error_type": "ValueError"},
    )
    trace.record_custom_run(
        first_question,
        run={"status": "completed", "final_metric": 0.25, "token": secret},
        event_id="evt_run_completed",
    )
    trace.record_comment(
        second_question,
        category="suggestion",
        text="Show the seed count.",
        event_id="evt_comment",
    )
    trace.record_answer(
        first_question,
        selected_letter="B",
        event_id="evt_answer_unknown",
    )
    trace.record_custom_run(
        second_question,
        run={"status": "failed", "error_type": "RuntimeError"},
        event_id="evt_run_failed",
    )

    summary = feedback.summarize_session_events(trace.events)

    assert summary["event_count"] == 8
    assert summary["event_type_counts"] == {
        "answer_submitted": 3,
        "comment_submitted": 1,
        "custom_run_completed": 1,
        "custom_run_failed": 1,
        "custom_setting_proposed": 1,
        "custom_setting_rejected": 1,
        "question_presented": 0,
        "question_reaction_submitted": 0,
    }
    assert summary["answers"] == {
        "total": 3,
        "unique_question_versions": 2,
        "known": 2,
        "correct": 1,
        "incorrect": 1,
        "unknown": 1,
        "accuracy": 0.5,
    }
    assert summary["settings"] == {"proposed": 1, "rejected": 1}
    assert summary["runs"] == {"completed": 1, "failed": 1}
    assert summary["comments"] == 1

    assert [row["sequence"] for row in summary["answer_rows"]] == [1, 3, 7]
    assert [row["is_correct"] for row in summary["answer_rows"]] == [
        True,
        False,
        None,
    ]
    assert [row["sequence"] for row in summary["proposal_rows"]] == [2, 4]
    assert summary["proposal_rows"][0] == {
        "sequence": 2,
        "occurred_at": trace.events[1]["occurred_at"],
        "question_id": "q_example",
        "question_version": feedback.question_version(first_question),
        "status": "proposed",
        "label": "Wide MLP",
        "candidate_id": "c_custom",
        "model_type": "mlp",
        "optimizer_type": "AdamW",
        "loss_id": "mse",
        "total_samples_seen": 2_048,
        "batch_size": 32,
        "error_type": None,
    }
    assert summary["proposal_rows"][1]["status"] == "rejected"
    assert summary["proposal_rows"][1]["optimizer_type"] == "SGD"
    assert summary["proposal_rows"][1]["error_type"] == "ValueError"
    assert summary["comment_rows"] == [
        {
            "sequence": 6,
            "occurred_at": trace.events[5]["occurred_at"],
            "question_id": "q_second",
            "question_version": feedback.question_version(second_question),
            "category": "suggestion",
            "text": "Show the seed count.",
        }
    ]
    assert secret not in json.dumps(summary)
    assert "session_id" not in json.dumps(summary)
    assert "event_id" not in json.dumps(summary)


def test_empty_session_summary_has_zero_counts_and_no_accuracy() -> None:
    summary = feedback.summarize_session_events([])

    assert summary["event_count"] == 0
    assert set(summary["event_type_counts"]) == feedback.EVENT_TYPES
    assert all(count == 0 for count in summary["event_type_counts"].values())
    assert summary["answers"] == {
        "total": 0,
        "unique_question_versions": 0,
        "known": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "accuracy": None,
    }
    assert summary["settings"] == {"proposed": 0, "rejected": 0}
    assert summary["runs"] == {"completed": 0, "failed": 0}
    assert summary["comments"] == 0
    assert summary["answer_rows"] == []
    assert summary["proposal_rows"] == []
    assert summary["comment_rows"] == []


def test_session_summary_rejects_non_append_only_or_mixed_session_events() -> None:
    first = feedback.SessionTrace("anon_first")
    first.record_answer(_question(), selected_letter="A", event_id="evt_first")
    first.record_answer(_question(), selected_letter="B", event_id="evt_second")
    second = feedback.SessionTrace("anon_second")
    second.record_answer(_question(), selected_letter="A", event_id="evt_other")

    duplicate = list(first.events)
    duplicate.append(first.events[0])
    with pytest.raises(feedback.FeedbackValidationError, match="duplicate event_id"):
        feedback.summarize_session_events(duplicate)

    out_of_order = list(first.events)
    out_of_order[1]["sequence"] = 1
    with pytest.raises(feedback.FeedbackValidationError, match="strictly increasing"):
        feedback.summarize_session_events(out_of_order)

    with pytest.raises(feedback.FeedbackValidationError, match="same session_id"):
        feedback.summarize_session_events([first.events[0], second.events[0]])


@pytest.mark.parametrize("category", sorted(feedback.COMMENT_CATEGORIES))
def test_comment_accepts_each_declared_category(category: str) -> None:
    resolved_category, resolved_text = feedback.validate_comment(category, " useful ")
    assert (resolved_category, resolved_text) == (category, "useful")


def test_comment_rejects_unknown_empty_and_oversized_values() -> None:
    with pytest.raises(feedback.FeedbackValidationError, match="category"):
        feedback.validate_comment("private_note", "hello")
    with pytest.raises(feedback.FeedbackValidationError, match="empty"):
        feedback.validate_comment("other", "  \n ")
    with pytest.raises(feedback.FeedbackValidationError, match="at most"):
        feedback.validate_comment("other", "x" * (feedback.MAX_COMMENT_LENGTH + 1))


def test_comment_and_identifier_limits_count_unicode_code_points() -> None:
    emoji = "😀"
    comment = emoji * feedback.MAX_COMMENT_LENGTH

    assert feedback.validate_comment("other", comment) == ("other", comment)
    with pytest.raises(feedback.FeedbackValidationError, match="at most"):
        feedback.validate_comment(
            "other",
            emoji * (feedback.MAX_COMMENT_LENGTH + 1),
        )

    event = feedback.build_event(
        "comment_submitted",
        session_id=emoji * 200,
        question=_question(),
        payload={"category": "other", "text": "valid emoji"},
        event_id="evt_emoji_identifier",
    )
    assert len(event["session_id"]) == 200
    with pytest.raises(feedback.FeedbackValidationError, match="at most 200"):
        feedback.build_event(
            "comment_submitted",
            session_id=emoji * 201,
            question=_question(),
            payload={"category": "other", "text": "invalid identifier"},
            event_id="evt_emoji_identifier_too_long",
        )


def test_config_explicit_values_override_environment_and_token_repr_is_redacted() -> (
    None
):
    environ = {
        feedback.FEEDBACK_ENDPOINT_ENV: "https://env.example/upload",
        feedback.FEEDBACK_TOKEN_ENV: "env-secret",
        feedback.FEEDBACK_TIMEOUT_ENV: "8.5",
    }
    config = feedback.FeedbackConfig.from_sources(
        endpoint="https://explicit.example/feedback",
        bearer_token="explicit-secret",
        timeout_seconds=2,
        environ=environ,
    )

    assert config.endpoint == "https://explicit.example/feedback"
    assert config.bearer_token == "explicit-secret"
    assert config.timeout_seconds == 2
    assert config.is_configured
    assert "explicit-secret" not in repr(config)


def test_config_can_be_detected_without_attempting_upload() -> None:
    config = feedback.FeedbackConfig.from_env({})
    client = feedback.FeedbackClient(config)

    assert not config.is_configured
    assert not client.is_configured
    assert not feedback.is_feedback_configured(environ={})
    with pytest.raises(feedback.FeedbackNotConfiguredError, match="not configured"):
        client.post_json({"event": "anything"})


def test_config_requires_bearer_token_before_endpoint_is_ready() -> None:
    config = feedback.FeedbackConfig.from_sources(
        endpoint="https://collector.example/feedback",
        environ={},
    )
    client = feedback.FeedbackClient(config)

    assert not config.is_configured
    assert not client.is_configured
    assert not feedback.is_feedback_configured(
        endpoint="https://collector.example/feedback",
        environ={},
    )
    with pytest.raises(
        feedback.FeedbackNotConfiguredError,
        match=feedback.FEEDBACK_TOKEN_ENV,
    ):
        config.require_configured()
    with pytest.raises(
        feedback.FeedbackNotConfiguredError,
        match=feedback.FEEDBACK_TOKEN_ENV,
    ):
        client.post_json({"event": "anything"})
    assert feedback.is_feedback_configured(
        endpoint="https://collector.example/feedback",
        bearer_token="ingest-secret",
        environ={},
    )


@pytest.mark.parametrize(
    ("endpoint", "bearer_token", "missing_env"),
    [
        (None, "ingest-secret", feedback.FEEDBACK_ENDPOINT_ENV),
        ("   ", "ingest-secret", feedback.FEEDBACK_ENDPOINT_ENV),
        ("https://collector.example/feedback", None, feedback.FEEDBACK_TOKEN_ENV),
        ("https://collector.example/feedback", "   ", feedback.FEEDBACK_TOKEN_ENV),
    ],
)
def test_config_requires_both_nonempty_authenticated_sources(
    endpoint: str | None,
    bearer_token: str | None,
    missing_env: str,
) -> None:
    config = feedback.FeedbackConfig.from_sources(
        endpoint=endpoint,
        bearer_token=bearer_token,
        environ={},
    )
    assert not config.is_configured
    with pytest.raises(feedback.FeedbackNotConfiguredError, match=missing_env):
        config.require_configured()


def test_config_is_ready_when_endpoint_and_token_come_from_environment() -> None:
    environ = {
        feedback.FEEDBACK_ENDPOINT_ENV: "https://collector.example/feedback",
        feedback.FEEDBACK_TOKEN_ENV: "ingest-secret",
    }
    assert feedback.FeedbackConfig.from_env(environ).is_configured
    assert feedback.is_feedback_configured(environ=environ)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": 123},
        {"endpoint": "https://example.test", "bearer_token": 456},
    ],
)
def test_config_rejects_non_string_endpoint_and_token(kwargs: dict[str, Any]) -> None:
    with pytest.raises(feedback.FeedbackConfigurationError, match="must be strings"):
        feedback.FeedbackConfig.from_sources(environ={}, **kwargs)


def _receipt(
    response: Any,
    *,
    status_code: int = 202,
    request_id: str | None = RECEIPT_REQUEST_ID,
    include_body_request_id: bool = True,
    body_request_id: Any = None,
) -> feedback.UploadReceipt:
    resolved_response = response
    if isinstance(response, dict):
        resolved_response = dict(response)
        if include_body_request_id:
            resolved_response.setdefault(
                "request_id",
                request_id if body_request_id is None else body_request_id,
            )
    return feedback.UploadReceipt(
        status_code=status_code,
        endpoint="https://collector.example/feedback",
        response=resolved_response,
        request_id=request_id,
    )


@pytest.mark.parametrize("status_code", [200, 202, 299])
@pytest.mark.parametrize(
    "response",
    [None, "ok", "<html>ok</html>", {}, {"receipt_id": "r_1"}],
)
def test_generic_2xx_upload_receipt_is_not_acknowledged_by_default(
    response: Any,
    status_code: int,
) -> None:
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(response, status_code=status_code),
        sent_count=3,
    )


@pytest.mark.parametrize("status_code", [200, 202, 299])
@pytest.mark.parametrize("response", [None, "ok", {}, {"receipt_id": "r_1"}])
def test_legacy_generic_2xx_requires_explicit_opt_in(
    response: Any,
    status_code: int,
) -> None:
    assert feedback.upload_receipt_acknowledges_all(
        _receipt(response, status_code=status_code),
        sent_count=3,
        allow_legacy_generic_2xx=True,
    )


@pytest.mark.parametrize("legacy_flag", [None, 0, 1, "true", "false"])
def test_legacy_generic_2xx_only_accepts_literal_true(legacy_flag: Any) -> None:
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(None),
        sent_count=3,
        allow_legacy_generic_2xx=legacy_flag,
    )


@pytest.mark.parametrize("status_code", [199, 300, 503])
def test_non_2xx_upload_receipt_never_acknowledges_all(status_code: int) -> None:
    receipt = _receipt(
        {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
        status_code=status_code,
    )
    assert not feedback.upload_receipt_acknowledges_all(receipt, sent_count=3)
    assert not feedback.upload_receipt_acknowledges_all(
        receipt,
        sent_count=3,
        allow_legacy_generic_2xx=True,
    )


@pytest.mark.parametrize(
    ("sent_count", "response"),
    [
        (0, {"accepted": 0, "duplicate": 0, "conflict": 0, "rejected": 0}),
        (3, {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0}),
        (3, {"accepted": 0, "duplicate": 3, "conflict": 0, "rejected": 0}),
        (3, {"accepted": 1, "duplicate": 2, "conflict": 0, "rejected": 0}),
    ],
)
@pytest.mark.parametrize("status_code", [200, 202, 299])
def test_complete_upload_counters_acknowledge_all(
    sent_count: int,
    response: dict[str, int],
    status_code: int,
) -> None:
    assert feedback.upload_receipt_acknowledges_all(
        _receipt(response, status_code=status_code),
        sent_count=sent_count,
    )


@pytest.mark.parametrize(
    "response",
    [
        {"accepted": 3},
        {"accepted": 3, "duplicate": 0},
        {"accepted": 3, "duplicate": 0, "rejected": 0},
        {"accepted": True, "duplicate": 2, "conflict": 0, "rejected": 0},
        {"accepted": 1.0, "duplicate": 2, "conflict": 0, "rejected": 0},
        {"accepted": "1", "duplicate": 2, "conflict": 0, "rejected": 0},
        {"accepted": -1, "duplicate": 4, "conflict": 0, "rejected": 0},
        {"accepted": 2, "duplicate": 0, "conflict": 0, "rejected": 0},
        {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 1},
        {"accepted": 1, "duplicate": 2, "conflict": 1, "rejected": 0},
        {"accepted": 1, "duplicate": 2, "conflict": True, "rejected": 0},
        {"accepted": 1, "duplicate": 2, "conflict": 0.0, "rejected": 0},
        {"accepted": 1, "duplicate": 2, "conflict": -1, "rejected": 0},
        {"conflict": 0},
    ],
)
def test_partial_or_invalid_upload_counters_do_not_acknowledge_all(
    response: dict[str, Any],
) -> None:
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(response),
        sent_count=3,
    )
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(response),
        sent_count=3,
        allow_legacy_generic_2xx=True,
    )


@pytest.mark.parametrize(
    "missing_counter", ["accepted", "duplicate", "conflict", "rejected"]
)
def test_each_missing_counter_stays_unacknowledged_in_legacy_mode(
    missing_counter: str,
) -> None:
    response = {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0}
    response.pop(missing_counter)
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(response),
        sent_count=3,
        allow_legacy_generic_2xx=True,
    )


@pytest.mark.parametrize(
    "response",
    [
        {"accepted": True, "duplicate": 0, "conflict": 0, "rejected": 0},
        {"accepted": 3, "duplicate": True, "conflict": 0, "rejected": 0},
        {"accepted": 3, "duplicate": 0, "conflict": True, "rejected": 0},
        {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": True},
    ],
)
def test_boolean_receipt_counters_never_acknowledge(response: dict[str, Any]) -> None:
    assert not feedback.upload_receipt_acknowledges_all(
        _receipt(response),
        sent_count=3,
    )


@pytest.mark.parametrize(
    "receipt",
    [
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id=None,
            body_request_id=RECEIPT_REQUEST_ID,
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            include_body_request_id=False,
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            body_request_id=OTHER_REQUEST_ID,
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id=RECEIPT_REQUEST_ID.upper(),
            body_request_id=RECEIPT_REQUEST_ID.upper(),
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id="not-a-uuid",
            body_request_id="not-a-uuid",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id=f" {RECEIPT_REQUEST_ID}",
            body_request_id=f" {RECEIPT_REQUEST_ID}",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id=f"{{{RECEIPT_REQUEST_ID}}}",
            body_request_id=f"{{{RECEIPT_REQUEST_ID}}}",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id="00000000-0000-0000-0000-000000000000",
            body_request_id="00000000-0000-0000-0000-000000000000",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id="72aee12d-7742-44ea-73d9-f056ae5c8ac2",
            body_request_id="72aee12d-7742-44ea-73d9-f056ae5c8ac2",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            request_id="72aee12d-7742-64ea-b3d9-f056ae5c8ac2",
            body_request_id="72aee12d-7742-64ea-b3d9-f056ae5c8ac2",
        ),
        _receipt(
            {"accepted": 3, "duplicate": 0, "conflict": 0, "rejected": 0},
            body_request_id=123,
        ),
    ],
)
def test_receipt_requires_matching_canonical_header_and_body_uuid(
    receipt: feedback.UploadReceipt,
) -> None:
    assert not feedback.upload_receipt_acknowledges_all(receipt, sent_count=3)


@pytest.mark.parametrize("sent_count", [-1, True, False, 1.0, "1", None])
def test_upload_receipt_acknowledgment_validates_sent_count(
    sent_count: Any,
) -> None:
    with pytest.raises(feedback.FeedbackValidationError, match="sent_count"):
        feedback.upload_receipt_acknowledges_all(
            _receipt(None),
            sent_count=sent_count,
        )


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 202,
        request_id: str | None = RECEIPT_REQUEST_ID,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = Message()
        if request_id:
            self.headers["X-Request-ID"] = request_id

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_http_client_posts_json_with_auth_timeout_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeResponse(
            json.dumps(
                {
                    "accepted": 1,
                    "duplicate": 0,
                    "conflict": 0,
                    "rejected": 0,
                    "request_id": RECEIPT_REQUEST_ID,
                }
            ).encode(),
            status=200,
        )

    monkeypatch.setattr(feedback, "_open_feedback_request", fake_urlopen)
    trace = feedback.SessionTrace("anon_upload")
    trace.record_answer(
        _question(),
        selected_letter="A",
        selected_candidate_id="c_a",
        event_id="evt_answer_before_comment",
    )
    event = trace.record_comment(
        _question(),
        category="question_quality",
        text="The wording is ambiguous.",
        event_id="evt_upload",
    )
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/v1/feedback?deployment=dev",
        bearer_token="top-secret-token",
        timeout_seconds=3.25,
        environ={},
    )

    receipt = client.post_event(event)
    request = captured["request"]
    posted = json.loads(request.data.decode("utf-8"))

    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer top-secret-token"
    assert request.get_header("Idempotency-key") == "evt_upload"
    assert request.get_header("Content-type") == "application/json; charset=utf-8"
    assert captured["timeout"] == 3.25
    assert posted["envelope_type"] == "session_trace"
    assert posted["event_count"] == 1
    assert posted["events"][0]["event_id"] == "evt_upload"
    assert posted["events"][0]["sequence"] == 2
    full_trace = trace.to_envelope()
    assert full_trace["events"][1]["sequence"] == 2
    assert posted["events"][0] == full_trace["events"][1]
    assert posted["trace_id"] != full_trace["trace_id"]
    assert receipt.ok and receipt.status_code == 200
    assert receipt.response == {
        "accepted": 1,
        "duplicate": 0,
        "conflict": 0,
        "rejected": 0,
        "request_id": RECEIPT_REQUEST_ID,
    }
    assert receipt.request_id == RECEIPT_REQUEST_ID
    assert receipt.endpoint == "https://collector.example/v1/feedback"
    assert "top-secret-token" not in repr(receipt)


def test_post_trace_rejects_invalid_envelope_before_opening_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = 0

    def fail_if_opened(request: Any, *, timeout: float) -> _FakeResponse:
        nonlocal opened
        del request, timeout
        opened += 1
        raise AssertionError("invalid trace must not open the feedback transport")

    monkeypatch.setattr(feedback, "_open_feedback_request", fail_if_opened)
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/feedback",
        bearer_token="configured-token",
        environ={},
    )

    with pytest.raises(feedback.FeedbackValidationError, match="exact schema"):
        client.post_trace({"envelope_type": "session_trace"})

    assert opened == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (feedback.MAX_SAFE_JSON_INTEGER + 1, "integer-valued"),
        ("\ud800", "surrogate"),
    ],
)
def test_http_client_rejects_noninteroperable_json_before_opening_network(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
    message: str,
) -> None:
    opened = 0

    def fail_if_opened(request: Any, *, timeout: float) -> _FakeResponse:
        nonlocal opened
        del request, timeout
        opened += 1
        raise AssertionError("non-interoperable JSON must not open the transport")

    monkeypatch.setattr(feedback, "_open_feedback_request", fail_if_opened)
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/feedback",
        bearer_token="configured-token",
        environ={},
    )

    with pytest.raises(feedback.FeedbackValidationError, match=message):
        client.post_json({"nested": [{"value": value}]})

    assert opened == 0


def test_http_error_is_diagnostic_but_never_contains_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "bearer-value-that-must-not-leak"

    def fake_urlopen(request: Any, *, timeout: float) -> Any:
        del request, timeout
        headers = Message()
        error = urllib.error.HTTPError(
            "https://collector.example/feedback",
            503,
            "Service unavailable",
            headers,
            io.BytesIO(f'{{"error":"retry; token={secret}"}}'.encode()),
        )
        raise error

    monkeypatch.setattr(feedback, "_open_feedback_request", fake_urlopen)
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/feedback?key=hidden-query",
        bearer_token=secret,
        environ={},
    )

    with pytest.raises(feedback.FeedbackUploadError) as caught:
        client.post_json({"hello": "world"}, idempotency_key="evt_retry")

    message = str(caught.value)
    assert "status=503" in message
    assert "retry" in message
    assert "https://collector.example/feedback" in message
    assert "hidden-query" not in message
    assert secret not in message
    assert secret not in repr(caught.value)


def test_event_id_conflict_is_a_structured_non_retryable_upload_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "conflict-secret-that-must-not-leak"
    request_id = "5a0d1e31-906b-4b33-98e6-f71fd70e8cd2"

    def fake_urlopen(request: Any, *, timeout: float) -> Any:
        del request, timeout
        headers = Message()
        headers["X-Request-ID"] = request_id
        body = {
            "accepted": 0,
            "duplicate": 1,
            "conflict": 1,
            "rejected": 1,
            "request_id": request_id,
            "error": {
                "code": "EVENT_ID_CONFLICT",
                "message": f"batch rejected; token={secret}",
            },
        }
        raise urllib.error.HTTPError(
            "https://collector.example/feedback",
            409,
            "Conflict",
            headers,
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(feedback, "_open_feedback_request", fake_urlopen)
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/feedback?key=hidden-query",
        bearer_token=secret,
        environ={},
    )

    with pytest.raises(feedback.FeedbackUploadConflictError) as caught:
        client.post_json({"hello": "world"}, idempotency_key="evt_conflict")

    error = caught.value
    assert error.status_code == 409
    assert error.error_code == "EVENT_ID_CONFLICT"
    assert error.conflict_count == 1
    assert error.request_id == request_id
    assert error.response["duplicate"] == 1
    assert error.response["rejected"] == 1
    assert "event ID conflicts" in str(error)
    assert "hidden-query" not in str(error)
    assert secret not in str(error)
    assert secret not in repr(error)


def test_network_error_is_wrapped_with_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "network-secret"

    def fake_urlopen(request: Any, *, timeout: float) -> Any:
        del request, timeout
        raise urllib.error.URLError(f"connection reset near {secret}")

    monkeypatch.setattr(feedback, "_open_feedback_request", fake_urlopen)
    client = feedback.FeedbackClient(
        endpoint="https://collector.example/feedback",
        bearer_token=secret,
        environ={},
    )

    with pytest.raises(feedback.FeedbackUploadError) as caught:
        client.post_json({"hello": "world"})

    assert "connection reset" in str(caught.value)
    assert secret not in str(caught.value)


def test_feedback_transport_refuses_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeOpener:
        def open(self, request: Any, *, timeout: float) -> _FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(b"{}")

    def fake_build_opener(*handlers: Any) -> FakeOpener:
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(feedback.urllib.request, "build_opener", fake_build_opener)
    request = urllib.request.Request(
        "https://collector.example/feedback",
        method="POST",
    )
    response = feedback._open_feedback_request(request, timeout=2.5)

    assert isinstance(response, _FakeResponse)
    assert captured["request"] is request
    assert captured["timeout"] == 2.5
    handler = captured["handlers"][0]
    assert isinstance(handler, urllib.request.HTTPRedirectHandler)
    assert (
        handler.redirect_request(
            request,
            response,
            307,
            "Temporary Redirect",
            Message(),
            "https://other.example/feedback",
        )
        is None
    )
