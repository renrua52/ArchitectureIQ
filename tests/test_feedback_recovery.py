"""Recovery-outbox tests for downloaded ArchitectureIQ session traces."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest


REPO = Path(__file__).resolve().parents[1]
INSPECTOR = REPO / "tools" / "question_inspector"
sys.path.insert(0, str(INSPECTOR))

import app as inspector_app  # noqa: E402
import feedback  # noqa: E402
import feedback_outbox  # noqa: E402
import feedback_recovery  # noqa: E402


class _SessionState(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _question() -> dict[str, Any]:
    return {"question_id": "q_recovery", "type": "mixed"}


def _trace(
    count: int,
    *,
    session_id: str = "anon_recovery",
    event_id_prefix: str = "evt_recovery",
    comment_prefix: str = "saved comment",
) -> feedback.SessionTrace:
    trace = feedback.SessionTrace(
        session_id,
        created_at="2026-07-12T00:00:00Z",
    )
    for index in range(count):
        trace.record_comment(
            _question(),
            category="other",
            text=f"{comment_prefix} {index}",
            event_id=f"{event_id_prefix}_{index:04d}",
            occurred_at="2026-07-12T00:00:00Z",
        )
    return trace


def _presentation_trace() -> tuple[
    feedback.SessionTrace,
    tuple[dict[str, Any], dict[str, Any]],
]:
    trace = feedback.SessionTrace(
        "anon_recovery_presented",
        created_at="2026-07-12T01:00:00Z",
    )
    acknowledged = trace.record_question_presented(
        _question(),
        attempt_id="attempt_recovery_presented",
        release_id="release_recovery_presented",
        decision_id="decision_recovery_explore",
        policy_version="surprise_policy_v1",
        mode="explore",
        propensity=0.2,
        source="next",
        position=5,
        event_id="evt_recovery_presented_ack",
        occurred_at="2026-07-12T01:00:01Z",
    )
    quarantined = trace.record_question_presented(
        _question(),
        attempt_id="attempt_recovery_presented",
        release_id="release_recovery_presented",
        decision_id="decision_recovery_random",
        policy_version="random_policy_v1",
        mode="fallback",
        propensity=0.125,
        source="random",
        position=6,
        event_id="evt_recovery_presented_quarantine",
        occurred_at="2026-07-12T01:00:02Z",
    )
    return trace, (acknowledged, quarantined)


def _raw_trace(trace: feedback.SessionTrace) -> bytes:
    return json.dumps(
        trace.to_envelope(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _request_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


def _receipt(
    *,
    accepted: int,
    duplicate: int = 0,
    index: int,
) -> feedback.UploadReceipt:
    request_id = _request_id(index)
    return feedback.UploadReceipt(
        status_code=200,
        endpoint="https://ingest.example/feedback-ingest",
        request_id=request_id,
        response={
            "accepted": accepted,
            "duplicate": duplicate,
            "conflict": 0,
            "rejected": 0,
            "request_id": request_id,
        },
    )


def _conflict(*, index: int) -> feedback.FeedbackUploadConflictError:
    request_id = _request_id(index)
    return feedback.FeedbackUploadConflictError(
        "feedback event ID conflicts with stored content",
        endpoint="https://ingest.example/feedback-ingest",
        status_code=409,
        request_id=request_id,
        response={
            "accepted": 0,
            "duplicate": 0,
            "conflict": 1,
            "rejected": 1,
            "request_id": request_id,
            "error": {"code": "EVENT_ID_CONFLICT"},
        },
    )


class _SuccessClient:
    is_configured = True

    def __init__(self) -> None:
        self.traces: list[Mapping[str, Any]] = []
        self.events: list[Mapping[str, Any]] = []
        self.request_index = 1

    def post_trace(self, trace: Mapping[str, Any]) -> feedback.UploadReceipt:
        self.traces.append(trace)
        receipt = _receipt(
            accepted=len(trace["events"]),
            index=self.request_index,
        )
        self.request_index += 1
        return receipt

    def post_event(self, event: Mapping[str, Any]) -> feedback.UploadReceipt:
        self.events.append(event)
        receipt = _receipt(accepted=1, index=self.request_index)
        self.request_index += 1
        return receipt


def _stub_recovery_ui(
    monkeypatch: pytest.MonkeyPatch,
    uploaded_file: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    buttons: list[dict[str, Any]] = []
    monkeypatch.setattr(inspector_app.st, "divider", lambda: None)
    monkeypatch.setattr(inspector_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector_app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector_app.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector_app.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector_app.st, "error", errors.append)
    monkeypatch.setattr(
        inspector_app.st,
        "file_uploader",
        lambda *args, **kwargs: uploaded_file,
    )

    def button(*args: Any, **kwargs: Any) -> bool:
        buttons.append(dict(kwargs))
        return False

    monkeypatch.setattr(inspector_app.st, "button", button)
    return errors, buttons


def test_recovery_ui_rejects_10_mib_plus_one_before_read_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedUpload:
        size = feedback_recovery.MAX_RECOVERY_FILE_BYTES + 1

        def __init__(self) -> None:
            self.getvalue_called = False

        def getvalue(self) -> bytes:
            self.getvalue_called = True
            raise AssertionError("oversized upload must not be read")

    class NetworkSentinel:
        is_configured = True

        def post_trace(self, trace: Mapping[str, Any]) -> feedback.UploadReceipt:
            del trace
            raise AssertionError("oversized upload must not reach the network")

        def post_event(self, event: Mapping[str, Any]) -> feedback.UploadReceipt:
            del event
            raise AssertionError("oversized upload must not reach the network")

    uploaded = OversizedUpload()
    errors, buttons = _stub_recovery_ui(monkeypatch, uploaded)

    inspector_app._render_recovery_upload(NetworkSentinel())  # type: ignore[arg-type]

    assert uploaded.getvalue_called is False
    assert buttons == []
    assert any(
        "10 MiB" in message and "not read or sent" in message for message in errors
    )


def test_recovery_ui_rejects_malformed_json_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"schema_version":"1.0","schema_version":"1.0"}'

    class InvalidUpload:
        size = len(raw)

        @staticmethod
        def getvalue() -> bytes:
            return raw

    class NetworkSentinel:
        is_configured = True

        def post_trace(self, trace: Mapping[str, Any]) -> feedback.UploadReceipt:
            del trace
            raise AssertionError("malformed upload must not reach the network")

        def post_event(self, event: Mapping[str, Any]) -> feedback.UploadReceipt:
            del event
            raise AssertionError("malformed upload must not reach the network")

    errors, buttons = _stub_recovery_ui(monkeypatch, InvalidUpload())

    inspector_app._render_recovery_upload(NetworkSentinel())  # type: ignore[arg-type]

    assert buttons == []
    assert any("duplicate JSON object key" in message for message in errors)


def test_recovery_ui_reads_a_file_at_the_declared_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExactLimitUpload:
        size = feedback_recovery.MAX_RECOVERY_FILE_BYTES

        def __init__(self) -> None:
            self.getvalue_called = False

        def getvalue(self) -> bytes:
            self.getvalue_called = True
            return b"{}"

    uploaded = ExactLimitUpload()
    errors, buttons = _stub_recovery_ui(monkeypatch, uploaded)

    inspector_app._render_recovery_upload(None)

    assert uploaded.getvalue_called is True
    assert buttons == []
    assert any("invalid saved session trace" in message for message in errors)


def test_recovery_entry_chunks_501_events_as_500_and_1() -> None:
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(_trace(501)))
    client = _SuccessClient()

    result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        recovered,
    )

    assert [len(trace["events"]) for trace in client.traces] == [500, 1]
    assert result.acknowledged_event_ids == tuple(
        event["event_id"] for event in recovered.events
    )
    assert result.remaining_event_ids == ()
    assert result.batch_request_count == 2
    assert result.all_acknowledged is True


def test_download_recovery_preserves_presentations_through_ack_and_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, presentations = _presentation_trace()
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(trace))
    conflict_id = presentations[1]["event_id"]

    class ConflictClient(_SuccessClient):
        def post_trace(self, envelope: Mapping[str, Any]) -> feedback.UploadReceipt:
            self.traces.append(envelope)
            raise _conflict(index=70)

        def post_event(self, event: Mapping[str, Any]) -> feedback.UploadReceipt:
            self.events.append(event)
            if event["event_id"] == conflict_id:
                raise _conflict(index=72)
            return _receipt(accepted=1, index=71)

    client = ConflictClient()
    result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        recovered,
    )

    assert recovered.events == presentations
    assert client.traces[0]["events"] == list(presentations)
    assert client.events == list(presentations)
    assert result.acknowledged_event_ids == ("evt_recovery_presented_ack",)
    assert [item.to_dict() for item in result.quarantined_events] == [
        {
            "event_id": "evt_recovery_presented_quarantine",
            "request_id": _request_id(72),
            "error_code": "EVENT_ID_CONFLICT",
        }
    ]
    expected_policy_fields = [
        {
            "policy_version": "surprise_policy_v1",
            "decision_id": "decision_recovery_explore",
            "mode": "explore",
            "source": "next",
            "position": 5,
            "propensity": 0.2,
        },
        {
            "policy_version": "random_policy_v1",
            "decision_id": "decision_recovery_random",
            "mode": "fallback",
            "source": "random",
            "position": 6,
            "propensity": 0.125,
        },
    ]
    assert [
        {
            field: event["payload"][field]
            for field in (
                "policy_version",
                "decision_id",
                "mode",
                "source",
                "position",
                "propensity",
            )
        }
        for event in recovered.events
    ] == expected_policy_fields

    state = _SessionState(feedback_recovery_outboxes={})
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    inspector_app._apply_recovery_outbox_result(recovered, result)

    acknowledged, quarantined = inspector_app._recovery_outbox_state(recovered)
    assert acknowledged == {"evt_recovery_presented_ack"}
    assert set(quarantined) == {"evt_recovery_presented_quarantine"}
    assert inspector_app._recovery_pending_events(recovered) == ()
    assert [
        {
            field: event["payload"][field]
            for field in (
                "policy_version",
                "decision_id",
                "mode",
                "source",
                "position",
                "propensity",
            )
        }
        for event in recovered.events
    ] == expected_policy_fields


def test_recovery_quarantines_local_body_limit_without_blocking_later_event() -> None:
    trace = feedback.SessionTrace(
        "anon_recovery_large",
        created_at="2026-07-12T00:00:00Z",
    )
    trace.record_comment(
        _question(),
        category="other",
        text="x" * 1_800,
        event_id="evt_recovery_large",
        occurred_at="2026-07-12T00:00:00Z",
    )
    trace.record_comment(
        _question(),
        category="other",
        text="small later event",
        event_id="evt_recovery_small",
        occurred_at="2026-07-12T00:00:01Z",
    )
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(trace))
    client = _SuccessClient()

    result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        recovered,
        max_body_bytes=1_000,
    )

    assert result.pending_event_ids == (
        "evt_recovery_large",
        "evt_recovery_small",
    )
    assert result.acknowledged_event_ids == ("evt_recovery_small",)
    assert [item.to_dict() for item in result.quarantined_events] == [
        {
            "event_id": "evt_recovery_large",
            "request_id": None,
            "error_code": feedback_outbox.LOCAL_BODY_LIMIT_ERROR_CODE,
        }
    ]
    assert result.remaining_event_ids == ()
    assert result.issue is None
    assert result.settled is True
    assert result.all_acknowledged is False
    assert [
        event["event_id"] for envelope in client.traces for event in envelope["events"]
    ] == ["evt_recovery_small"]
    assert client.events == []


def test_recovery_entry_revalidates_content_before_network() -> None:
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(_trace(1)))
    recovered.events[0]["payload"]["text"] = "changed after validation"
    client = _SuccessClient()

    with pytest.raises(feedback_recovery.FeedbackRecoveryError, match="changed"):
        feedback_recovery.upload_recovered_trace(
            client,  # type: ignore[arg-type]
            recovered,
        )

    assert client.traces == []


def test_recovery_entry_isolates_one_409_without_blocking_other_events() -> None:
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(_trace(3)))
    conflict_id = recovered.events[1]["event_id"]

    class ConflictClient(_SuccessClient):
        def post_trace(self, trace: Mapping[str, Any]) -> feedback.UploadReceipt:
            self.traces.append(trace)
            raise _conflict(index=10)

        def post_event(self, event: Mapping[str, Any]) -> feedback.UploadReceipt:
            self.events.append(event)
            if event["event_id"] == conflict_id:
                raise _conflict(index=20)
            return _receipt(accepted=1, index=30 + len(self.events))

    client = ConflictClient()
    result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        recovered,
    )

    assert result.acknowledged_event_ids == (
        recovered.events[0]["event_id"],
        recovered.events[2]["event_id"],
    )
    assert [item.event_id for item in result.quarantined_events] == [conflict_id]
    assert result.remaining_event_ids == ()
    assert result.batch_request_count == 1
    assert result.isolation_request_count == 3
    assert result.settled is True


def test_content_scoped_state_skips_same_trace_but_not_changed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _SessionState(feedback_recovery_outboxes={})
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    original = feedback_recovery.parse_recovered_trace(
        _raw_trace(
            _trace(
                1,
                session_id="anon_same_id",
                event_id_prefix="evt_same_id",
                comment_prefix="original",
            )
        )
    )
    client = _SuccessClient()

    first_result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        original,
    )
    inspector_app._apply_recovery_outbox_result(original, first_result)
    requests_after_first_upload = len(client.traces)

    replay = feedback_recovery.parse_recovered_trace(
        _raw_trace(
            _trace(
                1,
                session_id="anon_same_id",
                event_id_prefix="evt_same_id",
                comment_prefix="original",
            )
        )
    )
    acknowledged, quarantined = inspector_app._recovery_outbox_state(replay)
    replay_result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        replay,
        acknowledged_event_ids=acknowledged,
        quarantined_event_ids=quarantined,
    )

    assert replay.recovery_id == original.recovery_id
    assert replay_result.pending_event_ids == ()
    assert len(client.traces) == requests_after_first_upload

    changed = feedback_recovery.parse_recovered_trace(
        _raw_trace(
            _trace(
                1,
                session_id="anon_same_id",
                event_id_prefix="evt_same_id",
                comment_prefix="changed",
            )
        )
    )
    assert changed.trace_id == original.trace_id
    assert changed.events[0]["event_id"] == original.events[0]["event_id"]
    assert changed.recovery_id != original.recovery_id
    assert [
        event["event_id"] for event in inspector_app._recovery_pending_events(changed)
    ] == [changed.events[0]["event_id"]]

    changed_result = feedback_recovery.upload_recovered_trace(
        client,  # type: ignore[arg-type]
        changed,
    )
    assert changed_result.pending_event_ids == (changed.events[0]["event_id"],)
    assert len(client.traces) == requests_after_first_upload + 1


def test_recovery_state_does_not_pollute_live_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_quarantine = {
        "evt_live_conflict": {
            "event_id": "evt_live_conflict",
            "request_id": _request_id(91),
            "error_code": "EVENT_ID_CONFLICT",
        }
    }
    live_notices = {
        "q_live": {
            "level": "warning",
            "message": "live comment is pending",
            "event_id": "evt_live_pending",
        }
    }
    state = _SessionState(
        feedback_uploaded_event_ids=["evt_live_ack"],
        feedback_quarantined_events=deepcopy(live_quarantine),
        comment_notices=deepcopy(live_notices),
        feedback_recovery_outboxes={},
    )
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(_trace(2)))
    result = feedback_outbox.OutboxUploadResult(
        pending_event_ids=tuple(event["event_id"] for event in recovered.events),
        acknowledged_event_ids=(recovered.events[0]["event_id"],),
        quarantined_events=(
            feedback_outbox.QuarantinedEvent(
                event_id=recovered.events[1]["event_id"],
                request_id=_request_id(92),
                error_code="EVENT_ID_CONFLICT",
            ),
        ),
        remaining_event_ids=(),
        request_ids=(_request_id(92),),
        batch_request_count=1,
        isolation_request_count=2,
    )

    inspector_app._apply_recovery_outbox_result(recovered, result)

    assert state.feedback_uploaded_event_ids == ["evt_live_ack"]
    assert state.feedback_quarantined_events == live_quarantine
    assert state.comment_notices == live_notices
    acknowledged, quarantined = inspector_app._recovery_outbox_state(recovered)
    assert acknowledged == {recovered.events[0]["event_id"]}
    assert set(quarantined) == {recovered.events[1]["event_id"]}


@pytest.mark.parametrize(
    ("endpoint", "bearer_token"),
    [
        (None, "ingest-secret"),
        ("https://ingest.example/feedback-ingest", None),
    ],
)
def test_unconfigured_recovery_cannot_open_network(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
    bearer_token: str | None,
) -> None:
    opened = False

    def fail_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal opened
        opened = True
        raise AssertionError("incomplete feedback configuration must not open network")

    monkeypatch.setattr(feedback, "_open_feedback_request", fail_open)
    recovered = feedback_recovery.parse_recovered_trace(_raw_trace(_trace(1)))
    client = feedback.FeedbackClient(
        endpoint=endpoint,
        bearer_token=bearer_token,
        environ={},
    )

    result = feedback_recovery.upload_recovered_trace(client, recovered)

    assert opened is False
    assert result.acknowledged_event_ids == ()
    assert result.remaining_event_ids == (recovered.events[0]["event_id"],)
    assert result.issue is not None
    assert result.issue.kind == "batch_upload_failed"
    assert result.issue.retryable is False


@pytest.mark.parametrize(
    ("endpoint", "bearer_token"),
    [
        (None, "ingest-secret"),
        ("https://ingest.example/feedback-ingest", None),
    ],
)
def test_recovery_ui_disables_upload_with_incomplete_configuration(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str | None,
    bearer_token: str | None,
) -> None:
    raw = _raw_trace(_trace(1))

    class SavedUpload:
        size = len(raw)

        @staticmethod
        def getvalue() -> bytes:
            return raw

    state = _SessionState(feedback_recovery_outboxes={})
    monkeypatch.setattr(inspector_app.st, "session_state", state)
    errors, buttons = _stub_recovery_ui(monkeypatch, SavedUpload())
    client = feedback.FeedbackClient(
        endpoint=endpoint,
        bearer_token=bearer_token,
        environ={},
    )

    inspector_app._render_recovery_upload(client)

    assert errors == []
    assert len(buttons) == 1
    assert buttons[0]["disabled"] is True


def test_streamlit_recovery_widget_validates_saved_trace_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    for name in (
        feedback.FEEDBACK_ENDPOINT_ENV,
        feedback.FEEDBACK_URL_ENV,
        feedback.FEEDBACK_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    app = AppTest.from_file(str(INSPECTOR / "app.py")).run(timeout=30)
    app.file_uploader(key="recover_session_trace_json").upload(
        "architectureiq-saved-session.json",
        _raw_trace(_trace(1)),
        "application/json",
    ).run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any(
        "Validated trace · 1 event(s) · 1 pending" in value for value in captions
    )
    assert any(
        "Configure both the upload endpoint and Bearer token" in value
        for value in captions
    )
    button = app.button(key="upload_recovered_session_trace")
    assert button.disabled is True
