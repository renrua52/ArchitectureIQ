#!/usr/bin/env python3
"""Dependency-light smoke checks for the optional telemetry API.

The production API needs psycopg and a database.  These checks intentionally
exercise validation/auth/rate-limit boundaries without opening a database or
sending any event externally.
"""

from __future__ import annotations

import os
import sys
import types


def _install_psycopg_stub() -> None:
    if "psycopg" in sys.modules:
        return
    json_mod = types.SimpleNamespace(Json=lambda value: value)
    stub = types.ModuleType("psycopg")
    stub.Error = Exception
    stub.types = types.SimpleNamespace(json=json_mod)
    sys.modules["psycopg"] = stub


def main() -> int:
    os.environ.setdefault("DATABASE_URL", "postgresql://smoke.invalid/db")
    os.environ.setdefault("TELEMETRY_API_KEY", "smoke-key")
    _install_psycopg_stub()
    from fastapi import HTTPException
    from services.telemetry_api import app as module

    assert module.health()["status"] == "ok"
    module._authorize("Bearer smoke-key")
    try:
        module._authorize("Bearer wrong")
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("invalid bearer token was accepted")

    event = module._normalize(
        module.TelemetryEvent(session_id="session_smoke", event_type="question_view")
    )
    assert event.payload["schema_version"] == 1
    feedback = module._normalize(
        module.TelemetryEvent(
            session_id="session_smoke",
            event_type="audit_feedback",
            question_id="q_smoke",
            payload={"confidence": 4, "decision": "keep"},
        )
    )
    assert feedback.payload["decision"] == "keep"
    try:
        module._normalize(
            module.TelemetryEvent(session_id="session_smoke", event_type="unknown")
        )
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("unknown event type was accepted")
    print("telemetry API smoke passed: health/auth/event validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
