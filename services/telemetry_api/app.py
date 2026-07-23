"""Thin telemetry ingest API: validate → rate-limit → insert into Postgres."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Any

import psycopg
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ALLOWED_EVENTS = frozenset(
    {
        "session_start",
        "question_view",
        "stage_change",
        "answer_submit",
        "question_leave",
        "audit_feedback",
    }
)

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
TELEMETRY_API_KEY = os.environ.get("TELEMETRY_API_KEY", "").strip()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
RATE_LIMIT_PER_MINUTE = int(os.environ.get("TELEMETRY_RATE_LIMIT_PER_MINUTE", "120"))

app = FastAPI(title="ArchitectureIQ Telemetry", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_hits: dict[str, deque[float]] = defaultdict(deque)


class TelemetryEvent(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    event_type: str
    question_id: str | None = None
    duration_ms: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TelemetryBatch(BaseModel):
    events: list[TelemetryEvent] = Field(min_length=1, max_length=50)


def _authorize(authorization: str | None) -> None:
    if not TELEMETRY_API_KEY:
        raise HTTPException(status_code=503, detail="TELEMETRY_API_KEY is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != TELEMETRY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)


def _normalize(event: TelemetryEvent) -> TelemetryEvent:
    if event.event_type not in ALLOWED_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported event_type: {event.event_type}")
    payload = dict(event.payload)
    payload.setdefault("schema_version", 1)
    if event.duration_ms is not None and event.duration_ms < 0:
        raise HTTPException(status_code=400, detail="duration_ms must be non-negative")
    return event.model_copy(update={"payload": payload})


def _insert_many(events: list[TelemetryEvent]) -> int:
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    rows = [
        (
            e.session_id,
            e.event_type,
            e.question_id,
            psycopg.types.json.Json(e.payload),
            e.duration_ms,
        )
        for e in events
    ]
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    insert into quiz_events
                      (session_id, event_type, question_id, payload, duration_ms)
                    values (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()
    except psycopg.Error as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    return len(rows)


@app.get("/api/health")
def health() -> dict[str, str]:
    status = "ok" if DATABASE_URL and TELEMETRY_API_KEY else "misconfigured"
    return {"status": status}


@app.post("/api/telemetry/events", status_code=202)
def post_events(
    body: TelemetryEvent | TelemetryBatch,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    _rate_limit(request)
    raw = body.events if isinstance(body, TelemetryBatch) else [body]
    events = [_normalize(item) for item in raw]
    inserted = _insert_many(events)
    return {"accepted": inserted}
