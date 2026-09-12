"""Headless E2E for quiz recording/user-system/replay against local dist build.

Prereq: static server on 127.0.0.1:5210 serving frontend/quiz/dist with the
pack copied into dist/data/packs/.

Verifies: auth gate -> register -> record -> upload to Supabase -> export ->
replay player renders recorded events.
"""

import json
import os
import sys
import tempfile
import time
import pathlib

import psycopg
from playwright.sync_api import sync_playwright

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:5210/")
DB = os.environ.get("AIQ_DATABASE_URL", "")
USERNAME = f"e2e_bot_{int(time.time())}"
PASSWORD = os.environ.get("AIQ_TEST_GROUP_PASSWORD", "")
if not DB or not PASSWORD:
    print("FAIL: set AIQ_DATABASE_URL and AIQ_TEST_GROUP_PASSWORD")
    sys.exit(1)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(BASE)
    page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
    print("ok: pack loaded")

    # 1. Begin -> auth gate appears
    page.click(".menu-btn.begin")
    page.wait_for_selector(".auth-modal", timeout=5_000)
    print("ok: auth gate shown")

    # wrong password first
    page.fill(".auth-field >> nth=0 >> input", USERNAME)
    page.fill(".auth-field >> nth=1 >> input", "wrong-password")
    page.click(".auth-actions .cta")
    page.wait_for_selector(".auth-error", timeout=8_000)
    print("ok: wrong password rejected")

    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".stage-kicker", timeout=15_000)
    print("ok: signed in, quiz screen reached")

    # 2. generate pointer trajectory
    for i in range(40):
        page.mouse.move(200 + i * 20, 300 + (i % 5) * 40)
        time.sleep(0.05)
    page.click(".cta")  # See choices ->
    page.wait_for_selector(".choice-card", timeout=5_000)
    for i in range(30):
        page.mouse.move(900 - i * 15, 500 - (i % 4) * 30)
        time.sleep(0.05)
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    print("ok: answered question 1")

    # next question, answer again (single-page: several .cta, match by text)
    page.click("text=Next question")
    page.wait_for_timeout(1_000)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    page.click(".choice-card >> nth=1")
    page.wait_for_selector(".verdict", timeout=5_000)
    print("ok: answered question 2")

    # 3. wait for recorder flush (10s interval) then check DB
    page.wait_for_timeout(11_500)

    conn = psycopg.connect(DB)
    rows = conn.execute(
        "select session_id, pack, score_correct, score_total from quiz_sessions order by started_at desc limit 3"
    ).fetchall()
    sess = [r for r in rows if r[1] == "v15-launch50-seed42"]
    if not sess:
        fail(f"no session row for pack in DB (rows={rows})")
    session_id, _, sc, st = sess[0]
    print(f"ok: session row {session_id} score {sc}/{st}")
    if st != 2:
        fail(f"expected score_total=2, got {st}")

    chunks = conn.execute(
        "select seq, jsonb_array_length(events) from recording_chunks where session_id=%s order by seq",
        (session_id,),
    ).fetchall()
    if not chunks:
        fail("no recording chunks in DB")
    total_events = sum(c[1] for c in chunks)
    print(f"ok: {len(chunks)} chunk(s), {total_events} events")

    events = []
    for (seq,) in conn.execute(
        "select seq from recording_chunks where session_id=%s order by seq", (session_id,)
    ).fetchall():
        pass
    raw = conn.execute(
        "select events from recording_chunks where session_id=%s order by seq", (session_id,)
    ).fetchall()
    for (evs,) in raw:
        events.extend(evs)
    types = {e[1] for e in events}
    for needed in ("m", "c", "q", "g", "a"):
        if needed not in types:
            fail(f"event type {needed!r} missing; have {types}")
    print(f"ok: event types present: {sorted(types)}")

    user = conn.execute(
        "select u.username from quiz_users u join quiz_sessions s on s.user_id=u.id where s.session_id=%s",
        (session_id,),
    ).fetchone()
    if not user or user[0] != USERNAME:
        fail(f"session not linked to {USERNAME} (got {user})")
    print(f"ok: session owned by {USERNAME}")
    conn.close()

    # 4. export contains recording
    with page.expect_download() as dl_info:
        page.click("text=Export")
    download = dl_info.value
    tmp = pathlib.Path(tempfile.mkdtemp()) / "session.json"
    download.save_as(tmp)
    exported = json.loads(tmp.read_text())
    rec = exported.get("recording")
    if not rec or rec.get("schema_version") != 1 or not rec.get("events"):
        fail("export missing recording")
    print(f"ok: export has recording with {len(rec['events'])} events, meta user={rec['meta'].get('username')}")

    # 5. replay the exported file
    page.click(".brand-btn")  # back home
    page.wait_for_selector(".menu-btn.begin", timeout=5_000)
    page.set_input_files("input[type=file]", str(tmp))
    page.wait_for_selector(".replay-viewport", timeout=5_000)
    # cursor only exists once the playhead passes the first pointer event
    page.locator(".replay-transport input[type=range]").fill("2000")
    page.wait_for_selector(".replay-cursor", timeout=5_000)
    n_log = page.locator(".replay-log-list button").count()
    if n_log < 4:
        fail(f"replay log too short: {n_log}")
    page.click(".replay-transport .cta")  # play
    page.wait_for_timeout(1_200)
    clock = page.locator(".replay-clock").inner_text()
    print(f"ok: replay renders (log={n_log} events, clock={clock})")

    browser.close()

print("ALL E2E CHECKS PASSED")
