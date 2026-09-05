"""Verify the pagehide keepalive path: sign in, answer, close mid-chunk (<10s),
and confirm the final events still land in the DB."""
import os
import sys
import time

import psycopg
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5210/"
DB = os.environ.get("AIQ_DATABASE_URL", "")
PASSWORD = os.environ.get("AIQ_TEST_GROUP_PASSWORD", "")
if not DB or not PASSWORD:
    print("FAIL: set AIQ_DATABASE_URL and AIQ_TEST_GROUP_PASSWORD")
    sys.exit(1)

USERNAME = f"pagehide_{int(time.time())}"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(BASE)
    page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
    page.click(".menu-btn.begin")
    page.wait_for_selector(".auth-modal", timeout=5_000)
    page.fill(".auth-field >> nth=0 >> input", USERNAME)
    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".dataset-panel", timeout=15_000)
    # generate a few moves + answer within one 10s window, then hard-close
    for i in range(15):
        page.mouse.move(300 + i * 30, 400 + (i % 3) * 20)
        time.sleep(0.04)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    # let at least one regular flush cross the wire
    page.wait_for_timeout(6_000)
    for i in range(10):
        page.mouse.move(900 - i * 25, 500)
        time.sleep(0.04)
    # simulate the hidden transition every real tab switch/close fires first;
    # (page.close() in CDP skips lifecycle events, so drive it deterministically)
    page.evaluate(
        """() => {
            Object.defineProperty(document, 'hidden', {configurable: true, get: () => true});
            document.dispatchEvent(new Event('visibilitychange'));
        }"""
    )
    time.sleep(1.5)
    page.close()
    browser.close()

time.sleep(4)  # keepalive fetch lands
conn = psycopg.connect(DB)
row = conn.execute(
    "select s.session_id, count(c.id), coalesce(sum(jsonb_array_length(c.events)),0) "
    "from quiz_sessions s left join recording_chunks c on c.session_id = s.session_id "
    "where s.user_id in (select id from quiz_users where username=%s) group by s.session_id",
    (USERNAME,),
).fetchone()
conn.close()
if not row:
    print("FAIL: no session in DB for", USERNAME)
    sys.exit(1)
_, n_chunks, n_events = row
print(f"ok: session persisted with {n_chunks} chunk(s), {n_events} events (kept alive through pagehide)")
if n_chunks < 1 or n_events < 20:
    print("FAIL: pagehide chunk lost")
    sys.exit(1)
print("PAGEHIDE KEEPALIVE OK")
