"""Deep-link entry must now go through sign-in + recorder: answer via ?q=,
export, verify the export contains a real recording, replay it."""
import json
import os
import sys
import tempfile
import time
import pathlib

import psycopg
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5210/"
DB = os.environ.get("AIQ_DATABASE_URL", "")
PASSWORD = os.environ.get("AIQ_TEST_GROUP_PASSWORD", "")
if not DB or not PASSWORD:
    print("FAIL: set AIQ_DATABASE_URL and AIQ_TEST_GROUP_PASSWORD")
    sys.exit(1)

USERNAME = f"deeplink_{int(time.time())}"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    # deep link straight into a question
    page.goto(BASE + "?q=q_134324")
    page.wait_for_selector(".auth-modal", timeout=60_000)
    print("ok: deep link now requires sign-in")
    page.fill(".auth-field >> nth=0 >> input", USERNAME)
    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".dataset-panel", timeout=20_000)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    for i in range(20):
        page.mouse.move(300 + i * 25, 400)
        time.sleep(0.04)
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    page.wait_for_timeout(6_000)  # let a flush land

    with page.expect_download() as dl_info:
        page.click("text=Export")
    tmp = pathlib.Path(tempfile.mkdtemp()) / "s.json"
    dl_info.value.save_as(tmp)
    exported = json.loads(tmp.read_text())
    rec = exported.get("recording")
    assert rec and rec.get("schema_version") == 1 and rec.get("events"), "export still lacks recording!"
    types = {}
    for e in rec["events"]:
        types[e[1]] = types.get(e[1], 0) + 1
    assert "a" in types and "m" in types, f"recording missing key events: {types}"
    print(f"ok: deep-link export has recording ({len(rec['events'])} events, types {types})")

    # replay it
    page.click(".brand-btn")
    page.wait_for_selector(".menu-btn.begin", timeout=5_000)
    page.set_input_files("input[type=file]", str(tmp))
    page.wait_for_selector(".replay-viewport", timeout=5_000)
    page.locator(".replay-transport input[type=range]").fill("2000")
    page.wait_for_selector(".replay-cursor", timeout=5_000)
    print("ok: replay renders from deep-link export")
    browser.close()

conn = psycopg.connect(DB)
n = conn.execute(
    "select count(*) from recording_chunks c join quiz_sessions s on s.session_id=c.session_id "
    "join quiz_users u on u.id=s.user_id where u.username=%s",
    (USERNAME,),
).fetchone()[0]
conn.close()
assert n >= 1, "no chunks in DB"
print(f"ok: {n} chunk(s) auto-uploaded to DB")
print("DEEPLINK RECORDING OK")
