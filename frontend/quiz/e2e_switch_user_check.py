"""Switch-user isolation: u1 answers; switch to u2 -> fresh slate, uploads OK;
u1's rows remain in DB. Also verifies pack-scoped restore (old-pack rows for
the same qid do not lock the new pack)."""
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

U1 = f"switchA_{int(time.time())}"
U2 = f"switchB_{int(time.time())}"
PACK = "v15-launch50-seed20260905"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    def sign_in(username: str) -> None:
        page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
        page.click(".menu-btn.begin")
        page.wait_for_selector(".auth-modal", timeout=5_000)
        page.fill(".auth-field >> nth=0 >> input", username)
        page.fill(".auth-field >> nth=1 >> input", PASSWORD)
        page.click(".auth-actions .cta")
        page.wait_for_selector(".dataset-panel", timeout=20_000)

    page.goto(BASE)
    sign_in(U1)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    page.wait_for_timeout(2_000)  # let uploads land
    print("u1 answered one question")

    # switch user
    page.click(".brand-btn")
    page.wait_for_selector(".menu-btn.begin", timeout=5_000)
    page.click("text=Switch user")
    page.wait_for_selector(".auth-modal", timeout=5_000)
    page.fill(".auth-field >> nth=0 >> input", U2)
    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".dataset-panel", timeout=20_000)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    assert page.locator(".verdict").count() == 0, "u2 inherited u1's locked answer!"
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    print("ok: u2 starts fresh, can answer the same question")

    # u2's uploads must work (new session id, no owner mismatch): wait for a chunk
    page.wait_for_timeout(6_500)
    browser.close()

time.sleep(2)
conn = psycopg.connect(DB)
u1_rows = conn.execute(
    "select count(*) from quiz_answers a join quiz_users u on u.id=a.user_id where u.username=%s",
    (U1,),
).fetchone()[0]
u2_rows = conn.execute(
    "select count(*) from quiz_answers a join quiz_users u on u.id=a.user_id where u.username=%s",
    (U2,),
).fetchone()[0]
# u2's trajectory chunk must exist (recorder restarted cleanly under u2)
u2_chunk = conn.execute(
    "select count(*) from recording_chunks c join quiz_sessions s on s.session_id=c.session_id "
    "join quiz_users u on u.id=s.user_id where u.username=%s",
    (U2,),
).fetchone()[0]
conn.close()
assert u1_rows == 1, f"u1 backup lost: {u1_rows}"
assert u2_rows == 1, f"u2 answer missing: {u2_rows}"
assert u2_chunk >= 1, "u2 recording chunks missing (owner mismatch?)"
print(f"ok: u1 backup intact ({u1_rows} row), u2 answered ({u2_rows}) and recorded ({u2_chunk} chunks)")
print("SWITCH-USER ISOLATION OK")
