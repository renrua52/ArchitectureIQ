"""Persistence check: answers survive refresh, locked against re-answering."""
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

USERNAME = f"persist_{int(time.time())}"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(BASE)
    page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
    page.click(".menu-btn.begin")
    page.wait_for_selector(".auth-modal", timeout=5_000)
    page.fill(".auth-field >> nth=0 >> input", USERNAME)
    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".dataset-panel", timeout=15_000)
    qid = page.evaluate("location.search")  # not used; read from provenance instead
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    page.click(".choice-card >> nth=0")
    page.wait_for_selector(".verdict", timeout=5_000)
    verdict_before = page.locator(".verdict").inner_text()
    # give the server-side answer record time to land before reloading
    page.wait_for_timeout(2_500)
    # which question are we on? read the counter "n / 50"
    counter = page.locator(".quiz-count, .topnav").first.inner_text()
    print("answered once; verdict:", verdict_before.strip()[:60])

    # refresh mid-quiz: local results are gone, server must restore them
    page.reload()
    page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=30_000)
    page.click(".menu-btn.begin")
    page.wait_for_selector(".dataset-panel", timeout=15_000)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)
    verdict_after = page.locator(".verdict").inner_text()
    assert verdict_after.strip() == verdict_before.strip(), (
        f"answer not restored after refresh: {verdict_after!r} vs {verdict_before!r}"
    )
    print("ok: refresh restored locked answer with same verdict")

    # clicking a choice card must do nothing (still one verdict, no change)
    page.click(".choice-card >> nth=1")
    page.wait_for_timeout(500)
    assert page.locator(".verdict").count() == 1, "re-answer went through!"
    print("ok: re-answering blocked")
    browser.close()

time.sleep(2)
conn = psycopg.connect(DB)
n, attempts = conn.execute(
    "select count(*), max(attempts) from quiz_answers a join quiz_users u on u.id=a.user_id where u.username=%s",
    (USERNAME,),
).fetchone()
conn.close()
assert n == 1 and attempts == 1, f"expected 1 answer row with 1 attempt, got {n}/{attempts}"
print(f"ok: DB has exactly 1 answer row for {USERNAME} (attempts=1)")
print("PERSISTENCE CHECK OK")
