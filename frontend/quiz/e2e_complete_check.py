"""Completion celebration check: answering all 50 triggers the overlay.

Also verifies the cross-pack restore fetch (list without pack filter).
"""
import json
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

PORT = 5217
BASE = f"http://127.0.0.1:{PORT}/"
PASSWORD = "tccrzrgsy"
USERNAME = f"complete_{int(time.time())}"

server = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(PORT)],
    cwd="dist",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(1.0)

list_responses = []

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        def on_response(resp):
            if "quiz_list_answers" in resp.url:
                try:
                    list_responses.append(len(resp.json()))
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(BASE)
        page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
        page.click(".menu-btn.begin")
        page.wait_for_selector(".auth-modal", timeout=5_000)
        page.fill(".auth-field >> nth=0 >> input", USERNAME)
        page.fill(".auth-field >> nth=1 >> input", PASSWORD)
        page.click(".auth-actions .cta")
        page.wait_for_selector(".dataset-panel", timeout=15_000)

        answered = 0
        for i in range(50):
            see = page.locator("text=See choices")
            if see.count():
                see.first.click()
            page.wait_for_selector(".choice-card", timeout=5_000)
            page.click(".choice-card >> nth=0")
            page.wait_for_selector(".verdict", timeout=5_000)
            answered += 1
            if i < 49:
                page.locator(".top-actions button", has_text="Next").click()
                page.wait_for_timeout(150)
            else:
                break

        print(f"answered {answered} questions")
        page.wait_for_selector(".completion-overlay", timeout=8_000)
        card = page.locator(".completion-card").inner_text()
        assert "All done!" in card, f"overlay text unexpected: {card!r}"
        assert "/ 50" in card or "/50" in card, f"score missing: {card!r}"
        print("ok: completion overlay appeared with score:", " ".join(card.split())[:90])

        # close the overlay, UI still usable
        page.locator(".completion-btn", has_text="Back to questions").click()
        page.wait_for_timeout(300)
        assert page.locator(".completion-overlay").count() == 0
        print("ok: overlay dismissible")

        # cross-pack restore: open the OLD pack link, the fetch must return
        # all 50 answers (no pack filter) and overlapping questions lock.
        page.goto(BASE + "?question_pack=v15-launch50-seed42")
        page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=30_000)
        page.click(".menu-btn.begin")
        page.wait_for_selector(".dataset-panel", timeout=15_000)
        time.sleep(1.5)
        assert list_responses and list_responses[-1] == 50, (
            f"expected restore fetch of 50 records, got {list_responses}"
        )
        print("ok: old-pack link restored all 50 records (cross-pack fetch)")
        browser.close()
finally:
    server.terminate()

print("COMPLETION CHECK OK")
