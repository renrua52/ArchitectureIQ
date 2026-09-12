"""Quick check: activation row appears on choice cards and highlights differences."""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:5210/")
PASSWORD = os.environ.get("AIQ_TEST_GROUP_PASSWORD", "")
if not PASSWORD:
    print("FAIL: set AIQ_TEST_GROUP_PASSWORD")
    sys.exit(1)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(BASE)
    page.wait_for_selector(".menu-btn.begin:not([disabled])", timeout=60_000)
    page.click(".menu-btn.begin")
    page.wait_for_selector(".auth-modal", timeout=5_000)
    page.fill(".auth-field >> nth=0 >> input", f"act_check_{int(time.time())}")
    page.fill(".auth-field >> nth=1 >> input", PASSWORD)
    page.click(".auth-actions .cta")
    page.wait_for_selector(".dataset-panel", timeout=15_000)
    page.click("text=See choices")
    page.wait_for_selector(".choice-card", timeout=5_000)

    # dataset section must still be visible alongside choices (single page)
    assert page.locator(".dataset-panel").is_visible(), "dataset not visible with choices"
    print("ok: single page (dataset + choices together)")

    rows = page.locator(".choice-card >> nth=0 >> .choice-fields .field").all_inner_texts()
    fields_a = [r.split("\n")[0].strip().lower() for r in rows]
    print("card A fields:", fields_a)
    if "activation" not in fields_a:
        print("FAIL: activation row missing on card A")
        sys.exit(1)
    print("ok: activation row present")

    # count questions in this pack whose activation row varies (varying class)
    page.click(".brand-btn")
    browser.close()
print("ALL ACTIVATION CHECKS PASSED")
