"""Display check for xor/spiral classification questions (deep link ?q=)."""
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5210/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    for qid, expect in [("q_134324", "xor"), ("q_04316e", "spiral")]:
        page.goto(f"{BASE}?q={qid}")
        page.wait_for_selector(".task-description", timeout=60_000)
        summary = page.locator(".task-summary").inner_text()
        print(f"[{qid}] task summary: {summary[:180]}...")
        assert expect.upper() in summary or expect in summary.lower(), f"wrong summary: {summary}"
        assert "one-dimensional" not in summary, "regression template leaked into classification"
        rule = page.locator(".attr-list div", has_text="Label rule").inner_text()
        print(f"[{qid}] label rule row: {rule.replace(chr(10), ' ')}")
        assert "score" in rule or "arm" in rule, "label rule missing"
        if expect == "xor":
            balance = page.locator(".attr-list div", has_text="Class balance").inner_text()
            print(f"[{qid}] balance row: {balance.replace(chr(10), ' ')}")
        else:
            # spiral: arms are 50/50 by construction, no calibration row expected
            assert page.locator(".attr-list div", has_text="Class balance").count() == 0
            print(f"[{qid}] no balance row (spiral is 50/50 by construction)")
    browser.close()
print("XOR/SPIRAL DISPLAY OK")
