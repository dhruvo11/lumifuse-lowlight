"""Capture screenshots of the Streamlit app for the report (report/figures/app_*.png).
Start the app first:  streamlit run webapp/streamlit_app.py --server.port 8502"""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
os.makedirs(OUT, exist_ok=True)


def pick(page, example, method):
    page.locator('div[data-testid="stSelectbox"]').first.click()
    page.locator('li[role="option"]', has_text=example).first.click()
    page.wait_for_timeout(1500)
    page.locator("button", has_text=method).first.click()
    page.wait_for_timeout(800)
    page.locator("button", has_text="Enhance").first.click()


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1360, "height": 1000}, device_scale_factor=2)
    page = ctx.new_page()
    page.goto(URL)
    page.wait_for_selector("text=See in the dark", timeout=60000)
    page.wait_for_timeout(2500)

    # 1. compare-all on an extremely dark synthetic input
    pick(page, "Synthetic — extremely dark", "Compare all")
    page.wait_for_selector("text=All four methods", timeout=120000)
    page.wait_for_timeout(2500)
    page.locator('button[role="tab"]', has_text="Outputs").first.click()
    page.wait_for_timeout(2000)
    page.screenshot(path=os.path.join(OUT, "app_compare.png"), full_page=True)
    print("app_compare")

    # 2. hybrid on a real, very dark photo + pipeline stages
    pick(page, "Monument at night", "Hybrid")
    page.wait_for_selector("text=Stage timing", timeout=120000)
    page.wait_for_timeout(2500)
    page.locator('button[role="tab"]', has_text="Pipeline stages").first.click()
    page.wait_for_timeout(2000)
    page.screenshot(path=os.path.join(OUT, "app_hybrid_dark.png"), full_page=True)
    print("app_hybrid_dark")

    # 3. backlit real photo (global trust, highlights)
    pick(page, "Backlit building", "Hybrid")
    page.wait_for_selector("text=Stage timing", timeout=120000)
    page.wait_for_timeout(2500)
    page.screenshot(path=os.path.join(OUT, "app_backlit.png"), full_page=True)
    print("app_backlit")

    # 4. error state: a non-image file
    page.goto(URL)
    page.wait_for_selector("text=See in the dark", timeout=60000)
    page.wait_for_timeout(2000)
    page.locator('input[type="file"]').set_input_files(
        files=[{"name": "not_a_photo.jpg", "mimeType": "image/jpeg", "buffer": b"this is not an image"}])
    page.wait_for_selector("text=couldn't be read", timeout=60000)
    page.wait_for_timeout(1500)
    page.screenshot(path=os.path.join(OUT, "app_error.png"), clip={"x": 0, "y": 0, "width": 1360, "height": 900})
    print("app_error")

    # 5. phone-width layout
    mctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
    mp = mctx.new_page()
    mp.goto(URL)
    mp.wait_for_selector("text=See in the dark", timeout=60000)
    mp.wait_for_timeout(3000)
    mp.screenshot(path=os.path.join(OUT, "app_mobile.png"))
    print("app_mobile")
    b.close()
