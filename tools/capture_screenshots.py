"""Capture the README screenshots by driving the Streamlit app in a headless browser.

Optional tool, not part of the pipeline. Needs the app running and playwright installed:

    pip install playwright && playwright install chromium
    streamlit run app.py
    python tools/capture_screenshots.py

Scripting these rather than taking them by hand means they can be regenerated whenever
the UI changes, and they always show the same questions.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs"
APP = "http://localhost:8501"


def ask(page, question, wait_for_sources=True):
    box = page.get_by_placeholder("e.g.")
    box.fill("")
    box.fill(question)
    box.press("Enter")
    # Wait for the spinner to disappear, then for the rendered heading -- matching on
    # loose text caught the page mid-generation.
    page.get_by_text("Retrieving and answering").first.wait_for(state="hidden", timeout=300_000)
    if wait_for_sources:
        page.get_by_role("heading", name="Sources").wait_for(timeout=300_000)
    page.wait_for_timeout(3000)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1150},
                                device_scale_factor=2)
        page.goto(APP, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(4000)

        # 1. Answer with a source expanded -- the citation shot.
        ask(page, "How many samples does the CubiCasa5K dataset contain?")
        # Target the source expander by its label -- the sidebar filter is also a
        # <summary> and comes first in the DOM.
        page.locator("summary").filter(has_text="[1]").first.click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "screenshot-answer.png"), full_page=False)
        print("saved screenshot-answer.png")

        # 2. Refusal.
        ask(page, "What is the architecture of the BERT language model?",
            wait_for_sources=False)
        page.screenshot(path=str(OUT / "screenshot-refusal.png"), full_page=False)
        print("saved screenshot-refusal.png")

        browser.close()


if __name__ == "__main__":
    sys.exit(main())
