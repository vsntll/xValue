"""Shared plumbing for the FBref nodriver scrapers.

FBref sits behind Cloudflare. We drive it with **nodriver** (pure Chrome
DevTools Protocol, no chromedriver binary) because soccerdata's seleniumbase-UC
path uses a patched ``uc_driver.exe`` that crashes on this machine
(docs/fbref_ingestion.md). Needs Python 3.11 here (``py -3.11 ...``).
"""

from __future__ import annotations

import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FBREF_RAW = PROJECT_ROOT / "data" / "raw" / "fbref"

BROWSER = r"C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe"
CHALLENGE_MARKERS = ("Just a moment", "Enable JavaScript and cookies")
NAV_PAUSE = (1.0, 2.5)  # polite random gap between page loads

# league key -> (FBref competition id, URL slug, short code for filenames)
COMPS = {
    "ENG-Premier League": (9, "Premier-League", "ENG1"),
    "ENG-Championship": (10, "Championship", "ENG2"),
    "GER-Bundesliga": (20, "Bundesliga", "GER1"),
    "GER-2. Bundesliga": (33, "2-Bundesliga", "GER2"),
    "ESP-La Liga": (12, "La-Liga", "ESP1"),
    "ESP-La Liga 2": (17, "Segunda-Division", "ESP2"),
}
DEFAULT_LEAGUES = list(COMPS)
DEFAULT_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]


def season4(season: str) -> str:
    """'2023-24' -> '2023-2024' (FBref's URL form)."""
    start, end = season.split("-")
    return f"{start}-{start[:2]}{end}"


def current_season() -> str:
    """European football season in progress today, as 'YYYY-YY'."""
    t = datetime.date.today()
    start = t.year if t.month >= 7 else t.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def slug_to_name(slug: str) -> str:
    return slug.replace("-", " ")


async def get_html(browser, url: str, want_sel: str = "table", tries: int = 4) -> str:
    """Navigate to url (reusing the main tab) and return the DOM once the wanted
    element is present. Cloudflare's challenge auto-clears in a few seconds; we
    wait it out rather than clicking anything. Caller checks the result for a
    CHALLENGE_MARKER / the table it wanted."""
    page = await browser.get(url)
    html = ""
    for attempt in range(1, tries + 1):
        try:
            await page.select(want_sel, timeout=12)
        except Exception:
            pass
        await page.sleep(1.5)
        html = await page.get_content()
        if not any(m in html for m in CHALLENGE_MARKERS):
            return html
        await page.sleep(4 + 2 * attempt)
    return html
