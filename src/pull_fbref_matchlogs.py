"""Step 1c + step 2 groundwork - FETCH per-team all-competitions match-log pages
from FBref.

Uses **nodriver** (pure Chrome DevTools Protocol, no chromedriver binary). The
seleniumbase-UC path that soccerdata uses relies on a patched ``uc_driver.exe``
that *crashes* on this machine (Windows APPCRASH, faulting module uc_driver.exe,
~2 s after the first navigation - see docs/fbref_ingestion.md). nodriver has no
driver binary to crash and clears FBref's Cloudflare challenge reliably here.

This module only downloads and caches raw HTML. Turning it into a table is
``src/parse_fbref_matchlogs.py`` (no browser - run it any time to see progress).

Flow, per (league, season):
  1. GET the competition "Stats" page, read the 18-24 squad ids + name slugs
     from the ``stats_squads_standard_for`` table.
  2. For each squad x stat type, GET
     ``/en/squads/<id>/<season>/matchlogs/all_comps/<stat>/`` and save it.
Everything is skipped if already on disk, so re-running resumes for free.

Run (Python 3.11 - nodriver needs it here):
    py -3.11 src/pull_fbref_matchlogs.py --stats schedule     # cups + results first
    py -3.11 src/pull_fbref_matchlogs.py                       # then every stat type
    py -3.11 src/pull_fbref_matchlogs.py --leagues "ENG-Premier League" --seasons 2023-24
    py -3.11 src/pull_fbref_matchlogs.py --status              # progress, no scraping

A visible Chrome window opens and drives itself. You can use the machine while it
runs (nodriver doesn't need mouse focus), just don't close that window.

Output:
    data/raw/fbref/pages/<comp>_<season>_<slug>_<stat>.html    raw match-log pages
    data/raw/fbref/pages/_squads_<comp>_<season>.json          cached squad lists
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_ROOT / "data" / "raw" / "fbref" / "pages"

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
STAT_TYPES = ["schedule", "shooting", "keeper", "misc"]

BROWSER = r"C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe"
CHALLENGE_MARKERS = ("Just a moment", "Enable JavaScript and cookies")
NAV_PAUSE = (1.0, 2.5)  # polite random gap between page loads


def _season4(season: str) -> str:
    """'2023-24' -> '2023-2024' (FBref's URL form)."""
    start, end = season.split("-")
    return f"{start}-{start[:2]}{end}"


def _current_season() -> str:
    """European football season in progress today, as 'YYYY-YY'."""
    import datetime
    t = datetime.date.today()
    start = t.year if t.month >= 7 else t.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def _slug_to_name(slug: str) -> str:
    return slug.replace("-", " ")


async def _get_html(browser, url: str, want_sel: str = "table", tries: int = 4) -> str:
    """Navigate to url (reusing the main tab) and return the DOM once the wanted
    element is present. Cloudflare's challenge auto-clears in a few seconds; we
    just wait it out rather than clicking anything."""
    page = await browser.get(url)
    for attempt in range(1, tries + 1):
        try:
            await page.select(want_sel, timeout=12)
        except Exception:
            pass
        await page.sleep(1.5)
        html = await page.get_content()
        if not any(m in html for m in CHALLENGE_MARKERS):
            return html
        await page.sleep(4 + 2 * attempt)  # let the challenge finish
    return html  # caller checks for the table / challenge marker


def _parse_squads(html: str) -> list[tuple[str, str]]:
    """(squad_id, name_slug) from the squad standard-stats table. FBref hides some
    secondary tables inside HTML comments, so search the whole doc uncommented."""
    doc = html.replace("<!--", "").replace("-->", "")
    m = re.search(r'id="stats_squads_standard_for".*?</table>', doc, re.S)
    seg = m.group(0) if m else doc
    # season segment is present on historical pages, absent on the current one
    pairs = re.findall(
        r'/en/squads/([0-9a-f]{8})/(?:\d{4}-\d{4}/)?([A-Za-z0-9\-]+)-Stats', seg)
    return sorted(set(pairs))


async def _squad_list(browser, cid: int, slug: str, season4: str, is_current: bool,
                      cache: Path) -> list[tuple[str, str]]:
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    urls = [f"https://fbref.com/en/comps/{cid}/{season4}/{season4}-{slug}-Stats"]
    if is_current:
        # FBref serves the in-progress season at the season-less URL
        urls.append(f"https://fbref.com/en/comps/{cid}/{slug}-Stats")
    for url in urls:
        html = await _get_html(browser, url, want_sel="#stats_squads_standard_for")
        squads = _parse_squads(html)
        if squads:
            cache.write_text(json.dumps(squads))
            return squads
    return []


async def scrape(leagues: list[str], seasons: list[str], stats: list[str]) -> None:
    import nodriver as uc

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    current = _current_season()
    browser = await uc.start(browser_executable_path=BROWSER, headless=False)
    try:
        for league in leagues:
            cid, slug, code = COMPS[league]
            for season in seasons:
                s4 = _season4(season)
                sq_cache = PAGES_DIR / f"_squads_{code}_{season}.json"
                squads = await _squad_list(browser, cid, slug, s4,
                                           season == current, sq_cache)
                if not squads:
                    print(f"{code} {season}: no squads found (season may not exist yet)")
                    continue
                print(f"{code} {season}: {len(squads)} squads")
                for sid, name_slug in squads:
                    for stat in stats:
                        out = PAGES_DIR / f"{code}_{season}_{name_slug}_{stat}.html"
                        if out.exists() and out.stat().st_size > 50_000:
                            continue
                        url = (f"https://fbref.com/en/squads/{sid}/{s4}"
                               f"/matchlogs/all_comps/{stat}/")
                        html = await _get_html(browser, url, want_sel="#matchlogs_for")
                        if any(m in html for m in CHALLENGE_MARKERS):
                            print(f"  !! {name_slug} {stat}: stuck on challenge, skipping")
                            continue
                        if 'id="matchlogs_for"' not in html and stat != "schedule":
                            # some clubs have no keeper/shooting log for a season
                            out.write_text(html, encoding="utf-8")
                            print(f"  -  {name_slug} {stat}: no table (saved anyway)")
                        else:
                            out.write_text(html, encoding="utf-8")
                            print(f"  ok {name_slug} {stat}")
                        await asyncio.sleep(random.uniform(*NAV_PAUSE))
    finally:
        browser.stop()


def status() -> None:
    if not PAGES_DIR.exists():
        print("no pages yet")
        return
    pages = list(PAGES_DIR.glob("*.html"))
    by_stat: dict[str, int] = {}
    by_ls: dict[str, int] = {}
    for p in pages:
        parts = p.stem.split("_")
        stat = parts[-1]
        by_stat[stat] = by_stat.get(stat, 0) + 1
        by_ls["_".join(parts[:2])] = by_ls.get("_".join(parts[:2]), 0) + 1
    print(f"{len(pages)} pages cached")
    print("  by stat:  ", dict(sorted(by_stat.items())))
    print("  by league-season:")
    for k, v in sorted(by_ls.items()):
        print(f"    {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES, choices=list(COMPS))
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--stats", nargs="+", default=STAT_TYPES, choices=STAT_TYPES)
    ap.add_argument("--status", action="store_true", help="print cache summary and exit")
    args = ap.parse_args()

    if args.status:
        status()
        return

    import nodriver as uc
    uc.loop().run_until_complete(scrape(args.leagues, args.seasons, args.stats))
    print("\nnow build the table:  py -3.11 src/parse_fbref_matchlogs.py")


if __name__ == "__main__":
    main()
