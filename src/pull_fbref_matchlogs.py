"""Step 1c + step 2 groundwork - FETCH per-team all-competitions match-log pages
from FBref via soccerdata.

This module only *downloads and caches* the raw HTML. Turning it into a table is
``src/parse_fbref_matchlogs.py`` (browser-free — run it any time to see progress).
Splitting fetch from parse matters here because soccerdata drives FBref through
seleniumbase-UC + Cloudflare, whose browser session dies often; we don't want a
parse quirk to look like a scrape failure, or a scrape failure to lose parsed
work.

FBref's team "Scores & Fixtures" page scoped to all competitions
(``/squads/<id>/<season>/matchlogs/all_comps/<stat>/``) lists every fixture a
team played — league, domestic cup, league cup, European, super cup, friendly —
each tagged with a ``Comp`` column. ``soccerdata.FBref.read_team_match_stats()``
targets that URL and caches one HTML page per team per stat type.

Cache / resume:
* ``SOCCERDATA_DIR`` -> ``data/soccerdata`` — every byte stays in the project tree.
* Per (stat, league, season) chunking. A chunk that finishes without the browser
  dying is marked ``done`` in ``data/raw/fbref/scrape_progress.json`` and skipped
  next run. A chunk that dies part-way is left un-done; re-running resumes, and
  soccerdata itself skips pages already on disk. **Expect to run this several
  times** — the UC browser session is not reliable for long runs.

The scrape needs a real, VISIBLE browser (headless can't solve a fresh Cloudflare
challenge). Run it in your own terminal, not unattended in the background, and
keep hands off the mouse while the Chrome window is up.

Run:
    python src/pull_fbref_matchlogs.py --stats schedule      # cups + results first
    python src/pull_fbref_matchlogs.py                       # then all stat types
    python src/pull_fbref_matchlogs.py --leagues "ENG-Premier League" --seasons 2023-24
    python src/parse_fbref_matchlogs.py                      # build the CSV from cache

Output:
    data/soccerdata/data/FBref/matchlogs_<Team>_<code>_<stat>.html   raw pages
    data/raw/fbref/scrape_progress.json                              resume state
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep every scraped byte inside the project (see the "data stays in project"
# rule). Must be set before soccerdata is imported.
os.environ.setdefault("SOCCERDATA_DIR", str(PROJECT_ROOT / "data" / "soccerdata"))
# Cached pages never expire — this is an archive pull, not a refresh of volatile
# data. Re-scrape a season by deleting its cached pages.
os.environ.setdefault("SOCCERDATA_MAXAGE", "3650")

DEFAULT_BROWSER = Path("C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe")

# Top two tiers of each country. Second-tier keys resolve via the custom league
# dict at data/soccerdata/config/league_dict.json.
DEFAULT_LEAGUES = [
    "ENG-Premier League",
    "ENG-Championship",
    "GER-Bundesliga",
    "GER-2. Bundesliga",
    "ESP-La Liga",
    "ESP-La Liga 2",
]
DEFAULT_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]

# schedule = Comp/Round/Venue/Result/GF/GA/Opponent/Poss/xG/xGA  (all comps)
# shooting = Sh/SoT/Dist/FK/PK/PKatt                             (league only)
# keeper   = GK saves / PSxG / etc.                              (league only)
# misc     = CrdY/CrdR/Fls/Fld/Off/Crs/Int/OG/PKwon/PKcon        (league only)
STAT_TYPES = ["schedule", "shooting", "keeper", "misc"]

CACHE_DIR = PROJECT_ROOT / "data" / "soccerdata" / "data" / "FBref"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fbref"
PROGRESS_PATH = RAW_DIR / "scrape_progress.json"


def _season_code(season: str) -> str:
    """'2023-24' -> '2324' (soccerdata's cache-file season code)."""
    start, end = season.split("-")
    return start[2:] + end


def _cached_pages(season: str, stat: str) -> int:
    code = _season_code(season)
    return len(list(CACHE_DIR.glob(f"matchlogs_*_{code}_{stat}.html")))


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, sort_keys=True))


def pull(leagues: list[str], seasons: list[str], stats: list[str],
         browser: str | None, headless: bool, retries: int) -> None:
    import soccerdata as sd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()

    for stat in stats:
        for league in leagues:
            for season in seasons:
                key = f"{stat}|{league}|{season}"
                if progress.get(key, {}).get("status") == "done":
                    print(f"  skip (done): {key}")
                    continue

                before = _cached_pages(season, stat)
                for attempt in range(1, retries + 1):
                    try:
                        fb = sd.FBref(
                            leagues=[league],
                            seasons=[season],
                            path_to_browser=browser,
                            headless=headless,
                        )
                        # Fetches + caches one HTML page per team. We ignore the
                        # parsed return value — parse_fbref_matchlogs.py owns that.
                        fb.read_team_match_stats(stat_type=stat)
                        after = _cached_pages(season, stat)
                        progress[key] = {"status": "done", "pages": after}
                        _save_progress(progress)
                        print(f"  done: {key}  ({after} pages cached)")
                        break
                    except Exception as exc:  # noqa: BLE001
                        after = _cached_pages(season, stat)
                        gained = after - before
                        progress[key] = {"status": "partial", "pages": after,
                                         "last_error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                        _save_progress(progress)
                        print(f"  FAIL {attempt}/{retries}: {key} "
                              f"(+{gained} pages this attempt, {after} total) "
                              f"-- {type(exc).__name__}")
                        before = after
                        time.sleep(5 * attempt)
                else:
                    print(f"  moving on from {key}; re-run to keep going")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--stats", nargs="+", default=STAT_TYPES, choices=STAT_TYPES)
    ap.add_argument("--browser", default=str(DEFAULT_BROWSER),
                    help="path to chrome.exe (needs a real browser for Cloudflare)")
    ap.add_argument("--headless", action="store_true",
                    help="run headless (WARNING: cannot solve a fresh Cloudflare challenge)")
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per (stat, league, season) chunk before moving on")
    ap.add_argument("--status", action="store_true",
                    help="print scrape_progress.json summary and exit")
    args = ap.parse_args()

    if args.status:
        prog = _load_progress()
        if not prog:
            print("no progress yet")
            return
        done = sum(1 for v in prog.values() if v.get("status") == "done")
        print(f"{done}/{len(prog)} chunks done, "
              f"{sum(v.get('pages', 0) for v in prog.values())} pages cached total")
        for k, v in sorted(prog.items()):
            print(f"  {v['status']:>7}  {v.get('pages', 0):>3}p  {k}")
        return

    browser = args.browser if args.browser and Path(args.browser).exists() else None
    if browser is None:
        print(f"WARNING: browser not found at {args.browser} — letting seleniumbase locate one")
    pull(args.leagues, args.seasons, args.stats, browser, args.headless, args.retries)

    print("\nnow build the table:  python src/parse_fbref_matchlogs.py")


if __name__ == "__main__":
    main()
