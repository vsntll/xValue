"""Step 2 (historical) - FETCH FBref competition player season-stats pages.

Each page lists **every player in a league-season** for one stat category, so the
whole job is 6 leagues x 7 seasons x ~11 categories ~= 460 pages (vs. thousands
of per-player pages). Downloads raw HTML only; ``src/parse_fbref_player_stats.py``
turns it into the table.

Categories -> (FBref URL segment, table id):
    standard      stats          stats_standard
    shooting      shooting       stats_shooting
    passing       passing        stats_passing
    passing_types passing_types  stats_passing_types
    gca           gca            stats_gca
    defense       defense        stats_defense
    possession    possession     stats_possession
    playing_time  playingtime    stats_playing_time
    misc          misc           stats_misc
    keeper        keepers        stats_keeper
    keeper_adv    keepersadv     stats_keeper_adv

Run (Python 3.11):
    py -3.11 src/pull_fbref_player_stats.py
    py -3.11 src/pull_fbref_player_stats.py --leagues "ENG-Premier League" --seasons 2023-24
    py -3.11 src/pull_fbref_player_stats.py --status

Output:
    data/raw/fbref/player_stats/<COMP>_<season>_<category>.html
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re

from fbref_common import (
    CHALLENGE_MARKERS, COMPS, DEFAULT_LEAGUES, DEFAULT_SEASONS, FBREF_RAW,
    BROWSER, NAV_PAUSE, current_season, get_html, season4,
)

STATS_DIR = FBREF_RAW / "player_stats"

# category -> (URL segment, table id)
CATEGORIES = {
    "standard": ("stats", "stats_standard"),
    "shooting": ("shooting", "stats_shooting"),
    "passing": ("passing", "stats_passing"),
    "passing_types": ("passing_types", "stats_passing_types"),
    "gca": ("gca", "stats_gca"),
    "defense": ("defense", "stats_defense"),
    "possession": ("possession", "stats_possession"),
    "playing_time": ("playingtime", "stats_playing_time"),
    "misc": ("misc", "stats_misc"),
    "keeper": ("keepers", "stats_keeper"),
    "keeper_adv": ("keepersadv", "stats_keeper_adv"),
}


def _url(cid: int, s4: str, slug: str, segment: str, is_current: bool) -> str:
    if is_current:
        return f"https://fbref.com/en/comps/{cid}/{segment}/{slug}-Stats"
    return f"https://fbref.com/en/comps/{cid}/{s4}/{segment}/{s4}-{slug}-Stats"


_ID_STATS_JS = ("ranker player nationality position team age birth_year "
                "minutes_90s games matches").split()


def _settle_js(table_id: str) -> str:
    """Truthy once the table's *stat* cells (not the identity columns, which
    render first) have filled in - FBref populates advanced tables a beat late."""
    ids = ",".join(f"'{s}'" for s in _ID_STATS_JS)
    return (
        f"(() => {{const t=document.getElementById('{table_id}');"
        f"if(!t)return false;const skip=new Set([{ids}]);"
        f"const c=t.querySelectorAll('tbody td[data-stat]');let n=0;"
        f"for(const x of c){{if(!skip.has(x.getAttribute('data-stat'))"
        f"&&x.textContent.trim())n++;}}"
        f"return n>50;}})()"
    )


# cells present even before the stat numbers load - don't count these
_ID_STATS = {
    "ranker", "player", "nationality", "position", "team", "age", "birth_year",
    "minutes_90s", "games", "matches", "assists",
}


def _has_data(html: str, table_id: str) -> bool:
    """True once the table has real numbers in a spread of stat columns (FBref
    serves the skeleton, then some pages fill their advanced columns a beat
    later - or not at all if we grabbed the DOM too soon)."""
    doc = html.replace("<!--", "").replace("-->", "")
    m = re.search(rf'id="{table_id}".*?</table>', doc, re.S)
    if not m:
        return False
    filled = set(re.findall(r'data-stat="([a-z_0-9]+)"[^>]*>\s*[\d.]', m.group(0)))
    return len(filled - _ID_STATS) >= 6


async def scrape(leagues: list[str], seasons: list[str], cats: list[str]) -> None:
    import nodriver as uc

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season()
    browser = await uc.start(browser_executable_path=BROWSER, headless=False)
    try:
        for league in leagues:
            cid, slug, code = COMPS[league]
            for season in seasons:
                s4 = season4(season)
                is_current = season == current
                for cat in cats:
                    segment, table_id = CATEGORIES[cat]
                    out = STATS_DIR / f"{code}_{season}_{cat}.html"
                    if out.exists() and _has_data(out.read_text(encoding="utf-8"), table_id):
                        continue
                    url = _url(cid, s4, slug, segment, is_current)
                    html = await get_html(browser, url, want_sel=f"#{table_id}",
                                          settle_js=_settle_js(table_id))
                    if any(m in html for m in CHALLENGE_MARKERS):
                        print(f"  !! {code} {season} {cat}: stuck on challenge")
                        continue
                    doc = html.replace("<!--", "").replace("-->", "")
                    if f'id="{table_id}"' not in doc:
                        print(f"  -  {code} {season} {cat}: no table (season/category absent)")
                        continue
                    if not _has_data(html, table_id):
                        print(f"  ?? {code} {season} {cat}: table still unfilled, saving anyway")
                    out.write_text(html, encoding="utf-8")
                    print(f"  ok {code} {season} {cat}")
                    await asyncio.sleep(random.uniform(*NAV_PAUSE))
    finally:
        browser.stop()


def status() -> None:
    if not STATS_DIR.exists():
        print("no player-stats pages yet")
        return
    pages = sorted(STATS_DIR.glob("*.html"))
    by_ls: dict[str, int] = {}
    for p in pages:
        parts = p.stem.split("_")
        by_ls["_".join(parts[:2])] = by_ls.get("_".join(parts[:2]), 0) + 1
    print(f"{len(pages)} player-stats pages ({len(CATEGORIES)} categories x "
          f"{len(COMPS)} leagues x {len(DEFAULT_SEASONS)} seasons = "
          f"{len(CATEGORIES) * len(COMPS) * len(DEFAULT_SEASONS)} max)")
    for k, v in sorted(by_ls.items()):
        print(f"  {k}: {v}/{len(CATEGORIES)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES, choices=list(COMPS))
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--cats", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return

    import nodriver as uc
    uc.loop().run_until_complete(scrape(args.leagues, args.seasons, args.cats))
    print("\nnow build the table:  py -3.11 src/parse_fbref_player_stats.py")


if __name__ == "__main__":
    main()
