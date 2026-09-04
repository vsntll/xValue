"""Step 3 (recent seasons) - scrape Transfermarkt squad market values with nodriver.

The worldfootballR mirror stops at 2022-23 and TM blocks every non-browser client
(HTTP 405). nodriver (real Chrome) gets past that plus the GDPR consent wall.

Per (league, season): the competition page gives the ~20 club ids/slugs, then
each club's squad page (`/kader/verein/<id>/saison_id/<y>/plus/1`) lists every
player with their market value at that time. Pages are cached, so re-runs resume.

Run (Python 3.11, visible Chrome):
    py -3.11 src/pull_transfermarkt_scrape.py                       # 2023-24..2025-26
    py -3.11 src/pull_transfermarkt_scrape.py --seasons 2024-25
    py -3.11 src/pull_transfermarkt_scrape.py --parse-only

Output:
    data/raw/tm/squads/<COMP>_<season>_<slug>.html
    data/processed/tm_values_scraped.csv   -> folded into tm_player_values.csv by
                                              parse_fbref_player_stats via a union
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BROWSER = r"C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe"
SQUAD_DIR = PROJECT_ROOT / "data" / "raw" / "tm" / "squads"
OUT = PROJECT_ROOT / "data" / "processed" / "tm_values_scraped.csv"

COMPS = {"ENG1": "GB1", "GER1": "L1", "ESP1": "ES1"}
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
BASE = "https://www.transfermarkt.com"


def _saison(season: str) -> str:
    return season.split("-")[0]


def _value_eur(txt: str) -> float | None:
    m = re.search(r"([\d.,]+)\s*([mk])?", txt.replace("\u20ac", "").replace("EUR", "").strip(), re.I)
    if not m:
        return None
    n = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "").lower()
    return n * (1_000_000 if unit == "m" else 1_000 if unit == "k" else 1)


async def _accept_consent(page) -> None:
    for label in ("Accept all", "Accept & continue", "AGREE", "Accept"):
        try:
            btn = await page.find(label, best_match=True, timeout=4)
            if btn:
                await btn.click()
                await page.sleep(2)
                return
        except Exception:
            pass


async def _get(browser, url: str, tries: int = 3) -> str:
    for _ in range(tries):
        page = await browser.get(url)
        await page.sleep(3.5)
        html = await page.get_content()
        if len(html) > 60_000 and "verein" in html:
            return html
        await _accept_consent(page)
        await page.sleep(2)
        html = await page.get_content()
        if len(html) > 60_000:
            return html
    return html


async def scrape(seasons: list[str]) -> None:
    import nodriver as uc

    SQUAD_DIR.mkdir(parents=True, exist_ok=True)
    browser = await uc.start(browser_executable_path=BROWSER, headless=False)
    try:
        first = await browser.get(f"{BASE}/premier-league/startseite/wettbewerb/GB1")
        await _accept_consent(first)
        await first.sleep(2)

        for code, wett in COMPS.items():
            for season in seasons:
                y = _saison(season)
                comp_html = await _get(
                    browser, f"{BASE}/{wett.lower()}/startseite/wettbewerb/{wett}/saison_id/{y}")
                clubs = sorted(set(re.findall(
                    rf'/([a-z0-9-]+)/startseite/verein/(\d+)/saison_id/{y}', comp_html)))
                if not clubs:
                    clubs = sorted(set(re.findall(
                        r'/([a-z0-9-]+)/startseite/verein/(\d+)', comp_html)))[:24]
                print(f"{code} {season}: {len(clubs)} clubs")
                for slug, cid in clubs:
                    out = SQUAD_DIR / f"{code}_{season}_{slug}.html"
                    if out.exists() and out.stat().st_size > 80_000:
                        continue
                    url = f"{BASE}/{slug}/kader/verein/{cid}/saison_id/{y}/plus/1"
                    html = await _get(browser, url)
                    if 'profil/spieler/' not in html:
                        print(f"  !! {slug}: no squad table")
                        continue
                    out.write_text(html, encoding="utf-8")
                    print(f"  ok {slug}")
                    await asyncio.sleep(random.uniform(1.5, 3.0))
    finally:
        browser.stop()


def parse() -> None:
    rows = []
    for path in sorted(SQUAD_DIR.glob("*.html")):
        code, season, slug = path.stem.split("_", 2)
        html = path.read_text(encoding="utf-8")
        try:
            tbl = pd.read_html(StringIO(html), attrs={"class": "items"})[0]
        except (ValueError, IndexError):
            continue
        tbl.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in tbl.columns]
        # player id + name in document order, aligned to the value column
        ids = re.findall(r'/profil/spieler/(\d+)"\s+id="\d+">', html) \
            or re.findall(r'href="/[^"]+/profil/spieler/(\d+)"', html)
        val_col = next((c for c in tbl.columns if "market value" in c.lower()
                        or c.lower() in ("mv", "value")), tbl.columns[-1])
        name_col = next((c for c in tbl.columns if c.lower() in ("player", "name")), None)
        vals = tbl[val_col].astype(str).tolist()
        names = tbl[name_col].astype(str).tolist() if name_col else [None] * len(vals)
        for i, v in enumerate(vals):
            mv = _value_eur(v)
            if mv is None:
                continue
            rows.append({
                "season": season, "src_league": code, "squad_slug": slug,
                "tm_player_id": ids[i] if i < len(ids) else None,
                "player_name": names[i] if i < len(names) else None,
                "market_value_eur": mv,
            })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} player-values)")
    if len(df):
        print(df.groupby(["season", "src_league"]).size().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()

    if not args.parse_only:
        import nodriver as uc
        uc.loop().run_until_complete(scrape(args.seasons))
    parse()


if __name__ == "__main__":
    main()
