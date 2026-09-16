"""Step 3 (recent seasons) - scrape Transfermarkt squad market values with nodriver.

The worldfootballR mirror stops at 2022-23 and TM blocks every non-browser client
(HTTP 405). nodriver (real Chrome) gets past that plus the GDPR consent wall.

Per (league, season): the competition page gives the ~20 club ids/slugs, then
each club's squad page (`/kader/verein/<id>/saison_id/<y>/plus/1`) lists every
player with their market value at that time. Pages are cached, so re-runs resume.

ENG1/GER1/ESP1 are the modelled leagues. ITA1/FRA1 aren't modelled but their
past-season values go into value_history.csv, so a Serie A / Ligue 1 -> Premier
League mover isn't a cold start for the value model (a big chunk of the ~30%
that had no prior value).

Run (Python 3.11, visible Chrome - click the consent / WAF wall once):
    py -3.11 src/pull_transfermarkt_scrape.py                        # 2023-24..2025-26, all 5 leagues
    py -3.11 src/pull_transfermarkt_scrape.py --comps ITA1 FRA1 --seasons 2023-24 2024-25 2025-26
    py -3.11 src/pull_transfermarkt_scrape.py --parse-only

Output:
    data/raw/tm/squads/<COMP>_<season>_<slug>.html
    data/processed/tm_values_scraped.csv   -> folded into value_history.csv (all
                                              leagues) + fbref_player_season_stats
                                              (our 3) by their union steps

--profiles mode: a targeted backfill for the value model's cold-start segment
(has_any_prev == 0 - see train_value_model.py and build_cold_start_list.py). A
player's TM profile /transfers/ sub-page lists his whole transfer history with
the market value AND fee at each move, so one page visit per cold-start player
gives both prev_any_mv (the value at the transfer that brought him to his
current club) and a transfer-fee signal, without mirroring a whole league:

    py -3.11 src/build_cold_start_list.py          # -> cold_start_players.csv
    py -3.11 src/pull_transfermarkt_scrape.py --profiles
    py -3.11 src/pull_transfermarkt_scrape.py --profiles --parse-only

Player ids are resolved from data the pipeline already has - tm_player_values.csv
(the worldfootballR mirror's player_url, 2020-23) and the squad HTML this same
script caches under data/raw/tm/squads/ (2023-26; run the squad scrape for
whatever seasons your cold-start list covers first if an id doesn't resolve).

Output:
    data/raw/tm/profiles/<tm_player_id>.json   (transfer rows extracted from the
                                                 shadow-DOM grid via JS, not raw HTML)
    data/processed/tm_transfer_history.csv  -> folds into value_history.csv,
                                                same union pattern as tm_values_scraped.csv
    data/processed/tm_transfer_fees.csv     -> last_transfer_fee_eur / fee_is_loan /
                                                fee_is_free features in train_value_model.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live.schema import deaccent, normalize_team  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BROWSER = r"C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe"
SQUAD_DIR = PROJECT_ROOT / "data" / "raw" / "tm" / "squads"
OUT = PROJECT_ROOT / "data" / "processed" / "tm_values_scraped.csv"

PROFILE_DIR = PROJECT_ROOT / "data" / "raw" / "tm" / "profiles"
COLD_LIST = PROJECT_ROOT / "data" / "processed" / "cold_start_players.csv"
TM_PLAYER_VALUES = PROJECT_ROOT / "data" / "processed" / "tm_player_values.csv"
TRANSFER_OUT = PROJECT_ROOT / "data" / "processed" / "tm_transfer_history.csv"
FEE_OUT = PROJECT_ROOT / "data" / "processed" / "tm_transfer_fees.csv"

COMPS = {"ENG1": "GB1", "GER1": "L1", "ESP1": "ES1", "ITA1": "IT1", "FRA1": "FR1"}
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]
BASE = "https://www.transfermarkt.com"


def _key(s) -> str:
    """Same bare-ascii-lowercase join key as build_value_history.py, so a
    fbref Player name and a TM slug/player_name land on the same key."""
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


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


async def scrape(seasons: list[str], comps: list[str]) -> None:
    import nodriver as uc

    SQUAD_DIR.mkdir(parents=True, exist_ok=True)
    browser = await uc.start(browser_executable_path=BROWSER, headless=False)
    try:
        first = await browser.get(f"{BASE}/premier-league/startseite/wettbewerb/GB1")
        await _accept_consent(first)
        await first.sleep(2)

        for code in comps:
            wett = COMPS[code]
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


def _resolve_ids(names: set[str]) -> dict[str, tuple[str, str]]:
    """{_key(name): (slug, tm_player_id)} for as many `names` as we can resolve
    from data the pipeline already has, no new requests: the worldfootballR
    mirror's player_url (2020-23) and the profile links already sitting in this
    script's own cached squad HTML (2023-26, whatever's been scraped so far)."""
    found: dict[str, tuple[str, str]] = {}
    if TM_PLAYER_VALUES.exists():
        d = pd.read_csv(TM_PLAYER_VALUES, usecols=["player_url"]).dropna()
        for m in d["player_url"]:
            mm = re.search(r"transfermarkt\.com/([a-z0-9-]+)/profil/spieler/(\d+)", str(m))
            if mm:
                slug, pid = mm.groups()
                found.setdefault(_key(slug.replace("-", " ")), (slug, pid))
    for path in sorted(SQUAD_DIR.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        for slug, pid in re.findall(r'href="/([a-z0-9-]+)/profil/spieler/(\d+)"', html):
            found.setdefault(_key(slug.replace("-", " ")), (slug, pid))
    return {k: v for k, v in found.items() if k in names}


# the transfer-history grid is a Svelte web component (<tm-player-transfer-history>)
# rendered into an OPEN shadow root, not a <table> - nothing server-rendered, so
# pd.read_html can never see it. Pull the rows out with JS instead. Scoped-CSS
# class names (the "svelte-xxxxx" hashes) change on every TM deploy, so this
# matches structurally: a real transfer row is any <section> containing a club
# link, not the header row. JSON.stringify inside the JS itself - nodriver's
# evaluate() returns object results as CDP remote-object wrappers, not plain
# Python values, but a plain string round-trips cleanly through json.loads.
_EXTRACT_JS = r"""
JSON.stringify((() => {
    const el = document.querySelector('tm-player-transfer-history');
    if (!el || !el.shadowRoot) return null;
    const rows = Array.from(el.shadowRoot.querySelectorAll('section'))
        .filter(s => s.querySelector('a[href*="/verein/"]'));
    return rows.map(row => {
        const divs = Array.from(row.querySelectorAll(':scope > div'));
        const text = i => divs[i] ? divs[i].textContent.replace(/\s+/g, ' ').trim() : null;
        return {season: text(0), date: text(1), left: text(2), joined: text(3),
                mv: text(4), fee: text(5)};
    });
})())
"""
PAGE_TIMEOUT = 25    # seconds - a wedged navigation/evaluate call raises instead of hanging


async def _start_browser(uc):
    browser = await uc.start(browser_executable_path=BROWSER, headless=False)
    first = await browser.get(f"{BASE}/premier-league/startseite/wettbewerb/GB1")
    await _accept_consent(first)
    await first.sleep(2)
    return browser


async def scrape_profiles() -> None:
    import nodriver as uc

    if not COLD_LIST.exists():
        raise SystemExit(f"no {COLD_LIST} - run build_cold_start_list.py first")
    cold = pd.read_csv(COLD_LIST)
    names = {_key(n) for n in cold["Player"].dropna().unique()}
    ids = _resolve_ids(names)
    missing = len(names) - len(ids)
    print(f"resolved {len(ids)}/{len(names)} cold-start players to a TM id "
          f"({missing} unmatched - not in tm_player_values.csv or any cached "
          f"squad page; scrape squad pages for their season/comp first)")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(s, p) for s, p in sorted(set(ids.values()), key=lambda t: t[1])
            if not (PROFILE_DIR / f"{p}.json").exists()]
    print(f"{len(todo)} profiles to fetch")

    # one browser session for the whole run - restarting means a fresh profile,
    # which means re-clicking consent every time. Only restart reactively, on
    # an actual error, not on a fixed schedule.
    browser = await _start_browser(uc)
    try:
        for slug, pid in todo:
            try:
                page = await asyncio.wait_for(
                    browser.get(f"{BASE}/{slug}/transfers/spieler/{pid}"), PAGE_TIMEOUT)
                await page.sleep(4)
                raw = await asyncio.wait_for(
                    page.evaluate(_EXTRACT_JS, await_promise=True), PAGE_TIMEOUT)
                rows = json.loads(raw) if raw else None
            except Exception as e:
                print(f"  !! {slug} ({pid}): {type(e).__name__}: {str(e)[:100]} "
                      f"- restarting browser")
                try:
                    browser.stop()
                except Exception:
                    pass
                browser = await _start_browser(uc)
                continue
            if not rows:
                print(f"  !! {slug} ({pid}): no transfer rows in shadow DOM")
                continue
            (PROFILE_DIR / f"{pid}.json").write_text(
                json.dumps(rows), encoding="utf-8")
            print(f"  ok {slug} ({pid}) - {len(rows)} transfers")
            await asyncio.sleep(random.uniform(1.5, 3.0))
    finally:
        browser.stop()


def _season_from_short(txt: str) -> str | None:
    """TM's short season form ("23/24") -> our "2023-24"."""
    m = re.search(r"(\d{2})\s*/\s*(\d{2})", str(txt))
    if not m:
        return None
    y = int(m.group(1))
    y += 2000 if y < 50 else 1900
    return f"{y}-{str(y + 1)[-2:]}"


def parse_profiles() -> None:
    """Each cached profile's transfer-row JSON (see _EXTRACT_JS in
    scrape_profiles) -> a value snapshot per transfer (feeds value_history.csv
    the same way tm_values_scraped.csv does) + the most recent transfer's fee
    (feeds train_value_model.py)."""
    val_rows, fee_rows = [], []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        pid = path.stem
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not rows:
            continue

        seasons_seen = []
        for r in rows:
            season = _season_from_short(r.get("season") or "")
            if season is None:
                continue
            seasons_seen.append(season)
            mv = _value_eur(str(r.get("mv") or ""))
            if mv is not None:
                val_rows.append({"tm_player_id": pid, "season": season,
                                  "joined": r.get("joined"), "market_value_eur": mv})
            fee_txt = str(r.get("fee") or "").strip().lower()
            is_loan = "loan" in fee_txt
            is_free = "free" in fee_txt or fee_txt in ("", "-")
            fee = None if (is_loan or is_free) else _value_eur(fee_txt)
            fee_rows.append({"tm_player_id": pid, "season": season,
                              "joined": r.get("joined"),
                              "fee_eur": fee, "fee_is_loan": int(is_loan),
                              "fee_is_free": int(is_free)})
        if not seasons_seen:
            print(f"  !! {pid}: {len(rows)} rows but none had a parseable season "
                  f"(sample: {rows[0]})")

    if not val_rows and not fee_rows:
        print("no transfer rows parsed - inspect a cached profile page's table "
              "layout by hand and adjust parse_profiles()")
        return

    # tm_transfer_history.csv unions into value_history.csv exactly like
    # tm_values_scraped.csv - one row per (player, season, value) snapshot.
    # It's keyed by tm_player_id here; build_value_history.py joins by name, so
    # we need the id -> name map we already have from cold_start_players.csv's
    # resolution step - re-derive it rather than thread it through.
    cold = pd.read_csv(COLD_LIST) if COLD_LIST.exists() else pd.DataFrame()
    id_to_name = {}
    id_to_squad = {}
    if len(cold):
        ids = _resolve_ids({_key(n) for n in cold["Player"].dropna().unique()})
        name_by_key = {_key(n): n for n in cold["Player"].dropna().unique()}
        # most recent cold-start row per player = the squad his current-season
        # value should be anchored to, to sanity-check which transfer is "his
        # move to his current club" rather than trusting season-string order
        # alone (a fallen-through rumour or same-season loan-then-permanent
        # move could otherwise pick the wrong row).
        squad_by_key = (cold.assign(_k=cold["Player"].map(_key))
                        .sort_values("season").groupby("_k")["Squad"].last())
        for k, (slug, pid) in ids.items():
            id_to_name[pid] = name_by_key.get(k, slug.replace("-", " ").title())
            if k in squad_by_key.index:
                id_to_squad[pid] = squad_by_key[k]

    if val_rows:
        vdf = pd.DataFrame(val_rows)
        vdf["player_name"] = vdf["tm_player_id"].astype(str).map(
            lambda i: id_to_name.get(i, i))
        vdf[["player_name", "season", "joined", "market_value_eur"]].drop_duplicates().to_csv(
            TRANSFER_OUT, index=False)
        print(f"wrote {TRANSFER_OUT}  ({len(vdf)} transfer-value snapshots)")

    if fee_rows:
        fdf = pd.DataFrame(fee_rows).sort_values("season")
        # the move to his CURRENT club is the feature; prefer the row whose
        # "joined" club matches his known current squad over just trusting the
        # latest season string, which a fallen-through rumour or a same-season
        # loan-then-permanent move could get wrong.
        target = fdf["tm_player_id"].map(id_to_squad).map(
            lambda s: normalize_team(s) if pd.notna(s) else None)
        joined_norm = fdf["joined"].map(lambda s: normalize_team(s) if pd.notna(s) else None)
        fdf["_matches_squad"] = (target.notna() & (target == joined_norm))
        mismatched = (fdf.groupby("tm_player_id")["_matches_squad"].transform("any")
                      & ~fdf["_matches_squad"])
        fdf = fdf[~mismatched]
        # groupby().last() takes each COLUMN's last non-null value independently,
        # which mixes fields across different transfers (e.g. a free transfer's
        # fee_is_free=1 paired with an earlier transfer's real fee_eur) - tail(1)
        # keeps one real row intact.
        fdf = fdf.groupby("tm_player_id").tail(1)
        unmatched = int((~fdf["_matches_squad"] & fdf["tm_player_id"].map(id_to_squad).notna()).sum())
        if unmatched:
            print(f"  note: {unmatched} players' latest transfer row doesn't "
                  f"name their known current squad as 'joined' - used anyway "
                  f"(no other row matched either); spot-check these")
        fdf["player_name"] = fdf["tm_player_id"].astype(str).map(
            lambda i: id_to_name.get(i, i))
        fdf[["player_name", "season", "joined", "fee_eur", "fee_is_loan",
             "fee_is_free"]].to_csv(FEE_OUT, index=False)
        print(f"wrote {FEE_OUT}  ({len(fdf)} players' most recent transfer fee)")


_POSITIONS = [
    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Sweeper",
    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
    "Second Striker", "Centre-Forward", "Defender", "Midfielder", "Midfield",
    "Forward", "Attack", "midfielder", "defender",
]
_POS_RE = re.compile(r"\s*(" + "|".join(re.escape(p) for p in _POSITIONS) + r")\s*$")


def _strip_pos(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = _POS_RE.sub("", name).strip()
    return name


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
        val_col = next((c for c in tbl.columns if "market value" in c.lower()), tbl.columns[-1])
        name_col = next((c for c in tbl.columns if c.lower() == "player"), tbl.columns[1])
        real = tbl[tbl[val_col].notna() & (tbl[val_col].astype(str) != "-")].copy()

        # id/slug from the squad-table profile links, deduped in document order.
        # NB read_html's row order and this list can drift on odd pages - if the
        # counts don't line up we keep the (mojibake-prone) read_html name and
        # skip the id rather than mis-attach.
        body = html.split('class="responsive-table"', 1)[-1]
        pairs = list(dict.fromkeys(
            re.findall(r'href="/([a-z0-9-]+)/profil/spieler/(\d+)"', body)))
        aligned = len(pairs) == len(real)
        for i, (_, r) in enumerate(real.iterrows()):
            mv = _value_eur(str(r[val_col]))
            if mv is None:
                continue
            nm = _strip_pos(str(r[name_col]))
            if aligned:
                nm = pairs[i][0].replace("-", " ")
            rows.append({
                "season": season, "src_league": code, "squad_slug": slug,
                "tm_player_id": pairs[i][1] if aligned else None,
                "player_name": nm,
                "market_value_eur": mv,
            })
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["season", "src_league", "squad_slug", "player_name"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} player-values)")
    if len(df):
        print(df.groupby(["season", "src_league"]).size().to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--comps", nargs="+", default=list(COMPS), choices=list(COMPS))
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--profiles", action="store_true",
                     help="targeted cold-start backfill (see module docstring) "
                          "instead of the squad-page scrape")
    args = ap.parse_args()

    if args.profiles:
        if not args.parse_only:
            import nodriver as uc
            uc.loop().run_until_complete(scrape_profiles())
        parse_profiles()
        return

    if not args.parse_only:
        import nodriver as uc
        uc.loop().run_until_complete(scrape(args.seasons, args.comps))
    parse()


if __name__ == "__main__":
    main()
