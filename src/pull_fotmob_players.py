"""Per-player, per-match stats from FotMob's hidden matchDetails API - the
browser-free source for the stats Understat does not carry (shots on target,
fouls, tackles / interceptions / blocks / clearances / recoveries, touches),
so the every-other-day refresh no longer needs a manual FBref scrape for the
current season.

FotMob's `playerStats` block is keyed with stable slugs (`ShotsOnTarget`,
`fouls`, `interceptions` ...), so parsing is just a key lookup. Each match is
cached under data/raw/live/fotmob_players/<id>.json - re-runs only fetch new
matches. FotMob's ToS restricts programmatic use; keep volume low.

The same matchDetails payload also carries the starting-XI lineup
(content.lineup), so it's saved alongside the player stats at zero extra
request cost, under data/raw/live/fotmob_lineups/<id>.json - see
build_match_lineups.py. Same for the goal/red-card event timeline
(content.matchFacts.events), under data/raw/live/fotmob_events/<id>.json -
see live_win_prob.py's backtest. The events cache is opportunistic for new
matches only; backfilling it for already-pulled seasons needs the separate
--events flag (a deliberate, bounded re-fetch, not automatic).

Run:  py -3.11 src/pull_fotmob_players.py --current      # season in progress
      py -3.11 src/pull_fotmob_players.py --seasons 2025-26 2026-27
      py -3.11 src/pull_fotmob_players.py --events 2025-26   # backfill event timelines
Output: data/processed/fotmob_player_season.csv  (one row per player-team-season)
        data/processed/fotmob_match_meta.csv     (match id -> season/league/date)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fbref_common import current_season  # noqa: E402
from live.schema import deaccent, normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "raw" / "live" / "fotmob_players"
LINEUP_CACHE = ROOT / "data" / "raw" / "live" / "fotmob_lineups"
EVENTS_CACHE = ROOT / "data" / "raw" / "live" / "fotmob_events"
OUT = PROC / "fotmob_player_season.csv"
MATCH_META = PROC / "fotmob_match_meta.csv"
BASE = "https://www.fotmob.com/api/data"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
SLEEP = 0.5

LEAGUE_ID = {"ENG1": 47, "GER1": 54, "ESP1": 87, "ITA1": 55, "FRA1": 53}

# FotMob playerStats key -> our column. `value` is taken even for the
# fraction-with-percentage entries (aerials etc.), which is the "won"/"made" count.
STAT_KEYS = {
    "minutes_played": "minutes", "goals": "goals", "assists": "assists",
    "total_shots": "shots", "ShotsOnTarget": "sot",
    "expected_goals": "xg", "expected_goals_non_penalty": "npxg", "expected_assists": "xa",
    "fouls": "fouls", "was_fouled": "fouled",
    "matchstats.headers.tackles": "tackles", "interceptions": "interceptions",
    "shot_blocks": "blocks", "clearances": "clearances", "recoveries": "recoveries",
    "touches": "touches", "touches_opp_box": "touches_box", "dispossessed": "dispossessed",
    "aerials_won": "aerials_won", "chances_created": "chances_created",
    "saves": "saves", "goals_conceded": "goals_conceded",
}
SUM_COLS = [c for c in STAT_KEYS.values() if c != "minutes"] + ["minutes", "matches"]


def _get(path: str, **params) -> object:
    r = requests.get(f"{BASE}/{path}", params=params, headers=HDRS, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def _stat(entry: dict):
    s = entry.get("stat") or {}
    return s.get("value")


def _extract_events(d: dict) -> list[dict]:
    """Goal + red-card timeline from matchFacts.events - each goal's newScore
    is the authoritative running score (own goals already resolved into it;
    the top-level homeScore/awayScore fields on each event are stale and NOT
    reliable). Used by live_win_prob.py's backtest to replay a finished match
    minute-by-minute."""
    mf = (d.get("content") or {}).get("matchFacts") or {}
    ev = (mf.get("events") or {}).get("events") or []
    out = []
    for e in ev:
        t = e.get("type")
        if t == "Goal":
            ns = e.get("newScore") or [None, None]
            out.append({"time": e.get("time"), "type": "Goal",
                        "isHome": bool(e.get("isHome")),
                        "home_score": ns[0], "away_score": ns[1]})
        elif t == "Card" and e.get("card") in ("Red", "YellowRed"):
            out.append({"time": e.get("time"), "type": "Red", "isHome": bool(e.get("isHome"))})
    return out


def _parse_match(mid: str) -> list[dict] | None:
    """Per-player rows for one match, plus its starting-XI lineup - both come
    off the same matchDetails payload, so caching the lineup costs zero extra
    requests. Each cached independently (a finished match never changes); a
    match cached before the lineup cache existed gets exactly one re-fetch to
    backfill it (bounded to that known set, not open-ended re-scraping)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    LINEUP_CACHE.mkdir(parents=True, exist_ok=True)
    EVENTS_CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{mid}.json"
    lf = LINEUP_CACHE / f"{mid}.json"
    ef = EVENTS_CACHE / f"{mid}.json"
    if cf.exists() and lf.exists():
        return json.loads(cf.read_text())
    try:
        d = _get("matchDetails", matchId=mid)
    except requests.HTTPError:
        return json.loads(cf.read_text()) if cf.exists() else None

    lineup = (d.get("content") or {}).get("lineup") or {}
    lf.write_text(json.dumps({k: lineup.get(k) for k in ("homeTeam", "awayTeam")}))
    # opportunistic, best-effort - written whenever we already fetched `d` for
    # a NEW match, but its absence never forces its own re-fetch (unlike cf/lf
    # above) so this doesn't re-trigger a full historical re-scrape; a
    # deliberate, bounded backfill for existing matches is --events below.
    if not ef.exists():
        ef.write_text(json.dumps(_extract_events(d)))
    if cf.exists():
        return json.loads(cf.read_text())

    ps = ((d.get("content") or {}).get("playerStats")) or {}
    if not ps:
        cf.write_text("[]")
        return []
    g = d.get("general") or {}
    date = (g.get("matchTimeUTCDate") or "")[:10]
    rows = []
    for pid, p in ps.items():
        flat = {}
        for grp in p.get("stats", []):
            for _title, entry in (grp.get("stats") or {}).items():
                key = entry.get("key")
                if key in STAT_KEYS:
                    flat.setdefault(STAT_KEYS[key], _stat(entry))
        if flat.get("minutes") in (None, 0):
            continue
        rows.append({
            "fotmob_id": int(pid), "player": p.get("name"),
            "team": p.get("teamName"), "team_id": p.get("teamId"),
            "date": date, "match_id": f"fotmob:{mid}",
            "position": p.get("usualPosition"),
            **{c: flat.get(c) for c in STAT_KEYS.values()},
        })
    cf.write_text(json.dumps(rows))
    return rows


def pull_season(code: str, season: str) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    start = int(season.split("-")[0])
    try:
        fixtures = _get("fixtures", id=LEAGUE_ID[code], season=f"{start}/{start + 1}")
    except requests.HTTPError as e:
        print(f"  {code} {season}: fixtures failed ({e})")
        return None, pd.DataFrame()
    if not isinstance(fixtures, list):
        return None, pd.DataFrame()
    done = [f for f in fixtures if f.get("status", {}).get("finished")]
    print(f"  {code} {season}: {len(done)} finished matches", end="", flush=True)
    frames, meta_rows, n_new = [], [], 0
    for fx in done:
        mid = str(fx["id"])
        if not (CACHE / f"{mid}.json").exists():
            n_new += 1
        rows = _parse_match(mid)
        if rows:
            frames.append(pd.DataFrame(rows))
        # match metadata (id -> season/league/date) so build_match_lineups.py can
        # resolve the lineup cache offline, without re-hitting the fixtures endpoint
        meta_rows.append({"match_id": mid, "season": season, "src_league": code,
                          "date": (fx.get("status") or {}).get("utcTime", "")[:10]})
    print(f"  ({n_new} newly fetched)")
    meta = pd.DataFrame(meta_rows)
    if not frames:
        return None, meta
    pm = pd.concat(frames, ignore_index=True)
    pm["season"], pm["src_league"] = season, code
    return pm, meta


def _aggregate(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    for c in STAT_KEYS.values():
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    pm["matches"] = 1
    pm["team_key"] = pm["team"].map(normalize_team)
    agg = (pm.groupby(["season", "src_league", "fotmob_id", "team_key"], as_index=False)
             .agg({**{c: "sum" for c in SUM_COLS},
                   "player": "last", "team": "last", "position": "last", "team_id": "last"}))
    return agg


def backfill_events(seasons: list[str]) -> None:
    """Deliberate, bounded backfill of the goal/red-card timeline for matches
    that already have player stats + a lineup cached (from an earlier pull)
    but never got matchFacts.events saved - a match needs a fresh request for
    this regardless of what's already cached, since events weren't captured
    before this feature existed. Scoped to specific seasons, not the whole
    history, per the ToS "keep volume low" note above."""
    if not MATCH_META.exists():
        raise SystemExit(f"{MATCH_META} not found - pull player stats for these seasons first")
    EVENTS_CACHE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(MATCH_META, dtype={"match_id": str})
    meta = meta[meta["season"].isin(seasons)]
    print(f"backfilling match events for {len(meta)} matches ({seasons})")
    n_new = n_fail = 0
    for i, mid in enumerate(meta["match_id"]):
        ef = EVENTS_CACHE / f"{mid}.json"
        if ef.exists():
            continue
        try:
            d = _get("matchDetails", matchId=mid)
        except requests.HTTPError:
            n_fail += 1
            continue
        ef.write_text(json.dumps(_extract_events(d)))
        n_new += 1
        if n_new % 200 == 0:
            print(f"  {i + 1}/{len(meta)}  ({n_new} newly fetched, {n_fail} failed)")
    print(f"done: {n_new} newly fetched, {n_fail} failed, "
          f"{len(list(EVENTS_CACHE.glob('*.json')))} total cached")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=[current_season()])
    ap.add_argument("--current", action="store_true", help="just the season in progress")
    ap.add_argument("--events", nargs="+", metavar="SEASON",
                    help="backfill the goal/red-card event timeline for these seasons "
                         "(needs fotmob_match_meta.csv already built) instead of the "
                         "normal player-stats pull")
    args = ap.parse_args()
    if args.events:
        backfill_events(args.events)
        return
    seasons = [current_season()] if args.current else args.seasons

    frames, metas, refreshed, meta_refreshed = [], [], set(), set()
    for season in seasons:
        for code in LEAGUE_ID:
            pm, meta = pull_season(code, season)
            if not meta.empty:
                metas.append(meta)
                meta_refreshed.add((code, season))
            if pm is not None and not pm.empty:
                frames.append(_aggregate(pm))
                refreshed.add((code, season))
    if not frames:
        raise SystemExit("no FotMob player data pulled")
    new = pd.concat(frames, ignore_index=True)

    if OUT.exists():
        prior = pd.read_csv(OUT)
        keep = ~prior.set_index(["src_league", "season"]).index.isin(refreshed)
        new = pd.concat([prior[keep], new], ignore_index=True)
    PROC.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(new)} player-team-seasons, "
          f"{new['fotmob_id'].nunique()} players, seasons {sorted(new['season'].unique())})")

    if metas:
        new_meta = pd.concat(metas, ignore_index=True)
        if MATCH_META.exists():
            prior_meta = pd.read_csv(MATCH_META, dtype={"match_id": str})
            keep = ~prior_meta.set_index(["src_league", "season"]).index.isin(meta_refreshed)
            new_meta = pd.concat([prior_meta[keep], new_meta], ignore_index=True)
        new_meta.to_csv(MATCH_META, index=False)
        print(f"wrote {MATCH_META}  ({len(new_meta)} matches)")


if __name__ == "__main__":
    main()
