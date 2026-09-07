"""Per-player, per-match stats from FotMob's hidden matchDetails API - the
browser-free source for the stats Understat does not carry (shots on target,
fouls, tackles / interceptions / blocks / clearances / recoveries, touches),
so the every-other-day refresh no longer needs a manual FBref scrape for the
current season.

FotMob's `playerStats` block is keyed with stable slugs (`ShotsOnTarget`,
`fouls`, `interceptions` ...), so parsing is just a key lookup. Each match is
cached under data/raw/live/fotmob_players/<id>.json - re-runs only fetch new
matches. FotMob's ToS restricts programmatic use; keep volume low.

Run:  py -3.11 src/pull_fotmob_players.py --current      # season in progress
      py -3.11 src/pull_fotmob_players.py --seasons 2025-26 2026-27
Output: data/processed/fotmob_player_season.csv  (one row per player-team-season)
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
OUT = PROC / "fotmob_player_season.csv"
BASE = "https://www.fotmob.com/api/data"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
SLEEP = 0.5

LEAGUE_ID = {"ENG1": 47, "GER1": 54, "ESP1": 87}

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


def _parse_match(mid: str) -> list[dict] | None:
    """Per-player rows for one match. Cached (a finished match never changes)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{mid}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    try:
        d = _get("matchDetails", matchId=mid)
    except requests.HTTPError:
        return None
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


def pull_season(code: str, season: str) -> pd.DataFrame | None:
    start = int(season.split("-")[0])
    try:
        fixtures = _get("fixtures", id=LEAGUE_ID[code], season=f"{start}/{start + 1}")
    except requests.HTTPError as e:
        print(f"  {code} {season}: fixtures failed ({e})")
        return None
    if not isinstance(fixtures, list):
        return None
    done = [f for f in fixtures if f.get("status", {}).get("finished")]
    print(f"  {code} {season}: {len(done)} finished matches", end="", flush=True)
    frames, n_new = [], 0
    for fx in done:
        if not (CACHE / f"{fx['id']}.json").exists():
            n_new += 1
        rows = _parse_match(str(fx["id"]))
        if rows:
            frames.append(pd.DataFrame(rows))
    print(f"  ({n_new} newly fetched)")
    if not frames:
        return None
    pm = pd.concat(frames, ignore_index=True)
    pm["season"], pm["src_league"] = season, code
    return pm


def _aggregate(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    for c in STAT_KEYS.values():
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    pm["matches"] = 1
    pm["team_key"] = pm["team"].map(normalize_team)
    agg = (pm.groupby(["season", "src_league", "fotmob_id", "team_key"], as_index=False)
             .agg({**{c: "sum" for c in SUM_COLS},
                   "player": "last", "team": "last", "position": "last"}))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=[current_season()])
    ap.add_argument("--current", action="store_true", help="just the season in progress")
    args = ap.parse_args()
    seasons = [current_season()] if args.current else args.seasons

    frames, refreshed = [], set()
    for season in seasons:
        for code in LEAGUE_ID:
            pm = pull_season(code, season)
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


if __name__ == "__main__":
    main()
