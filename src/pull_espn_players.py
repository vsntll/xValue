"""Per-player, per-match box scores for the five leagues from ESPN's public site
API - an independent second source to cross-check FotMob's rows against
(src/verify_player_matches.py) while Understat is unavailable. ESPN has no xG
and no minutes, but it does carry goals / assists / shots / shots on target /
fouls / saves per player.

One scoreboard request per league-month finds the finished matches, one summary
request per match gets both rosters. Every match's rows are committed per
season to data/processed/espn/player_matches_<season>.csv and listed in
espn_match_meta.csv, so a re-run (even on a fresh CI runner) only requests
matches not committed yet. --budget-min caps the fetching time per run.

Run:  py -3.11 src/pull_espn_players.py --current
      py -3.11 src/pull_espn_players.py --seasons 2025-26 2026-27
Output: data/processed/espn/player_matches_<season>.csv  (one row per player who played)
        data/processed/espn_match_meta.csv               (one row per match: teams, score)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store as store  # noqa: E402
from fbref_common import current_season  # noqa: E402
from live.espn import _season_months  # noqa: E402
from live.schema import ALL_COMPS  # noqa: E402
from net import get_json  # noqa: E402

PROC = ROOT / "data" / "processed"
DIR = PROC / "espn"
META = PROC / "espn_match_meta.csv"
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
HDRS = {"User-Agent": "Mozilla/5.0"}
SLEEP = 0.3
LEAGUES = ["ENG1", "GER1", "ESP1", "ITA1", "FRA1"]
# ESPN roster stat name -> our column (FotMob's names, so the check lines up)
STATS = {"totalGoals": "goals", "goalAssists": "assists", "totalShots": "shots",
         "shotsOnTarget": "sot", "foulsCommitted": "fouls", "foulsSuffered": "fouled",
         "saves": "saves", "goalsConceded": "goals_conceded", "ownGoals": "own_goals",
         "yellowCards": "yellow_cards", "redCards": "red_cards"}


def _get(path: str, **params) -> dict:
    out = get_json(f"{BASE}/{path}", params=params, headers=HDRS)
    time.sleep(SLEEP)
    return out


def _player_rows(summary: dict, eid: str) -> list[dict]:
    rows = []
    for side in summary.get("rosters") or []:
        team = side.get("team") or {}
        for p in side.get("roster") or []:
            st = {s["name"]: s.get("value") for s in p.get("stats") or []}
            if not st.get("appearances"):
                continue   # an unused sub
            a = p.get("athlete") or {}
            rows.append({"match_id": f"espn:{eid}", "team": team.get("displayName"),
                         "team_id": team.get("id"), "is_home": int(side.get("homeAway") == "home"),
                         "espn_id": a.get("id"), "player": a.get("displayName"),
                         "starter": int(bool(p.get("starter"))),
                         **{c: st.get(k) for k, c in STATS.items()}})
    return rows


def pull_season(code: str, season: str, done: set[str], deadline: float) -> tuple[int, int]:
    lc = ALL_COMPS[code]["espn"]
    finished = {}
    for month in _season_months(season):
        try:
            board = _get(f"{lc}/scoreboard", dates=month, limit=1000)
        except requests.RequestException as e:
            print(f"  {code} {month}: scoreboard failed ({e})")
            continue
        for ev in board.get("events", []):
            if ((ev.get("status") or {}).get("type") or {}).get("completed"):
                finished[ev["id"]] = ev
    meta = []
    for eid, ev in finished.items():
        cs = {c["homeAway"]: c for c in ev["competitions"][0]["competitors"]}
        meta.append({"match_id": f"espn:{eid}", "season": season, "src_league": code,
                     "date": ev["date"][:10], "home": cs["home"]["team"]["displayName"],
                     "away": cs["away"]["team"]["displayName"],
                     "home_score": cs["home"].get("score"), "away_score": cs["away"].get("score")})
    todo = [e for e in finished if f"espn:{e}" not in done]
    print(f"  {code} {season}: {len(finished)} finished matches ({len(todo)} to fetch)", flush=True)
    rows, fetched, n_left = [], [], 0
    for eid in todo:
        if time.monotonic() > deadline:
            n_left += 1
            continue
        try:
            rows += [{**r, "season": season, "src_league": code}
                     for r in _player_rows(_get(f"{lc}/summary", event=eid), eid)]
        except requests.RequestException:
            continue
        fetched.append(f"espn:{eid}")
    store.upsert("player_matches", season, pd.DataFrame(rows), base=DIR)
    if meta:
        m = pd.DataFrame(meta)
        m.loc[m["match_id"].isin(fetched), "fetched"] = 1
        store.upsert_meta(m, META)
    return len(fetched), n_left


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=[current_season()])
    ap.add_argument("--current", action="store_true", help="just the season in progress")
    ap.add_argument("--budget-min", type=float, default=10,
                    help="stop fetching new matches after this many minutes (rest next run)")
    args = ap.parse_args()
    seasons = [current_season()] if args.current else args.seasons

    meta = store.read_meta(META)
    done = set(meta.loc[meta["fetched"] == 1, "match_id"])
    deadline = time.monotonic() + args.budget_min * 60
    n_new = n_left = 0
    for season in seasons:
        for code in LEAGUES:
            a, b = pull_season(code, season, done, deadline)
            n_new, n_left = n_new + a, n_left + b
    print(f"\n{n_new} matches fetched" + (f", {n_left} left for next run (budget spent)" if n_left else ""))


if __name__ == "__main__":
    main()
