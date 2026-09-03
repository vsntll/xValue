"""Understat xG - the historical + current xG source for the three leagues.

FBref gates its xG columns from scrapers and the worldfootballR mirror stops at
2022-23, so xG comes from Understat (EPL / Bundesliga / La Liga, 2014-present).
soccerdata's reader uses a TLS-fingerprint HTTP client - no browser, no
Cloudflare. Understat has **no cup / European coverage** - that xG comes from
FotMob (src/live/fotmob.py).

Run (Python 3.11):
    py -3.11 src/pull_understat.py                       # all seasons, both tables
    py -3.11 src/pull_understat.py --seasons 2026-27     # refresh the live season
    py -3.11 src/pull_understat.py --what matches

Output:
    data/processed/understat_matches.csv         one row per match, home_xg/away_xg
    data/processed/understat_player_season.csv   one row per player-team-season, xg/xa/npxg/...
    data/understat_cache/                        soccerdata cache (SOCCERDATA_DIR)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SOCCERDATA_DIR", str(PROJECT_ROOT / "data" / "understat_cache"))

import pandas as pd  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
LEAGUE_KEY = {
    "ENG1": "ENG-Premier League",
    "GER1": "GER-Bundesliga",
    "ESP1": "ESP-La Liga",
}
DEFAULT_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]


def _sd_season(season: str) -> str:
    """soccerdata wants an unambiguous form - a bare start year like '2021' is
    read as the 2020-21 season, not 2021-22. Use YYYY-YYYY."""
    start = int(season.split("-")[0])
    return f"{start}-{start + 1}"


def pull_matches(seasons: list[str]) -> None:
    import soccerdata as sd

    frames = []
    for code, key in LEAGUE_KEY.items():
        for season in seasons:
            try:
                sch = sd.Understat(leagues=key, seasons=_sd_season(season)).read_schedule()
            except Exception as e:  # noqa: BLE001
                print(f"  {code} {season}: {type(e).__name__}: {str(e)[:90]}")
                continue
            sch = sch.reset_index()
            sch["src_league"] = code
            sch["season"] = season
            frames.append(sch)
            played = int(sch.get("is_result", pd.Series(dtype=bool)).sum())
            print(f"  {code} {season}: {len(sch)} matches ({played} played)")
    if not frames:
        raise SystemExit("no match data")
    out = pd.concat(frames, ignore_index=True)
    keep = ["src_league", "season", "game_id", "date", "home_team", "away_team",
            "home_goals", "away_goals", "home_xg", "away_xg", "is_result", "url"]
    out = out[[c for c in keep if c in out.columns]]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "understat_matches.csv", index=False)
    print(f"\nwrote understat_matches.csv  ({len(out)} rows, "
          f"{int(out['is_result'].sum())} with xG)")


def pull_players(seasons: list[str]) -> None:
    import soccerdata as sd

    frames = []
    for code, key in LEAGUE_KEY.items():
        for season in seasons:
            try:
                ps = sd.Understat(leagues=key, seasons=_sd_season(season)).read_player_season_stats()
            except Exception as e:  # noqa: BLE001
                print(f"  {code} {season}: {type(e).__name__}: {str(e)[:90]}")
                continue
            ps = ps.reset_index()
            ps["src_league"] = code
            ps["season"] = season
            frames.append(ps)
            print(f"  {code} {season}: {len(ps)} players")
    if not frames:
        raise SystemExit("no player data")
    out = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "understat_player_season.csv", index=False)
    print(f"\nwrote understat_player_season.csv  ({len(out)} rows, "
          f"{out['player'].nunique()} players)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--what", choices=["matches", "players", "both"], default="both")
    args = ap.parse_args()

    if args.what in ("matches", "both"):
        print("[matches]")
        pull_matches(args.seasons)
    if args.what in ("players", "both"):
        print("[players]")
        pull_players(args.seasons)


if __name__ == "__main__":
    main()
