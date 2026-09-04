"""Per-player, per-match contribution (goals / assists / xG / xA / minutes) -
the match-level half of the value<->outcome coupling in build_form_momentum.py.

Understat's match endpoint returns one row per player who featured, per match
(one HTTP fetch per match - slower than the season-total pulls elsewhere in
this pipeline: ~0.5s/match, so a full season is a few minutes and all three
leagues x all seasons is over an hour). Results are cached by soccerdata
(SOCCERDATA_DIR) and this script itself skips seasons already written, so a
second run only fills in what's missing (e.g. new matches this week).

Run (Python 3.11):
    py -3.11 src/pull_understat_player_matches.py                     # all seasons
    py -3.11 src/pull_understat_player_matches.py --seasons 2026-27   # just the live one

Output:
    data/processed/understat_player_matches.csv
        season, src_league, date, game_id, team, player, position, minutes,
        goals, assists, xg, xa, key_passes, shots
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("SOCCERDATA_DIR", str(PROJECT_ROOT / "data" / "understat_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from fbref_common import current_season  # noqa: E402
from live.schema import deaccent  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROCESSED / "understat_player_matches.csv"
LEAGUE_KEY = {"ENG1": "ENG-Premier League", "GER1": "GER-Bundesliga", "ESP1": "ESP-La Liga"}
DEFAULT_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]
KEEP = ["season", "src_league", "date", "game_id", "team", "player", "position",
        "minutes", "goals", "assists", "xg", "xa", "key_passes", "shots"]


def _sd_season(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start}-{start + 1}"


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def pull_one(code: str, key: str, season: str) -> pd.DataFrame | None:
    import soccerdata as sd

    u = sd.Understat(leagues=key, seasons=_sd_season(season))
    sch = u.read_schedule(include_matches_without_data=False).reset_index()
    if sch.empty:
        return None
    df = u.read_player_match_stats(match_id=sch["game_id"].astype(int).tolist())
    df = df.reset_index()
    dates = sch.set_index("game_id")["date"]
    df["date"] = df["game_id"].map(dates)
    df["season"] = season
    df["src_league"] = code
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--force", action="store_true", help="re-pull seasons already in the output file")
    args = ap.parse_args()

    live_season = current_season()
    have = set()
    prior = None
    if OUT.exists() and not args.force:
        prior = pd.read_csv(OUT)
        have = set(zip(prior["src_league"], prior["season"]))

    frames = [prior] if prior is not None else []
    t0 = time.time()
    for code, key in LEAGUE_KEY.items():
        for season in args.seasons:
            # a completed season never changes once pulled, but the season in
            # progress gains new matches every week - always re-pull that one
            if (code, season) in have and season != live_season:
                print(f"  {code} {season}: already have it, skipping (--force to redo)")
                continue
            try:
                df = pull_one(code, key, season)
            except Exception as e:  # noqa: BLE001 - one bad season shouldn't kill the run
                print(f"  {code} {season}: {type(e).__name__}: {str(e)[:100]}")
                continue
            if df is None or df.empty:
                print(f"  {code} {season}: no data yet")
                continue
            df["_pk"] = df["player"].map(_norm)
            df["_tk"] = df["team"].map(_norm)
            frames.append(df[KEEP + ["_pk", "_tk"]])
            elapsed = time.time() - t0
            print(f"  {code} {season}: {len(df)} player-match rows "
                  f"({df['game_id'].nunique()} matches)  [{elapsed / 60:.1f} min elapsed]")

    if not frames:
        raise SystemExit("no player-match data pulled")
    out = pd.concat(frames, ignore_index=True)
    # a match sometimes gets re-pulled across seasons of the same run - keep one
    out = out.drop_duplicates(subset=["src_league", "season", "game_id", "player_id"]
                              if "player_id" in out.columns
                              else ["src_league", "season", "game_id", "player", "team"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} player-match rows, "
          f"{out['game_id'].nunique()} matches, {(time.time() - t0) / 60:.1f} min this run)")


if __name__ == "__main__":
    main()
