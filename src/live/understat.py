"""Understat via soccerdata - the free xG source.

Understat covers exactly EPL / La Liga / Bundesliga (+ Serie A, Ligue 1, RFPL),
2014-15 to present, with per-match team xG and per-player season xG. soccerdata's
reader uses a TLS-fingerprint client (no browser) and works from here.

Only the three leagues - **no cup / European coverage** (that xG needs FotMob or
Sofascore). In the merge this source just fills HxG/AxG on league rows.

Cache: ``data/understat_cache/`` (SOCCERDATA_DIR, set below before the import).
"""

from __future__ import annotations

import os
from pathlib import Path

from .schema import LEAGUES, blank_frame, finalize

_PROJECT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("SOCCERDATA_DIR", str(_PROJECT / "data" / "understat_cache"))
os.environ.setdefault("SOCCERDATA_MAXAGE", "30")  # refresh current season weekly-ish

# our comp code -> soccerdata Understat league key
_LEAGUE_KEY = {
    "ENG1": "ENG-Premier League",
    "GER1": "GER-Bundesliga",
    "ESP1": "ESP-La Liga",
}


def fetch(comp_codes: list[str], season: str, with_stats: bool = True) -> "object":
    import soccerdata as sd

    y = int(season.split("-")[0])
    sd_season = f"{y}-{y + 1}"  # a bare year is ambiguous to soccerdata
    rows = []
    for code in comp_codes:
        if code not in _LEAGUE_KEY:
            continue
        cfg = LEAGUES[code]
        try:
            us = sd.Understat(leagues=_LEAGUE_KEY[code], seasons=sd_season)
            sch = us.read_schedule().reset_index()
        except Exception as e:  # noqa: BLE001
            print(f"  understat {code}: {type(e).__name__}: {str(e)[:100]}")
            continue
        print(f"  understat {code}: {len(sch)} matches")
        for _, m in sch.iterrows():
            played = bool(m.get("is_result", False))
            rows.append({
                "season": season, "league": cfg["name"], "tier": cfg["tier"],
                "competition_type": "league", "comp_code": code,
                "Date": str(m["date"])[:10],
                "Time": str(m["date"])[11:16] if len(str(m["date"])) > 11 else None,
                "HomeTeam": m["home_team"], "AwayTeam": m["away_team"],
                "status": "FT" if played else "SCHEDULED",
                "FTHG": m["home_goals"] if played else None,
                "FTAG": m["away_goals"] if played else None,
                "HxG": round(float(m["home_xg"]), 3) if played and m.get("home_xg") is not None else None,
                "AxG": round(float(m["away_xg"]), 3) if played and m.get("away_xg") is not None else None,
                "match_id": f"understat:{m['game_id']}",
            })
    return finalize(rows, "understat") if rows else blank_frame()
