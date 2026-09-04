"""Step 6 - squad rollup: aggregate the player table to one row per (team, season).

Feeds the outcome model a cheap proxy for squad strength (total market value,
value of the most-used XI, age profile, attacking output). Per-match starting-XI
rollups would need lineup data we only have for 2026-27, so this is season-level.

Run:  py -3.11 src/build_squad_features.py
Output: data/processed/squad_season_features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
SRC = PROC / "fbref_player_season_stats.csv"
OUT = PROC / "squad_season_features.csv"


def main() -> None:
    d = pd.read_csv(SRC, low_memory=False)
    d["min"] = pd.to_numeric(d["standard__Playing Time_Min"], errors="coerce").fillna(0)
    d["age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["mv"] = pd.to_numeric(d["market_value_eur"], errors="coerce")
    d["xg"] = pd.to_numeric(d["understat__xg"], errors="coerce")
    d["xa"] = pd.to_numeric(d["understat__xa"], errors="coerce")
    d["team_key"] = d["Squad"].map(normalize_team)

    rows = []
    for (season, league, team), g in d.groupby(["season", "src_league", "Squad"]):
        g = g.sort_values("min", ascending=False)
        core = g.head(18)  # ~ rotation squad
        xi = g.head(11)
        rows.append({
            "season": season, "src_league": league, "team": team,
            "team_key": normalize_team(team),
            "squad_value_eur": g["mv"].sum(skipna=True),
            "core18_value_eur": core["mv"].sum(skipna=True),
            "xi_value_eur": xi["mv"].sum(skipna=True),
            "value_known_frac": g["mv"].notna().mean(),
            "mean_age_wtd": np.average(g["age"].fillna(g["age"].mean()),
                                       weights=g["min"] + 1),
            "squad_xg": g["xg"].sum(skipna=True),
            "squad_xa": g["xa"].sum(skipna=True),
            "n_players": len(g),
        })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} team-seasons)")
    print(out.groupby("src_league").agg(
        n=("team", "size"),
        median_squad_value_m=("squad_value_eur", lambda s: round(s.median() / 1e6)),
    ).to_string())
    print("\ntop squads by value:")
    print(out.nlargest(5, "squad_value_eur")[
        ["season", "team", "squad_value_eur", "value_known_frac"]].to_string())


if __name__ == "__main__":
    main()
