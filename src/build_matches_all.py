"""Union every match source into one modelling table: data/processed/matches_all.csv

One row per match, all competitions, 2020-21 .. 2026-27:

  league history  football-data.co.uk (match_features.csv) + Understat xG
  cup / Europe    FBref team match logs (fbref_team_matchlogs.csv), deduped to
                  one row per match  (no shot detail - FBref gates it for cups)
  2026-27 (all)   live_matches_<season>.csv  (ESPN stats + Understat/FotMob xG)

Cross-source team names are reconciled through live/schema.normalize_team.

Run:  py -3.11 src/build_matches_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "matches_all.csv"

COLS = [
    "season", "competition_type", "comp", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "HPoss", "APoss", "HxG", "AxG", "source",
]
DIV_LEAGUE = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga"}
DIV_CODE = {"E0": "ENG1", "D1": "GER1", "SP1": "ESP1"}
COMP_TO_CODE = {"Premier League": "ENG1", "Bundesliga": "GER1", "La Liga": "ESP1"}


def _key(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h = df["HomeTeam"].map(normalize_team)
    a = df["AwayTeam"].map(normalize_team)
    return df["season"].astype(str) + "|" + df["competition_type"].astype(str) + "|" + d + "|" + h + "|" + a


def _clean_opponent(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"^[a-z]{2,3} ", "", regex=True)


def league_history() -> pd.DataFrame:
    mf = pd.read_csv(PROC / "match_features.csv")
    mf = mf[mf["season"] != "2026-27"].copy()
    mf["competition_type"] = "league"
    mf["comp"] = mf["Div"].map(DIV_LEAGUE)
    mf["src_league"] = mf["Div"].map(DIV_CODE)
    mf["source"] = "football-data.co.uk"

    us = pd.read_csv(PROC / "understat_matches.csv")
    us["_k"] = (us["src_league"] + "|" + us["season"].astype(str) + "|"
                + pd.to_datetime(us["date"]).dt.strftime("%Y-%m-%d") + "|"
                + us["home_team"].map(normalize_team) + "|" + us["away_team"].map(normalize_team))
    xg = us.set_index("_k")[["home_xg", "away_xg"]].to_dict("index")

    mf["_k"] = (mf["src_league"] + "|" + mf["season"].astype(str) + "|"
                + pd.to_datetime(mf["Date"]).dt.strftime("%Y-%m-%d") + "|"
                + mf["HomeTeam"].map(normalize_team) + "|" + mf["AwayTeam"].map(normalize_team))
    mf["HxG"] = mf["_k"].map(lambda k: xg.get(k, {}).get("home_xg"))
    mf["AxG"] = mf["_k"].map(lambda k: xg.get(k, {}).get("away_xg"))
    got = mf["HxG"].notna().mean()
    print(f"league history: {len(mf)} rows, xG matched on {got:.0%}")
    return mf.reindex(columns=COLS)


def cup_history() -> pd.DataFrame:
    tm = pd.read_csv(PROC / "fbref_team_matchlogs.csv")
    tm = tm[(tm["competition_type"] != "league") & (tm["season"] != "2026-27")].copy()
    tm = tm[tm["GF"].notna()]  # played only
    tm["Opponent"] = _clean_opponent(tm["Opponent"])

    home_is_team = tm["Venue"].eq("Home")
    tm["HomeTeam"] = tm["team"].where(home_is_team, tm["Opponent"])
    tm["AwayTeam"] = tm["Opponent"].where(home_is_team, tm["team"])
    tm["FTHG"] = tm["GF"].where(home_is_team, tm["GA"])
    tm["FTAG"] = tm["GA"].where(home_is_team, tm["GF"])
    # neutral venue: order the pair deterministically so both perspectives collapse
    neu = tm["Venue"].eq("Neutral")
    swap = neu & (tm["team"].map(normalize_team) > tm["Opponent"].map(normalize_team))
    tm.loc[swap, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]] = tm.loc[
        swap, ["AwayTeam", "HomeTeam", "FTAG", "FTHG"]].values

    tm["comp"] = tm["Comp"]
    tm["source"] = "fbref"
    tm["_k"] = _key(tm)
    tm = tm.drop_duplicates(subset="_k")
    print(f"cup/Europe history: {len(tm)} matches "
          f"({tm['competition_type'].value_counts().to_dict()})")
    return tm.reindex(columns=COLS)


def live_season() -> pd.DataFrame:
    f = PROC / "live_matches_2026-27.csv"
    if not f.exists():
        print("(no live_matches_2026-27.csv - skipping current season)")
        return pd.DataFrame(columns=COLS)
    lv = pd.read_csv(f)
    lv = lv[lv["FTHG"].notna()].copy()  # played
    lv["comp"] = lv["league"]
    print(f"2026-27 live: {len(lv)} played matches")
    return lv.reindex(columns=COLS)


def main() -> None:
    parts = [league_history(), cup_history(), live_season()]
    allm = pd.concat(parts, ignore_index=True)
    allm["_k"] = _key(allm)
    allm = allm.drop_duplicates(subset="_k").drop(columns="_k")
    # derive FTR where missing
    h, a = pd.to_numeric(allm["FTHG"], errors="coerce"), pd.to_numeric(allm["FTAG"], errors="coerce")
    allm["FTR"] = allm["FTR"].where(allm["FTR"].notna(),
                                    pd.Series("H", index=allm.index).where(h > a,
                                    pd.Series("A", index=allm.index).where(h < a, "D")))
    allm.loc[h.isna() | a.isna(), "FTR"] = pd.NA
    allm = allm.sort_values(["season", "competition_type", "Date"]).reset_index(drop=True)
    allm.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}  ({len(allm)} matches, {allm.shape[1]} cols)")
    print("\nby competition_type:")
    print(allm["competition_type"].value_counts().to_string())
    print(f"\nxG coverage: {allm['HxG'].notna().mean():.0%}  |  "
          f"shot stats: {allm['HS'].notna().mean():.0%}  |  "
          f"possession: {allm['HPoss'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
