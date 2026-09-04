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
    """season | competition_type | home | away. No date - sources disagree on
    kickoff day by a timezone, and a given pair meets at most once per comp per
    season per venue (two-legged ties differ by which side is home)."""
    h = df["HomeTeam"].map(normalize_team)
    a = df["AwayTeam"].map(normalize_team)
    return (df["season"].astype(str) + "|" + df["competition_type"].astype(str)
            + "|" + h + "|" + a)


def _clean_opponent(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"^[a-z]{2,3} ", "", regex=True)


def _tracked_by_season() -> dict[str, set]:
    """{season: {normalized club names that played in ENG1/GER1/ESP1 that year}}.
    Scopes cup/European rows to ties involving one of our clubs - matching FBref."""
    mf = pd.read_csv(PROC / "match_features.csv")
    out: dict[str, set] = {}
    for seas, g in mf.groupby("season"):
        clubs = set(g["HomeTeam"].map(normalize_team)) | set(g["AwayTeam"].map(normalize_team))
        out[str(seas)] = clubs
    # current season from Understat (football-data 2026-27 is thin early on)
    try:
        um = pd.read_csv(PROC / "understat_matches.csv")
        for seas, g in um.groupby("season"):
            out.setdefault(str(seas), set()).update(
                set(g["home_team"].map(normalize_team)) | set(g["away_team"].map(normalize_team)))
    except FileNotFoundError:
        pass
    return out


def _involves_tracked(df: pd.DataFrame, tracked: dict[str, set]) -> pd.Series:
    h = df["HomeTeam"].map(normalize_team)
    a = df["AwayTeam"].map(normalize_team)
    return df.apply(lambda r: h[r.name] in tracked.get(str(r["season"]), set())
                    or a[r.name] in tracked.get(str(r["season"]), set()), axis=1)


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


def live_matches(tracked: dict[str, set]) -> pd.DataFrame:
    """Every played row from every live_matches_<season>.csv, scoped to matches
    involving one of our clubs (drops e.g. PSG-Porto in the UCL)."""
    frames = []
    for f in sorted(PROC.glob("live_matches_*.csv")):
        lv = pd.read_csv(f)
        lv = lv[lv["FTHG"].notna()].copy()
        lv["comp"] = lv["league"]
        frames.append(lv)
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    lg = out["competition_type"].eq("league")
    out = out[lg | _involves_tracked(out, tracked)]
    print(f"live snapshots: {len(out)} rows across {sorted(out['season'].unique())}")
    return out.reindex(columns=COLS)


_STAT_FILL = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY",
              "HR", "AR", "HPoss", "APoss", "HxG", "AxG"]


def _enrich(allm: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Fill missing stat/xG fields on allm rows from matching live rows."""
    if live.empty:
        return allm
    live = live.assign(_k=_key(live)).drop_duplicates("_k").set_index("_k")
    allm = allm.assign(_k=_key(allm))
    filled = 0
    for col in _STAT_FILL:
        need = allm[col].isna() & allm["_k"].isin(live.index)
        vals = allm.loc[need, "_k"].map(live[col])
        allm.loc[need, col] = vals
        filled += vals.notna().sum()
    print(f"enrich: filled {filled} stat/xG cells from live snapshots")
    return allm.drop(columns="_k")


def main() -> None:
    tracked = _tracked_by_season()
    live = live_matches(tracked)
    allm = pd.concat([league_history(), cup_history(), live], ignore_index=True)
    allm["_k"] = _key(allm)
    allm = allm.drop_duplicates(subset="_k").drop(columns="_k")
    allm = _enrich(allm, live)
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
