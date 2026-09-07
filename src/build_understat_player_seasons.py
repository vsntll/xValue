"""Synthesise Serie A / Ligue 1 player-season rows for
`fbref_player_season_stats.csv` without an FBref scrape.

FBref gates its advanced tables and this project only runs the browser FBref
player-stats scrape for the three original leagues (PL / Bundesliga / La Liga).
Serie A and Ligue 1 were added as full modelled leagues 2026-09-06, so their
player-season rows are built here from the browser-free feeds instead:

    understat_player_season.csv   goals / assists / xG / npxG / xA / shots /
                                  key passes / cards / xgChain / xgBuildup /
                                  minutes / MP  - every season, both leagues
    understat_player_matches.csv  per-appearance rows -> Starts (position != 'Sub')
    wfr_player_advanced.csv       passing / defense / possession / GCA / SoT /
                                  progressive - the worldfootballR mirror,
                                  2020-21..2022-23 only
    fotmob_player_season.csv      current-season SoT / fouls / tackles / int /
                                  blocks / clearances / touches / saves
    value_history.csv             the season's market value (the model target)

Birth year + nationality come from any row the same player already has in
fbref_player_season_stats.csv (a big-5 transfer) or in the mirror; players with
neither keep a blank Born and fall back to the value model's age handling.

Runs AFTER parse_fbref_player_stats.py (which rewrites the file from scratch for
the scraped leagues) and BEFORE build_current_season_stats.py (which then
refreshes 2026-27 for all five leagues uniformly and does the cross-source name
dedup). Idempotent: drops any existing ITA1/FRA1 rows and rebuilds them.

Run:  py -3.11 src/build_understat_player_seasons.py
Output: data/processed/fbref_player_season_stats.csv  (+ ITA1/FRA1 rows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fbref_common import current_season  # noqa: E402
from live.schema import deaccent, normalize_team  # noqa: E402
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
FBREF = PROC / "fbref_player_season_stats.csv"
LEAGUES = ("ITA1", "FRA1")

# understat_player_season column -> fbref column it fills. Mirrors
# build_current_season_stats.US_TO_FBREF; per-90s + a few derived cols are
# computed afterwards.
US_TO_FBREF = {
    "matches": "standard__Playing Time_MP",
    "minutes": "standard__Playing Time_Min",
    "goals": "standard__Performance_Gls",
    "assists": "standard__Performance_Ast",
    "np_goals": "standard__Performance_G-PK",
    "yellow_cards": "standard__Performance_CrdY",
    "red_cards": "standard__Performance_CrdR",
    "shots": "shooting__Standard_Sh",
    "xg": "standard__xG_Expected",
    "np_xg": "standard__npxG_Expected",
    "xa": "standard__xAG_Expected",
    "key_passes": "passing__KP",
}
US_PASSTHROUGH = ["xg", "np_xg", "xa", "np_goals", "shots", "key_passes",
                  "xg_chain", "xg_buildup"]

# The worldfootballR mirror already names its advanced columns exactly the way
# the FBref parser does (passing__ / defense__ / possession__ / gca__ /
# passing_types__ are identical), so those copy straight across. Only standard__
# and shooting__ use the mirror's own scheme - map the few we need.
WFR_EXTRA = {
    "shooting__SoT_Standard": "shooting__Standard_SoT",
    "shooting__SoT_percent_Standard": "shooting__Standard_SoT%",
    "shooting__Dist_Standard": "shooting__Standard_Dist",
    "shooting__G_per_Sh_Standard": "shooting__Standard_G/Sh",
    "shooting__FK_Standard": "shooting__Standard_FK",
    "standard__PKatt": "standard__Performance_PKatt",
    "standard__PK": "standard__Performance_PK",
}

FOTMOB_TO_FBREF = {
    "sot": ["shooting__Standard_SoT"],
    "fouls": ["misc__Performance_Fls"], "fouled": ["misc__Performance_Fld"],
    "tackles": ["defense__Tkl_Tackles", "misc__Performance_TklW"],
    "interceptions": ["defense__Int", "misc__Performance_Int"],
    "blocks": ["defense__Blocks_Blocks"], "clearances": ["defense__Clr"],
    "touches": ["possession__Touches_Touches"], "saves": ["keeper__Performance_Saves"],
}


def _us_pos(p) -> str:
    """Understat's season position string ('D S', 'F M S', 'GK', 'S') -> our
    group. The trailing/leading 'S' is a sub marker; the first real letter is
    the position."""
    letters = [c for c in str(p).upper().split() if c and c != "S"]
    if not letters:
        return "MF"
    return {"G": "GK", "GK": "GK", "D": "DF", "M": "MF", "F": "FW"}.get(letters[0], "MF")


def _born_nation_lookup(f: pd.DataFrame) -> dict:
    """slug key -> (Born, Nation) from every existing fbref row, the mirror, and
    the full worldfootballR TM value dump (2010-22, all big-5 - the widest birth-
    year source we have, reached by slug and by tm_player_id via the scrape)."""
    out: dict[str, tuple] = {}
    src = [f[["player_slug", "Born", "Nation"]]]
    wfr = PROC / "wfr_player_advanced.csv"
    if wfr.exists():
        w = pd.read_csv(wfr, low_memory=False)
        src.append(w.assign(player_slug=w["Player"])[["player_slug", "Born", "Nation"]])
    tm = PROC / "tm_player_values.csv"
    if tm.exists():
        t = pd.read_csv(tm, low_memory=False)
        if "player_dob" in t.columns:
            t["Born"] = pd.to_datetime(t["player_dob"], errors="coerce").dt.year
            src.append(t.rename(columns={"player_name": "player_slug",
                                         "player_nationality": "Nation"})[
                ["player_slug", "Born", "Nation"]])

    # full mirror dump: slug -> (Born, Nation) and tm_player_id -> (Born, Nation)
    rds = ROOT / "data" / "raw" / "tm" / "big5_player_vals.rds"
    by_tmid: dict[str, tuple] = {}
    if rds.exists():
        try:
            import pyreadr
            m = next(iter(pyreadr.read_r(str(rds)).values()))
            m["Born"] = pd.to_datetime(m["player_dob"], errors="coerce").dt.year
            m["tmid"] = m["player_url"].astype(str).str.extract(r"/spieler/(\d+)")[0]
            m["slug"] = (m["player_url"].astype(str)
                         .str.extract(r"transfermarkt\.com/([a-z0-9-]+)/profil")[0]
                         .str.replace("-", " "))
            m = m.sort_values("season_start_year").drop_duplicates("player_url", keep="last")
            src.append(m.rename(columns={"slug": "player_slug",
                                         "player_nationality": "Nation"})[
                ["player_slug", "Born", "Nation"]])
            for r in m[["tmid", "Born", "player_nationality"]].itertuples(index=False):
                if pd.notna(r.tmid):
                    by_tmid[str(r.tmid)] = (r.Born, r.player_nationality)
        except Exception as exc:  # noqa: BLE001 - birth year is a nice-to-have
            print(f"  (full TM mirror unreadable for birth years: {exc})")

    for part in src:
        for _, r in part.dropna(subset=["player_slug"]).iterrows():
            k = _norm_name(r["player_slug"])
            if not k:
                continue
            born = pd.to_numeric(r.get("Born"), errors="coerce")
            cur = out.get(k, (np.nan, np.nan))
            out[k] = (born if pd.notna(born) else cur[0],
                      r["Nation"] if pd.notna(r.get("Nation")) else cur[1])

    # reach 2023-26 players who were in the mirror under another club/name via
    # the Transfermarkt scrape's tm_player_id
    scr = PROC / "tm_values_scraped.csv"
    if scr.exists() and by_tmid:
        s = pd.read_csv(scr, low_memory=False)
        for r in s.itertuples(index=False):
            k = _norm_name(getattr(r, "player_name", ""))
            bn = by_tmid.get(str(getattr(r, "tm_player_id", "")))
            if k and bn and pd.notna(bn[0]) and k not in out:
                out[k] = bn
    return out


def _starts_by_season() -> dict:
    """(season, _pk, _tk) -> count of appearances the player started."""
    pm = pd.read_csv(PROC / "understat_player_matches.csv", low_memory=False)
    pm = pm[pm["src_league"].isin(LEAGUES)]
    started = pm[pm["position"].astype(str) != "Sub"]
    g = started.groupby(["season", "_pk", "_tk"]).size()
    return g.to_dict()


def _club_matches_by_season() -> dict:
    """(season, normalize_team(club)) -> played matches, for Min%."""
    m = pd.read_csv(PROC / "understat_matches.csv", low_memory=False)
    m = m[(m["src_league"].isin(LEAGUES)) & (m["is_result"] == True)]  # noqa: E712
    out: dict[tuple, int] = {}
    for col in ("home_team", "away_team"):
        for (season, tk), n in m.groupby(["season", m[col].map(normalize_team)]).size().items():
            out[(season, tk)] = out.get((season, tk), 0) + int(n)
    return out


def _wfr_rows() -> dict:
    """(season, _pk, _tk) -> mirror row (advanced stats), 2020-22."""
    wfr = PROC / "wfr_player_advanced.csv"
    if not wfr.exists():
        return {}
    w = pd.read_csv(wfr, low_memory=False)
    w = w[w["src_league"].isin(LEAGUES)].copy()
    w["_pk"] = w["Player"].map(_norm_name)
    w["_tk"] = w["Squad"].map(normalize_team)
    w = w.drop_duplicates(subset=["season", "_pk", "_tk"], keep="last")
    return {(r["season"], r["_pk"], r["_tk"]): r for _, r in w.iterrows()}


def _value_lookup() -> dict:
    """(season, _pk) -> market_value_eur from value_history.csv."""
    vh = PROC / "value_history.csv"
    if not vh.exists():
        return {}
    v = pd.read_csv(vh)
    v["_pk"] = v["player_key"].map(_norm_name)
    v = v.dropna(subset=["market_value_eur"])
    v = v.sort_values("market_value_eur").drop_duplicates(["season", "_pk"], keep="last")
    return {(r["season"], r["_pk"]): r["market_value_eur"] for _, r in v.iterrows()}


def build() -> pd.DataFrame:
    cur = current_season()
    f = pd.read_csv(FBREF, low_memory=False)
    cols = list(f.columns)
    f = f[~f["src_league"].isin(LEAGUES)].copy()   # rebuild ITA1/FRA1 from scratch

    us = pd.read_csv(PROC / "understat_player_season.csv", low_memory=False)
    us = us[us["src_league"].isin(LEAGUES)].copy()
    if us.empty:
        print("no Understat Serie A / Ligue 1 rows - nothing to build")
        return f
    for c in list(US_TO_FBREF) + ["xg_chain", "xg_buildup"]:
        us[c] = pd.to_numeric(us[c], errors="coerce")
    us["_pk"] = us["player"].map(_norm_name)
    us["_tk"] = us["team"].map(normalize_team)
    us = us.drop_duplicates(subset=["season", "_pk", "_tk"], keep="last")

    bn = _born_nation_lookup(pd.read_csv(FBREF, low_memory=False))
    starts = _starts_by_season()
    club_mp = _club_matches_by_season()
    wfr = _wfr_rows()
    vals = _value_lookup()
    yr_of = {s: int(s[:4]) for s in us["season"].unique()}

    rows = []
    for _, d in us.iterrows():
        season, tk, pk = d["season"], d["_tk"], d["_pk"]
        mins = d["minutes"] or 0
        n90 = mins / 90.0 if mins else np.nan
        row = {c: np.nan for c in cols}
        row["season"] = season
        row["src_league"] = d["src_league"]
        row["Squad"] = d["team"]
        row["Player"] = deaccent(str(d["player"]))
        row["player_slug"] = deaccent(str(d["player"]))
        row["Pos"] = _us_pos(d["position"])

        for us_col, fb_col in US_TO_FBREF.items():
            row[fb_col] = d[us_col]
        for c in US_PASSTHROUGH:
            row[f"understat__{c}"] = d[c]
        row["standard__Performance_G+A"] = _add(d["goals"], d["assists"])
        row["standard__Performance_PK"] = _sub(d["goals"], d["np_goals"])
        row["standard__Playing Time_90s"] = n90
        row["shooting__90s"] = n90
        row["misc__90s"] = n90
        row["standard__Playing Time_Starts"] = int(starts.get((season, pk, tk), 0))
        cmp_ = club_mp.get((season, tk))
        row["playing_time__Playing Time_Min%"] = (
            round(100 * mins / (cmp_ * 90), 1) if cmp_ else np.nan)
        row["playing_time__Playing Time_MP"] = d["matches"]
        row["playing_time__Playing Time_Min"] = mins
        row["playing_time__Starts_Starts"] = row["standard__Playing Time_Starts"]

        born, nation = bn.get(pk, (np.nan, np.nan))
        if pd.notna(born):
            row["Born"] = born
            row["Age"] = yr_of[season] - born
        if pd.notna(nation):
            row["Nation"] = nation

        w = wfr.get((season, pk, tk))
        if w is not None:
            for c in cols:
                if c.split("__")[0] in ("passing", "passing_types", "defense",
                                        "possession", "gca") and c in w.index:
                    row[c] = w[c]
            for src_c, dst_c in WFR_EXTRA.items():
                if src_c in w.index and pd.notna(w[src_c]) and dst_c in row:
                    row[dst_c] = w[src_c]
            for c in ("Born", "Nation", "Pos"):
                if pd.notna(w.get(c)):
                    row[c] = w[c]
            if pd.notna(pd.to_numeric(w.get("Born"), errors="coerce")):
                row["Age"] = yr_of[season] - int(pd.to_numeric(w["Born"], errors="coerce"))

        for tot, per in (("standard__Performance_Gls", "standard__Per 90 Minutes_Gls"),
                         ("standard__Performance_Ast", "standard__Per 90 Minutes_Ast"),
                         ("standard__xG_Expected", "standard__xG_Per"),
                         ("standard__npxG_Expected", "standard__npxG_Per"),
                         ("standard__xAG_Expected", "standard__xAG_Per")):
            row[per] = (pd.to_numeric(row[tot], errors="coerce") / n90) if n90 else np.nan
        sh = pd.to_numeric(row.get("shooting__Standard_Sh"), errors="coerce")
        if pd.notna(sh) and n90:
            row["shooting__Standard_Sh/90"] = round(sh / n90, 2)

        mv = vals.get((season, pk))
        if mv is not None and pd.notna(mv):
            row["market_value_eur"] = mv
            row["market_value_imputed"] = 0

        rows.append(row)

    add = pd.DataFrame(rows)[cols]
    # a player with a mirror row in one season but not another (mirror stops at
    # 2022-23) still has a fixed birth year / nationality - propagate it across
    # all of that player's rows.
    add["_pk"] = add["player_slug"].map(_norm_name)
    for col in ("Born", "Nation"):
        filled = add.groupby("_pk")[col].transform(
            lambda s: s.ffill().bfill() if s.notna().any() else s)
        add[col] = add[col].where(add[col].notna(), filled)
    yr = add["season"].str[:4].astype(int)
    born_num = pd.to_numeric(add["Born"], errors="coerce")
    add["Age"] = add["Age"].where(add["Age"].notna(), yr - born_num)
    add = add.drop(columns="_pk")
    add = _fill_from_fotmob(add, cur, cols)
    out = pd.concat([f, add], ignore_index=True)
    out.to_csv(FBREF, index=False)

    by = add.groupby(["src_league", "season"]).size()
    print(f"built {len(add)} Serie A / Ligue 1 player-season rows "
          f"({add['market_value_eur'].notna().sum()} with a value, "
          f"{add['Born'].notna().sum()} with a birth year, "
          f"{sum(1 for _ in wfr)} mirror rows available) -> {FBREF.name}")
    for (lg, s), n in by.items():
        print(f"    {lg} {s}: {n}")
    return out


def _fill_from_fotmob(add: pd.DataFrame, cur: str, cols) -> pd.DataFrame:
    fm_path = PROC / "fotmob_player_season.csv"
    if not fm_path.exists():
        return add
    fm = pd.read_csv(fm_path, low_memory=False)
    fm = fm[(fm["season"] == cur) & (fm["src_league"].isin(LEAGUES))].copy()
    if fm.empty:
        return add
    for c in FOTMOB_TO_FBREF:
        fm[c] = pd.to_numeric(fm[c], errors="coerce")
    fm["pk"] = fm["player"].map(_norm_name)
    fm["tk"] = fm["team"].map(normalize_team)
    by_kt = {(r.pk, r.tk): r for r in fm.itertuples(index=False)}
    valid = {k: [c for c in v if c in cols] for k, v in FOTMOB_TO_FBREF.items()}
    filled = 0
    for i in add.index:
        if add.at[i, "season"] != cur:
            continue
        r = by_kt.get((_norm_name(add.at[i, "player_slug"]),
                       normalize_team(add.at[i, "Squad"])))
        if r is None:
            continue
        n90 = pd.to_numeric(add.at[i, "standard__Playing Time_90s"], errors="coerce")
        for fm_col, fb_cols in valid.items():
            v = getattr(r, fm_col)
            if pd.notna(v):
                for fb_col in fb_cols:
                    add.at[i, fb_col] = v
        sot = getattr(r, "sot")
        sh = pd.to_numeric(add.at[i, "shooting__Standard_Sh"], errors="coerce")
        if pd.notna(sot) and sh:
            add.at[i, "shooting__Standard_SoT%"] = round(100 * sot / sh, 1)
        if pd.notna(sot) and pd.notna(n90) and n90:
            add.at[i, "shooting__Standard_SoT/90"] = round(sot / n90, 2)
        filled += 1
    print(f"  filled {filled} current-season rows' SoT/fouls/tackles/... from FotMob")
    return add


def _sub(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a - b) if pd.notna(a) and pd.notna(b) else np.nan


def _add(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a + b) if pd.notna(a) and pd.notna(b) else np.nan


if __name__ == "__main__":
    build()
