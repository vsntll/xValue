"""Refresh the CURRENT season's rows in fbref_player_season_stats.csv from
Understat, so the every-other-day pipeline doesn't wait on a manual,
browser-gated FBref scrape for goals / assists / minutes / xG to go stale.

FBref stays the source of record for finished seasons and for the deep stats
(shots on target, fouls, passing / defense / possession / GCA / keeper) that only
a full FBref parse brings - run pull_fbref_player_stats.py + parse_fbref_player_stats.py
monthly for those. Between those runs this script rewrites ONLY the current season
and ONLY the volatile counting columns, keeping each player's birth year,
nationality, position and last-known deep stats from the most recent FBref parse.

Sources (all browser-free, pulled in the same workflow):
  understat_player_season.csv    goals / np_goals / assists / xg / np_xg / xa /
                                 shots / key_passes / cards / xg_chain / xg_buildup
  understat_player_matches.csv   per-appearance rows -> Starts (position != 'Sub')
  understat_matches.csv          played-match count per club -> Min%

Run:  py -3.11 src/build_current_season_stats.py
Output: data/processed/fbref_player_season_stats.csv  (current-season rows only)
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

# understat_player_season column -> the fbref column it overwrites. Per-90s and a
# few mirrors are recomputed afterwards, not listed here.
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
                  "xg_chain", "xg_buildup"]  # -> understat__<col>, consumed directly


def _starts(cur: str) -> pd.Series:
    """(_pk, _tk) -> count of appearances that season where the player started
    (understat's per-match position is 'Sub' only when they came off the bench)."""
    pm = pd.read_csv(PROC / "understat_player_matches.csv")
    pm = pm[pm["season"] == cur]
    started = pm[pm["position"].astype(str) != "Sub"]
    return started.groupby(["_pk", "_tk"]).size()


def _club_matches(cur: str) -> dict:
    """normalize_team(club) -> played matches this season, for Min%."""
    m = pd.read_csv(PROC / "understat_matches.csv")
    m = m[(m["season"] == cur) & (m["is_result"] == True)]  # noqa: E712
    played: dict[str, int] = {}
    for col in ("home_team", "away_team"):
        for name, n in m[col].map(normalize_team).value_counts().items():
            played[name] = played.get(name, 0) + int(n)
    return played


def build() -> pd.DataFrame:
    cur = current_season()
    yr = int(cur[:4])
    f = pd.read_csv(FBREF, low_memory=False)
    hist = f[f["season"] != cur]
    fcur = f[f["season"] == cur].copy()

    us = pd.read_csv(PROC / "understat_player_season.csv")
    us = us[us["season"] == cur].copy()
    if us.empty:
        print(f"no Understat rows for {cur} - leaving fbref_player_season_stats.csv untouched")
        return f
    for c in list(US_TO_FBREF) + ["xg_chain", "xg_buildup"]:
        us[c] = pd.to_numeric(us[c], errors="coerce")
    us["pk"] = us["player"].map(_norm_name)
    us["tk"] = us["team"].map(normalize_team)
    us = us.drop_duplicates(subset=["pk", "tk"], keep="last")

    # Understat's squads are chaos for a week or two after the window - a club
    # with a stub roster isn't ready, so don't spawn new rows from it (its real
    # players still get refreshed via an existing FBref row).
    ready = {tk for tk, n in us.groupby("tk").size().items() if n >= 16}
    if len(us) < 900:
        print(f"  understat {cur} has only {len(us)} players - too early/incomplete, "
              f"refreshing existing rows only, adding none")

    starts = _starts(cur)
    club_mp = _club_matches(cur)

    # Understat lookups: (name, team) first, then name-only for a club-string
    # mismatch or a just-completed transfer.
    us["nk"] = us["pk"]  # keep the name key reachable after set_index
    us_by_kt = us.set_index(["pk", "tk"])
    us_by_k = us.drop_duplicates("pk", keep="last").set_index("pk")

    def _us_row(pk, tk, allow_name_only: bool):
        if (pk, tk) in us_by_kt.index:
            v = us_by_kt.loc[(pk, tk)]
            return v.iloc[0] if isinstance(v, pd.DataFrame) else v
        # name-only is a club-string mismatch OR a transfer - only trust it to
        # fill an FBref row that has no minutes yet, never to move a played row.
        if allow_name_only and pk in us_by_k.index:
            return us_by_k.loc[pk]
        return None

    def _apply(row: dict, u, tk: str) -> None:
        """overwrite the volatile counting columns of `row` from Understat row `u`."""
        n90 = (u["minutes"] / 90.0) if u["minutes"] else np.nan
        for us_col, fb_col in US_TO_FBREF.items():
            row[fb_col] = u[us_col]
        for col in US_PASSTHROUGH:
            row[f"understat__{col}"] = u[col]
        row["standard__Performance_PK"] = _sub(u["goals"], u["np_goals"])
        row["standard__Playing Time_Starts"] = int(starts.get((_norm_name(u["player"]), tk), 0))
        row["standard__Playing Time_90s"] = n90
        cmp_ = club_mp.get(tk)
        row["playing_time__Playing Time_Min%"] = (
            round(100 * u["minutes"] / (cmp_ * 90), 1) if cmp_ else np.nan)
        born = pd.to_numeric(row.get("Born"), errors="coerce")
        if pd.notna(born):
            row["Age"] = yr - born
        for tot, per in (("standard__Performance_Gls", "standard__Per 90 Minutes_Gls"),
                         ("standard__Performance_Ast", "standard__Per 90 Minutes_Ast"),
                         ("standard__xG_Expected", "standard__xG_Per"),
                         ("standard__npxG_Expected", "standard__npxG_Per"),
                         ("standard__xAG_Expected", "standard__xAG_Per")):
            row[per] = (row[tot] / n90) if n90 else np.nan

    # 1. every FBref current-season row is kept; refreshed if Understat has it
    prior = (hist.assign(pk=hist["player_slug"].map(_norm_name))
                 .sort_values("season").drop_duplicates("pk", keep="last").set_index("pk"))
    fcur_names = set(fcur["player_slug"].map(_norm_name))
    rows, used = [], set()
    for row in fcur[f.columns].to_dict("records"):
        pk, tk = _norm_name(row["player_slug"]), normalize_team(row["Squad"])
        no_min = pd.to_numeric(pd.Series([row["standard__Playing Time_Min"]]),
                               errors="coerce").fillna(0).iat[0] == 0
        u = _us_row(pk, tk, allow_name_only=no_min)
        if u is not None:
            _apply(row, u, tk)
            used.add(u["nk"])
        rows.append(row)

    # 2. Understat players FBref has no current row for at all -> new rows.
    # Skip anyone whose name already exists in a current FBref row under any club
    # (Understat mislabels squads badly during the transfer window - a real
    # Newcastle player showing up as "Arsenal" must not become a duplicate).
    add_ok = len(us) >= 900
    for k in us_by_k.index:
        if k in used or k in fcur_names or not add_ok:
            continue
        u = us_by_k.loc[k]
        tk = normalize_team(u["team"])
        if tk not in ready:                       # club's Understat roster is a stub
            continue
        row = {c: np.nan for c in f.columns}
        p = prior.loc[u["nk"]] if u["nk"] in prior.index else None
        row["season"], row["src_league"], row["Squad"] = cur, u["src_league"], u["team"]
        row["Player"] = p["Player"] if p is not None else deaccent(str(u["player"]))
        row["player_slug"] = p["player_slug"] if p is not None else deaccent(str(u["player"]))
        for col in ("Nation", "Born"):
            row[col] = p[col] if p is not None else np.nan
        row["Pos"] = p["Pos"] if p is not None and pd.notna(p["Pos"]) else _us_pos(u["position"])
        _apply(row, u, tk)
        rows.append(row)

    updated = pd.DataFrame(rows)[f.columns]
    # a transfer-window mislabel can leave the same player under two clubs - keep
    # the one with more minutes (the real club), drop the ghost.
    updated["_m"] = pd.to_numeric(updated["standard__Playing Time_Min"], errors="coerce").fillna(0)
    updated = (updated.sort_values("_m", ascending=False)
                      .drop_duplicates("player_slug", keep="first").drop(columns="_m"))
    # FBref leaves Age blank all of the current season - derive it from Born for
    # every current row (kept or refreshed), matching build_xy's own fallback.
    born = pd.to_numeric(updated["Born"], errors="coerce")
    updated["Age"] = updated["Age"].where(updated["Age"].notna(), yr - born)
    out = pd.concat([hist, updated], ignore_index=True)
    out.to_csv(FBREF, index=False)
    n_ref = len(used)
    print(f"{cur}: {len(fcur)} FBref rows kept ({n_ref} refreshed from Understat), "
          f"{len(updated) - len(fcur)} new Understat-only rows, "
          f"{len(hist)} earlier-season rows untouched -> {FBREF.name}")
    return out


def _sub(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a - b) if pd.notna(a) and pd.notna(b) else np.nan


def _us_pos(p) -> str:
    """understat position ('D', 'M', 'F', 'GK', 'D M S', 'DC' ...) -> our group.
    Only a last-resort fallback for a signing with no FBref row at all."""
    c = str(p).strip()[:1].upper()
    return {"G": "GK", "D": "DF", "F": "FW"}.get(c, "MF")


if __name__ == "__main__":
    build()
