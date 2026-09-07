"""Refresh the CURRENT season's rows in fbref_player_season_stats.csv from
Understat, so the every-other-day pipeline doesn't wait on a manual,
browser-gated FBref scrape for goals / assists / minutes / xG to go stale.

FBref stays the source of record for finished seasons and for the deep stats
(shots on target, fouls, passing / defense / possession / GCA / keeper) that only
a full FBref parse brings - run pull_fbref_player_stats.py + parse_fbref_player_stats.py
monthly for those. Between those runs this script rewrites ONLY the current season
and ONLY the volatile counting columns, keeping each player's birth year,
nationality, position and last-known deep stats from the most recent FBref parse.

Also drops any club that has not been top-flight during the settled window
(2020-21 .. last complete season) - promoted / relegated churn is often wrong in
the current season's feeds and 2nd-tier sides aren't in scope. See
fbref_common.top_flight_clubs().

Sources (all browser-free, pulled in the same workflow):
  understat_player_season.csv    goals / np_goals / assists / xg / np_xg / xa /
                                 shots / key_passes / cards / xg_chain / xg_buildup
  understat_player_matches.csv   per-appearance rows -> Starts (position != 'Sub')
  understat_matches.csv          played-match count per club -> Min%, and the
                                 top-flight club list

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
from fbref_common import current_season, top_flight_clubs  # noqa: E402
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

# FotMob per-season column -> the fbref column it fills (the stats Understat
# doesn't carry). Totals; per-90s recomputed after. src/pull_fotmob_players.py.
FOTMOB_TO_FBREF = {
    "sot": "shooting__Standard_SoT",
    "fouls": "misc__Performance_Fls", "fouled": "misc__Performance_Fld",
    "tackles": "defense__Tkl_Tackles", "interceptions": "defense__Int",
    "blocks": "defense__Blocks_Blocks", "clearances": "defense__Clr",
    "touches": "possession__Touches_Touches", "saves": "keeper__Performance_Saves",
}


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

    # Understat lists exactly the players who have appeared - which is the right
    # current-season universe: a transfer out of the big three (Salah -> Saudi),
    # an all-season injury (Saliba), or an unused squad player genuinely has no
    # current stats. Trust it; the only safety check is a sanity floor in case a
    # pull half-failed.
    add_ok = len(us) >= 600
    if not add_ok:
        print(f"  understat {cur} has only {len(us)} players - looks like a partial "
              f"pull, refreshing existing rows only and adding none")

    starts = _starts(cur)
    club_mp = _club_matches(cur)

    # Understat lookups: (name, team) first, then name-only for a club-string
    # mismatch or a just-completed transfer, then a same-club fuzzy match for a
    # name form that differs between sources (Kylian Mbappe vs Mbappe-Lottin,
    # Ezri Konsa vs Ezri Konsa Ngoyo - one name's tokens are a subset of the
    # other's). The fuzzy step keeps my rule-2 additions from re-creating a
    # player FBref already has under a slightly different spelling.
    us["nk"] = us["pk"]  # keep the name key reachable after set_index
    us_by_kt = us.set_index(["pk", "tk"])
    us_by_k = us.drop_duplicates("pk", keep="last").set_index("pk")
    us_by_club: dict[str, list] = {}
    for _, u in us.iterrows():
        us_by_club.setdefault(u["tk"], []).append(u)

    def _fuzzy_same(a: str, b: str) -> bool:
        """same player, name form differs across sources - one token set is fully
        contained in the other, they share >=2 tokens, and either the first or
        the last token agrees (a dropped middle name: "Destiny Udogie" vs
        "Iyenoma Destiny Udogie"; or a compound surname: "Kylian Mbappe" vs
        "Kylian Mbappe Lottin")."""
        ta, tb = a.split(), b.split()
        sa, sb = set(ta), set(tb)
        return (len(sa) >= 2 and len(sb) >= 2 and (sa <= sb or sb <= sa)
                and len(sa & sb) >= 2 and (ta[0] == tb[0] or ta[-1] == tb[-1]))

    def _us_row(pk, tk, allow_name_only: bool):
        if (pk, tk) in us_by_kt.index:
            v = us_by_kt.loc[(pk, tk)]
            return v.iloc[0] if isinstance(v, pd.DataFrame) else v
        for u in us_by_club.get(tk, []):
            if _fuzzy_same(pk, u["pk"]):
                return u
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
    rows, used, xref = [], set(), []   # xref: (understat_name, fbref_name) for a non-exact match
    for row in fcur[f.columns].to_dict("records"):
        pk, tk = _norm_name(row["player_slug"]), normalize_team(row["Squad"])
        no_min = pd.to_numeric(pd.Series([row["standard__Playing Time_Min"]]),
                               errors="coerce").fillna(0).iat[0] == 0
        u = _us_row(pk, tk, allow_name_only=no_min)
        if u is not None:
            if _norm_name(u["player"]) != pk and pd.notna(row.get("Player")):
                xref.append((str(u["player"]), str(row["Player"])))
            _apply(row, u, tk)
            used.add(u["nk"])
        rows.append(row)

    # 2. Understat players FBref has no current row for at all -> new rows.
    # Understat lists whoever actually played, so a name it has and FBref doesn't
    # is a real signing (Bruno Guimaraes -> Arsenal) or a debutant - add them, as
    # long as the club is a genuine top-flight side (top_flight_clubs()) and the
    # pull isn't obviously partial.
    top = top_flight_clubs()
    for k in us_by_k.index:
        if k in used or not add_ok:
            continue
        u = us_by_k.loc[k]
        tk = normalize_team(u["team"])
        if tk not in top:
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
    # a stale FBref club + Understat's current one can leave a player under two
    # clubs - keep the row with more minutes (the club they actually play for).
    updated["_m"] = pd.to_numeric(updated["standard__Playing Time_Min"], errors="coerce").fillna(0)
    updated = (updated.sort_values("_m", ascending=False)
                      .drop_duplicates("player_slug", keep="first").drop(columns="_m"))
    # same player, different name form across sources ("Kylian Mbappe" vs
    # "Kylian Mbappe-Lottin", "Jose Gaya" vs "Jose Luis Gaya"): within a club,
    # cluster fuzzy-equal names, keep the fullest row (Born / deep stats) but
    # relabel it to the SHORTEST name in the cluster - that's the common form.
    updated["_n"] = updated.notna().sum(axis=1)
    updated["_tk"] = updated["Squad"].map(normalize_team)
    drop_idx, canon = set(), {}   # canon: any variant name -> the chosen one
    for _tk, g in updated.groupby("_tk"):
        clusters: list[list] = []
        for i in g.index:
            ki = _norm_name(updated.at[i, "player_slug"])
            for cl in clusters:
                if any(_fuzzy_same(ki, _norm_name(updated.at[j, "player_slug"])) for j in cl):
                    cl.append(i)
                    break
            else:
                clusters.append([i])
        for cl in clusters:
            if len(cl) < 2:
                continue
            keep_i = max(cl, key=lambda j: updated.at[j, "_n"])
            best = min((updated.at[j, "Player"] for j in cl),
                       key=lambda n: (len(str(n).split()), len(str(n))))
            for j in cl:
                canon[str(updated.at[j, "Player"])] = best
                if j != keep_i:
                    drop_idx.add(j)
            updated.at[keep_i, "Player"] = best
    updated = updated.drop(index=drop_idx).drop(columns=["_n", "_tk"])

    # one map: every non-canonical spelling (from a merged cluster or a rule-1
    # fuzzy match that had no dup to merge) -> the canonical name.
    name_map = {us: canon.get(fb, fb) for us, fb in xref}
    name_map.update(canon)
    name_map = {k: v for k, v in name_map.items() if k and v and str(k) != str(v)}
    if drop_idx or name_map:
        print(f"  merged {len(drop_idx)} duplicate name-form rows; "
              f"{len(name_map)} name aliases -> {sorted(set(name_map.values()))[:8]}...")
    # drop 2nd-tier clubs that leak into the current-season feeds, unless the club
    # has actually been top-flight during the observed window
    keep = updated["Squad"].map(normalize_team).isin(top)
    if (~keep).any():
        print(f"  dropped {int((~keep).sum())} current rows at non-top-flight clubs: "
              f"{sorted(updated.loc[~keep, 'Squad'].dropna().unique())}")
    updated = updated[keep]

    updated = _fill_from_fotmob(updated, cur, f.columns)

    # FBref leaves Age blank all of the current season - derive it from Born for
    # every current row (kept or refreshed), matching build_xy's own fallback.
    born = pd.to_numeric(updated["Born"], errors="coerce")
    updated["Age"] = updated["Age"].where(updated["Age"].notna(), yr - born)
    # apply the same top-flight filter to finished seasons (a club relegated
    # years ago but top-flight sometime in 2020-26 stays; one that never was goes)
    hist = hist[hist["Squad"].map(normalize_team).isin(top)]
    out = pd.concat([hist, updated], ignore_index=True)
    out.to_csv(FBREF, index=False)

    # variant spelling -> canonical, so downstream files that carry another
    # source's spelling (player_elo.csv from Understat) show one name per player.
    pd.DataFrame(sorted(name_map.items()), columns=["understat_name", "canonical_name"]).to_csv(
        PROC / "player_name_map.csv", index=False)

    n_new = max(0, len(updated) - fcur["Squad"].map(normalize_team).isin(top).sum())
    print(f"{cur}: {len(updated)} rows ({len(used)} refreshed from Understat, "
          f"~{n_new} new Understat-only), {len(hist)} earlier-season rows kept "
          f"-> {FBREF.name}")
    return out


def _fill_from_fotmob(updated: pd.DataFrame, cur: str, cols) -> pd.DataFrame:
    """Fill the stats Understat doesn't carry (SoT, fouls, tackles / int / blocks /
    clearances, touches, saves) from FotMob's per-match player stats. Matched by
    normalised name + club, with the same fuzzy fallback as the Understat join."""
    fm_path = PROC / "fotmob_player_season.csv"
    if not fm_path.exists():
        return updated
    fm = pd.read_csv(fm_path)
    fm = fm[fm["season"] == cur].copy()
    if fm.empty:
        return updated
    for c in FOTMOB_TO_FBREF:
        fm[c] = pd.to_numeric(fm[c], errors="coerce")
    fm["pk"] = fm["player"].map(_norm_name)
    fm["tk"] = fm["team"].map(normalize_team)
    by_kt = {(r.pk, r.tk): r for r in fm.itertuples(index=False)}
    by_club: dict[str, list] = {}
    for r in fm.itertuples(index=False):
        by_club.setdefault(r.tk, []).append(r)

    tgt = [c for c in FOTMOB_TO_FBREF.values() if c in cols]
    filled = 0
    for i in updated.index:
        pk = _norm_name(updated.at[i, "player_slug"])
        tk = normalize_team(updated.at[i, "Squad"])
        r = by_kt.get((pk, tk))
        if r is None:
            for cand in by_club.get(tk, []):
                if _names_close(pk, cand.pk):
                    r = cand
                    break
        if r is None:
            continue
        n90 = pd.to_numeric(pd.Series([updated.at[i, "standard__Playing Time_90s"]]),
                            errors="coerce").iat[0]
        for fm_col, fb_col in FOTMOB_TO_FBREF.items():
            if fb_col in tgt and pd.notna(getattr(r, fm_col)):
                updated.at[i, fb_col] = getattr(r, fm_col)
        sot, sh = getattr(r, "sot"), pd.to_numeric(updated.at[i, "shooting__Standard_Sh"], errors="coerce")
        gls = pd.to_numeric(updated.at[i, "standard__Performance_Gls"], errors="coerce")
        if "shooting__Standard_SoT%" in cols and pd.notna(sot) and sh:
            updated.at[i, "shooting__Standard_SoT%"] = round(100 * sot / sh, 1)
        if "shooting__Standard_G/Sh" in cols and pd.notna(gls) and sh:
            updated.at[i, "shooting__Standard_G/Sh"] = round(gls / sh, 2)
        if "shooting__Standard_SoT/90" in cols and pd.notna(sot) and n90:
            updated.at[i, "shooting__Standard_SoT/90"] = round(sot / n90, 2)
        filled += 1
    print(f"  filled {filled} rows' FBref-only stats (SoT/fouls/tackles/...) from FotMob")
    return updated


def _names_close(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    sa, sb = set(ta), set(tb)
    return (len(sa) >= 2 and len(sb) >= 2 and (sa <= sb or sb <= sa)
            and len(sa & sb) >= 2 and (ta[0] == tb[0] or ta[-1] == tb[-1]))


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
