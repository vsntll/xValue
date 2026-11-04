"""Cross-check every committed FotMob player-match row against a second source
and write the verdict into the committed CSVs.

Club matches (data/processed/fotmob/player_matches_<season>.csv) get two checks:
  * per player, against Understat's own box scores (understat_player_matches.csv):
    each FotMob match is paired with its Understat game by team + date, each
    player by name (exact, then a fuzzy fallback inside the same match, guarded
    by minutes). Goals, assists and shots must match exactly, minutes within
    MINUTES_TOL (the two count stoppage time differently, so a few minutes apart
    is normal). xG / xA are never compared: each provider runs its own model.
  * per match, against football-data.co.uk's result (match_features.csv):
    FotMob's score must equal it, and no team's players may be credited with
    more goals than the team scored. This one still works while Understat is
    unavailable.

Internationals (fotmob_intl_player_matches.csv) have no second per-player
source, so the check is per match: FotMob's score must equal the martj42
result (international_results.csv), and no team's players may be credited with
more goals than the team scored.

Columns written:
  verified    agree        every check that could run passed
              conflict     a check failed - `conflicts` lists which stats
              fotmob_only  no second source to check against (yet)
  checked_by  ;-joined second sources the row was checked against
  conflicts   ;-joined stat names that disagree (score / team_goals = match level)
  us_game_id / us_player_id  the Understat game + player the row paired with
                              (src/understat_gapfill.py maps ids through these)
  us_goals / us_assists / us_shots / us_minutes  Understat's values (club rows)

Rows are never dropped or overwritten with the other source's numbers -
consumers decide what to do with a conflict. Re-running is idempotent; a
fotmob_only row turns into agree / conflict once Understat has the match.

Run:  py -3.11 src/verify_player_matches.py --current
      py -3.11 src/verify_player_matches.py --seasons 2020-21 2021-22 ...
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store as store  # noqa: E402
from fbref_common import current_season  # noqa: E402
from live.schema import deaccent, normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
UNDERSTAT = PROC / "understat_player_matches.csv"
INTL_ROWS = PROC / "fotmob_intl_player_matches.csv"
INTL_RESULTS = PROC / "international_results.csv"
FD_RESULTS = PROC / "match_features.csv"

EXACT = ["goals", "assists", "shots"]
MINUTES_TOL = 10          # |FotMob - Understat| minutes beyond this is a conflict
FUZZY_MIN_RATIO = 0.6     # name similarity for the fallback pairing
FUZZY_MAX_MIN_GAP = 15    # ... and the paired rows' minutes must be this close
VERIFY_COLS = ["verified", "checked_by", "conflicts", "us_game_id", "us_player_id",
               "us_goals", "us_assists", "us_shots", "us_minutes"]


def _pk(s) -> str:
    if not isinstance(s, str):
        return ""
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def _game_ids(meta: pd.DataFrame, us: pd.DataFrame) -> pd.Series:
    """FotMob match_id -> Understat game_id: either team on the same date (a
    club plays once a day), allowing a day either way for UTC vs local dates."""
    by_team_day = {(tk, d): gid for gid, tk, d in
                   us[["game_id", "_tk2", "_day"]].drop_duplicates().itertuples(index=False)}
    out = {}
    for mid, home, away, day in meta[["match_id", "home", "away", "_day"]].itertuples(index=False):
        keys = [(normalize_team(t), day + pd.Timedelta(days=o))
                for o in (0, -1, 1) for t in (home, away)]
        gid = next((by_team_day[k] for k in keys if k in by_team_day), None)
        if gid is not None:
            out[f"fotmob:{mid}"] = gid
    return pd.Series(out, dtype="float64")


def _fuzzy_pairs(fm_left: pd.DataFrame, us_left: pd.DataFrame) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing of leftover names inside one match: string
    similarity, or a shared name token (\"Alisson Becker\" / \"Alisson\"), and
    minutes close enough that it's plausibly the same appearance."""
    scored = []
    for i, pk_a, min_a in fm_left[["_pk", "minutes"]].itertuples():
        for j, pk_b, min_b in us_left[["_pk", "minutes"]].itertuples():
            if abs(min_a - min_b) > FUZZY_MAX_MIN_GAP:
                continue
            r = SequenceMatcher(None, pk_a, pk_b).ratio()
            if set(pk_a.split()) & set(pk_b.split()):
                r = max(r, 0.75)
            if r >= FUZZY_MIN_RATIO:
                scored.append((r, i, j))
    used_i, used_j, pairs = set(), set(), []
    for _r, i, j in sorted(scored, reverse=True):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            pairs.append((i, j))
    return pairs


def _match_checks(fm: pd.DataFrame, meta: pd.DataFrame, fd: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per FotMob row: was its match found in football-data.co.uk, and which
    match-level checks failed ("score", "team_goals")."""
    by_team_day = {}
    for h, a, day, hg, ag in fd[["HomeTeam", "AwayTeam", "_day", "FTHG", "FTAG"]].itertuples(index=False):
        by_team_day[(normalize_team(h), day)] = (hg, ag)
        by_team_day[(normalize_team(a), day)] = (hg, ag)
    team_goals = fm.groupby(["match_id", "team"])["goals"].sum()
    found, bad = {}, {}
    for mid, home, away, day, hs, as_ in meta[["match_id", "home", "away", "_day",
                                               "home_score", "away_score"]].itertuples(index=False):
        key = f"fotmob:{mid}"
        keys = [(normalize_team(t), day + pd.Timedelta(days=o)) for o in (0, -1, 1) for t in (home, away)]
        sc = next((by_team_day[k] for k in keys if k in by_team_day), None)
        if sc is None or pd.isna(hs) or pd.isna(as_):
            continue
        found[key] = True
        b = []
        if (sc[0], sc[1]) != (hs, as_):
            b.append("score")
        if team_goals.get((key, home), 0) > hs or team_goals.get((key, away), 0) > as_:
            b.append("team_goals")
        bad[key] = ";".join(b)
    return fm["match_id"].map(found).fillna(False).astype(bool), fm["match_id"].map(bad).fillna("")


def verify_club_season(season: str, us_all: pd.DataFrame, meta_all: pd.DataFrame,
                       fd_all: pd.DataFrame) -> pd.Series:
    fm = store.read("player_matches", [season])
    if fm.empty:
        return pd.Series(dtype=int)
    fm = fm.drop(columns=[c for c in VERIFY_COLS if c in fm])
    us = us_all[us_all["season"] == season].copy()
    meta = meta_all[meta_all["season"] == season].copy()
    meta["_day"] = pd.to_datetime(meta["date"])

    fm["_game"] = fm["match_id"].map(_game_ids(meta, us)) if len(us) else np.nan
    fm["_pk"] = fm["player"].map(_pk)
    for c in EXACT:
        fm[c] = pd.to_numeric(fm[c], errors="coerce").fillna(0)
    fm["minutes"] = pd.to_numeric(fm["minutes"], errors="coerce").fillna(0)

    # exact name pairs first, then the fuzzy fallback for what's left per match
    us_idx = {(g, pk): i for i, g, pk in us[["game_id", "_pk"]].itertuples()}
    fm["_us_row"] = [us_idx.get((g, pk)) for g, pk in zip(fm["_game"], fm["_pk"])]
    paired_us = set(fm["_us_row"].dropna().astype(int))
    us_left_by_game = {g: grp for g, grp in us[~us.index.isin(paired_us)].groupby("game_id")}
    for g, grp in fm[fm["_us_row"].isna() & fm["_game"].notna()].groupby("_game"):
        cand = us_left_by_game.get(g)
        if cand is not None:
            for i, j in _fuzzy_pairs(grp, cand):
                fm.at[i, "_us_row"] = j

    hit = fm["_us_row"].notna()
    src = us.loc[fm.loc[hit, "_us_row"].astype(int)]
    for c in EXACT + ["minutes", "game_id", "player_id"]:
        fm[f"us_{c}"] = np.nan
        fm.loc[hit, f"us_{c}"] = pd.to_numeric(src[c], errors="coerce").to_numpy()
    for c in ("us_game_id", "us_player_id"):
        fm[c] = fm[c].astype("Int64")

    bad = {c: hit & (fm[c] != fm[f"us_{c}"]) for c in EXACT}
    bad["minutes"] = hit & ((fm["minutes"] - fm["us_minutes"]).abs() > MINUTES_TOL)
    fd_hit, fd_bad = _match_checks(fm, meta, fd_all[fd_all["season"] == season])
    fm["checked_by"] = [";".join(n for n, h in (("understat", a), ("football-data", b)) if h)
                        for a, b in zip(hit, fd_hit)]
    fm["conflicts"] = [";".join([c for c in bad if bad[c].iat[k]] + ([fd_bad.iat[k]] if fd_bad.iat[k] else []))
                       for k in range(len(fm))]
    fm["verified"] = np.where(fm["checked_by"] == "", "fotmob_only",
                              np.where(fm["conflicts"] != "", "conflict", "agree"))

    # write back only the verdict columns; the FotMob stats stay exactly as pulled
    out = store.read("player_matches", [season])
    out = out.drop(columns=[c for c in VERIFY_COLS if c in out])
    for c in VERIFY_COLS:
        out[c] = fm[c].to_numpy()
    out.to_csv(store.path("player_matches", season), index=False)
    return fm["verified"].value_counts()


def verify_internationals() -> pd.Series | None:
    if not (INTL_ROWS.exists() and INTL_RESULTS.exists() and store.INTL_META.exists()):
        return None
    meta = store.read_meta(store.INTL_META)
    res = pd.read_csv(INTL_RESULTS)
    res["_day"] = pd.to_datetime(res["date"])
    by_pair = {}
    for day, h, a, hs, as_ in res[["_day", "home_team", "away_team", "home_score", "away_score"]] \
            .itertuples(index=False):
        by_pair.setdefault(frozenset((h, a)), []).append((day, {h: hs, a: as_}))

    rows = pd.read_csv(INTL_ROWS)
    rows = rows.drop(columns=[c for c in ("verified", "conflicts") if c in rows])
    goals = (rows.assign(goals=pd.to_numeric(rows["goals"], errors="coerce").fillna(0))
                 .groupby(["match_id", "team"])["goals"].sum())

    verdict, conflicts = {}, {}
    for mid, day, home, away, hs, as_ in meta[["match_id", "date", "home", "away",
                                               "home_score", "away_score"]].itertuples(index=False):
        key = f"fotmob:{mid}"
        day = pd.to_datetime(day)
        cands = [(abs((d - day).days), sc) for d, sc in by_pair.get(frozenset((home, away)), [])]
        cands = [c for c in cands if c[0] <= 2]
        if not cands or pd.isna(hs) or pd.isna(as_):
            verdict[key] = "fotmob_only"
            continue
        sc = min(cands, key=lambda c: c[0])[1]
        bad = []
        if (sc[home], sc[away]) != (hs, as_):
            bad.append("score")
        if goals.get((key, home), 0) > hs or goals.get((key, away), 0) > as_:
            bad.append("goals")
        verdict[key] = "conflict" if bad else "agree"
        conflicts[key] = ";".join(bad)

    rows["verified"] = rows["match_id"].map(verdict).fillna("fotmob_only")
    rows["conflicts"] = rows["match_id"].map(conflicts).fillna("")
    rows.to_csv(INTL_ROWS, index=False)
    store.upsert_meta(pd.DataFrame({"match_id": [k.removeprefix("fotmob:") for k in verdict],
                                    "verified": list(verdict.values())}), store.INTL_META)
    return rows.drop_duplicates("match_id")["verified"].value_counts()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=None)
    ap.add_argument("--current", action="store_true", help="just the season in progress")
    args = ap.parse_args()
    seasons = [current_season()] if args.current or not args.seasons else args.seasons

    us = pd.read_csv(UNDERSTAT, low_memory=False) if UNDERSTAT.exists() else pd.DataFrame(
        columns=["season", "game_id", "team", "date", "player", *EXACT, "minutes"])
    us["_pk"] = us["player"].map(_pk)
    us["_tk2"] = us["team"].map(normalize_team)
    us["_day"] = pd.to_datetime(us["date"]).dt.normalize()
    fd = pd.read_csv(FD_RESULTS) if FD_RESULTS.exists() else pd.DataFrame(
        columns=["season", "HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"])
    fd["_day"] = pd.to_datetime(fd["Date"])
    meta = store.read_meta()
    for season in seasons:
        vc = verify_club_season(season, us, meta, fd)
        print(f"  {season}: " + ", ".join(f"{k} {v}" for k, v in vc.items()))
    vc = verify_internationals()
    if vc is not None:
        print("  internationals (matches): " + ", ".join(f"{k} {v}" for k, v in vc.items()))


if __name__ == "__main__":
    main()
