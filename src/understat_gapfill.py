"""Stand-in for Understat while it can't be pulled (it has been serving a
"We will come back soon" page since late September 2026). FotMob's committed
per-match player rows (data/processed/fotmob/) carry the same stats Understat
gave us - minutes / goals / assists / shots / xG / npxG / xA / chances created
(= key passes) - so every league match Understat has no result for yet is
rebuilt from FotMob in Understat's own shape, ids and spellings:

  understat_gapfill_player_matches.csv  rows for understat_player_matches.csv
  understat_gapfill_matches.csv         results + xG for understat_matches.csv
  understat_gapfill_player_season.csv   understat_player_season.csv rows with
                                        the gap matches added on top

Ids: a FotMob player / team is mapped to the Understat id it was paired with in
earlier matches (verify_player_matches.py's us_player_id, through the Understat
game it paired with), a match to the Understat fixture with the same team on the
same day. Anything with no Understat counterpart gets a negative id (-FotMob id)
so it can never collide with a real one. Every gap row carries source=fotmob.

Only matches after Understat's last result in that league-season are filled, so
this never duplicates a match Understat has; once Understat publishes a game,
the next build drops its gap rows. The Understat CSVs themselves are never
modified - read them through read() to get Understat + gap-fill. Not filled
(FotMob doesn't carry them): cards, xG chain / buildup; np_goals assumes each
penalty taken (npxG gap / PEN_XG) was scored.

Run:  py -3.11 src/understat_gapfill.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store  # noqa: E402
from live.schema import deaccent  # noqa: E402

PROC = ROOT / "data" / "processed"
PURE = {k: PROC / f"understat_{k}.csv" for k in ("player_matches", "matches", "player_season")}
GAP = {k: PROC / f"understat_gapfill_{k}.csv" for k in PURE}
PM_KEEP = ["season", "src_league", "date", "game_id", "team", "player", "player_id", "team_id",
           "position", "minutes", "goals", "assists", "xg", "xa", "key_passes", "shots"]
SEASON_ADD = {"minutes": "minutes", "goals": "goals", "xg": "xg", "np_xg": "npxg",
              "assists": "assists", "xa": "xa", "shots": "shots", "key_passes": "chances_created"}
POS = {0: "GK", 1: "DC", 2: "MC", 3: "FW"}   # FotMob usualPosition -> an Understat code
PEN_XG = 0.76                               # FotMob's xG for one penalty


def _norm(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def _mode_map(df: pd.DataFrame, key: str, val: str) -> dict:
    """key -> its most common val, keeping only keys whose val is not shared by
    another key (a one-to-one mapping)."""
    m = df.dropna(subset=[key, val]).groupby(key)[val].agg(lambda x: x.mode().iat[0])
    return m[~m.duplicated(keep=False)].to_dict()


def _cutoffs(um: pd.DataFrame) -> dict:
    """(src_league, season) -> date of Understat's last result."""
    done = um[um["is_result"].astype(str) == "True"]
    return pd.to_datetime(done["date"]).dt.normalize().groupby(
        [done["src_league"], done["season"]]).max().to_dict()


def build() -> None:
    um = pd.read_csv(PURE["matches"])
    upm = pd.read_csv(PURE["player_matches"], low_memory=False)
    ups = pd.read_csv(PURE["player_season"], low_memory=False)
    fm = fotmob_store.read("player_matches")
    meta = fotmob_store.read_meta()
    meta = meta[meta["fetched"] == 1].copy()
    meta["kday"] = pd.to_datetime(meta["date"])

    # id maps learnt from matches both sources have
    paired = fm.dropna(subset=["us_player_id", "us_game_id"]).copy()
    pid_of = _mode_map(paired, "fotmob_id", "us_player_id")
    paired = paired.merge(upm[["game_id", "player_id", "team", "team_id"]].rename(
                              columns={"team": "us_team", "team_id": "us_team_id"}),
                          left_on=["us_game_id", "us_player_id"], right_on=["game_id", "player_id"])
    # FotMob team id -> "Understat name\tUnderstat id" (FotMob spells one club
    # several ways across its payloads; its numeric id is the stable key)
    team_of = _mode_map(paired.assign(_t=paired["us_team"] + "\t" + paired["us_team_id"].astype(str)),
                        "team_id", "_t")
    name_of = upm.sort_values("date").drop_duplicates("player_id", keep="last").set_index("player_id")["player"]

    # which FotMob matches Understat has no result for, and their Understat fixture (if listed)
    # only league-seasons Understat hasn't finished (it still lists unplayed
    # fixtures) - a finished season stays exactly as Understat published it
    cut = _cutoffs(um)
    open_ls = set(zip(*um.loc[um["is_result"].astype(str) != "True", ["src_league", "season"]].T.values))
    meta = meta[[(lg, s) in open_ls and d > cut.get((lg, s), pd.Timestamp.min)
                 for lg, s, d in zip(meta["src_league"], meta["season"], meta["kday"])]]
    um["kday"] = pd.to_datetime(um["date"]).dt.normalize()
    fixture = {(t, d): r for r in um.itertuples() for t, d in ((r.home_team, r.kday), (r.away_team, r.kday))}
    # a rescheduled match keeps its original date on Understat's list - each
    # home/away pairing is played once a season, so fall back to the pair
    by_pair = {(r.season, r.home_team, r.away_team): r for r in um.itertuples()
               if str(r.is_result) != "True"}

    def _team(fid, name):
        t = team_of.get(fid)
        return (t.split("\t")[0], int(float(t.split("\t")[1]))) if t else (name, -int(fid))

    lu = fotmob_store.read("lineups")
    side = {(m, h): (tid, n) for m, h, tid, n in
            lu[["match_id", "is_home", "team_id", "team"]].drop_duplicates(["match_id", "is_home"])
            .itertuples(index=False)}
    game_of, match_rows = {}, []
    for r in meta.itertuples():
        mid = f"fotmob:{r.match_id}"
        sides = [side.get((r.match_id, h)) for h in (1, 0)]
        if None in sides:
            continue
        (h_id, _), (a_id, _) = sides
        home, away = _team(*sides[0])[0], _team(*sides[1])[0]
        fx = next((fixture[(t, r.kday + pd.Timedelta(days=o))] for o in (0, -1, 1) for t in (home, away)
                   if (t, r.kday + pd.Timedelta(days=o)) in fixture), None)
        if fx is None:
            fx = by_pair.get((r.season, home, away))
        g = fm[fm["match_id"] == mid]
        xg = pd.to_numeric(g["xg"], errors="coerce").groupby(g["team_id"]).sum()
        game_of[mid] = (int(fx.game_id), fx.date) if fx is not None else (-int(r.match_id), f"{r.date} 00:00:00")
        match_rows.append({
            "src_league": r.src_league, "season": r.season, "game_id": game_of[mid][0],
            "date": game_of[mid][1], "home_team": fx.home_team if fx is not None else home,
            "away_team": fx.away_team if fx is not None else away,
            "home_goals": float(r.home_score), "away_goals": float(r.away_score),
            "home_xg": xg.get(h_id, np.nan), "away_xg": xg.get(a_id, np.nan),
            "is_result": True, "url": fx.url if fx is not None else "", "source": "fotmob"})

    gm = fm[fm["match_id"].isin(game_of)].copy()
    starter = {(f"fotmob:{m}", p): s for m, p, s in lu[["match_id", "fotmob_id", "is_starter"]].itertuples(index=False)}
    teams = [_team(tid, t) for t, tid in zip(gm["team"], gm["team_id"].fillna(0))]
    gm["player_id"] = [int(pid_of.get(f, -f)) for f in gm["fotmob_id"]]
    gm_rows = pd.DataFrame({
        "season": gm["season"], "src_league": gm["src_league"],
        "date": [game_of[m][1] for m in gm["match_id"]], "game_id": [game_of[m][0] for m in gm["match_id"]],
        "team": [t[0] for t in teams], "team_id": [t[1] for t in teams],
        "player": [name_of.get(p, n) for p, n in zip(gm["player_id"], gm["player"])],
        "player_id": gm["player_id"],
        "position": ["Sub" if starter.get((m, f)) == 0 else POS.get(int(p) if pd.notna(p) else 2, "MC")
                     for m, f, p in zip(gm["match_id"], gm["fotmob_id"], gm["position"])],
        **{c: pd.to_numeric(gm[s], errors="coerce").fillna(0).to_numpy() for c, s in
           (("minutes", "minutes"), ("goals", "goals"), ("assists", "assists"), ("xg", "xg"),
            ("xa", "xa"), ("key_passes", "chances_created"), ("shots", "shots"))},
    })
    gm_rows["_pk"], gm_rows["_tk"] = gm_rows["player"].map(_norm), gm_rows["team"].map(_norm)
    gm_rows["source"] = "fotmob"
    gm_rows.to_csv(GAP["player_matches"], index=False)
    pd.DataFrame(match_rows).to_csv(GAP["matches"], index=False)

    # season totals: Understat's own row + the gap matches on top
    add = gm.assign(player_id=gm_rows["player_id"].to_numpy(), team=gm_rows["team"].to_numpy(),
                    team_id=gm_rows["team_id"].to_numpy())
    for c in set(SEASON_ADD.values()) | {"minutes", "goals", "xg", "npxg"}:
        add[c] = pd.to_numeric(add[c], errors="coerce").fillna(0)
    add["_pens"] = ((add["xg"] - add["npxg"]) / PEN_XG).round().clip(lower=0)
    key = ["season", "src_league", "player_id", "team_id"]
    agg = add.groupby(key).agg(**{c: (s, "sum") for c, s in SEASON_ADD.items()},
                               matches=("minutes", "size"), _pens=("_pens", "sum"),
                               player=("player", "last"), team=("team", "last")).reset_index()
    agg["np_goals"] = (agg["goals"] - agg["_pens"]).clip(lower=0)
    base = ups.merge(agg[key], on=key)
    base = base.set_index(key)
    a = agg.set_index(key)
    for c in list(SEASON_ADD) + ["matches", "np_goals"]:
        base[c] = base[c].fillna(0) + a.loc[base.index, c]
    new = a[~a.index.isin(base.index)].reset_index()
    new["player"] = [name_of.get(p, n) for p, n in zip(new["player_id"], new["player"])]
    new["league"] = new["src_league"]
    ps = pd.concat([base.reset_index(), new.drop(columns=["_pens"])], ignore_index=True)
    ps["source"] = "understat+fotmob"
    ps.loc[ps.index >= len(base), "source"] = "fotmob"
    ps[[c for c in list(ups.columns) + ["source"] if c in ps]].to_csv(GAP["player_season"], index=False)

    n_m = len(match_rows)
    print(f"gap-fill: {n_m} matches after Understat's last result "
          f"({sum(g > 0 for g, _ in game_of.values())} on Understat's fixture list), "
          f"{len(gm_rows)} player-match rows ({(gm_rows['player_id'] > 0).mean():.0%} on Understat ids), "
          f"{len(ps)} player-season rows")


def read(kind: str) -> pd.DataFrame:
    """Understat's `kind` table (player_matches / matches / player_season) with
    the FotMob gap-fill applied - what every downstream reader should load."""
    pure = pd.read_csv(PURE[kind], low_memory=False)
    if not GAP[kind].exists():
        return pure
    gap = pd.read_csv(GAP[kind], low_memory=False)
    if gap.empty:
        return pure
    if kind == "player_matches":
        gap = gap[~gap["game_id"].isin(pure["game_id"])]
        return pd.concat([pure, gap], ignore_index=True)
    if kind == "matches":
        done = pure.loc[pure["is_result"].astype(str) == "True", "game_id"]
        gap = gap[~gap["game_id"].isin(done)]
        return pd.concat([pure[~pure["game_id"].isin(gap["game_id"])], gap], ignore_index=True)
    key = ["season", "src_league", "player_id", "team_id"]
    keep = ~pure.set_index(key).index.isin(gap.set_index(key).index)
    return pd.concat([pure[keep], gap], ignore_index=True)


if __name__ == "__main__":
    build()
