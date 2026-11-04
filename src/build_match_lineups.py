"""Per-match starting-XI value rollups, built for free out of the lineup
block pull_fotmob_players.py already commits alongside its per-player stats
(zero extra requests - see that file's docstring).

For each committed match, resolves the 11 starters (+ bench) on each side to a
player_key, joins against the value model's predicted_eur (predicted_eur is
used rather than the raw listed value since it's populated for every
eligible row, including current-season and thin-minutes players the site
still needs a number for), and rolls up per (match, team):

  xi_value_eur          sum of predicted_eur over the resolved starters
  xi_mean_age           mean age of the XI (FotMob's own age field - always
                         present regardless of whether the player resolved)
  bench_value_eur       sum of predicted_eur over the resolved substitutes
  xi_value_known_frac   share of the 11 starters that resolved to a value -
                         low coverage means xi_value_eur understates the true
                         total; downstream consumers should fall back when
                         this is low, same pattern as market_value_imputed

Also writes a player-level roster (match_lineup_players.csv) so
build_form_momentum.py can restrict its recent-output rollup to players who
actually started, instead of the whole squad that featured.

Needs data/processed/fotmob/lineups_<season>.csv (pull_fotmob_players.py) and
data/processed/fotmob_match_meta.csv (same script) to resolve each match's
season/league/date without re-hitting FotMob.

Run:  py -3.11 src/build_match_lineups.py
Output: data/processed/match_lineup_features.csv   one row per (match, team)
        data/processed/match_lineup_players.csv    one row per (match, team, player)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store  # noqa: E402
from live.schema import normalize_team  # noqa: E402
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
META = PROC / "fotmob_match_meta.csv"
OUT_TEAM = PROC / "match_lineup_features.csv"
OUT_PLAYERS = PROC / "match_lineup_players.csv"


def _resolve(players: list[dict], vmap: dict) -> list[dict]:
    out = []
    for p in players:
        pk = _norm_name(p.get("player"))
        out.append({"fotmob_id": p.get("fotmob_id"), "player": p.get("player"), "_pk": pk,
                    "age": pd.to_numeric(p.get("age"), errors="coerce"),
                    "predicted_eur": vmap.get(pk)})
    return out


def main() -> None:
    if not META.exists():
        raise SystemExit(f"{META} not found - run src/pull_fotmob_players.py first")

    meta = pd.read_csv(META, dtype={"match_id": str}).set_index("match_id")
    meta = meta[~meta.index.duplicated(keep="last")]

    v = pd.read_csv(PROC / "value_model_predictions.csv")
    v["_pk"] = v["Player"].map(_norm_name)
    # real (non-imputed) values win a collision on the same key within a
    # season/league before an imputed baseline does
    v = v.sort_values("value_imputed").drop_duplicates(
        subset=["season", "src_league", "_pk"], keep="first")
    v_by_sl: dict[tuple, dict] = {
        k: dict(zip(g["_pk"], g["predicted_eur"]))
        for k, g in v.groupby(["season", "src_league"])
    }

    lu = fotmob_store.read("lineups")
    n_no_meta = lu.loc[~lu["match_id"].isin(meta.index), "match_id"].nunique()
    lu = lu[lu["match_id"].isin(meta.index)]
    team_rows, player_rows = [], []
    for (mid, _home), side in lu.groupby(["match_id", "is_home"], sort=False):
        m = meta.loc[mid]
        season, src_league, date = m["season"], m["src_league"], m["date"]
        vmap = v_by_sl.get((season, src_league), {})
        starters = side[side["is_starter"] == 1].to_dict("records")
        subs = side[side["is_starter"] == 0].to_dict("records")
        if not starters:
            continue
        team_name = side["team"].iat[0]
        team_key = normalize_team(team_name)

        st, be = _resolve(starters, vmap), _resolve(subs, vmap)
        known = [r["predicted_eur"] for r in st if r["predicted_eur"] is not None]
        bench_known = [r["predicted_eur"] for r in be if r["predicted_eur"] is not None]
        team_rows.append({
            "match_id": mid, "season": season, "src_league": src_league, "date": date,
            "team": team_name, "team_key": team_key,
            "xi_value_eur": sum(known) if known else np.nan,
            "xi_mean_age": np.nanmean([r["age"] for r in st]),
            "bench_value_eur": sum(bench_known) if bench_known else np.nan,
            "xi_value_known_frac": len(known) / len(st),
            "n_starters": len(st), "n_bench": len(be),
        })
        for is_starter, grp in ((1, st), (0, be)):
            for r in grp:
                player_rows.append({"match_id": mid, "season": season, "src_league": src_league,
                                    "date": date, "team_key": team_key, "_pk": r["_pk"],
                                    "player": r["player"], "fotmob_id": r["fotmob_id"],
                                    "is_starter": is_starter})

    if not team_rows:
        raise SystemExit("no lineups - check data/processed/fotmob/lineups_*.csv")

    team_out = pd.DataFrame(team_rows)
    team_out["xi_value_eur"] = team_out["xi_value_eur"].round(0)
    team_out["bench_value_eur"] = team_out["bench_value_eur"].round(0)
    team_out["xi_mean_age"] = team_out["xi_mean_age"].round(1)
    team_out["xi_value_known_frac"] = team_out["xi_value_known_frac"].round(2)
    player_out = pd.DataFrame(player_rows)

    PROC.mkdir(parents=True, exist_ok=True)
    team_out.to_csv(OUT_TEAM, index=False)
    player_out.to_csv(OUT_PLAYERS, index=False)

    print(f"wrote {OUT_TEAM}  ({len(team_out)} team-matches, "
          f"{team_out['team_key'].nunique()} teams, seasons {sorted(team_out['season'].unique())})")
    print(f"wrote {OUT_PLAYERS}  ({len(player_out)} player-match rows)")
    print(f"  {n_no_meta} lineup matches skipped (no match_meta row)")
    print(f"  median xi_value_known_frac = {team_out['xi_value_known_frac'].median():.0%}")


if __name__ == "__main__":
    main()
