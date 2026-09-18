"""Per-match starting-XI value rollups, built for free out of the lineup
block pull_fotmob_players.py already caches alongside its per-player stats
(zero extra requests - see that file's docstring).

For each cached match, resolves the 11 starters (+ bench) on each side to a
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

Needs data/raw/live/fotmob_lineups/*.json (pull_fotmob_players.py) and
data/processed/fotmob_match_meta.csv (same script) to resolve each match's
season/league/date without re-hitting FotMob.

Run:  py -3.11 src/build_match_lineups.py
Output: data/processed/match_lineup_features.csv   one row per (match, team)
        data/processed/match_lineup_players.csv    one row per (match, team, player)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
LINEUP_CACHE = ROOT / "data" / "raw" / "live" / "fotmob_lineups"
META = PROC / "fotmob_match_meta.csv"
OUT_TEAM = PROC / "match_lineup_features.csv"
OUT_PLAYERS = PROC / "match_lineup_players.csv"


def _resolve(players: list[dict], vmap: dict) -> list[dict]:
    out = []
    for p in players:
        pk = _norm_name(p.get("name"))
        out.append({"fotmob_id": p.get("id"), "player": p.get("name"), "_pk": pk,
                    "age": pd.to_numeric(p.get("age"), errors="coerce"),
                    "predicted_eur": vmap.get(pk)})
    return out


def main() -> None:
    if not LINEUP_CACHE.exists() or not META.exists():
        raise SystemExit(f"{LINEUP_CACHE} / {META} not found - "
                          "run src/pull_fotmob_players.py first")

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

    team_rows, player_rows = [], []
    n_no_meta = n_empty = 0
    for f in sorted(LINEUP_CACHE.glob("*.json")):
        mid = f.stem
        if mid not in meta.index:
            n_no_meta += 1
            continue
        m = meta.loc[mid]
        season, src_league, date = m["season"], m["src_league"], m["date"]
        lineup = json.loads(f.read_text())
        if not lineup.get("homeTeam") and not lineup.get("awayTeam"):
            n_empty += 1
            continue
        vmap = v_by_sl.get((season, src_league), {})

        for side_key in ("homeTeam", "awayTeam"):
            side = lineup.get(side_key) or {}
            starters, subs = side.get("starters") or [], side.get("subs") or []
            if not starters:
                continue
            team_name = side.get("name")
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
            for r in st:
                player_rows.append({"match_id": mid, "season": season, "src_league": src_league,
                                    "date": date, "team_key": team_key, "_pk": r["_pk"],
                                    "player": r["player"], "fotmob_id": r["fotmob_id"],
                                    "is_starter": 1})
            for r in be:
                player_rows.append({"match_id": mid, "season": season, "src_league": src_league,
                                    "date": date, "team_key": team_key, "_pk": r["_pk"],
                                    "player": r["player"], "fotmob_id": r["fotmob_id"],
                                    "is_starter": 0})

    if not team_rows:
        raise SystemExit("no lineups parsed - check the lineup cache isn't empty")

    team_out = pd.DataFrame(team_rows)
    team_out["xi_value_eur"] = team_out["xi_value_eur"].round(0)
    team_out["bench_value_eur"] = team_out["bench_value_eur"].round(0)
    team_out["xi_mean_age"] = team_out["xi_mean_age"].round(1)
    team_out["xi_value_known_frac"] = team_out["xi_value_known_frac"].round(2)
    player_out = pd.DataFrame(player_rows)

    # upsert by match_id, not overwrite: this environment's lineup cache may
    # only cover a subset of matches (e.g. CI's actions/cache, restored fresh
    # each run, vs. a one-off local historical backfill's much larger cache) -
    # a plain overwrite would silently regress the committed file back down
    # to whatever this run's cache happens to hold.
    if OUT_TEAM.exists():
        prior = pd.read_csv(OUT_TEAM, dtype={"match_id": str})
        keep = ~prior["match_id"].isin(team_out["match_id"])
        team_out = pd.concat([prior[keep], team_out], ignore_index=True)
    if OUT_PLAYERS.exists():
        prior_p = pd.read_csv(OUT_PLAYERS, dtype={"match_id": str})
        keep_p = ~prior_p["match_id"].isin(player_out["match_id"])
        player_out = pd.concat([prior_p[keep_p], player_out], ignore_index=True)

    PROC.mkdir(parents=True, exist_ok=True)
    team_out.to_csv(OUT_TEAM, index=False)
    player_out.to_csv(OUT_PLAYERS, index=False)

    print(f"wrote {OUT_TEAM}  ({len(team_out)} team-matches, "
          f"{team_out['team_key'].nunique()} teams, seasons {sorted(team_out['season'].unique())})")
    print(f"wrote {OUT_PLAYERS}  ({len(player_out)} player-match rows)")
    print(f"  {n_no_meta} lineup files skipped (no match_meta row), "
          f"{n_empty} skipped (empty lineup)")
    print(f"  median xi_value_known_frac = {team_out['xi_value_known_frac'].median():.0%}")


if __name__ == "__main__":
    main()
