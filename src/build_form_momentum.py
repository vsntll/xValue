"""Closes the value<->outcome loop the other way round.

The value model already feeds the outcome model through the squad-value
rollup - but that's static and season-level. This adds a dynamic signal: fit
a peer baseline ("given this predicted market value, position and age, what
attacking output does a player like this typically produce per 90?"), then
compare each player's REAL recent match-level output (Understat) to that
baseline. The gap is a form/momentum read: a squad whose valuable players are
outperforming what their price implies lately is worth an extra edge; one
whose stars are quietly underperforming isn't - and unlike the season-level
squad value, this moves week to week.

  1. peer baseline: contribution/90 = npxG/90 + 0.7*xA/90 (season totals),
     regressed on log(predicted value) + position + age, fit across every
     real (non-imputed) player-season in the whole panel.
  2. recent actual: from understat_player_matches.csv, each player's trailing,
     pre-match (leak-free) rolling mean of match-level contribution/90 over
     his last 6 appearances.
  3. gap = recent - baseline, minutes-weighted up across the players who
     actually featured in a given match, for the team that fielded them -
     one momentum value per (team, match).

Needs data/processed/understat_player_matches.csv - only a partial pull is
fine, a (team, date) with no player-match data simply gets no momentum value
(build_match_model_table.py leaves it NaN, same as any other missing feature).

Run:  py -3.11 src/build_form_momentum.py
Output: data/processed/squad_momentum.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import deaccent, normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
ROLL_N = 6                # trailing matches for "recent form"
MIN_MATCH_MINUTES = 10     # ignore cameo appearances when computing match contribution
FEAT_NUM, FEAT_CAT = ["log_value", "Age"], ["pos", "src_league"]


def _pk(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def build_baseline() -> Pipeline:
    """contribution_per90 ~ log(predicted value) + position + age, fit on the
    full historical player-season panel (real values only)."""
    v = pd.read_csv(PROC / "value_model_predictions.csv")
    v = v[v["value_imputed"] == 0]
    s = pd.read_csv(PROC / "fbref_player_season_stats.csv", low_memory=False)
    s["_n90"] = pd.to_numeric(s["standard__Playing Time_90s"], errors="coerce")
    s = s[s["_n90"] >= 3]  # need a real sample before trusting a season rate
    s["contribution_per90"] = (
        pd.to_numeric(s["understat__np_xg"], errors="coerce").fillna(0)
        + 0.7 * pd.to_numeric(s["understat__xa"], errors="coerce").fillna(0)
    ) / s["_n90"]

    j = s.merge(v[["season", "src_league", "Player", "Squad", "predicted_eur"]],
               on=["season", "src_league", "Player", "Squad"], how="inner")
    j = j.dropna(subset=["contribution_per90", "predicted_eur", "Age"])
    j["pos"] = j["Pos"].astype(str).str.split(",").str[0].replace({"": "MF"})
    j["log_value"] = np.log1p(j["predicted_eur"])

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEAT_CAT)],
                            remainder="passthrough")
    model = Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        min_samples_leaf=30, random_state=0))])
    X = j[FEAT_NUM + FEAT_CAT]
    model.fit(X, j["contribution_per90"])
    print(f"baseline: fit on {len(j)} player-seasons, R2(in-sample)={model.score(X, j['contribution_per90']):.3f}")
    return model


def main() -> None:
    pm_path = PROC / "understat_player_matches.csv"
    if not pm_path.exists():
        raise SystemExit(f"{pm_path} not found - run src/pull_understat_player_matches.py first")

    model = build_baseline()

    pm = pd.read_csv(pm_path)
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm = pm.dropna(subset=["date"])
    pm["minutes"] = pd.to_numeric(pm["minutes"], errors="coerce").fillna(0)
    pm = pm[pm["minutes"] >= MIN_MATCH_MINUTES].copy()
    if "_pk" not in pm.columns:
        pm["_pk"] = pm["player"].map(_pk)
    # cap a single-match blowout (a cameo hat-trick) so it can't dominate a
    # 6-match rolling mean
    pm["match_contribution_per90"] = ((
        pd.to_numeric(pm["xg"], errors="coerce").fillna(0)
        + 0.7 * pd.to_numeric(pm["xa"], errors="coerce").fillna(0)
    ) / pm["minutes"] * 90).clip(upper=6)

    pm = pm.sort_values(["_pk", "date"])
    pm["recent_contribution"] = (
        pm.groupby("_pk")["match_contribution_per90"]
          .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=2).mean())
    )

    # each match uses the PLAYER'S value/age/position for THAT season
    v = pd.read_csv(PROC / "value_model_predictions.csv")
    v["_pk"] = v["Player"].map(_pk)
    vbase = (v[["season", "src_league", "_pk", "predicted_eur", "age", "pos"]]
             .rename(columns={"age": "Age"})
             .drop_duplicates(subset=["season", "src_league", "_pk"]))
    pm = pm.merge(vbase, on=["season", "src_league", "_pk"], how="left")
    pm["log_value"] = np.log1p(pm["predicted_eur"])

    have = pm.dropna(subset=FEAT_NUM + FEAT_CAT + ["recent_contribution"]).copy()
    have["baseline"] = model.predict(have[FEAT_NUM + FEAT_CAT])
    have["gap"] = have["recent_contribution"] - have["baseline"]
    have["team_key"] = have["team"].map(normalize_team)

    sq = have.groupby(["season", "src_league", "team_key", "date", "game_id"]).apply(
        lambda g: pd.Series({"squad_momentum": np.average(g["gap"], weights=g["minutes"]),
                             "n_players": len(g)}),
        include_groups=False,
    ).reset_index()
    sq["squad_momentum"] = sq["squad_momentum"].round(4)
    sq.to_csv(PROC / "squad_momentum.csv", index=False)
    print(f"wrote {PROC / 'squad_momentum.csv'}  ({len(sq)} team-match rows, "
          f"{sq['team_key'].nunique()} teams, seasons {sorted(sq['season'].unique())}, "
          f"from {have['_pk'].nunique()} player-match rows with a baseline)")


if __name__ == "__main__":
    main()
