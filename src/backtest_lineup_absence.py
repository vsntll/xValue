"""Backtest: does a team whose most valuable player is NOT in the confirmed
starting XI underperform the outcome model?

The outcome model sees no lineups - it's built day(s) ahead - so if the absence
of a team's best player (injury, rotation, suspension) matters, it should show
up as a negative residual: actual points below the model's expected points
(3 x P(win) + P(draw)). If it does, scoring matches again with the confirmed XI
~1h before kickoff (a matchday re-score) is worth building; if not, it isn't.

"Top player" for a match = the highest-valued player (listed value, else the
value model's) who was in the team's matchday squad (XI or bench) at least once
in the previous WINDOW_DAYS - so a January signing isn't "absent" before he
arrived, and a player out for months drops out of the reckoning. Value is that
season's (value_model_predictions.csv); the XI is FotMob's confirmed lineup
(match_lineup_players.csv). Out-of-sample: the outcome model's 2024-26 test set,
pure and hybrid (the hybrid includes market opening odds, which may already
price in early team news).

Run:  py -3.11 src/backtest_lineup_absence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
WINDOW_DAYS = 45
BOOT = 2000


def _values() -> dict:
    v = pd.read_csv(PROC / "value_model_predictions.csv")
    v["val"] = v["market_value_eur"].where(v["value_imputed"] == 0).fillna(v["predicted_eur"])
    v["_pk"], v["tk"] = v["Player"].map(_norm_name), v["Squad"].map(normalize_team)
    return v.groupby(["season", "tk", "_pk"])["val"].max().to_dict()


def team_absence(lu: pd.DataFrame, values: dict) -> pd.DataFrame:
    """One row per (match, team): the top player's value and whether he started,
    plus the share of the team's top-3 value missing from the XI."""
    lu = lu.assign(day=pd.to_datetime(lu["date"]))
    lu["val"] = [values.get((s, t, p), np.nan) for s, t, p in zip(lu["season"], lu["team_key"], lu["_pk"])]
    rows = []
    for (season, tk), g in lu.groupby(["season", "team_key"]):
        g = g.sort_values("day")
        days = g["day"].unique()
        for d in days:
            recent = g[(g["day"] < d) & (g["day"] >= d - pd.Timedelta(days=WINDOW_DAYS))]
            pool = recent.dropna(subset=["val"]).drop_duplicates("_pk").nlargest(3, "val")
            if pool.empty:
                continue
            xi = set(g.loc[(g["day"] == d) & (g["is_starter"] == 1), "_pk"])
            top = pool.iloc[0]
            rows.append({"season": season, "team_key": tk, "day": d, "top_player": top["player"],
                         "top_value": top["val"], "top_absent": int(top["_pk"] not in xi),
                         "top3_absent_share": float(pool.loc[~pool["_pk"].isin(xi), "val"].sum() / pool["val"].sum())})
    return pd.DataFrame(rows)


def _residuals(pred: pd.DataFrame) -> pd.DataFrame:
    """Each match from both sides: actual points minus model expected points."""
    out = []
    for side, opp, p_w, p_l, w in (("HomeTeam", "AwayTeam", "p_home", "p_away", "H"),
                                   ("AwayTeam", "HomeTeam", "p_away", "p_home", "A")):
        pts = np.where(pred["FTR"] == w, 3, np.where(pred["FTR"] == "D", 1, 0))
        out.append(pd.DataFrame({"season": pred["season"], "day": pd.to_datetime(pred["Date"]),
                                 "team_key": pred[side].map(normalize_team),
                                 "resid": pts - (3 * pred[p_w] + pred["p_draw"]), "pts": pts}))
    return pd.concat(out, ignore_index=True)


def _join(res: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    """Match residual rows to lineup rows by team + date (FotMob UTC vs local, +-1 day)."""
    ab_idx = {(t, d): i for i, t, d in zip(ab.index, ab["team_key"], ab["day"])}
    hit = []
    for t, d in zip(res["team_key"], res["day"]):
        i = next((ab_idx[(t, d + pd.Timedelta(days=o))] for o in (0, -1, 1)
                  if (t, d + pd.Timedelta(days=o)) in ab_idx), None)
        hit.append(i)
    res = res.assign(_i=hit).dropna(subset=["_i"])
    return res.join(ab.drop(columns=["season", "team_key", "day"]), on="_i")


def _diff_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    d = a.mean() - b.mean()
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(BOOT)]
    return d, *np.percentile(boots, [2.5, 97.5])


def main() -> None:
    lu = pd.read_csv(PROC / "match_lineup_players.csv", dtype={"match_id": str})
    lu = lu[lu["season"].isin(["2023-24", "2024-25", "2025-26"])]   # 2023-24 only seeds the window
    ab = team_absence(lu, _values())
    rng = np.random.default_rng(0)
    for name in ("outcome_model_predictions.csv", "outcome_model_predictions_hybrid.csv"):
        pred = pd.read_csv(PROC / name).dropna(subset=["FTR"])
        j = _join(_residuals(pred), ab)
        absent, present = j[j["top_absent"] == 1], j[j["top_absent"] == 0]
        d, lo, hi = _diff_ci(absent["resid"].to_numpy(), present["resid"].to_numpy(), rng)
        print(f"\n{name}  ({len(j):,} team-matches joined to a confirmed XI)")
        print(f"  top player absent : n={len(absent):5,}  mean pts - expected = {absent['resid'].mean():+.3f}")
        print(f"  top player starts : n={len(present):5,}  mean pts - expected = {present['resid'].mean():+.3f}")
        print(f"  difference        : {d:+.3f} pts/match  (95% bootstrap CI {lo:+.3f} .. {hi:+.3f})")
        for lo_s, hi_s, lbl in ((0, .01, "none of top-3 missing"), (.01, .5, "<half of top-3 value missing"),
                                (.5, 1.01, ">=half of top-3 value missing")):
            m = (j["top3_absent_share"] >= lo_s) & (j["top3_absent_share"] < hi_s)
            print(f"  {lbl:30} n={m.sum():5,}  resid {j.loc[m, 'resid'].mean():+.3f}")
        heavy = j.loc[j["top3_absent_share"] >= .5, "resid"].to_numpy()
        full = j.loc[j["top3_absent_share"] < .01, "resid"].to_numpy()
        d3, lo3, hi3 = _diff_ci(heavy, full, rng)
        print(f"  >=half of top-3 missing vs none: {d3:+.3f} pts/match (95% CI {lo3:+.3f} .. {hi3:+.3f})")
        big = j[j["top_value"] >= 50e6]
        if len(big):
            d2, lo2, hi2 = _diff_ci(big.loc[big["top_absent"] == 1, "resid"].to_numpy(),
                                    big.loc[big["top_absent"] == 0, "resid"].to_numpy(), rng)
            print(f"  top player worth EUR50M+: absent-vs-starts {d2:+.3f} (CI {lo2:+.3f} .. {hi2:+.3f}), "
                  f"n absent={int((big['top_absent'] == 1).sum())}")


if __name__ == "__main__":
    main()
