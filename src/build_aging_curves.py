"""Population aging curves + player-specific 1-2 season value forecasts.

Everything needed is already sitting in value_model_predictions.csv (season,
Player, Squad, pos, age, market_value_eur, predicted_eur, value_imputed for
every player-season 2020-21..2026-27) - no new data.

  1. Population curve: log(value) ~ age + age^2, fit separately per position
     group (GK/DF/MF/FW - aging looks very different for a keeper than a
     winger), on real (non-imputed) player-seasons only, restricted to
     players with >=3 real-valued seasons so one-off flash-in-the-pan values
     don't dominate the slope. Season is included as a fixed effect (one
     dummy per season) and then discarded, so the fitted age/age^2
     coefficients aren't confounded by the market's broad value inflation
     over 2020-26 - only the SHAPE of the age effect is kept, centred so the
     adjustment is 0 at the position's own fitted peak age. This replaces
     train_value_model.py's hand-tuned `_age_adj` (flat 22-29, -6%/yr after
     30, +4%/yr under 21, the same for every position) with a curve actually
     fit to the data.

  2. Player forecast, Marcel/delta-method style (as in baseball aging
     projections): a player's own deviation from the population curve
     ("residual") is a decent proxy for his persistent quality/reputation
     level, recency-weighted (half-life 2 seasons, same convention as
     train_outcome_model.py's recency weight) and shrunk toward 0 - i.e.
     toward the bare population curve - the fewer real-valued seasons he has
     (the same Bayesian-shrinkage pattern train_value_model.py already uses
     for nation_premium, just at player instead of nation grain). The
     forecast for age+1/age+2 is the population curve at that future age,
     plus the shrunk residual (decayed slightly further for the 2-year
     horizon - more time for regression to the mean).

Run:  py -3.11 src/build_aging_curves.py
Output: data/processed/aging_curve.csv     pos, age, log_adj (ages 16-42)
        data/processed/value_forecast.csv  player_key, player, pos, season, age,
                                            n_real_seasons, forecast_1y_eur, forecast_2y_eur
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
SRC = PROC / "value_model_predictions.csv"
OUT_CURVE = PROC / "aging_curve.csv"
OUT_FORECAST = PROC / "value_forecast.csv"

MIN_SEASONS_FOR_CURVE = 3   # a player needs this many real-valued seasons to
                            # help fit the population curve (not to GET a forecast)
AGE_RANGE = range(16, 43)
SHRINK_K = 0.5               # player-level Bayesian shrinkage strength. Small:
                             # transfer-market value is far stickier year to
                             # year than the batting-average-style rate stats
                             # Marcel projections were designed for, so even a
                             # handful of real seasons should already inspire
                             # real confidence in a player's own level.
HALF_LIFE_SEASONS = 2.0
FORECAST_2Y_DECAY = 0.85    # extra pull toward the population curve at the
                            # 2-year horizon - more time for reversion to the mean
RATIO_CAP_1Y = (0.2, 3.0)   # forecast bounded to [0.2x, 3x] the player's own
RATIO_CAP_2Y = (0.12, 4.0)  # current value - see the guardrail note below

_SEASON_ORDER = {f"{y}-{str(y + 1)[-2:]}": y - 2013 for y in range(2013, 2028)}


def _fit_pos_curve(g: pd.DataFrame) -> tuple[float, float, float]:
    """OLS log(value) ~ age + age^2 + season-dummies on one position group;
    returns (beta_age, beta_age2, peak_age) - season coefficients are fit
    (to net out cross-season inflation) then discarded, only the age shape
    is kept."""
    X = pd.DataFrame({"season": g["season"], "age": g["age"], "age2": g["age"] ** 2})
    ct = ColumnTransformer([("season", OneHotEncoder(handle_unknown="ignore"), ["season"])],
                           remainder="passthrough")
    Xt = ct.fit_transform(X)
    m = LinearRegression().fit(Xt, g["log_value"])
    b_age, b_age2 = m.coef_[-2], m.coef_[-1]
    peak_age = -b_age / (2 * b_age2) if b_age2 < 0 else 26.0
    peak_age = float(np.clip(peak_age, 20, 34))
    return float(b_age), float(b_age2), peak_age


def _curve_fn(b_age: float, b_age2: float, peak_age: float):
    base = b_age * peak_age + b_age2 * peak_age ** 2

    def f(age):
        age = np.asarray(age, dtype=float)
        return b_age * age + b_age2 * age ** 2 - base
    return f


def build_curves(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    real = df[(df["value_imputed"] == 0) & df["market_value_eur"].notna()].copy()
    real["log_value"] = np.log1p(real["market_value_eur"])
    real["_pk"] = real["Player"].map(_norm_name)
    n_seasons = real.groupby("_pk")["season"].transform("nunique")
    fit_pop = real[n_seasons >= MIN_SEASONS_FOR_CURVE]

    curves, pop_level, rows = {}, {}, []
    for pos, g in fit_pop.groupby("pos"):
        b_age, b_age2, peak_age = _fit_pos_curve(g)
        fn = _curve_fn(b_age, b_age2, peak_age)
        curves[pos] = fn
        # log_adj is a pure SHAPE (0 at peak age) - the population's mean
        # residual (log_value with that shape backed out) is the meaningful
        # "typical player of this position" level to shrink a thin-history
        # player toward. Shrinking toward 0 instead would be shrinking toward
        # a value of ~EUR1, which swamps the real signal - the bug this
        # replaced. Computed over the FULL real population (not fit_pop,
        # the >=3-season set used for the curve shape above) - fit_pop
        # survivorship-biases toward established players, which would pull a
        # cheap fringe player's forecast up toward an "established squad
        # player" level instead of the level of players actually like him.
        allg = real[real["pos"] == pos]
        pop_level[pos] = float((allg["log_value"] - fn(allg["age"])).mean())
        print(f"  {pos:3} n={len(g):5d}  peak age {peak_age:.1f}  "
              f"pop level EUR{np.expm1(pop_level[pos]) / 1e6:.1f}M  "
              f"(fit on {g['_pk'].nunique()} players with >= {MIN_SEASONS_FOR_CURVE} real seasons)")
        for age in AGE_RANGE:
            rows.append({"pos": pos, "age": age, "log_adj": round(float(fn(age)), 4)})
    return curves, pop_level, pd.DataFrame(rows)


def build_forecasts(df: pd.DataFrame, curves: dict, pop_level: dict) -> pd.DataFrame:
    real = df[(df["value_imputed"] == 0) & df["market_value_eur"].notna()].copy()
    real["log_value"] = np.log1p(real["market_value_eur"])
    real["_pk"] = real["Player"].map(_norm_name)
    real["_ord"] = real["season"].map(_SEASON_ORDER)
    real = real.dropna(subset=["_ord", "age"])

    def _adj(pos, age):
        fn = curves.get(pos) or curves.get("MF")  # MF as a generic fallback
        return float(fn(np.clip(age, 16, 42)))

    def _pop(pos):
        return pop_level.get(pos, pop_level.get("MF"))

    # residual = log_value with the age SHAPE backed out - a level quantity
    # (not zero-centred), so shrinkage below pulls toward the population's
    # own mean residual (a typical player's level), not toward literal 0.
    real["log_adj"] = [_adj(p, a) for p, a in zip(real["pos"], real["age"])]
    real["residual"] = real["log_value"] - real["log_adj"]

    rows = []
    for pk, g in real.sort_values("_ord").groupby("_pk"):
        last = g.iloc[-1]
        pop = _pop(last["pos"])
        w = 0.5 ** ((last["_ord"] - g["_ord"]) / HALF_LIFE_SEASONS)
        player_level = float(np.average(g["residual"], weights=w))
        n = len(g)
        # Bayesian shrinkage toward the position's population level, same
        # pattern as train_value_model.py's nation_premium (there: toward the
        # global mean, strength 12; here: toward the position mean, strength
        # SHRINK_K - players have far fewer observations than nations do).
        shrunk = (n * player_level + SHRINK_K * pop) / (n + SHRINK_K)
        # extra pull toward the population level at the 2-year horizon - more
        # time for regression to the mean / uncertainty about whether current
        # form holds
        shrunk_2y = shrunk * FORECAST_2Y_DECAY + pop * (1 - FORECAST_2Y_DECAY)
        age1, age2 = last["age"] + 1, last["age"] + 2
        f1 = _adj(last["pos"], age1) + shrunk
        f2 = _adj(last["pos"], age2) + shrunk_2y
        # guardrail: log-space shrinkage toward a position-wide mean spans a
        # huge multiplicative range on a market this heavy-tailed (a cheap
        # fringe player's tiny value sits log-units away from the position
        # mean, so even a small shrinkage WEIGHT is a large swing) - cap the
        # forecast as a bounded multiple of the player's own current value,
        # same spirit as the value model's own CAP_EUR clip.
        cur_val = float(np.expm1(last["log_value"]))
        v1 = np.clip(float(np.expm1(f1)), cur_val * RATIO_CAP_1Y[0], cur_val * RATIO_CAP_1Y[1])
        v2 = np.clip(float(np.expm1(f2)), cur_val * RATIO_CAP_2Y[0], cur_val * RATIO_CAP_2Y[1])
        rows.append({
            "player_key": pk, "player": last["Player"], "squad": last["Squad"],
            "src_league": last["src_league"], "pos": last["pos"],
            "season": last["season"], "age": last["age"], "n_real_seasons": n,
            "forecast_1y_eur": round(v1, 0),
            "forecast_2y_eur": round(v2, 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(SRC)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])

    print("population aging curves (log-value adjustment, season-controlled):")
    curves, pop_level, curve_df = build_curves(df)
    curve_df.to_csv(OUT_CURVE, index=False)
    print(f"wrote {OUT_CURVE}  ({len(curve_df)} rows)")

    forecast = build_forecasts(df, curves, pop_level)
    forecast.to_csv(OUT_FORECAST, index=False)
    print(f"wrote {OUT_FORECAST}  ({len(forecast)} players)")
    print("\nsample forecasts:")
    print(forecast.sort_values("forecast_1y_eur", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
