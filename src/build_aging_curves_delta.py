"""Delta-method aging curves - a check on build_aging_curves.py's cross-sectional
curve, which compares DIFFERENT players' value levels at each age and so is
biased by who survives: the only 34-year-olds still in a big-5 league are the
ones good enough to stay, which makes old age look kinder than it is.

The delta method pairs each player with HIMSELF a season later: for every
player with real (non-imputed) values in consecutive seasons, the change in
log value; that change is de-meaned per season pair (to net out market-wide
inflation), then modelled against age per position group:

    d log(value) / d age  =  a + b * age          (OLS on the paired changes)

which integrates to a quadratic curve with its peak where the change crosses
zero (age = -a / b), centred to 0 at the peak like aging_curve.csv. It still
only sees players who stay in the data for two seasons (a milder selection
than the cross-sectional one, which conditions on the level itself).

Evaluated out of sample: both methods fitted on pairs/seasons up to 2023-24,
then asked to predict each player's 2024-25 -> 2025-26 (inflation-adjusted)
log-value change; mean squared error, per position and overall.

Run:  py -3.11 src/build_aging_curves_delta.py
Output: data/processed/aging_curve_delta.csv   pos, age, log_adj_delta, log_adj_cross
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_aging_curves import (AGE_RANGE, MIN_SEASONS_FOR_CURVE, _SEASON_ORDER, _curve_fn,  # noqa: E402
                                _fit_pos_curve)
from parse_fbref_player_stats import _norm_name  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "aging_curve_delta.csv"
TRAIN_LAST = "2023-24"              # pairs starting up to this season fit the curves
TEST_PAIR = ("2024-25", "2025-26")  # the out-of-sample season pair


def _real(df: pd.DataFrame) -> pd.DataFrame:
    real = df[(df["value_imputed"] == 0) & df["market_value_eur"].notna()].copy()
    real["age"] = pd.to_numeric(real["age"], errors="coerce")
    real = real.dropna(subset=["age"])
    real["log_value"] = np.log1p(real["market_value_eur"])
    real["_pk"] = real["Player"].map(_norm_name)
    real["_ord"] = real["season"].map(_SEASON_ORDER)
    # one row per player-season (a mid-season mover has two squads - keep the higher value)
    return real.sort_values("log_value").drop_duplicates(["_pk", "season"], keep="last")


def pairs(real: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-season pairs of the same player, inflation-adjusted change."""
    nxt = real[["_pk", "_ord", "log_value", "season"]].rename(
        columns={"log_value": "lv_next", "season": "season_next"})
    nxt["_ord"] -= 1
    p = real.merge(nxt, on=["_pk", "_ord"])
    p["d"] = p["lv_next"] - p["log_value"]
    p["d_adj"] = p["d"] - p.groupby("season")["d"].transform("mean")   # net out market inflation
    p["age_mid"] = p["age"] + 0.5
    return p


def fit_delta(p: pd.DataFrame) -> tuple[float, float]:
    """OLS d_adj = a + b * age_mid; returns (a, b)."""
    X = np.c_[np.ones(len(p)), p["age_mid"].to_numpy()]
    a, b = np.linalg.lstsq(X, p["d_adj"].to_numpy(), rcond=None)[0]
    return float(a), float(b)


def delta_curve(a: float, b: float):
    """Integral of a + b*age, centred to 0 at its peak (age = -a/b)."""
    peak = float(np.clip(-a / b, 18, 36)) if b < 0 else 27.0

    def f(age):
        age = np.asarray(age, dtype=float)
        return a * (age - peak) + b / 2 * (age ** 2 - peak ** 2)
    return f, peak


def main() -> None:
    real = _real(pd.read_csv(PROC / "value_model_predictions.csv"))
    p = pairs(real)
    rows, report = [], []
    test = p[(p["season"] == TEST_PAIR[0]) & (p["season_next"] == TEST_PAIR[1])]
    err = {"delta": [], "cross": [], "flat": []}
    for pos in sorted(p["pos"].dropna().unique()):
        # full-data curves, for the output table
        a, b = fit_delta(p[p["pos"] == pos])
        fd, peak_d = delta_curve(a, b)
        rp = real[real["pos"] == pos]
        keep = rp.groupby("_pk")["season"].transform("nunique") >= MIN_SEASONS_FOR_CURVE
        ba, ba2, peak_c = _fit_pos_curve(rp[keep])
        fc = _curve_fn(ba, ba2, peak_c)
        rows += [{"pos": pos, "age": age, "log_adj_delta": round(float(fd(age)), 4),
                  "log_adj_cross": round(float(fc(age)), 4)} for age in AGE_RANGE]

        # out-of-sample: fit on training seasons only, predict the test pair
        tr = p[(p["pos"] == pos) & (p["season"] <= TRAIN_LAST)]
        a_t, b_t = fit_delta(tr)
        fd_t, _ = delta_curve(a_t, b_t)
        rt = rp[rp["season"] <= TRAIN_LAST]
        keep_t = rt.groupby("_pk")["season"].transform("nunique") >= MIN_SEASONS_FOR_CURVE
        fc_t = _curve_fn(*_fit_pos_curve(rt[keep_t]))
        te = test[test["pos"] == pos]
        if len(te):
            age = te["age"].to_numpy()
            err["delta"].append((te["d_adj"] - (fd_t(age + 1) - fd_t(age))) ** 2)
            err["cross"].append((te["d_adj"] - (fc_t(age + 1) - fc_t(age))) ** 2)
            err["flat"].append(te["d_adj"] ** 2)          # "no age effect" baseline
        report.append({"pos": pos, "pairs": len(p[p["pos"] == pos]), "peak_delta": round(peak_d, 1),
                       "peak_cross": round(peak_c, 1),
                       "peak_to_33_delta": f"{np.expm1(fd(33)):+.0%}", "peak_to_33_cross": f"{np.expm1(fc(33)):+.0%}",
                       "test_n": len(te),
                       "mse_delta": round(float(err["delta"][-1].mean()), 4) if len(te) else None,
                       "mse_cross": round(float(err["cross"][-1].mean()), 4) if len(te) else None})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(report).to_string(index=False))
    tot = {k: float(pd.concat(v).mean()) for k, v in err.items()}
    print(f"\nout of sample ({TEST_PAIR[0]} -> {TEST_PAIR[1]}, n={len(test):,}): MSE of the "
          f"inflation-adjusted log-value change - delta {tot['delta']:.4f}, cross-sectional "
          f"{tot['cross']:.4f}, no-age baseline {tot['flat']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
