"""Fit the confirmed-XI adjustment the matchday re-score applies, and test it
out of sample.

The site's Dixon-Coles odds never see a lineup. backtest_lineup_absence.py
showed a team missing a big share of its top-3 players' value underperforms;
this turns that into an adjustment of the expected goals:

    xg_home' = xg_home * exp(OWN * share_home_missing + OPP * share_away_missing)
    xg_away' = xg_away * exp(OWN * share_away_missing + OPP * share_home_missing)

where share_*_missing = the share of that side's top-3 (by value) not in the
confirmed XI. Pre-match xG come from a walk-forward Dixon-Coles (refit every 28
days on matches strictly before - leak-free). OWN / OPP are fitted by Poisson
likelihood on FIT_SEASONS and evaluated, untouched, on EVAL_SEASONS: 1X2
log-loss with vs without the adjustment. The coefficients and the evaluation go
to data/processed/lineup_adjustment.json for src/matchday_rescore.py.

Run:  py -3.11 src/fit_lineup_adjustment.py           (reuses dc_walk_lambdas.csv)
      py -3.11 src/fit_lineup_adjustment.py --refit   (~12 min: redo the walk-forward fits)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dixon_coles as dc  # noqa: E402
from backtest_lineup_absence import _values, team_absence  # noqa: E402
from export_site_data import LEAGUES, prep_dc_matches  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "lineup_adjustment.json"
# walk-forward expected goals + missing shares per match (slow to build; --refit rebuilds)
CACHE = PROC / "dc_walk_lambdas.csv"
FIT_SEASONS = ["2021-22", "2022-23", "2023-24"]
EVAL_SEASONS = ["2024-25", "2025-26"]
MAXG = 10


def _lambdas(model, h, a):
    if model is None or h not in model["idx"] or a not in model["idx"]:
        return np.nan, np.nan
    i, j = model["idx"][h], model["idx"][a]
    return (float(np.exp(model["att"][i] + model["dfn"][j] + model["home"])),
            float(np.exp(model["att"][j] + model["dfn"][i])))


def walk_lambdas(matches: pd.DataFrame, targets: pd.DataFrame, refit_days: int = 28) -> pd.DataFrame:
    out, model, last = [], None, None
    for r in targets.sort_values("Date").itertuples():
        if last is None or (r.Date - last).days >= refit_days:
            model = dc.fit(matches, r.Date) or model
            last = r.Date
        out.append((r.Index, *_lambdas(model, r.h, r.a)))
    lam = pd.DataFrame(out, columns=["i", "lh", "la"]).set_index("i")
    return targets.join(lam)


def _shares(t: pd.DataFrame) -> pd.DataFrame:
    lu = pd.read_csv(PROC / "match_lineup_players.csv", dtype={"match_id": str})
    lu = lu[lu["season"].isin(["2020-21"] + FIT_SEASONS + EVAL_SEASONS)]
    ab = team_absence(lu, _values())
    idx = {(tk, d): s for tk, d, s in zip(ab["team_key"], ab["day"], ab["top3_absent_share"])}

    def look(tk, d):
        return next((idx[(tk, d + pd.Timedelta(days=o))] for o in (0, -1, 1)
                     if (tk, d + pd.Timedelta(days=o)) in idx), np.nan)
    return t.assign(sh=[look(h, d) for h, d in zip(t["h"], t["Date"])],
                    sa=[look(a, d) for a, d in zip(t["a"], t["Date"])])


def adjust(lh, la, sh, sa, own, opp):
    return lh * np.exp(own * sh + opp * sa), la * np.exp(own * sa + opp * sh)


def _nll(params, d):
    lh, la = adjust(d["lh"], d["la"], d["sh"], d["sa"], *params)
    return -(poisson.logpmf(d["FTHG"], lh) + poisson.logpmf(d["FTAG"], la)).sum()


def probs_1x2(lh, la) -> np.ndarray:
    """[home, draw, away] from independent Poissons (same form with and without
    the adjustment, so the comparison is like for like)."""
    g = np.arange(MAXG)
    ph = poisson.pmf(g[None, :], np.asarray(lh)[:, None])
    pa = poisson.pmf(g[None, :], np.asarray(la)[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    i, j = np.meshgrid(g, g, indexing="ij")
    p = np.stack([(grid * (i > j)).sum((1, 2)), (grid * (i == j)).sum((1, 2)), (grid * (i < j)).sum((1, 2))], 1)
    return p / p.sum(1, keepdims=True)


def _ll_each(p, ftr) -> np.ndarray:
    y = ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def _ll(p, ftr):
    return float(_ll_each(p, ftr).mean())


def main() -> None:
    if CACHE.exists() and "--refit" not in sys.argv:
        t = pd.read_csv(CACHE, parse_dates=["Date"])
    else:
        matches = prep_dc_matches()
        t = matches[(matches["comp"].isin(LEAGUES.values()))
                    & matches["season"].isin(FIT_SEASONS + EVAL_SEASONS)].copy()
        t = t.dropna(subset=["FTHG", "FTAG", "FTR"])
        print(f"walk-forward Dixon-Coles over {len(t):,} league matches ...", flush=True)
        t = _shares(walk_lambdas(matches, t)).dropna(subset=["lh", "la", "sh", "sa"])
        t[["season", "comp", "Date", "HomeTeam", "AwayTeam", "h", "a", "FTHG", "FTAG", "FTR",
           "lh", "la", "sh", "sa"]].to_csv(CACHE, index=False)
    fit, ev = t[t["season"].isin(FIT_SEASONS)], t[t["season"].isin(EVAL_SEASONS)]
    res = minimize(_nll, x0=[0.0, 0.0], args=(fit,), method="Nelder-Mead")
    own, opp = (float(x) for x in res.x)
    base = probs_1x2(ev["lh"], ev["la"])
    adj = probs_1x2(*adjust(ev["lh"], ev["la"], ev["sh"], ev["sa"], own, opp))
    # paired bootstrap over matches: is the log-loss gain distinguishable from noise?
    gain = _ll_each(base, ev["FTR"]) - _ll_each(adj, ev["FTR"])
    rng = np.random.default_rng(0)
    boots = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {"own": round(own, 4), "opp": round(opp, 4), "fit_seasons": FIT_SEASONS, "n_fit": int(len(fit)),
           "eval": {"seasons": EVAL_SEASONS, "n": int(len(ev)),
                    "log_loss_without": round(_ll(base, ev["FTR"]), 4),
                    "log_loss_with_xi": round(_ll(adj, ev["FTR"]), 4),
                    "gain": round(float(gain.mean()), 4),
                    "gain_ci95": [round(float(lo), 4), round(float(hi), 4)]},
           "note": "xg' = xg * exp(own * own_top3_missing_share + opp * opponent_top3_missing_share)"}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
