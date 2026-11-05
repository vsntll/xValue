"""Dixon-Coles time-weighted team-strength model.

The classic football model: each team has an attack and a defence strength;
expected goals are exp(att_home + def_away + home_adv) and exp(att_away +
def_home). A rho term corrects the correlation in low-scoring games. Parameters
are fitted by weighted maximum likelihood over recent matches
(weight = exp(-xi * days_ago)).

Used two ways downstream:
  * `fit` / `match_probs` - the site's published match odds (export_site_data.py)
    and the forward-test ledger.
  * `fit_predict_walk` - refit every 28 days, predict each match's W/D/L from
    only prior matches (leak-free) -> dixon_coles_probs.csv (main() below), the
    streak game's per-match reveal. (An xG-fitted variant was dropped: on the
    2024-26 test set it scored 1.079 log-loss, worse than the base rate - xG
    coverage is too patchy across competitions to fit team strengths on.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
MAXG = 10
XI = 0.0018          # time-decay per day (~half-life 385 days)


def _tau(i, j, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((i == 0) & (j == 0), 1 - lh * la * rho, t)
    t = np.where((i == 0) & (j == 1), 1 + lh * rho, t)
    t = np.where((i == 1) & (j == 0), 1 + la * rho, t)
    t = np.where((i == 1) & (j == 1), 1 - rho, t)
    return t


def _nll(params, teams, hi, ai, gh, ga, w):
    n = len(teams)
    att = params[:n]
    dfn = np.r_[params[n:2 * n - 1], -params[n:2 * n - 1].sum()]  # sum-to-zero
    home, rho = params[-2], params[-1]
    lh = np.exp(att[hi] + dfn[ai] + home)
    la = np.exp(att[ai] + dfn[hi])
    ll = (w * (poisson.logpmf(gh, lh) + poisson.logpmf(ga, la)
               + np.log(np.clip(_tau(gh, ga, lh, la, rho), 1e-6, None))))
    pen = 1e-3 * (att @ att)
    return -ll.sum() + pen


def fit(matches: pd.DataFrame, asof: pd.Timestamp, gcols=("FTHG", "FTAG"),
        window_days=900):
    m = matches[(matches["Date"] < asof)
                & (matches["Date"] >= asof - pd.Timedelta(days=window_days))].copy()
    if len(m) < 100:
        return None
    w = np.exp(-XI * (asof - m["Date"]).dt.days.to_numpy())
    teams = sorted(set(m["h"]) | set(m["a"]))
    idx = {t: k for k, t in enumerate(teams)}
    hi = m["h"].map(idx).to_numpy()
    ai = m["a"].map(idx).to_numpy()
    gh = pd.to_numeric(m[gcols[0]], errors="coerce").to_numpy()
    ga = pd.to_numeric(m[gcols[1]], errors="coerce").to_numpy()
    ok = ~(np.isnan(gh) | np.isnan(ga))
    hi, ai, gh, ga, w = hi[ok], ai[ok], gh[ok], ga[ok], w[ok]
    n = len(teams)
    x0 = np.r_[np.zeros(n), np.zeros(n - 1), 0.25, -0.05]
    res = minimize(_nll, x0, args=(teams, hi, ai, gh, ga, w),
                   method="L-BFGS-B", options={"maxiter": 200})
    att = res.x[:n]
    dfn = np.r_[res.x[n:2 * n - 1], -res.x[n:2 * n - 1].sum()]
    return {"idx": idx, "att": att, "dfn": dfn, "home": res.x[-2], "rho": res.x[-1]}


def match_probs(model, home: str, away: str) -> np.ndarray:
    """P(A, D, H)."""
    if model is None or home not in model["idx"] or away not in model["idx"]:
        return np.array([0.29, 0.26, 0.45])
    h, a = model["idx"][home], model["idx"][away]
    lh = np.exp(model["att"][h] + model["dfn"][a] + model["home"])
    la = np.exp(model["att"][a] + model["dfn"][h])
    i = np.arange(MAXG)[:, None]
    j = np.arange(MAXG)[None, :]
    grid = poisson.pmf(i, lh) * poisson.pmf(j, la)
    grid = grid * _tau(i, j, lh, la, model["rho"])
    grid = np.clip(grid, 0, None)
    P = np.array([grid[i < j].sum(), grid[i == j].sum(), grid[i > j].sum()])
    return P / P.sum()


def fit_predict_walk(matches: pd.DataFrame, targets: pd.DataFrame,
                     gcols=("FTHG", "FTAG"), refit_days=28) -> np.ndarray:
    """For each row of `targets`, W/D/L probs from a model fitted on matches
    strictly before it, refitted every `refit_days`."""
    out = np.zeros((len(targets), 3))
    model, last_fit = None, None
    for k, (_, r) in enumerate(targets.sort_values("Date").iterrows()):
        d = r["Date"]
        if last_fit is None or (d - last_fit).days >= refit_days:
            model = fit(matches, d, gcols) or model
            last_fit = d
        out[targets.index.get_loc(r.name)] = match_probs(model, r["h"], r["a"])
    return out


def _prep(path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date", "FTHG", "FTAG"])
    m["h"] = m["HomeTeam"].map(normalize_team)
    m["a"] = m["AwayTeam"].map(normalize_team)
    return m.sort_values("Date")


OUT = PROC / "dixon_coles_probs.csv"
KEY = ["season", "Date", "HomeTeam", "AwayTeam"]


def main() -> None:
    """Leak-free walk-forward dc_pA/pD/pH for every big-5 league match since
    2020-08 -> dixon_coles_probs.csv (the streak game's reveal). Incremental: a
    finished match's walk-forward probability never changes, so rows already in
    the file are kept and only new matches are fitted (seconds on a refresh);
    --full recomputes everything (~20 min)."""
    allm = _prep(PROC / "matches_all.csv")
    warm = ROOT / "data" / "processed" / "elo_warmup.csv"
    if warm.exists():
        w = pd.read_csv(warm)
        w["Date"] = pd.to_datetime(w["Date"], dayfirst=True, format="mixed", errors="coerce")
        w["h"] = w["HomeTeam"].map(normalize_team)
        w["a"] = w["AwayTeam"].map(normalize_team)
        allm = pd.concat([w.dropna(subset=["Date"]), allm], ignore_index=True).sort_values("Date")

    lg = allm[allm.get("competition_type", "league").fillna("league") == "league"].copy()
    lg = lg[lg["season"].notna()] if "season" in lg else lg
    tgt = lg[lg["Date"] >= "2020-08-01"].reset_index(drop=True)

    prior = pd.DataFrame()
    if OUT.exists() and "--full" not in sys.argv:
        prior = pd.read_csv(OUT, parse_dates=["Date"])
        prior = prior[[c for c in prior.columns if not c.startswith("dcx_")]]   # dropped xG variant
        have = set(map(tuple, prior[KEY].astype(str).to_numpy()))
        tgt = tgt[[k not in have for k in map(tuple, tgt[KEY].astype(str).to_numpy())]].reset_index(drop=True)
    print(f"walk-forward Dixon-Coles for {len(tgt):,} new league matches ({len(prior):,} kept)", flush=True)
    if len(tgt):
        tgt[["dc_pA", "dc_pD", "dc_pH"]] = fit_predict_walk(allm, tgt).round(4)
    out = pd.concat([prior, tgt[KEY + ["FTR", "dc_pA", "dc_pD", "dc_pH"]]], ignore_index=True)
    out = out.drop_duplicates(KEY, keep="last").sort_values(["Date", "HomeTeam"])
    out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"wrote {OUT}  ({len(out):,} matches, seasons {sorted(out['season'].unique())})")

    from sklearn.metrics import log_loss
    te = out[out["season"].isin(["2024-25", "2025-26"])].dropna(subset=["FTR"])
    if len(te):
        print(f"  dc test log-loss {log_loss(te['FTR'], te[['dc_pA', 'dc_pD', 'dc_pH']].to_numpy(), labels=['A', 'D', 'H']):.3f}"
              f"  (n={len(te):,})")


if __name__ == "__main__":
    main()
