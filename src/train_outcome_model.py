"""Step 8 - outcome classifier: pre-match home-win / draw / away-win.

Features from `src/build_match_model_table.py` (run it first): goals-Elo and
xG-Elo (both updated across all competitions), home/away-split rolling form,
squad-value ratio, promoted flags, rest days, head-to-head.

Temporal split: train <= 2022-23, validate 2023-24 (calibration), test
2024-25 + 2025-26. Benchmarked against Bet365 closing odds and a base-rate model.

Run:  py -3.11 src/build_match_model_table.py
      py -3.11 src/train_outcome_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
LABELS = ["A", "D", "H"]

FEATURES = [
    "elo_diff", "xelo_diff", "elo_exp_h", "xelo_exp_h",
    "fh_pts", "fh_gf", "fh_ga", "fh_xgf", "fh_xga",
    "fa_pts", "fa_gf", "fa_ga", "fa_xgf", "fa_xga",
    "all_h_pts", "all_a_pts", "all_h_xgd", "all_a_xgd",
    "h_att", "a_att", "form_pts_gap",
    "value_log_ratio", "age_gap", "days_rest_h", "days_rest_a",
    "promoted_h", "promoted_a", "h2h_h_pts",
]


def _ll(y, p) -> float:
    return log_loss(y, p, labels=LABELS)


def _order(clf, p):
    idx = [list(clf.classes_).index(c) for c in LABELS]
    return p[:, idx]


_H_FEATS = ["elo_diff", "xelo_diff", "elo_exp_h", "xelo_exp_h",
            "fh_xgf", "fa_xga", "fh_gf", "fa_ga", "fh_pts", "fa_pts",
            "all_h_xgd", "all_a_xgd", "all_h_pts", "value_log_ratio",
            "h_att", "a_att", "promoted_h", "promoted_a", "days_rest_h", "h2h_h_pts"]
_A_FEATS = ["elo_diff", "xelo_diff", "elo_exp_h", "xelo_exp_h",
            "fa_xgf", "fh_xga", "fa_gf", "fh_ga", "fa_pts", "fh_pts",
            "all_a_xgd", "all_h_xgd", "all_a_pts", "value_log_ratio",
            "a_att", "h_att", "promoted_a", "promoted_h", "days_rest_a", "h2h_h_pts"]
MAXG = 8


def poisson_probs(tr, te, ytr_h, ytr_a, rho=-0.11, alpha=1e-2, w=None):
    """Fit home-goals and away-goals Poisson GLMs, return P(A/D/H) for te via the
    scoreline distribution with a Dixon-Coles low-score tweak."""
    def fit(cols, y):
        p = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("m", PoissonRegressor(alpha=alpha, max_iter=800))])
        p.fit(tr[cols], y, m__sample_weight=w)
        return p
    mh, ma = fit(_H_FEATS, ytr_h), fit(_A_FEATS, ytr_a)
    lh = np.clip(mh.predict(te[_H_FEATS]), 0.15, 6)
    la = np.clip(ma.predict(te[_A_FEATS]), 0.15, 6)
    ph = poisson.pmf(np.arange(MAXG)[:, None], lh)   # (MAXG, n)
    pa = poisson.pmf(np.arange(MAXG)[:, None], la)
    grid = ph[:, None, :] * pa[None, :, :]           # (i, j, n)
    grid[0, 0] *= 1 - lh * la * rho
    grid[0, 1] *= 1 + lh * rho
    grid[1, 0] *= 1 + la * rho
    grid[1, 1] *= 1 - rho
    i = np.arange(MAXG)[:, None, None]
    j = np.arange(MAXG)[None, :, None]
    P = np.stack([
        np.where(i < j, grid, 0).sum((0, 1)),   # away
        np.where(i == j, grid, 0).sum((0, 1)),  # draw
        np.where(i > j, grid, 0).sum((0, 1)),   # home
    ], axis=1)
    return np.clip(P, 1e-6, None) / np.clip(P, 1e-6, None).sum(1, keepdims=True)


def _book(df: pd.DataFrame) -> pd.DataFrame | None:
    mf = pd.read_csv(PROC / "match_features.csv")
    cols = ["B365CH", "B365CD", "B365CA"] if "B365CH" in mf else ["B365H", "B365D", "B365A"]
    mf = mf.dropna(subset=cols)
    mf["k"] = (mf["season"] + "|" + pd.to_datetime(mf["Date"]).dt.strftime("%Y-%m-%d")
               + "|" + mf["HomeTeam"].map(normalize_team) + "|" + mf["AwayTeam"].map(normalize_team))
    inv = 1 / mf[cols].to_numpy()
    inv = inv / inv.sum(axis=1, keepdims=True)
    mf[["pH", "pD", "pA"]] = inv
    return mf.set_index("k")[["pA", "pD", "pH"]]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true",
                    help="add market-consensus opening odds as features")
    args = ap.parse_args()

    df = pd.read_csv(PROC / "match_model_table.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["promoted_h"] = df["promoted_h"].fillna(0)
    df["promoted_a"] = df["promoted_a"].fillna(0)
    df = df.dropna(subset=["elo_diff", "fh_pts", "fa_pts"])

    feats = list(FEATURES)
    if args.hybrid:
        feats += ["mkt_pH", "mkt_pD", "mkt_pA"]
        print("[hybrid: + market opening odds]")
    globals()["FEATURES"] = feats

    tr = df[df["season"] <= "2022-23"]
    va = df[df["season"] == "2023-24"]
    te = df[df["season"].isin(["2024-25", "2025-26"])]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    Xtr, ytr = tr[FEATURES], tr["FTR"]
    Xva, yva = va[FEATURES], va["FTR"]
    Xte, yte = te[FEATURES], te["FTR"]

    base = ytr.value_counts(normalize=True).reindex(LABELS).to_numpy()
    print(f"\n{'model':14}{'acc':>7}{'logloss':>10}")
    print(f"{'base-rate':14}{(yte == 'H').mean():7.3f}{_ll(yte, np.tile(base, (len(yte), 1))):10.3f}")

    logreg = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler()),
                       ("m", LogisticRegression(max_iter=4000, C=0.3))])
    hgb = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("m", HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.03, max_iter=600,
                        l2_regularization=2.0, min_samples_leaf=40))])

    preds, preds_va = {}, {}
    for name, clf in {"logreg": logreg, "hgb": hgb}.items():
        clf.fit(Xtr, ytr)
        preds[name] = _order(clf, clf.predict_proba(Xte))
        preds_va[name] = _order(clf, clf.predict_proba(Xva))
        print(f"{name:14}{accuracy_score(yte, LABELS_pred(preds[name])):7.3f}{_ll(yte, preds[name]):10.3f}")

    # recency weight: half-life ~2 seasons
    age_days = (tr["Date"].max() - tr["Date"]).dt.days.to_numpy()
    wt = 0.5 ** (age_days / 730)

    # tune rho + alpha on the validation season
    best = (None, 9.9)
    for rho in (-0.16, -0.13, -0.11, -0.08, -0.05):
        for alpha in (1e-3, 1e-2, 5e-2):
            pv = poisson_probs(tr, va, tr["FTHG"].astype(float), tr["FTAG"].astype(float),
                               rho=rho, alpha=alpha, w=wt)
            ll = _ll(yva, pv)
            if ll < best[1]:
                best = ((rho, alpha), ll)
    rho, alpha = best[0]
    print(f"poisson tuned: rho={rho} alpha={alpha}  (val logloss {best[1]:.3f})")

    preds["poisson"] = poisson_probs(tr, te, tr["FTHG"].astype(float), tr["FTAG"].astype(float),
                                     rho=rho, alpha=alpha, w=wt)
    preds_va["poisson"] = poisson_probs(tr, va, tr["FTHG"].astype(float), tr["FTAG"].astype(float),
                                        rho=rho, alpha=alpha, w=wt)
    print(f"{'poisson':14}{accuracy_score(yte, LABELS_pred(preds['poisson'])):7.3f}"
          f"{_ll(yte, preds['poisson']):10.3f}")

    # a second Poisson fitted to xG (a smoother scoring-rate estimate), blended in
    if "HxG" in tr.columns:
        trx = tr.dropna(subset=["HxG", "AxG"])
        wtx = 0.5 ** ((trx["Date"].max() - trx["Date"]).dt.days / 730).to_numpy()
        yhx, yax = trx["HxG"].astype(float), trx["AxG"].astype(float)
        preds["poisson"] = 0.6 * preds["poisson"] + 0.4 * poisson_probs(
            trx, te, yhx, yax, rho=rho, alpha=alpha, w=wtx)
        preds_va["poisson"] = 0.6 * preds_va["poisson"] + 0.4 * poisson_probs(
            trx, va, yhx, yax, rho=rho, alpha=alpha, w=wtx)
        print(f"{'poisson+xg':14}{accuracy_score(yte, LABELS_pred(preds['poisson'])):7.3f}"
              f"{_ll(yte, preds['poisson']):10.3f}")

    # geometric blend of poisson + logreg, weight tuned on val (keeps if it helps)
    def _gblend(ps, w):
        z = np.exp(sum(wi * np.log(np.clip(p, 1e-6, 1)) for wi, p in zip(w, ps)))
        return z / z.sum(1, keepdims=True)
    from scipy.optimize import minimize_scalar
    a = minimize_scalar(lambda a: _ll(yva, _gblend(
        [preds_va["poisson"], preds_va["logreg"]], [a, 1 - a])),
        bounds=(0.3, 1.0), method="bounded").x
    pf_va = _gblend([preds_va["poisson"], preds_va["logreg"]], [a, 1 - a])
    pf = _gblend([preds["poisson"], preds["logreg"]], [a, 1 - a])
    print(f"{'blend(a=%.2f)' % a:14}{accuracy_score(yte, LABELS_pred(pf)):7.3f}{_ll(yte, pf):10.3f}")

    if args.hybrid and {"mkt_pA", "mkt_pD", "mkt_pH"}.issubset(df.columns):
        mk_va = np.where(np.isnan(va[["mkt_pA", "mkt_pD", "mkt_pH"]].to_numpy()),
                         pf_va, va[["mkt_pA", "mkt_pD", "mkt_pH"]].to_numpy())
        mk_te = np.where(np.isnan(te[["mkt_pA", "mkt_pD", "mkt_pH"]].to_numpy()),
                         pf, te[["mkt_pA", "mkt_pD", "mkt_pH"]].to_numpy())
        b = minimize_scalar(lambda b: _ll(yva, _gblend([mk_va, pf_va], [b, 1 - b])),
                            bounds=(0.3, 1.0), method="bounded").x
        pf = _gblend([mk_te, pf], [b, 1 - b])
        print(f"{'hybrid(b=%.2f)' % b:14}{accuracy_score(yte, LABELS_pred(pf)):7.3f}{_ll(yte, pf):10.3f}")

    bp = _book(df)
    if bp is not None:
        t2 = te.copy()
        t2["k"] = (t2["season"] + "|" + t2["Date"].dt.strftime("%Y-%m-%d") + "|"
                   + t2["HomeTeam"].map(normalize_team) + "|" + t2["AwayTeam"].map(normalize_team))
        j = t2.join(bp, on="k").dropna(subset=["pA", "pD", "pH"])
        pb = j[["pA", "pD", "pH"]].to_numpy()
        print(f"{'bookmaker':14}{accuracy_score(j['FTR'], LABELS_pred(pb)):7.3f}"
              f"{_ll(j['FTR'], pb):10.3f}   ({len(j)}/{len(te)})")

    # pure output = the best single model (poisson+xg); hybrid = the market blend
    final = pf if args.hybrid else preds["poisson"]
    out = te[["season", "comp", "Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out[["p_away", "p_draw", "p_home"]] = np.round(final, 4)
    name = "outcome_model_predictions_hybrid.csv" if args.hybrid else "outcome_model_predictions.csv"
    out.to_csv(PROC / name, index=False)
    print(f"\nwrote {PROC / name}")


def LABELS_pred(p):
    return np.array(LABELS)[p.argmax(1)]


if __name__ == "__main__":
    main()
