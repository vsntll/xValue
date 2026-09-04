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


_H_FEATS = ["elo_diff", "xelo_diff", "elo_exp_h", "fh_xgf", "fa_xga", "fh_gf",
            "fa_ga", "all_h_xgd", "value_log_ratio", "h_att", "promoted_a"]
_A_FEATS = ["elo_diff", "xelo_diff", "elo_exp_h", "fa_xgf", "fh_xga", "fa_gf",
            "fh_ga", "all_a_xgd", "value_log_ratio", "a_att", "promoted_h"]
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
    df = pd.read_csv(PROC / "match_model_table.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["promoted_h"] = df["promoted_h"].fillna(0)
    df["promoted_a"] = df["promoted_a"].fillna(0)
    df = df.dropna(subset=["elo_diff", "fh_pts", "fa_pts"])

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

    parts = ["poisson", "logreg"]
    meta = LogisticRegression(max_iter=3000, C=0.5)
    meta.fit(np.column_stack([preds_va[p] for p in parts]), yva)
    pf = _order(meta, meta.predict_proba(np.column_stack([preds[p] for p in parts])))
    print(f"{'blend':14}{accuracy_score(yte, LABELS_pred(pf)):7.3f}{_ll(yte, pf):10.3f}")

    bp = _book(df)
    if bp is not None:
        t2 = te.copy()
        t2["k"] = (t2["season"] + "|" + t2["Date"].dt.strftime("%Y-%m-%d") + "|"
                   + t2["HomeTeam"].map(normalize_team) + "|" + t2["AwayTeam"].map(normalize_team))
        j = t2.join(bp, on="k").dropna(subset=["pA", "pD", "pH"])
        pb = j[["pA", "pD", "pH"]].to_numpy()
        print(f"{'bookmaker':14}{accuracy_score(j['FTR'], LABELS_pred(pb)):7.3f}"
              f"{_ll(j['FTR'], pb):10.3f}   ({len(j)}/{len(te)})")

    out = te[["season", "comp", "Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out[["p_away", "p_draw", "p_home"]] = pf.round(4)
    out.to_csv(PROC / "outcome_model_predictions.csv", index=False)
    print(f"\nwrote {PROC / 'outcome_model_predictions.csv'}")


def LABELS_pred(p):
    return np.array(LABELS)[p.argmax(1)]


if __name__ == "__main__":
    main()
