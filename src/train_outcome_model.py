"""Step 8 - outcome classifier: pre-match home-win / draw / away-win.

Features are all knowable before kickoff:
  * rolling form over the last 8 league games (points, goals & xG for/against),
    computed per team and shifted so the current match is excluded
  * squad-value ratio (step 6) and mean-age gap
  * home advantage is implicit (home vs away form are separate features)
  * days rest, matchweek

Target = FTR (H/D/A) on `matches_all.csv` league rows. Temporal split: train
<=2023-24, test 2024-25 + 2025-26. Benchmarked against the bookmaker's implied
probabilities (Bet365 closing odds from match_features.csv) and a naive
base-rate model.

Run:  py -3.11 src/train_outcome_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
K = 8
TRAIN_MAX = "2023-24"
TEST = ["2024-25", "2025-26"]


def _team_timeline(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team-perspective) with rolling pre-match form."""
    recs = []
    for _, m in matches.iterrows():
        for side in ("H", "A"):
            team = m["HomeTeam"] if side == "H" else m["AwayTeam"]
            gf = m["FTHG"] if side == "H" else m["FTAG"]
            ga = m["FTAG"] if side == "H" else m["FTHG"]
            xgf = m["HxG"] if side == "H" else m["AxG"]
            xga = m["AxG"] if side == "H" else m["HxG"]
            pts = 3 if gf > ga else 1 if gf == ga else 0
            recs.append({
                "match_id": m.name, "season": m["season"], "date": m["Date"],
                "team": normalize_team(team), "side": side,
                "gf": gf, "ga": ga, "xgf": xgf, "xga": xga, "pts": pts,
            })
    t = pd.DataFrame(recs).sort_values(["team", "date"])
    g = t.groupby("team", group_keys=False)
    for col in ("pts", "gf", "ga", "xgf", "xga"):
        t[f"roll_{col}"] = g[col].apply(
            lambda s: s.shift(1).rolling(K, min_periods=3).mean())
    t["roll_games"] = g.cumcount()
    return t


def build() -> pd.DataFrame:
    m = pd.read_csv(PROC / "matches_all.csv")
    m = m[(m["competition_type"] == "league") & m["FTR"].isin(["H", "D", "A"])].copy()
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ("FTHG", "FTAG", "HxG", "AxG"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    tl = _team_timeline(m)
    h = tl[tl.side == "H"].set_index("match_id").add_prefix("h_")
    a = tl[tl.side == "A"].set_index("match_id").add_prefix("a_")
    df = m.join(h, how="left").join(a, how="left")

    sq = pd.read_csv(PROC / "squad_season_features.csv")
    sq["k"] = sq["season"] + "|" + sq["team_key"]
    vmap = sq.set_index("k")["core18_value_eur"].to_dict()
    amap = sq.set_index("k")["mean_age_wtd"].to_dict()
    hk = df["season"] + "|" + df["HomeTeam"].map(normalize_team)
    ak = df["season"] + "|" + df["AwayTeam"].map(normalize_team)
    df["value_log_ratio"] = np.log((hk.map(vmap) + 1e6) / (ak.map(vmap) + 1e6))
    df["age_gap"] = hk.map(amap) - ak.map(amap)

    df["days_rest_h"] = (df["Date"] - df.groupby(df["HomeTeam"].map(normalize_team))["Date"]
                         .shift(1)).dt.days.clip(0, 14)
    df["days_rest_a"] = (df["Date"] - df.groupby(df["AwayTeam"].map(normalize_team))["Date"]
                         .shift(1)).dt.days.clip(0, 14)
    df["mw"] = df.groupby(["season", "comp"]).cumcount() // 5
    return df


FEATURES = [
    "h_roll_pts", "h_roll_gf", "h_roll_ga", "h_roll_xgf", "h_roll_xga",
    "a_roll_pts", "a_roll_gf", "a_roll_ga", "a_roll_xgf", "a_roll_xga",
    "value_log_ratio", "age_gap", "days_rest_h", "days_rest_a", "mw",
]


def _book_probs(df: pd.DataFrame) -> pd.DataFrame | None:
    mf = pd.read_csv(PROC / "match_features.csv")
    cols = ["B365CH", "B365CD", "B365CA"] if "B365CH" in mf else ["B365H", "B365D", "B365A"]
    mf = mf.dropna(subset=cols)
    mf["k"] = (mf["season"] + "|" + pd.to_datetime(mf["Date"]).dt.strftime("%Y-%m-%d")
               + "|" + mf["HomeTeam"].map(normalize_team) + "|" + mf["AwayTeam"].map(normalize_team))
    inv = 1 / mf[cols].to_numpy()
    inv = inv / inv.sum(axis=1, keepdims=True)
    mf[["pH", "pD", "pA"]] = inv
    return mf.set_index("k")[["pH", "pD", "pA"]]


LABELS = ["A", "D", "H"]  # lexicographic - what sklearn's log_loss expects


def _ll(y, prob_adh) -> float:
    return log_loss(y, prob_adh, labels=LABELS)


def main() -> None:
    df = build()
    df = df.dropna(subset=["h_roll_pts", "a_roll_pts"])
    tr = df[df["season"] <= TRAIN_MAX]
    te = df[df["season"].isin(TEST)]
    Xtr, ytr = tr[FEATURES].astype(float), tr["FTR"]
    Xte, yte = te[FEATURES].astype(float), te["FTR"]
    print(f"train {len(tr)}  test {len(te)}   train classes {dict(ytr.value_counts())}")

    base = ytr.value_counts(normalize=True).reindex(LABELS).to_numpy()
    print(f"\n{'model':10} {'acc':>6} {'logloss':>8}")
    print(f"{'base-rate':10} {(yte == 'H').mean():6.3f} "
          f"{_ll(yte, np.tile(base, (len(yte), 1))):8.3f}")

    fitted = {}
    for name, clf in {
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegression(max_iter=3000, C=0.4))]),
        "hgb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.04,
                                              max_iter=500, l2_regularization=1.0),
    }.items():
        Xt = Xtr.fillna(Xtr.median()) if name == "logreg" else Xtr
        Xv = Xte.fillna(Xtr.median()) if name == "logreg" else Xte
        clf.fit(Xt, ytr)
        order = [list(clf.classes_).index(c) for c in LABELS]
        p = clf.predict_proba(Xv)[:, order]
        fitted[name] = (clf, p)
        print(f"{name:10} {accuracy_score(yte, clf.predict(Xv)):6.3f} {_ll(yte, p):8.3f}")

    bp = _book_probs(df)
    if bp is not None:
        te2 = te.copy()
        te2["k"] = (te2["season"] + "|" + te2["Date"].dt.strftime("%Y-%m-%d") + "|"
                    + te2["HomeTeam"].map(normalize_team) + "|"
                    + te2["AwayTeam"].map(normalize_team))
        j = te2.join(bp, on="k").dropna(subset=["pA", "pD", "pH"])
        if len(j):
            pb = j[["pA", "pD", "pH"]].to_numpy()
            bacc = (pb.argmax(1) == j["FTR"].map({"A": 0, "D": 1, "H": 2})).mean()
            print(f"{'bookmaker':10} {bacc:6.3f} {_ll(j['FTR'], pb):8.3f}   "
                  f"({len(j)}/{len(te)} matches matched)")

    out = te[["season", "comp", "Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out[["p_away", "p_draw", "p_home"]] = fitted["hgb"][1]
    out.to_csv(PROC / "outcome_model_predictions.csv", index=False)
    print(f"\nwrote {PROC / 'outcome_model_predictions.csv'}")


if __name__ == "__main__":
    main()
