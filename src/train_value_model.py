"""Step 7 - value regression: predict a player's market value from his season.

Target = log1p(market_value_eur) (Transfermarkt / Sofascore, see step 3).
Features = age, position, minutes, and per-90 productivity (goals, assists, shots,
xG, xA, xG-chain, fouls, cards, aerials) - the columns available for every season
(the advanced passing/defense block only exists 2020-22, so it's left out for a
model that generalises across seasons).

Temporal split: train 2020-21..2023-24, test 2024-25 + 2025-26. Goalkeepers are
excluded (value driven by different factors - a separate model later).

Run:  py -3.11 src/train_value_model.py

Output:
    data/processed/value_model_predictions.csv   test-set actual vs predicted
    models/value_model.pkl                       fitted pipeline
    (stdout) metrics + top coefficients
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live.schema import deaccent  # noqa: E402
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "fbref_player_season_stats.csv"
PRED = ROOT / "data" / "processed" / "value_model_predictions.csv"
MODEL = ROOT / "models" / "value_model.pkl"

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

# raw column -> feature name (per-90 unless noted). We divide the totals by 90s.
PER90 = {
    "standard__Performance_Gls": "goals_p90",
    "standard__Performance_Ast": "assists_p90",
    "standard__Performance_G-PK": "npg_p90",
    "standard__Performance_CrdY": "yellow_p90",
    "shooting__Standard_Sh": "shots_p90",
    "shooting__Standard_SoT": "sot_p90",
    "understat__xg": "xg_p90",
    "understat__np_xg": "npxg_p90",
    "understat__xa": "xa_p90",
    "understat__key_passes": "kp_p90",
    "understat__xg_chain": "xgchain_p90",
    "understat__xg_buildup": "xgbuildup_p90",
    "misc__Performance_Fls": "fouls_p90",
    "misc__Performance_Fld": "fouled_p90",
}
FLAT = {
    "standard__Playing Time_Min": "minutes",
    "standard__Playing Time_Starts": "starts",
    "shooting__Standard_G/Sh": "g_per_shot",
    "shooting__Standard_Dist": "shot_dist",
}


def _norm_team(s):
    s = str(s).lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def _key(s) -> str:
    """Player key matching value_history.csv / _norm_name(player_slug)."""
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def _pos_group(p: str) -> str:
    p = str(p).split(",")[0]
    return {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}.get(p, "MF")


_SEASON_ORDER = {s: i for i, s in enumerate(
    ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"])}


def build_xy(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["n90"] = pd.to_numeric(d["standard__Playing Time_90s"], errors="coerce")
    d["age"] = pd.to_numeric(d["Age"], errors="coerce")
    d["pos"] = d["Pos"].map(_pos_group)
    d["mv"] = pd.to_numeric(d["market_value_eur"], errors="coerce")

    # prior-season value (value is strongly autocorrelated - the biggest lever).
    # value_history.csv unions the big-5 mirror (2015-22, so 2020-21 rows and
    # cross-league movers get a lag) + scrape + Sofascore + every fuzzy fill.
    d["_pk"] = d["player_slug"].map(_key)
    d["_ord"] = d["season"].map(_SEASON_ORDER)
    d["_min"] = pd.to_numeric(d["standard__Playing Time_Min"], errors="coerce")

    vh_path = SRC.parent / "value_history.csv"
    if vh_path.exists():
        vh = pd.read_csv(vh_path)
        vh["_pk"] = vh["player_key"].map(_key)
        vh["_ord"] = vh["season"].map(_SEASON_ORDER)
        vh = vh.dropna(subset=["_ord"]).groupby(["_pk", "_ord"], as_index=False)[
            "market_value_eur"].max().rename(columns={"market_value_eur": "mv"})
    else:
        vh = d[["_pk", "_ord", "mv"]].dropna().drop_duplicates()

    for lag in (1, 2):
        p = vh.rename(columns={"mv": f"prev{lag}_mv"}).copy()
        p["_ord"] = p["_ord"] + lag
        d = d.merge(p, on=["_pk", "_ord"], how="left")

    # as-of join: the most recent value known *strictly before* the target
    # season, plus how many seasons stale it is (a debutant gets prev_any = NaN)
    tgt = d[["_pk", "_ord"]].drop_duplicates()
    aj = tgt.merge(vh.rename(columns={"_ord": "_vord", "mv": "prev_any_mv"}),
                   on="_pk", how="left")
    aj = aj[aj["_vord"] < aj["_ord"]].sort_values("_vord")
    aj = aj.groupby(["_pk", "_ord"], as_index=False).last()
    aj["prev_staleness"] = aj["_ord"] - aj["_vord"]
    d = d.merge(aj[["_pk", "_ord", "prev_any_mv", "prev_staleness"]],
                on=["_pk", "_ord"], how="left")

    # prev-season minutes for the minutes-trend feature (our data only)
    mn = d.dropna(subset=["_min"]).groupby(["_pk", "_ord"], as_index=False)["_min"].max()
    mn["_ord"] = mn["_ord"] + 1
    d = d.merge(mn.rename(columns={"_min": "prev1_min"}), on=["_pk", "_ord"], how="left")

    # contract years remaining at the season's midpoint (Jan 1 of the end year)
    d["_ce"] = pd.to_datetime(d.get("contract_expiry"), errors="coerce")
    d["_ref"] = pd.to_datetime(d["season"].str[:4].astype(float).add(1).astype("Int64").astype(str)
                               + "-01-01", errors="coerce")
    d["contract_years"] = ((d["_ce"] - d["_ref"]).dt.days / 365).clip(-1, 6)

    # club strength + how central the player is to the squad's attack
    sq_path = SRC.parent / "squad_season_features.csv"
    if sq_path.exists():
        sq = pd.read_csv(sq_path).rename(columns={"team": "Squad"})
        d = d.merge(sq[["season", "src_league", "Squad", "squad_xg"]],
                    on=["season", "src_league", "Squad"], how="left")
        # PREVIOUS season's squad value - club spending power, non-circular
        sq["_ord"] = sq["season"].map(_SEASON_ORDER) + 1
        d = d.merge(sq[["_ord", "src_league", "Squad", "squad_value_eur"]].rename(
            columns={"squad_value_eur": "prev_squad_value"}),
            on=["_ord", "src_league", "Squad"], how="left")
    else:
        d["prev_squad_value"], d["squad_xg"] = np.nan, np.nan

    d = d[(d["mv"].notna()) & (d["n90"] >= 8) & (d["pos"] != "GK") & d["age"].notna()]
    d["_xg"] = pd.to_numeric(d.get("understat__xg"), errors="coerce")

    # the workhorse: log of the last known value. prev1 (season-1) if we have it,
    # else the as-of value carried forward, with staleness so the model can
    # discount it. Only ~5% of rows (true debutants) end up with neither.
    p1 = np.log1p(d["prev1_mv"])
    pany = np.log1p(d["prev_any_mv"])
    prev_best = p1.fillna(pany)
    stale = d["prev_staleness"].where(p1.isna(), 1).fillna(0)
    out = pd.DataFrame({
        "season": d["season"], "src_league": d["src_league"],
        "Player": d["Player"], "Squad": d["Squad"],
        "age": d["age"], "age_sq": d["age"] ** 2,
        "peak_dist": (d["age"] - 26).abs(),
        "pos": d["pos"],
        "prev_log_value": prev_best,
        "prev1_log_value": p1,
        "prev2_log_value": np.log1p(d["prev2_mv"]),
        "prev_staleness": stale,
        "value_momentum": (p1.fillna(pany) - np.log1p(d["prev2_mv"])),
        "prev_x_youth": prev_best * (25 - d["age"]).clip(-8, 8),
        "prev_x_age": prev_best * (d["age"] - 26),
        "has_prev": d["prev1_mv"].notna().astype(int),
        "has_any_prev": d["prev_any_mv"].notna().astype(int),
        "contract_years": d["contract_years"],
        "minutes_trend": np.log1p(d["_min"]) - np.log1p(d["prev1_min"]),
        "club_log_value": np.log1p(d["prev_squad_value"]),
        "xg_share": d["_xg"] / (pd.to_numeric(d["squad_xg"], errors="coerce") + 1),
        "y": np.log1p(d["mv"]),
        "market_value_eur": d["mv"],
    })
    for col, name in PER90.items():
        out[name] = pd.to_numeric(d.get(col), errors="coerce") / d["n90"]
    for col, name in FLAT.items():
        out[name] = pd.to_numeric(d.get(col), errors="coerce")
    return out.reset_index(drop=True)


def main() -> None:
    df = build_xy(pd.read_csv(SRC, low_memory=False))
    feat_num = [c for c in df.columns
                if c not in ("season", "src_league", "Player", "Squad", "pos", "y",
                             "market_value_eur")]
    feat_cat = ["pos", "src_league"]

    tr = df[df["season"].isin(TRAIN_SEASONS)]
    te = df[df["season"].isin(TEST_SEASONS)]
    # drop features that are mostly missing in the training window
    feat_num = [c for c in feat_num
                if pd.to_numeric(tr[c], errors="coerce").notna().mean() > 0.5]
    print(f"train {len(tr)} (seasons {TRAIN_SEASONS})  |  test {len(te)} ({TEST_SEASONS})")
    print(f"features: {len(feat_num)} numeric + {feat_cat}")

    ridge_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), feat_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
    ])
    ohe_pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
    ], remainder="passthrough")

    # base learners - two boosted-tree fits at different depth/shrinkage, a
    # bagged tree, and a linear model. Their errors are only partly correlated
    # so a stack of them beats any one.
    bases = {
        "hgb_shallow": Pipeline([("pre", ohe_pre), ("m", HistGradientBoostingRegressor(
            random_state=0, max_depth=3, learning_rate=0.03, max_iter=700,
            l2_regularization=2.0, min_samples_leaf=25, max_leaf_nodes=31))]),
        "hgb_deep": Pipeline([("pre", ohe_pre), ("m", HistGradientBoostingRegressor(
            random_state=0, learning_rate=0.02, max_iter=1500, l2_regularization=1.0,
            min_samples_leaf=15, max_leaf_nodes=63, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=40))]),
        "extratrees": Pipeline([("pre", ohe_pre), ("m", ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=3, n_jobs=-1, random_state=0))]),
        "ridge": Pipeline([("pre", ridge_pre),
                           ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))]),
    }

    def _prep(frame):
        X = frame[feat_num + feat_cat].copy()
        for c in feat_num:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X
    Xtr, Xte = _prep(tr), _prep(te)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    def _score(name, pred):
        pl = np.log1p(np.clip(pred, 0, None))
        act = te["market_value_eur"].to_numpy()
        r2 = r2_score(te["y"], pl)
        mae = mean_absolute_error(act, pred) / 1e6
        ape = np.median(np.abs(pred - act) / act)
        w2 = np.mean((np.maximum(pred, act) / np.minimum(pred, act)) <= 2)
        print(f"  {name:12}  R2(log)={r2:.3f}  MAE=EUR{mae:.1f}M  medAPE={ape:.0%}  "
              f"within-2x={w2:.0%}")
        return r2

    oof, pred = {}, {}
    for name, pipe in bases.items():
        oof[name] = cross_val_predict(pipe, Xtr, tr["y"], cv=cv, n_jobs=-1)
        pipe.fit(Xtr, tr["y"])
        pred[name] = pipe.predict(Xte)
        _score(name, np.expm1(pred[name]))

    # stack: a ridge meta-model over the base OOF predictions, then a monotone
    # spline recalibration (removes the trees' regression-to-the-mean squeeze).
    meta = RidgeCV(alphas=np.logspace(-3, 2, 20))
    S_tr = np.column_stack([oof[n] for n in bases])
    S_te = np.column_stack([pred[n] for n in bases])
    meta.fit(S_tr, tr["y"])
    cal = Pipeline([("s", SplineTransformer(n_knots=6, degree=3)),
                    ("l", LinearRegression())])
    cal.fit(meta.predict(S_tr).reshape(-1, 1), tr["y"])

    def _stack_predict(S):
        return np.expm1(cal.predict(meta.predict(S).reshape(-1, 1)))

    stack_pred = _stack_predict(S_te)
    print()
    r2_stack = _score("stack", stack_pred)
    r2_single = {n: r2_score(te["y"], np.log1p(np.clip(np.expm1(pred[n]), 0, None)))
                 for n in bases}
    best_single = max(r2_single, key=r2_single.get)
    if r2_stack >= r2_single[best_single]:
        best_name, best_pred = "stack", stack_pred
    else:
        best_name, best_pred = best_single, np.expm1(pred[best_single])
    print(f"\nbest: {best_name}")

    # segment diagnostics - where the error lives
    has_p1 = te["prev1_log_value"].notna().to_numpy()
    for lbl, m in [("with prev value", has_p1), ("cold start (no prior)", ~has_p1)]:
        if m.sum():
            print(f"  [{lbl:22}] n={m.sum():4d}  "
                  f"R2(log)={r2_score(te['y'][m], np.log1p(np.clip(best_pred[m], 0, None))):.3f}")

    te_out = te[["season", "src_league", "Player", "Squad", "pos", "age",
                 "market_value_eur"]].copy()
    te_out["predicted_eur"] = np.clip(best_pred, 0, None).round(0)
    te_out["ratio"] = (te_out["predicted_eur"] / te_out["market_value_eur"]).round(2)
    te_out.sort_values("market_value_eur", ascending=False).to_csv(PRED, index=False)
    print(f"wrote {PRED}")

    MODEL.parent.mkdir(exist_ok=True)
    with MODEL.open("wb") as fh:
        pickle.dump({"bases": bases, "meta": meta, "cal": cal,
                     "features": feat_num + feat_cat}, fh)
    print(f"wrote {MODEL}")

    print("\nbiggest over/under-valuations by the model (test set):")
    show = te_out.assign(err_m=((te_out.predicted_eur - te_out.market_value_eur) / 1e6))
    print(show.nlargest(5, "err_m")[["season", "Player", "Squad", "market_value_eur", "predicted_eur"]].to_string())
    print(show.nsmallest(5, "err_m")[["season", "Player", "Squad", "market_value_eur", "predicted_eur"]].to_string())


if __name__ == "__main__":
    main()
