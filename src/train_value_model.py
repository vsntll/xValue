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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    import re as _re
    s = str(s).lower()
    return _re.sub(r"[^a-z0-9 ]", " ", s).strip()


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

    # prior-season value for the same player (value is strongly autocorrelated)
    d["_pk"] = d["Player"].astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True)
    d["_ord"] = d["season"].map(_SEASON_ORDER)
    d["_min"] = pd.to_numeric(d["standard__Playing Time_Min"], errors="coerce")
    hist = d[["_pk", "src_league", "_ord", "mv", "_min"]].drop_duplicates()
    for lag in (1, 2):
        p = hist.copy()
        p["_ord"] = p["_ord"] + lag
        d = d.merge(p[["_pk", "src_league", "_ord", "mv", "_min"]].rename(
            columns={"mv": f"prev{lag}_mv", "_min": f"prev{lag}_min"}),
            on=["_pk", "src_league", "_ord"], how="left")

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
    out = pd.DataFrame({
        "season": d["season"], "src_league": d["src_league"],
        "Player": d["Player"], "Squad": d["Squad"],
        "age": d["age"], "age_sq": d["age"] ** 2,
        "peak_dist": (d["age"] - 26).abs(),
        "pos": d["pos"],
        "prev_log_value": np.log1p(d["prev1_mv"]),
        "prev2_log_value": np.log1p(d["prev2_mv"]),
        "value_momentum": np.log1p(d["prev1_mv"]) - np.log1p(d["prev2_mv"]),
        "prev_x_youth": np.log1p(d["prev1_mv"]) * (25 - d["age"]).clip(-8, 8),
        "has_prev": d["prev1_mv"].notna().astype(int),
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
    hgb_pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
    ], remainder="passthrough")
    def _hgb(**kw):
        return Pipeline([("pre", hgb_pre), ("m", HistGradientBoostingRegressor(
            random_state=0, early_stopping=False, **kw))])
    models = {
        "ridge": Pipeline([("pre", ridge_pre),
                           ("m", RidgeCV(alphas=np.logspace(-2, 3, 30)))]),
        "hgb": _hgb(max_depth=3, learning_rate=0.03, max_iter=700,
                    l2_regularization=2.0, min_samples_leaf=25,
                    max_leaf_nodes=31),
    }

    def _prep(frame):
        X = frame[feat_num + feat_cat].copy()
        for c in feat_num:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X
    Xtr, Xte = _prep(tr), _prep(te)

    from sklearn.model_selection import cross_val_predict

    def _score(name, pred, cal):
        pl = np.log1p(np.clip(pred, 0, None))
        act = te["market_value_eur"].to_numpy()
        r2 = r2_score(te["y"], pl)
        mae = mean_absolute_error(act, pred) / 1e6
        ape = np.median(np.abs(pred - act) / act)
        w2 = np.mean((np.maximum(pred, act) / np.minimum(pred, act)) <= 2)
        print(f"  {name:8}  R2(log)={r2:.3f}  MAE=EUR{mae:.1f}M  medAPE={ape:.0%}  "
              f"within-2x={w2:.0%}" + (f"  (cal {cal[0]:+.2f},{cal[1]:.2f})" if cal else ""))
        return r2

    results, oofs = {}, {}
    for name, pipe in models.items():
        oof = cross_val_predict(pipe, Xtr, tr["y"], cv=4)
        b1, b0 = np.polyfit(oof, tr["y"], 1)
        pipe.fit(Xtr, tr["y"])
        pred = np.expm1(b0 + b1 * pipe.predict(Xte))
        oofs[name] = np.expm1(b0 + b1 * oof)
        results[name] = (pipe, pred, (b0, b1))
        _score(name, pred, (b0, b1))

    # ridge/hgb ensemble, weight chosen on OOF (kept only if it beats hgb)
    from scipy.optimize import minimize_scalar
    a = minimize_scalar(lambda a: -r2_score(
        tr["y"], np.log1p(a * oofs["hgb"] + (1 - a) * oofs["ridge"])),
        bounds=(0, 1), method="bounded").x
    ens = a * results["hgb"][1] + (1 - a) * results["ridge"][1]
    r2_ens = _score(f"ens(a={a:.2f})", ens, None)

    if r2_ens > r2_score(te["y"], np.log1p(np.clip(results["hgb"][1], 0, None))):
        best_name, best_pred = "ensemble", ens
    else:
        best_name, best_pred = "hgb", results["hgb"][1]
    best_pipe = results["hgb"][0]
    print(f"\nbest: {best_name}")

    te_out = te[["season", "src_league", "Player", "Squad", "pos", "age",
                 "market_value_eur"]].copy()
    te_out["predicted_eur"] = best_pred.round(0)
    te_out["ratio"] = (te_out["predicted_eur"] / te_out["market_value_eur"]).round(2)
    te_out.sort_values("market_value_eur", ascending=False).to_csv(PRED, index=False)
    print(f"wrote {PRED}")

    MODEL.parent.mkdir(exist_ok=True)
    with MODEL.open("wb") as fh:
        pickle.dump({"pipeline": best_pipe, "features": feat_num + feat_cat}, fh)
    print(f"wrote {MODEL}")

    if best_name == "ridge":
        pre_f = best_pipe.named_steps["pre"]
        names = feat_num + list(pre_f.named_transformers_["cat"].get_feature_names_out(feat_cat))
        coefs = pd.Series(best_pipe.named_steps["m"].coef_, index=names).sort_values()
        print("\ntop negative / positive coefficients (on log value):")
        print(pd.concat([coefs.head(6), coefs.tail(8)]).round(3).to_string())

    print("\nbiggest over/under-valuations by the model (test set):")
    show = te_out.assign(err_m=((te_out.predicted_eur - te_out.market_value_eur) / 1e6))
    print(show.nlargest(5, "err_m")[["season", "Player", "Squad", "market_value_eur", "predicted_eur"]].to_string())
    print(show.nsmallest(5, "err_m")[["season", "Player", "Squad", "market_value_eur", "predicted_eur"]].to_string())


if __name__ == "__main__":
    main()
