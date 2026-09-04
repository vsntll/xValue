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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "fbref_player_season_stats.csv"
PRED = ROOT / "data" / "processed" / "value_model_predictions.csv"
MODEL = ROOT / "models" / "value_model.pkl"

TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

CAP_EUR = 220_000_000    # Transfermarkt's effective ceiling - predictions clip here
BLEND_W = 0.5            # weight on the change-from-last-value view vs. the direct one

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


_SEASON_ORDER = {f"{y}-{str(y + 1)[-2:]}": y - 2013 for y in range(2013, 2028)}


def build_xy(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["n90"] = pd.to_numeric(d["standard__Playing Time_90s"], errors="coerce")
    d["age"] = pd.to_numeric(d["Age"], errors="coerce")
    # FBref leaves Age blank early in a season - fall back to (start year - birth year)
    _by = pd.to_numeric(d["Born"], errors="coerce")
    d["age"] = d["age"].fillna(d["season"].str[:4].astype(float) - _by)
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

    d["imputed"] = pd.to_numeric(d.get("market_value_imputed"), errors="coerce").fillna(0)
    # keep anyone with a value or any minutes (the n90 >= 8 cut for fit/eval is
    # applied in main; current-season rows have few minutes but still need a
    # predicted value for the site)
    d = d[((d["mv"].notna()) | (d["_min"].fillna(0) > 0)) & d["age"].notna()]
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
        "imputed": d["imputed"].astype(int),
        "n90": d["n90"],
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
                             "market_value_eur", "imputed", "n90")]
    feat_cat = ["pos", "src_league"]

    # fit + evaluate only on rows with a *real* value and a real sample of the
    # season (>= 8 full 90s); never on the parser's peer-median imputations.
    # Predictions are still written for every row, current season included.
    fit_ok = (df["imputed"] == 0) & (df["y"].notna()) & (df["n90"] >= 8)
    tr = df[fit_ok & df["season"].isin(TRAIN_SEASONS)]
    te = df[fit_ok & df["season"].isin(TEST_SEASONS)]
    # drop features that are mostly missing in the training window
    feat_num = [c for c in feat_num
                if pd.to_numeric(tr[c], errors="coerce").notna().mean() > 0.5]
    print(f"train {len(tr)} (seasons {TRAIN_SEASONS})  |  test {len(te)} ({TEST_SEASONS})"
          f"  |  GK in train: {(tr['pos'] == 'GK').sum()}")
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
        # low-minute current-season rows can blow a per-90 rate up to inf
        X[feat_num] = X[feat_num].replace([np.inf, -np.inf], np.nan)
        return X
    Xtr, Xte = _prep(tr), _prep(te)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    ytr = tr["y"].to_numpy()
    # Transfermarkt caps its listings near this; the best players sit just under
    # it. Predictions are squeezed into (~0, CAP] so no one is valued above it
    # and the elite gravitate toward it.
    CAP_LOG = float(np.log1p(CAP_EUR))
    anchor_fill = float(np.nanmedian(tr["prev_log_value"].to_numpy()))

    def _anchor(frame):
        return frame["prev_log_value"].fillna(anchor_fill).to_numpy()

    def _train_stack(target):
        """Fit the base learners + a ridge meta-model on `target`. Returns the
        fitted {name: pipe} dict + meta model, and the OOF meta prediction."""
        bs = {n: clone(p) for n, p in bases.items()}
        oof_ = {}
        for n, p in bs.items():
            oof_[n] = cross_val_predict(p, Xtr, target, cv=cv, n_jobs=-1)
            p.fit(Xtr, target)
        cols = list(bs)
        mt = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(
            np.column_stack([oof_[n] for n in cols]), target)
        return {"bases": bs, "meta": mt, "cols": cols}, \
            mt.predict(np.column_stack([oof_[n] for n in cols]))

    def _stack_pred(st, Xf):
        return st["meta"].predict(
            np.column_stack([st["bases"][n].predict(Xf) for n in st["cols"]]))

    # two views of the target: predict log-value directly, and predict the
    # *change* from the last known value (small, bounded - keeps mid-range tight
    # and lets a superstar's prediction climb back toward his prior value).
    direct_st, direct_oof = _train_stack(ytr)
    anch_tr = _anchor(tr)
    resid_tgt = ytr - anch_tr
    resid_st, resid_oof = _train_stack(resid_tgt)

    # de-shrink the DIRECT view against y; de-shrink only the residual *delta*
    # (the anchor passes through at slope 1, so a EUR200M prior stays near EUR200M).
    da1, da0 = np.polyfit(direct_oof, ytr, 1)
    ra1, ra0 = np.polyfit(resid_oof, resid_tgt, 1)

    # small age curve for carrying a value forward: flat through the mid-20s,
    # about -6%/yr after 30, +4%/yr for U21 (fit loosely to how TM values age)
    def _age_adj(frame):
        a = pd.to_numeric(frame["age"], errors="coerce").fillna(26).to_numpy()
        return np.where(a >= 30, -0.06 * (a - 30),
                        np.where(a <= 21, 0.04 * (21 - a), 0.0))

    def _predict_eur(frame):
        Xf = _prep(frame)
        d = da0 + da1 * _stack_pred(direct_st, Xf)
        r = _anchor(frame) + ra0 + ra1 * _stack_pred(resid_st, Xf)
        blended = np.clip(BLEND_W * r + (1 - BLEND_W) * d, None, CAP_LOG)
        # too few minutes this season for any form signal - the best estimate is
        # simply last known value, nudged along an age curve.
        n90 = pd.to_numeric(frame["n90"], errors="coerce").fillna(0).to_numpy()
        carried = np.clip(_anchor(frame) + _age_adj(frame), None, CAP_LOG)
        thin = n90 < 8
        log_v = np.where(thin, carried, blended)
        return np.clip(np.expm1(log_v), 1e4, CAP_EUR)

    best_pred = _predict_eur(te)
    act = te["market_value_eur"].to_numpy()
    r2 = r2_score(te["y"], np.log1p(best_pred))
    ape = np.abs(best_pred - act) / act
    print(f"\n  blend(w={BLEND_W}, cap EUR{CAP_EUR/1e6:.0f}M)  R2(log)={r2:.3f}  "
          f"MAE=EUR{mean_absolute_error(act, best_pred)/1e6:.1f}M  "
          f"medAPE={np.median(ape):.0%}  within-2x="
          f"{np.mean(np.maximum(best_pred, act) / np.minimum(best_pred, act) <= 2):.0%}")

    # segment diagnostics - error by value band and by group
    for lbl, m in [("outfield", (te["pos"] != "GK").to_numpy()),
                   ("goalkeepers", (te["pos"] == "GK").to_numpy()),
                   ("with prev value", te["prev1_log_value"].notna().to_numpy()),
                   ("cold start", te["prev1_log_value"].isna().to_numpy()),
                   ("mid  EUR3-40M", ((act >= 3e6) & (act < 40e6))),
                   ("high EUR40-100M", ((act >= 40e6) & (act < 100e6))),
                   ("elite EUR100M+", (act >= 100e6))]:
        if m.sum():
            bias = np.median(best_pred[m] / act[m])
            print(f"  [{lbl:16}] n={m.sum():4d}  R2(log)="
                  f"{r2_score(te['y'][m], np.log1p(best_pred[m])):.3f}  "
                  f"medAPE={np.median(ape[m]):.0%}  pred/listed(med)={bias:.2f}")

    # predictions for EVERY eligible row (all seasons incl. 2026-27, GK included,
    # peer-imputed rows flagged) so the site has a value for every current player.
    allp = df.copy()
    allp["predicted_eur"] = _predict_eur(allp).round(0)
    out = allp[["season", "src_league", "Player", "Squad", "pos", "age",
                "market_value_eur", "predicted_eur", "imputed"]].rename(
        columns={"imputed": "value_imputed"})
    out["ratio"] = (out["predicted_eur"] / out["market_value_eur"]).round(2)
    out.sort_values(["season", "market_value_eur"], ascending=[True, False]).to_csv(
        PRED, index=False)
    print(f"wrote {PRED}  ({len(out)} rows, seasons {sorted(out['season'].unique())})")

    MODEL.parent.mkdir(exist_ok=True)
    with MODEL.open("wb") as fh:
        pickle.dump({"direct": direct_st, "resid": resid_st,
                     "anchor_fill": anchor_fill, "blend_w": BLEND_W,
                     "cap_eur": CAP_EUR, "cal": (da0, da1, ra0, ra1),
                     "features": feat_num + feat_cat}, fh)
    print(f"wrote {MODEL}")

    print("\nbiggest over/under-valuations by the model (test set):")
    show = te[["season", "Player", "Squad", "market_value_eur"]].copy()
    show["predicted_eur"] = best_pred.round(0)
    show["err_m"] = (show["predicted_eur"] - show["market_value_eur"]) / 1e6
    print(show.nlargest(5, "err_m").to_string())
    print(show.nsmallest(5, "err_m").to_string())


if __name__ == "__main__":
    main()
