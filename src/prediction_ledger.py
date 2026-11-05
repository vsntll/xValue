"""Forward-test ledger: a frozen record of the match odds BEFORE each kickoff,
from two models side by side, and a running scoreboard of how they did against
the market.

Every other prediction file is regenerated on each refresh, so on its own it
proves nothing about what a model said in advance. Here, each refresh appends
one row per upcoming league fixture (kickoff on a later UTC day than the
snapshot, so always pre-match) per model to data/processed/predictions_ledger.csv
with the snapshot time, the data date and a model-version hash of the code that
produced the odds:

  classifier    the headline outcome model (Poisson-Skellam + Elo/form/xG
                blend, train_outcome_model.py) - versions "clf-<hash>"
  dixon_coles   the site's Dixon-Coles attack/defence model (dixon_coles.py) -
                versions "<hash>" (the ledger's first model, so unprefixed)

Rows are never rewritten or deleted - tests/ checks the committed file is a
prefix of every new one.

The scoreboard takes each finished match's LAST pre-kickoff snapshot per model
and scores it against the result, against the de-vigged average CLOSING odds
(football-data.co.uk AvgC*, Bet365 closing as fallback - the market's final,
best-informed price) and against the de-vigged average OPENING odds (AvgH/D/A -
the price around when the snapshot was taken: whether there was an edge at
publication, the betting backtester's benchmark). Head-to-head numbers are on
the fixtures both models have, so they compare like with like.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from live.schema import normalize_team

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
LEDGER = PROC / "predictions_ledger.csv"
COLS = ["snapshot_at", "model_version", "data_asof", "league", "kickoff_date", "home", "away",
        "home_key", "away_key", "p_home", "p_draw", "p_away", "xg_home", "xg_away"]
HORIZON_DAYS = 10
CLF_PREFIX = "clf-"
# the code that turns data into each model's odds - a change to it is a new version
VERSIONED = [ROOT / "src" / "dixon_coles.py", ROOT / "src" / "export_site_data.py"]
VERSIONED_CLF = [ROOT / "src" / "train_outcome_model.py", ROOT / "src" / "build_match_model_table.py"]
MODELS = ("classifier", "dixon_coles")


def _hash(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


def model_version() -> str:
    """Dixon-Coles odds version (unprefixed - the ledger's original model)."""
    return _hash(VERSIONED)


def classifier_version() -> str:
    return CLF_PREFIX + _hash(VERSIONED_CLF)


def model_kind(version: str) -> str:
    return "classifier" if str(version).startswith(CLF_PREFIX) else "dixon_coles"


def read() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(LEDGER)


def _append(cands: list[dict], now: pd.Timestamp, data_asof: pd.Timestamp) -> int:
    """Append the candidate rows whose (match, model) latest snapshot doesn't
    already have the same version and odds."""
    prior = read()
    last = {}
    for r in prior.sort_values("snapshot_at").itertuples(index=False):
        last[(r.kickoff_date, r.home_key, r.away_key, model_kind(r.model_version))] = r
    rows = []
    for c in cands:
        prev = last.get((c["kickoff_date"], c["home_key"], c["away_key"], model_kind(c["model_version"])))
        if (prev is not None and prev.model_version == c["model_version"]
                and np.allclose([prev.p_home, prev.p_draw, prev.p_away],
                                [c["p_home"], c["p_draw"], c["p_away"]], atol=5e-4)):
            continue
        rows.append({"snapshot_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "data_asof": data_asof.strftime("%Y-%m-%d"), **c})
    if rows:
        pd.DataFrame(rows, columns=COLS).to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)
    return len(rows)


def _window(dates: pd.Series, now: pd.Timestamp) -> pd.Series:
    today = now.normalize()
    return (dates.dt.normalize() > today) & (dates <= today + pd.Timedelta(days=HORIZON_DAYS))


def _now(now):
    return pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else now


def snapshot(fixtures: pd.DataFrame, predict, data_asof: pd.Timestamp,
             now: pd.Timestamp | None = None) -> int:
    """Dixon-Coles: every fixture kicking off on a later day than `now` (UTC)
    and within HORIZON_DAYS. `fixtures`: league, Date, HomeTeam, AwayTeam, h, a.
    `predict(h, a)` -> (p_away, p_draw, p_home, xg_home, xg_away)."""
    now = _now(now)
    fx = fixtures[_window(fixtures["Date"], now)]
    ver, cands = model_version(), []
    for r in fx.itertuples(index=False):
        pA, pD, pH, lh, la = predict(r.h, r.a)
        cands.append({"model_version": ver, "league": r.league, "kickoff_date": r.Date.strftime("%Y-%m-%d"),
                      "home": r.HomeTeam, "away": r.AwayTeam, "home_key": r.h, "away_key": r.a,
                      "p_home": round(float(pH), 4), "p_draw": round(float(pD), 4),
                      "p_away": round(float(pA), 4), "xg_home": round(float(lh), 3),
                      "xg_away": round(float(la), 3)})
    return _append(cands, now, data_asof)


def snapshot_classifier(preds: pd.DataFrame, data_asof: pd.Timestamp,
                        now: pd.Timestamp | None = None) -> int:
    """The headline outcome model: its `split == "upcoming"` rows in
    outcome_model_predictions_all.csv (pre-match features from
    build_match_model_table.py), same window as `snapshot`."""
    now = _now(now)
    up = preds[preds["split"] == "upcoming"].copy()
    if up.empty:
        return 0
    up["Date"] = pd.to_datetime(up["Date"])
    up = up[_window(up["Date"], now)]
    ver = classifier_version()
    cands = [{"model_version": ver, "league": r.comp, "kickoff_date": r.Date.strftime("%Y-%m-%d"),
              "home": r.HomeTeam, "away": r.AwayTeam, "home_key": normalize_team(r.HomeTeam),
              "away_key": normalize_team(r.AwayTeam), "p_home": float(r.p_home), "p_draw": float(r.p_draw),
              "p_away": float(r.p_away), "xg_home": np.nan, "xg_away": np.nan}
             for r in up.itertuples(index=False)]
    return _append(cands, now, data_asof)


def _by_team_day(df: pd.DataFrame, home: str, away: str, date: str) -> dict:
    out = {}
    for i, h, a, d in zip(df.index, df[home], df[away], pd.to_datetime(df[date])):
        out[(normalize_team(h), normalize_team(a), d.normalize())] = i
    return out


def _find(idx: dict, h: str, a: str, day: pd.Timestamp):
    for o in (0, -1, 1):
        i = idx.get((h, a, day + pd.Timedelta(days=o)))
        if i is not None:
            return i
    return None


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def _devig(row: pd.Series, groups) -> np.ndarray | None:
    for cols in groups:
        vals = pd.to_numeric(row[list(cols)], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).all() and (vals > 1).all():
            inv = 1 / vals
            return inv / inv.sum()          # proportional de-vig -> [home, draw, away]
    return None


def _outcome_lookup():
    """(home_key, away_key, day) -> (result 0/1/2 = H/D/A, de-vigged closing
    market, de-vigged opening market or None) for finished league matches with
    a closing price, else None."""
    res = pd.read_csv(PROC / "matches_all.csv", low_memory=False)
    res = res[(res["competition_type"] == "league") & res["FTR"].notna()]
    odds = pd.read_csv(PROC / "match_features.csv", low_memory=False)
    r_idx = _by_team_day(res, "HomeTeam", "AwayTeam", "Date")
    o_idx = _by_team_day(odds, "HomeTeam", "AwayTeam", "Date")

    def look(h, a, day):
        ri, oi = _find(r_idx, h, a, day), _find(o_idx, h, a, day)
        if ri is None or oi is None:
            return None
        o = odds.loc[oi]
        close = _devig(o, (("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")))
        if close is None:
            return None
        return ({"H": 0, "D": 1, "A": 2}[res.loc[ri, "FTR"]], close,
                _devig(o, (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"))))
    return look


def scoreboard_xi() -> dict | None:
    """The matchday XI ledger (matchday_rescore.py): plain vs XI-adjusted odds
    vs the closing market, on the same finished matches."""
    f = PROC / "predictions_ledger_xi.csv"
    if not f.exists():
        return None
    led = pd.read_csv(f, dtype={"fotmob_id": str})
    look = _outcome_lookup()
    rows = []
    for r in led.itertuples(index=False):
        hit = look(r.home_key, r.away_key, pd.Timestamp(r.kickoff_utc[:10]))
        if hit is not None:
            rows.append((hit[0], [r.p_home_base, r.p_draw_base, r.p_away_base], [r.p_home, r.p_draw, r.p_away],
                         hit[1].tolist()))
    if not rows:
        return {"n": 0, "pending": int(len(led))}
    y = np.array([r[0] for r in rows])
    ll = {k: round(float(_ll(np.array([r[i] for r in rows]), y).mean()), 4)
          for k, i in (("base_ll", 1), ("xi_ll", 2), ("market_ll", 3))}
    return {"n": len(rows), "pending": int(len(led) - len(rows)), **ll}


def scoreboard() -> dict | None:
    """Running log-loss of each model's frozen pre-match odds vs the closing and
    opening market, head to head on the fixtures both models have."""
    led = read()
    if led.empty:
        return None
    led["kick"] = pd.to_datetime(led["kickoff_date"])
    led = led[pd.to_datetime(led["snapshot_at"].str[:10]) < led["kick"]]
    led["model"] = led["model_version"].map(model_kind)
    last = led.sort_values("snapshot_at").drop_duplicates(
        ["kickoff_date", "home_key", "away_key", "model"], keep="last")
    wide = last.pivot_table(index=["kickoff_date", "home_key", "away_key"], columns="model",
                            values=["p_home", "p_draw", "p_away"], aggfunc="first")
    names = last.drop_duplicates(["kickoff_date", "home_key", "away_key"]).set_index(
        ["kickoff_date", "home_key", "away_key"])[["league", "home", "away"]]
    both = [m for m in MODELS if ("p_home", m) in wide.columns]
    out = {"since": led["snapshot_at"].min()[:10], "models": both,
           "versions": {m: int(last.loc[last["model"] == m, "model_version"].nunique()) for m in both}}

    look = _outcome_lookup()
    rows = []
    for key, w in wide.iterrows():
        probs = {m: [w[("p_home", m)], w[("p_draw", m)], w[("p_away", m)]] for m in both}
        if any(np.isnan(v).any() for v in probs.values()):
            continue                         # head to head: only fixtures every model has
        hit = look(key[1], key[2], pd.Timestamp(key[0]))
        if hit is None:
            continue
        y, close, open_ = hit
        rows.append({"date": key[0], **names.loc[key].to_dict(), "y": y, "close": close.tolist(),
                     "open": None if open_ is None else open_.tolist(), **probs})
    out["pending"] = int(len(wide) - len(rows))
    if not rows:
        return {**out, "n": 0}
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    y = df["y"].to_numpy()
    cols = list(both) + ["close"]
    for c in cols:
        df[f"ll_{c}"] = _ll(np.array(df[c].tolist()), y)
    has_open = df["open"].notna()
    df["ll_open"] = np.nan
    if has_open.any():
        df.loc[has_open, "ll_open"] = _ll(np.array(df.loc[has_open, "open"].tolist()), y[has_open.to_numpy()])
    cum = df.groupby("date")[[f"ll_{c}" for c in cols + ["open"]]].agg(["sum", "count"]).cumsum()
    series = [{"date": d, "n": int(r[(f"ll_{cols[0]}", "count")]),
               **{c: round(r[(f"ll_{c}", "sum")] / r[(f"ll_{c}", "count")], 4)
                  for c in cols + ["open"] if r[(f"ll_{c}", "count")]}}
              for d, r in cum.iterrows()]
    acc = {c: round(float((np.array(df[c].tolist()).argmax(1) == y).mean()), 3) for c in cols}
    recent = [{"date": x["date"], "home": x["home"], "away": x["away"], "result": "HDA"[x["y"]],
               **{c: [round(v, 3) for v in x[c]] for c in cols},
               "open": None if x["open"] is None else [round(v, 3) for v in x["open"]]}
              for x in df.tail(20).iloc[::-1].to_dict("records")]
    return {**out, "n": int(len(df)), "n_open": int(has_open.sum()),
            "ll": {c: round(float(df[f"ll_{c}"].mean()), 4) for c in cols + ["open"] if df[f"ll_{c}"].notna().any()},
            "acc": acc, "series": series, "recent": recent}
