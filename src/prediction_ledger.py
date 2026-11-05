"""Forward-test ledger: a frozen record of the match odds the site published
BEFORE each kickoff, and a running scoreboard of how they did against the
market.

Every other prediction file is regenerated on each refresh, so on its own it
proves nothing about what the model said in advance. Here, each refresh
appends one row per upcoming league fixture (kickoff on a later UTC day than
the snapshot, so it is always pre-match) to data/processed/predictions_ledger.csv
with the snapshot time, the data date and a model-version hash (of the code
that produces the odds). Rows are never rewritten or deleted - tests/ checks
the committed file is a prefix of every new one.

The scoreboard takes each finished match's LAST pre-kickoff snapshot and
scores it against the result and against the de-vigged average CLOSING odds
(football-data.co.uk AvgC*, Bet365 closing as fallback) - the market's final,
best-informed price, which had every bit of news the snapshot didn't.
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
# the code that turns data into the published odds - a change to it is a new model version
VERSIONED = [ROOT / "src" / "dixon_coles.py", ROOT / "src" / "export_site_data.py"]


def model_version() -> str:
    h = hashlib.sha256()
    for f in VERSIONED:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


def read() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(LEDGER)


def snapshot(fixtures: pd.DataFrame, predict, data_asof: pd.Timestamp,
             now: pd.Timestamp | None = None) -> int:
    """Append a snapshot of every fixture kicking off on a later day than `now`
    (UTC) and within HORIZON_DAYS. `fixtures`: league, Date, HomeTeam, AwayTeam,
    h, a. `predict(h, a)` -> (p_away, p_draw, p_home, xg_home, xg_away). A
    fixture whose latest snapshot has the same model version and odds is
    skipped. Returns the number of rows appended."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else now
    today = now.normalize()
    fx = fixtures[(fixtures["Date"].dt.normalize() > today)
                  & (fixtures["Date"] <= today + pd.Timedelta(days=HORIZON_DAYS))]
    if fx.empty:
        return 0
    prior = read()
    last = {}
    if len(prior):
        prior = prior.sort_values("snapshot_at")
        for r in prior.itertuples(index=False):
            last[(r.kickoff_date, r.home_key, r.away_key)] = r
    ver = model_version()
    rows = []
    for r in fx.itertuples(index=False):
        pA, pD, pH, lh, la = predict(r.h, r.a)
        key = (r.Date.strftime("%Y-%m-%d"), r.h, r.a)
        prev = last.get(key)
        if (prev is not None and prev.model_version == ver
                and np.allclose([prev.p_home, prev.p_draw, prev.p_away], [pH, pD, pA], atol=5e-4)):
            continue
        rows.append({"snapshot_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "model_version": ver,
                     "data_asof": data_asof.strftime("%Y-%m-%d"), "league": r.league,
                     "kickoff_date": key[0], "home": r.HomeTeam, "away": r.AwayTeam,
                     "home_key": r.h, "away_key": r.a,
                     "p_home": round(float(pH), 4), "p_draw": round(float(pD), 4),
                     "p_away": round(float(pA), 4), "xg_home": round(float(lh), 3),
                     "xg_away": round(float(la), 3)})
    if rows:
        out = pd.DataFrame(rows, columns=COLS)
        out.to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)
    return len(rows)


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


def scoreboard() -> dict | None:
    """Running log-loss of the frozen pre-match odds vs the closing market."""
    led = read()
    if led.empty:
        return None
    led["kick"] = pd.to_datetime(led["kickoff_date"])
    led = led[pd.to_datetime(led["snapshot_at"].str[:10]) < led["kick"]]
    last = led.sort_values("snapshot_at").drop_duplicates(["kickoff_date", "home_key", "away_key"], keep="last")

    res = pd.read_csv(PROC / "matches_all.csv", low_memory=False)
    res = res[(res["competition_type"] == "league") & res["FTR"].notna()]
    odds = pd.read_csv(PROC / "match_features.csv", low_memory=False)
    r_idx = _by_team_day(res, "HomeTeam", "AwayTeam", "Date")
    o_idx = _by_team_day(odds, "HomeTeam", "AwayTeam", "Date")

    rows = []
    for r in last.itertuples(index=False):
        ri = _find(r_idx, r.home_key, r.away_key, r.kick)
        if ri is None:
            continue
        oi = _find(o_idx, r.home_key, r.away_key, r.kick)
        mk = None
        if oi is not None:
            o = odds.loc[oi]
            for cols in (("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")):
                vals = pd.to_numeric(o[list(cols)], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(vals).all() and (vals > 1).all():
                    inv = 1 / vals
                    mk = inv / inv.sum()     # proportional de-vig -> [home, draw, away]
                    break
        if mk is None:
            continue
        ftr = res.loc[ri, "FTR"]
        rows.append({"date": r.kickoff_date, "league": r.league, "home": r.home, "away": r.away,
                     "p": [r.p_home, r.p_draw, r.p_away], "m": mk.tolist(),
                     "y": {"H": 0, "D": 1, "A": 2}[ftr], "snapshot_at": r.snapshot_at,
                     "model_version": r.model_version})
    if not rows:
        return {"since": led["snapshot_at"].min(), "n": 0, "pending": int(len(last))}
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    p, m, y = np.array(df["p"].tolist()), np.array(df["m"].tolist()), df["y"].to_numpy()
    df["ll_model"], df["ll_market"] = _ll(p, y), _ll(m, y)
    daily = df.groupby("date").agg(n=("y", "size"), lm=("ll_model", "sum"), lk=("ll_market", "sum"))
    cum = daily.cumsum()
    series = [{"date": d, "n": int(c.n), "model": round(c.lm / c.n, 4), "market": round(c.lk / c.n, 4)}
              for d, c in cum.iterrows()]
    recent = [{"date": x.date, "league": x.league, "home": x.home, "away": x.away,
               "p": [round(v, 3) for v in x.p], "m": [round(v, 3) for v in x.m],
               "result": "HDA"[x.y]} for x in df.tail(20).iloc[::-1].itertuples()]
    return {
        "since": led["snapshot_at"].min()[:10], "n": int(len(df)),
        "pending": int(len(last) - len(df)),
        "model_ll": round(float(df["ll_model"].mean()), 4), "market_ll": round(float(df["ll_market"].mean()), 4),
        "model_acc": round(float((p.argmax(1) == y).mean()), 3),
        "market_acc": round(float((m.argmax(1) == y).mean()), 3),
        "versions": int(df["model_version"].nunique()),
        "series": series, "recent": recent,
    }
