"""Feature table for the outcome model - one row per league match, all features
knowable before kickoff.

Built from `matches_all.csv` (all competitions, so Elo/form pick up midweek
European games) + `squad_season_features.csv`.

Features:
  elo_h, elo_a, elo_diff        goals-based Elo (all comps), pre-match
  xelo_h, xelo_a, xelo_diff     same but updated on xG, not goals
  form_* (home/away split)      rolling-6 pts / GF / GA / xGF / xGA, split by venue
  gf/ga/xg rolling (all venues) rolling-8
  value_log_ratio, age_gap      squad strength
  promoted_h, promoted_a        first season in this league
  days_rest_h/a, h2h_h_pts      congestion, head-to-head
Output: data/processed/match_model_table.csv
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "match_model_table.csv"

HFA = 65.0            # home-field advantage in Elo points
K = 24.0
REVERT = 0.25         # fraction reverted to 1500 between seasons
PROMOTED_ELO = 1400.0


def _elo_update(eh: float, ea: float, res_h: float, gd: float) -> tuple[float, float]:
    exp_h = 1 / (1 + 10 ** ((ea - eh - HFA) / 400))
    mov = np.log(abs(gd) + 1) * (2.2 / (abs(eh - ea) * 0.001 + 2.2))
    delta = K * mov * (res_h - exp_h)
    return eh + delta, ea - delta


def _res(a: float, b: float) -> float:
    return 1.0 if a > b else 0.5 if a == b else 0.0


DIV_COMP = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga"}


def _warmup() -> pd.DataFrame:
    """2014-20 league results (football-data.co.uk) - fed into Elo/form so ratings
    have converged before the modelling window; never emitted as feature rows."""
    f = ROOT / "data" / "raw" / "football_data" / "_elo_warmup.csv"
    if not f.exists():
        return pd.DataFrame()
    w = pd.read_csv(f)
    w["Date"] = pd.to_datetime(w["Date"], dayfirst=True, errors="coerce")
    w["comp"] = w["Div"].map(DIV_COMP)
    w["competition_type"] = "league"
    w["HxG"] = np.nan
    w["AxG"] = np.nan
    w["_warm"] = True
    return w.dropna(subset=["Date"])


def build() -> pd.DataFrame:
    m = pd.read_csv(PROC / "matches_all.csv")
    m["_warm"] = False
    m = pd.concat([_warmup(), m], ignore_index=True)
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
    m = m.dropna(subset=["Date", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
    m["h"] = m["HomeTeam"].map(normalize_team)
    m["a"] = m["AwayTeam"].map(normalize_team)
    for c in ("FTHG", "FTAG", "HxG", "AxG"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    xelo: dict[str, float] = defaultdict(lambda: 1500.0)
    last_season: dict[str, str] = {}
    # rolling deques keyed by (team, 'home'|'away'|'all')
    roll: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=8))
    last_date: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=5))
    league_teams: dict[str, set] = defaultdict(set)

    feats = []
    for _, r in m.iterrows():
        h, a, seas = r["h"], r["a"], r["season"]
        league = r["comp"] if r["competition_type"] == "league" else None

        # season reversion + promoted detection (only for league rows we output)
        for t in (h, a):
            if last_season.get(t) != seas:
                elo[t] = 1500 + (1 - REVERT) * (elo[t] - 1500)
                xelo[t] = 1500 + (1 - REVERT) * (xelo[t] - 1500)
                last_season[t] = seas

        promoted_h = promoted_a = np.nan
        if league is not None:
            prev = league_teams.get(f"{league}|prev|{seas}", None)

        def _roll_mean(key, idx):
            d = roll[key]
            return np.mean([x[idx] for x in d]) if len(d) >= 3 else np.nan

        emit = not r["_warm"]
        row = {
            "season": seas, "comp": r["comp"], "competition_type": r["competition_type"],
            "Date": r["Date"], "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
            "FTR": r["FTR"], "FTHG": r["FTHG"], "FTAG": r["FTAG"],
            "elo_h": elo[h], "elo_a": elo[a], "elo_diff": elo[h] - elo[a] + HFA,
            "xelo_h": xelo[h], "xelo_a": xelo[a], "xelo_diff": xelo[h] - xelo[a] + HFA,
            "fh_pts": _roll_mean((h, "home"), 0), "fh_gf": _roll_mean((h, "home"), 1),
            "fh_ga": _roll_mean((h, "home"), 2), "fh_xgf": _roll_mean((h, "home"), 3),
            "fh_xga": _roll_mean((h, "home"), 4),
            "fa_pts": _roll_mean((a, "away"), 0), "fa_gf": _roll_mean((a, "away"), 1),
            "fa_ga": _roll_mean((a, "away"), 2), "fa_xgf": _roll_mean((a, "away"), 3),
            "fa_xga": _roll_mean((a, "away"), 4),
            "all_h_pts": _roll_mean((h, "all"), 0), "all_a_pts": _roll_mean((a, "all"), 0),
            "all_h_xgd": (_roll_mean((h, "all"), 3) or np.nan) - (_roll_mean((h, "all"), 4) or np.nan),
            "all_a_xgd": (_roll_mean((a, "all"), 3) or np.nan) - (_roll_mean((a, "all"), 4) or np.nan),
            "days_rest_h": (r["Date"] - last_date[h]).days if h in last_date else np.nan,
            "days_rest_a": (r["Date"] - last_date[a]).days if a in last_date else np.nan,
            "h2h_h_pts": np.mean(list(h2h[(h, a)])) if h2h[(h, a)] else np.nan,
        }
        if emit:
            feats.append(row)

        # --- post-match updates ---
        gh, ga_, xh, xa_ = r["FTHG"], r["FTAG"], r["HxG"], r["AxG"]
        elo[h], elo[a] = _elo_update(elo[h], elo[a], _res(gh, ga_), gh - ga_)
        if pd.notna(xh) and pd.notna(xa_):
            xelo[h], xelo[a] = _elo_update(xelo[h], xelo[a], _res(xh, xa_), xh - xa_)
        else:
            xelo[h], xelo[a] = _elo_update(xelo[h], xelo[a], _res(gh, ga_), gh - ga_)
        ph, pa = _res(gh, ga_) * 3 - (gh == ga_), _res(ga_, gh) * 3 - (gh == ga_)
        ph = 3 if gh > ga_ else 1 if gh == ga_ else 0
        pa = 3 if ga_ > gh else 1 if gh == ga_ else 0
        xh2 = xh if pd.notna(xh) else gh
        xa2 = xa_ if pd.notna(xa_) else ga_
        roll[(h, "home")].append((ph, gh, ga_, xh2, xa2))
        roll[(a, "away")].append((pa, ga_, gh, xa2, xh2))
        roll[(h, "all")].append((ph, gh, ga_, xh2, xa2))
        roll[(a, "all")].append((pa, ga_, gh, xa2, xh2))
        last_date[h] = last_date[a] = r["Date"]
        h2h[(h, a)].append(ph)
        h2h[(a, h)].append(pa)

    df = pd.DataFrame(feats)

    # promoted flag: a league team not seen in that league the previous season
    df["_lk"] = df["season"] + "|" + df["comp"]
    seen: dict[str, set] = {}
    prom_h, prom_a = [], []
    order = sorted(df.loc[df.competition_type == "league", "_lk"].unique())
    league_seasons = defaultdict(list)
    for lk in order:
        comp, season = lk.split("|")[1], lk.split("|")[0]
        league_seasons[comp].append(season)
    for _, r in df.iterrows():
        if r["competition_type"] != "league":
            prom_h.append(np.nan); prom_a.append(np.nan); continue
        comp = r["comp"]
        seasons = league_seasons[comp]
        i = seasons.index(r["season"])
        prevset = seen.get(f"{comp}|{seasons[i-1]}") if i > 0 else None
        prom_h.append(np.nan if prevset is None else int(normalize_team(r["HomeTeam"]) not in prevset))
        prom_a.append(np.nan if prevset is None else int(normalize_team(r["AwayTeam"]) not in prevset))
        seen.setdefault(f"{comp}|{r['season']}", set()).update(
            [normalize_team(r["HomeTeam"]), normalize_team(r["AwayTeam"])])
    df["promoted_h"], df["promoted_a"] = prom_h, prom_a

    # squad value / age
    sq = pd.read_csv(PROC / "squad_season_features.csv")
    sq["k"] = sq["season"] + "|" + sq["team_key"]
    vmap = sq.set_index("k")["core18_value_eur"].to_dict()
    amap = sq.set_index("k")["mean_age_wtd"].to_dict()
    hk = df["season"] + "|" + df["HomeTeam"].map(normalize_team)
    ak = df["season"] + "|" + df["AwayTeam"].map(normalize_team)
    df["value_log_ratio"] = np.log((hk.map(vmap).fillna(0) + 5e6) / (ak.map(vmap).fillna(0) + 5e6))
    df["age_gap"] = hk.map(amap) - ak.map(amap)
    df["days_rest_h"] = df["days_rest_h"].clip(0, 14)
    df["days_rest_a"] = df["days_rest_a"].clip(0, 14)

    # Elo -> expected home score (a calibrated starting point for the classifier)
    df["elo_exp_h"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))
    df["xelo_exp_h"] = 1 / (1 + 10 ** (-df["xelo_diff"] / 400))
    # recent scoring / conceding rate blended across venues
    df["h_att"] = df[["fh_xgf", "all_h_xgd"]].mean(axis=1)
    df["a_att"] = df[["fa_xgf", "all_a_xgd"]].mean(axis=1)
    df["form_pts_gap"] = df["all_h_pts"] - df["all_a_pts"]

    out = df[df["competition_type"] == "league"].drop(columns=["_lk"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} league matches, {out.shape[1]} cols)")
    print(out[["elo_diff", "xelo_diff", "value_log_ratio"]].describe().round(1).to_string())
    return out


if __name__ == "__main__":
    build()
