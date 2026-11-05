"""Matchday re-score on the confirmed XI (the matchday workflow runs this every
30 minutes when matchday_due.py says a league match is close).

For each big-5 league match kicking off within the next WINDOW minutes whose
lineup FotMob marks as confirmed (lineupType "standard" - before that it serves
a "predicted" XI, which is skipped): the share of each side's top-3 players (by
value, among its matchday squads in the last 45 days) missing from the XI,
the fitted adjustment (data/processed/lineup_adjustment.json, from
fit_lineup_adjustment.py) applied to the site's Dixon-Coles expected goals, and
one row appended to data/processed/predictions_ledger_xi.csv - both the plain
and the XI-adjusted odds, snapshot time and version - so the scoreboard can
show, on real matches, whether the confirmed XI helps. Append-only; one row
per match.

Run:  py -3.11 src/matchday_rescore.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dixon_coles as dc  # noqa: E402
import prediction_ledger  # noqa: E402
from backtest_lineup_absence import WINDOW_DAYS, _values  # noqa: E402
from export_site_data import prep_dc_matches  # noqa: E402
from fit_lineup_adjustment import _lambdas, adjust  # noqa: E402
from live.schema import normalize_team  # noqa: E402
from matchday_due import kickoffs_in_window  # noqa: E402
from parse_fbref_player_stats import _norm_name  # noqa: E402
from pull_fotmob_players import _get  # noqa: E402

PROC = ROOT / "data" / "processed"
LEDGER_XI = PROC / "predictions_ledger_xi.csv"
ADJ = PROC / "lineup_adjustment.json"
COLS = ["snapshot_at", "model_version", "xi_version", "kickoff_utc", "league", "fotmob_id", "home", "away",
        "home_key", "away_key", "home_missing_share", "away_missing_share", "home_missing", "away_missing",
        "p_home_base", "p_draw_base", "p_away_base", "p_home", "p_draw", "p_away"]
LEAGUE_NAME = {"ENG1": "Premier League", "GER1": "Bundesliga", "ESP1": "La Liga",
               "ITA1": "Serie A", "FRA1": "Ligue 1"}


def dc_probs(lh: float, la: float, rho: float) -> np.ndarray:
    """[home, draw, away] from a Dixon-Coles grid - the same form as
    dixon_coles.match_probs, but from given expected goals."""
    i = np.arange(dc.MAXG)[:, None]
    j = np.arange(dc.MAXG)[None, :]
    grid = np.clip(dc.poisson.pmf(i, lh) * dc.poisson.pmf(j, la) * dc._tau(i, j, lh, la, rho), 0, None)
    p = np.array([grid[i > j].sum(), grid[i == j].sum(), grid[i < j].sum()])
    return p / p.sum()


def xi_version() -> str:
    h = hashlib.sha256(ADJ.read_bytes().replace(b"\r\n", b"\n"))
    h.update(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


def missing_share(team_key: str, starters: set[str], recent: pd.DataFrame, values: dict, season: str):
    """Share of the team's top-3 (by value, matchday squads in the window) not
    in `starters`, and who they are."""
    g = recent[recent["team_key"] == team_key].drop_duplicates("_pk")
    g = g.assign(val=[values.get((season, team_key, p), np.nan) for p in g["_pk"]]).dropna(subset=["val"])
    top = g.nlargest(3, "val")
    if top.empty:
        return np.nan, ""
    miss = top[~top["_pk"].isin(starters)]
    return float(miss["val"].sum() / top["val"].sum()), "; ".join(miss["player"])


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    due = kickoffs_in_window(now)
    if not due:
        print("no league kickoff in the window - nothing to do")
        return
    done = set(pd.read_csv(LEDGER_XI, dtype={"fotmob_id": str})["fotmob_id"]) if LEDGER_XI.exists() else set()
    due = [(c, m) for c, m in due if m not in done]
    if not due:
        print("every due match already has an XI snapshot")
        return
    adj = json.loads(ADJ.read_text(encoding="utf-8"))
    matches = prep_dc_matches()
    model = dc.fit(matches, pd.Timestamp(now.date()))
    season = f"{now.year if now.month >= 7 else now.year - 1}-{(now.year + (now.month >= 7)) % 100:02d}"
    lu = pd.read_csv(PROC / "match_lineup_players.csv", dtype={"match_id": str})
    lu["day"] = pd.to_datetime(lu["date"])
    recent = lu[lu["day"] >= pd.Timestamp(now.date()) - pd.Timedelta(days=WINDOW_DAYS)]
    values = _values()

    rows = []
    for code, mid in due:
        try:
            d = _get("matchDetails", matchId=mid)
        except requests.RequestException as e:
            print(f"  {mid}: matchDetails failed ({e})")
            continue
        lineup = (d.get("content") or {}).get("lineup") or {}
        if lineup.get("lineupType") != "standard":
            print(f"  {mid}: lineup not confirmed yet ({lineup.get('lineupType')}) - next run")
            continue
        sides = [lineup.get("homeTeam") or {}, lineup.get("awayTeam") or {}]
        keys = [normalize_team(s.get("name")) for s in sides]
        xis = [{_norm_name(p.get("name")) for p in s.get("starters") or []} for s in sides]
        (sh, mh), (sa, ma) = (missing_share(k, xi, recent, values, season) for k, xi in zip(keys, xis))
        lh, la = _lambdas(model, *keys)
        if np.isnan([lh, la, sh, sa]).any():
            print(f"  {mid}: no rating / squad history for {keys} - skipped")
            continue
        base = dc_probs(lh, la, model["rho"])
        xi = dc_probs(*adjust(lh, la, sh, sa, adj["own"], adj["opp"]), model["rho"])
        g = d.get("general") or {}
        rows.append({"snapshot_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "model_version": prediction_ledger.model_version(), "xi_version": xi_version(),
                     "kickoff_utc": (g.get("matchTimeUTCDate") or "")[:16], "league": LEAGUE_NAME[code],
                     "fotmob_id": mid, "home": sides[0].get("name"), "away": sides[1].get("name"),
                     "home_key": keys[0], "away_key": keys[1],
                     "home_missing_share": round(sh, 3), "away_missing_share": round(sa, 3),
                     "home_missing": mh, "away_missing": ma,
                     **{f"p_{k}_base": round(float(v), 4) for k, v in zip(("home", "draw", "away"), base)},
                     **{f"p_{k}": round(float(v), 4) for k, v in zip(("home", "draw", "away"), xi)}})
        print(f"  {rows[-1]['home']} v {rows[-1]['away']}: missing {sh:.0%} / {sa:.0%}  "
              f"odds {np.round(base, 3)} -> {np.round(xi, 3)}")
    if rows:
        pd.DataFrame(rows, columns=COLS).to_csv(LEDGER_XI, mode="a", header=not LEDGER_XI.exists(), index=False)
    print(f"appended {len(rows)} XI snapshots to {LEDGER_XI.name}")


if __name__ == "__main__":
    main()
