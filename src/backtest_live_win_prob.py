"""Backtest for live_win_prob.py: replays finished matches minute-by-minute
against their actual goal/red-card timeline (data/raw/live/fotmob_events,
populated by `py -3.11 src/pull_fotmob_players.py --events <season>`) and
scores the live win probability against the REAL final outcome, Brier score
by minute bucket - exactly what the docstring on live_win_prob.py asks for
before wiring it into anything live.

Matching: a FotMob match_id resolves to a match_model_table.csv row (for its
pre-match Elo/form/value features) via the SET of the two teams involved
(season + comp + {team_key, team_key}) - FotMob's own home/away labelling on
the fixtures-by-league endpoint isn't reliable (seen inconsistent against the
actual match report), so match_model_table.csv's own HomeTeam/AwayTeam
(sourced from football-data.co.uk, authoritative) decides which side is home.

Two comparisons:
  1. live_win_prob at each minute checkpoint vs. the SAME pre-match
     probability held static throughout (does re-conditioning on the game
     state actually help, and from when).
  2. On the red-card subset only: applying the disadvantaged-team multiplier
     vs. ignoring the red card entirely, for checkpoints after it - does the
     0.73 factor actually improve calibration here.

Run:  py -3.11 src/backtest_live_win_prob.py --season 2025-26
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402
from live_win_prob import live_win_prob  # noqa: E402
from train_outcome_model import _A_FEATS, _H_FEATS, fit_lambdas  # noqa: E402

PROC = ROOT / "data" / "processed"
EVENTS_CACHE = ROOT / "data" / "raw" / "live" / "fotmob_events"
MATCH_META = PROC / "fotmob_match_meta.csv"
LINEUP_FEATURES = PROC / "match_lineup_features.csv"

SRC_LEAGUE_TO_COMP = {"ENG1": "Premier League", "GER1": "Bundesliga", "ESP1": "La Liga",
                      "ITA1": "Serie A", "FRA1": "Ligue 1"}
TRAIN_SEASONS_MAX = "2022-23"   # same split as train_outcome_model.py's default
MINUTE_CHECKPOINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _resolve_matches(season: str) -> pd.DataFrame:
    """FotMob match_id -> the match_model_table.csv row it corresponds to,
    via the SET of teams involved (season + comp), not FotMob's own home/away
    label."""
    meta = pd.read_csv(MATCH_META, dtype={"match_id": str})
    meta = meta[meta["season"] == season]
    lf = pd.read_csv(LINEUP_FEATURES, dtype={"match_id": str})
    lf = lf[lf["season"] == season]
    teams_by_match = lf.groupby("match_id")["team_key"].apply(frozenset)

    mm = pd.read_csv(PROC / "match_model_table.csv")
    mm = mm[mm["season"] == season].copy()
    mm["Date"] = pd.to_datetime(mm["Date"], errors="coerce")
    mm["hk"] = mm["HomeTeam"].map(normalize_team)
    mm["ak"] = mm["AwayTeam"].map(normalize_team)
    mm["pair"] = [frozenset((h, a)) for h, a in zip(mm["hk"], mm["ak"])]
    mm_by_pair: dict[frozenset, list] = {}
    for i, p in enumerate(mm["pair"]):
        mm_by_pair.setdefault(p, []).append(i)

    meta_date = pd.to_datetime(meta["date"], errors="coerce")
    DATE_TOL_DAYS = 5   # a league season plays each pair twice, home and away -
                        # the pair alone is ambiguous, so pick whichever of the
                        # two meetings' dates is closest to FotMob's own date
    rows, n_no_pair, n_too_far = [], 0, 0
    for mid, comp, mdate in zip(meta["match_id"], meta["src_league"].map(SRC_LEAGUE_TO_COMP), meta_date):
        pair = teams_by_match.get(mid)
        if pair is None or len(pair) != 2:
            n_no_pair += 1
            continue
        cand = [i for i in mm_by_pair.get(pair, []) if mm.iloc[i]["comp"] == comp]
        if not cand:
            n_no_pair += 1
            continue
        best = min(cand, key=lambda i: abs((mm.iloc[i]["Date"] - mdate).days)
                   if pd.notna(mdate) and pd.notna(mm.iloc[i]["Date"]) else 999)
        if pd.notna(mdate) and pd.notna(mm.iloc[best]["Date"]) and \
           abs((mm.iloc[best]["Date"] - mdate).days) > DATE_TOL_DAYS:
            n_too_far += 1
            continue
        rows.append({"match_id": mid, **mm.iloc[best].to_dict()})
    print(f"resolved {len(rows)}/{len(meta)} matches to a match_model_table.csv row "
          f"({n_no_pair} no pair match, {n_too_far} nearest date > {DATE_TOL_DAYS}d)")
    return pd.DataFrame(rows)


def _load_events(match_id: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, str]]]:
    """(goal timeline [(minute, home_score, away_score)], red timeline
    [(minute, 'home'/'away')]) for one match, from the cached event list."""
    f = EVENTS_CACHE / f"{match_id}.json"
    if not f.exists():
        return [], []
    ev = json.loads(f.read_text())
    goals = sorted((e["time"], e["home_score"], e["away_score"])
                   for e in ev if e["type"] == "Goal" and e["home_score"] is not None)
    reds = sorted((e["time"], "home" if e["isHome"] else "away")
                  for e in ev if e["type"] == "Red")
    return goals, reds


def _state_at(goals, reds, minute: int) -> tuple[int, int, int, int]:
    h = a = hr = ar = 0
    for t, gh, ga in goals:
        if t <= minute:
            h, a = gh, ga
    for t, side in reds:
        if t <= minute:
            hr += side == "home"
            ar += side == "away"
    return h, a, hr, ar


def _brier(p: np.ndarray, outcome: np.ndarray) -> float:
    """p: (n, 3) [home, draw, away]; outcome: (n,) in {'H','D','A'}."""
    y = np.zeros_like(p)
    y[:, 0] = outcome == "H"
    y[:, 1] = outcome == "D"
    y[:, 2] = outcome == "A"
    return float(((p - y) ** 2).sum(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    args = ap.parse_args()

    if not EVENTS_CACHE.exists():
        raise SystemExit(f"{EVENTS_CACHE} not found - run "
                          f"pull_fotmob_players.py --events {args.season} first")

    matches = _resolve_matches(args.season)
    matches = matches.dropna(subset=_H_FEATS + _A_FEATS + ["FTHG", "FTAG"], how="any")
    have_events = matches["match_id"].map(lambda m: (EVENTS_CACHE / f"{m}.json").exists())
    matches = matches[have_events].reset_index(drop=True)
    print(f"{len(matches)} matches have a cached event timeline and complete features")

    df = pd.read_csv(PROC / "match_model_table.csv")
    tr = df[df["season"] <= TRAIN_SEASONS_MAX]
    lh, la = fit_lambdas(tr, matches, tr["FTHG"].astype(float), tr["FTAG"].astype(float))
    matches["lh"], matches["la"] = lh, la

    outcome = np.select([matches["FTHG"] > matches["FTAG"], matches["FTHG"] == matches["FTAG"]],
                        ["H", "D"], default="A")

    events = {mid: _load_events(mid) for mid in matches["match_id"]}
    n_red_matches = sum(1 for g, r in events.values() if r)
    print(f"{n_red_matches} matches had at least one red card")

    print(f"\n{'minute':>7}{'live Brier':>12}{'static Brier':>14}{'n':>6}")
    for minute in MINUTE_CHECKPOINTS:
        states = [_state_at(*events[mid], minute) for mid in matches["match_id"]]
        cur_h, cur_a, hr, ar = (np.array(x) for x in zip(*states))
        live = live_win_prob(matches["lh"].to_numpy(), matches["la"].to_numpy(), minute,
                             cur_h, cur_a, home_reds=hr, away_reds=ar)
        static = live_win_prob(matches["lh"].to_numpy(), matches["la"].to_numpy(), 0, 0, 0)
        print(f"{minute:>7}{_brier(live, outcome):>12.4f}{_brier(static, outcome):>14.4f}{len(matches):>6}")

    # red-card check: with vs without the multiplier, on checkpoints strictly
    # after each match's first red card
    print("\nred-card subset: with vs. without the disadvantaged-team multiplier "
          "(checkpoints after the first red)")
    print(f"{'minute':>7}{'with-mult Brier':>18}{'no-mult Brier':>16}{'n':>6}")
    red_mids = [mid for mid in matches["match_id"] if events[mid][1]]
    red_matches = matches[matches["match_id"].isin(red_mids)].reset_index(drop=True)
    red_outcome = np.select(
        [red_matches["FTHG"] > red_matches["FTAG"], red_matches["FTHG"] == red_matches["FTAG"]],
        ["H", "D"], default="A")
    for minute in MINUTE_CHECKPOINTS:
        first_red = np.array([min(t for t, _ in events[mid][1]) for mid in red_matches["match_id"]])
        after = first_red <= minute
        if after.sum() < 5:
            continue
        states = [_state_at(*events[mid], minute) for mid in red_matches["match_id"]]
        cur_h, cur_a, hr, ar = (np.array(x) for x in zip(*states))
        m = red_matches.loc[after]
        idx = after
        with_mult = live_win_prob(m["lh"].to_numpy(), m["la"].to_numpy(), minute,
                                  cur_h[idx], cur_a[idx], home_reds=hr[idx], away_reds=ar[idx])
        no_mult = live_win_prob(m["lh"].to_numpy(), m["la"].to_numpy(), minute,
                                cur_h[idx], cur_a[idx])
        print(f"{minute:>7}{_brier(with_mult, red_outcome[idx]):>18.4f}"
              f"{_brier(no_mult, red_outcome[idx]):>16.4f}{idx.sum():>6}")


if __name__ == "__main__":
    main()
