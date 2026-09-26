"""National-team Elo, World Football Elo style (eloratings.net's published
method), over the full international results history - the opponent-strength
input for international appearances in build_player_elo.py, the counterpart of
the club goals-Elo build_match_model_table.py supplies for club matches.

  new = old + K * G * (result - expected)
  expected = 1 / (10^(-dr/400) + 1),  dr = rating gap (+HOME_ADV unless neutral)
  K by match weight (TOURNAMENT_K), G by goal margin (1, 1.5, (11+n)/8)

Everyone starts at 1500 in 1872, so by the window the ratings have had 150
years to converge - no warm-up seeding needed.

Needs data/processed/international_results.csv (src/pull_international_results.py).

Run:  py -3.11 src/build_national_elo.py
Output: data/processed/national_team_elo.csv  (one row per team-match since
        KEEP_FROM: date, team, opponent, pre-match elo of each side)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import deaccent  # noqa: E402

PROC = ROOT / "data" / "processed"
SRC = PROC / "international_results.csv"
OUT = PROC / "national_team_elo.csv"

START = 1500.0
HOME_ADV = 100.0
KEEP_FROM = "2020-07-01"  # only the rows anything downstream reads; the fit uses all history
FRIENDLY_K = 20


def tournament_k(t: str) -> int:
    """eloratings.net's match weights: World Cup finals 60, continental finals
    and intercontinental 50, qualifiers and major tournaments (Nations League
    included) 40, other tournaments 30, friendlies 20."""
    t = deaccent(t).lower()  # "Copa América"
    if t == "fifa world cup":
        return 60
    if "qualification" in t or "nations league" in t:
        return 40
    if t in ("uefa euro", "copa america", "african cup of nations", "afc asian cup",
             "gold cup", "confederations cup", "concacaf championship", "ofc nations cup"):
        return 50
    if t == "friendly":
        return FRIENDLY_K
    return 30


def goal_mult(margin: int) -> float:
    n = abs(margin)
    return 1.0 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found - run src/pull_international_results.py first")
    df = pd.read_csv(SRC).sort_values("date", kind="stable")
    rating: dict[str, float] = {}
    rows = []
    for r in df.itertuples(index=False):
        rh, ra = rating.get(r.home_team, START), rating.get(r.away_team, START)
        dr = rh - ra + (0.0 if r.neutral else HOME_ADV)
        exp_h = 1.0 / (10 ** (-dr / 400) + 1)
        res_h = 1.0 if r.home_score > r.away_score else 0.5 if r.home_score == r.away_score else 0.0
        delta = tournament_k(r.tournament) * goal_mult(r.home_score - r.away_score) * (res_h - exp_h)
        rating[r.home_team], rating[r.away_team] = rh + delta, ra - delta
        if r.date >= KEEP_FROM:
            rows.append({"date": r.date, "team": r.home_team, "opponent": r.away_team,
                         "elo": round(rh, 1), "opp_elo": round(ra, 1)})
            rows.append({"date": r.date, "team": r.away_team, "opponent": r.home_team,
                         "elo": round(ra, 1), "opp_elo": round(rh, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    top = sorted(rating.items(), key=lambda kv: -kv[1])[:10]
    print(f"wrote {OUT}  ({len(out)} team-match rows since {KEEP_FROM})")
    print("top 10 now: " + ", ".join(f"{t} {v:.0f}" for t, v in top))


if __name__ == "__main__":
    main()
