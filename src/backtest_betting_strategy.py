"""Step 9 - betting-strategy backtester: does the outcome model actually have
exploitable edge against the market, and where?

Pure downstream analysis, no new data - `outcome_model_predictions.csv` /
`_hybrid.csv` already carry the test-set (2024-25 + 2025-26, genuinely
out-of-sample) 1X2 probabilities, and `match_features.csv` already carries
the market's opening consensus (`AvgH/D/A` - the closing-average-style column
but for OPENING lines: the average across every book football-data.co.uk
tracks, a cleaner "true market" reference than any single bookmaker, and
what you could actually have bet at before kickoff - using the CLOSING line
here would be leakage, since it reflects information revealed after the bet
would have been placed).

Devig `AvgH/D/A` into implied probabilities (remove the bookmaker's margin,
the same normalise-by-sum-of-inverse-odds as `_book()` in
train_outcome_model.py, which devigs the CLOSING line for benchmarking
instead). `edge = model_p - implied_p` is the model's claimed edge on that
outcome. Two staking strategies:
  - flat: a fixed unit on every outcome where edge clears a threshold
  - kelly: a fraction of a unit sized by the full Kelly criterion (computed
    against the actual, vigged odds - that's what determines the real
    payout), capped - raw Kelly on a noisy edge estimate is a bankroll-ruin
    machine, so both a fractional multiplier and a hard per-bet cap apply

The hybrid model already blends in the market's OWN opening odds as a
feature, so its "edge" against that same market is close to tautologically
small by construction - the PURE model is the more meaningful test of
genuine, model-only edge; both are reported for comparison.

Sliced by odds bucket (favourite vs. longshot) and league (`comp`, already a
column on the predictions) - the hypothesis under test is that any edge
concentrates in longshots or lower-liquidity leagues, where the market is
thinner and slower to correct.

Run:  py -3.11 src/backtest_betting_strategy.py
Output: (stdout) ROI / n bets / win rate, sliced multiple ways, both models
        data/processed/betting_backtest.csv - bet-level detail (pure model)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"

EDGE_THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.12]
KELLY_FRACTION = 0.25   # quarter-Kelly - the standard discount for edge-estimate noise
KELLY_CAP = 0.05        # never stake more than 5% of bankroll on one bet
ODDS_BINS = [1.0, 1.5, 2.0, 3.0, 5.0, 1000.0]
ODDS_LABELS = ["1.0-1.5 (fav)", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+ (longshot)"]


def load_market() -> pd.DataFrame:
    """Opening market consensus, devigged - AvgH/D/A, not AvgCH/CD/CA (the
    closing line isn't a real bet you could have placed at prediction time)."""
    mf = pd.read_csv(PROC / "match_features.csv")
    cols = ["AvgH", "AvgD", "AvgA"]
    mf = mf.dropna(subset=cols).copy()
    mf["k"] = (mf["season"] + "|" + pd.to_datetime(mf["Date"]).dt.strftime("%Y-%m-%d")
               + "|" + mf["HomeTeam"].map(normalize_team) + "|" + mf["AwayTeam"].map(normalize_team))
    inv = 1 / mf[cols].to_numpy()
    dvig = inv / inv.sum(axis=1, keepdims=True)
    mf[["mkt_pH", "mkt_pD", "mkt_pA"]] = dvig
    mf = mf.rename(columns={"AvgH": "oddsH", "AvgD": "oddsD", "AvgA": "oddsA"})
    return mf.set_index("k")[["oddsH", "oddsD", "oddsA", "mkt_pH", "mkt_pD", "mkt_pA"]]


def load_bets(model_file: str) -> pd.DataFrame:
    """One row per (match, candidate outcome) - 3 candidate bets per match,
    filtered to a positive-edge strategy downstream. `model_file` is one of
    outcome_model_predictions.csv (pure) / outcome_model_predictions_hybrid.csv."""
    pred = pd.read_csv(PROC / model_file)
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred["k"] = (pred["season"] + "|" + pred["Date"].dt.strftime("%Y-%m-%d") + "|"
                + pred["HomeTeam"].map(normalize_team) + "|" + pred["AwayTeam"].map(normalize_team))

    market = load_market()
    j = pred.join(market, on="k").dropna(subset=["oddsH", "oddsD", "oddsA"])

    rows = []
    for outcome, p_col, odds_col, mkt_col in (
        ("H", "p_home", "oddsH", "mkt_pH"), ("D", "p_draw", "oddsD", "mkt_pD"),
        ("A", "p_away", "oddsA", "mkt_pA"),
    ):
        rows.append(pd.DataFrame({
            "season": j["season"], "comp": j["comp"], "Date": j["Date"],
            "match": j["HomeTeam"] + " vs " + j["AwayTeam"], "outcome": outcome,
            "model_p": j[p_col], "mkt_p": j[mkt_col], "odds": j[odds_col],
            "won": (j["FTR"] == outcome).astype(int),
        }))
    bets = pd.concat(rows, ignore_index=True)
    bets["edge"] = bets["model_p"] - bets["mkt_p"]
    # full Kelly against the actual (vigged) odds - that's what pays out,
    # regardless of what the devigged "true" probability is
    b = bets["odds"] - 1
    bets["kelly_raw"] = (bets["model_p"] * bets["odds"] - 1) / b
    return bets


def _roi_table(bets: pd.DataFrame, stake: pd.Series) -> dict:
    placed = stake > 0
    if placed.sum() == 0:
        return {"n_bets": 0, "win_rate": np.nan, "roi": np.nan, "staked": 0.0}
    profit = np.where(bets["won"] == 1, stake * (bets["odds"] - 1), -stake)
    return {
        "n_bets": int(placed.sum()),
        "win_rate": float(bets.loc[placed, "won"].mean()),
        "roi": float(profit[placed].sum() / stake[placed].sum()),
        "staked": float(stake[placed].sum()),
    }


def flat_stake(bets: pd.DataFrame, threshold: float) -> pd.Series:
    return (bets["edge"] > threshold).astype(float)


def kelly_stake(bets: pd.DataFrame) -> pd.Series:
    return np.clip(bets["kelly_raw"] * KELLY_FRACTION, 0, KELLY_CAP)


def _sliced(bets: pd.DataFrame, stake: pd.Series, by: str) -> None:
    for name, g in bets.assign(_stake=stake).groupby(by, observed=True):
        r = _roi_table(g, g["_stake"])
        if r["n_bets"] == 0:
            continue
        print(f"  {str(name):20} n={r['n_bets']:4d}  win_rate={r['win_rate']:.0%}  "
              f"ROI={r['roi']:+.1%}  staked={r['staked']:.1f}u")


def run_model(model_file: str, label: str) -> pd.DataFrame:
    bets = load_bets(model_file)
    print(f"\n{'=' * 70}\n{label}  ({len(bets) // 3} matches, {len(bets)} candidate outcomes)\n{'=' * 70}")

    print("\nflat stake, by edge threshold:")
    print(f"  {'threshold':>10}{'n_bets':>8}{'win_rate':>10}{'ROI':>9}")
    for t in EDGE_THRESHOLDS:
        stake = flat_stake(bets, t)
        r = _roi_table(bets, stake)
        print(f"  {t:>10.0%}{r['n_bets']:>8d}{r['win_rate']:>10.0%}{r['roi']:>9.1%}"
              if r["n_bets"] else f"  {t:>10.0%}{'—':>8}")

    kstake = kelly_stake(bets)
    r = _roi_table(bets, kstake)
    print(f"\nfractional Kelly (x{KELLY_FRACTION}, capped {KELLY_CAP:.0%}/bet): "
          f"n={r['n_bets']}  win_rate={r['win_rate']:.0%}  ROI={r['roi']:+.1%}  staked={r['staked']:.1f}u")

    print("\nflat stake (edge > 2%), by odds bucket:")
    bets["odds_bucket"] = pd.cut(bets["odds"], ODDS_BINS, labels=ODDS_LABELS)
    _sliced(bets, flat_stake(bets, 0.02), "odds_bucket")

    print("\nflat stake (edge > 2%), by league:")
    _sliced(bets, flat_stake(bets, 0.02), "comp")

    print("\nfractional Kelly, by league:")
    _sliced(bets, kstake, "comp")

    return bets


def main() -> None:
    pure = run_model("outcome_model_predictions.csv", "PURE MODEL vs. opening market")
    run_model("outcome_model_predictions_hybrid.csv", "HYBRID MODEL vs. opening market "
              "(already blends the market's own opening odds - expect a much smaller edge)")

    pure["flat_stake_2pct"] = flat_stake(pure, 0.02)
    pure["kelly_stake"] = kelly_stake(pure)
    out = PROC / "betting_backtest.csv"
    pure.drop(columns=["kelly_raw"]).to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(pure)} candidate bets, pure model)")


if __name__ == "__main__":
    main()
