# Models (steps 6-8)

**Current results** (temporal holdout 2024-25 + 2025-26):

| | value model | outcome (pure) | outcome (hybrid) |
| --- | --- | --- | --- |
| metric | R²(log) **0.89**, MAE **€4.9M**, within-2x **91%**, medAPE 23% | log-loss **0.978**, acc **0.532** | log-loss **0.975**, acc **0.531** |
| v1 was | 0.70 / €9.1M / 68% | 1.014 / 0.511 | — |
| reference | — | Bet365 closing 0.973 / 0.537 | (uses market opening odds as a feature) |

Trajectory: value R²(log) 0.70 → 0.82 (prev-value) → 0.835 (contract/minutes) →
0.87 (prev-season club value + xG-share) → 0.88 (name-resolution, coverage
81 → 95%) → 0.90 (`value_history.csv`: big-5 mirror back to 2015 + cross-league)
→ **0.89** (goalkeepers folded in, test set widened to transferred players, and a
direct + change-from-last-value blend with a €220M cap that fixes the elite tail
at a ~0.005 cost to overall R²). Outcome 1.014 → 0.997 → 0.990 → 0.988 → **0.978**
(cleaner team keys). Hybrid **0.975** ≈ bookmaker.

The value model splits cleanly by whether a prior-season value exists:
**R²(log) 0.93** for the 77% that have one, **0.72** for cold-start arrivals from
outside the big-5 (promoted-club squads, Eredivisie/Primeira/Championship
signings, academy graduates). Keepers score **0.88**. Closing the cold-start gap -
and pushing past 0.90 - needs data we don't have: lower-league value history,
transfer fees, or salaries.

**The ceiling and the high end**: Transfermarkt caps listings near **€220M**, so
predictions are clipped there in log space and the elite gravitate toward it
(2025-26: Yamal €200M→€220M, Haaland €200M→€209M, Bellingham €130M→€139M).
`medAPE` for the **€100M+ band is 12%** (was ~30-40% before — the trees squashed
the tail toward the mean). Mid-range predictions carry a **median bias of ~4%**
(pred/listed 1.04) but individual scatter stays ~24% — that's roughly how much a
crowd-sourced value moves between updates, and the season stats don't explain the
revisions, so "every mid-range player within 10%" is not reachable from this data.

**Coverage guarantee**: `parse_fbref_player_stats.py` gives every player with
minutes a `market_value_eur` - a real feed value, else the player's most recent
value carried forward across a transfer, else a position × league × age
peer-median (flagged `market_value_imputed`, never used to fit the model). The
only players left unvalued are arrivals on a just-promoted club with no market
history. `value_model_predictions.csv` is written for every season including the
current one, so the site has a value and a prediction for every current player.

"Hybrid" (`train_outcome_model.py --hybrid`) blends market-consensus **opening**
odds into the stack - it beats the opening line and lands ~level with the
bookmaker's own **closing**-odds performance. The pure model uses only data.


## Step 7 - value regression  (`src/train_value_model.py`)

Predicts a player's market value from his season + his value history.

- **Data**: `fbref_player_season_stats.csv`, players (outfield **and keepers**)
  with a real value and >= 8 full-90s. ~4,160 train / ~2,070 test. Value coverage
  is ~99% of season-rows that have minutes (81% before this work); the gap is
  2022-23 (thin TM mirror) and just-promoted clubs.
- **Prev-value lags** (`build_value_history.py` → `value_history.csv`): the model
  is dominated by last season's value, so coverage of that lag caps accuracy.
  The history table unions the worldfootballR big-5 mirror **back to 2015-16 and
  including Ligue 1 / Serie A** (so 2020-21 rows and cross-league movers get a
  lag), the browser scrape (2023-26), Sofascore (2026-27), and every fuzzy-filled
  label. Features: `prev_log_value` (season-1, else the most recent known value
  carried forward), `prev_staleness`, `prev1/prev2_log_value`, `value_momentum`,
  `prev × age/youth` interactions, `has_prev` / `has_any_prev`.
- **Other features** (~33 total): age / age² / distance-from-26; minutes, starts,
  minutes-trend; per-90 goals / assists / npg / shots / SoT / xG / npxG / xA /
  key passes / xG-chain / fouls; prev-season squad value; xG-share; contract
  years remaining; one-hot position + league.
- **Target**: `log1p(market_value_eur)`. **Split**: train 2020-24, test 2024-26.
  Fit + eval only on real values (peer-median imputations excluded); age falls
  back to (season year - birth year) when FBref leaves it blank early in a season.
- **Model**: two ridge stacks (each over two HGB fits + ExtraTrees + ridge) -
  one predicts `log1p(value)` directly, the other predicts the *change* from the
  player's last known value (a small, bounded quantity; the anchor passes through
  at slope 1 so a €200M prior isn't shrunk toward the mean). The two are
  de-shrunk on OOF and blended `0.5 / 0.5`, then clipped to €220M in log space.
  Rows with < 8 full-90s this season (all of the current season, early on) skip
  the models and predict the last known value along a light age curve.
- **Result**: R2(log) **0.89**, MAE **EUR4.8M**, medAPE **23%**, within-2x
  **91%** (0.93 with a prior value, 0.72 cold-start, 0.88 keepers). €100M+ band
  medAPE **12%**, median pred/listed **1.00**.
- Output: `value_model_predictions.csv` (**every player, every season incl. the
  current one**, with `value_imputed` flag), `models/value_model.pkl`
  (`{bases, meta, cal, features}`).

## Step 6 - squad rollup  (`src/build_squad_features.py`)

`data/processed/squad_season_features.csv` - one row per (team, season):
`squad_value_eur`, `core18_value_eur`, `xi_value_eur` (top-N by minutes),
`value_known_frac`, minutes-weighted `mean_age_wtd`, `squad_xg`, `squad_xa`.
Season-level (per-match XI rollups need lineup data we only have for 2026-27).

## Step 8 - outcome classifier  (`src/train_outcome_model.py` + `build_match_model_table.py`)

Pre-match home-win / draw / away-win on `matches_all.csv` league rows.

- **Feature table** `build_match_model_table.py`: **goals-Elo and xG-Elo**
  (updated across all competitions, seeded with 2014-20 warmup results so ratings
  have converged); home/away-**split** rolling-8 form (pts / goals / xG for &
  against); Elo-implied home prob; squad-value ratio; promoted flags; days rest;
  head-to-head.
- **Models**:
  - logreg on the features
  - **Poisson-Skellam**: two Poisson GLMs (home goals, away goals) -> full
    scoreline distribution with a Dixon-Coles low-score correction (rho tuned on
    a validation season); recency-weighted (2-season half-life). **Best single
    model.**
  - a logreg stack of {poisson, logreg} calibrated on 2023-24
- **Split**: train <= 2022-23, val 2023-24, test 2024-26 (~2,120).
- **Result** (log-loss / accuracy):

  | model | acc | log-loss |
  | --- | --- | --- |
  | base rate | .433 | 1.074 |
  | logreg | .516 | 0.997 |
  | poisson | .519 | 0.990 |
  | **blend** | **.523** | **0.990** |
  | Bet365 closing | .537 | 0.973 |

  Closes ~55% of the base-rate -> bookmaker gap on log-loss. The remaining gap is
  information the market has and we don't (confirmed lineups, injuries, sharp
  money).
- Output: `data/processed/outcome_model_predictions.csv`.

## Obvious next improvements

- **Value model → 0.95**: the has-prev segment is already at 0.93 (Transfermarkt's
  own estimate noise is roughly the ceiling there), so the gain is in the
  cold-start 23%. Needs a prior value for players arriving from outside the
  big-5: a Championship / Eredivisie / Primeira Liga TM scrape, or transfer-fee
  data, or salaries. Also worth trying: a dedicated cold-start sub-model, and a
  real 2022-23 TM scrape to fill the mirror hole.
- Per-match XI values once lineup history exists (API-Football paid, or FotMob).
- Advanced player stats for 2023-26 (blocked - FBref gates, mirror stale).
- Outcome model: confirmed lineups / injuries to close the last of the bookmaker gap.
