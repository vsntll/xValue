# Models (steps 6-8)

**v2 results** (temporal holdout 2024-25 + 2025-26):

| | value model | outcome model |
| --- | --- | --- |
| metric | R²(log) **0.82**, MAE €6.6M, within-2x **82%** | log-loss **0.990**, acc **0.523** |
| v1 was | 0.70 / €9.1M / 68% | 1.014 / 0.511 |
| reference | — | Bet365 closing 0.973 / 0.537 |


## Step 7 - value regression  (`src/train_value_model.py`)

Predicts a player's market value from his season + his value history.

- **Data**: `fbref_player_season_stats.csv`, outfield players with a
  Transfermarkt/Sofascore value and >= 8 full-90s. ~3,300 train / ~1,570 test.
- **Features** (~23): **prev-season value + 2-seasons-ago value + momentum +
  prev*youth interaction** (value is strongly autocorrelated - the biggest
  lever); age / age^2 / distance-from-26; minutes, starts; per-90 goals /
  assists / npg / shots / SoT / xG / npxG / xA / key passes / xG-chain / fouls;
  one-hot position + league.
- **Target**: `log1p(market_value_eur)`. **Split**: train 2020-24, test 2024-26.
  A CV-based linear de-shrink is applied in log space.
- **Result**: HGB R2(log) **0.82**, MAE **EUR6.6M**, median APE 30%, within-2x
  **82%**. Well-calibrated 5-15M; still ~0.8x on the EUR60M+ tail (superstar
  premium) and ~1.3x on sub-5M players (regression to the mean).
- Output: `data/processed/value_model_predictions.csv`, `models/value_model.pkl`.

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

- Fuzzy player-name matching (accented names like Mbappe lose their xG join).
- Home/away-split rolling form; head-to-head; promoted-team flag; midweek-European
  fatigue.
- Per-match XI values once lineup history exists (API-Football paid, or FotMob).
- Advanced player stats for 2023-26 (blocked - FBref gates, mirror stale).
