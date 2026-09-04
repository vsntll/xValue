# Models (steps 6-8)

## Step 7 - value regression  (`src/train_value_model.py`)

Predicts a player's market value from his season's per-90 productivity + age.

- **Data**: `fbref_player_season_stats.csv`, outfield players with a
  Transfermarkt/Sofascore value and >= 8 full-90s. 3,287 train / 1,586 test.
- **Features** (20): age, age^2, distance-from-26, minutes, starts, per-90 goals /
  assists / npg / shots / SoT / xG / npxG / xA / key passes / xG-chain /
  xG-buildup / fouls / fouled, plus g/shot, shot distance; one-hot position +
  league. (The advanced passing/defense block is 2020-22 only, so it's excluded
  for cross-season generalisation.)
- **Target**: `log1p(market_value_eur)`. **Split**: train 2020-24, test 2024-26.
- **Result**: ridge R2(log) **0.70**, MAE EUR9.1M, median APE 43%, within-2x 68%.
  Systematically under-values the EUR100M+ superstars (Mbappe/Haaland/Bellingham -
  brand/marketability premium the stats don't see) and over-values high-minutes
  role players.
- Output: `data/processed/value_model_predictions.csv`, `models/value_model.pkl`.

## Step 6 - squad rollup  (`src/build_squad_features.py`)

`data/processed/squad_season_features.csv` - one row per (team, season):
`squad_value_eur`, `core18_value_eur`, `xi_value_eur` (top-N by minutes),
`value_known_frac`, minutes-weighted `mean_age_wtd`, `squad_xg`, `squad_xa`.
Season-level (per-match XI rollups need lineup data we only have for 2026-27).

## Step 8 - outcome classifier  (`src/train_outcome_model.py`)

Pre-match home-win / draw / away-win on `matches_all.csv` league rows.

- **Features** (15, all pre-kickoff): rolling mean over the last 8 games (shifted)
  of points / goals for / goals against / xG for / xG against, for home and away
  team; `value_log_ratio` and `age_gap` from step 6; days rest; matchweek bucket.
- **Split**: train <= 2023-24 (4,124), test 2024-25 + 2025-26 (2,111).
- **Result** (log-loss / accuracy):

  | model | acc | log-loss |
  | --- | --- | --- |
  | base rate | .432 | 1.074 |
  | **logreg** | **.511** | **1.014** |
  | hgb | .501 | 1.053 |
  | Bet365 closing | .537 | 0.974 |

  Beats the naive base rate, trails the bookmaker (who also prices injuries,
  lineups, sharp money). Trees don't help - signal is near-linear at this data
  size.
- Output: `data/processed/outcome_model_predictions.csv`.

## Obvious next improvements

- Fuzzy player-name matching (accented names like Mbappe lose their xG join).
- Home/away-split rolling form; head-to-head; promoted-team flag; midweek-European
  fatigue.
- Per-match XI values once lineup history exists (API-Football paid, or FotMob).
- Advanced player stats for 2023-26 (blocked - FBref gates, mirror stale).
