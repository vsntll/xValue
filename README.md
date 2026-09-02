# premierliga — merged predictor

Two models over a shared feature layer (see `merged_predictor_architecture.png`):

- **Value regression** — ridge on `log(market value)` from player features.
- **Outcome classifier** — random forest / XGBoost on match features + squad rollup,
  predicting home win / draw / away win.

## Data sources

| Source | What | Used by |
| --- | --- | --- |
| football-data.co.uk | Match archive CSVs (E0, D1, SP1) | Match features |
| FBref | Season player stats + per-match starting XIs | Player features, squad rollup |
| Transfermarkt | Market value per player per season | Value model target |

## Pipeline steps

1. **Match archive** — `src/pull_match_archive.py` → `data/processed/match_features.csv`
2. Player stats + lineups (FBref)
3. Market values (Transfermarkt)
4. Name resolution across sources
5. Build the three feature tables
6. Squad rollup
7. Train value regression
8. Train outcome classifier

## Setup

```
python -m pip install -r requirements.txt
```

## Step 1 — match archive

```
python src/pull_match_archive.py
```

Downloads the last 6 completed seasons (2020-21 … 2025-26) for the Premier League
(E0), Bundesliga (D1) and La Liga (SP1) from football-data.co.uk, keeps the
goals / shots / corners / cards / fouls / result columns plus Bet365 and market-average
odds (kept for the step 8 benchmark), and concatenates them into one table.

- Raw per-season files: `data/raw/football_data/<season>_<div>.csv`
- Combined table: `data/processed/match_features.csv`
