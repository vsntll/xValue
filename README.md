# xValue — a football predictor

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

Downloads football-data.co.uk season CSVs for the **top two tiers** of England,
Germany and Spain (E0/E1, D1/D2, SP1/SP2) across the last 6 completed seasons
**plus the ongoing 2026-27 season** (re-run to refresh). Keeps the
goals / shots / corners / cards / fouls / result columns plus Bet365 and
market-average odds (for the step 8 benchmark), and concatenates them into one
table (~14,470 matches).

- Raw per-season files: `data/raw/football_data/<season>_<div>.csv`
- Combined table: `data/processed/match_features.csv`
- Data dictionary + caveats: `docs/match_features.md`

**Friendlies and knockout cups are not in football-data.co.uk** — no feed exists.
Those come from FBref team match logs in step 2 and union into the same table via
the `competition_type` column.
