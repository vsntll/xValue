# xValue — a football predictor

Two models over a shared feature layer (see `merged_predictor_architecture.png`):

- **Value regression** — ridge on `log(market value)` from player features.
- **Outcome classifier** — random forest / XGBoost on match features + squad rollup,
  predicting home win / draw / away win.

## Data sources

| Source | What | Script |
| --- | --- | --- |
| football-data.co.uk | league match archive (E0/D1/SP1), 2020-26 | `pull_match_archive.py` |
| FBref (nodriver) | player season stats, all-comps team match logs | `pull_fbref_*.py` |
| Understat | per-match + per-player xG, 3 leagues, 2020-now | `pull_understat.py` |
| ESPN / FotMob / football-data.org | current season + cup stats/xG | `pull_live.py`, `src/live/` |
| worldfootballR mirror | advanced player stats (2020-22), Transfermarkt values (2020-23) | `pull_wfr_advanced.py`, `pull_transfermarkt.py` |

## Pipeline steps

1. ✅ **Match archive** — `pull_match_archive.py`
2. ✅ **Player stats** — `fbref_player_season_stats.csv` (11k players, xG, value)
3. ✅ **Market values** — mirror 2020-23 + TM scrape 2023-26 + Sofascore 2026-27
4. 🔨 Name resolution across sources (alias map in `live/schema.py`, ~90%)
5. ✅ **Match table** — `build_matches_all.py` → `matches_all.csv` (9.1k, xG 75%)
6. ✅ **Squad rollup** — `build_squad_features.py`
7. ✅ **Value regression** — `train_value_model.py` (R²(log) 0.87)
8. ✅ **Outcome classifier** — `train_outcome_model.py` (0.988 pure, 0.975 hybrid vs book 0.973)

See `docs/models.md` for model detail.

See `docs/` for per-source detail.

## Setup

```
python -m pip install -r requirements.txt
```

## Step 1 — match archive

```
python src/pull_match_archive.py
```

Downloads football-data.co.uk season CSVs for the **top flight** of England,
Germany and Spain (E0, D1, SP1) across the last 6 completed seasons **plus the
ongoing 2026-27 season** (re-run to refresh). Keeps the
goals / shots / corners / cards / fouls / result columns plus Bet365 and
market-average odds (for the step 8 benchmark), and concatenates them into one
table (~6,455 matches).

- Raw per-season files: `data/raw/football_data/<season>_<div>.csv`
- Combined table: `data/processed/match_features.csv`
- Data dictionary + caveats: `docs/match_features.md`

**Friendlies and knockout cups are not in football-data.co.uk** — no feed exists.
Those come from FBref team match logs in step 2 and union into the same table via
the `competition_type` column.
