# xValue — a football predictor

Two models over a shared data layer for the **big-5 top flights — England,
Spain, Germany, Italy, France, 2020-21 → 2026-27** (+ the cups those clubs play in):

- **Value regression** — predicts a player's market value from his season + value
  history. Stacked trees (direct + change-from-last-value blend, capped at
  Transfermarkt's €220M ceiling), R²(log) **0.89**, MAE **€4.1M**, within-2×
  **91%** on a 2024-26 big-5 holdout — **0.92** with a prior-season value, **15 %**
  median error in the €100M+ band. Outfield + keepers; scores every season
  including the current one.
- **Outcome classifier** — pre-match home-win / draw / away-win. Poisson-Skellam
  on Elo + form + xG, log-loss **0.981** (pure) / **0.972** (with market odds),
  vs Bet365 closing 0.971 — at the market ceiling.

`docs/models.md` has the full model detail; `run_pipeline.md` is the runbook.

## Data sources

| Source | What | Script |
| --- | --- | --- |
| football-data.co.uk | league results + bookmaker odds, 2014-26 (all big-5) | `pull_match_archive.py` |
| FBref (nodriver) | player season stats (PL/BL/La Liga), all-comps team match logs | `pull_fbref_*.py` |
| Understat (soccerdata) | per-match + per-player xG, 2020-now; the Serie A / Ligue 1 player spine | `pull_understat*.py` |
| Transfermarkt (mirror + nodriver) + Sofascore | market values, 2020-27 | `pull_transfermarkt*.py`, `pull_sofascore_values.py` |
| worldfootballR mirror | advanced player stats (xAG, progressive, tackles…), 2020-22 | `pull_wfr_advanced.py` |
| ESPN / FotMob / football-data.org | current season + cup stats & xG | `pull_live.py`, `src/live/` |

Unofficial JSON APIs are behind one pluggable schema (`src/live/`) so a source
going dark is swapped without touching anything downstream.

## Pipeline

| # | step | script | output |
| --- | --- | --- | --- |
| 1 | Match archive | `pull_match_archive.py` | `match_features.csv` |
| 2 | Player stats + xG + value | `pull_fbref_player_stats.py` → `parse_fbref_player_stats.py` → `build_understat_player_seasons.py` (Serie A / Ligue 1 spine) | `fbref_player_season_stats.csv` (~19k rows, 290 cols) |
| 3 | Market values | `pull_transfermarkt*.py`, `pull_sofascore_values.py` | folded into step 2 (~90% labelled) |
| 4 | Cross-source name resolution | alias map + transliteration in `src/live/schema.py`, fuzzy value fill in the parser (~95%) | — |
| 5 | Unified match table | `build_matches_all.py` | `matches_all.csv` (13k matches, all comps) |
| 6 | Squad rollup | `build_squad_features.py` | `squad_season_features.csv` |
| 7 | Value model | `build_value_history.py` → `train_value_model.py` | `models/value_model.pkl` |
| 8 | Outcome model | `build_match_model_table.py` → `train_outcome_model.py` | `outcome_model_predictions*.csv` |

## Setup

```
py -3.11 -m pip install -r requirements.txt   # Python 3.11 required
```

Then follow `run_pipeline.md`.

## Notes

- **Python 3.11** — `nodriver` and soccerdata's TLS client don't work on 3.14 here.
- Browser scrapes (FBref, Transfermarkt) open a visible Chrome window; TM needs
  its AWS-WAF captcha clicked once per session.
- `data/processed/` is committed (built tables + pulled sources); `data/raw/`
  and the fetch caches are gitignored (large and reproducible).
