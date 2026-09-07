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
| FBref (nodriver) | finished-season player stats (PL/BL/La Liga) + the deep columns nothing live carries, all-comps team match logs | `pull_fbref_*.py` |
| Understat (soccerdata) | per-match + per-player xG, 2020-now; the current-season goals/assists/minutes/xG and the Serie A / Ligue 1 player spine | `pull_understat*.py` |
| FotMob | current-season SoT / fouls / tackles / interceptions / blocks / clearances / touches / saves (per-match box scores) | `pull_fotmob_players.py` |
| Transfermarkt (mirror + nodriver) + Sofascore | market values, 2020-27 | `pull_transfermarkt*.py`, `pull_sofascore_values.py` |
| worldfootballR mirror | advanced player stats (xAG, progressive, tackles…), 2020-22 | `pull_wfr_advanced.py` |
| ESPN / FotMob / football-data.org | current-season fixtures/results + cup stats & xG | `pull_live.py`, `src/live/` |

Unofficial JSON APIs are behind one pluggable schema (`src/live/`) so a source
going dark is swapped without touching anything downstream.

## Pipeline

| # | step | script | output |
| --- | --- | --- | --- |
| 1 | Match archive + live fixtures | `pull_match_archive.py`, `pull_live.py` | `match_features.csv`, `live_matches_<season>.csv` |
| 2 | Finished-season player stats | `pull_fbref_player_stats.py` → `parse_fbref_player_stats.py` | `fbref_player_season_stats.csv` (~18.9k rows, 291 cols) |
| 3 | Market values | `pull_transfermarkt*.py`, `pull_sofascore_values.py` → `build_value_history.py` | `value_history.csv`; folded into step 2 (~90% labelled) |
| 4 | Serie A / Ligue 1 player spine | `build_understat_player_seasons.py` (Understat + wfr mirror + FotMob + value history) | adds ~7.7k player-seasons to step 2 |
| 5 | Current-season stats | `pull_understat.py --current`, `pull_fotmob_players.py --current` → `build_current_season_stats.py` | rewrites the 2026-27 rows of step 2 — no browser |
| 6 | Cross-source name resolution | alias map + transliteration in `src/live/schema.py`, fuzzy value fill in the parser, `player_name_map.csv` (~95%) | — |
| 7 | Unified match table | `build_matches_all.py` | `matches_all.csv` (~13.5k matches, all comps) |
| 8 | Squad rollup | `build_squad_features.py` | `squad_season_features.csv` |
| 9 | Value model | `train_value_model.py` | `models/value_model.pkl`, `value_model_predictions.csv` |
| 10 | Form / momentum | `build_form_momentum.py` (peer-baseline value vs. actual output) → re-run `build_match_model_table.py` | `squad_momentum.csv` |
| 11 | Outcome model | `build_match_model_table.py` → `train_outcome_model.py [--hybrid]` | `outcome_model_predictions*.csv` |
| 12 | Player Elo | `build_player_elo.py` (opponent-adjusted, no market value) | `player_elo.csv` |
| 13 | Site | `export_site_data.py` | `site/index.html` |

## Setup

```
py -3.11 -m pip install -r requirements.txt   # Python 3.11 required
```

Then follow `run_pipeline.md`.

## Notes

- **Python 3.11** — `nodriver` and soccerdata's TLS client don't work on 3.14 here.
- The ongoing season is **browser-free**: goals/assists/minutes/xG from Understat,
  SoT/fouls/tackles from FotMob, folded in by `build_current_season_stats.py`.
  The FBref browser scrape (visible Chrome) is now only for finished seasons and
  the deep columns nothing else carries — run it roughly monthly. The
  Transfermarkt scrape (visible Chrome + a one-off WAF-captcha click) is also
  monthly-manual.
- `data/processed/` is committed (built tables + pulled sources); `data/raw/`
  and the fetch caches are gitignored (large and reproducible).
