# Running the pipeline

All scripts run under **Python 3.11** (`py -3.11 ...`). Raw data caches under
`data/raw/` and `data/*_cache/`; built tables land in `data/processed/`.
Everything is resumable - scripts skip work already cached.

## 1. Ingest (slow; browser scrapes need a visible Chrome window)

```
py -3.11 src/pull_match_archive.py                 # league results, 2020-26      (football-data.co.uk)
py -3.11 src/pull_fbref_matchlogs.py --stats schedule   # cup/European rows       (FBref, nodriver, ~1.5h)
py -3.11 src/pull_fbref_matchlogs.py                    # + shooting/keeper/misc  (~4h, re-run to resume)
py -3.11 src/pull_fbref_player_stats.py                 # league-wide player stats (FBref, ~1h)
py -3.11 src/pull_wfr_advanced.py                       # advanced player stats    (mirror, 2020-22)
py -3.11 src/pull_understat.py                          # per-match + per-player xG (Understat)
py -3.11 src/pull_transfermarkt.py                      # market values 2020-23    (mirror)
py -3.11 src/pull_transfermarkt_scrape.py               # market values 2023-26    (TM, click the WAF captcha once)
py -3.11 src/pull_sofascore_values.py                   # current-season values    (Sofascore)
py -3.11 src/pull_live.py --season 2026-27              # live season, all comps   (ESPN+fdorg+Understat+FotMob)
py -3.11 src/pull_live.py --seasons 2024-25 2025-26 --comps UCL UEL UECL FA EFL DFB CDR --sources espn fotmob
```

## 2. Build the modelling tables

```
py -3.11 src/parse_fbref_matchlogs.py       -> fbref_team_matchlogs.csv
py -3.11 src/parse_fbref_player_stats.py    -> fbref_player_season_stats.csv   (folds in xG + values)
py -3.11 src/build_matches_all.py           -> matches_all.csv                 (9.1k matches, all comps)
py -3.11 src/build_squad_features.py        -> squad_season_features.csv
py -3.11 src/build_match_model_table.py     -> match_model_table.csv           (Elo, form, odds)
```

## 3. Train

```
py -3.11 src/train_value_model.py           -> models/value_model.pkl, value_model_predictions.csv
py -3.11 src/train_outcome_model.py         -> outcome_model_predictions.csv          (pure, log-loss 0.988)
py -3.11 src/train_outcome_model.py --hybrid -> outcome_model_predictions_hybrid.csv  (+ market odds, 0.975)
```

## Refresh the current season (weekly)

```
py -3.11 src/pull_live.py                    # current season
py -3.11 src/pull_understat.py --seasons 2026-27
py -3.11 src/pull_sofascore_values.py
# then re-run step 2 + step 3
```
