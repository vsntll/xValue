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
py -3.11 src/build_matches_all.py           -> matches_all.csv                 (9k matches, all comps)
py -3.11 src/build_squad_features.py        -> squad_season_features.csv
py -3.11 src/build_match_model_table.py     -> match_model_table.csv           (Elo, form, odds)
py -3.11 src/build_value_history.py         -> value_history.csv               (prev-value lags: big-5 mirror 2015-22 + scrape + Sofascore)
```

## 3. Train

```
py -3.11 src/train_value_model.py           -> models/value_model.pkl, value_model_predictions.csv  (R2(log) 0.89; predicts every player/season incl. current)
py -3.11 src/train_outcome_model.py         -> outcome_model_predictions.csv          (pure, log-loss 0.988)
py -3.11 src/train_outcome_model.py --hybrid -> outcome_model_predictions_hybrid.csv  (+ market odds, 0.975)
```

## 4. Build the site

```
py -3.11 src/export_site_data.py   -> site/data.json + site/index.html
```

`export_site_data.py` now splices the payload into `site/template.html` itself;
`site/index.html` is the committed, self-contained deliverable.

## Refresh the current season

**Automatic (weekly):** `.github/workflows/weekly-refresh.yml` runs every Monday
(and on manual dispatch). It pulls everything that doesn't need a browser -
football-data.co.uk results, live fixtures + match stats (ESPN / football-data.org
/ FotMob / Understat), Sofascore market values - then rebuilds the match table,
retrains both models, regenerates `site/index.html` and commits it.

Setup: add repo secret `FOOTBALL_DATA_ORG_KEY`; seed the (gitignored) data cache
once with a GitHub Release the workflow can fall back to:

```
tar czf pipeline-seed.tar.gz data/processed data/raw/football_data/_elo_warmup.csv
gh release create pipeline-seed pipeline-seed.tar.gz -t "pipeline data seed"
```

**Manual (goals / assists / minutes — needs a real Chrome window):** run this
every week or two; it re-scrapes the current season's FBref player stats,
rebuilds the tables, retrains the value model and regenerates the page.

```
py -3.11 refresh_player_stats.py            # scrape + rebuild; you commit
py -3.11 refresh_player_stats.py --commit   # also commit + push (fires the Pages deploy)
```

**Manual (less often — cup xG and Transfermarkt scrape values):**

```
py -3.11 src/pull_fbref_matchlogs.py --stats schedule
py -3.11 src/pull_transfermarkt_scrape.py     # click the WAF captcha once
# then re-run steps 2-4 (or refresh_player_stats.py)
```
