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
py -3.11 src/pull_understat_player_matches.py           # per-player per-match xG/xA (Understat, ~1-1.5h, resumable)
py -3.11 src/pull_fotmob_players.py --current           # per-player SoT/fouls/tackles/... (FotMob, current season, cached)
```

## 2. Build the modelling tables

```
py -3.11 src/parse_fbref_matchlogs.py       -> fbref_team_matchlogs.csv
py -3.11 src/parse_fbref_player_stats.py    -> fbref_player_season_stats.csv   (folds in xG + values)
py -3.11 src/build_current_season_stats.py  -> rewrites the current season's rows in fbref_player_season_stats.csv from Understat (goals/assists/minutes/xG) + FotMob (SoT/fouls/tackles/int/blocks/clearances) - no browser scrape
py -3.11 src/build_matches_all.py           -> matches_all.csv                 (9k matches, all comps)
py -3.11 src/build_squad_features.py        -> squad_season_features.csv
py -3.11 src/build_match_model_table.py     -> match_model_table.csv           (Elo, form, odds, momentum if squad_momentum.csv exists yet)
py -3.11 src/build_value_history.py         -> value_history.csv               (prev-value lags: big-5 mirror 2015-22 + scrape + Sofascore)
```

## 3. Train

```
py -3.11 src/train_value_model.py           -> models/value_model.pkl, value_model_predictions.csv  (R2(log) 0.89; predicts every player/season incl. current)
py -3.11 src/build_form_momentum.py         -> squad_momentum.csv    (peer-baseline value vs. actual recent output - needs value_model_predictions.csv above, so run this after it; re-run build_match_model_table.py once more to fold it in)
py -3.11 src/train_outcome_model.py         -> outcome_model_predictions.csv          (pure, log-loss 0.988; + O/U, BTTS, correct-score)
py -3.11 src/train_outcome_model.py --hybrid -> outcome_model_predictions_hybrid.csv  (+ market odds, 0.975)
```

## 4. Build the site

```
py -3.11 src/build_player_elo.py   -> player_elo.csv  (genuine, no-value-model player Elo, last 3 seasons - needs match_model_table.csv + understat_player_matches.csv from steps 1-2)
py -3.11 src/export_site_data.py   -> site/data.json + site/index.html
```

`export_site_data.py` now splices the payload into `site/template.html` itself;
`site/index.html` is the committed, self-contained deliverable.

## 5. Dashboard + source health (optional, local)

```
py -3.11 -m streamlit run src/dashboard.py   # predictions vs results, value leaderboard, source health
py -3.11 src/health_check.py                 # probes every scrape source, writes health_check.json
```

`.github/workflows/health-check.yml` runs the health check every 3 days and
commits the report, so the dashboard's "Source health" tab stays current
without a local run. A transient upstream outage (5xx, an IP block on CI's
range) is reported as `unreachable` but does not fail the run - only a real
schema drift (`DEGRADED` / `ERROR`) does.

## Refresh the current season

**Automatic (every 2 days):** `.github/workflows/weekly-refresh.yml` runs on a
`*/2` cron (and on manual dispatch). It pulls everything that doesn't need a
browser - football-data.co.uk results, live fixtures + match stats (ESPN /
football-data.org / FotMob / Understat), Sofascore market values - **refreshes
the current season's player stats from Understat + FotMob** (`pull_understat.py
--current`, `pull_fotmob_players.py --current`, then `build_current_season_stats.py`),
rebuilds the match table, retrains both models, regenerates `site/index.html`,
and commits the refreshed `data/processed/` **and** `site/index.html` back to
`master`.

`build_current_season_stats.py` rewrites only the current season's rows in
`fbref_player_season_stats.csv`: goals / assists / minutes / MP / cards / shots /
xG / npxG / xAG + the `understat__*` block from Understat; starts from
`understat_player_matches.csv` (`position != 'Sub'`); shots on target / fouls /
tackles / interceptions / blocks / clearances / touches / saves from FotMob's
per-match box scores (`fotmob_player_season.csv`). Each player's birth year,
nationality, and the ~200 unused deep FBref columns (progressive passing, SCA/GCA,
zonal touches, PSxG) are kept from the most recent FBref parse. Finished seasons
are never touched. Names are consolidated across sources - see `player_name_map.csv`.

`data/processed/` is committed (checkout brings it, `git pull` gets you the
latest data locally). `data/raw/` + the soccerdata/Understat fetch caches stay
gitignored - CI keeps them in a non-load-bearing `actions/cache` blob only to
skip re-downloading. Setup: just add repo secret `FOOTBALL_DATA_ORG_KEY`.

**Manual (rarely — only the deep FBref columns nothing currently reads, and
Transfermarkt values; needs a real Chrome window):** re-scrape the full FBref
player-stats set for the depth columns Understat + FotMob don't carry
(progressive passing / carries, SCA/GCA breakdown, zonal touches, PSxG,
pressures), then re-apply the current-season refresh on top.

```
py -3.11 src/pull_fbref_player_stats.py       # league-wide player stats (FBref, ~1h)
py -3.11 src/parse_fbref_player_stats.py      # -> fbref_player_season_stats.csv
py -3.11 src/build_current_season_stats.py    # re-fold Understat's live season back in
# then re-run steps 2-4, or just let the next every-2-days run rebuild the site
```

**Manual (one-off — Serie A / Ligue 1 past values, so a Serie A/Ligue 1 -> our
leagues mover has a prior value; needs visible Chrome + a consent click):**

```
py -3.11 src/pull_transfermarkt_scrape.py --comps ITA1 FRA1 --seasons 2023-24 2024-25 2025-26
py -3.11 src/build_value_history.py        # folds the new values in
# then retrain: train_value_model.py + downstream
```

**Manual (less often — cup xG and Transfermarkt scrape values):**

```
py -3.11 src/pull_fbref_matchlogs.py --stats schedule
py -3.11 src/pull_transfermarkt_scrape.py     # click the WAF captcha once
# then re-run steps 2-4 (or refresh_player_stats.py)
```
