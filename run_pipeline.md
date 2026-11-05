# Running the pipeline

All scripts run under **Python 3.11** (`py -3.11 ...`). Pulled and built
tables land in `data/processed/` (committed); `data/raw/` and
`data/understat_cache/` are gitignored scratch. Everything is resumable - the
match-level pulls skip every match already in their committed CSVs, and the
FotMob / ESPN pulls take `--budget-min` to cap fetching time per run.

## 1. Ingest (slow; browser scrapes need a visible Chrome window)

```
py -3.11 src/pull_match_archive.py                 # league results, 2020-26      (football-data.co.uk)
py -3.11 src/pull_fbref_matchlogs.py --stats schedule   # cup/European rows       (FBref, nodriver, ~1.5h)
py -3.11 src/pull_fbref_matchlogs.py                    # + shooting/keeper/misc  (~4h, re-run to resume)
py -3.11 src/pull_fbref_player_stats.py                 # league-wide player stats (FBref, ~1h)
py -3.11 src/pull_wfr_advanced.py                       # advanced player stats    (mirror, 2020-22)
py -3.11 src/pull_understat.py                          # per-match + per-player xG, all big-5 (Understat)
py -3.11 src/pull_transfermarkt.py                      # market values 2020-23    (mirror)
py -3.11 src/pull_transfermarkt_scrape.py               # market values 2023-26    (TM, click the WAF captcha once)
py -3.11 src/pull_sofascore_values.py                   # current-season values    (Sofascore; stops at the first IP block and keeps the last good CSV)
py -3.11 src/pull_live.py --season 2026-27              # live season, all comps   (ESPN+fdorg+Understat+FotMob)
py -3.11 src/pull_live.py --seasons 2024-25 2025-26 --comps UCL UEL UECL FA EFL DFB CDR --sources espn fotmob
py -3.11 src/pull_understat_player_matches.py           # per-player per-match xG/xA (Understat, ~1-1.5h first time; then only games not yet in the CSV)
py -3.11 src/pull_fotmob_players.py --current           # per-player SoT/fouls/tackles/passes/... (FotMob, current season; only uncommitted matches are fetched)
py -3.11 src/pull_fotmob_players.py --backfill-passes 2024-25 2025-26 2026-27   # one-off: re-fetch committed matches that predate passes_completed/attempted (resumable, skips done ones)
py -3.11 src/pull_international_results.py             # every men's international result since 1872 (martj42 dataset, one download)
py -3.11 src/pull_fotmob_internationals.py             # per-player international box scores since 2024-07-01 (FotMob; only matches not yet committed)
py -3.11 src/pull_espn_players.py --current             # per-player box scores, five leagues (ESPN) - second source for the cross-check
py -3.11 src/verify_player_matches.py --current         # cross-check FotMob rows vs Understat / ESPN / football-data (internationals: martj42) -> `verified` column
py -3.11 src/understat_gapfill.py                       # while Understat is down: rebuild its missing matches from FotMob -> understat_gapfill_*.csv
```

## 2. Build the modelling tables

```
py -3.11 src/parse_fbref_matchlogs.py       -> fbref_team_matchlogs.csv
py -3.11 src/parse_fbref_player_stats.py       -> fbref_player_season_stats.csv   (folds in xG + values; PL/BL/La Liga only)
py -3.11 src/build_value_history.py            -> value_history.csv               (prev-value lags + the Serie A/Ligue 1 season value: big-5 mirror 2015-22 + scrape + Sofascore. A clean first build converges on the 2nd pass.)
py -3.11 src/build_understat_player_seasons.py -> + Serie A / Ligue 1 rows, all seasons, synthesised from Understat + wfr mirror + FotMob + value_history (idempotent)
py -3.11 src/build_current_season_stats.py     -> rewrites the current season's rows in fbref_player_season_stats.csv from Understat (goals/assists/minutes/xG) + FotMob (SoT/fouls/tackles/int/blocks/clearances) - no browser scrape; also gives every unvalued current player with minutes a value (division-adjusted positional median for just-promoted squads, flagged imputed)
py -3.11 src/build_matches_all.py           -> matches_all.csv                 (~13.5k matches, all comps)
py -3.11 src/build_squad_features.py        -> squad_season_features.csv
py -3.11 src/build_match_model_table.py     -> match_model_table.csv           (Elo, form, odds, momentum if squad_momentum.csv exists yet)
```

## 3. Train

```
py -3.11 src/train_value_model.py           -> models/value_model.pkl, value_model_predictions.csv  (R2(log) 0.89; predicts every player/season incl. current)
py -3.11 src/build_form_momentum.py         -> squad_momentum.csv    (peer-baseline value vs. actual recent output - needs value_model_predictions.csv above, so run this after it; re-run build_match_model_table.py once more to fold it in)
py -3.11 src/train_outcome_model.py         -> outcome_model_predictions.csv + _all.csv (market-free; _all also holds the upcoming fixtures' pre-match odds)
py -3.11 src/train_outcome_model.py --hybrid -> outcome_model_predictions_hybrid.csv  (+ market opening odds as features)
py -3.11 src/dixon_coles.py                  -> dixon_coles_probs.csv  (walk-forward, incremental; --full rebuilds, ~20 min)
# current scores: README / docs/models.md (generated from model_metrics.json)
```

## 4. Build the site

```
py -3.11 src/build_national_elo.py -> national_team_elo.csv  (World Football Elo over all international history - opponent strength for international appearances)
py -3.11 src/build_player_elo.py   -> player_elo.csv  (genuine, no-value-model player Elo, last 3 seasons, role-weighted attack/defence/passing - needs match_model_table.csv + understat_player_matches.csv (+ the FotMob gap-fill) from steps 1-2, and FotMob's committed per-match rows (data/processed/fotmob/) for the def/pass components, plus fotmob_intl_player_matches.csv + national_team_elo.csv for international appearances; prints the FotMob join, pass-data and international coverage)
py -3.11 src/export_site_data.py   -> site/data.json + site/index.html
```

`export_site_data.py` now splices the payload into `site/template.html` itself;
`site/index.html` is the self-contained deliverable - published by the refresh
(release asset + Pages), not committed.

```
py -3.11 src/build_sqlite_db.py   -> data/processed/xvalue.db  (read-only SQL mirror of every processed CSV, one table per file)
```

The SQLite mirror is local-only: it's gitignored and not built in CI (~150 MB,
over GitHub's 100 MB file limit, and nothing in the pipeline reads it).

## Matchday re-score (automatic)

`.github/workflows/matchday.yml` (every 30 min): `src/matchday_due.py` (stdlib
only - exits in seconds unless a big-5 league match kicks off within ~2 h), then
`src/matchday_rescore.py` re-scores matches whose FotMob XI is confirmed and
appends to `data/processed/predictions_ledger_xi.csv`. The adjustment it applies
comes from `src/fit_lineup_adjustment.py` (re-fit occasionally:
`py -3.11 src/fit_lineup_adjustment.py --refit`, ~12 min; without `--refit` it
reuses `dc_walk_lambdas.csv`). See `docs/models.md`.

## 5. Tests, model scores + regression gate

```
py -3.11 -m pytest -q tests                  # data invariants on the processed CSVs + unit tests
py -3.11 src/check_model_metrics.py          # new model scores vs last commit + accepted_metrics.json
py -3.11 src/update_doc_metrics.py           # write the scores into README.md + docs/models.md
```

`train_value_model.py` and `train_outcome_model.py` write their test-set scores to
`data/processed/model_metrics.json`. It's the only source of the model numbers
in README.md and docs/models.md (between `<!-- metrics:... -->` markers - edit the
text around them, not inside), and the baseline for the regression gate: the
refresh fails, before committing, if value R²(log) drops more than 0.02 or 1X2
log-loss rises more than 0.01 against the committed scores. `tests/` checks the
rebuilt CSVs (unique match keys, 1X2 rows summing to 1, predicted values under
the €220M cap, no row-count collapse vs the committed files, FotMob rows unique)
plus unit tests for `dixon_coles.py`, `live_win_prob.py` (at minute 90 the
current score is final), the team-name alias map and the gate itself.

The gate checks the same tolerances against two baselines: the last committed
`model_metrics.json` (run to run) and the hand-pinned `accepted_metrics.json` at
the repo root - so a model can't be walked down a little each run (every step
inside tolerance of the last, the total well outside it). Only you change the
pinned one: after a deliberate model change, check the numbers, run
`py -3.11 src/check_model_metrics.py --accept`, and commit `model_metrics.json`,
`accepted_metrics.json` and the regenerated docs together.

## 6. Dashboard + source health (optional, local)

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

**Automatic (every 2 days):** `.github/workflows/refresh.yml` runs on a
`*/2` cron (and on manual dispatch). It pulls everything that doesn't need a
browser - football-data.co.uk results, live fixtures + match stats (ESPN /
football-data.org / FotMob / Understat), Sofascore market values, FotMob + ESPN
per-player box scores, internationals - cross-checks the per-player rows,
fills Understat's gaps from FotMob, **refreshes the current season's player
stats** (`build_current_season_stats.py`), rebuilds the match table, retrains
both models, **stops if either model got worse** (regression gate), regenerates
`site/index.html`, writes the new model scores into the docs, **runs `tests/`**,
and commits the refreshed sources, ledgers, model scores and docs back to the
branch it ran on. On `master` it then packs the rebuilt tables + `site/index.html`
into the `latest-build` release (`src/sync_build.py pack`, replaced each run) and
triggers the Pages deploy, which publishes that page - a manual dispatch on a
test branch never touches either. Each run starts by restoring the last build
(`sync_build.py fetch`), since a few steps read last run's tables before
rebuilding them. A run takes ~15 min against the 45-min job limit.

Every pull step is `continue-on-error`, so a source being down never stops the
refresh - it keeps its committed data and catches up on the next run that gets
through (the match-level pulls fetch everything they're missing). The FotMob /
ESPN pulls cap their fetching (`--budget-min`) and each request is hard-capped at
45 s (`src/net.py`), with step timeouts as a backstop. Python output is
unbuffered, so a cancelled step still shows how far it got. Sofascore is the
exception: it only exposes current values, so a blocked run's values are stale,
not recoverable - set the `SOFASCORE_PROXY` secret (a residential proxy) to
get fresh values from CI.

`build_current_season_stats.py` rewrites only the current season's rows in
`fbref_player_season_stats.csv`: goals / assists / minutes / MP / cards / shots /
xG / npxG / xAG + the `understat__*` block from Understat; starts from
`understat_player_matches.csv` (`position != 'Sub'`); shots on target / fouls /
tackles / interceptions / blocks / clearances / touches / saves from FotMob's
per-match box scores (`fotmob_player_season.csv`). Each player's birth year,
nationality, and the ~200 unused deep FBref columns (progressive passing, SCA/GCA,
zonal touches, PSxG) are kept from the most recent FBref parse. Finished seasons
are never touched. Names are consolidated across sources - see `player_name_map.csv`.

`data/processed/`'s sources, ledgers and model scores are committed (`git pull`
gets you the latest locally; `py -3.11 src/sync_build.py fetch` adds the rebuilt
tables and site from the `latest-build` release) and are the only record of what
has been pulled - CI keeps no fetch cache. Every pull skips what is already committed and only requests
new matches: FotMob's per-match player rows / lineups / goal timelines live in
`data/processed/fotmob/<kind>_<season>.csv`, the live-match source snapshots in
`data/processed/live/`. `verify_player_matches.py` then cross-checks each FotMob
row against Understat, ESPN's box scores (`pull_espn_players.py`, committed under
`data/processed/espn/`) and football-data.co.uk (internationals: martj42) and
writes `verified` = agree / conflict / fotmob_only into those CSVs. `data/raw/`
stays gitignored scratch.

Understat has been down since late September 2026. Until it's back,
`understat_gapfill.py` rebuilds every league match it has no result for from
FotMob, in Understat's own shape and ids (`understat_gapfill_*.csv`, tagged
`source=fotmob`); everything that used to read `understat_*.csv` now reads it
through `understat_gapfill.read()`. The Understat files themselves are left
untouched, and gap rows drop out on their own once Understat publishes those
games. A season keeps being pulled until Understat's fixture list has no
unplayed games, so an outage that outlasts the season still backfills.

Setup: add repo secret `FOOTBALL_DATA_ORG_KEY`; optionally `SOFASCORE_PROXY`.

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
