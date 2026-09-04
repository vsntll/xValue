# Player / odds website ("xValue"; `src/export_site_data.py`, `site/`)

A single self-contained HTML page (`site/index.html`) with the season's player
stats, the value model's predictions, and next-fixture odds across the Premier
League, La Liga and Bundesliga — no server, no fetch, data is embedded inline.
A league chip filter in the top bar (All / Premier League / La Liga /
Bundesliga) scopes both the player table and the fixture grid.

## Data (`src/export_site_data.py` -> `site/data.json`)

`LEAGUES` in the script maps FBref's `src_league` codes to display names
(`ENG1`/`ESP1`/`GER1` -> Premier League/La Liga/Bundesliga); add an entry there
to cover another top-5 league already in the underlying data.

- **Players**: current-season (2026-27) FBref stats, min 45 minutes played, for
  all three leagues. npxG/90 and xA/90 come from Understat season totals / 90s
  played — FBref's own `*_Per` xG columns are empty for every season/league in
  this dataset (the advanced FBref tables are gated), Understat is the real
  source here.
- **Last season** (2025-26) line shown alongside, where the player has one.
- **Projected 38-game pace**: current per-90 rate x projected minutes over a full
  season. A simple pace projection, not a trained model — labelled as such in the UI.
- **Predicted market value**: from `value_model_predictions.csv` (current-season
  rows, the trained stacked regressor, R2(log) 0.89 / 12% median error in the
  €100M+ band, capped at €220M) vs the player's listed value. `listed_is_estimate`
  is set when the listed value is a peer-median fill rather than a real feed value.
- **Next-fixture odds**: a *single* Dixon-Coles attack/defence model
  (`src/dixon_coles.py`, fit once on all competitions through today, 900-day
  window - not per-league) gives P(home/draw/away) and expected goals for each
  team's next scheduled league match (`live_matches_2026-27.csv`, status
  SCHEDULED/TIMED). Fitting across competitions (not per-league) is what lets
  Bundesliga/La Liga/Premier League teams share one attack/defence scale via
  their Champions/Europa League meetings. Fixture counts per league vary
  matchday to matchday (10 for a clean Premier League round, 9 for Bundesliga's
  18 teams, more than 10 for La Liga when its schedule is staggered/rescheduled
  that week) - not a bug, just how synced each league's "next round" is.
- **Player props (anytime goal/assist)**: the fixture's team-expected-goals split
  across the matchday squad by each player's (non-penalty xG90 / xA90, shrunk
  toward last season's rate while the current season is small-sample) x season
  minutes-share, then `P(>=1) = 1 - exp(-lambda)`. A transparent heuristic, not a
  trained per-player model — there isn't one in this pipeline yet.

Encoding: `fbref_player_season_stats.csv` and `value_model_predictions.csv` are
UTF-8 (they carry accented names correctly since the deaccent/slug work);
`live_matches_2026-27.csv` is still read latin-1 (ESPN team names mojibake under
UTF-8).

Run: `python src/export_site_data.py` — regenerates `site/data.json` **and**
splices it into `site/template.html` to produce `site/index.html` (the committed,
self-contained page; `__SITE_DATA_JSON__` placeholder).

## Weekly auto-refresh

`.github/workflows/weekly-refresh.yml` (Mondays 07:00 UTC + manual dispatch)
re-pulls every browser-free source, rebuilds `matches_all`, retrains both models,
regenerates `site/index.html` and commits it. It does **not** refresh the FBref
counting stats or the Transfermarkt scrape values (those need Chrome + a WAF
captcha) — run those locally now and then per `run_pipeline.md`.

Setup: repo secret `FOOTBALL_DATA_ORG_KEY`, plus a one-off `pipeline-seed`
GitHub Release holding `data/processed/` (the workflow's cache fallback -
`data/` is gitignored). Command in `run_pipeline.md`.

## Known data-scope note

The 2026-27 Premier League club list in the underlying data (Coventry City, Hull
City, Ipswich Town in; West Ham, Wolves, Burnley out, vs. 2025-26) was
cross-checked for internal consistency against known 2024-25/2025-26
promotion/relegation and is plausible, not independently verified against a live
source.
