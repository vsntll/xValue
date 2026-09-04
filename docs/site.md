# Player / odds website ("xValue"; `src/export_site_data.py`, `site/`)

A single self-contained HTML page (`site/index.html`) with the season's player
stats, the value model's predictions, next-fixture odds, and club pages, across
the Premier League, La Liga and Bundesliga — no server, no fetch, data is
embedded inline. A league chip filter in the top bar (All / Premier League /
La Liga / Bundesliga) scopes the player table and the fixture grid. Three tabs:
Players, Fixtures & Odds, Teams.

The **Teams** tab (a club picker, searchable, grouped by league) shows: current
league position, last 8 results in any competition (score, venue, competition,
shots/possession/xG where available), the full roster (reusing the Players
data - clicking a name jumps to their row on the Players tab), league position
by season back to 2020-21, and cup finals the club reached.

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
- **Predicted market value**: from `value_model_predictions.csv` (the trained
  regressor, R2(log) 0.89 / MAE EUR4.9M / within-2x 91%) vs the player's listed
  value. `value_model_predictions.csv` now covers every season 2020-21..2026-27
  (it used to stop at 2025-26); `load_value_predictions` prefers the *current*
  season's row per player+team and falls back to 2025-26 for anyone not yet
  covered there (below the model's minutes threshold so far this season) - so
  `value.as_of_season` on a player record varies, it's not always "2025-26"
  anymore. The file also gained a `value_imputed` flag (true when
  `market_value_eur` itself was filled in rather than sourced) that isn't
  currently surfaced in the UI.
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
- **Recent results** (`build_recent_matches`): last 8 matches per team, any
  competition, from both sides of `matches_all.csv` (each match produces one
  row for each team's perspective).
- **League standings** (`build_standings`): a full table per (league, season),
  computed directly from match results (3 pts/win) - not scraped from anywhere,
  so it's exactly consistent with `recent_matches`/the model's training data.
  The 2026-27 table is simply the live in-progress standing.
- **Cup finals** (`build_cup_finals`): for each domestic cup / league cup /
  European competition, the **last-dated match of that season+competition** is
  treated as the final - there's no round/stage column to identify it properly.
  Spot-checked against several known real finals (2021-22 UCL: Real Madrid 1-0
  Liverpool; 2023-24 Copa del Rey: Athletic Club 1-1 Mallorca, correctly
  flagged as penalties; etc.) and it held up, but it's an inference, not sourced
  from an official bracket - a season/competition with under 2 recorded matches
  is skipped rather than guessed, and **when the final finished level, no winner
  is shown** (penalty-shootout results aren't in this data). The in-progress
  2026-27 season is excluded entirely (its cups haven't reached a final).
  `matches_all.csv` also carries two source labels for the same European
  competitions in overlapping seasons ("Champions Lg" vs "Champions League",
  etc.) - `COMP_ALIASES` normalises them before grouping, otherwise a team's
  run could be split across two differently-labelled "competitions."

Watch for: `fbref_player_season_stats.csv`, `value_model_predictions.csv`,
`live_matches_2026-27.csv`, `matches_all.csv` and `squad_season_features.csv`
are read as **latin-1**, not utf-8 (accented names mojibake under the default
encoding - e.g. "Supercopa de España" in `matches_all.csv`'s `comp` column).
That alone isn't quite enough, though: some *individual rows* within these same
files are actually utf-8 that a different, earlier scraper run mis-decoded as
latin-1 before writing them back out - reading the whole file as latin-1 fixes
the majority but doubly-mangles those rows the other way (e.g. a name with "ć"
or "š" survives as two garbage characters instead of one). `_fix_names()` runs
`ftfy.fix_text()` over every name/team/competition column after loading -
it detects and repairs exactly that double-encoding mismatch and leaves
already-correct text untouched. This matters for more than display: an
unrepaired name normalizes (`normalize_team`) to the *wrong* `team_key`,
which would silently split one real team's matches/roster/standings row across
two different keys - so `build_teams_list` also recomputes `team_key` locally
from the fixed name rather than trusting `squad_season_features.csv`'s own
(possibly pre-fix) `team_key` column. Also: current-season (2026-27) `Age` is
unpopulated upstream for every row - `build_players_payload` falls back to last
season's age + 1; players with no 2025-26 record (new signings, debutants)
still show no age.

Run: `python src/export_site_data.py` (regenerates `site/data.json`), then splice
it into `site/template.html` to produce `site/index.html` (the `__SITE_DATA_JSON__`
placeholder).

## Known data-scope note

The 2026-27 Premier League club list in the underlying data (Coventry City, Hull
City, Ipswich Town in; West Ham, Wolves, Burnley out, vs. 2025-26) was
cross-checked for internal consistency against known 2024-25/2025-26
promotion/relegation and is plausible, not independently verified against a live
source.
