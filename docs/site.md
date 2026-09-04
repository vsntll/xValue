# Player / odds website ("xValue"; `src/export_site_data.py`, `site/`)

A single self-contained HTML page (`site/index.html`) with the season's player
stats, the value model's predictions, next-fixture odds, and club pages, across
the Premier League, La Liga and Bundesliga — no server, no fetch, data is
embedded inline. A league chip filter in the top bar (All / Premier League /
La Liga / Bundesliga) scopes the player table, fixture grid and leaderboard.
Four tabs: Players, Fixtures & Odds, Teams, How It Works.

The **Players** tab opens with a "Biggest bargains" / "Most overpriced"
leaderboard (`build_value_leaderboard`) above the search table - the value
model's predicted-vs-listed gap, both directions, gated to players with 180+
minutes and a EUR1.5M+ listed value so it isn't dominated by noisy fringe
players with nominal Transfermarkt listings.

The **Fixtures** tab's team names now carry a generated crest (a colour hashed
from the club name, initials on top - no external assets) and a 5-match form
strip, both client-side (`crest()`, `formPills()` in the template).

The **Teams** tab (a club picker, searchable, grouped by league) shows: current
league position with crest + form strip in the header, last 8 results in any
competition (score, venue, competition, shots/possession/xG where available),
a **projected final table** (below), the full roster (reusing the Players data
- clicking a name jumps to their row on the Players tab), league position by
season back to 2020-21, and cup finals the club reached.

The **projected final table** (`build_projected_table`) takes each team's
current 2026-27 points and adds *expected* points from every remaining
fixture that season - `3*P(win) + 1*P(draw)` per game, via the same
Dixon-Coles model as the match odds (`build_full_schedule` pulls every
remaining SCHEDULED/TIMED fixture per team, not just the next one). It's an
expected-value sum, not a season simulation - no single simulated result ever
gets played out - and obviously can't see injuries, transfers, or a manager
getting sacked in November.

The **How It Works** tab is the point of transparency: six cards, one per
"kind of conclusion" (match odds, player props, value model, pace projection,
standings/projected table, cup finals), each with a plain-language explanation
and - for the three model-driven ones - the *actual arithmetic*, run on one
real upcoming fixture and one real player (`build_methodology_example`): the
Dixon-Coles attack/defence ratings and resulting expected goals for that
fixture, the anytime-goalscorer share-of-xG breakdown for its top-attack
player, and the value model's inputs/output for the single most expensive
player in the dataset (picked for recognisability, not cherry-picked results).

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
  covered there (below the model's minutes threshold so far this season), so
  `value.as_of_season` on a player record varies by player. `listed_is_estimate`
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
- **Value leaderboard** (`build_value_leaderboard`): top 8 biggest predicted-vs-listed
  gaps each direction, min 180 minutes + EUR1.5M listed value.
- **Projected table** (`build_full_schedule` + `build_projected_table`): see above.
- **Methodology example** (`build_methodology_example`): the earliest upcoming
  fixture where both teams are in the fitted Dixon-Coles model's team index
  (falls back gracefully - `methodology_example: null` - if somehow none
  qualify), its top-attacking-weight home player for the prop-odds worked
  example, and the highest-listed-value player across all three leagues for
  the value-model worked example.

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
