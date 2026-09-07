# Player / odds website ("xValue"; `src/export_site_data.py`, `site/`)

A single self-contained HTML page (`site/index.html`) with the season's player
stats, the value model's predictions, next-fixture odds, and club pages, across
the Premier League, La Liga, Bundesliga, Serie A and Ligue 1 — no server, no
fetch, data is embedded inline. The league chip filter in the top bar is built
from `DATA.leagues`, so it auto-expands as leagues are added; it scopes the
player table, fixture grid and leaderboard. Seven tabs: **Players**, **Fixtures
& Odds**, **Teams**, **Rankings** (team + player Elo), **Match Sim** (run any
matchup with goalscorers), **Streak** (call a real season one match at a time),
**How It Works**.

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

The **Rankings** tab has two from-scratch Elo systems, no market value in either
(`build_team_elo_rankings`, `build_player_elo_leaderboard` over
`build_player_elo.py`'s output):

- **Team power rankings** — goals-based Elo across *all* competitions, the same
  rating the outcome model uses. "Overall" carries across seasons (a quarter of
  each team's gap from 1500 reverts each summer); a per-season ladder resets
  harder (starts at 1500 + half the previous final's gap, promoted sides at
  1400) and counts only that season's matches.
- **Player Elo leaderboard** — an opponent-adjusted rating built purely from
  real match output (non-penalty xG + 0.7 × xA per appearance) vs. an
  expectation that folds in the opponent's strength and the player's own current
  rating, over the last three seasons. Half the gap from 1500 reverts between
  seasons; a player who stops featuring bleeds toward 1500 for every match his
  club plays without him, after a two-game grace. Overall (≥ 5 appearances in
  the window) plus per-position and per-season views.

The **Match Sim** tab (`build_games_data` → `sim`) runs any matchup between two
current-squad clubs: draws each side's goals from a Poisson on the Dixon-Coles
expected goals, then hands each goal to a player picked by his npxG-per-90 ×
minutes share. One random game per run — or simulate 1,000 for the score spread.

The **Streak** tab (`build_games_data` → `streak`) is a "predict the season"
game: every league match 2020-21 → 2026-27 in date order with its actual score,
and the leak-free Dixon-Coles pre-match W/D/L probabilities revealed after each
call. One wrong call ends the run.

The **How It Works** tab is the point of transparency: a card per kind of
conclusion (match odds, player props, value model, pace projection,
standings/projected table, cup finals; the Elo systems and the two games are
explained in the footer notes), each with a plain-language explanation and - for
the model-driven ones - the *actual arithmetic*, run on one real upcoming
fixture and one real player (`build_methodology_example`): the Dixon-Coles
attack/defence ratings and resulting expected goals for that fixture, the
anytime-goalscorer share-of-xG breakdown for its top-attack player, and the
value model's inputs/output for the single most expensive player in the dataset
(picked for recognisability, not cherry-picked results).

## Data (`src/export_site_data.py` -> `site/data.json`)

`LEAGUES` in the script maps FBref's `src_league` codes to display names
(`ENG1`/`ESP1`/`GER1`/`ITA1`/`FRA1` -> Premier League / La Liga / Bundesliga /
Serie A / Ligue 1); add an entry there to cover another league already in the
underlying data.

- **Players**: current-season (2026-27) stats, min 45 minutes played, for
  all five leagues. npxG/90 and xA/90 come from Understat season totals / 90s
  played — FBref's own `*_Per` xG columns are empty for every season/league in
  this dataset (the advanced FBref tables are gated), Understat is the real
  source here.
- **Last season** (2025-26) line shown alongside, where the player has one.
- **Projected 38-game pace**: current per-90 rate x projected minutes over a full
  season. A simple pace projection, not a trained model — labelled as such in the UI.
- **Predicted market value**: from `value_model_predictions.csv` (the trained
  regressor, R2(log) 0.89 / MAE EUR4.1M / within-2x 91%) vs the player's listed
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
  example, and the highest-listed-value player across all five leagues for
  the value-model worked example.

Watch for: `fbref_player_season_stats.csv`, `value_model_predictions.csv`,
`live_matches_2026-27.csv`, `matches_all.csv` and `squad_season_features.csv`
are **utf-8** (confirmed 2026-09-04: `parse_fbref_player_stats.py`'s rewrite
made this correct at the source - plain utf-8 decodes every one of these files
without error, and disagrees with a latin-1 read on real names like "Håvard
Nordtveit" and "Šime Vrsaljko", which only the utf-8 read gets right). Read as
latin-1 instead - as this script did until 2026-09-04 - and most names still
look fine (`ftfy.fix_text()` repairs the common single-encoding mismatch), but
a handful don't: ftfy's heuristics didn't catch every case, so a small number
of names stayed silently wrong. `_fix_names()` still runs `ftfy.fix_text()`
after the (now correct) utf-8 read, as a defense-in-depth no-op safety net -
verified to change 0 rows across every processed CSV once the encoding is
right, so it costs nothing and catches a future regression. This matters for
more than display: a mis-decoded name normalizes (`normalize_team`) to the
*wrong* `team_key`, which would silently split one real team's
matches/roster/standings row across two different keys - so `build_teams_list`
also recomputes `team_key` locally from the loaded name rather than trusting
`squad_season_features.csv`'s own (baked-in-earlier) `team_key` column. Also:
current-season (2026-27) `Age` is unpopulated upstream for every row -
`build_players_payload` falls back to last season's age + 1; players with no
2025-26 record (new signings, debutants) still show no age.

Run: `python src/export_site_data.py` — regenerates `site/data.json` **and**
splices it into `site/template.html` to produce `site/index.html` (the committed,
self-contained page; `__SITE_DATA_JSON__` placeholder), in one run
(`splice_index_html()`, called at the end of `main()`). This is what
`weekly-refresh.yml`'s "Regenerate the site" step relies on - before
2026-09-04 the script only wrote `data.json`, so that CI step silently did
nothing to `site/index.html` and the commit-and-deploy step downstream always
saw "site unchanged."

## Auto-refresh (every 2 days)

`.github/workflows/weekly-refresh.yml` — "Site refresh (every 2 days)", cron
`0 7 */2 * *` + manual dispatch — re-pulls every browser-free source,
**refreshes the current season's player stats from Understat + FotMob**
(`build_current_season_stats.py`: goals / assists / minutes / xG / SoT / fouls /
tackles, no browser), rebuilds `matches_all`, retrains both models, rebuilds the
player Elo, regenerates `site/index.html`, and commits the refreshed
`data/processed/` **and** `site/index.html` back to `master`.

It does **not** re-scrape the deep FBref columns (progressive passing, SCA/GCA,
zonal touches, PSxG — nothing the models or site currently read) or the
Transfermarkt values; those need Chrome + a WAF captcha, so run them locally
roughly monthly per `run_pipeline.md`.

Setup: just the repo secret `FOOTBALL_DATA_ORG_KEY`. `data/processed/` is
committed, so there's no seed release to maintain; `data/raw/` and the fetch
caches stay gitignored (CI keeps them in a throwaway `actions/cache` blob).

## Known data-scope note

The 2026-27 promoted-club lists across all five leagues (e.g. Coventry / Hull /
Ipswich into the Premier League) were cross-checked for internal consistency
against ESPN + football-data.org + Understat (all three agree on the 20/20/18/20/18
club counts) and are plausible, not independently verified against an official
source. Serie A / Ligue 1 player rows are Understat-derived (see
`docs/fbref_ingestion.md`), so a player who has not yet appeared in a match has
no row until he does.
