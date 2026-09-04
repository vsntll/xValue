# Player / odds website (`src/export_site_data.py`, `site/`)

A single self-contained HTML page (`site/index.html`) with the season's Premier
League player stats, the value model's predictions, and next-fixture odds — no
server, no fetch, data is embedded inline.

## Data (`src/export_site_data.py` -> `site/data.json`)

- **Players**: current-season (2026-27) FBref stats, min 45 minutes played.
  npxG/90 and xA/90 come from Understat season totals / 90s played — FBref's own
  `*_Per` xG columns are empty for every EPL season in this dataset (the advanced
  FBref tables are gated), Understat is the real source here.
- **Last season** (2025-26) line shown alongside, where the player has one.
- **Projected 38-game pace**: current per-90 rate x projected minutes over a full
  season. A simple pace projection, not a trained model — labelled as such in the UI.
- **Predicted market value**: from `value_model_predictions.csv` (2025-26 season,
  the trained HGB regressor, R2(log) 0.82) vs the player's listed value.
- **Next-fixture odds**: a Dixon-Coles attack/defence model (`src/dixon_coles.py`,
  fit on all competitions through today, 900-day window) gives P(home/draw/away)
  and expected goals for each team's next scheduled Premier League match
  (`live_matches_2026-27.csv`, status SCHEDULED/TIMED).
- **Player props (anytime goal/assist)**: the fixture's team-expected-goals split
  across the matchday squad by each player's (non-penalty xG90 / xA90, shrunk
  toward last season's rate while the current season is small-sample) x season
  minutes-share, then `P(>=1) = 1 - exp(-lambda)`. A transparent heuristic, not a
  trained per-player model — there isn't one in this pipeline yet.

Watch for: `fbref_player_season_stats.csv` and `value_model_predictions.csv` are
**latin-1**, not utf-8 (accented names mojibake under the default encoding).

Run: `python src/export_site_data.py` (regenerates `site/data.json`), then splice
it into `site/template.html` to produce `site/index.html` (the `__SITE_DATA_JSON__`
placeholder).

## Known data-scope note

The 2026-27 Premier League club list in the underlying data (Coventry City, Hull
City, Ipswich Town in; West Ham, Wolves, Burnley out, vs. 2025-26) was
cross-checked for internal consistency against known 2024-25/2025-26
promotion/relegation and is plausible, not independently verified against a live
source.
