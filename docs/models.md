# Models (pipeline steps 8-12)

<!-- metrics:models-summary -->
**Current results** (temporal holdout 2024-25 + 2025-26, all **big-5** leagues; generated from `data/processed/model_metrics.json`):

| | value model | outcome (pure) | outcome (hybrid) |
| --- | --- | --- | --- |
| metric | R²(log) **0.89**, MAE **€4.2M**, within-2x **91%**, medAPE 22% | log-loss **0.977**, acc **0.535** | log-loss **0.974**, acc **0.533** |
| v1 was | 0.70 / €9.1M / 68% | 1.014 / 0.511 | — |
| reference | — | Bet365 closing 0.971 / 0.539 | (uses market opening odds as a feature) |
<!-- /metrics:models-summary -->

Trajectory: value R²(log) 0.70 → 0.82 (prev-value) → 0.835 (contract/minutes) →
0.87 (prev-season club value + xG-share) → 0.88 (name-resolution, coverage
81 → 95%) → 0.90 (`value_history.csv`: big-5 mirror back to 2015 + cross-league)
→ 0.907 (defensive-volume + team-success features, Serie A/Ligue 1 value backfill)
→ **0.89** (Serie A + Ligue 1 folded in as full leagues 2026-09-06: +7.7k
player-seasons, most without the team-success columns and cold-start-heavy, so
the holdout got bigger and harder even as MAE fell to €4.2M). Outcome
1.014 → 0.997 → 0.990 → 0.988 → 0.981 → 0.980 (pure - HGB's blend weight
was 0 until this was tuned; now genuinely contributes); hybrid went from closing
~90% of the base-rate → Bet365-closing gap to ~97%. Current scores: the table
above, regenerated after every retrain.

<!-- metrics:models-value-split -->
The value model splits cleanly by whether a prior-season value exists:
**R²(log) 0.90** for the ~93% that have one, **0.64** for cold-start arrivals with
none (promoted-club squads, Eredivisie/Primeira/Championship signings, academy
graduates, and Serie A / Ligue 1 players whose only prior value predates the
2015 mirror). Keepers score **0.87**.
<!-- /metrics:models-value-split -->
Closing the cold-start gap - and pushing
past 0.90 - needs data we don't have: lower-league value history, transfer fees,
or salaries.

**The ceiling and the high end**: Transfermarkt caps listings near **€220M**, so
predictions are clipped there in log space and the elite gravitate toward it.
<!-- metrics:models-elite -->
`medAPE` for the **€100M+ band is 18%** (was ~30-40% before the cap - the trees squashed
the tail toward the mean), though the ~38-row band sits slightly *under* the
listed values (median pred/listed 0.84). Mid-range (€3-40M) predictions carry a
**median pred/listed of 0.99** but individual scatter stays ~21% —
<!-- /metrics:models-elite -->
that's roughly how much a crowd-sourced value moves between updates, and the
season stats don't explain the revisions, so "every mid-range player within 10%"
is not reachable from this data.

**Coverage guarantee**: every current-season player with minutes gets a
`market_value_eur` - a real feed value, else the player's most recent value
carried forward across a transfer, else a position × league × age peer-median
(`parse_fbref_player_stats.py`), and finally, for the just-promoted squads the
transfermarkt / sofascore feeds don't reach (Coventry, Hull, Elversberg,
Deportivo, ...), a **division-adjusted** baseline in
`build_current_season_stats.py`: the top-flight positional median scaled by
`PROMO_FACTOR` (~0.38 for England down to the Championship, ~0.42 for Serie A /
Ligue 1 - the narrower gap to their 2nd tiers). All impute paths are flagged
`market_value_imputed = 1` and excluded from both model fitting and the
prev-value history; the model's own `predicted_eur` is still computed for these
players, so `ratio` = model vs. the promoted-club baseline is a real over/under
signal. `value_model_predictions.csv` covers every season including the current
one, so the site has a value and a prediction for every current player.

"Hybrid" (`train_outcome_model.py --hybrid`) blends market-consensus **opening**
odds into the stack - it beats the opening line and lands ~level with the
bookmaker's own **closing**-odds performance. The pure model uses only data.


## Step 9 - value regression  (`src/train_value_model.py`)

Predicts a player's market value from his season + his value history.

- **Data**: `fbref_player_season_stats.csv`, players (outfield **and keepers**)
  with a real value and >= 8 full-90s, across all five leagues. ~6,700 train /
  ~3,000 test. ~90% of season-rows with minutes carry a *real* value (the gap is
  2022-23's thin TM mirror and Serie A / Ligue 1 rows the scrape missed); the
  rest get a flagged peer-median or, for just-promoted squads, a
  division-adjusted baseline - never used to fit.
- **Prev-value lags** (`build_value_history.py` → `value_history.csv`): the model
  is dominated by last season's value, so coverage of that lag caps accuracy.
  The history table unions the worldfootballR big-5 mirror **back to 2015-16 and
  including Ligue 1 / Serie A** (so 2020-21 rows and cross-league movers get a
  lag), the browser scrape (2023-26), Sofascore (2026-27), and every fuzzy-filled
  label. Features: `prev_log_value` (season-1, else the most recent known value
  carried forward), `prev_staleness`, `prev1/prev2_log_value`, `value_momentum`,
  `prev × age/youth` interactions, `has_prev` / `has_any_prev`.
- **Other features** (~33 total): age / age² / distance-from-26; minutes, starts,
  minutes-trend; per-90 goals / assists / npg / shots / SoT / xG / npxG / xA /
  key passes / xG-chain / fouls; prev-season squad value; xG-share; contract
  years remaining; one-hot position + league.
- **Target**: `log1p(market_value_eur)`. **Split**: train 2020-24, test 2024-26.
  Fit + eval only on real values (peer-median imputations excluded); age falls
  back to (season year - birth year) when FBref leaves it blank early in a season.
- **Model**: two ridge stacks (each over two HGB fits + ExtraTrees + ridge) -
  one predicts `log1p(value)` directly, the other predicts the *change* from the
  player's last known value (a small, bounded quantity; the anchor passes through
  at slope 1 so a €200M prior isn't shrunk toward the mean). The two are
  de-shrunk on OOF and blended `0.5 / 0.5`, then clipped to €220M in log space.
  Rows with < 8 full-90s this season (all of the current season, early on) skip
  the models and predict the last known value along a light age curve.
- **Cold-start rows** (`has_any_prev == 0`, ~10% of fit rows - debutants,
  promotions, cross-league movers) get their own direct-only stack: no
  residual-from-anchor view (there's no real prior value to residual
  against), and a feature set with every prior-value-derived column dropped
  (`prev_log_value`, `value_momentum`, ...). They also get three cheap
  derived flags (`is_promoted`, `is_academy_age`, `is_brand_new`) and, where
  scraped, `last_transfer_fee_log` / `fee_is_loan` / `fee_is_free` from a
  targeted Transfermarkt profile backfill (`pull_transfermarkt_scrape.py`) -
  a fee negotiated for the move is a strong value proxy even at zero minutes.
<!-- metrics:models-value-result -->
- **Result**: R2(log) **0.89**, MAE **€4.2M**, medAPE **22%**, within-2x
  **91%** (0.90 with a prior value, 0.64 cold-start on its own stack -
  medAPE 42% - 0.87 keepers). €100M+ band medAPE **18%**, median pred/listed **0.84**.
- **Thin-minutes players** (< 8 full 90s - priced by carrying last season's
  value forward one year along the age curve, not by the trees; 40% of all
  published predictions, never in the test set): R2(log) **0.69**, medAPE **28%**, within-2x 77%
  against their real listed values.
<!-- /metrics:models-value-result -->
- Output: `value_model_predictions.csv` (**every player, every season incl. the
  current one**, with `value_imputed` flag), `models/value_model.pkl`
  (`{direct, resid, cold, cal, cold_cal, features, features_cold}`).

## Step 8 - squad rollup  (`src/build_squad_features.py`)

`data/processed/squad_season_features.csv` - one row per (team, season):
`squad_value_eur`, `core18_value_eur`, `xi_proxy_value_eur` (top-11 by
minutes - a season-level proxy, not any specific match's real lineup),
`value_known_frac`, minutes-weighted `mean_age_wtd`, `squad_xg`, `squad_xa`.
The real per-match XI rollup (confirmed starters, from FotMob's committed
lineup rows in `data/processed/fotmob/lineups_<season>.csv` - `src/build_match_lineups.py` -> `match_lineup_features.csv`)
feeds `build_match_model_table.py`'s `xi_value_ratio` directly; see Step 11.

## Step 11 - outcome classifier  (`src/train_outcome_model.py` + `build_match_model_table.py`)

Pre-match home-win / draw / away-win on `matches_all.csv` league rows.

- **Feature table** `build_match_model_table.py` (56 cols): **goals-Elo and
  xG-Elo** (updated across all competitions, seeded with 2014-20 warmup results
  so ratings have converged); home/away-**split** rolling-8 form (pts / goals /
  xG for & against); Elo-implied home prob; squad-value ratio; promoted flags;
  days rest; head-to-head; **squad momentum** (`h/a_momentum` — each side's
  recent player output vs. the value model's peer baseline, from
  `build_form_momentum.py` → `squad_momentum.csv`, folded in on a second
  `build_match_model_table.py` pass, restricted to the confirmed starting XI
  where a lineup is known rather than every player who featured). The
  season-scoped Elo twins (`elo_*_s`) are emitted for the site Rankings view
  only; the model uses the continuous columns.
- **`xi_value_ratio`** refines the squad-value ratio with the confirmed
  starting XI's value (`build_match_lineups.py`, off the lineup block
  `pull_fotmob_players.py` already commits for free) in place of last season's
  whole-squad value, for any match where that lineup is known. Falls back to
  `value_log_ratio` otherwise (flag `xi_value_known`) - same pattern as
  `market_value_imputed`. Since confirmed lineups exist only for matches
  already played or ~1hr pre-kickoff, this is populated for backtesting and
  near-kickoff scoring, essentially never for the model's normal day(s)-ahead
  use case - it's used in `train_outcome_model.py`'s `FEATURES` in place of
  `value_log_ratio` precisely because it degrades to that column when unknown.
- **Models**:
  - logreg and HGB on the features, each with its own small hyperparameter
    search (logreg's `C`, HGB's depth/learning-rate/l2/min-leaf) picked on
    the validation season only - both were hardcoded guesses before
  - **Poisson-Skellam**: two Poisson GLMs (home goals, away goals) -> full
    scoreline distribution with a Dixon-Coles low-score correction (rho tuned on
    a validation season); recency-weighted (2-season half-life). **Best single
    model.**
  - a 3-way geometric blend of {poisson, logreg, hgb}, weights grid-searched
    on the validation season (2 free weights on the simplex) - HGB used to be
    fit and reported but never actually blended in; its errors are only
    partly correlated with the other two, same reasoning train_value_model.py's
    multi-learner stacks already use
- **Split**: train <= 2022-23, val 2023-24, test 2024-26 (~3,470).
<!-- metrics:models-outcome-table -->
- **Result** (log-loss / accuracy, test 2024-25 + 2025-26, n=3,476):

  | model | acc | log-loss |
  | --- | --- | --- |
  | base rate | 0.429 | 1.075 |
  | logreg | 0.527 | 0.983 |
  | hgb | 0.523 | 0.985 |
  | poisson-Skellam | 0.528 | 0.982 |
  | poisson + xG | 0.529 | 0.981 |
  | blend {poisson, logreg, hgb} (**pure**) | 0.535 | 0.977 |
  | **hybrid (+ market opening odds)** | 0.533 | 0.974 |
  | Bet365 closing | 0.539 | 0.971 |

  The pure model closes ~94% of the base-rate → bookmaker gap on log-loss; the
  hybrid, which adds the market's *opening* line as a feature, closes ~97% of
  it - the last sliver is information the market has and we don't
  (confirmed lineups, injuries, sharp money).
<!-- /metrics:models-outcome-table -->
- Output: `data/processed/outcome_model_predictions.csv` (pure),
  `outcome_model_predictions_hybrid.csv` (+ opening odds),
  `outcome_model_predictions_all.csv` (every split, incl. the live 2026-27 rows).

## Step 12 - player Elo  (`src/build_player_elo.py`)

A self-contained, match-by-match performance rating for individual players -
no value-model inputs anywhere in it, and not a feature of the outcome model
(it feeds the site's Rankings tab only). Same mechanic as the team Elo: start
at 1500, move by `K * (actual - expected)` (`K=20`, per-match delta clipped at
±40). Windowed to 2024-25 → 2026-27. Ratings carry across seasons with no
summer reset; the only pull back toward 1500 is inactivity - a player who
stops featuring decays 3% toward 1500 per club match missed after a
two-match grace.

- **Actual** is a role-weighted blend of three per-match components, not
  attacking output alone:
  - `atk` - xG + 0.7 × xA (Understat, `understat_player_matches.csv` read
    through `understat_gapfill.read()`, so matches Understat hasn't published
    while it's down come from FotMob's xG / xA instead; xG includes penalties)
  - `def` - tackles + interceptions + blocks + clearances for outfield
    players; saves − goals conceded for keepers (FotMob's committed per-match
    rows, `data/processed/fotmob/player_matches_<season>.csv`)
  - `pass` - passes completed minus what a position-average passer would
    complete on the same attempts (FotMob; matches with < 5 attempts are
    treated as missing)

  Each component is z-scored against its own position-group baseline/90 and
  residual std before blending with `ROLE_WEIGHTS`:

  | pos | atk | def | pass |
  | --- | --- | --- | --- |
  | FW | 0.70 | 0.15 | 0.15 |
  | MF | 0.45 | 0.30 | 0.25 |
  | DF | 0.20 | 0.50 | 0.30 |
  | GK | 0.05 | 0.55 | 0.40 |

  A component missing for a match (no FotMob row, or pre-backfill pass data)
  drops out and the remaining weights renormalize - never counted as zero.
- **Expected**, in the same z units, per 90 minutes played:
  `baseline + (rating − 1500) / SPREAD + OPP_BETA × (opponent Elo − average opponent)`.
  `SPREAD = 500`, so a player settles at 1500 + 500 × his average blended z
  per 90 (top 1% ≈ 1,900). `OPP_BETA` is fitted each run - the minutes-weighted
  slope of output on opponent Elo, per position (≈ −0.09 z/90 per 100 Elo for
  forwards, −0.02 for defenders) and separately for international rows.
  Opponent Elo is the club goals-Elo from `match_model_table.csv`, reused so
  the two Elo systems agree. (An earlier version multiplied the z-score
  baseline - ≈ 0 - by opponent and rating factors, so neither had any effect
  and strong players' ratings climbed without limit.)
- **Identity**: Understat's numeric `player_id` (two different "Alvaro
  Fernandez"es share a normalized name). The FotMob join has no shared id, so
  it matches on (normalized name, team, date) - scoped to one real match.
  Position comes from `fbref_player_season_stats.csv` (Understat says "Sub"
  for anyone off the bench).
- **Pass data** needs `pull_fotmob_players.py --backfill-passes <seasons>` for
  matches committed before `passes_completed`/`passes_attempted` were captured;
  the script prints its FotMob join + pass coverage each run.
- **International appearances** move the same rating. Box scores come from
  FotMob (`pull_fotmob_internationals.py` → `fotmob_intl_player_matches.csv`:
  World Cup, continental finals, every confederation's qualifiers, Nations
  Leagues, friendlies - FotMob only exposes friendlies from 2026) with the same
  three components. A player is linked by his FotMob player id, paired with
  his Understat id through his own matched club rows (one-to-one pairs only) -
  never through a club or a name - so international rows carry the national
  team, not a club. Opponent strength is a national-team Elo
  (`build_national_elo.py`, World Football Elo weights over every result since
  1872 from `pull_international_results.py`), recentred so the average
  international opponent sits at 1500. Position baselines stay club-only, so
  internationals are measured against the same bar. Friendlies count at half
  K (`INTL_K_MULT`). Inactivity decay stays club-only: an international
  appearance never counts as a played or missed club match, and a call-up
  mid-absence doesn't reset the two-match grace. Tournaments in June/July
  belong to the season just ended (seasons split on 1 August).
- Output: `player_elo.csv` (one row per player-match, rating before/after;
  `is_international` / `competition` mark international rows). The site
  always lists a player at the club of his last club appearance.

## Value screen + bargain validation  (`src/export_site_data.py`)

The homepage tile only ever showed the top-8 predicted-vs-listed value gaps
each way. The **Value Screen** tab exposes the full qualifying pool (same
guards as the tile: 180+ minutes, listed >= EUR1.5M, not an imputed listing)
with client-side league/position/direction filters and sort, plus a
**validation** table: the model's biggest "bargain" calls from
`BARGAIN_VALIDATION_SEASON` (2025-26), checked against what actually happened
to the listing (and, where scraped, a real transfer fee via
`tm_transfer_fees.csv`) by 2026-27 - most of these calls are small-value
players (near the EUR1.5M floor) whose listings mostly didn't move toward the
model's prediction, which is a useful, honest finding in itself: the value
gap is least trustworthy exactly where it's largest in relative terms.

## Aging curves + value forecast  (`src/build_aging_curves.py`)

Population curve: `log(value) ~ age + age^2`, fit separately per position
group (GK/DF/MF/FW) with season fixed effects (to net out cross-season market
inflation, then discarded - only the age SHAPE is kept), on real-valued
player-seasons with 3+ real seasons of history. Peak ages land at 23-25
across positions (steeper decline after than the old flat-22-29 guess), and
this curve now drives `train_value_model.py`'s `_age_adj` (falls back to the
old hand-tuned guess if `aging_curve.csv` doesn't exist yet - a fresh clone,
before this script has run once).

Player forecast (Marcel/delta-method style): a player's own deviation from
the curve, recency-weighted (2-season half-life) and Bayesian-shrunk toward
the position's population level (not toward 0 - shrinking toward a value of
~EUR1 was the first, wrong version of this) by how many real-valued seasons
he has. `SHRINK_K=0.5` - small, since transfer-market value is far stickier
year to year than the batting-average-style rate stats Marcel projections
were designed for. A hard multiplicative cap (`RATIO_CAP_1Y`/`_2Y`, e.g.
[0.2x, 3x] at 1 year) guards against the shrinkage's own heavy-tail
sensitivity: on a market this skewed, even a small shrinkage *weight* toward
the position mean is a large swing for a cheap fringe player whose own level
sits many log-units below it.

Output: `value_forecast.csv` (`player_key, player, squad, pos, season, age,
n_real_seasons, forecast_1y_eur, forecast_2y_eur`), folded into the player
detail panel on the site (not a separate tab).

**Delta method** (`src/build_aging_curves_delta.py` -> `aging_curve_delta.csv`).
The cross-sectional curve compares *different* players' value levels at each
age, so it is biased by who is in the sample. The delta method pairs each
player with himself a season later (3.3k players, 9.6k consecutive-season
pairs), de-means the log-value change per season pair (market inflation), fits
`d log(value)/d age = a + b·age` per position and integrates it:

| pos | peak (delta) | peak (cross-sectional) | peak → 33 (delta) | peak → 33 (cross) |
| --- | --- | --- | --- | --- |
| DF | 26.4 | 23.4 | −72% | −78% |
| FW | 25.5 | 24.4 | −76% | −70% |
| GK | 28.5 | 25.3 | −35% | −64% |
| MF | 25.7 | 23.5 | −78% | −77% |

The delta curves peak **1-3 years later** and keepers decline far more gently.
The bias runs the opposite way to the usual survivorship story: young players
in a big-5 league are a pre-selected elite (only the best prospects make it),
which inflates the young end of the cross-sectional curve and pulls its peak
early. Out of sample (both fit up to 2023-24, predicting each player's
2024-25 → 2025-26 inflation-adjusted log-value change, n=1,580) the delta
method wins in every position: MSE **0.310** vs 0.336 cross-sectional vs 0.385
with no age effect.

**Carried path fix.** The value model prices thin-minutes players (< 8 full
90s - ~40% of published predictions, none in the test set) by carrying last
season's value forward along the age curve. It used to add the curve's
*level* (0 at the peak, ≈ −0.7 at 33) to last season's value - carrying a
33-year-old forward at about half his value and a teenager below his - instead
of one year's *change*. Fixed to `curve(age) − curve(age − 1)`; scored against
the real listed values of those 5.2k rows: R²(log) **0.53 → 0.69**, medAPE
**37% → 28%**, within-2x 69% → 77% (cross-sectional increments; the delta
curve's increments run high here - 1.21 median pred/listed - because a player
not getting minutes loses value faster than the inflation-adjusted average).

## In-play win probability  (`src/live_win_prob.py`)

Not a new model - `live_win_prob(lh, la, minute, cur_h, cur_a, home_reds=,
away_reds=)` rescales the SAME pre-match Poisson lambdas
(`train_outcome_model.fit_lambdas`, extracted from `_fit_grid` for reuse
here) to the goals still expected in the time remaining (linear in minutes
left), re-runs the same Dixon-Coles-corrected grid on that REMAINING-goals
distribution, and reads the 1X2 probabilities off `(i+cur_h) vs (j+cur_a)`
instead of `i vs j`. A red card multiplies the disadvantaged team's remaining
lambda by a fixed `RED_CARD_FACTOR=0.73` (literature range ~0.70-0.75 for a
team down to 10 men), applied per card.

Data: the goal/red-card timeline comes from the same FotMob `matchDetails`
payload `pull_fotmob_players.py` already polls (`content.matchFacts.events`)
- committed for every new match at zero extra request cost
(`data/processed/fotmob/events_<season>.csv`, alongside the lineups); backfilling
it for matches pulled before events were captured needs
`pull_fotmob_players.py --events <season>` (a deliberate, bounded re-fetch).

**Backtest** (`src/backtest_live_win_prob.py`): replays every 2025-26 match
with an event timeline minute-by-minute against the REAL final
outcome (1,620 matches, 1,718/1,752 resolved to a match_model_table.csv row
- a full-season backfill via `pull_fotmob_players.py --events 2025-26`),
Brier score by minute checkpoint, against a static (never-updated, minute-0)
baseline:

| minute | live Brier | static Brier |
| --- | --- | --- |
| 0 | 0.585 | 0.585 |
| 30 | 0.532 | 0.585 |
| 60 | 0.425 | 0.585 |
| 90 | 0.000 | 0.585 |

Live re-conditioning beats the static baseline at every checkpoint past
kickoff, monotonically improving as the match progresses (minute 90 is
trivially perfect - no time left means the current score IS the final
score). On the red-card subset (292 matches), checkpoints in the first
~40 minutes after the card show the 0.73 multiplier clearly outperforming
ignoring the card entirely (e.g. 0.46 vs 0.52 Brier at the 30-minute mark,
0.40 vs 0.44 at 40); the gap narrows through 50-70 minutes and mildly
reverses by 80, where little remaining time makes the multiplier matter
less either way and the shrinking red-card subsample (12 matches at the
10-minute checkpoint, growing to 292 by full time) gets noisier.

**Not built**: a live poller / in-play scoring service - this is deliberately
scoped as a standalone, backtested function first, per the caveat that
confirmed lineups (and by extension live match state) aren't available at
the model's normal day(s)-ahead prediction time anyway.

## Betting-strategy backtester  (`src/backtest_betting_strategy.py`)

Pure downstream analysis of data that already exists: the test-set (2024-26,
genuinely out-of-sample) 1X2 probabilities against the market's OPENING
consensus (`AvgH/D/A` - the odds actually bettable at prediction time; the
closing line would be leakage), devigged the same way `_book()` already does
for the closing line. `edge = model_p - devigged_market_p`; flat-stake (bet a
unit past an edge threshold) and fractional-Kelly (quarter-Kelly, capped at
5%/bet - raw Kelly on a noisy edge estimate is a bankroll-ruin machine)
staking, both reported as ROI per unit staked.

**Result, pooled**: flat and Kelly are both roughly break-even-to-negative
across every edge threshold, for both the pure and hybrid model - no free
lunch against an efficient market, as expected.

**Sliced by odds bucket, the story flips**: edge concentrates in
**favourites**, not longshots - the opposite of the hypothesis that
motivated this backtest. Pure model, edge > 2%, flat stake:

| odds bucket | n bets | win rate | ROI |
| --- | --- | --- | --- |
| 1.0-1.5 (fav) | 188 | 77% | **+3.0%** |
| 1.5-2.0 | 419 | 58% | **+2.0%** |
| 2.0-3.0 | 724 | 41% | -0.1% |
| 3.0-5.0 | 1,603 | 26% | -2.9% |
| 5.0+ (longshot) | 502 | 13% | **-18.6%** |

This is the textbook favourite-longshot bias (bettors systematically
overvalue longshots, the market prices that in, so longshots are
structurally worse bets) - consistent across both the pure and hybrid model,
and the opposite of "residual edge sits in the longshots." The exact
bucket-by-bucket numbers move with each retrain (the favourite/mid-range
buckets have swung between roughly +2% and +12% across reruns so far, always
positive; the longshot bucket has stayed clearly, and increasingly,
negative) - take any single bucket's number with real caution, n=188-1,603;
the favourite-side-vs-longshot-side DIRECTION is the robust part.

**Sliced by league**: no clean signal - Bundesliga and Premier League
positive, La Liga/Ligue 1/Serie A negative, for the pure model; a different
mix for the hybrid model. Likely noise at ~500-900 bets/league rather than a
structural effect. Also: **the "low-liquidity leagues" hypothesis isn't
really testable with this data** - all 5 tracked leagues are big-5,
high-liquidity top-flight competitions; there's no genuinely thin market
(Eredivisie, Championship, ...) in the dataset to compare against.

Output: `data/processed/betting_backtest.csv` (bet-level detail, pure model).

## Lineup-absence backtest  (`src/backtest_lineup_absence.py`)

Does a team missing its best players from the confirmed XI underperform the
outcome model (which never sees lineups)? Residual = actual points minus the
model's expected points (3 × P(win) + P(draw)), on the out-of-sample 2024-26
test set, ~6,700 team-matches joined to a FotMob confirmed XI. A team's "top
players" for a match are its highest-valued players who were in its matchday
squad in the previous 45 days (so a January signing isn't "absent" before he
arrived).

| | n | pts − expected |
| --- | --- | --- |
| top player starts | 5,085 | +0.002 |
| top player **not** in the XI | 1,611 | −0.019 |
| none of the top-3 (by value) missing | 2,578 | +0.045 |
| < half of the top-3's value missing | 2,859 | −0.019 |
| **≥ half of the top-3's value missing** | 1,259 | **−0.063** |

- The single best player missing is **not** a signal: −0.02 pts/match, 95%
  bootstrap CI −0.09 .. +0.05. For €50M+ stars it even leans positive -
  clubs rest stars in games they expect to win anyway, so "star benched" partly
  marks an easy fixture.
- How much of the core is missing **is**: ≥ half of the top-3's value out vs
  none out is **−0.107 pts/match (95% CI −0.19 .. −0.03)**, and the same for
  the hybrid model (−0.108) - the market's *opening* odds don't price it in
  either. That makes a matchday re-score on the confirmed XI worth having -
  keyed on the share of top-3 value missing, not on the star alone.

## Matchday re-score on the confirmed XI  (`src/fit_lineup_adjustment.py`, `src/matchday_rescore.py`)

The site's Dixon-Coles odds never see a lineup. The adjustment:

    xg_side' = xg_side × exp(OWN × own_top3_missing_share + OPP × opponent_top3_missing_share)

fitted by Poisson likelihood on walk-forward (leak-free, refit every 28 days)
Dixon-Coles expected goals for 2021-24 league matches, and evaluated untouched on
2024-26 (`data/processed/lineup_adjustment.json`):

- **OWN = −0.126**: a side missing all of its top-3 by value scores ~12% fewer
  goals than expected; **OPP = +0.088**: a side scores ~9% more when the
  *opponent* is missing all of its top-3.
- Out of sample (2024-26, 3,327 matches): 1X2 log-loss **0.9983 → 0.9949**,
  a gain of **0.0034 (paired-bootstrap 95% CI 0.0016 .. 0.0053)** - small, but
  clearly not noise. (Independent-Poisson grids here, so the plain number is
  above the site's Dixon-Coles; the comparison is like for like.)

Live: `.github/workflows/matchday.yml` runs every 30 minutes; a stdlib-only
check (`matchday_due.py`) exits unless a big-5 league match kicks off within
~2 hours. For those whose FotMob lineup is confirmed (`lineupType` "standard" -
before that FotMob serves a "predicted" XI, which is ignored),
`matchday_rescore.py` appends the plain and XI-adjusted odds to the append-only
`predictions_ledger_xi.csv`; the Track Record tab scores both against results
and the closing market. Opening matchdays are skipped (no squad history in the
45-day window yet).

## Obvious next improvements

- **Value model → 0.95**: the has-prev segment is already at 0.90 (Transfermarkt's
  own estimate noise is roughly the ceiling there); cold-start rows have their
  own sub-model now (R²(log) 0.62, up from a 0.72 gap-to-warm pre-split) but
  are still the weak segment. Remaining lever: a prior value for players
  arriving from outside the big-5 - a Championship / Eredivisie / Primeira
  Liga TM scrape, or more transfer-fee / salary coverage beyond the current
  targeted profile backfill - plus a real 2022-23 TM scrape to fill the
  mirror hole.
- Advanced player stats for 2023-26 (blocked - FBref gates, mirror stale).
- Outcome model: `xi_value_ratio` (Step 11) only closes the gap for matches
  already played or scored near kickoff, not the model's normal day(s)-ahead
  predictions - confirmed lineups / injuries are still the sliver Bet365's
  closing line has that this model doesn't. The lineup-absence backtest and
  the matchday re-score above now use the confirmed XI for the site's odds;
  folding the same top-3-missing share into the classifier's features is the
  next step.
- A live poller / in-play scoring service on top of `live_win_prob.py`,
  backed by the FotMob `matchDetails` clock (`d["general"]`) - the backtest
  validates the math; nothing polls a live match yet.
- Aging curves: the delta method (below, under Aging curves) predicts
  year-over-year value changes better than the cross-sectional curve but runs
  high for thin-minutes players - an age curve conditioned on minutes played
  (playing vs. not) would likely beat both for the carried path.
- Betting backtester: the favourite-side edge (the 1.0-1.5 odds bucket in the
  table above - small-n and it has swung between +2% and +12% across reruns)
  is worth re-checking as more test-set seasons accumulate before trusting it
  - and testing the actual "low-liquidity leagues" hypothesis needs a market
  outside the big-5 (Eredivisie, Championship, ...) this dataset doesn't have.
