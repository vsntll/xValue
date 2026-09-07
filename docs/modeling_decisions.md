# Modeling decisions

Decisions locked in while building the data layer, so downstream steps don't
re-litigate them. Revisit if evaluation says otherwise.

## Competition scope for the outcome classifier

**Train and evaluate on league matches only; let the Elo / form features carry
the cross-competition signal.**

- The original plan was to pool every competition type with `competition_type` as
  a categorical feature. In practice `build_match_model_table.py` filters
  `matches_all.csv` to `competition_type == 'league'` (10.9k rows) before
  building the feature table — cup rows have patchy shot/xG stats and no
  bookmaker line, so they added more noise than signal to the classifier.
- The cross-competition information isn't lost: the goals-Elo and xG-Elo in the
  feature table **are** updated on every competition (league + domestic cup +
  European), so a team's rating going into a league match already reflects its
  midweek Champions League form.
- `matches_all.csv` still carries all competition types (`league`,
  `domestic_cup`, `league_cup`, `european`, `super_cup`, `playoff`) for the Elo
  update, the squad rollup, the site's recent-results / cup-finals views, and a
  future cup model. Friendlies are not currently ingested from any source.
- The bookmaker-odds benchmark exists only for league matches (football-data.co.uk
  has no cup odds), which is the other reason the eval is league-only.

## Target definition for knockout matches

**3-class W/D/L, decided by the score after 90' + extra time, before penalties.**

- A cup tie level after extra time and decided on penalties is a **draw** for the
  classifier target. Penalty shootouts are ~coin-flips; a 4th class would add
  noise, not signal.
- Extra columns preserve the detail for anyone who wants a separate
  "who advances" model later:
  - `decided_by`: `regulation` | `extra_time` | `penalties`
  - `went_to_penalties`: bool
  - `pens_home`, `pens_away`: shootout score (nullable)
- Two-legged ties: each leg is its own row with its own 90' result. Aggregate
  advancement is out of scope for a per-match outcome model.

*Current state:* `matches_all.csv` carries only `FTHG` / `FTAG` / `FTR` — the
extra-time / penalties detail above is not populated by any source we pull, so a
cup tie that went to a shootout is recorded as the drawn score after ET. The
site's cup-finals view shows no winner when the final finished level.

## Null handling for match stats

Stat completeness degrades away from league play (domestic cups usually full,
European solid, lower-round cups and friendlies often goals + cards only).

- Downstream feature engineering must treat `NaN` shots/possession as missing,
  never coerce to 0. Models that can't take NaN get an explicit imputation step
  with a companion missingness indicator. (`matches_all.csv` leaves the cells
  `NaN`; there is no `has_shot_data` flag column — absence of `HS`/`AS` is the
  signal.)
