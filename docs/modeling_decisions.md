# Modeling decisions

Decisions locked in while building the data layer, so downstream steps don't
re-litigate them. Revisit if evaluation says otherwise.

## Competition scope for the outcome classifier

**Train on all competition types pooled, with `competition_type` as a categorical
feature. Additionally hold out cup matches as a separate eval slice.**

- Pooling gives the tree model more data and, crucially, more variety in
  squad-strength vs. outcome pairings — cup matches with rotated XIs are exactly
  where the squad rollup feature (step 6) earns its keep.
- `competition_type` (values: `league`, `domestic_cup`, `league_cup`,
  `european`, `super_cup`, `friendly`) lets the model separate the distributions
  rather than smearing them into one average.
- Report metrics three ways: overall, league-only, cups-only. The bookmaker-odds
  benchmark (step 8) only exists for league matches — football-data.co.uk has no
  cup odds — so the cup slice is model-vs-model, not model-vs-market.
- **Friendlies**: kept in the table for completeness and for form/squad-rollup
  lookback, but excluded from the classifier's training set by default (dead
  rubbers, experimental lineups, unreliable stats). Flag: rows where
  `competition_type == 'friendly'`.

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

## Null handling for match stats

Stat completeness degrades away from league play (domestic cups usually full,
European solid, lower-round cups and friendlies often goals + cards only).

- `has_shot_data`: bool — distinguishes "0 shots" from "shots unknown".
- Downstream feature engineering must treat `NaN` shots/possession as missing,
  never coerce to 0. Models that can't take NaN get an explicit imputation step
  with a companion missingness indicator.
