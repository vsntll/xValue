# `match_features.csv` — data dictionary

Built by `src/pull_match_archive.py` from football-data.co.uk season CSVs.
One row per match.

## Coverage

- **Competitions:** the top flight of each country — Premier League (England),
  Bundesliga (Germany), La Liga (Spain). All league play; see "Not included"
  below. (The three 2nd tiers were in scope until 2026-09-03, then dropped.)
- **Seasons:** 2020-21 … 2025-26 complete, plus **2026-27 ongoing** (live —
  re-running the script pulls in matches played since the last run).
- **Rows:** ~6,455. Per completed season: 1066 (380 E0 + 306 D1 + 380 SP1).
  2026-27 grows over time.
- **Date range:** 2020-09-12 … whatever the latest played match is.

## Columns

| Column | Meaning |
| --- | --- |
| `season` | e.g. `2024-25` (injected) |
| `league` | human-readable league name (injected) |
| `tier` | always 1 (injected) — kept for schema stability |
| `competition_type` | always `league` here — the column exists so cup/friendly rows from another source can be unioned in later |
| `Div` | football-data division code (`E0`/`D1`/`SP1`) |
| `Date` | match date, ISO (parsed from dd/mm/yy) |
| `Time` | kickoff local time |
| `HomeTeam`, `AwayTeam` | football-data team names — **not yet resolved** to FBref names (step 4) |
| `Referee` | England only (`E0`); blank for German/Spanish leagues (source does not provide it) |
| `FTHG`, `FTAG`, `FTR` | full-time home goals, away goals, result (`H`/`D`/`A`) |
| `HTHG`, `HTAG`, `HTR` | half-time equivalents |
| `HS`, `AS` | shots |
| `HST`, `AST` | shots on target |
| `HC`, `AC` | corners |
| `HF`, `AF` | fouls committed |
| `HY`, `AY` | yellow cards |
| `HR`, `AR` | red cards |
| `B365H/D/A` | Bet365 pre-match 1X2 odds |
| `B365CH/CD/CA` | Bet365 closing 1X2 odds |
| `AvgH/D/A` | market-average pre-match 1X2 odds |
| `AvgCH/CD/CA` | market-average closing 1X2 odds |

Odds columns are carried only for the step 8 benchmark (model vs. bookmaker);
the outcome model itself does not train on them. `AvgH/D/A` is the most complete
odds column (no nulls); `B365H` has a handful of gaps in the ongoing season where
Bet365 prices had not posted at pull time.

## Not included — friendlies and knockout cups

football-data.co.uk distributes **domestic league play only**. It has no feed for
friendlies or for knockout competitions (FA Cup, EFL Cup, DFB-Pokal, Copa del Rey,
Champions League, Europa League, ...).

To add those, the source is **FBref team match logs** (step 2), which list every
competitive fixture per team per season with the same shot/card/corner detail.
Friendlies are also on FBref but shot-level detail is spotty. When pulled, those
rows get `competition_type` of `cup` / `european` / `friendly` and union into this
same table on the shared schema.

## Known data-quality notes

- **Union Berlin 0–2 Bochum, 2024-12-14** — result present, all match stats `NaN`.
  Match abandoned after ~15 min and the score awarded by the DFB, so no
  shot/corner/card data exists. Legitimate, not corruption.
- `FTR` class balance across all rows: ~H 44% / A 30% / D 26%. Draws are the
  minority class — matters for step 8 per-class metrics.
- Team names are raw football-data strings (e.g. "Man City", "Ath Bilbao").
  Cross-source resolution happens in step 4.
