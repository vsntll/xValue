# `match_features.csv` — data dictionary

Built by `src/pull_match_archive.py` from football-data.co.uk season CSVs.
One row per match. 6396 rows = 6 seasons × (380 E0 + 380 SP1 + 306 D1).

## Coverage

- Leagues: Premier League (`E0`), La Liga (`SP1`), Bundesliga (`D1`)
- Seasons: 2020-21 … 2025-26 (all complete)
- Date range: 2020-09-12 … 2026-05-24

## Columns

| Column | Meaning |
| --- | --- |
| `season` | e.g. `2024-25` (injected) |
| `league` | human-readable league name (injected) |
| `Div` | football-data division code (`E0`/`D1`/`SP1`) |
| `Date` | match date, ISO (parsed from dd/mm/yy) |
| `Time` | kickoff local time |
| `HomeTeam`, `AwayTeam` | football-data team names — **not yet resolved** to FBref names (step 4) |
| `Referee` | Premier League only; blank for D1/SP1 (source does not provide it) |
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
the outcome model itself does not train on them.

## Known data-quality notes

- **Union Berlin 0–2 Bochum, 2024-12-14** — result present, all match stats `NaN`.
  The match was abandoned after ~15 min and the score awarded by the DFB, so no
  shot/corner/card data exists. Legitimate, not corruption.
- `FTR` class balance across all rows: H 44.0% / A 30.9% / D 25.1%. Draws are the
  minority class — matters for step 8 per-class metrics.
- Team names are raw football-data strings (e.g. "Man City", "Ath Bilbao").
  Cross-source resolution happens in step 4.
