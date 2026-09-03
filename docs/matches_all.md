# `matches_all.csv` - the modelling match table

Built by `src/build_matches_all.py` from every match source. One row per match,
all competitions, 2020-21 .. 2026-27, top flight of England / Germany / Spain
plus the cups their clubs play in.

## Composition (9,133 matches as of 2026-09-03)

| competition_type | rows | source | stats |
| --- | --- | --- | --- |
| league | 6,456 | football-data.co.uk (2020-26) + live sources (2026-27) | shots/SoT/corners/fouls/cards everywhere; **xG 93%** (Understat); possession only 2026-27 |
| european (UCL/UEL/UECL) | 1,346 | FBref match logs + ESPN/FotMob (2024-25 on) | result-only history; **xG + shots for 2024-25, 2025-26, 2026-27** |
| domestic_cup (FA/DFB/Copa) | 1,005 | same | same |
| league_cup (EFL) | 277 | same | same |
| super_cup | 37 | same | |
| playoff | 12 | same | |

Overall: **xG 75%**, shots 80%, possession 10%. Cup xG: 0% for 2020-24,
76-78% for 2024-25 & 2025-26, ~100% for 2026-27.

Cup/European rows are scoped to ties involving a club from ENG1/GER1/ESP1 that
season (matches FBref). ~10% of the 2024-25/2025-26 live cup rows don't dedupe
cleanly against the FBref rows yet (team-name drift - step 4).

## Gaps and how to close them

- **Historical cup xG + shot stats** (~2,300 matches, 2020-26): FBref gates shot
  detail for cups and Understat has no cup coverage. FotMob has both -
  `py -3.11 src/pull_live.py --season <s> --comps UCL UEL UECL FA EFL DFB CDR`
  per historical season, then re-run `build_matches_all.py` after teaching it to
  read `data/raw/live/*_<season>.csv`. ~2,300 FotMob matchDetails calls - ToS-
  sensitive at that volume.
- **Historical league possession** (all 6,396 rows): football-data.co.uk has no
  possession. ESPN box scores do - `pull_live.py` per historical season with
  `--sources espn`, ~14k summary calls.
- **Name-match misses**: ~7% of league rows don't get Understat xG (spelling
  drift the alias map in `live/schema.py` doesn't cover yet).

## Columns

`season, competition_type, comp, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
HTHG, HTAG, HTR, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR, HPoss, APoss,
HxG, AxG, source`

Team names are still per-source (football-data / FBref / ESPN / Understat /
FotMob) - full resolution to a canonical id is step 4.
