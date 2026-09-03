# Current-season ingestion

FBref covers the historical seasons (2020-21..2025-26); it's slow and gates
advanced stats, so the **ongoing 2026-27 season** comes from lighter unofficial
JSON APIs instead. API-Football was the plan but its free tier is unusable
(seasons 2022-24 only, 100 req/day, **page parameter capped at 3** → 60 records
max); a paid plan would work but isn't in place.

## Design — `src/pull_live.py` + `src/live/`

None of the free options have an SLA, so the design is **redundancy**: a priority
list of sources, each a module exposing

    fetch(comp_codes, season, with_stats) -> DataFrame[schema.MATCH_COLS]

Downstream only sees `MATCH_COLS` (aligned with `match_features.csv` + possession
+ `source`/`match_id`). Swap or reorder sources without touching anything else.
`pull_live.py` runs them in priority order and fills each merged row field-by-
field from the first source that has it (key: comp_code + normalised home + away).

| module | source | key? | gives | status |
| --- | --- | --- | --- | --- |
| `live/football_data_org.py` | football-data.org v4 | `FOOTBALL_DATA_ORG_KEY` in `.env` | fixtures, results, HT score — **sanctioned, stable**. No shot/possession. 10 req/min, PL/BL1/PD/CL. | working |
| `live/espn.py` | site.api.espn.com | none | fixtures, results, **+ box-score stats** (shots, SoT, corners, fouls, cards, possession) for the 3 leagues + UCL/UEL/UECL + FA/EFL/DFB/Copa. Unofficial. Summaries cached. | working |
| `live/understat.py` | understat.com (via soccerdata, TLS client) | none | **+ xG** (per-match team xG). The 3 leagues only — no cups. | working |
| `live/fotmob.py` | fotmob.com/api/data | none (may need `x-mas` later) | **+ xG** for cups/Europe (and leagues). Reads teams+xG from `matchDetails` (fixture-list home/away is unreliable). Details cached. ToS restricts — low volume. | working |
| `live/sofascore.py` | api.sofascore.com | none | + xG | stub — fallback |

Merge is **spine + fuzzy-match** (`_same_match`): first source to carry a comp
defines its match list; later sources are matched on comp + date (±3d) + both
team names (`teams_match`: normalized-equal or decisive token overlap, with a
score+date fallback for names one source mangles) and fill missing fields.

Comp codes: `ENG1 GER1 ESP1` (leagues) + `UCL UEL UECL FA EFL DFB CDR` (cups).
Default = leagues + UCL/FA/EFL/DFB/CDR.

## Run

    py -3.11 src/pull_live.py                          # current season, default comps, ESPN
    py -3.11 src/pull_live.py --season 2026-27 --comps ENG1 GER1 ESP1 UCL
    py -3.11 src/pull_live.py --no-stats               # fixtures/results only, fast
    py -3.11 src/pull_live.py --sources espn           # force a single source

Output: `data/raw/live/<source>_<season>.csv` (per-source) and
`data/processed/live_matches_<season>.csv` (merged).

First full run (2026-09-03): **1,594 fixtures** across ENG1/GER1/ESP1 + UCL/UEL/
UECL/EFL/DFB. 152 played — **100 % shots/possession, 99 % xG** (2 garbled-name
DFB matches missed xG). FA Cup / Copa del Rey not started yet.

## Known issues

- ESPN returns some lower-league German cup team names already mojibake'd
  (`Preu�en M�nster`) — their bug, not recoverable. `normalize_team()` strips
  non-ASCII so joins still work; only display is affected.
- No half-time score or xG from ESPN. HT would need parsing `keyEvents`; xG needs
  FotMob/Sofascore or a paid API.
- To add football-data.org: register free at football-data.org, put the token in
  `.env` as `FOOTBALL_DATA_ORG_KEY`. It then becomes the results backbone
  automatically.
