# FBref ingestion — status and blockers

## The plan (steps 1-cups and step 2)

Pull per-team all-competitions match logs from FBref
(`/squads/{id}/{season}/matchlogs/all_comps/...`) to get cup / European /
friendly fixtures with the same shot/card/corner detail as league play, plus
season-level player stats. `soccerdata.FBref.read_team_match_stats()` already
targets the `all_comps` URL, so this would be a wrapper around that.

## RESOLVED 2026-09-02 (evening) — switched to nodriver

`soccerdata` is dropped for this. Its seleniumbase-UC path drives a patched
`uc_driver.exe` (undetected-chromedriver 152) that **crashes** on this machine —
Windows `APPCRASH`, faulting module `uc_driver.exe`, ~2 s after the first
navigation, repeatedly (confirmed in the Application event log, ~20 crashes while
debugging; not Defender — no detection events). That's why every soccerdata
attempt got ~10 pages then `connection refused` forever after.

**`nodriver`** (pure Chrome DevTools Protocol, *no* driver binary) on **Python
3.11** clears FBref's Cloudflare challenge reliably and fetched schedule +
shooting + keeper pages back to back with no crash. New scripts:

- `src/pull_fbref_matchlogs.py` — nodriver. Per (league, season): read the squad
  list off the competition Stats page (`stats_squads_standard_for` table), then
  GET each `/squads/<id>/<season>/matchlogs/all_comps/<stat>/` and save it to
  `data/raw/fbref/pages/<COMP>_<season>_<slug>_<stat>.html`. Skips files already
  on disk → resumable. `--status` for progress.
- `src/parse_fbref_matchlogs.py` — unchanged in spirit: `pandas.read_html` over
  the cached pages → `data/processed/fbref_team_matchlogs.csv`, `Comp` mapped to
  `competition_type`.

Competition ids: PL 9, Championship 10, Bundesliga 20, 2.Bundesliga 33,
La Liga 12, La Liga 2 (Segunda) 17.

## Match-log scrape COMPLETE 2026-09-03 (~02:15)

Full run: 6 leagues x 7 seasons (2020-21 … 2026-27) x 4 stat types (schedule,
shooting, keeper, misc) = **~3,380 pages** in `data/raw/fbref/pages/`, ~4 h at
~5 s/page, one Cloudflare skip (backfilled). nodriver never crashed.

`data/processed/fbref_team_matchlogs.csv`: **~37,500 rows**, 159 teams, one row
per team per match, schedule+shooting+keeper+misc stitched (54 cols). Stat
columns ~86 % populated on league rows (the gap is mostly unplayed 2026-27
fixtures, which FBref lists in full).

    competition_type   rows
    league            33,158
    domestic_cup        1,961   (FA Cup, DFB-Pokal, Copa del Rey)
    european            1,615   (UCL / UEL / UECL)
    league_cup            696   (EFL Cup)
    super_cup              75
    playoff               36    (promotion/relegation)

## Player season stats 2026-09-03

`src/pull_fbref_player_stats.py` + `parse_fbref_player_stats.py`. FBref's
league-wide player-stats pages, 11 categories x 6 leagues x 7 seasons = 462 pages
in `data/raw/fbref/player_stats/`.

**FBref gates its advanced (Opta) tables from scrapers** — passing, passing_types,
defense, possession, gca, keeper_adv come back as empty skeletons (no xG on the
standard page either). Confirmed across comp/squad/player pages with 10 s settle
waits. Only the basic tables come through: standard (goals/assists/minutes/cards/
pens), shooting (Sh/SoT/dist), misc (fouls/offsides/recoveries/aerials), keeper
(saves/GA/CS), playing_time.

The advanced stats for **PL / Bundesliga / La Liga, 2020-21..2022-23** come from
the worldfootballR mirror instead — `src/pull_wfr_advanced.py` reads
`fb_big5_advanced_season_stats/*.rds` (xG/xAG/npxG, progressive passing, tackles,
possession, GCA). The mirror stopped updating advanced stats after 2022-23, so
2023-24 onward + the three 2nd tiers wait for API-Football.

`data/processed/fbref_player_season_stats.csv` = one row per (player, squad,
season), **24,081 rows, 274 cols**. Basic stats everywhere; advanced on ~4,150
big-5 2020-22 rows (84 % of mirror rows joined — name-mismatch losses; a
player-id join would lift this).

## Live season -> API-Football

`src/pull_api_football.py` is scaffolded (fixtures / match-stats / players,
env-var or `.env` auth, league-id map). Needs a **paid** API-Football key (free
plan is seasons 2021-23 only). Once keyed: run for 2026-27, add a parser onto
the `match_features.csv` / player-stats schema.

## Still to do

- Dedupe the ~4,380 FBref cup/European/etc. rows to one-per-match, union into
  `match_features.csv` (needs FBref<->football-data name resolution — step 4).
- Cross-source player id resolution (FBref <-> Transfermarkt) for steps 3/4.

Everything below is the earlier debugging record, kept for context.

## (obsolete) Update 2026-09-02 — Smart App Control OFF, but the scrape still won't run headless-or-agent

The user disabled Smart App Control and restarted. That clears blocker #2 below.
Chrome-for-Testing now runs. `soccerdata` + seleniumbase-UC **does** get past
Cloudflare — but only for the first ~10 pages of a session, then it breaks
systemically and cannot recover:

- seleniumbase UC mode monkeypatches `driver.get` → `uc_special_open_if_cf`,
  which **disconnects chromedriver and reconnects** to dodge bot detection. After
  the initial working window the reconnect fails every time:
  `HTTPConnectionPool(host='localhost', port=…): connection refused` on
  `/session/…/window/handles`. soccerdata burns its 5 retries (each re-init also
  dead) and raises `ConnectionError`.
- Reproduced identically across: headless & non-headless, Python 3.14 & 3.11,
  soccerdata & raw `seleniumbase.Driver`, `driver.get` & `uc_open_with_reconnect`.
  Not a Cloudflare problem, not a Defender/SAC problem (no detections, drivers
  present) — it's UC-mode's reconnect failing in a background/non-interactive
  context. Likely works in a genuine **interactive foreground terminal** on the
  user's desktop (real display + focus), which the agent session can't provide.
- 10 PL 2023-24 schedule pages were cached before it broke; they parse cleanly
  with `pandas.read_html(..., attrs={"id": "matchlogs_for"})` — **no browser
  needed to parse, only to fetch.** `Comp` column separates league / cup /
  European rows exactly as step 1c needs.

**Where this leaves ingestion — pick one:**
1. User runs `python src/pull_fbref_matchlogs.py` in their own terminal (not via
   the agent). It's now chunked per (league, season, stat) and resumable via
   `data/raw/fbref/scrape_progress.json` + the page cache. Multi-hour grind.
2. worldfootballR mirror for the back-catalogue (2019-20…2023-24 complete), tiny
   live scrape for the current season only.
3. **API-Football** (the user already plans to switch to it): a **paid** plan
   ($19–39/mo) unlocks all historical seasons + cups + lineups + events via clean
   JSON — no scraping at all, for history *and* ongoing. Free plan is capped to
   seasons 2021–2023 so it won't cover 2020-21 or 2024-25+.

## Historical blocker detail (pre-2026-09-02)

Hit three compounding problems:

1. **Cloudflare.** `fbref.com` returns HTTP 403 (managed challenge) to plain
   `requests` and to `curl_cffi` browser-impersonation. It needs a real automated
   browser that executes the challenge JS.
2. **Smart App Control is ON** (`VerifiedAndReputablePolicyState = 1`). It blocks
   every unsigned / unknown-reputation executable, which is every browser+driver
   combo installable without admin:
   - Chrome for Testing `chrome.exe` — not Authenticode-signed → blocked
     (`WinError 4551: An Application Control policy has blocked this file`).
   - seleniumbase's patched `uc_driver.exe` — patched binary, unknown reputation
     → blocked.
   - Google Chrome's own installer needs UAC elevation, which can't be approved
     from a non-interactive shell.
3. **Edge** (the one Microsoft-signed browser present) drives fine via
   `msedgedriver`, but headless it stalls on the Cloudflare "Just a moment"
   challenge, and non-headless it exits immediately in this non-interactive
   session. `nodriver` (pure-CDP, no driver binary) would sidestep the driver
   issue but is broken on this machine's Python 3.14 (source-encoding bug in its
   generated CDP module).

## Fallback in use: the worldfootballR data mirror

`github.com/JaseZiv/worldfootballR_data` publishes FBref data pre-scraped by the
R-ecosystem equivalent of `soccerdata`, served from GitHub releases (no
Cloudflare). Relevant assets:

| Asset | Contents | Our use |
| --- | --- | --- |
| `match_results_cups/*` | FA Cup, EFL Cup, Copa del Rey, DFB-Pokal, UCL, UEL, UECL results (+ xG for UCL/UEL) | cup / European match rows |
| `match_results/*` | domestic league results | cross-check vs football-data |
| `fb_big5_advanced_season_stats/*` | season player stats (goals, assists, minutes, per-90) | step 2 player features |
| `fb_advanced_match_stats/*`, `fb_match_summary/*`, `fb_match_shooting/*` | per-match stats — **league only** | form / squad rollup |
| `fbref-tm-player-mapping/*` | FBref ↔ Transfermarkt player-ID crosswalk | step 4 |
| `tm_player_vals/*` | Transfermarkt player values by season | step 3 |
| `understat_shots/*` | Understat shot/xG events | the diagram's 4th source |

### Limitations of the mirror

- **Not live.** Cup results are complete for 2019-20 … 2023-24, partial for
  2024-25, absent for 2025-26 and 2026-27. Refreshes when the maintainer reruns,
  not on demand.
- **Cup matches carry results only** — no shots/corners/cards per cup match
  (xG only, and only for UCL/UEL). Matches the expected stat degradation for
  knockout football, so `has_shot_data = False` on all these rows.
- **No club friendlies.** The mirror's friendlies file is national-team only.
- **No domestic super cups** (Community Shield, Supercopa, DFL-Supercup) as
  separate assets.
- String fields have latin-1 mojibake (`Cádiz` → `C�diz`) needing repair.

## Open decision

Whether to also stand up the live Chrome-based scrape (needs Google Chrome
installed with a UAC approval, or Smart App Control turned off) to cover the
current season and get full stats where FBref has them. Until then, cup coverage
stops at the 2024-25 season.
