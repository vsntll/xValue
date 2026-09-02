# FBref ingestion — status and blockers

## The plan (steps 1-cups and step 2)

Pull per-team all-competitions match logs from FBref
(`/squads/{id}/{season}/matchlogs/all_comps/...`) to get cup / European /
friendly fixtures with the same shot/card/corner detail as league play, plus
season-level player stats. `soccerdata.FBref.read_team_match_stats()` already
targets the `all_comps` URL, so this would be a wrapper around that.

## Blocker: FBref can't be scraped from this machine

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
