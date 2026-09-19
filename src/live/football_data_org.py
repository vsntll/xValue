"""football-data.org v4 - the one *sanctioned*, stable free API in the mix.

Free tier: 10 req/min, 12 competitions incl. PL (PL) / Bundesliga (BL1) /
La Liga (PD) / Champions League (CL). Fixtures, results, standings - **no**
shot/possession detail. Use it as the reliability backbone; layer ESPN on top for
stats.

Needs a free key: register at football-data.org, put it in .env as
FOOTBALL_DATA_ORG_KEY.
"""

from __future__ import annotations

import os
import time

import requests

from .schema import ALL_COMPS, blank_frame, finalize

BASE = "https://api.football-data.org/v4"


def _token() -> str:
    tok = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
    if not tok:
        raise SystemExit(
            "No FOOTBALL_DATA_ORG_KEY (env or .env). Register free at "
            "football-data.org."
        )
    return tok


def fetch(comp_codes: list[str], season: str, with_stats: bool = False) -> "object":
    """with_stats is accepted for interface parity but ignored - this source has
    no match stats."""
    s = requests.Session()
    s.headers["X-Auth-Token"] = _token()
    start_year = season.split("-")[0]
    rows = []
    for code in comp_codes:
        cfg = ALL_COMPS.get(code)
        if not cfg or not cfg.get("fdorg"):
            continue
        r = s.get(f"{BASE}/competitions/{cfg['fdorg']}/matches",
                  params={"season": start_year}, timeout=30)
        if r.status_code == 429:
            print("  football-data.org: rate limited (10/min) - waiting 60s")
            time.sleep(60)
            r = s.get(f"{BASE}/competitions/{cfg['fdorg']}/matches",
                      params={"season": start_year}, timeout=30)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        print(f"  football-data.org {code}: {len(matches)} matches")
        for m in matches:
            # fullTime score is only meaningful once the match has actually
            # started - for TIMED/SCHEDULED fixtures the API sometimes returns
            # {home: 0, away: 0} instead of null, which would look like a real
            # 0-0 result if taken unconditionally.
            live_score = m["status"] not in ("TIMED", "SCHEDULED", "POSTPONED", "CANCELLED")
            ft = m["score"]["fullTime"]
            row = {
                "season": season, "league": cfg["name"], "tier": cfg["tier"],
                "competition_type": cfg["competition_type"], "comp_code": code,
                "Date": m["utcDate"][:10], "Time": m["utcDate"][11:16],
                "HomeTeam": m["homeTeam"]["name"], "AwayTeam": m["awayTeam"]["name"],
                "status": m["status"],
                "FTHG": ft["home"] if live_score else None,
                "FTAG": ft["away"] if live_score else None,
                "match_id": f"fdorg:{m['id']}",
            }
            if m["status"] == "FINISHED":
                htv = m["score"].get("halfTime", {})
                row["HTHG"], row["HTAG"] = htv.get("home"), htv.get("away")
            rows.append(row)
        time.sleep(6.5)  # stay under 10/min
    return finalize(rows, "football-data.org") if rows else blank_frame()
