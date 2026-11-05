"""Cheap gate for the matchday workflow: is any big-5 league match kicking off
in the next WINDOW (minutes)? Standard library only, so the half-hourly cron
can answer it before installing anything; the heavy re-score
(matchday_rescore.py) runs only when this prints due=true.

Writes `due=true|false` to $GITHUB_OUTPUT when set, else prints it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request

LEAGUE_ID = {"ENG1": 47, "GER1": 54, "ESP1": 87, "ITA1": 55, "FRA1": 53}
WINDOW = (10, 130)   # minutes before kickoff - lineups are confirmed ~60-75 min out


def season_label(now: dt.datetime) -> str:
    y = now.year if now.month >= 7 else now.year - 1
    return f"{y}/{y + 1}"


def kickoffs_in_window(now: dt.datetime) -> list[tuple[str, str]]:
    out = []
    for code, lid in LEAGUE_ID.items():
        url = f"https://www.fotmob.com/api/data/fixtures?id={lid}&season={season_label(now).replace('/', '%2F')}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                fixtures = json.load(r)
        except (OSError, ValueError):
            continue
        for f in fixtures if isinstance(fixtures, list) else []:
            st = f.get("status") or {}
            if st.get("started") or st.get("finished") or st.get("cancelled") or not st.get("utcTime"):
                continue
            ko = dt.datetime.fromisoformat(st["utcTime"].replace("Z", "+00:00"))
            mins = (ko - now).total_seconds() / 60
            if WINDOW[0] <= mins <= WINDOW[1]:
                out.append((code, str(f["id"])))
    return out


def main() -> None:
    due = kickoffs_in_window(dt.datetime.now(dt.timezone.utc))
    line = f"due={'true' if due else 'false'}"
    print(f"{len(due)} league kickoffs in the next {WINDOW[0]}-{WINDOW[1]} min: {due}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
