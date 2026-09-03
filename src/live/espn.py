"""ESPN hidden JSON API (site.api.espn.com) - no key, no Cloudflare.

Scoreboard gives fixtures + results for a date range; the per-event summary gives
box-score stats (shots, SoT, possession, corners, fouls, cards). Unofficial and
unsanctioned - keep volume low, expect endpoints to shift.
"""

from __future__ import annotations

import time

import requests

from .schema import ALL_COMPS, blank_frame, finalize

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SLEEP = 0.3

# ESPN box-score stat name -> our column, per home/away side
_STAT_MAP = {
    "totalShots": "S", "shotsOnTarget": "ST", "wonCorners": "C",
    "foulsCommitted": "F", "yellowCards": "Y", "redCards": "R",
    "possessionPct": "Poss",
}


def _season_range(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start}0701-{start + 1}0701"


def _get(url: str, **params) -> dict:
    r = requests.get(url, params=params, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def _event_stats(league_code: str, event_id: str) -> dict:
    """{'HS':.., 'AS':.., 'HPoss':.., ...} from the summary box score."""
    try:
        s = _get(f"{BASE}/{league_code}/summary", event=event_id)
    except requests.HTTPError:
        return {}
    out: dict[str, float] = {}
    for t in s.get("boxscore", {}).get("teams", []):
        side = "H" if t.get("homeAway") == "home" else "A"
        vals = {st["name"]: st.get("displayValue") for st in t.get("statistics", [])}
        for espn_name, suffix in _STAT_MAP.items():
            v = vals.get(espn_name)
            if v not in (None, ""):
                out[f"{side}{suffix}"] = float(v) if "." in str(v) else int(v)
    return out


def fetch(comp_codes: list[str], season: str, with_stats: bool = True) -> "object":
    rows = []
    for code in comp_codes:
        cfg = ALL_COMPS.get(code)
        if not cfg or not cfg.get("espn"):
            continue
        lc = cfg["espn"]
        try:
            board = _get(f"{BASE}/{lc}/scoreboard", dates=_season_range(season), limit=1000)
        except requests.HTTPError as e:
            print(f"  espn {code}: scoreboard failed ({e})")
            continue
        events = board.get("events", [])
        print(f"  espn {code}: {len(events)} events")
        for ev in events:
            comp = ev["competitions"][0]
            cs = {c["homeAway"]: c for c in comp["competitors"]}
            if "home" not in cs or "away" not in cs:
                continue
            st = ev["status"]["type"]
            row = {
                "season": season, "league": cfg["name"], "tier": cfg["tier"],
                "competition_type": cfg["competition_type"], "comp_code": code,
                "Date": ev["date"][:10], "Time": ev["date"][11:16],
                "HomeTeam": cs["home"]["team"]["displayName"],
                "AwayTeam": cs["away"]["team"]["displayName"],
                "status": st["name"].replace("STATUS_", ""),
                "match_id": f"espn:{ev['id']}",
            }
            if st.get("completed"):
                row["FTHG"] = int(cs["home"]["score"])
                row["FTAG"] = int(cs["away"]["score"])
                if with_stats:
                    row.update(_event_stats(lc, ev["id"]))
            rows.append(row)
    return finalize(rows, "espn") if rows else blank_frame()
