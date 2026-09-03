"""FotMob unofficial JSON API - the xG source for cups / Europe.

Understat has no cup coverage, ESPN has no xG, so FotMob fills xG (and a rich
stat set) for UCL / UEL / UECL / FA Cup / EFL Cup / DFB-Pokal / Copa del Rey - and
works for the leagues too as a backup.

Endpoints (no key today; may need an ``x-mas`` header later):
    /api/data/fixtures?id=<leagueId>&season=YYYY%2FYYYY   season fixture list
    /api/data/matchDetails?matchId=<id>                   xG + full box score

FotMob's fixture-list home/away fields are team-relative and unreliable, so we
read the correct teams + score + xG from matchDetails. Each match detail is
cached (``data/raw/live/fotmob/<id>.json``, trimmed) so re-runs cost almost
nothing. FotMob's ToS restricts programmatic use - keep volume low.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .schema import ALL_COMPS, blank_frame, finalize

BASE = "https://www.fotmob.com/api/data"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "live" / "fotmob"
SLEEP = 0.5

# our comp code -> FotMob league id
_LID = {
    "ENG1": 47, "GER1": 54, "ESP1": 87,
    "UCL": 42, "UEL": 73, "UECL": 10216,
    "FA": 132, "EFL": 133, "DFB": 209, "CDR": 453,
}
# FotMob "Top stats" title -> (home col, away col) target
_STAT_TITLES = {
    "Expected goals (xG)": ("HxG", "AxG"),
    "Ball possession": ("HPoss", "APoss"),
    "Total shots": ("HS", "AS"),
    "Shots on target": ("HST", "AST"),
    "Corners": ("HC", "AC"),
    "Fouls": ("HF", "AF"),
    "Yellow cards": ("HY", "AY"),
    "Red cards": ("HR", "AR"),
}


def _get(url: str, **params) -> dict | list:
    r = requests.get(url, params=params, headers=HDRS, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def _num(v):
    if v is None:
        return None
    s = str(v).split(" ")[0].replace("%", "")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def _detail(match_id: str) -> dict | None:
    """Trimmed match detail: correct teams, score, xG + box-score stats."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{match_id}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    try:
        d = _get(f"{BASE}/matchDetails", matchId=match_id)
    except requests.HTTPError:
        return None
    g = d.get("general", {})
    out = {
        "match_id": f"fotmob:{match_id}",
        "HomeTeam": g.get("homeTeam", {}).get("name"),
        "AwayTeam": g.get("awayTeam", {}).get("name"),
        "Date": (g.get("matchTimeUTCDate") or "")[:10],
        "Time": (g.get("matchTimeUTCDate") or "")[11:16],
    }
    hteams = d.get("header", {}).get("teams", [])
    if len(hteams) == 2:
        out["FTHG"], out["FTAG"] = hteams[0].get("score"), hteams[1].get("score")
    for grp in d.get("content", {}).get("stats", {}).get("Periods", {}).get("All", {}).get("stats", []):
        if grp.get("title") != "Top stats":
            continue
        for s in grp.get("stats", []):
            tgt = _STAT_TITLES.get(s.get("title"))
            vals = s.get("stats")
            if tgt and isinstance(vals, list) and len(vals) == 2:
                out[tgt[0]], out[tgt[1]] = _num(vals[0]), _num(vals[1])
    dst.write_text(json.dumps(out))
    return out


def fetch(comp_codes: list[str], season: str, with_stats: bool = True) -> "object":
    start = int(season.split("-")[0])
    fm_season = f"{start}/{start + 1}"  # requests encodes the slash
    rows = []
    for code in comp_codes:
        lid = _LID.get(code)
        cfg = ALL_COMPS.get(code)
        if not lid or not cfg:
            continue
        try:
            fixtures = _get(f"{BASE}/fixtures", id=lid, season=fm_season)
        except requests.HTTPError as e:
            print(f"  fotmob {code}: fixtures failed ({e})")
            continue
        if not isinstance(fixtures, list):
            continue
        finished = [f for f in fixtures if f.get("status", {}).get("finished")]
        print(f"  fotmob {code}: {len(fixtures)} fixtures, {len(finished)} finished")
        for fx in finished:
            det = _detail(str(fx["id"])) if with_stats else None
            base = {
                "season": season, "league": cfg["name"], "tier": cfg["tier"],
                "competition_type": cfg["competition_type"], "comp_code": code,
                "status": "FT", "match_id": f"fotmob:{fx['id']}",
            }
            sc = fx.get("status", {}).get("scoreStr", "")
            if " - " in sc:
                pass  # home/away unreliable here; trust the detail
            if det:
                base.update({k: v for k, v in det.items() if v is not None})
            rows.append(base)
    return finalize(rows, "fotmob") if rows else blank_frame()
