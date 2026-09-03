"""Live-season ingestion from API-Football (api-football.com).

FBref covers the historical seasons (see src/pull_fbref_matchlogs.py). This module
keeps the **current** season current via API-Football's REST API - clean JSON, no
scraping. Same output schema as the FBref path so downstream code doesn't branch.

Auth - set ONE of these (an untracked .env in the project root is read
automatically; see .env.example):
    API_FOOTBALL_KEY=...          # direct account at dashboard.api-football.com
                                  # -> https://v3.football.api-sports.io
    RAPIDAPI_KEY=...              # key obtained through RapidAPI
                                  # -> https://api-football-v1.p.rapidapi.com/v3

NOTE: the free plan only exposes seasons 2021-2023, so the live season needs a
paid plan. Nothing here runs without a key.

Run (Python 3.11+):
    py -3.11 src/pull_api_football.py --what fixtures        # results + fixtures
    py -3.11 src/pull_api_football.py --what match-stats     # lineups/events/stats per fixture
    py -3.11 src/pull_api_football.py --what players         # player season stats
    py -3.11 src/pull_api_football.py --season 2026 --leagues ENG1

Output:
    data/raw/api_football/<endpoint>_<league>_<season>[_<id>].json   raw responses
    data/processed/api_football_fixtures_<season>.csv                flattened
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "api_football"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# our short code -> API-Football league id (differs from FBref's ids)
LEAGUE_IDS = {
    "ENG1": 39, "ENG2": 40,
    "GER1": 78, "GER2": 79,
    "ESP1": 140, "ESP2": 141,
}

# our competition_type for API-Football's fixture "league" ids we may see in cups
CUP_LEAGUE_TYPE = {
    2: "european", 3: "european", 848: "european",          # UCL / UEL / UECL
    45: "domestic_cup", 48: "league_cup",                    # FA Cup / EFL Cup
    81: "domestic_cup", 529: "super_cup",                    # DFB-Pokal / Community Shield
    143: "domestic_cup", 556: "super_cup",                   # Copa del Rey / Supercopa
    556.1: "super_cup",
}

RATE_SLEEP = 0.4  # seconds between calls; paid plans cap per-minute


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _session() -> tuple[requests.Session, str]:
    _load_dotenv()
    s = requests.Session()
    if os.environ.get("API_FOOTBALL_KEY"):
        s.headers["x-apisports-key"] = os.environ["API_FOOTBALL_KEY"]
        return s, "https://v3.football.api-sports.io"
    if os.environ.get("RAPIDAPI_KEY"):
        s.headers.update({
            "x-rapidapi-key": os.environ["RAPIDAPI_KEY"],
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        })
        return s, "https://api-football-v1.p.rapidapi.com/v3"
    raise SystemExit(
        "No API key. Set API_FOOTBALL_KEY or RAPIDAPI_KEY (env var or .env). "
        "See .env.example and docs/fbref_ingestion.md."
    )


def _get(s: requests.Session, base: str, path: str, **params) -> dict:
    """One API call, with paging handled by the caller via params['page']."""
    r = s.get(f"{base}/{path}", params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise SystemExit(f"API-Football error on {path} {params}: {body['errors']}")
    time.sleep(RATE_SLEEP)
    return body


def _paged(s: requests.Session, base: str, path: str, **params) -> list[dict]:
    out, page = [], 1
    while True:
        body = _get(s, base, path, page=page, **params)
        out.extend(body.get("response", []))
        paging = body.get("paging", {})
        if page >= paging.get("total", 1):
            break
        page += 1
    return out


def pull_fixtures(leagues: list[str], season: int) -> None:
    """Every fixture (played + scheduled) for the given leagues/season, plus the
    cup fixtures those clubs played (API-Football returns those under their own
    league ids - we pick them up via team fixture lists)."""
    import pandas as pd

    s, base = _session()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in leagues:
        lid = LEAGUE_IDS[code]
        resp = _paged(s, base, "fixtures", league=lid, season=season)
        (RAW_DIR / f"fixtures_{code}_{season}.json").write_text(json.dumps(resp))
        for fx in resp:
            f, g, t = fx["fixture"], fx["goals"], fx["teams"]
            rows.append({
                "fixture_id": f["id"],
                "src_league": code,
                "season": f"{season}-{str(season + 1)[2:]}",
                "competition_type": "league",
                "Date": f["date"][:10],
                "status": f["status"]["short"],
                "HomeTeam": t["home"]["name"], "AwayTeam": t["away"]["name"],
                "FTHG": g["home"], "FTAG": g["away"],
                "went_to_penalties": fx["score"]["penalty"]["home"] is not None,
                "pens_home": fx["score"]["penalty"]["home"],
                "pens_away": fx["score"]["penalty"]["away"],
            })
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / f"api_football_fixtures_{season}.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} fixtures, {df['status'].eq('FT').sum()} played)")


def pull_match_stats(leagues: list[str], season: int) -> None:
    """lineups + events + team statistics for every played fixture. Reads the
    fixture list produced by pull_fixtures; call that first."""
    s, base = _session()
    for code in leagues:
        fx_file = RAW_DIR / f"fixtures_{code}_{season}.json"
        if not fx_file.exists():
            print(f"{code}: run --what fixtures first")
            continue
        fixtures = json.loads(fx_file.read_text())
        played = [x for x in fixtures if x["fixture"]["status"]["short"] == "FT"]
        for i, fx in enumerate(played, 1):
            fid = fx["fixture"]["id"]
            for ep in ("lineups", "events", "statistics"):
                dst = RAW_DIR / f"{ep}_{fid}.json"
                if dst.exists():
                    continue
                body = _get(s, base, f"fixtures/{ep}", fixture=fid)
                dst.write_text(json.dumps(body.get("response", [])))
            if i % 25 == 0:
                print(f"  {code}: {i}/{len(played)} fixtures")
        print(f"{code}: {len(played)} fixtures done")


def pull_players(leagues: list[str], season: int) -> None:
    """Player season stats (paged ~20/player-page)."""
    s, base = _session()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for code in leagues:
        resp = _paged(s, base, "players", league=LEAGUE_IDS[code], season=season)
        (RAW_DIR / f"players_{code}_{season}.json").write_text(json.dumps(resp))
        print(f"{code}: {len(resp)} player-season records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--what", choices=["fixtures", "match-stats", "players"],
                    default="fixtures")
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUE_IDS), choices=list(LEAGUE_IDS))
    ap.add_argument("--season", type=int, default=None,
                    help="start year, e.g. 2026 for 2026-27 (default: current)")
    args = ap.parse_args()

    season = args.season
    if season is None:
        import datetime
        t = datetime.date.today()
        season = t.year if t.month >= 7 else t.year - 1

    {"fixtures": pull_fixtures, "match-stats": pull_match_stats,
     "players": pull_players}[args.what](args.leagues, season)


if __name__ == "__main__":
    main()
