"""Per-player, per-match stats from FotMob's hidden matchDetails API - the
browser-free source for the stats Understat does not carry (shots on target,
fouls, tackles / interceptions / blocks / clearances / recoveries, touches),
so the every-other-day refresh no longer needs a manual FBref scrape for the
current season.

FotMob's `playerStats` block is keyed with stable slugs (`ShotsOnTarget`,
`fouls`, `interceptions` ...), so parsing is just a key lookup. Every match's
rows are committed to the per-season CSVs in data/processed/fotmob/ (see
src/fotmob_store.py) - a re-run, even on a fresh CI runner, only fetches
matches not committed yet. --budget-min caps the fetching time per run.
FotMob's ToS restricts programmatic use; keep volume low.

The same matchDetails payload also carries the matchday squads
(content.lineup -> lineups_<season>.csv, see build_match_lineups.py) and the
goal/red-card timeline (content.matchFacts.events -> events_<season>.csv, see
live_win_prob.py's backtest), saved at zero extra request cost. Matches pulled
before events were captured need the separate --events flag (a deliberate,
bounded re-fetch, not automatic).

Run:  py -3.11 src/pull_fotmob_players.py --current      # season in progress
      py -3.11 src/pull_fotmob_players.py --seasons 2025-26 2026-27
      py -3.11 src/pull_fotmob_players.py --events 2025-26   # backfill event timelines
      py -3.11 src/pull_fotmob_players.py --backfill-passes 2024-25 2025-26 2026-27
Output: data/processed/fotmob/{player_matches,lineups,events}_<season>.csv
        data/processed/fotmob_player_season.csv  (one row per player-team-season)
        data/processed/fotmob_match_meta.csv     (match id -> season/league/date/score)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store as store  # noqa: E402
from fbref_common import current_season  # noqa: E402
from live.schema import normalize_team  # noqa: E402
from net import get_json  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "fotmob_player_season.csv"
BASE = "https://www.fotmob.com/api/data"
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
SLEEP = 0.5
MAX_REQUEST_S = 45  # hard cap on one request, start to last byte (see src/net.py)

LEAGUE_ID = {"ENG1": 47, "GER1": 54, "ESP1": 87, "ITA1": 55, "FRA1": 53}

# FotMob playerStats key -> our column. `value` is taken even for the
# fraction-with-percentage entries (aerials etc.), which is the "won"/"made" count.
STAT_KEYS = {
    "minutes_played": "minutes", "goals": "goals", "assists": "assists",
    "total_shots": "shots", "ShotsOnTarget": "sot",
    "expected_goals": "xg", "expected_goals_non_penalty": "npxg", "expected_assists": "xa",
    "fouls": "fouls", "was_fouled": "fouled",
    "matchstats.headers.tackles": "tackles", "interceptions": "interceptions",
    "shot_blocks": "blocks", "clearances": "clearances", "recoveries": "recoveries",
    "touches": "touches", "touches_opp_box": "touches_box", "dispossessed": "dispossessed",
    "aerials_won": "aerials_won", "chances_created": "chances_created",
    "saves": "saves", "goals_conceded": "goals_conceded",
}
SUM_COLS = [c for c in STAT_KEYS.values() if c != "minutes"] + ["minutes", "matches",
                                                                 "passes_completed", "passes_attempted"]
PASS_KEY = "accurate_passes"  # fractionWithPercentage: {value: completed, total: attempted} -
                              # needs both, unlike every other STAT_KEYS entry (value only)


def _get(path: str, **params) -> object:
    """GET one FotMob endpoint as JSON, hard-capped at MAX_REQUEST_S (src/net.py).
    Raises requests.RequestException - callers skip the match, retry next run."""
    out = get_json(f"{BASE}/{path}", params=params, headers=HDRS, max_s=MAX_REQUEST_S)
    time.sleep(SLEEP)
    return out


def _stat(entry: dict):
    s = entry.get("stat") or {}
    return s.get("value")


def _extract_events(d: dict) -> list[dict]:
    """Goal + red-card timeline from matchFacts.events - each goal's newScore
    is the authoritative running score (own goals already resolved into it;
    the top-level homeScore/awayScore fields on each event are stale and NOT
    reliable). Used by live_win_prob.py's backtest to replay a finished match
    minute-by-minute."""
    mf = (d.get("content") or {}).get("matchFacts") or {}
    ev = (mf.get("events") or {}).get("events") or []
    out = []
    for e in ev:
        t = e.get("type")
        if t == "Goal":
            ns = e.get("newScore") or [None, None]
            out.append({"time": e.get("time"), "type": "Goal",
                        "isHome": bool(e.get("isHome")),
                        "home_score": ns[0], "away_score": ns[1]})
        elif t == "Card" and e.get("card") in ("Red", "YellowRed"):
            out.append({"time": e.get("time"), "type": "Red", "isHome": bool(e.get("isHome"))})
    return out


def _extract_player_rows(d: dict, mid: str) -> list[dict]:
    ps = ((d.get("content") or {}).get("playerStats")) or {}
    if not ps:
        return []
    g = d.get("general") or {}
    date = (g.get("matchTimeUTCDate") or "")[:10]
    rows = []
    for pid, p in ps.items():
        flat = {}
        for grp in p.get("stats", []):
            for _title, entry in (grp.get("stats") or {}).items():
                key = entry.get("key")
                if key == PASS_KEY:
                    s = entry.get("stat") or {}
                    flat.setdefault("passes_completed", s.get("value"))
                    flat.setdefault("passes_attempted", s.get("total"))
                elif key in STAT_KEYS:
                    flat.setdefault(STAT_KEYS[key], _stat(entry))
        if flat.get("minutes") in (None, 0):
            continue
        rows.append({
            "fotmob_id": int(pid), "player": p.get("name"),
            "team": p.get("teamName"), "team_id": p.get("teamId"),
            "date": date, "match_id": f"fotmob:{mid}",
            "position": p.get("usualPosition"),
            **{c: flat.get(c) for c in STAT_KEYS.values()},
            "passes_completed": flat.get("passes_completed"), "passes_attempted": flat.get("passes_attempted"),
        })
    return rows


def _extract_lineup_rows(lineup: dict, mid: str) -> list[dict]:
    """One row per player in each side's matchday squad (content.lineup)."""
    rows = []
    for side_key, is_home in (("homeTeam", 1), ("awayTeam", 0)):
        side = lineup.get(side_key) or {}
        for is_starter, grp in ((1, "starters"), (0, "subs")):
            for p in side.get(grp) or []:
                rows.append({"match_id": str(mid), "is_home": is_home, "team": side.get("name"),
                             "team_id": side.get("id"), "fotmob_id": p.get("id"),
                             "player": p.get("name"), "age": p.get("age"),
                             "position_id": p.get("positionId"), "is_starter": is_starter})
    return rows


def _fetch_match(mid: str, season: str, code: str) -> dict | None:
    """Player, lineup and event rows for one finished match off a single
    matchDetails request; None when FotMob refuses it (retried next run)."""
    try:
        d = _get("matchDetails", matchId=mid)
    except requests.RequestException:
        return None
    lineup = (d.get("content") or {}).get("lineup") or {}
    return {
        "player_matches": [{**r, "season": season, "src_league": code}
                           for r in _extract_player_rows(d, mid)],
        "lineups": _extract_lineup_rows(lineup, mid),
        "events": [{"match_id": mid, **e} for e in _extract_events(d)],
    }


def _fixture_meta(fx: dict, season: str, code: str) -> dict:
    home, away = fx.get("home") or {}, fx.get("away") or {}
    return {"match_id": str(fx["id"]), "season": season, "src_league": code,
            "date": (fx.get("status") or {}).get("utcTime", "")[:10],
            "home": home.get("name"), "away": away.get("name"),
            "home_score": home.get("score"), "away_score": away.get("score")}


def pull_season(code: str, season: str, done: set[str], deadline: float) -> int:
    """Fetch this league-season's finished matches that aren't committed yet
    and upsert them into the per-season CSVs. Returns how many were left over
    because the time budget ran out."""
    start = int(season.split("-")[0])
    try:
        fixtures = _get("fixtures", id=LEAGUE_ID[code], season=f"{start}/{start + 1}")
    except requests.RequestException as e:
        print(f"  {code} {season}: fixtures failed ({e})")
        return 0
    if not isinstance(fixtures, list):
        return 0
    finished = [f for f in fixtures if (f.get("status") or {}).get("finished")]
    if not finished:
        return 0
    meta = pd.DataFrame([_fixture_meta(f, season, code) for f in finished])
    todo = [m for m in meta["match_id"] if m not in done]
    print(f"  {code} {season}: {len(finished)} finished matches ({len(todo)} to fetch)", flush=True)

    got = {k: [] for k in store.KINDS}
    fetched, n_left = [], 0
    for mid in todo:
        if time.monotonic() > deadline:
            n_left += 1
            continue
        res = _fetch_match(mid, season, code)
        if res is None:
            continue
        for k in store.KINDS:
            got[k] += res[k]
        fetched.append(mid)
    for k in store.KINDS:
        store.upsert(k, season, pd.DataFrame(got[k]))
    meta.loc[meta["match_id"].isin(fetched), ["fetched", "events_fetched"]] = 1
    store.upsert_meta(meta)
    return n_left


def _aggregate(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    for c in STAT_KEYS.values():
        pm[c] = pd.to_numeric(pm[c], errors="coerce")
    pm["matches"] = 1
    pm["team_key"] = pm["team"].map(normalize_team)
    agg = (pm.groupby(["season", "src_league", "fotmob_id", "team_key"], as_index=False)
             .agg({**{c: "sum" for c in SUM_COLS},
                   "player": "last", "team": "last", "position": "last", "team_id": "last"}))
    return agg


def backfill_events(seasons: list[str]) -> None:
    """Deliberate, bounded backfill of the goal/red-card timeline for committed
    matches pulled before events were captured (events_fetched != 1) - each
    needs a fresh request. Scoped to specific seasons, per the ToS "keep volume
    low" note above."""
    meta = store.read_meta()
    meta = meta[meta["season"].isin(seasons) & (meta["events_fetched"] != 1)]
    print(f"backfilling match events for {len(meta)} matches ({seasons})")
    n_new = n_fail = 0
    for season, g in meta.groupby("season"):
        rows, ok = [], []
        for mid in g["match_id"]:
            try:
                d = _get("matchDetails", matchId=mid)
            except requests.RequestException:
                n_fail += 1
                continue
            rows += [{"match_id": mid, **e} for e in _extract_events(d)]
            ok.append(mid)
        store.upsert("events", season, pd.DataFrame(rows))
        if ok:
            store.upsert_meta(pd.DataFrame({"match_id": ok, "events_fetched": 1}))
        n_new += len(ok)
    print(f"done: {n_new} newly fetched, {n_fail} failed")


def backfill_player_stats(seasons: list[str]) -> None:
    """Deliberate, bounded re-fetch of committed matches whose player rows
    predate passes_completed/passes_attempted (added for build_player_elo.py's
    role-weighted rating - see PASS_KEY above). Matches that already carry them
    are skipped, so a re-run after an interruption picks up where it left off.
    Scoped to specific seasons, per the ToS "keep volume low" note above."""
    n_new = n_fail = 0
    for season in seasons:
        pm = store.read("player_matches", [season])
        if pm.empty:
            continue
        if "passes_attempted" not in pm:
            pm["passes_attempted"] = None
        has = pm.groupby("match_id")["passes_attempted"].apply(lambda s: s.notna().any())
        todo = has[~has].index.tolist()
        print(f"backfilling pass stats for {len(todo)} matches ({season})")
        code_of = pm.drop_duplicates("match_id").set_index("match_id")["src_league"]
        rows = []
        for mid in todo:
            try:
                d = _get("matchDetails", matchId=mid.removeprefix("fotmob:"))
            except requests.RequestException:
                n_fail += 1
                continue
            rows += [{**r, "season": season, "src_league": code_of[mid]}
                     for r in _extract_player_rows(d, mid.removeprefix("fotmob:"))]
            n_new += 1
        store.upsert("player_matches", season, pd.DataFrame(rows))
    print(f"done: {n_new} newly fetched, {n_fail} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", nargs="+", default=[current_season()])
    ap.add_argument("--current", action="store_true", help="just the season in progress")
    ap.add_argument("--budget-min", type=float, default=20,
                    help="stop fetching new matches after this many minutes (rest next run)")
    ap.add_argument("--events", nargs="+", metavar="SEASON",
                    help="backfill the goal/red-card event timeline for these seasons' "
                         "committed matches instead of the normal player-stats pull")
    ap.add_argument("--backfill-passes", nargs="+", metavar="SEASON",
                    help="backfill passes_completed/passes_attempted for these seasons' "
                         "committed matches instead of the normal player-stats pull")
    args = ap.parse_args()
    if args.events:
        backfill_events(args.events)
        return
    if args.backfill_passes:
        backfill_player_stats(args.backfill_passes)
        return
    seasons = [current_season()] if args.current else args.seasons

    meta = store.read_meta()
    done = set(meta.loc[meta["fetched"] == 1, "match_id"])
    deadline = time.monotonic() + args.budget_min * 60
    n_left = sum(pull_season(code, season, done, deadline)
                 for season in seasons for code in LEAGUE_ID)
    if n_left:
        print(f"\n  fetch budget ({args.budget_min:g} min) spent - {n_left} matches left for next run")

    # season totals are rebuilt from the committed per-match rows, so they always
    # agree with them (and with whatever an earlier run already committed)
    pm = store.read("player_matches", seasons)
    if pm.empty:
        raise SystemExit("no FotMob player data for " + ", ".join(seasons))
    new = _aggregate(pm)
    if OUT.exists():
        prior = pd.read_csv(OUT)
        new = pd.concat([prior[~prior["season"].isin(seasons)], new], ignore_index=True)
    PROC.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(new)} player-team-seasons, "
          f"{new['fotmob_id'].nunique()} players, seasons {sorted(new['season'].unique())})")


if __name__ == "__main__":
    main()
