"""Per-player, per-match box scores for senior men's internationals, from the
same FotMob matchDetails payload src/pull_fotmob_players.py parses for club
matches (same STAT_KEYS, same _extract_player_rows) - so an international
appearance carries the same xG / xA / defensive / passing columns and can feed
build_player_elo.py exactly like a club one.

Every senior men's competition FotMob lists (INTL_LEAGUES): World Cup, the
continental finals, every confederation's qualifiers, the Nations Leagues, and
friendlies - FotMob only exposes the current friendlies season (2026 here), so
older friendlies are simply absent rather than guessed at. Team names are
aliased onto international_results.csv's spellings (TEAM_ALIASES) so
build_national_elo.py's ratings join on them.

Rows carry the player's FotMob id and national team - no club. build_player_elo.py
links a player to his Understat identity through the fotmob_id his club rows
already carry.

No fetch cache: a match is done once it's in the committed output CSV or marked
fetched in fotmob_intl_match_meta.csv (which also records matches FotMob has no
player data for, plus every match's teams and score), so a fresh CI runner only
requests new matches; new rows are merged into the CSV. --budget-min caps the
time spent fetching so the step can never run the job into its timeout.
FotMob's ToS restricts programmatic use; keep volume low.

Run:  py -3.11 src/pull_fotmob_internationals.py                 # since SINCE
      py -3.11 src/pull_fotmob_internationals.py --since 2025-07-01
Output: data/processed/fotmob_intl_player_matches.csv  (one row per player-match)
        data/processed/fotmob_intl_match_meta.csv      (one row per match: teams, score, fetched)
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
from pull_fotmob_players import _extract_player_rows, _get  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "fotmob_intl_player_matches.csv"
SINCE = "2024-07-01"  # build_player_elo.py's window start (2024-25)

# FotMob league id -> (name, "competitive" | "friendly")
INTL_LEAGUES = {
    77: ("World Cup", "competitive"), 50: ("EURO", "competitive"),
    44: ("Copa America", "competitive"), 289: ("Africa Cup of Nations", "competitive"),
    290: ("Asian Cup", "competitive"), 298: ("CONCACAF Gold Cup", "competitive"),
    10304: ("Finalissima", "competitive"),
    9806: ("UEFA Nations League A", "competitive"), 9807: ("UEFA Nations League B", "competitive"),
    9808: ("UEFA Nations League C", "competitive"), 9809: ("UEFA Nations League D", "competitive"),
    9821: ("CONCACAF Nations League", "competitive"),
    10607: ("EURO Qualification", "competitive"), 10608: ("AFCON Qualification", "competitive"),
    10195: ("WC Qualification UEFA", "competitive"), 10196: ("WC Qualification CAF", "competitive"),
    10197: ("WC Qualification AFC", "competitive"), 10198: ("WC Qualification CONCACAF", "competitive"),
    10199: ("WC Qualification CONMEBOL", "competitive"), 10200: ("WC Qualification OFC", "competitive"),
    10201: ("WC Qualification Inter-confederation", "competitive"),
    114: ("Friendlies", "friendly"),
}

# FotMob spelling -> international_results.csv spelling (the rest already agree)
TEAM_ALIASES = {
    "Curacao": "Curaçao", "Czechia": "Czech Republic", "Ireland": "Republic of Ireland",
    "Macao": "Macau", "Saint Vincent and The Grenadines": "Saint Vincent and the Grenadines",
    "Sao Tome and Principe": "São Tomé and Príncipe", "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Turkiye": "Turkey", "U.S. Virgin Islands": "United States Virgin Islands",
    "UAE": "United Arab Emirates", "USA": "United States",
}


def _team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _seasons(lid: int, since: str) -> list[str]:
    """FotMob season labels ("2026", "2024/2025") that can hold matches on/after `since`."""
    d = _get("leagues", id=lid)
    first_year = int(since[:4])
    return [s for s in (d.get("allAvailableSeasons") or []) if int(s.split("/")[-1]) >= first_year]


def _match_rows(fx: dict, comp: str, comp_type: str) -> list[dict] | None:
    """Player rows for one finished international; None when FotMob refuses
    the request (retried next run)."""
    mid = str(fx["id"])
    try:
        d = _get("matchDetails", matchId=mid)
    except requests.RequestException:
        return None
    home, away = fx["home"], fx["away"]
    side = {str(home["id"]): (_team(home["name"]), _team(away["name"]), 1),
            str(away["id"]): (_team(away["name"]), _team(home["name"]), 0)}
    rows = []
    for r in _extract_player_rows(d, mid):
        team, opp, is_home = side.get(str(r["team_id"]), (None, None, None))
        if team is None:
            continue
        rows.append({**r, "team": team, "opponent": opp, "is_home": is_home,
                     "competition": comp, "comp_type": comp_type})
    return rows


def _fixture_meta(fx: dict, comp: str) -> dict:
    home, away = fx.get("home") or {}, fx.get("away") or {}
    return {"match_id": str(fx["id"]), "date": ((fx.get("status") or {}).get("utcTime") or "")[:10],
            "competition": comp, "home": _team(home.get("name")), "away": _team(away.get("name")),
            "home_score": home.get("score"), "away_score": away.get("score")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default=SINCE, help="earliest match date (YYYY-MM-DD)")
    ap.add_argument("--budget-min", type=float, default=15,
                    help="stop fetching new matches after this many minutes (rest next run)")
    args = ap.parse_args()
    deadline = time.monotonic() + args.budget_min * 60

    prev = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    meta = store.read_meta(store.INTL_META)
    done = set(meta.loc[meta["fetched"] == 1, "match_id"])
    if len(prev):
        done |= set(prev["match_id"].astype(str).str.removeprefix("fotmob:"))
    rows, metas, fetched, n_left = [], [], [], 0
    for lid, (comp, comp_type) in INTL_LEAGUES.items():
        try:
            seasons = _seasons(lid, args.since)
        except requests.RequestException as e:
            print(f"  {comp}: league lookup failed ({e})")
            continue
        for season in seasons:
            try:
                fixtures = _get("fixtures", id=lid, season=season)
            except requests.RequestException as e:
                print(f"  {comp} {season}: fixtures failed ({e})")
                continue
            finished = [f for f in (fixtures if isinstance(fixtures, list) else [])
                        if (f.get("status") or {}).get("finished")
                        and ((f.get("status") or {}).get("utcTime") or "")[:10] >= args.since]
            metas += [_fixture_meta(f, comp) for f in finished]
            todo = [f for f in finished if str(f["id"]) not in done]
            print(f"  {comp} {season}: {len(finished)} finished matches ({len(todo)} to fetch)")
            for fx in todo:
                if time.monotonic() > deadline:
                    n_left += 1
                    continue
                got = _match_rows(fx, comp, comp_type)
                if got is not None:
                    rows += got
                    fetched.append(str(fx["id"]))

    if n_left:
        print(f"\n  fetch budget ({args.budget_min:g} min) spent - {n_left} matches left for next run")
    if metas:
        meta = pd.DataFrame(metas).drop_duplicates("match_id")
        meta.loc[meta["match_id"].isin(done | set(fetched)), "fetched"] = 1
        store.upsert_meta(meta, store.INTL_META)
    if not rows and not len(prev):
        raise SystemExit("no FotMob international player data pulled")
    out = (pd.concat([pd.DataFrame(rows), prev], ignore_index=True)
           .drop_duplicates(subset=["match_id", "fotmob_id"]))
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} player-matches, {out['match_id'].nunique()} matches, "
          f"{out['fotmob_id'].nunique()} players, {out['date'].min()} -> {out['date'].max()})")


if __name__ == "__main__":
    main()
