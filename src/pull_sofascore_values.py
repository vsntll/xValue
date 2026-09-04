"""Current player market values from Sofascore.

Transfermarkt blocks every non-browser client (HTTP 405) and its *history*
endpoint needs a real browser. Sofascore's API (TLS-fingerprint client, no
browser) exposes ``proposedMarketValueRaw`` per player in the squad response -
**current value only, no history**, so this covers the ongoing season.

For each league: current-season standings -> team ids -> ``/team/<id>/players``
(one call per club, ~20 clubs x 3 leagues). ~60 requests total.

Run:  py -3.11 src/pull_sofascore_values.py
      py -3.11 src/pull_sofascore_values.py --season 25/26   # a past season's squads

Output: data/processed/sofascore_values.csv  (player, club, market value EUR, +bio)
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import tls_requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "processed" / "sofascore_values.csv"
API = "https://api.sofascore.com/api/v1"

# our code -> Sofascore unique-tournament id + {season year: season id}
LEAGUES = {
    "ENG1": (17, {"26/27": 96668, "25/26": 76986, "24/25": 61627}),
    "GER1": (35, {"26/27": 97464, "25/26": 77333, "24/25": 63516}),
    "ESP1": (8, {"26/27": 97268, "25/26": 77559, "24/25": 61643}),
}
SLEEP = 0.6


def _get(path: str) -> dict:
    for attempt in range(4):
        r = tls_requests.get(f"{API}/{path}", timeout=25)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    raise RuntimeError(f"429 loop on {path}")


def _season_label(y: str) -> str:
    a, b = y.split("/")
    return f"20{a}-{b}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", default=None, help="e.g. 26/27 (default: current)")
    args = ap.parse_args()

    if args.season:
        year = args.season
    else:
        t = datetime.date.today()
        s = t.year if t.month >= 7 else t.year - 1
        year = f"{str(s)[2:]}/{str(s + 1)[2:]}"

    rows = []
    for code, (utid, seasons) in LEAGUES.items():
        sid = seasons.get(year)
        if not sid:
            print(f"{code}: no season id for {year}")
            continue
        st = _get(f"unique-tournament/{utid}/season/{sid}/standings/total")
        teams = [(r["team"]["name"], r["team"]["id"])
                 for r in st["standings"][0]["rows"]]
        print(f"{code} {year}: {len(teams)} clubs")
        for tname, tid in teams:
            squad = _get(f"team/{tid}/players").get("players", [])
            for entry in squad:
                p = entry["player"]
                mv = (p.get("proposedMarketValueRaw") or {}).get("value")
                cu = p.get("contractUntilTimestamp")
                dob = p.get("dateOfBirthTimestamp")
                rows.append({
                    "season": _season_label(year), "src_league": code,
                    "club": tname, "player_name": p.get("name"),
                    "sofascore_id": p.get("id"),
                    "position": p.get("position"),
                    "dob": datetime.date.fromtimestamp(dob).isoformat() if dob else None,
                    "foot": p.get("preferredFoot"),
                    "height_cm": p.get("height"),
                    "country": (p.get("country") or {}).get("name"),
                    "contract_until": datetime.date.fromtimestamp(cu).isoformat() if cu else None,
                    "market_value_eur": mv,
                })
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out)} players, "
          f"{out['market_value_eur'].notna().sum()} with a value)")
    print(out.groupby("src_league").agg(
        n=("player_name", "size"),
        median_value_m=("market_value_eur", lambda s: round(s.median() / 1e6, 1)),
    ).to_string())


if __name__ == "__main__":
    main()
