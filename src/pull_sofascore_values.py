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
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

# our code -> Sofascore unique-tournament id + {season year: season id}
LEAGUES = {
    "ENG1": (17, {"26/27": 96668, "25/26": 76986, "24/25": 61627}),
    "GER1": (35, {"26/27": 97464, "25/26": 77333, "24/25": 63516}),
    "ESP1": (8, {"26/27": 97268, "25/26": 77559, "24/25": 61643}),
}
SLEEP = 0.6


class Blocked(Exception):
    """Sofascore refused us (403/429/5xx) after retries - usually an IP block on
    datacenter ranges (GitHub Actions, most VPS). Locally it's fine."""


def _get(path: str) -> dict:
    last = None
    for attempt in range(5):
        try:
            r = tls_requests.get(f"{API}/{path}", headers=HEADERS, timeout=25)
        except Exception as exc:  # noqa: BLE001 - transport hiccup, retry
            last = exc
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in (403, 429) or r.status_code >= 500:
            last = f"{r.status_code} on {path}"
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(SLEEP)
        return r.json()
    raise Blocked(str(last))


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
    blocked = []
    for code, (utid, seasons) in LEAGUES.items():
        sid = seasons.get(year)
        if not sid:
            print(f"{code}: no season id for {year}")
            continue
        try:
            st = _get(f"unique-tournament/{utid}/season/{sid}/standings/total")
            teams = [(r["team"]["name"], r["team"]["id"])
                     for r in st["standings"][0]["rows"]]
        except Blocked as exc:
            print(f"{code} {year}: BLOCKED ({exc}) - skipping this league")
            blocked.append(code)
            continue
        print(f"{code} {year}: {len(teams)} clubs")
        for tname, tid in teams:
            try:
                squad = _get(f"team/{tid}/players").get("players", [])
            except Blocked as exc:
                print(f"  {tname}: BLOCKED ({exc}) - skipping")
                continue
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
    # Sofascore IP-blocks datacenter ranges, so a CI run often gets little or
    # nothing - never clobber the last good file with a partial/empty scrape.
    if OUT.exists() and (blocked or len(out) < 1200):
        n_prev = len(pd.read_csv(OUT))
        print(f"\nonly got {len(out)} rows (blocked: {blocked or 'none'}) - keeping "
              f"the existing {OUT.name} ({n_prev} rows) unchanged.")
        return
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
