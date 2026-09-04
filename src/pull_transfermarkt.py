"""Step 3 - Transfermarkt market values (the value model's target).

Direct Transfermarkt scraping is blocked here (HTTP 405 to `curl_cffi` /
`tls_requests`; a GDPR consent wall behind `nodriver`). The worldfootballR mirror
publishes TM values pre-scraped as `.rds` - big-5 leagues, **season_start_year
2010..2022** (i.e. up to 2022-23). That's three of our seasons (2020-21..2022-23)
- enough to train the value model and hold out a season. 2023-24 onward would
need a browser scrape with consent-wall handling (not built).

Run:  py -3.11 src/pull_transfermarkt.py

Output:
    data/raw/tm/big5_player_vals.rds        raw
    data/processed/tm_player_values.csv     one row per player-club-season, EUR value
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "tm"
OUT = PROJECT_ROOT / "data" / "processed" / "tm_player_values.csv"
URL = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
       "data/tm_player_vals/big5_player_vals.rds")

COMP_CODE = {"Premier League": "ENG1", "Bundesliga": "GER1", "LaLiga": "ESP1"}
SEASON = {2020: "2020-21", 2021: "2021-22", 2022: "2022-23"}


def main() -> None:
    import pyreadr

    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / "big5_player_vals.rds"
    if not dst.exists() or dst.stat().st_size < 10_000:
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        dst.write_bytes(r.content)

    df = next(iter(pyreadr.read_r(str(dst)).values()))
    df = df[df["comp_name"].isin(COMP_CODE) & df["season_start_year"].isin(SEASON)].copy()
    df["src_league"] = df["comp_name"].map(COMP_CODE)
    df["season"] = df["season_start_year"].map(SEASON)
    df["tm_player_id"] = df["player_url"].astype(str).str.extract(r"/spieler/(\d+)")
    # TM's served names are mojibake'd for accents; the URL slug is clean
    slug = df["player_url"].astype(str).str.extract(r"transfermarkt\.com/([a-z0-9-]+)/profil")[0]
    df["player_name"] = slug.str.replace("-", " ").fillna(df["player_name"])

    keep = ["season", "src_league", "squad", "player_name", "tm_player_id",
            "player_position", "player_dob", "player_age", "player_nationality",
            "player_height_mtrs", "player_foot", "date_joined", "joined_from",
            "contract_expiry", "player_market_value_euro", "player_url"]
    out = df[keep].rename(columns={"player_market_value_euro": "market_value_eur"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"wrote {OUT}  ({len(out)} rows)")
    print(out.groupby(["season", "src_league"]).agg(
        players=("player_name", "size"),
        median_value_m=("market_value_eur", lambda s: round(s.median() / 1e6, 1)),
    ).to_string())


if __name__ == "__main__":
    main()
