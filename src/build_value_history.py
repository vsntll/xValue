"""Build a long market-value history for the value model's lag features.

The value model is dominated by a player's *previous-season* market value, so its
accuracy is capped by how often that lag is populated. The training seasons only
had ~54% coverage because:
  - 2020-21 rows need a 2019-20 value we never pulled;
  - a cross-league move (Ligue 1 / Serie A -> one of our leagues) broke the join;
  - the 2022-23 worldfootballR mirror is thin.

This script unions every value we can get into one lookup keyed by
(player_key, season) - and crucially pulls the mirror for **all big-5 leagues,
season_start_year 2015..2022**, not just our three leagues / three seasons:

    data/raw/tm/big5_player_vals.rds     worldfootballR mirror (2015-22, big-5)
    data/processed/tm_values_scraped.csv nodriver scrape        (2023-26, our 3)
    data/processed/sofascore_values.csv  Sofascore              (2026-27, our 3)
    data/processed/fbref_player_season_stats.csv  final labelled values (all fills)

Output: data/processed/value_history.csv  -  player_key, season, market_value_eur
(one row per player-season, the max when sources disagree).

Run:  py -3.11 src/build_value_history.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live.schema import deaccent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RDS = ROOT / "data" / "raw" / "tm" / "big5_player_vals.rds"
OUT = PROC / "value_history.csv"
RDS_URL = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
           "data/tm_player_vals/big5_player_vals.rds")

MIRROR_FROM = 2015  # season_start_year; earlier values are too stale to help


def _key(s) -> str:
    """Player join key: bare-ascii lowercase, no punctuation - matches the
    _norm_name(player_slug) key used elsewhere in the pipeline."""
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def _from_mirror() -> pd.DataFrame:
    if not RDS.exists() or RDS.stat().st_size < 10_000:
        RDS.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(RDS_URL, timeout=120)
        r.raise_for_status()
        RDS.write_bytes(r.content)
    import pyreadr
    df = next(iter(pyreadr.read_r(str(RDS)).values()))
    df = df[df["season_start_year"] >= MIRROR_FROM].copy()
    slug = df["player_url"].astype(str).str.extract(
        r"transfermarkt\.com/([a-z0-9-]+)/profil")[0]
    name = slug.str.replace("-", " ").fillna(df["player_name"])
    y = df["season_start_year"].astype(int)
    return pd.DataFrame({
        "player_key": name.map(_key),
        "season": y.astype(str) + "-" + (y + 1).astype(str).str[-2:],
        "market_value_eur": pd.to_numeric(df["player_market_value_euro"], errors="coerce"),
    })


def _simple(path: Path, namecol: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["player_key", "season", "market_value_eur"])
    d = pd.read_csv(path)
    return pd.DataFrame({
        "player_key": d[namecol].map(_key),
        "season": d["season"],
        "market_value_eur": pd.to_numeric(d["market_value_eur"], errors="coerce"),
    })


def main() -> None:
    parts = [
        _from_mirror(),
        _simple(PROC / "tm_values_scraped.csv", "player_name"),
        _simple(PROC / "sofascore_values.csv", "player_name"),
    ]
    # the final labelled column carries every fuzzy fill from the parser
    fb = PROC / "fbref_player_season_stats.csv"
    if fb.exists():
        d = pd.read_csv(fb, low_memory=False)
        d = d[d["market_value_eur"].notna()]
        parts.append(pd.DataFrame({
            "player_key": d["player_slug"].map(_key),
            "season": d["season"],
            "market_value_eur": pd.to_numeric(d["market_value_eur"], errors="coerce"),
        }))

    hist = pd.concat(parts, ignore_index=True)
    hist = hist[(hist["player_key"] != "") & hist["market_value_eur"].notna()]
    hist = (hist.groupby(["player_key", "season"], as_index=False)["market_value_eur"]
                .max())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(hist)} player-seasons, "
          f"{hist['player_key'].nunique()} players)")
    print(hist.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
