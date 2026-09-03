"""Big-5 advanced player stats from the worldfootballR data mirror.

FBref serves its Opta-derived player tables (xG/xAG, progressive passing, tackles,
possession, shot-creation) as empty skeletons to scrapers - see
docs/fbref_ingestion.md. ``github.com/JaseZiv/worldfootballR_data`` publishes them
pre-scraped as ``.rds`` (no browser, no Cloudflare).

Coverage: Premier League / Bundesliga / La Liga (of our six), **Season_End_Year
2021-2023** i.e. seasons 2020-21 .. 2022-23. The mirror stopped updating advanced
stats after 2022-23; 2023-24 onward and the three 2nd tiers wait for API-Football.

Run:  py -3.11 src/pull_wfr_advanced.py

Output:
    data/raw/wfr/big5_player_<category>.rds            raw
    data/processed/wfr_player_advanced.csv             merged, our schema
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wfr"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "wfr_player_advanced.csv"

BASE = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
        "data/fb_big5_advanced_season_stats")

# the categories FBref gated from us (standard is here too - for xG/xAG)
CATEGORIES = ["standard", "shooting", "passing", "passing_types",
              "gca", "defense", "possession", "keepers_adv"]

# mirror Comp -> our src_league code (only the three we cover)
COMP_CODE = {"Premier League": "ENG1", "Bundesliga": "GER1", "La Liga": "ESP1"}
SEASON_END_YEARS = {2021: "2020-21", 2022: "2021-22", 2023: "2022-23"}

# identity columns to keep once (the rest get a category prefix)
ID_OUT = ["season", "src_league", "Player", "Squad", "Born", "Nation", "Pos", "Age",
          "fbref_player_id"]
JOIN_KEYS = ["season", "src_league", "Player", "Squad", "Born"]


def _read_rds(cat: str) -> pd.DataFrame:
    import pyreadr

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / f"big5_player_{cat}.rds"
    if not dst.exists() or dst.stat().st_size < 1000:
        r = requests.get(f"{BASE}/big5_player_{cat}.rds", timeout=120)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return next(iter(pyreadr.read_r(str(dst)).values()))


def _shape(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    df = df[df["Comp"].isin(COMP_CODE) & df["Season_End_Year"].isin(SEASON_END_YEARS)].copy()
    df["season"] = df["Season_End_Year"].map(SEASON_END_YEARS)
    df["src_league"] = df["Comp"].map(COMP_CODE)
    if "Url" in df.columns:
        df["fbref_player_id"] = df["Url"].astype(str).str.extract(r"/players/([0-9a-f]{8})/")
    keep_id = [c for c in ID_OUT if c in df.columns]
    stat_cols = [c for c in df.columns
                 if c not in keep_id + ["Season_End_Year", "Comp", "Url", "Rk"]]
    df = df.rename(columns={c: f"{cat}__{c}" for c in stat_cols})
    return df[keep_id + [f"{cat}__{c}" for c in stat_cols]]


def main() -> None:
    combined = None
    for cat in CATEGORIES:
        try:
            raw = _read_rds(cat)
        except Exception as exc:  # noqa: BLE001
            print(f"  {cat}: skip ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        part = _shape(raw, cat)
        print(f"  {cat}: {len(part)} player-seasons, {part.shape[1]} cols")
        if combined is None:
            combined = part
        else:
            on = [k for k in JOIN_KEYS if k in part.columns and k in combined.columns]
            dup_ids = [c for c in ID_OUT if c not in on and c in part.columns]
            combined = combined.merge(part.drop(columns=dup_ids).drop_duplicates(subset=on),
                                      on=on, how="outer")

    if combined is None:
        raise SystemExit("nothing downloaded")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    print(f"seasons: {sorted(combined['season'].dropna().unique())}  "
          f"leagues: {sorted(combined['src_league'].dropna().unique())}")


if __name__ == "__main__":
    main()
