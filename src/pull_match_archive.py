"""Step 1 — pull the match archive from football-data.co.uk.

Downloads one CSV per (season, division), trims to the columns we care about
(goals / shots / corners / cards / fouls / result + a slice of odds for the
step 8 benchmark), and concatenates everything into a single match-features table.

Run:
    python src/pull_match_archive.py

Outputs:
    data/raw/football_data/<season>_<div>.csv   one file per download, untouched
    data/processed/match_features.csv           combined + column-trimmed
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk division codes
DIVISIONS = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
}

# Last 6 completed seasons. Codes are the two end-years, e.g. 2020-21 -> "2021".
SEASONS = {
    "2021": "2020-21",
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
}

# Columns to keep when present. Not every season carries every odds column, so
# selection is "keep the intersection with what the file actually has".
IDENTITY_COLS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "Referee"]
RESULT_COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]
MATCH_STAT_COLS = [
    "HS", "AS",      # shots
    "HST", "AST",    # shots on target
    "HC", "AC",      # corners
    "HF", "AF",      # fouls
    "HY", "AY",      # yellow cards
    "HR", "AR",      # red cards
]
ODDS_COLS = [
    "B365H", "B365D", "B365A",        # Bet365 pre-match
    "B365CH", "B365CD", "B365CA",     # Bet365 closing
    "AvgH", "AvgD", "AvgA",           # market average pre-match
    "AvgCH", "AvgCD", "AvgCA",        # market average closing
]
KEEP_COLS = IDENTITY_COLS + RESULT_COLS + MATCH_STAT_COLS + ODDS_COLS

RAW_DIR = Path("data/raw/football_data")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "match_features.csv"


def fetch_csv(season_code: str, div_code: str) -> pd.DataFrame:
    """Download one season/division CSV, cache the raw bytes, return a DataFrame."""
    url = f"{BASE_URL}/{season_code}/{div_code}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw_path = RAW_DIR / f"{season_code}_{div_code}.csv"
    raw_path.write_bytes(resp.content)

    # Older files are Windows-1252, newer ones are UTF-8 with a BOM. Try the
    # BOM-aware decode first and fall back to latin-1 (which never raises).
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(resp.content), encoding=encoding, on_bad_lines="skip"
            )
            break
        except UnicodeDecodeError:
            continue

    df.columns = [c.lstrip("﻿").strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    return df[present], missing


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season_code, season_label in SEASONS.items():
        for div_code, div_label in DIVISIONS.items():
            df = fetch_csv(season_code, div_code)
            trimmed, missing = trim_columns(df)
            trimmed = trimmed.copy()
            trimmed.insert(0, "season", season_label)
            trimmed.insert(1, "league", div_label)
            # Parse the dd/mm/yy or dd/mm/yyyy date to a real timestamp for later
            # time-based splits.
            trimmed["Date"] = pd.to_datetime(
                trimmed["Date"], dayfirst=True, errors="coerce"
            )
            frames.append(trimmed)

            note = f"  missing: {missing}" if missing else ""
            print(
                f"{season_label} {div_label:<14} "
                f"{len(trimmed):>3} matches, {trimmed.shape[1]} cols{note}"
            )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Date", "league"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nwrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    print(f"date range: {combined['Date'].min().date()} .. {combined['Date'].max().date()}")


if __name__ == "__main__":
    main()
