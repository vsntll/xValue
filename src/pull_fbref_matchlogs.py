"""Step 1c + step 2 groundwork — pull per-team all-competitions match logs from
FBref via soccerdata.

FBref's team "Scores & Fixtures" page, scoped to all competitions, lists every
fixture a team played in a season — league, domestic cup, league cup, European
competition, super cup, friendly — each row tagged with a ``comp`` column.
``soccerdata.FBref.read_team_match_stats()`` targets that ``/matchlogs/all_comps/``
URL. This module wraps it with:

* an in-project soccerdata cache (``SOCCERDATA_DIR`` -> ``data/soccerdata``) so
  every scraped page is visible under the project tree, never in the home dir;
* **per (league, season, stat-type) chunking** — a browser/driver death (common
  when scraping FBref through seleniumbase-UC + Cloudflare) loses only the
  current chunk, and re-running resumes from the page cache;
* a combined long table (one row per team per match) for the downstream
  ``build_match_features`` union.

The scrape needs a real browser for the Cloudflare challenge. Point ``--browser``
at a Chrome / Chrome-for-Testing ``chrome.exe``. Headless CANNOT solve a fresh
Cloudflare challenge (seleniumbase disables its GUI solver in headless mode), so
the default is a visible window — don't fight it for mouse focus while it runs.

Run:
    python src/pull_fbref_matchlogs.py                       # everything, resumable
    python src/pull_fbref_matchlogs.py --leagues "ENG-Premier League" --seasons 2023-24
    python src/pull_fbref_matchlogs.py --combine-only        # rebuild table from cache

Outputs (all under the project tree):
    data/soccerdata/data/FBref/matchlogs_*.html    raw pages (soccerdata-managed)
    data/raw/fbref/matchlogs_<stat>.parquet        one per stat type (accumulated)
    data/raw/fbref/scrape_progress.json            per-chunk status for resume
    data/processed/fbref_team_matchlogs.csv        combined long table
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep every scraped byte inside the project (see the "data stays in project"
# rule). Must be set before soccerdata is imported.
os.environ.setdefault("SOCCERDATA_DIR", str(PROJECT_ROOT / "data" / "soccerdata"))
# Pages already on disk never expire — this is an archive pull, not a refresh of
# volatile data. (Re-scrape a season by deleting its cached pages.)
os.environ.setdefault("SOCCERDATA_MAXAGE", "3650")

DEFAULT_BROWSER = Path("C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe")

# Top two tiers of each country. Second-tier keys resolve via the custom league
# dict at data/soccerdata/config/league_dict.json.
DEFAULT_LEAGUES = [
    "ENG-Premier League",
    "ENG-Championship",
    "GER-Bundesliga",
    "GER-2. Bundesliga",
    "ESP-La Liga",
    "ESP-La Liga 2",
]
DEFAULT_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]

# schedule = comp/round/venue/result/GF/GA/opponent/poss/xG/xGA
# shooting = sh/sot/dist/fk/pk/pkatt
# keeper   = GK saves/PSxG/etc.
# misc     = crdy/crdr/fls/fld/off/crs/int/og/pkwon/pkcon
STAT_TYPES = ["schedule", "shooting", "keeper", "misc"]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fbref"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROGRESS_PATH = RAW_DIR / "scrape_progress.json"
OUT_PATH = PROCESSED_DIR / "fbref_team_matchlogs.csv"


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, sort_keys=True))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """soccerdata returns MultiIndex columns for stat pages; flatten to
    ``group_stat`` (dropping the ``Unnamed``/empty top level)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[-1] if (not c[0] or str(c[0]).startswith("Unnamed"))
            else "_".join(str(p) for p in c).strip("_")
            for c in df.columns
        ]
    return df


def pull(leagues: list[str], seasons: list[str], stats: list[str],
         browser: str | None, headless: bool, retries: int) -> None:
    """Scrape one (league, season, stat) chunk at a time, appending each chunk to
    its per-stat parquet. Resumable: cached pages and completed chunks are
    skipped."""
    import soccerdata as sd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()

    for stat in stats:
        parts: list[pd.DataFrame] = []
        parquet_path = RAW_DIR / f"matchlogs_{stat}.parquet"
        if parquet_path.exists():
            parts.append(pd.read_parquet(parquet_path))

        for league in leagues:
            for season in seasons:
                key = f"{stat}|{league}|{season}"
                if progress.get(key) == "done":
                    print(f"  skip (done): {key}")
                    continue

                for attempt in range(1, retries + 1):
                    try:
                        fb = sd.FBref(
                            leagues=[league],
                            seasons=[season],
                            path_to_browser=browser,
                            headless=headless,
                        )
                        df = fb.read_team_match_stats(stat_type=stat)
                        df = _flatten_columns(df.reset_index())
                        df.insert(0, "src_league", league)
                        df.insert(1, "src_season", season)
                        parts.append(df)
                        pd.concat(parts, ignore_index=True).to_parquet(parquet_path)
                        progress[key] = "done"
                        _save_progress(progress)
                        print(f"  ok: {key}  (+{len(df)} team-match rows)")
                        break
                    except Exception as exc:  # noqa: BLE001
                        print(f"  FAIL {attempt}/{retries}: {key} -- {type(exc).__name__}: "
                              f"{str(exc)[:160]}")
                        progress[key] = f"error: {type(exc).__name__}"
                        _save_progress(progress)
                        time.sleep(5 * attempt)
                else:
                    print(f"  giving up on {key} this run; re-run to retry")


def combine() -> pd.DataFrame:
    """Join the per-stat parquets into one long per-team-match table."""
    frames: dict[str, pd.DataFrame] = {}
    for stat in STAT_TYPES:
        p = RAW_DIR / f"matchlogs_{stat}.parquet"
        if p.exists():
            frames[stat] = pd.read_parquet(p)

    if "schedule" not in frames:
        raise SystemExit("no schedule parquet yet — run the scrape first")

    combined = frames["schedule"]
    key_cols = [c for c in ["src_league", "src_season", "team", "date", "game",
                            "opponent", "comp", "round"]
                if c in combined.columns]

    for stat, df in frames.items():
        if stat == "schedule":
            continue
        on = [c for c in key_cols if c in df.columns]
        new_cols = [c for c in df.columns if c not in combined.columns or c in on]
        combined = combined.merge(df[new_cols], on=on, how="left",
                                  suffixes=("", f"_{stat}"))
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--stats", nargs="+", default=STAT_TYPES, choices=STAT_TYPES)
    ap.add_argument("--browser", default=str(DEFAULT_BROWSER),
                    help="path to chrome.exe (needs a real browser for Cloudflare)")
    ap.add_argument("--headless", action="store_true",
                    help="run headless (WARNING: cannot solve a fresh Cloudflare challenge)")
    ap.add_argument("--retries", type=int, default=3,
                    help="attempts per (league, season, stat) chunk before moving on")
    ap.add_argument("--combine-only", action="store_true",
                    help="skip scraping; rebuild the combined table from parquet")
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        browser = args.browser if args.browser and Path(args.browser).exists() else None
        if browser is None:
            print(f"WARNING: browser not found at {args.browser} — letting seleniumbase "
                  f"locate one")
        pull(args.leagues, args.seasons, args.stats, browser, args.headless, args.retries)

    combined = combine()
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    for col in ("comp", "Comp"):
        if col in combined.columns:
            print(f"\nrows per competition ({col}):")
            print(combined[col].value_counts().to_string())
            break


if __name__ == "__main__":
    main()
