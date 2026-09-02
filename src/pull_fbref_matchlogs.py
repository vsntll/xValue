"""Step 1 (cups) + step 2 groundwork — pull per-team all-competitions match logs
from FBref via soccerdata.

FBref's team "Scores & Fixtures" page, scoped to all competitions, lists every
fixture a team played in a season — league, domestic cup, league cup, European
competition, super cup, friendly — each row tagged with a `Comp` column.
`soccerdata.FBref.read_team_match_stats()` already targets that
`/matchlogs/all_comps/` URL, so this module is a thin wrapper: pull each stat
type, cache it, and stitch the pieces into one long table (one row per team per
match).

The per-match table is deduped into `match_features.csv` rows by
`src/build_match_features.py`.

Prereq: a working browser for soccerdata (Cloudflare needs a real one). See
docs/fbref_ingestion.md. Point `--browser` at chrome.exe.

Run (example):
    python src/pull_fbref_matchlogs.py --browser "C:/Users/avasa/chrome-for-testing/chrome-win64/chrome.exe"
    python src/pull_fbref_matchlogs.py --leagues "ENG-Premier League" --seasons 2024-25

Outputs:
    <soccerdata cache>/FBref/matchlogs_*.html      raw pages (soccerdata-managed)
    data/raw/fbref/matchlogs_<stat>.parquet        one per stat type
    data/processed/fbref_team_matchlogs.csv         combined long table
"""

from __future__ import annotations

# NOTE: soccerdata's exact index/column names for read_team_match_stats have not
# yet been verified on a live run (scraping was blocked at authoring time — see
# docs/fbref_ingestion.md). The join keys in combine() and the MultiIndex
# flattening may need adjustment after the first successful pull.

import argparse
from pathlib import Path

import pandas as pd

# Top two tiers of each country. Second-tier keys come from the custom league
# dict at ~/soccerdata/config/league_dict.json (created during setup).
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

# schedule = Comp/Round/Venue/Result/GF/GA/Opponent/Poss/xG/xGA
# shooting = Sh/SoT/Dist/FK/PK/PKatt
# misc     = CrdY/CrdR/Fls/Fld/Off/Crs/Int/OG/PKwon/PKcon
STAT_TYPES = ["schedule", "shooting", "misc", "keeper"]

RAW_DIR = Path("data/raw/fbref")
PROCESSED_DIR = Path("data/processed")
OUT_PATH = PROCESSED_DIR / "fbref_team_matchlogs.csv"


def pull(leagues: list[str], seasons: list[str], stats: list[str],
         browser: str | None, headless: bool) -> dict[str, pd.DataFrame]:
    import soccerdata as sd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"leagues": leagues, "seasons": seasons}
    if browser:
        kwargs["path_to_browser"] = browser
    kwargs["headless"] = headless
    fb = sd.FBref(**kwargs)

    frames: dict[str, pd.DataFrame] = {}
    for stat in stats:
        print(f"\n=== {stat} ===")
        df = fb.read_team_match_stats(stat_type=stat)
        # soccerdata returns a MultiIndex column frame for stat pages; flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                c[-1] if c[0].startswith("Unnamed") or c[0] == "" else "_".join(c).strip("_")
                for c in df.columns
            ]
        out = RAW_DIR / f"matchlogs_{stat}.parquet"
        df.to_parquet(out)
        frames[stat] = df
        print(f"  {len(df)} team-match rows -> {out}")
    return frames


def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the stat frames on (league, season, team, date, game) into one long
    per-team-match table."""
    base = frames["schedule"].reset_index()
    # Identify the join keys soccerdata used in the index.
    key_cols = [c for c in ["league", "season", "team", "date", "game", "opponent"]
                if c in base.columns]

    combined = base
    for stat, df in frames.items():
        if stat == "schedule":
            continue
        right = df.reset_index()
        add_cols = [c for c in right.columns if c not in combined.columns or c in key_cols]
        on = [c for c in key_cols if c in right.columns]
        combined = combined.merge(right[add_cols], on=on, how="left", suffixes=("", f"_{stat}"))

    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--stats", nargs="+", default=STAT_TYPES, choices=STAT_TYPES)
    ap.add_argument("--browser", default=None, help="path to chrome.exe")
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--combine-only", action="store_true",
                    help="skip download, rebuild the combined table from cached parquet")
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.combine_only:
        frames = {s: pd.read_parquet(RAW_DIR / f"matchlogs_{s}.parquet")
                  for s in args.stats if (RAW_DIR / f"matchlogs_{s}.parquet").exists()}
    else:
        frames = pull(args.leagues, args.seasons, args.stats, args.browser, args.headless)

    combined = combine(frames)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    if "Comp" in combined.columns:
        print("\nrows per competition:")
        print(combined["Comp"].value_counts().to_string())


if __name__ == "__main__":
    main()
