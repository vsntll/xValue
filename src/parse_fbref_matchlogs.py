"""Parse cached FBref all-competitions match-log HTML into one long table.

Browser-free: reads the raw pages ``src/pull_fbref_matchlogs.py`` cached under
``data/raw/fbref/pages/<comp>_<season>_<slug>_<stat>.html`` and turns them into
``data/processed/fbref_team_matchlogs.csv`` — one row per team per match, league +
cup + European + friendly, with the ``Comp`` column mapped to a
``competition_type`` that unions onto ``match_features.csv``.

Every stat page (schedule / shooting / keeper / misc) exposes its table as
``id="matchlogs_for"``; we stitch them on (team, season, date, comp, round,
opponent).

Run:
    py -3.11 src/parse_fbref_matchlogs.py
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_ROOT / "data" / "raw" / "fbref" / "pages"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fbref_team_matchlogs.csv"

# cache filename: <COMP>_<season>_<name-slug>_<stat>.html  e.g. ENG1_2023-24_Arsenal_schedule.html
FNAME_RE = re.compile(
    r"^(?P<comp>[A-Z]{3}\d)_(?P<season>\d{4}-\d{2})_(?P<team>.+)_(?P<stat>[a-z]+)\.html$"
)

STITCH_KEYS = ["team", "season", "Date", "Comp", "Round", "Opponent"]

# FBref `Comp` value -> our competition_type (see docs/modeling_decisions.md).
COMPETITION_TYPE = {
    "Premier League": "league", "Championship": "league",
    "Bundesliga": "league", "2. Bundesliga": "league",
    "La Liga": "league", "Segunda División": "league", "La Liga 2": "league",
    "FA Cup": "domestic_cup", "DFB-Pokal": "domestic_cup", "Copa del Rey": "domestic_cup",
    "EFL Cup": "league_cup",
    "Champions Lg": "european", "Europa Lg": "european",
    "Europa Conf Lg": "european", "Conf Lg": "european",
    "Community Shield": "super_cup", "FA Community Shield": "super_cup",
    "Supercopa de España": "super_cup", "DFL-Supercup": "super_cup",
    "UEFA Super Cup": "super_cup", "Super Cup": "super_cup",
    "Club Friendlies": "friendly", "Friendlies (M)": "friendly",
    "Relegation/Promotion Play-offs": "playoff", "Rel/Pro play-offs": "playoff",
    "Promotion play-offs": "playoff",
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten FBref's MultiIndex stat columns. The identity block sits under a
    'For <Team>' (or Unnamed/empty) super-header -> keep just the leaf; real stat
    groups ('Standard', 'Expected', ...) -> 'Group_Leaf'."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        def name(top, leaf):
            top = str(top)
            if not top or top.startswith(("Unnamed", "For ")):
                return leaf
            return f"{top}_{leaf}".strip("_")
        df.columns = [name(c[0], c[-1]) for c in df.columns]
    return df


def _clean_opponent(s: pd.Series) -> pd.Series:
    """FBref prefixes foreign opponents with a lowercase country code
    ('nl PSV', 'fr Lens'). Strip it."""
    return s.astype("string").str.replace(r"^[a-z]{2,3} ", "", regex=True)


def _parse_result(df: pd.DataFrame) -> pd.DataFrame:
    """Result cells for shootouts read '1 (4)'. Split the shootout score out and
    keep GF/GA as the 90'+ET score."""
    for col in ("GF", "GA"):
        if col not in df.columns:
            continue
        pens = df[col].astype("string").str.extract(r"\((\d+)\)")[0]
        base = df[col].astype("string").str.replace(r"\s*\(\d+\)", "", regex=True)
        df[col] = pd.to_numeric(base, errors="coerce")
        df[f"{col}_pens"] = pd.to_numeric(pens, errors="coerce")
    if {"GF_pens", "GA_pens"}.issubset(df.columns):
        df["went_to_penalties"] = df["GF_pens"].notna() & df["GA_pens"].notna()
    return df


def load_stat(stat: str) -> pd.DataFrame:
    frames = []
    for path in sorted(PAGES_DIR.glob(f"*_{stat}.html")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        html = path.read_text(encoding="utf-8")
        try:
            tbl = pd.read_html(StringIO(html), attrs={"id": "matchlogs_for"})[0]
        except (ValueError, IndexError):
            continue  # club had no log for this stat/season
        tbl = _flatten(tbl)
        tbl = tbl[tbl["Date"].notna() & (tbl["Date"] != "Date")]  # drop repeated headers
        tbl.insert(0, "team", m["team"].replace("-", " "))
        tbl.insert(1, "season", m["season"])
        tbl.insert(2, "src_league", m["comp"])
        frames.append(tbl)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if not PAGES_DIR.exists():
        raise SystemExit(f"no FBref cache at {PAGES_DIR} — run pull_fbref_matchlogs.py first")

    sched = load_stat("schedule")
    if sched.empty:
        raise SystemExit("no schedule pages cached yet")

    sched = _parse_result(sched)
    sched["Opponent"] = _clean_opponent(sched["Opponent"])
    sched["Date"] = pd.to_datetime(sched["Date"], errors="coerce")
    sched["competition_type"] = sched["Comp"].map(COMPETITION_TYPE).fillna("other")
    unmapped = sorted(set(sched.loc[sched["competition_type"] == "other", "Comp"].dropna()))
    if unmapped:
        print(f"unmapped Comp values -> 'other': {unmapped}")

    combined = sched
    for stat in ("shooting", "keeper", "misc"):
        df = load_stat(stat)
        if df.empty:
            print(f"({stat}: no pages cached, skipping)")
            continue
        df["Opponent"] = _clean_opponent(df["Opponent"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        on = [k for k in STITCH_KEYS if k in df.columns and k in combined.columns]
        new = [c for c in df.columns if c not in combined.columns or c in on]
        combined = combined.merge(df[new], on=on, how="left", suffixes=("", f"_{stat}"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    print(f"teams: {combined['team'].nunique()}  seasons: {sorted(combined['season'].unique())}")
    print("\nrows per competition_type:")
    print(combined["competition_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
