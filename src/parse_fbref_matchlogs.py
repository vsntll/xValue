"""Parse cached FBref all-competitions match-log HTML into one long table.

Browser-free: reads the raw pages that ``src/pull_fbref_matchlogs.py`` (via
soccerdata) cached under ``data/soccerdata/data/FBref/matchlogs_*.html`` and turns
them into ``data/processed/fbref_team_matchlogs.csv`` — one row per team per
match, league + cup + European + friendly, with the ``Comp`` column mapped to a
``competition_type`` that unions onto ``match_features.csv``.

Every stat page (schedule / shooting / keeper / misc) exposes its table as
``id="matchlogs_for"``; we stitch them on (team, season, date, comp, round,
opponent).

Run:
    python src/parse_fbref_matchlogs.py
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "soccerdata" / "data" / "FBref"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fbref_team_matchlogs.csv"

# soccerdata cache filename: matchlogs_<Team Name>_<seasoncode>_<stat>.html
FNAME_RE = re.compile(r"^matchlogs_(?P<team>.+)_(?P<season>\d{4})_(?P<stat>[a-z]+)\.html$")

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
    "Community Shield": "super_cup", "Supercopa de España": "super_cup",
    "DFL-Supercup": "super_cup", "UEFA Super Cup": "super_cup",
    "Club Friendlies": "friendly", "Friendlies (M)": "friendly",
    "Relegation/Promotion Play-offs": "playoff",
}


def _season_label(code: str) -> str:
    """'2324' -> '2023-24'."""
    return f"20{code[:2]}-{code[2:]}"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[-1] if (not c[0] or str(c[0]).startswith("Unnamed"))
            else "_".join(str(p) for p in c).strip("_")
            for c in df.columns
        ]
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
    for path in sorted(CACHE_DIR.glob(f"matchlogs_*_{stat}.html")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        html = path.read_text(encoding="utf-8")
        try:
            tbl = pd.read_html(StringIO(html), attrs={"id": "matchlogs_for"})[0]
        except (ValueError, IndexError):
            print(f"  no table in {path.name}")
            continue
        tbl = _flatten(tbl)
        tbl = tbl[tbl["Date"].notna() & (tbl["Date"] != "Date")]  # drop repeated headers
        tbl.insert(0, "team", m["team"])
        tbl.insert(1, "season", _season_label(m["season"]))
        frames.append(tbl)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if not CACHE_DIR.exists():
        raise SystemExit(f"no FBref cache at {CACHE_DIR} — run pull_fbref_matchlogs.py first")

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
