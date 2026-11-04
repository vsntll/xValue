"""Committed per-match FotMob tables - the pipeline's only record of what has
been pulled. No fetch cache: a match is "done" once its rows are in these CSVs,
so a fresh CI runner skips everything already committed and only requests new
matches. Split per season so a refresh only rewrites the live season's files.

  data/processed/fotmob/player_matches_<season>.csv  one row per player-match
  data/processed/fotmob/lineups_<season>.csv         one row per player in a matchday squad
  data/processed/fotmob/events_<season>.csv          goal / red-card timeline
  data/processed/fotmob_match_meta.csv               one row per match, with
      fetched / events_fetched flags (a match FotMob has no player stats for
      still counts as done, so it isn't re-requested every run)
  data/processed/fotmob_intl_match_meta.csv          the same for internationals
      (their player rows: fotmob_intl_player_matches.csv)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DIR = PROC / "fotmob"
MATCH_META = PROC / "fotmob_match_meta.csv"
INTL_META = PROC / "fotmob_intl_match_meta.csv"
KINDS = ("player_matches", "lineups", "events")


def path(kind: str, season: str) -> Path:
    return DIR / f"{kind}_{season}.csv"


def read(kind: str, seasons: list[str] | None = None) -> pd.DataFrame:
    """All rows of `kind`, optionally limited to `seasons`."""
    files = sorted(DIR.glob(f"{kind}_*.csv")) if seasons is None else \
        [path(kind, s) for s in seasons if path(kind, s).exists()]
    frames = [pd.read_csv(f, dtype={"match_id": str}, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def upsert(kind: str, season: str, rows: pd.DataFrame) -> None:
    """Replace `rows`' matches in the season file, keep every other match."""
    if rows.empty:
        return
    DIR.mkdir(parents=True, exist_ok=True)
    f = path(kind, season)
    rows = rows.assign(match_id=rows["match_id"].astype(str))
    if f.exists():
        prior = pd.read_csv(f, dtype={"match_id": str}, low_memory=False)
        rows = pd.concat([prior[~prior["match_id"].isin(rows["match_id"])], rows], ignore_index=True)
    rows.sort_values(["match_id"], kind="stable").to_csv(f, index=False)


def read_meta(f: Path = MATCH_META) -> pd.DataFrame:
    if not f.exists():
        return pd.DataFrame(columns=["match_id", "date", "fetched", "events_fetched"])
    return pd.read_csv(f, dtype={"match_id": str})


def upsert_meta(rows: pd.DataFrame, f: Path = MATCH_META) -> pd.DataFrame:
    """Merge match rows into a match-meta file by match_id; a column the new
    rows don't carry (or carry as NaN) keeps its committed value."""
    prior = read_meta(f)
    rows = rows.assign(match_id=rows["match_id"].astype(str)).set_index("match_id")
    out = prior.set_index("match_id")
    out = rows.combine_first(out) if len(out) else rows
    out = out.reset_index()
    for c in ("home_score", "away_score", "fetched", "events_fetched"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out.to_csv(f, index=False)
    return out
