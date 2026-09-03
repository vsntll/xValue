"""Parse cached FBref player season-stats HTML into one wide table.

Browser-free: reads ``data/raw/fbref/player_stats/<COMP>_<season>_<category>.html``
and joins the ~11 stat categories into
``data/processed/fbref_player_season_stats.csv`` - one row per (player, squad,
season), with per-90 and totals for the value model.

A player who moved mid-season shows once per club stint (FBref splits them), plus
a league total row (Squad ends in ``2 Squads`` etc.) which we drop.

Run:
    py -3.11 src/parse_fbref_player_stats.py
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = PROJECT_ROOT / "data" / "raw" / "fbref" / "player_stats"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "fbref_player_season_stats.csv"

FNAME_RE = re.compile(r"^(?P<comp>[A-Z]{3}\d)_(?P<season>\d{4}-\d{2})_(?P<cat>[a-z_]+)\.html$")

CATEGORY_TABLE = {
    "standard": "stats_standard", "shooting": "stats_shooting",
    "passing": "stats_passing", "passing_types": "stats_passing_types",
    "gca": "stats_gca", "defense": "stats_defense", "possession": "stats_possession",
    "playing_time": "stats_playing_time", "misc": "stats_misc",
    "keeper": "stats_keeper", "keeper_adv": "stats_keeper_adv",
}

# identity columns present in every category table
ID_COLS = ["Player", "Nation", "Pos", "Squad", "Age", "Born"]
JOIN_KEYS = ["season", "src_league", "Player", "Squad", "Born"]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            c[-1] if (not str(c[0]) or str(c[0]).startswith("Unnamed"))
            else f"{c[0]}_{c[-1]}".strip("_")
            for c in df.columns
        ]
    return df


def _player_ids(html_doc: str, table_id: str) -> dict[tuple[str, str], str]:
    """(Player, Squad) -> FBref player id, scraped from data-append-csv rows."""
    m = re.search(rf'id="{table_id}".*?</table>', html_doc, re.S)
    if not m:
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in re.findall(r"<tr.*?</tr>", m.group(0), re.S):
        pid = re.search(r'data-append-csv="([^"]+)"', row)
        pname = re.search(r'data-append-csv="[^"]+">([^<]+)</a>', row)
        squad = re.search(r'/squads/[0-9a-f]{8}/[^"]*">([^<]+)</a>', row)
        if pid and pname:
            out[(pname.group(1), squad.group(1) if squad else "")] = pid.group(1)
    return out


def load_category(cat: str) -> pd.DataFrame:
    table_id = CATEGORY_TABLE[cat]
    frames = []
    for path in sorted(STATS_DIR.glob(f"*_{cat}.html")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        doc = path.read_text(encoding="utf-8").replace("<!--", "").replace("-->", "")
        try:
            df = pd.read_html(StringIO(doc), attrs={"id": table_id})[0]
        except (ValueError, IndexError):
            continue
        df = _flatten(df)
        df = df[df["Player"].notna() & (df["Player"] != "Player")]
        df = df[~df["Squad"].astype(str).str.contains("Squads", na=False)]  # drop league-total rows
        df.insert(0, "season", m["season"])
        df.insert(1, "src_league", m["comp"])
        # non-identity stat columns get a category prefix to stay distinct
        ren = {c: f"{cat}__{c}" for c in df.columns
               if c not in (ID_COLS + ["season", "src_league", "Rk", "Matches"])}
        df = df.rename(columns=ren).drop(columns=["Rk", "Matches"], errors="ignore")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    if not STATS_DIR.exists() or not any(STATS_DIR.glob("*.html")):
        raise SystemExit(f"no player-stats pages in {STATS_DIR} - run pull_fbref_player_stats.py")

    base = load_category("standard")
    if base.empty:
        raise SystemExit("no standard-category pages cached yet")

    combined = base
    for cat in CATEGORY_TABLE:
        if cat == "standard":
            continue
        df = load_category(cat)
        if df.empty:
            print(f"({cat}: no pages, skipping)")
            continue
        on = [k for k in JOIN_KEYS if k in df.columns and k in combined.columns]
        drop_dupe_ids = [c for c in ID_COLS if c not in on and c in df.columns]
        df = df.drop(columns=drop_dupe_ids).drop_duplicates(subset=on)
        combined = combined.merge(df, on=on, how="left")

    for col in ("Age", "Born"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(combined)} rows, {combined.shape[1]} cols)")
    print(f"players: {combined['Player'].nunique()}  "
          f"seasons: {sorted(combined['season'].dropna().unique())}")
    print(f"rows per season:\n{combined.groupby('season').size().to_string()}")


if __name__ == "__main__":
    main()
