"""Fetch the current season from pluggable unofficial sources, merge to one table.

None of these APIs come with an SLA, so the design is redundancy: a priority list
of sources, each returning the same schema (src/live/schema.py). We take results/
fixtures from the most authoritative source that has them and let lower-priority
sources fill gaps (notably ESPN's shot/possession stats).

Priority default: football-data.org (sanctioned, stable - results backbone) then
ESPN (unofficial - adds stats). football-data.org is skipped automatically if
FOOTBALL_DATA_ORG_KEY isn't set, so ESPN alone works out of the box.

Run (Python 3.11+):
    py -3.11 src/pull_live.py                       # current season, default comps
    py -3.11 src/pull_live.py --season 2026-27 --comps ENG1 GER1 ESP1 UCL
    py -3.11 src/pull_live.py --sources espn        # force a single source
    py -3.11 src/pull_live.py --no-stats            # fixtures/results only, fast

Output:
    data/raw/live/<source>_<season>.csv      per-source snapshot
    data/processed/live_matches_<season>.csv merged, schema.MATCH_COLS
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import pandas as pd

from live import espn, fotmob, football_data_org, sofascore
from live.schema import DEFAULT_COMPS, MATCH_COLS, normalize_team

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "live"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

SOURCES = {
    "football-data-org": football_data_org.fetch,
    "espn": espn.fetch,
    "fotmob": fotmob.fetch,
    "sofascore": sofascore.fetch,
}
# authority order for filling a merged row
DEFAULT_PRIORITY = ["football-data-org", "espn"]


def _current_season() -> str:
    t = datetime.date.today()
    y = t.year if t.month >= 7 else t.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def _load_dotenv() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _merge(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """frames in priority order (most authoritative first). One row per
    (comp_code, home, away); each field taken from the first source that has it."""
    merged: dict[tuple, dict] = {}
    for _src, df in frames:
        if df.empty:
            continue
        df = df.copy()
        df["_h"] = df["HomeTeam"].map(normalize_team)
        df["_a"] = df["AwayTeam"].map(normalize_team)
        for _, r in df.iterrows():
            key = (r["comp_code"], r["_h"], r["_a"])
            slot = merged.setdefault(key, {})
            for c in MATCH_COLS:
                if slot.get(c) in (None, "") or pd.isna(slot.get(c)):
                    v = r.get(c)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        slot[c] = v
            slot.setdefault("_sources", set()).add(r["source"])
    out = pd.DataFrame(list(merged.values()))
    if out.empty:
        return out
    out["source"] = out.pop("_sources").map(lambda s: "+".join(sorted(s)))
    return out[MATCH_COLS].sort_values(["comp_code", "Date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: current)")
    ap.add_argument("--comps", nargs="+", default=DEFAULT_COMPS)
    ap.add_argument("--sources", nargs="+", default=None,
                    help=f"subset/order of {list(SOURCES)} (default: available of {DEFAULT_PRIORITY})")
    ap.add_argument("--no-stats", dest="stats", action="store_false")
    args = ap.parse_args()

    _load_dotenv()
    season = args.season or _current_season()

    if args.sources:
        order = args.sources
    else:
        order = [s for s in DEFAULT_PRIORITY
                 if s != "football-data-org" or os.environ.get("FOOTBALL_DATA_ORG_KEY")]
    if not order:
        raise SystemExit("no sources selected")
    print(f"season {season} | comps {args.comps} | sources {order}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in order:
        print(f"[{name}]")
        try:
            df = SOURCES[name](args.comps, season, with_stats=args.stats)
        except NotImplementedError as e:
            print(f"  skipped: {e}")
            continue
        df.to_csv(RAW_DIR / f"{name}_{season}.csv", index=False)
        frames.append((name, df))
        print(f"  -> {len(df)} matches")

    merged = _merge(frames)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"live_matches_{season}.csv"
    merged.to_csv(out, index=False)
    played = merged["FTHG"].notna().sum() if len(merged) else 0
    with_stats = merged["HS"].notna().sum() if len(merged) else 0
    print(f"\nwrote {out}  ({len(merged)} matches, {played} played, {with_stats} with stats)")
    if len(merged):
        print(merged.groupby("comp_code").size().to_string())


if __name__ == "__main__":
    main()
