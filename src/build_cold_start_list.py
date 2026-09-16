"""List the value model's cold-start players (has_any_prev == 0) for a targeted
Transfermarkt profile backfill - see train_value_model.py's cold-start segment.

Rather than mirror an entire second-tier league's history to fill prev_any_mv,
we only need it for these specific players: this script lists them so
pull_transfermarkt_scrape.py's --profiles mode knows who to target.

Run:  py -3.11 src/build_cold_start_list.py

Output: data/processed/cold_start_players.csv
    (season, src_league, Squad, Player, age, is_promoted, is_academy_age, is_brand_new)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_value_model import SRC, build_xy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "cold_start_players.csv"


def main() -> None:
    df = build_xy(pd.read_csv(SRC, low_memory=False))
    cold = df[df["has_any_prev"] == 0].drop_duplicates(subset=["season", "Squad", "Player"])
    out = cold[["season", "src_league", "Squad", "Player", "age",
                "is_promoted", "is_academy_age", "is_brand_new"]].sort_values(
        ["season", "Squad", "Player"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} cold-start player-seasons across "
          f"{sorted(out['season'].unique())})")
    print(out[["is_promoted", "is_academy_age", "is_brand_new"]].sum().to_string())


if __name__ == "__main__":
    main()
