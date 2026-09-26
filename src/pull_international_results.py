"""Every senior men's international result since 1872 (friendlies included),
from the open martj42/international_results dataset on GitHub - the history
build_national_elo.py rates national teams from, so a player's international
appearance can be opponent-adjusted the same way a club one is.

Results only (no player stats) - those come from FotMob, see
src/pull_fotmob_internationals.py. Team names here are the canonical ones;
FotMob's spellings are aliased onto them there.

Run:  py -3.11 src/pull_international_results.py
Output: data/processed/international_results.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "international_results.csv"
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def main() -> None:
    df = pd.read_csv(URL)
    df = df.dropna(subset=["home_score", "away_score"])  # scheduled, not yet played
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} matches, {df['date'].min()} -> {df['date'].max()})")


if __name__ == "__main__":
    main()
