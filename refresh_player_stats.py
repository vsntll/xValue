#!/usr/bin/env python3
"""Local refresh for the FBref counting stats the weekly workflow can't touch.

Re-scrapes the CURRENT season's goals / assists / minutes / cards / shots /
keeper numbers from FBref (a visible Chrome window - nodriver), rebuilds the
player table (folding in fresh xG + market values), retrains the value model,
and regenerates ``site/index.html``.

    py -3.11 refresh_player_stats.py            # scrape + rebuild; you commit
    py -3.11 refresh_player_stats.py --commit   # also commit + push (fires the Pages deploy)
    py -3.11 refresh_player_stats.py --season 2025-26   # a specific season instead

A Chrome window opens for the scrape. If it shows a "Just a moment..." challenge,
leave it - it usually clears itself within a few seconds. Only the in-progress
season is re-fetched; the historical pages stay cached.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fbref_common import current_season  # noqa: E402

PY = [sys.executable]  # the interpreter running this script (py -3.11)

# the FBref player-stats categories that actually return data - the advanced
# ones (passing/defense/possession/gca/keeper_adv) come back as empty skeletons.
CATS = ["standard", "shooting", "misc", "keeper", "playing_time"]


def run(label: str, *args: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    if subprocess.run(PY + list(args), cwd=ROOT).returncode != 0:
        sys.exit(f"\n{label} FAILED - fix it and re-run (cached pages make it resume).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", default=None, help="default: the season in progress")
    ap.add_argument("--commit", action="store_true",
                    help="commit + push site/index.html (triggers the GitHub Pages deploy)")
    args = ap.parse_args()

    season = args.season or current_season()
    print(f"Refreshing FBref player stats for {season}")

    pages = ROOT / "data" / "raw" / "fbref" / "player_stats"
    stale = sorted(pages.glob(f"*_{season}_*.html"))
    for f in stale:
        f.unlink()
    print(f"cleared {len(stale)} cached {season} page(s) so they re-download fresh")

    run(f"FBref player-stats scrape ({season}) - a Chrome window will open",
        "src/pull_fbref_player_stats.py", "--seasons", season, "--cats", *CATS)
    run("parse -> fbref_player_season_stats.csv", "src/parse_fbref_player_stats.py")
    run("build matches_all.csv", "src/build_matches_all.py")
    run("build squad_season_features.csv", "src/build_squad_features.py")
    run("build match_model_table.csv", "src/build_match_model_table.py")
    run("build value_history.csv", "src/build_value_history.py")
    run("train value model", "src/train_value_model.py")
    run("train outcome model (hybrid)", "src/train_outcome_model.py", "--hybrid")
    run("regenerate site/index.html", "src/export_site_data.py")

    subprocess.run(["git", "add", "site/index.html"], cwd=ROOT)
    unchanged = subprocess.run(["git", "diff", "--cached", "--quiet"],
                               cwd=ROOT).returncode == 0
    print()
    if unchanged:
        print("site/index.html is unchanged - nothing to commit.")
    elif args.commit:
        subprocess.run(["git", "commit", "-m",
                        f"Refresh FBref player stats ({season})"], cwd=ROOT, check=True)
        subprocess.run(["git", "pull", "--rebase", "--autostash"], cwd=ROOT, check=True)
        subprocess.run(["git", "push"], cwd=ROOT, check=True)
        print("committed + pushed - GitHub Pages will redeploy.")
    else:
        print("site/index.html updated. Review it, then:")
        print(f'  git commit -m "Refresh FBref player stats ({season})" && git push')


if __name__ == "__main__":
    main()
