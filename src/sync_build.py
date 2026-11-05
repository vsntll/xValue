"""Move the rebuilt-every-run outputs through a GitHub release asset instead of
git history.

The refresh regenerates these from scratch on every run, so committing them
grew the repo ~5 MB a run for no reproducibility gain - the pulled sources,
ledgers and model scores that ARE the record stay committed on master. Each
refresh `fetch`es the last build (some steps read the previous run's copy
before rebuilding it, and tests compare against it), rebuilds, then `pack`s a
new zip that the workflow uploads to the `latest-build` release
(`gh release upload --clobber`), replacing the old one - constant size.

Locally (public repo, no auth needed):
    py -3.11 src/sync_build.py fetch           # -> data/processed/*, site/index.html
or rebuild them yourself with the pipeline (run_pipeline.md).

Run:  py -3.11 src/sync_build.py fetch [--baseline DIR]   # also extract a pristine copy into DIR
      py -3.11 src/sync_build.py pack  [--out FILE]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "vsntll/xValue"
TAG = "latest-build"
ASSET = "processed-build.zip"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
# rebuilt from the committed sources on every refresh - see run_pipeline.md
BUILD_FILES = [
    "data/processed/player_elo.csv",
    "data/processed/match_model_table.csv",
    "data/processed/match_model_upcoming.csv",
    "data/processed/matches_all.csv",
    "data/processed/outcome_model_predictions.csv",
    "data/processed/outcome_model_predictions_hybrid.csv",
    "data/processed/outcome_model_predictions_all.csv",
    "data/processed/value_model_predictions.csv",
    "data/processed/squad_season_features.csv",
    "data/processed/squad_momentum.csv",
    "data/processed/match_lineup_features.csv",
    "data/processed/match_lineup_players.csv",
    "site/index.html",
]


def fetch(baseline: Path | None = None) -> None:
    try:
        with urllib.request.urlopen(URL, timeout=120) as r:
            blob = r.read()
    except OSError as e:
        print(f"no build to restore ({e}) - the pipeline will rebuild everything from the sources")
        return
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        meta = json.loads(z.read("build.json")) if "build.json" in z.namelist() else {}
        for name in z.namelist():
            if name == "build.json":
                continue
            for dest in [ROOT] + ([baseline] if baseline else []):
                out = dest / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    print(f"restored {len(z.namelist()) - 1} build files from {TAG} "
          f"(built {meta.get('built_at', '?')} from {meta.get('commit', '?')[:7]})"
          + (f"; baseline copy in {baseline}" if baseline else ""))


def pack(out: Path) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": commit or os.environ.get("GITHUB_SHA", ""), "files": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in BUILD_FILES:
            f = ROOT / rel
            if f.exists():
                z.write(f, rel)
                meta["files"].append(rel)
        z.writestr("build.json", json.dumps(meta, indent=2))
    print(f"packed {len(meta['files'])} build files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["fetch", "pack"])
    ap.add_argument("--baseline", type=Path, help="fetch: also extract a pristine copy here")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / ASSET, help="pack: zip path")
    args = ap.parse_args()
    fetch(args.baseline) if args.action == "fetch" else pack(args.out)


if __name__ == "__main__":
    main()
