"""One-off: fold the old data/raw/live FotMob JSON caches (fotmob_players,
fotmob_lineups, fotmob_events, fotmob_intl_players) into the committed per-season CSVs that
src/fotmob_store.py describes, and flag fotmob_match_meta.csv's fetched /
events_fetched columns from what was cached. Safe to re-run; the caches can be
deleted afterwards.

Run:  py -3.11 src/migrate_fotmob_cache.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import fotmob_store as store  # noqa: E402
from pull_fotmob_players import _extract_lineup_rows  # noqa: E402

LIVE = ROOT / "data" / "raw" / "live"


def _load(d: Path, mid: str):
    f = d / f"{mid}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main() -> None:
    meta = store.read_meta()
    meta = meta.drop_duplicates("match_id", keep="last")
    fetched, ev_fetched = {}, {}
    for season, g in meta.groupby("season"):
        players, lineups, events = [], [], []
        for mid, code in zip(g["match_id"], g["src_league"]):
            p = _load(LIVE / "fotmob_players", mid)
            ln = _load(LIVE / "fotmob_lineups", mid)
            ev = _load(LIVE / "fotmob_events", mid)
            fetched[mid] = int(p is not None and ln is not None)
            ev_fetched[mid] = int(ev is not None)
            players += [{**r, "season": season, "src_league": code} for r in p or []]
            lineups += _extract_lineup_rows(ln or {}, mid)
            events += [{"match_id": mid, **e} for e in ev or []]
        for kind, rows in (("player_matches", players), ("lineups", lineups), ("events", events)):
            store.upsert(kind, season, pd.DataFrame(rows))
        print(f"  {season}: {len(g)} matches, {len(players)} player rows, "
              f"{len(lineups)} lineup rows, {len(events)} events")
    meta = meta.assign(fetched=meta["match_id"].map(fetched),
                       events_fetched=meta["match_id"].map(ev_fetched))
    meta.to_csv(store.MATCH_META, index=False)
    print(f"wrote {store.MATCH_META}  ({meta['fetched'].sum()}/{len(meta)} fetched, "
          f"{meta['events_fetched'].sum()} with events)")

    # internationals: their rows are already in fotmob_intl_player_matches.csv;
    # record every cached match as fetched so ones with no player data aren't
    # re-requested (pull_fotmob_internationals.py fills in teams / scores)
    intl = sorted(f.stem for f in (LIVE / "fotmob_intl_players").glob("*.json"))
    if intl:
        store.upsert_meta(pd.DataFrame({"match_id": intl, "fetched": 1}), store.INTL_META)
        print(f"wrote {store.INTL_META}  ({len(intl)} matches)")


if __name__ == "__main__":
    main()
