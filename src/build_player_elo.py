"""Player Elo: a genuine, self-contained performance rating, updated match by
match from real output vs. an opponent-adjusted expectation - no value-model
inputs anywhere in it. Same mechanic as the team Elo already used for the
outcome model (src/build_match_model_table.py): everyone starts at 1500 and
moves by K * (actual - expected), where "expected" bakes in opponent strength
(harder to produce against a good defence) via the same base-10/exponent
curve as classical Elo's expected-score formula. The player's OWN current
rating feeds back into the NEXT match's expectation too - the whole point of
an Elo system - so a player who's already rated well has to keep performing
at that level to gain more, not just clear a fixed bar forever.

Windowed to the last three seasons (WINDOW_SEASONS below): this is about who
is playing well NOW, not a decade-old peak, and every player starts near the
population mean at the top of the window rather than dragging in a rating
from a different team, league, or level of first-team involvement.

"Expected" per match = position-group baseline output/90 (purely empirical,
computed from this same windowed data - nothing from train_value_model.py or
build_form_momentum.py touches this file) x minutes played x an opponent-
strength multiplier x the player's own current-rating multiplier.

Identity is tracked by Understat's own numeric player_id, NOT by name -
verified two real, different players ("Alvaro Fernandez": a Sevilla keeper
and an unrelated Real Madrid full-back) share a normalized name, which would
silently fuse their histories into one rating if grouped by name.

Needs data/processed/understat_player_matches.csv (match-level actual output,
with player_id/team_id - re-pull via src/pull_understat_player_matches.py if
missing) and data/processed/match_model_table.csv (opponent's own team Elo -
reused, not recomputed, so the two Elo systems stay consistent with each
other) and fbref_player_season_stats.csv (position only, name-joined - a
lookup, not a model input; Understat's own per-match position field just
says "Sub" for anyone who came off the bench, which is useless for grouping).

Run:  py -3.11 src/build_player_elo.py
Output: data/processed/player_elo.csv  (one row per player-match: rating before/after)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from live.schema import deaccent, normalize_team  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "player_elo.csv"

WINDOW_SEASONS = ["2024-25", "2025-26", "2026-27"]  # last 2-3 seasons, per design
MIN_MATCH_MINUTES = 10
START_RATING = 1500.0
REVERT = 0.25          # fraction reverted to 1500 between seasons - matches team Elo
K = 20.0
OPP_SCALE = 1000.0     # opponent-strength exponent divisor (gentler than classical Elo's 400 -
                       # output varies ~2x facing a good vs. bad defence, not 10x)
FORM_SCALE = 1000.0    # same curve, applied to the player's own rating
MULT_CLIP = (0.5, 2.0)
DELTA_CLIP = 40.0

POS_MAP = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
# Understat's own per-match position codes -> GK/DF/MF/FW. Id-scoped (no name
# collision possible) and always available for anyone who's started a match
# in the window, unlike the fbref lookup below, which needs both a name match
# AND fbref to have that player at all (verified gap: a 2025-26 Real Madrid
# debutant with no fbref row yet, full-back mislabeled GK by name-only fallback
# onto an unrelated same-named keeper before this fix).
UNDERSTAT_POS_MAP = {
    "GK": "GK",
    "DC": "DF", "DL": "DF", "DR": "DF",
    "DMC": "MF", "DML": "MF", "DMR": "MF",
    "MC": "MF", "ML": "MF", "MR": "MF",
    "AMC": "MF", "AML": "MF", "AMR": "MF",
    "FW": "FW", "FWL": "FW", "FWR": "FW",
}


def _understat_pos_lookup(raw: pd.DataFrame) -> dict[int, str]:
    """player_id -> GK/DF/MF/FW, from the mode of that player's own recorded
    starting positions (excludes "Sub" rows, which carry no real position)."""
    starts = raw[raw["position"] != "Sub"].copy()
    starts["pos_group"] = starts["position"].map(UNDERSTAT_POS_MAP)
    starts = starts.dropna(subset=["pos_group", "player_id"])
    return starts.groupby("player_id")["pos_group"].agg(lambda x: x.mode().iat[0]).to_dict()


def _pk(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", deaccent(s).lower().replace("'", "")).split())


def _pos_lookup() -> tuple[dict[tuple, str], dict[str, str]]:
    """(name-key, team-key) -> GK/DF/MF/FW, from fbref (Understat's own
    per-match position is just "Sub" for anyone who didn't start - unusable
    for grouping). fbref has no numeric id to join on, so this is a name
    lookup like player_id's collision case above - keying on (name, team) as
    well fixes the concrete one found (a Sevilla keeper and a Real Madrid
    full-back both "Alvaro Fernandez": name-only picked the keeper's GK label
    for the full-back too, since fbref carries both under the same name).
    A name-only fallback dict (most common position across all that name's
    rows) covers a mid-window transfer where the current team isn't the one
    on record; still collision-prone, so it's the fallback, not the primary."""
    s = pd.read_csv(PROC / "fbref_player_season_stats.csv", low_memory=False)
    s["_pk"] = s["Player"].map(_pk)
    s["team_key"] = s["Squad"].map(normalize_team)
    s["pos_group"] = s["Pos"].astype(str).str.split(",").str[0].map(lambda p: POS_MAP.get(p, "MF"))
    by_team = (s.sort_values("season").drop_duplicates(subset=["_pk", "team_key"], keep="last")
              .set_index(["_pk", "team_key"])["pos_group"].to_dict())
    by_name = s.groupby("_pk")["pos_group"].agg(lambda x: x.mode().iat[0]).to_dict()
    return by_team, by_name


def _opponent_elo_lookup() -> pd.DataFrame:
    """(season, team_key, 'YYYY-MM-DD') -> that team's pre-match Elo, straight
    from build_match_model_table.py's own goals-based Elo - reused, not
    recomputed, so a team's rating here always agrees with the outcome model."""
    mm = pd.read_csv(PROC / "match_model_table.csv")
    mm["d"] = pd.to_datetime(mm["Date"]).dt.strftime("%Y-%m-%d")
    h = mm[["season", "d"]].assign(team_key=mm["HomeTeam"].map(normalize_team), elo=mm["elo_h"])
    a = mm[["season", "d"]].assign(team_key=mm["AwayTeam"].map(normalize_team), elo=mm["elo_a"])
    return pd.concat([h, a], ignore_index=True).drop_duplicates(subset=["season", "team_key", "d"])


def main() -> None:
    pm_path = PROC / "understat_player_matches.csv"
    mm_path = PROC / "match_model_table.csv"
    if not pm_path.exists():
        raise SystemExit(f"{pm_path} not found - run src/pull_understat_player_matches.py first")
    if not mm_path.exists():
        raise SystemExit(f"{mm_path} not found - run src/build_match_model_table.py first")

    pm = pd.read_csv(pm_path)
    pm = pm[pm["season"].isin(WINDOW_SEASONS)].copy()
    if "player_id" not in pm.columns or pm["player_id"].isna().all():
        raise SystemExit(f"{pm_path} has no player_id for {WINDOW_SEASONS} - re-pull with "
                         f"`py -3.11 src/pull_understat_player_matches.py --seasons {' '.join(WINDOW_SEASONS)} --force`")
    pm = pm.dropna(subset=["player_id"])  # every WINDOW_SEASONS row has it (backfilled)
    pm["player_id"] = pm["player_id"].astype("int64")
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm = pm.dropna(subset=["date"])
    pm["minutes"] = pd.to_numeric(pm["minutes"], errors="coerce").fillna(0)
    pm = pm[pm["minutes"] >= MIN_MATCH_MINUTES].copy()
    pm["contribution"] = (pd.to_numeric(pm["xg"], errors="coerce").fillna(0)
                          + 0.7 * pd.to_numeric(pm["xa"], errors="coerce").fillna(0))
    pm["team_key"] = pm["team"].map(normalize_team)
    pm["_pk"] = pm["player"].map(_pk)
    pm["d"] = pm["date"].dt.strftime("%Y-%m-%d")

    # opponent = the other team_key sharing this game_id (every game_id has
    # exactly 2 teams in this data - verified on the full pull)
    sides = pm[["game_id", "team_key"]].drop_duplicates()
    opp_map = sides.merge(sides.rename(columns={"team_key": "opp_key"}), on="game_id")
    opp_map = opp_map[opp_map["team_key"] != opp_map["opp_key"]]
    pm = pm.merge(opp_map, on=["game_id", "team_key"], how="left")
    pm = pm.dropna(subset=["opp_key"])

    understat_pos = _understat_pos_lookup(
        pd.read_csv(PROC / "understat_player_matches.csv")
          .pipe(lambda d: d[d["season"].isin(WINDOW_SEASONS)]))
    pos_by_team, pos_by_name = _pos_lookup()
    pos_us = pm["player_id"].map(understat_pos)
    pos_team = pd.Series(list(zip(pm["_pk"], pm["team_key"]))).map(pos_by_team)
    pos_name = pm["_pk"].map(pos_by_name)
    pm["pos_group"] = pos_us.fillna(pos_team).fillna(pos_name).fillna("MF").to_numpy()

    elo_lu = _opponent_elo_lookup().rename(columns={"team_key": "opp_key", "elo": "opp_elo"})
    pm = pm.merge(elo_lu, on=["season", "opp_key", "d"], how="left")
    pm["opp_elo"] = pm["opp_elo"].fillna(START_RATING)

    # position-group baseline: minutes-weighted output/90, purely empirical
    # over this same window - the only thing "expected" is built from.
    baseline90, resid_std = {}, {}
    for pos, g in pm.groupby("pos_group"):
        n90 = (g["minutes"] / 90.0).sum()
        baseline90[pos] = float(g["contribution"].sum() / n90) if n90 > 0 else 0.05
        resid = g["contribution"] - baseline90[pos] * (g["minutes"] / 90.0)
        resid_std[pos] = float(resid.std()) or 0.2
    print("position baselines (output/90) and residual std, this window:")
    for pos in POS_MAP.values():
        print(f"  {pos}: baseline={baseline90.get(pos, 0):.3f}  std={resid_std.get(pos, 0):.3f}")

    pm = pm.sort_values(["player_id", "date"]).reset_index(drop=True)

    rating: dict[int, float] = {}
    last_season: dict[int, str] = {}
    rows = []
    for r in pm.itertuples(index=False):
        pid = r.player_id
        if pid not in rating:
            rating[pid] = START_RATING
        elif last_season.get(pid) != r.season:
            rating[pid] = START_RATING + (1 - REVERT) * (rating[pid] - START_RATING)
        last_season[pid] = r.season

        rp = rating[pid]
        opp_mult = np.clip(10 ** ((START_RATING - r.opp_elo) / OPP_SCALE), *MULT_CLIP)
        form_mult = np.clip(10 ** ((rp - START_RATING) / FORM_SCALE), *MULT_CLIP)
        expected = baseline90[r.pos_group] * (r.minutes / 90.0) * opp_mult * form_mult
        actual = r.contribution
        std = resid_std[r.pos_group]
        delta = float(np.clip(K * (actual - expected) / std, -DELTA_CLIP, DELTA_CLIP))
        rating[pid] = rp + delta

        rows.append({
            "season": r.season, "src_league": r.src_league, "date": r.d, "game_id": r.game_id,
            "player": r.player, "player_id": pid, "team_key": r.team_key, "opp_key": r.opp_key,
            "pos_group": r.pos_group, "minutes": r.minutes,
            "actual": round(actual, 4), "expected": round(expected, 4),
            "opp_elo": round(r.opp_elo, 1),
            "rating_before": round(rp, 1), "rating_after": round(rating[pid], 1),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    latest = out.sort_values("date").drop_duplicates(subset="player_id", keep="last")
    print(f"\nwrote {OUT}  ({len(out)} player-match rows, {out['player_id'].nunique()} players, "
          f"seasons {WINDOW_SEASONS})")
    print("\ntop 10 current ratings:")
    print(latest.nlargest(10, "rating_after")[["player", "team_key", "pos_group", "rating_after"]]
         .to_string(index=False))


if __name__ == "__main__":
    main()
