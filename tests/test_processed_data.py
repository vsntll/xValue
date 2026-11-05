"""Invariants on the processed CSVs the models and the site are built from.

The refresh runs these after rebuilding everything and before committing, so a
silent upstream change (a source renaming a column, a pull returning half the
rows, a broken join duplicating matches) fails the run instead of shipping.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import PROC, committed_lines

MATCH_KEY = ["season", "comp", "Date", "HomeTeam", "AwayTeam"]
VALUE_CAP_EUR = 220e6
PROB_COLS = ["p_away", "p_draw", "p_home"]

# file -> smallest share of its committed row count a rebuild may keep
NO_COLLAPSE = {
    "matches_all.csv": 0.95,
    "match_model_table.csv": 0.95,
    "outcome_model_predictions_all.csv": 0.95,
    "value_model_predictions.csv": 0.9,
    "fbref_player_season_stats.csv": 0.9,
    "understat_player_matches.csv": 0.95,
    "player_elo.csv": 0.9,
    "fotmob_intl_player_matches.csv": 0.95,
}


def _csv(name: str) -> pd.DataFrame:
    path = PROC / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return pd.read_csv(path, low_memory=False)


@pytest.mark.parametrize("name", ["matches_all.csv", "match_model_table.csv",
                                  "outcome_model_predictions_all.csv"])
def test_match_keys_unique(name):
    df = _csv(name)
    dup = df[df.duplicated(MATCH_KEY, keep=False)]
    assert dup.empty, f"{len(dup)} rows share a match key in {name}:\n{dup[MATCH_KEY].head()}"


@pytest.mark.parametrize("name", ["outcome_model_predictions.csv", "outcome_model_predictions_hybrid.csv",
                                  "outcome_model_predictions_all.csv"])
def test_1x2_probabilities_sum_to_one(name):
    p = _csv(name)[PROB_COLS]
    assert p.notna().all().all(), f"{name}: missing 1X2 probabilities"
    assert ((p >= 0) & (p <= 1)).all().all(), f"{name}: a probability outside [0, 1]"
    # written rounded to 4 dp, so each row can be up to ~1.5e-4 off
    assert np.allclose(p.sum(axis=1), 1.0, atol=5e-4), f"{name}: 1X2 rows don't sum to 1"


def test_predicted_values_capped_and_positive():
    v = _csv("value_model_predictions.csv")["predicted_eur"]
    assert v.notna().all()
    assert (v > 0).all()
    assert (v <= VALUE_CAP_EUR * 1.001).all(), f"max predicted value {v.max():,.0f} above the €220M cap"


def test_value_predictions_one_row_per_player_season():
    v = _csv("value_model_predictions.csv")
    assert not v.duplicated(["season", "src_league", "Player", "Squad"]).any()


@pytest.mark.parametrize("name,share", sorted(NO_COLLAPSE.items()))
def test_no_row_count_collapse(name, share):
    rel = f"data/processed/{name}"
    before = committed_lines(rel)
    if before is None:
        pytest.skip(f"{name} not committed yet")
    now = (PROC / name).read_bytes().count(b"\n")
    assert now >= share * before, f"{name}: {now:,} lines vs {before:,} committed (< {share:.0%})"


def test_fotmob_player_rows_unique_and_verified():
    from fotmob_store import read
    pm = read("player_matches")
    if pm.empty:
        pytest.skip("no FotMob per-match rows")
    assert not pm.duplicated(["match_id", "fotmob_id"]).any()
    if "verified" in pm:
        assert set(pm["verified"].dropna()) <= {"agree", "conflict", "fotmob_only"}


def test_player_elo_ratings_finite_and_plausible():
    r = _csv("player_elo.csv")["rating_after"]
    assert np.isfinite(r).all()
    assert r.between(500, 3500).all(), f"rating range {r.min():.0f}..{r.max():.0f}"


def test_model_metrics_present():
    path = PROC / "model_metrics.json"
    if not path.exists():
        pytest.skip("no model_metrics.json yet")
    m = json.loads(path.read_text(encoding="utf-8"))
    assert 0 < m["value"]["r2_log"] <= 1
    for k in ("pure", "bookmaker", "base_rate"):
        assert 0.5 < m["outcome"][k]["log_loss"] < 1.2
