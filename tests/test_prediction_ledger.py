"""prediction_ledger.py - snapshotting and the two-model scoreboard, on a
synthetic ledger (no dependence on the real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import prediction_ledger as pl


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "LEDGER", tmp_path / "predictions_ledger.csv")
    return pl.LEDGER


def _row(snap, ver, day, h, a, p):
    return {"snapshot_at": snap, "model_version": ver, "data_asof": snap[:10], "league": "L",
            "kickoff_date": day, "home": h, "away": a, "home_key": h, "away_key": a,
            "p_home": p[0], "p_draw": p[1], "p_away": p[2], "xg_home": np.nan, "xg_away": np.nan}


def test_model_kind():
    assert pl.model_kind("clf-abc") == "classifier"
    assert pl.model_kind("df99a60559") == "dixon_coles"


def test_append_skips_unchanged_snapshot(ledger):
    now = pd.Timestamp("2026-11-05 07:00")
    c = {"model_version": "v1", "league": "L", "kickoff_date": "2026-11-07", "home": "a", "away": "b",
         "home_key": "a", "away_key": "b", "p_home": .5, "p_draw": .3, "p_away": .2, "xg_home": 1, "xg_away": 1}
    assert pl._append([c], now, now) == 1
    assert pl._append([c], now, now) == 0                      # same version + odds: skipped
    assert pl._append([{**c, "p_home": .55, "p_away": .15}], now, now) == 1   # odds moved: logged
    assert pl._append([{**c, "model_version": "clf-x"}], now, now) == 1       # other model: logged


def test_scoreboard_head_to_head(ledger, monkeypatch):
    rows = [
        # match 1: both models, two classifier snapshots - the later one counts
        _row("2026-11-03T07:00:00Z", "clf-1", "2026-11-06", "a", "b", (.40, .30, .30)),
        _row("2026-11-05T07:00:00Z", "clf-1", "2026-11-06", "a", "b", (.60, .25, .15)),
        _row("2026-11-05T07:00:00Z", "dc1", "2026-11-06", "a", "b", (.50, .25, .25)),
        # match 2: both models
        _row("2026-11-05T07:00:00Z", "clf-1", "2026-11-07", "c", "d", (.20, .30, .50)),
        _row("2026-11-05T07:00:00Z", "dc1", "2026-11-07", "c", "d", (.30, .30, .40)),
        # match 3: dixon-coles only -> not head to head
        _row("2026-11-05T07:00:00Z", "dc1", "2026-11-08", "e", "f", (.5, .3, .2)),
        # match 4: snapshot on kickoff day -> never counts
        _row("2026-11-09T07:00:00Z", "clf-1", "2026-11-09", "g", "h", (.5, .3, .2)),
        _row("2026-11-09T07:00:00Z", "dc1", "2026-11-09", "g", "h", (.5, .3, .2)),
    ]
    pd.DataFrame(rows, columns=pl.COLS).to_csv(ledger, index=False)
    results = {("a", "b"): (0, np.array([.55, .25, .20]), np.array([.5, .3, .2])),
               ("c", "d"): (2, np.array([.25, .30, .45]), None)}
    monkeypatch.setattr(pl, "_outcome_lookup", lambda: lambda h, a, day: results.get((h, a)))
    sb = pl.scoreboard()
    assert sb["models"] == ["classifier", "dixon_coles"]
    assert sb["n"] == 2                                   # matches 1 + 2 only
    exp_clf = -(np.log(.60) + np.log(.50)) / 2            # later classifier snapshot for match 1
    assert sb["ll"]["classifier"] == pytest.approx(exp_clf, abs=1e-4)
    assert sb["ll"]["dixon_coles"] == pytest.approx(-(np.log(.50) + np.log(.40)) / 2, abs=1e-4)
    assert sb["n_open"] == 1 and sb["ll"]["open"] == pytest.approx(-np.log(.5), abs=1e-4)
    assert sb["recent"][0]["result"] == "A" and sb["recent"][1]["result"] == "H"
