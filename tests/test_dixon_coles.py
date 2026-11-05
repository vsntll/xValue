"""dixon_coles.py - match probabilities from a fitted attack/defence model."""

from __future__ import annotations

import numpy as np
import pandas as pd

import dixon_coles as dc


def _model(att, dfn, home=0.25, rho=-0.05):
    return {"idx": {t: k for k, t in enumerate(att)}, "att": np.array(list(att.values())),
            "dfn": np.array([dfn[t] for t in att]), "home": home, "rho": rho}


def test_probabilities_sum_to_one_and_ordered_a_d_h():
    m = _model({"a": 0.2, "b": -0.1}, {"a": -0.1, "b": 0.1})
    p = dc.match_probs(m, "a", "b")
    assert p.shape == (3,)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()
    assert p[2] > p[0]   # [away, draw, home]: the stronger home side is favoured


def test_home_advantage_alone_favours_home():
    m = _model({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, home=0.3)
    p = dc.match_probs(m, "a", "b")
    assert p[2] > p[0]
    flipped = dc.match_probs(_model({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, home=0.0), "a", "b")
    assert np.isclose(flipped[0], flipped[2])   # no home edge, equal teams -> symmetric


def test_unknown_team_falls_back_to_prior():
    m = _model({"a": 0.0}, {"a": 0.0})
    assert np.allclose(dc.match_probs(m, "a", "zzz"), [0.29, 0.26, 0.45])
    assert np.allclose(dc.match_probs(None, "a", "b"), [0.29, 0.26, 0.45])


def test_fit_recovers_the_stronger_team():
    rng = np.random.default_rng(0)
    teams = ["strong", "mid1", "mid2", "weak"]
    rate = {"strong": 2.2, "mid1": 1.3, "mid2": 1.3, "weak": 0.6}
    rows, day = [], pd.Timestamp("2024-08-01")
    for k in range(400):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"Date": day + pd.Timedelta(days=k), "h": h, "a": a,
                     "FTHG": rng.poisson(rate[h] * 1.1), "FTAG": rng.poisson(rate[a])})
    matches = pd.DataFrame(rows)
    model = dc.fit(matches, asof=matches["Date"].max() + pd.Timedelta(days=1))
    assert model is not None
    att = dict(zip(model["idx"], model["att"]))
    assert att["strong"] > att["mid1"] > att["weak"]
    p = dc.match_probs(model, "strong", "weak")
    assert p[2] > 0.6


def test_fit_needs_enough_matches():
    few = pd.DataFrame({"Date": pd.date_range("2024-08-01", periods=20), "h": "a", "a": "b",
                        "FTHG": 1, "FTAG": 0})
    assert dc.fit(few, asof=pd.Timestamp("2025-01-01")) is None
