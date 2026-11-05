"""live_win_prob.py - in-play P(home, draw, away) from the match state."""

from __future__ import annotations

import numpy as np
import pytest

from live_win_prob import live_win_prob


@pytest.mark.parametrize("h,a,winner", [(2, 1, 0), (1, 1, 1), (0, 3, 2), (0, 0, 1)])
def test_minute_90_current_score_is_final(h, a, winner):
    p = live_win_prob(1.6, 1.1, minute=90, cur_h=h, cur_a=a)
    assert p[winner] > 0.999, p


def test_stoppage_time_is_treated_as_minute_90():
    assert np.allclose(live_win_prob(1.6, 1.1, minute=97, cur_h=1, cur_a=0),
                       live_win_prob(1.6, 1.1, minute=90, cur_h=1, cur_a=0))


def test_probabilities_sum_to_one_at_every_minute():
    for minute in range(0, 91, 15):
        p = live_win_prob(1.4, 1.2, minute=minute, cur_h=1, cur_a=1)
        assert np.isclose(p.sum(), 1.0)


def test_leading_side_gains_as_time_runs_out():
    early = live_win_prob(1.4, 1.4, minute=20, cur_h=1, cur_a=0)[0]
    late = live_win_prob(1.4, 1.4, minute=80, cur_h=1, cur_a=0)[0]
    assert late > early


def test_red_card_hurts_the_side_down_a_man():
    base = live_win_prob(1.6, 1.1, minute=45, cur_h=0, cur_a=0)
    red = live_win_prob(1.6, 1.1, minute=45, cur_h=0, cur_a=0, home_reds=1)
    assert red[0] < base[0] and red[2] > base[2]


def test_vectorised_matches_scalar_calls():
    lh, la = np.array([1.6, 0.9]), np.array([1.1, 1.4])
    mins, hs, as_ = np.array([30, 75]), np.array([1, 0]), np.array([0, 2])
    vec = live_win_prob(lh, la, mins, hs, as_)
    for k in range(2):
        assert np.allclose(vec[k], live_win_prob(lh[k], la[k], mins[k], hs[k], as_[k]))


def test_module_self_test_passes():
    from live_win_prob import _self_test
    _self_test()   # minute-0 == pre-match Dixon-Coles grid, plus the cases above
