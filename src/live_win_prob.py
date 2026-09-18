"""Live in-play win probability, re-conditioned on the current match state.

Not a new model - it reuses the exact scoreline-grid machinery
train_outcome_model.py already fits pre-match (Poisson lambdas -> Dixon-Coles
grid -> grid_1x2), just rescaled and re-convolved:

  1. Scale the pre-match full-90 lambdas (lh, la - train_outcome_model.py's
     fit_lambdas()) down to the goals still expected in the TIME REMAINING,
     linear in minutes left. A simplification: it ignores the real (higher)
     early-game and stoppage-time scoring rates - fine for a first pass.
  2. A red card should shift the disadvantaged team's remaining lambda, not
     just leave it to the time-remaining scaling - literature puts a team
     down to 10 men at roughly 0.70-0.75x its expected-goals rate for the
     rest of the match. Applied here as a fixed multiplier per red card
     (compounds for a second), until there's enough red-card matches in this
     project's own data to fit the multiplier directly.
  3. Poisson(remaining_lh) x Poisson(remaining_la), same Dixon-Coles
     low-score correction (same rho) as the pre-match grid, gives the
     distribution of ADDITIONAL goals from here. Add the current score and
     compare (i+cur_h) vs (j+cur_a) instead of the fresh-match (i vs j) -
     everything else about the grid is identical to _fit_grid.

Vectorized: every argument may be a scalar or an equal-length array, so a
whole backtest (many matches x many minute-checkpoints) runs in one call.

Run: py -3.11 src/live_win_prob.py   (self-test against hand-worked cases)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAXG_REM = 8
RED_CARD_FACTOR = 0.73  # literature range ~0.70-0.75 for a team down a man


def live_win_prob(lh, la, minute, cur_h, cur_a, *, home_reds=0, away_reds=0,
                   rho=-0.11, red_card_factor=RED_CARD_FACTOR, maxg=MAXG_REM):
    """P(home win, draw, away win) from the current match state.

    lh, la: pre-match full-90-minute expected goals (home, away) - from
        train_outcome_model.fit_lambdas().
    minute: elapsed minutes (0-90; clipped, so stoppage time is treated as
        minute 90 - no more remaining-time goals expected beyond it).
    cur_h, cur_a: current score.
    home_reds, away_reds: red cards received by each side so far this match.

    Returns an (n, 3) array [P(home), P(draw), P(away)] - or a (3,) array if
    every argument was a scalar.
    """
    scalar_input = np.ndim(lh) == 0 and all(
        np.ndim(x) == 0 for x in (la, minute, cur_h, cur_a, home_reds, away_reds))

    lh, la, minute, cur_h, cur_a, home_reds, away_reds = (
        np.atleast_1d(np.asarray(x, dtype=float))
        for x in (lh, la, minute, cur_h, cur_a, home_reds, away_reds))
    lh, la, minute, cur_h, cur_a, home_reds, away_reds = np.broadcast_arrays(
        lh, la, minute, cur_h, cur_a, home_reds, away_reds)

    frac = np.clip(90 - np.clip(minute, 0, 90), 0, 90) / 90
    rem_lh = np.clip(lh * frac * (red_card_factor ** home_reds), 1e-6, None)
    rem_la = np.clip(la * frac * (red_card_factor ** away_reds), 1e-6, None)

    ph = poisson.pmf(np.arange(maxg)[:, None], rem_lh)   # (maxg, n)
    pa = poisson.pmf(np.arange(maxg)[:, None], rem_la)
    grid = ph[:, None, :] * pa[None, :, :]                # (i, j, n) additional goals
    grid[0, 0] *= 1 - rem_lh * rem_la * rho
    grid[0, 1] *= 1 + rem_lh * rho
    grid[1, 0] *= 1 + rem_la * rho
    grid[1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    grid = grid / grid.sum((0, 1), keepdims=True)

    i = np.arange(maxg)[:, None, None] + cur_h[None, None, :]
    j = np.arange(maxg)[None, :, None] + cur_a[None, None, :]
    p_home = np.where(i > j, grid, 0).sum((0, 1))
    p_draw = np.where(i == j, grid, 0).sum((0, 1))
    p_away = np.where(i < j, grid, 0).sum((0, 1))
    out = np.stack([p_home, p_draw, p_away], axis=1)
    out = np.clip(out, 1e-9, None)
    out = out / out.sum(1, keepdims=True)
    return out[0] if scalar_input else out


def _self_test() -> None:
    from train_outcome_model import grid_1x2  # noqa: PLC0415 - test-only import

    # 1. minute=0, 0-0, no reds: must reduce exactly to the pre-match grid
    lh, la = 1.6, 1.1
    live = live_win_prob(lh, la, minute=0, cur_h=0, cur_a=0)
    i = np.arange(MAXG_REM)[:, None, None]
    j = np.arange(MAXG_REM)[None, :, None]
    lh_a, la_a = np.array([lh]), np.array([la])
    ph = poisson.pmf(np.arange(MAXG_REM)[:, None], lh_a)
    pa = poisson.pmf(np.arange(MAXG_REM)[:, None], la_a)
    grid = ph[:, None, :] * pa[None, :, :]
    rho = -0.11
    grid[0, 0] *= 1 - lh_a * la_a * rho
    grid[0, 1] *= 1 + lh_a * rho
    grid[1, 0] *= 1 + la_a * rho
    grid[1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    grid = grid / grid.sum((0, 1), keepdims=True)
    ref = grid_1x2(grid)[0]  # [away, draw, home] order
    ref = ref[[2, 1, 0]]     # -> [home, draw, away]
    assert np.allclose(live, ref, atol=1e-9), f"minute=0 mismatch: {live} vs {ref}"
    print(f"minute=0, 0-0: live={live.round(3)}  ref={ref.round(3)}  OK")

    # 2. minute=90, home leading: must be ~certain home win regardless of lambdas
    live = live_win_prob(1.6, 1.1, minute=90, cur_h=2, cur_a=1)
    assert live[0] > 0.999, live
    print(f"minute=90, 2-1: live={live.round(4)}  (home win near-certain)  OK")

    # 3. minute=90, level score: must be ~certain draw
    live = live_win_prob(1.6, 1.1, minute=90, cur_h=1, cur_a=1)
    assert live[1] > 0.999, live
    print(f"minute=90, 1-1: live={live.round(4)}  (draw near-certain)  OK")

    # 4. a red card should lower the disadvantaged team's win chance, holding
    #    everything else fixed
    base = live_win_prob(1.6, 1.1, minute=45, cur_h=0, cur_a=0)
    down_a_man = live_win_prob(1.6, 1.1, minute=45, cur_h=0, cur_a=0, home_reds=1)
    assert down_a_man[0] < base[0], (base, down_a_man)
    assert down_a_man[2] > base[2], (base, down_a_man)
    print(f"minute=45, 0-0: no-red={base.round(3)}  home-red={down_a_man.round(3)}  OK")

    # 5. vectorized call matches scalar calls element-by-element
    lhs, las = np.array([1.6, 0.9]), np.array([1.1, 1.4])
    mins, hs, as_ = np.array([30, 75]), np.array([1, 0]), np.array([0, 2])
    vec = live_win_prob(lhs, las, mins, hs, as_)
    for k in range(2):
        s = live_win_prob(lhs[k], las[k], mins[k], hs[k], as_[k])
        assert np.allclose(vec[k], s, atol=1e-9)
    print("vectorized batch matches per-row scalar calls  OK")

    print("\nall self-tests passed")


if __name__ == "__main__":
    _self_test()
