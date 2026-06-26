"""Validate the closed-form touch probability against Monte Carlo, and its boundary behavior.

The GBM barrier formula is the foundation of the whole forecast subsystem — if it's wrong, every
probability and every calibration number is wrong. So we check it against a fine-grained simulation
(many sub-daily steps ≈ continuous monitoring), plus the limiting cases that must hold exactly.
"""

import numpy as np
import pytest

from analyst_scorecard.forecast.probability import (
    GbmTouchModel,
    empirical_touch_probability,
    touch_probability,
)
from analyst_scorecard.schemas import Direction


def _mc_touch(s0, k, n_days, mu, sigma, direction, n_paths=60000, sub=20, seed=0):
    """Monte-Carlo touch probability with sub-daily steps (approximates continuous monitoring)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / sub
    steps = n_days * sub
    incr = mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, steps))
    x = np.cumsum(incr, axis=1)  # log price relative to s0
    if direction == Direction.UP:
        touched = x.max(axis=1) >= np.log(k / s0)
    else:
        touched = x.min(axis=1) <= np.log(k / s0)
    return float(np.mean(touched))


@pytest.mark.parametrize(
    "s0,k,n,mu,sigma,direction",
    [
        (100, 110, 60, 0.0, 0.02, Direction.UP),     # +10% up, no drift
        (100, 110, 60, 0.0008, 0.02, Direction.UP),  # +10% up, mild positive drift
        (100, 90, 60, 0.0, 0.02, Direction.DOWN),    # -10% down, no drift
        (100, 120, 120, -0.0005, 0.025, Direction.UP),  # far up target, negative drift, long horizon
        (50, 45, 40, 0.0003, 0.03, Direction.DOWN),  # down target with positive drift (fights it)
    ],
)
def test_closed_form_matches_monte_carlo(s0, k, n, mu, sigma, direction):
    analytic = touch_probability(s0, k, n, mu, sigma, direction)
    mc = _mc_touch(s0, k, n, mu, sigma, direction)
    assert abs(analytic - mc) < 0.02, f"analytic={analytic:.3f} mc={mc:.3f}"


def test_already_touched_is_certain():
    # UP target at/below current price -> already there -> prob 1.
    assert touch_probability(100, 95, 30, 0.0, 0.02, Direction.UP) == 1.0
    # DOWN target at/above current price -> already there -> prob 1.
    assert touch_probability(100, 105, 30, 0.0, 0.02, Direction.DOWN) == 1.0


def test_farther_targets_are_less_likely():
    p_near = touch_probability(100, 105, 60, 0.0, 0.02, Direction.UP)
    p_far = touch_probability(100, 130, 60, 0.0, 0.02, Direction.UP)
    assert 0.0 < p_far < p_near < 1.0


def test_more_time_and_more_vol_raise_touch_probability():
    base = touch_probability(100, 115, 30, 0.0, 0.02, Direction.UP)
    assert touch_probability(100, 115, 120, 0.0, 0.02, Direction.UP) > base   # more time
    assert touch_probability(100, 115, 30, 0.0, 0.04, Direction.UP) > base    # more vol


def test_zero_vol_is_floored_not_a_crash():
    # A perfectly flat lookback must not divide by zero; with no drift it can't reach a far target.
    p = touch_probability(100, 110, 30, 0.0, 0.0, Direction.UP)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_empirical_bootstrap_agrees_with_closed_form_on_normal_returns():
    rng = np.random.default_rng(42)
    daily = rng.normal(0.0, 0.02, size=750)  # ~3y of ~N(0,2%) daily log returns
    emp = empirical_touch_probability(100, 110, 60, daily, Direction.UP, n_paths=20000, rng=rng)
    ana = touch_probability(100, 110, 60, float(np.mean(daily)), float(np.std(daily, ddof=1)), Direction.UP)
    # Bootstrap monitors at daily steps (vs continuous) so it sits a touch lower — close, not equal.
    assert abs(emp - ana) < 0.06


def test_model_wrapper_matches_function():
    m = GbmTouchModel()
    assert m.probability(s0=100, k=110, n_days=60, mu_daily=0.0003, sigma_daily=0.02, direction=Direction.UP) == \
        touch_probability(100, 110, 60, 0.0003, 0.02, Direction.UP)
