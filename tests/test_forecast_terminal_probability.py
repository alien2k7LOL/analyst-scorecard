"""Validate the closed-form TERMINAL-value probability against Monte Carlo and its invariants.

Where ``touch_probability`` asks "did the path ever reach k?", ``terminal_probability`` asks "is the
price AT/NEAR k on the final day?" — a strictly harder question. Under GBM the terminal log return is
exactly Gaussian, so the closed form is a difference of normal CDFs; we check it against simulation
of the terminal price (endpoint only) plus the limiting/identity cases that must hold exactly.
"""

import numpy as np
import pytest

from analyst_scorecard.forecast.probability import (
    GbmTerminalModel,
    empirical_terminal_probability,
    terminal_probability,
    touch_probability,
)
from analyst_scorecard.schemas import Direction


def _mc_terminal(s0, k, n_days, mu, sigma, direction, band_pct=None, n_paths=120000, seed=0):
    """Monte-Carlo terminal probability: only the END price of each path matters."""
    rng = np.random.default_rng(seed)
    incr = mu + sigma * rng.standard_normal((n_paths, n_days))
    terminal = s0 * np.exp(incr.sum(axis=1))
    if band_pct is not None:
        hit = (terminal >= k * (1 - band_pct)) & (terminal <= k * (1 + band_pct))
    elif direction == Direction.UP:
        hit = terminal >= k
    else:
        hit = terminal <= k
    return float(np.mean(hit))


@pytest.mark.parametrize(
    "s0,k,n,mu,sigma,direction",
    [
        (100, 110, 60, 0.0, 0.02, Direction.UP),
        (100, 110, 60, 0.0008, 0.02, Direction.UP),
        (100, 90, 60, 0.0, 0.02, Direction.DOWN),
        (100, 120, 120, -0.0005, 0.025, Direction.UP),
        (50, 45, 40, 0.0003, 0.03, Direction.DOWN),
    ],
)
def test_at_or_beyond_matches_monte_carlo(s0, k, n, mu, sigma, direction):
    analytic = terminal_probability(s0, k, n, mu, sigma, direction)
    mc = _mc_terminal(s0, k, n, mu, sigma, direction)
    assert abs(analytic - mc) < 0.01, f"analytic={analytic:.3f} mc={mc:.3f}"


@pytest.mark.parametrize(
    "s0,k,n,mu,sigma,band",
    [
        (100, 110, 60, 0.0, 0.02, 0.03),
        (100, 100, 30, 0.0, 0.02, 0.02),
        (100, 120, 120, 0.0006, 0.025, 0.05),
        (200, 180, 90, -0.0003, 0.018, 0.04),
    ],
)
def test_band_matches_monte_carlo(s0, k, n, mu, sigma, band):
    analytic = terminal_probability(s0, k, n, mu, sigma, Direction.UP, band_pct=band)
    mc = _mc_terminal(s0, k, n, mu, sigma, Direction.UP, band_pct=band)
    assert abs(analytic - mc) < 0.012, f"analytic={analytic:.3f} mc={mc:.3f}"


def test_terminal_is_never_more_likely_than_touch():
    # The path can reach a level and drift away, so ending at/through it is rarer than ever touching.
    for k, d in [(115, Direction.UP), (88, Direction.DOWN)]:
        term = terminal_probability(100, k, 90, 0.0, 0.02, d)
        touch = touch_probability(100, k, 90, 0.0, 0.02, d)
        assert term < touch, f"{d}: terminal {term:.3f} should be < touch {touch:.3f}"


def test_up_and_down_at_a_level_partition_to_one():
    # P(S_T >= k) + P(S_T <= k) = 1 (the equality set has probability zero).
    up = terminal_probability(100, 107, 60, 0.0004, 0.02, Direction.UP)
    down = terminal_probability(100, 107, 60, 0.0004, 0.02, Direction.DOWN)
    assert up + down == pytest.approx(1.0, abs=1e-9)


def test_wider_band_is_more_likely():
    narrow = terminal_probability(100, 105, 60, 0.0, 0.02, Direction.UP, band_pct=0.01)
    wide = terminal_probability(100, 105, 60, 0.0, 0.02, Direction.UP, band_pct=0.06)
    assert 0.0 < narrow < wide < 1.0


def test_drift_toward_target_raises_at_or_beyond():
    flat = terminal_probability(100, 110, 90, 0.0, 0.02, Direction.UP)
    pushed = terminal_probability(100, 110, 90, 0.0010, 0.02, Direction.UP)
    assert pushed > flat


def test_more_vol_helps_a_far_at_or_beyond_target():
    base = terminal_probability(100, 115, 60, 0.0, 0.02, Direction.UP)
    assert terminal_probability(100, 115, 60, 0.0, 0.04, Direction.UP) > base


def test_zero_vol_is_floored_not_a_crash():
    p = terminal_probability(100, 110, 30, 0.0, 0.0, Direction.UP)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_empirical_bootstrap_agrees_with_closed_form():
    rng = np.random.default_rng(42)
    daily = rng.normal(0.0003, 0.02, size=900)
    mu, sigma = float(np.mean(daily)), float(np.std(daily, ddof=1))
    for band in (None, 0.04):
        emp = empirical_terminal_probability(100, 112, 60, daily, Direction.UP, band_pct=band,
                                             n_paths=40000, rng=np.random.default_rng(1))
        ana = terminal_probability(100, 112, 60, mu, sigma, Direction.UP, band_pct=band)
        assert abs(emp - ana) < 0.03, f"band={band}: emp={emp:.3f} ana={ana:.3f}"


def test_model_wrapper_matches_function():
    m = GbmTerminalModel()
    assert m.probability(s0=100, k=110, n_days=60, mu_daily=0.0003, sigma_daily=0.02,
                         direction=Direction.UP, band_pct=0.03) == \
        terminal_probability(100, 110, 60, 0.0003, 0.02, Direction.UP, 0.03)
