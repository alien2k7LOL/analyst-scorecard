"""Touch-by-deadline probability — the closed-form GBM barrier-hitting model (+ an empirical check).

The headline question is: starting from price S0, what is the probability the price TOUCHES a
target K at least once within ``n_days`` trading days? Under geometric Brownian motion, the log
price X_t = ln(S_t/S0) is an arithmetic BM with per-day drift ``mu`` and per-day vol ``sigma``, and
the first-passage (reflection-principle) result gives, for an up-barrier b = ln(K/S0) > 0:

    P(max_{0<=t<=T} X_t >= b) = Φ((mu·T - b)/(sigma·√T)) + exp(2·mu·b/sigma²)·Φ((-mu·T - b)/(sigma·√T))

DOWN targets use the same formula on the mirrored barrier (distance ln(S0/K), drift -mu).

This assumes CONTINUOUS monitoring; real resolution uses daily closes, so the raw model slightly
overstates touch probability. That bias is exactly what the calibration backtest measures and the
recalibration layer corrects — see calibration.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm

from ..schemas import Direction

MIN_SIGMA = 1e-6  # floor so a flat lookback can't divide by zero


def touch_probability(
    s0: float, k: float, n_days: int, mu_daily: float, sigma_daily: float, direction: Direction
) -> float:
    """Closed-form GBM probability that ``k`` is touched within ``n_days`` (continuous monitoring)."""
    if s0 <= 0 or k <= 0:
        raise ValueError("prices must be positive")
    if n_days < 1:
        raise ValueError("n_days must be >= 1")

    sigma = max(float(sigma_daily), MIN_SIGMA)
    if direction == Direction.UP:
        if k <= s0:
            return 1.0  # already at/above the target
        b = float(np.log(k / s0))
        nu = float(mu_daily)
    elif direction == Direction.DOWN:
        if k >= s0:
            return 1.0  # already at/below the target
        b = float(np.log(s0 / k))
        nu = -float(mu_daily)
    else:
        raise ValueError("direction must be UP or DOWN")

    T = float(n_days)
    sqrt_t = sigma * np.sqrt(T)
    term1 = norm.cdf((nu * T - b) / sqrt_t)
    expo = np.clip(2.0 * nu * b / (sigma ** 2), -50.0, 50.0)  # guard overflow on extreme drift
    term2 = np.exp(expo) * norm.cdf((-nu * T - b) / sqrt_t)
    return float(np.clip(term1 + term2, 0.0, 1.0))


def empirical_touch_probability(
    s0: float,
    k: float,
    n_days: int,
    daily_log_returns: np.ndarray,
    direction: Direction,
    n_paths: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Non-parametric cross-check: bootstrap the lookback's own daily returns into touch paths.

    Resamples observed daily log returns (with replacement) to build ``n_paths`` price paths of
    length ``n_days`` and returns the fraction that touch ``k``. Captures the lookback's actual
    return distribution (fat tails, skew) rather than assuming normality. Deterministic given ``rng``.
    """
    r = np.asarray(daily_log_returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    draws = r[rng.integers(0, len(r), size=(n_paths, n_days))]
    paths = s0 * np.exp(np.cumsum(draws, axis=1))
    if direction == Direction.UP:
        if k <= s0:
            return 1.0
        hit = paths.max(axis=1) >= k
    elif direction == Direction.DOWN:
        if k >= s0:
            return 1.0
        hit = paths.min(axis=1) <= k
    else:
        raise ValueError("direction must be UP or DOWN")
    return float(np.mean(hit))


class GbmTouchModel:
    """Thin wrapper exposing the closed-form touch probability as a named model."""

    name = "gbm_touch"

    def probability(
        self, *, s0: float, k: float, n_days: int, mu_daily: float, sigma_daily: float, direction: Direction
    ) -> float:
        return touch_probability(s0, k, n_days, mu_daily, sigma_daily, direction)
