"""Deterministic synthetic 30-min intraday data — for the intraday demo, proof, and tests.

Real 30-min history is only ~60 days deep (yfinance), too thin for a stable offline demo, so this
module manufactures a reproducible intraday world instead. Two honesty-preserving choices:

  * It is a clean US market-hours grid (13 bars/day, 09:30–15:30 starts) so bar-counting lines up
    with ``interval.bars_between``.
  * It plants a SMALL, REAL momentum signal (AR(1) autocorrelation in bar returns), so the model has
    something genuine to discover — the trailing-trend features earn their AUC instead of fitting
    noise. The signal is deliberately weak (intraday is mostly noise), which is the honest regime.

Nothing here is committed as a file; it regenerates identically from the seed, so the app and tests
share one source of truth without repo bloat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..providers.historical_price_provider import HistoricalPriceFileProvider
from .interval import BARS_PER_TRADING_DAY_30M

BENCHMARK = "IDX"
# Fictional tickers (clearly not real securities), with annualized vol and a planted momentum phi.
_UNIVERSE = [
    ("VELO", 0.34, 0.08), ("NOVA", 0.41, 0.10), ("HALC", 0.28, 0.06),
    ("RIVE", 0.46, 0.09), ("QUARK", 0.37, 0.07), ("ZEPH", 0.31, 0.085),
]
_BARS_PER_YEAR = 252 * BARS_PER_TRADING_DAY_30M


def market_hours_index(n_days: int, start: str = "2023-01-03") -> pd.DatetimeIndex:
    """A tz-naive 30-min market-hours index: ``n_days`` business days × 13 bars (09:30…15:30)."""
    days = pd.bdate_range(start, periods=n_days)
    times = [pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30) * k
             for k in range(BARS_PER_TRADING_DAY_30M)]
    return pd.DatetimeIndex([d + t for d in days for t in times])


def _ar1_path(n: int, sigma_bar: float, phi: float, s0: float, rng: np.random.Generator) -> np.ndarray:
    """A price path whose per-bar log returns follow AR(1) (phi = planted momentum), unit-vol scaled."""
    eps = rng.standard_normal(n - 1)
    r = np.empty(n - 1)
    innov = sigma_bar * np.sqrt(1.0 - phi ** 2)  # keep the unconditional per-bar vol == sigma_bar
    r[0] = sigma_bar * eps[0]
    for i in range(1, n - 1):
        r[i] = phi * r[i - 1] + innov * eps[i]
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def generate_intraday_frame(seed: int = 7, n_days: int = 200) -> pd.DataFrame:
    """Reproducible wide frame (index = 30-min bars, columns = tickers + benchmark)."""
    idx = market_hours_index(n_days)
    n = len(idx)
    data: dict[str, np.ndarray] = {}
    for j, (sym, vol_ann, phi) in enumerate(_UNIVERSE):
        rng = np.random.default_rng(seed * 1000 + j)
        data[sym] = _ar1_path(n, vol_ann / np.sqrt(_BARS_PER_YEAR), phi, 100.0 + 10 * j, rng)
    rng_b = np.random.default_rng(seed * 1000 + 999)
    data[BENCHMARK] = _ar1_path(n, 0.16 / np.sqrt(_BARS_PER_YEAR), 0.03, 5000.0, rng_b)
    return pd.DataFrame(data, index=idx)


def intraday_demo_provider(seed: int = 7, n_days: int = 200) -> HistoricalPriceFileProvider:
    """A look-ahead-safe provider over the synthetic intraday world (datetime index preserved)."""
    frame = generate_intraday_frame(seed, n_days)
    return HistoricalPriceFileProvider.from_frame(frame, BENCHMARK)
