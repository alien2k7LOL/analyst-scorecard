"""Price data: the ``PriceDataProvider`` interface and a deterministic synthetic default.

Two ideas carry the fairness guarantee here:

1. ``SyntheticPriceDataProvider`` generates fully reproducible daily price paths from the
   seed in ``ScorecardConfig`` (seeded geometric Brownian motion, one independent RNG
   stream per symbol). Same seed -> identical prices, forever.

2. ``PriceWindow`` is a BOUNDED slice of price history, [call_date, resolution_date]. The
   resolver is only ever handed a window, never the full provider, so it is structurally
   incapable of seeing a price after the resolution date. Constructing a window that
   contains future data, or asking a window for a price outside its bounds, RAISES — which
   is how the look-ahead tests catch a leakage attempt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Union

import numpy as np
import pandas as pd

from ..config import DEFAULT_CONFIG, TRADING_DAYS_PER_YEAR, ScorecardConfig

DateLike = Union[date, pd.Timestamp, str]


def _ts(d: DateLike) -> pd.Timestamp:
    """Normalize any date-like to a midnight Timestamp (the provider's index convention)."""
    return pd.Timestamp(d).normalize()


# --------------------------------------------------------------------------------------
# PriceWindow — the structural look-ahead guard
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceWindow:
    """An immutable price slice [call_date, resolution_date] for one stock + the benchmark.

    INVARIANT (enforced in __post_init__): each series starts exactly at ``call_date`` and
    ends exactly at ``resolution_date``. A series carrying any observation past the
    resolution date is rejected as a leakage attempt. Reading a price outside the window
    raises. This is what makes look-ahead bias structurally impossible in the resolver.
    """

    stock_symbol: str
    benchmark_symbol: str
    call_date: pd.Timestamp
    resolution_date: pd.Timestamp
    stock: pd.Series       # indexed by Timestamp, sorted ascending
    benchmark: pd.Series

    def __post_init__(self) -> None:
        for name, s in (("stock", self.stock), ("benchmark", self.benchmark)):
            if not isinstance(s, pd.Series) or len(s) < 2:
                raise ValueError(f"{name} window needs >= 2 observations, got {len(s)}")
            if not s.index.is_monotonic_increasing:
                raise ValueError(f"{name} window index must be sorted ascending")
            idx_min, idx_max = s.index.min(), s.index.max()
            if idx_min != self.call_date:
                raise ValueError(
                    f"{name} window must start exactly at call_date {self.call_date.date()}, "
                    f"starts at {idx_min}"
                )
            if idx_max != self.resolution_date:
                # The crucial leakage guard: any data at/after a later date than the agreed
                # resolution date means future information has leaked into the window.
                raise ValueError(
                    f"LOOK-AHEAD BLOCKED: {name} window must end exactly at resolution_date "
                    f"{self.resolution_date.date()}, but ends at {idx_max} — future data leaked."
                )
            if (s <= 0).any():
                raise ValueError(f"{name} window contains a non-positive price")

    # -- bounded reads -----------------------------------------------------------------
    def _series(self, which: str) -> pd.Series:
        if which == "stock":
            return self.stock
        if which == "benchmark":
            return self.benchmark
        raise ValueError(f"which must be 'stock' or 'benchmark', got {which!r}")

    def price(self, which: str, on: DateLike) -> float:
        """Read a price inside the window. Reading outside [call,resolution] RAISES."""
        on = _ts(on)
        if on < self.call_date or on > self.resolution_date:
            raise KeyError(
                f"LOOK-AHEAD BLOCKED: {on.date()} is outside the permitted window "
                f"[{self.call_date.date()}, {self.resolution_date.date()}]"
            )
        series = self._series(which)
        if on not in series.index:
            raise KeyError(f"{on.date()} is not a trading day in the window")
        return float(series.loc[on])

    @property
    def call_price(self) -> float:
        return float(self.stock.loc[self.call_date])

    @property
    def actual_price(self) -> float:
        return float(self.stock.loc[self.resolution_date])

    @property
    def benchmark_call_price(self) -> float:
        return float(self.benchmark.loc[self.call_date])

    @property
    def benchmark_actual_price(self) -> float:
        return float(self.benchmark.loc[self.resolution_date])

    @property
    def n_observations(self) -> int:
        return int(len(self.stock))

    @property
    def horizon_steps(self) -> int:
        """Number of daily steps in the window (observations - 1)."""
        return self.n_observations - 1

    def stock_log_returns(self) -> np.ndarray:
        """Daily log returns of the stock over the window (used for realized vol)."""
        prices = self.stock.to_numpy(dtype=float)
        return np.diff(np.log(prices))


# --------------------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------------------


class PriceDataProvider(ABC):
    """Interface for daily price history of a stock universe plus one benchmark index.

    A future real provider (market-data API / yfinance) implements the same surface; the
    engine never depends on the concrete class.
    """

    @property
    @abstractmethod
    def benchmark_symbol(self) -> str: ...

    @abstractmethod
    def tickers(self) -> list[str]:
        """Tradeable tickers (universe), excluding the benchmark."""

    @abstractmethod
    def trading_days(self) -> pd.DatetimeIndex: ...

    @abstractmethod
    def price_series(self, symbol: str) -> pd.Series: ...

    def price_on(self, symbol: str, on: DateLike) -> float:
        on = _ts(on)
        s = self.price_series(symbol)
        if on not in s.index:
            raise KeyError(f"{on.date()} is not a trading day for {symbol}")
        return float(s.loc[on])

    def price_window_series(self, symbol: str, start: DateLike, end: DateLike) -> pd.Series:
        """Inclusive [start, end] slice of one symbol's series."""
        start, end = _ts(start), _ts(end)
        if end <= start:
            raise ValueError(f"end {end.date()} must be after start {start.date()}")
        s = self.price_series(symbol)
        return s.loc[start:end]

    def window_for_call(self, ticker: str, call_date: DateLike, resolution_date: DateLike) -> PriceWindow:
        """Build the bounded [call_date, resolution_date] window for a stock + benchmark.

        This is the ONLY sanctioned way to feed prices to the resolver. The slice is taken
        up to and including ``resolution_date`` and nothing later, so the resolver cannot
        see the future even in principle.
        """
        call_ts, res_ts = _ts(call_date), _ts(resolution_date)
        stock = self.price_window_series(ticker, call_ts, res_ts)
        bench = self.price_window_series(self.benchmark_symbol, call_ts, res_ts)
        return PriceWindow(
            stock_symbol=ticker,
            benchmark_symbol=self.benchmark_symbol,
            call_date=call_ts,
            resolution_date=res_ts,
            stock=stock,
            benchmark=bench,
        )

    def trading_day_offset(self, start: DateLike, n_trading_days: int) -> pd.Timestamp:
        """The trading day exactly ``n_trading_days`` after ``start`` (for fixing deadlines)."""
        days = self.trading_days()
        start = _ts(start)
        pos = days.get_indexer([start])[0]
        if pos == -1:
            raise KeyError(f"{start.date()} is not a trading day")
        target = pos + n_trading_days
        if target >= len(days):
            raise IndexError(
                f"{n_trading_days} trading days after {start.date()} runs past the data range"
            )
        return days[target]

    def is_trading_day(self, on: DateLike) -> bool:
        return _ts(on) in self.trading_days()


# --------------------------------------------------------------------------------------
# Synthetic, seeded, reproducible implementation (offline default)
# --------------------------------------------------------------------------------------


class SyntheticPriceDataProvider(PriceDataProvider):
    """Reproducible synthetic daily prices via seeded geometric Brownian motion.

    For each symbol with annualized (drift mu, vol sigma) and start price P0, daily log
    steps are  (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z,  dt = 1/252, Z ~ N(0,1) i.i.d.,
    and prices are P0 * exp(cumsum(steps)) with the first day pinned to P0 exactly.

    Each symbol draws from its OWN RNG, seeded by ``config.ticker_seed(symbol)`` (a hash of
    the master seed and the symbol). Consequence: a symbol's path depends only on the seed
    and its own name — adding, removing, or reordering tickers never perturbs another's
    path. This is the backbone of reproducibility.
    """

    def __init__(self, config: ScorecardConfig = DEFAULT_CONFIG):
        self._config = config
        self._index = pd.bdate_range(start=config.sim_start, periods=config.n_trading_days)
        self._frame = self._generate()

    @property
    def config(self) -> ScorecardConfig:
        return self._config

    @property
    def benchmark_symbol(self) -> str:
        return self._config.benchmark.symbol

    def tickers(self) -> list[str]:
        return [t.symbol for t in self._config.universe]

    def trading_days(self) -> pd.DatetimeIndex:
        return self._index

    def price_series(self, symbol: str) -> pd.Series:
        if symbol not in self._frame.columns:
            raise KeyError(f"Unknown symbol: {symbol!r}")
        return self._frame[symbol]

    @property
    def frame(self) -> pd.DataFrame:
        """Full price panel (index = trading days, columns = symbols incl. benchmark)."""
        return self._frame

    # -- generation --------------------------------------------------------------------
    def _simulate_one(self, spec) -> np.ndarray:
        cfg = self._config
        n = cfg.n_trading_days
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        rng = np.random.default_rng(cfg.ticker_seed(spec.symbol))
        z = rng.standard_normal(n - 1)
        drift_term = (spec.drift - 0.5 * spec.vol ** 2) * dt
        diffusion = spec.vol * np.sqrt(dt) * z
        log_steps = drift_term + diffusion
        log_path = np.concatenate([[0.0], np.cumsum(log_steps)])
        return spec.start_price * np.exp(log_path)

    def _generate(self) -> pd.DataFrame:
        cfg = self._config
        data = {}
        for spec in cfg.universe:
            data[spec.symbol] = self._simulate_one(spec)
        data[cfg.benchmark.symbol] = self._simulate_one(cfg.benchmark)
        frame = pd.DataFrame(data, index=self._index)
        frame.index.name = "date"
        return frame
