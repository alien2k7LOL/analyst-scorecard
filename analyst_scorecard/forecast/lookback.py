"""``LookbackWindow`` — the look-ahead guard for the PREDICTION side.

Where ``PriceWindow`` bounds resolution to [call_date, resolution_date], ``LookbackWindow`` bounds
the *information a prediction is allowed to use* to (…, as_of]: a price slice that ENDS exactly at
the as_of trading day and physically contains nothing after it. Constructing one with a later
observation raises — so a probability estimate is structurally incapable of seeing its own future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..providers.price_provider import DateLike, PriceDataProvider
from .interval import BarInterval, to_ts


@dataclass(frozen=True)
class LookbackWindow:
    """An immutable price history ENDING exactly at ``as_of`` (nothing later is reachable)."""

    symbol: str
    as_of: pd.Timestamp
    prices: pd.Series  # indexed by Timestamp, sorted ascending, last index == as_of

    def __post_init__(self) -> None:
        s = self.prices
        if not isinstance(s, pd.Series) or len(s) < 2:
            raise ValueError(f"lookback needs >= 2 observations, got {len(s)}")
        if not s.index.is_monotonic_increasing:
            raise ValueError("lookback index must be sorted ascending")
        if (s <= 0).any():
            raise ValueError("lookback contains a non-positive price")
        if s.index.max() != self.as_of:
            # The crucial guard: any observation dated after as_of is future information.
            raise ValueError(
                f"LOOK-AHEAD BLOCKED: lookback must end exactly at as_of {self.as_of.date()}, "
                f"but ends at {s.index.max()} — future data leaked into the prediction inputs."
            )

    @property
    def last_price(self) -> float:
        """The price the prediction acts on (close on the as_of day)."""
        return float(self.prices.iloc[-1])

    @property
    def n_observations(self) -> int:
        return int(len(self.prices))

    def daily_log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.prices.to_numpy(dtype=float)))

    def drift_vol(self) -> tuple[float, float]:
        """(mean, std) of daily LOG returns — the per-day drift and volatility the model uses."""
        r = self.daily_log_returns()
        if len(r) < 2:
            return 0.0, 0.0
        return float(np.mean(r)), float(np.std(r, ddof=1))

    def momentum(self, days: int = 20) -> float:
        """Trailing total log return over the last ``days`` (a simple, testable trend feature)."""
        p = self.prices.to_numpy(dtype=float)
        k = min(days, len(p) - 1)
        return float(np.log(p[-1] / p[-1 - k]))


def lookback_window(
    provider: PriceDataProvider,
    symbol: str,
    as_of: DateLike,
    lookback_days: int = 252,
    interval: BarInterval = BarInterval.DAILY,
) -> LookbackWindow:
    """Build the [.., as_of] lookback for ``symbol`` using ONLY prices on/before as_of.

    ``as_of`` is snapped back to the last available bar on/before it, and the window is the trailing
    ``lookback_days + 1`` bars ending there (bars = trading days for daily, 30-min bars for intraday).
    Nothing after as_of is ever included. ``interval`` only changes how ``as_of`` is normalized — for
    intraday the exact time of day is kept, so the slice doesn't collapse onto midnight.
    """
    series = provider.price_series(symbol)
    asof_ts = to_ts(as_of, interval)
    past = series.loc[:asof_ts]  # label slice: everything on/before as_of
    if len(past) < 2:
        raise ValueError(f"no usable price history for {symbol!r} on/before {asof_ts}")
    tail = past.iloc[-(lookback_days + 1):]
    return LookbackWindow(symbol=symbol, as_of=tail.index[-1], prices=tail)
