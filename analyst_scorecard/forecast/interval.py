"""Bar interval — the one place that knows whether a "step" is a trading day or a 30-min bar.

The probability engine is scale-free: it works in *bars*, with a per-bar drift and volatility and a
horizon counted in bars. This module is the thin adapter that lets the SAME engine run at daily or
intraday resolution by answering two questions for each interval:

  * ``to_ts`` — how to normalize a timestamp (daily collapses to midnight, the daily index
    convention; intraday keeps the exact 30-min time, or the bars would all collapse onto one day).
  * ``bars_between`` — how many bars separate two timestamps (business-day count for daily; a
    market-hours-aware 30-min count for intraday).

Defaulting everything to DAILY makes the daily path byte-identical to before (``to_ts`` ==
the old ``_ts``; ``bars_between`` == the old ``trading_days_to``), so nothing daily changes.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR

# US equities regular session 09:30–16:00 = 6.5h = 13 thirty-minute bars (labelled by start time).
BARS_PER_TRADING_DAY_30M = 13
_OPEN = pd.Timedelta(hours=9, minutes=30)
_BAR = pd.Timedelta(minutes=30)


class BarInterval(str, Enum):
    DAILY = "1d"
    MIN30 = "30m"

    @property
    def bars_per_year(self) -> int:
        if self is BarInterval.DAILY:
            return TRADING_DAYS_PER_YEAR
        return TRADING_DAYS_PER_YEAR * BARS_PER_TRADING_DAY_30M

    @property
    def bars_per_day(self) -> int:
        return 1 if self is BarInterval.DAILY else BARS_PER_TRADING_DAY_30M

    @property
    def label(self) -> str:
        return "trading day" if self is BarInterval.DAILY else "30-min bar"

    @property
    def horizon_word(self) -> str:
        return "trading days" if self is BarInterval.DAILY else "30-min bars"


def to_ts(x, interval: BarInterval = BarInterval.DAILY) -> pd.Timestamp:
    """Daily: normalize to midnight (the daily index convention). Intraday: keep the exact time."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize() if interval is BarInterval.DAILY else ts


def _intraday_bars_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Count 30-min market bars in (a, b] — market-hours aware (skips nights/weekends)."""
    if b <= a:
        return 0
    total = 0
    for day in pd.bdate_range(a.normalize(), b.normalize()):
        open_ = day + _OPEN
        for k in range(BARS_PER_TRADING_DAY_30M):
            bar_ts = open_ + _BAR * k  # bar start times 09:30, 10:00, … 15:30
            if a < bar_ts <= b:
                total += 1
    return total


def bars_between(a, b, interval: BarInterval = BarInterval.DAILY) -> int:
    """Number of bars from ``a`` (exclusive) to ``b`` (inclusive) at this interval."""
    if interval is BarInterval.DAILY:
        return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(b).date()))
    return _intraday_bars_between(pd.Timestamp(a), pd.Timestamp(b))
