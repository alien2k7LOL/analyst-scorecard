"""Resolution engine — the look-ahead-safe core.

``resolve_call`` turns a ``Call`` + a bounded ``PriceWindow`` into a ``Resolution``: the
actual horizon price, the stock and benchmark returns over the window, and the realized
volatility the accuracy stage needs. It computes the inputs for ALL THREE scoring stages
but makes no judgements — scoring lives in scoring.py.

THE LOOK-AHEAD GUARANTEE (structural, not by convention)
--------------------------------------------------------
``resolve_call`` is handed a ``PriceWindow``, never a provider. A window physically contains
only the prices in [call_date, resolution_date] and raises if asked for anything outside that
range or if constructed with data past the resolution date. Therefore the resolver cannot use
future information even in principle. ``resolve_call_with_provider`` is the only place a window
is built, and it slices strictly up to the (record-time-fixed) resolution date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .providers.price_provider import PriceDataProvider, PriceWindow, _ts
from .schemas import Call, Resolution


def resolve_call(call: Call, window: PriceWindow) -> Resolution:
    """Resolve a single call from its bounded price window. No future data is reachable.

    The window must correspond exactly to the call (same ticker and the same record-time
    call/resolution dates); a mismatch is a programming error and raises, so a misaligned
    (and possibly leaky) window can never be silently scored.
    """
    _assert_window_matches_call(call, window)

    call_price = window.call_price
    actual_price = window.actual_price
    bench_call = window.benchmark_call_price
    bench_actual = window.benchmark_actual_price

    stock_return = actual_price / call_price - 1.0
    benchmark_return = bench_actual / bench_call - 1.0

    realized_horizon_vol = _realized_horizon_vol(window)

    return Resolution(
        call_id=call.call_id,
        call_date=call.call_date,
        resolution_date=call.resolution_date,
        call_price=call_price,
        target_price=call.target_price,
        actual_price=actual_price,
        benchmark_call_price=bench_call,
        benchmark_actual_price=bench_actual,
        stock_return=stock_return,
        benchmark_return=benchmark_return,
        realized_horizon_vol=realized_horizon_vol,
        n_observations=window.n_observations,
    )


def resolve_call_with_provider(call: Call, provider: PriceDataProvider) -> Resolution:
    """Build the bounded window for the call (sliced up to its resolution date) and resolve.

    This is the ONLY sanctioned bridge from a full provider to the resolver. The slice ends
    at the call's record-time resolution date, so no later price ever enters the window.
    """
    window = provider.window_for_call(call.ticker, call.call_date, call.resolution_date)
    return resolve_call(call, window)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _assert_window_matches_call(call: Call, window: PriceWindow) -> None:
    if window.stock_symbol != call.ticker:
        raise ValueError(
            f"window is for {window.stock_symbol!r} but call is for {call.ticker!r}"
        )
    if window.call_date != _ts(call.call_date):
        raise ValueError(
            f"window call_date {window.call_date.date()} != call.call_date {call.call_date}"
        )
    if window.resolution_date != _ts(call.resolution_date):
        # Guards against scoring a call against a window resolved on the wrong (e.g. later)
        # date — a back door for look-ahead. The deadline is fixed at record time; honor it.
        raise ValueError(
            f"window resolution_date {window.resolution_date.date()} != "
            f"call.resolution_date {call.resolution_date}"
        )


def _realized_horizon_vol(window: PriceWindow) -> float:
    """Realized stock volatility over the horizon = daily-return std * sqrt(horizon days).

    Sample std (ddof=1) of the daily LOG returns inside the window, scaled to the full
    horizon by sqrt(number of daily steps). This is the per-stock yardstick the accuracy
    stage divides by, so that a given price miss is judged against how much THAT stock
    actually moved around over THIS horizon.
    """
    log_rets = window.stock_log_returns()
    if len(log_rets) < 2:
        return 0.0
    sigma_daily = float(np.std(log_rets, ddof=1))
    return sigma_daily * np.sqrt(window.horizon_steps)
