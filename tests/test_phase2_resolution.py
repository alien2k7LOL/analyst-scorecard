"""Phase 2 — resolution engine: prove the look-ahead guarantee is structural.

Two flavors of proof:
  (a) BEHAVIORAL: mutating every price AFTER the resolution date does not change the
      resolution at all — the resolver provably ignores the future.
  (b) STRUCTURAL: a deliberately constructed leakage attempt (a window that extends past
      the resolution date, or asking a window for a future price, or handing the resolver a
      window resolved on the wrong date) is rejected with an error.
"""

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.providers.call_provider import FixtureCallProvider
from analyst_scorecard.providers.price_provider import (
    PriceWindow,
    SyntheticPriceDataProvider,
    _ts,
)
from analyst_scorecard.resolution import resolve_call, resolve_call_with_provider
from analyst_scorecard.schemas import Call, Rating


@pytest.fixture(scope="module")
def provider():
    return SyntheticPriceDataProvider(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def sample_call():
    # A mid-history Buy with room for its full horizon.
    return FixtureCallProvider().get_calls()[10]


def _window_from_frame(frame: pd.DataFrame, call: Call, benchmark: str) -> PriceWindow:
    c, r = _ts(call.call_date), _ts(call.resolution_date)
    return PriceWindow(
        stock_symbol=call.ticker,
        benchmark_symbol=benchmark,
        call_date=c,
        resolution_date=r,
        stock=frame[call.ticker].loc[c:r],
        benchmark=frame[benchmark].loc[c:r],
    )


# --------------------------------------------------------------------------------------
# Basic correctness
# --------------------------------------------------------------------------------------


def test_resolution_returns_match_manual_math(provider, sample_call):
    res = resolve_call_with_provider(sample_call, provider)
    p0 = provider.price_on(sample_call.ticker, sample_call.call_date)
    p1 = provider.price_on(sample_call.ticker, sample_call.resolution_date)
    b0 = provider.price_on("MKT", sample_call.call_date)
    b1 = provider.price_on("MKT", sample_call.resolution_date)
    assert res.call_price == pytest.approx(p0)
    assert res.actual_price == pytest.approx(p1)
    assert res.stock_return == pytest.approx(p1 / p0 - 1)
    assert res.benchmark_return == pytest.approx(b1 / b0 - 1)
    assert res.realized_horizon_vol > 0
    assert res.n_observations == sample_call.horizon_days + 1


def test_resolution_is_deterministic(provider, sample_call):
    assert resolve_call_with_provider(sample_call, provider) == resolve_call_with_provider(
        sample_call, provider
    )


# --------------------------------------------------------------------------------------
# (a) BEHAVIORAL leakage proof — future prices cannot change the resolution
# --------------------------------------------------------------------------------------


def test_mutating_future_prices_does_not_change_resolution(provider, sample_call):
    res_clean = resolve_call_with_provider(sample_call, provider)

    # Replace EVERY price strictly after the resolution date with garbage (x1000).
    tampered = provider.frame.copy()
    future = tampered.index > _ts(sample_call.resolution_date)
    assert future.any(), "test needs data after the resolution date to be meaningful"
    tampered.loc[future] = tampered.loc[future] * 1000.0

    res_tampered = resolve_call(sample_call, _window_from_frame(tampered, sample_call, "MKT"))
    assert res_tampered == res_clean  # identical: the future never entered the calculation


def test_mutating_pre_call_prices_does_not_change_resolution(provider, sample_call):
    """Symmetrically, prices BEFORE the call date are also outside the window and ignored."""
    res_clean = resolve_call_with_provider(sample_call, provider)
    tampered = provider.frame.copy()
    past = tampered.index < _ts(sample_call.call_date)
    tampered.loc[past] = tampered.loc[past] * 0.001
    res_tampered = resolve_call(sample_call, _window_from_frame(tampered, sample_call, "MKT"))
    assert res_tampered == res_clean


# --------------------------------------------------------------------------------------
# (b) STRUCTURAL leakage attempts are caught
# --------------------------------------------------------------------------------------


def test_leakage_window_extending_past_resolution_is_rejected(provider, sample_call):
    days = provider.trading_days()
    c = _ts(sample_call.call_date)
    r = _ts(sample_call.resolution_date)
    r_plus = days[days.get_indexer([r])[0] + 5]  # 5 trading days of FUTURE data
    leaky_stock = provider.price_window_series(sample_call.ticker, c, r_plus)
    bench = provider.price_window_series("MKT", c, r)
    with pytest.raises(ValueError, match="LOOK-AHEAD BLOCKED"):
        PriceWindow(
            stock_symbol=sample_call.ticker,
            benchmark_symbol="MKT",
            call_date=c,
            resolution_date=r,
            stock=leaky_stock,
            benchmark=bench,
        )


def test_asking_window_for_a_future_price_is_rejected(provider, sample_call):
    days = provider.trading_days()
    w = provider.window_for_call(sample_call.ticker, sample_call.call_date, sample_call.resolution_date)
    future_day = days[days.get_indexer([_ts(sample_call.resolution_date)])[0] + 1]
    with pytest.raises(KeyError, match="LOOK-AHEAD BLOCKED"):
        w.price("stock", future_day)


def test_resolver_rejects_window_resolved_on_wrong_date(provider, sample_call):
    """A window resolved LATER than the call's record-time deadline is a look-ahead back
    door — the resolver refuses it instead of silently scoring extra horizon."""
    days = provider.trading_days()
    later = days[days.get_indexer([_ts(sample_call.resolution_date)])[0] + 10]
    wrong_window = provider.window_for_call(sample_call.ticker, sample_call.call_date, later)
    with pytest.raises(ValueError, match="resolution_date"):
        resolve_call(sample_call, wrong_window)


def test_resolver_rejects_mismatched_ticker(provider, sample_call):
    other = "JUNI" if sample_call.ticker != "JUNI" else "EVRG"
    wrong_window = provider.window_for_call(other, sample_call.call_date, sample_call.resolution_date)
    with pytest.raises(ValueError, match="but call is for"):
        resolve_call(sample_call, wrong_window)


def test_every_fixture_call_resolves(provider):
    """The whole synthetic dataset resolves cleanly (no off-by-one / out-of-range calls)."""
    for call in FixtureCallProvider().get_calls():
        res = resolve_call_with_provider(call, provider)
        assert res.actual_price > 0
        assert res.n_observations == call.horizon_days + 1
