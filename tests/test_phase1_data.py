"""Phase 1 — data layer: reproducibility, benchmark existence, fixture validation."""

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.config import DEFAULT_CONFIG, ScorecardConfig
from analyst_scorecard.providers.call_provider import FixtureCallProvider, InMemoryCallProvider
from analyst_scorecard.providers.price_provider import (
    PriceWindow,
    SyntheticPriceDataProvider,
)
from analyst_scorecard.schemas import Call, Direction, Rating


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_prices_are_deterministic_under_fixed_seed():
    p1 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    p2 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    assert p1.frame.equals(p2.frame)


def test_different_seed_gives_different_prices():
    p1 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    p2 = SyntheticPriceDataProvider(DEFAULT_CONFIG.with_overrides(seed=DEFAULT_CONFIG.seed + 1))
    assert not p1.frame.equals(p2.frame)


def test_ticker_seed_is_order_independent():
    """A symbol's path depends only on (seed, its own name) — not on the universe order."""
    base = DEFAULT_CONFIG
    reordered = base.with_overrides(universe=tuple(reversed(base.universe)))
    p_base = SyntheticPriceDataProvider(base)
    p_re = SyntheticPriceDataProvider(reordered)
    for sym in ("ASTR", "HELX", "FALC"):
        pd.testing.assert_series_equal(
            p_base.price_series(sym), p_re.price_series(sym), check_names=True
        )


def test_first_day_price_is_start_price():
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    for spec in DEFAULT_CONFIG.universe:
        assert p.price_series(spec.symbol).iloc[0] == pytest.approx(spec.start_price)
    bspec = DEFAULT_CONFIG.benchmark
    assert p.price_series(bspec.symbol).iloc[0] == pytest.approx(bspec.start_price)


# --------------------------------------------------------------------------------------
# Benchmark & calendar
# --------------------------------------------------------------------------------------


def test_benchmark_series_exists_and_is_separate():
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    assert p.benchmark_symbol == "MKT"
    assert p.benchmark_symbol not in p.tickers()
    s = p.price_series(p.benchmark_symbol)
    assert len(s) == DEFAULT_CONFIG.n_trading_days
    assert (s > 0).all()


def test_trading_day_offset_counts_trading_days():
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    days = p.trading_days()
    start = days[100]
    assert p.trading_day_offset(start, 252) == days[352]
    with pytest.raises(IndexError):
        p.trading_day_offset(days[-5], 252)  # runs past the data range


def test_price_on_rejects_non_trading_day():
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    # A Saturday is not in the business-day index.
    with pytest.raises(KeyError):
        p.price_on("ASTR", "2021-01-09")  # Saturday


# --------------------------------------------------------------------------------------
# PriceWindow bounds (the look-ahead guard surfaces here too)
# --------------------------------------------------------------------------------------


def test_window_for_call_is_bounded_exactly():
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    days = p.trading_days()
    w = p.window_for_call("ASTR", days[10], days[10 + 252])
    assert w.stock.index.min() == days[10]
    assert w.stock.index.max() == days[10 + 252]
    assert w.n_observations == 253
    # reading inside the window works; outside raises
    assert w.price("stock", days[10]) == pytest.approx(w.call_price)
    with pytest.raises(KeyError):
        w.price("stock", days[10 + 253])  # one day past resolution


def test_window_rejects_future_data():
    """Constructing a window whose series extends past the resolution date is rejected."""
    p = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    days = p.trading_days()
    call_d, res_d = days[10], days[10 + 252]
    leaky_stock = p.price_window_series("ASTR", call_d, days[10 + 260])  # too long
    leaky_bench = p.price_window_series("MKT", call_d, res_d)
    with pytest.raises(ValueError, match="LOOK-AHEAD BLOCKED"):
        PriceWindow(
            stock_symbol="ASTR",
            benchmark_symbol="MKT",
            call_date=call_d,
            resolution_date=res_d,
            stock=leaky_stock,
            benchmark=leaky_bench,
        )


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def test_fixture_calls_load_and_validate():
    calls = FixtureCallProvider().get_calls()
    assert len(calls) >= 8 * 10  # >=8 analysts, ~10+ calls each
    analysts = {c.analyst_id for c in calls}
    assert len(analysts) >= 8
    # every rating type appears -> the full rating->direction map is exercised
    ratings = {c.rating for c in calls}
    assert ratings == set(Rating)


def test_calls_are_sorted_and_consistent():
    calls = FixtureCallProvider().get_calls()
    # sorted by (call_date, call_id)
    keys = [(c.call_date, c.call_id) for c in calls]
    assert keys == sorted(keys)
    # every call's resolution date is strictly after its call date
    for c in calls:
        assert c.resolution_date > c.call_date
        assert c.target_price > 0
        assert c.initial_price > 0


def test_implied_direction_and_position_mapping():
    p = lambda **kw: Call(
        call_id="t",
        analyst_id="a",
        analyst_name="A",
        firm="F",
        ticker="ASTR",
        call_date="2021-02-01",
        horizon_days=252,
        resolution_date="2022-02-01",
        initial_price=100.0,
        target_price=110.0,
        **kw,
    )
    assert p(rating=Rating.BUY).implied_direction == Direction.UP
    assert p(rating=Rating.OVERWEIGHT).implied_position == +1
    assert p(rating=Rating.HOLD).implied_direction == Direction.FLAT
    assert p(rating=Rating.HOLD).implied_position == 0
    assert p(rating=Rating.HOLD).is_directional is False
    assert p(rating=Rating.SELL).implied_position == -1
    assert p(rating=Rating.UNDERWEIGHT).implied_direction == Direction.DOWN


def test_call_rejects_bad_values():
    with pytest.raises(Exception):
        Call(
            call_id="t", analyst_id="a", analyst_name="A", firm="F", ticker="ASTR",
            rating=Rating.BUY, target_price=-1.0, call_date="2021-02-01",
            horizon_days=252, resolution_date="2022-02-01", initial_price=100.0,
        )
    with pytest.raises(Exception):  # resolution before call
        Call(
            call_id="t", analyst_id="a", analyst_name="A", firm="F", ticker="ASTR",
            rating=Rating.BUY, target_price=110.0, call_date="2021-02-01",
            horizon_days=252, resolution_date="2020-02-01", initial_price=100.0,
        )


def test_in_memory_provider_roundtrip():
    calls = FixtureCallProvider().get_calls()
    mem = InMemoryCallProvider(calls)
    assert len(mem.get_calls()) == len(calls)
