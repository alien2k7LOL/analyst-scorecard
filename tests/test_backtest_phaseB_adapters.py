"""Back-test Phase B — file-based adapters load real-shaped data and enforce the policies."""

import json
from pathlib import Path

import pandas as pd
import pytest

from analyst_scorecard.providers.historical_call_provider import (
    HistoricalCallFileProvider,
    normalize_rating,
)
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider
from analyst_scorecard.providers.price_provider import _ts
from analyst_scorecard.schemas import Rating


def _write_dataset(tmp_path: Path) -> Path:
    """Tiny but real-shaped dataset: benchmark + a full ticker + a delisted ticker."""
    days = pd.bdate_range(start="2020-01-06", periods=520)  # ~2 trading years, starts Monday
    rows = []
    for i, d in enumerate(days):
        rows.append((d.date().isoformat(), "BENCH", round(100 + 0.10 * i, 4)))
        rows.append((d.date().isoformat(), "AAA", round(50 + 0.20 * i, 4)))
        if i < 120:  # DEAD delists after ~120 sessions
            rows.append((d.date().isoformat(), "DEAD", round(40 - 0.10 * i, 4)))
    prices = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    prices.to_csv(tmp_path / "prices.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"benchmark_symbol": "BENCH", "default_horizon_months": 12}))

    last_day = days[-1].date().isoformat()
    after_end = (days[-1] + pd.Timedelta(days=10)).date().isoformat()
    near_end = days[400].date().isoformat()
    calls = [
        # good: call_date is a Saturday -> must snap forward; blank horizon -> default 12mo (252td)
        {"call_id": "c-good", "analyst_id": "a1", "analyst_name": "A One", "firm": "Firm1",
         "ticker": "AAA", "rating": "Buy", "target_price": 120, "call_date": "2020-01-11", "horizon_months": ""},
        # synonym rating -> Overweight; explicit 6-month horizon -> 126td
        {"call_id": "c-syn", "analyst_id": "a1", "analyst_name": "A One", "firm": "Firm1",
         "ticker": "AAA", "rating": "Outperform", "target_price": 130, "call_date": "2020-02-03", "horizon_months": 6},
        # bad rating -> dropped
        {"call_id": "c-badrating", "analyst_id": "a2", "analyst_name": "A Two", "firm": "Firm2",
         "ticker": "AAA", "rating": "Frobnicate", "target_price": 100, "call_date": "2020-02-03", "horizon_months": 12},
        # unknown ticker -> dropped
        {"call_id": "c-unknown", "analyst_id": "a2", "analyst_name": "A Two", "firm": "Firm2",
         "ticker": "ZZZ", "rating": "Buy", "target_price": 100, "call_date": "2020-02-03", "horizon_months": 12},
        # no entry price: DEAD has no data at this later date -> dropped
        {"call_id": "c-noentry", "analyst_id": "a2", "analyst_name": "A Two", "firm": "Firm2",
         "ticker": "DEAD", "rating": "Buy", "target_price": 100, "call_date": days[200].date().isoformat(), "horizon_months": 12},
        # horizon beyond data (still open) -> dropped
        {"call_id": "c-open", "analyst_id": "a2", "analyst_name": "A Two", "firm": "Firm2",
         "ticker": "AAA", "rating": "Buy", "target_price": 100, "call_date": near_end, "horizon_months": 12},
        # call date past the last session -> dropped
        {"call_id": "c-future", "analyst_id": "a2", "analyst_name": "A Two", "firm": "Firm2",
         "ticker": "AAA", "rating": "Buy", "target_price": 100, "call_date": after_end, "horizon_months": 12},
    ]
    pd.DataFrame(calls).to_csv(tmp_path / "calls.csv", index=False)
    return tmp_path


# --------------------------------------------------------------------------------------
# Price provider
# --------------------------------------------------------------------------------------


def test_price_provider_loads_calendar_and_ragged_coverage(tmp_path):
    data = _write_dataset(tmp_path)
    p = HistoricalPriceFileProvider(data)
    assert p.benchmark_symbol == "BENCH"
    assert p.tickers() == ["AAA", "DEAD"]  # sorted, benchmark excluded
    assert len(p.trading_days()) == 520     # calendar == benchmark dates
    # DEAD has ragged (early-terminating) coverage
    assert len(p.price_series("DEAD")) == 120
    assert len(p.price_series("AAA")) == 520
    with pytest.raises(KeyError):
        p.price_series("NOPE")


def test_snapping_and_windowing(tmp_path):
    p = HistoricalPriceFileProvider(_write_dataset(tmp_path))
    days = p.trading_days()
    # Saturday snaps forward to the next session
    snapped = p.next_trading_day_on_or_after("2020-01-11")
    assert snapped in days and snapped >= _ts("2020-01-11")
    # window for a delisted ticker whose horizon ends after it stopped trading -> rejected
    with pytest.raises(ValueError):
        p.window_for_call("DEAD", days[10], days[262])


def test_rating_normalization():
    assert normalize_rating("Strong Buy") == Rating.BUY
    assert normalize_rating(" outperform ") == Rating.OVERWEIGHT
    assert normalize_rating("Equal Weight") == Rating.HOLD
    assert normalize_rating("Underperform") == Rating.UNDERWEIGHT
    assert normalize_rating("SELL") == Rating.SELL
    assert normalize_rating("banana") is None


# --------------------------------------------------------------------------------------
# Call provider + ingest policies
# --------------------------------------------------------------------------------------


def test_call_provider_builds_valid_calls_and_logs_drops(tmp_path):
    data = _write_dataset(tmp_path)
    prices = HistoricalPriceFileProvider(data)
    calls_provider = HistoricalCallFileProvider(data, prices)
    calls = calls_provider.get_calls()

    by_id = {c.call_id: c for c in calls}
    assert set(by_id) == {"c-good", "c-syn"}  # only the two resolvable, valid calls survive

    good = by_id["c-good"]
    # snapped forward off the Saturday, onto a real trading day
    assert prices.is_trading_day(good.call_date)
    assert good.call_date > __import__("datetime").date(2020, 1, 11)
    assert good.horizon_days == 252  # default 12 months
    assert good.initial_price == prices.price_on("AAA", good.call_date)
    # deadline = call + 252 benchmark trading days, fixed at record time
    assert good.resolution_date == prices.trading_day_offset(good.call_date, 252).date()

    assert by_id["c-syn"].rating == Rating.OVERWEIGHT
    assert by_id["c-syn"].horizon_days == 126  # explicit 6 months

    reasons = {iss["call_id"]: iss["reason"] for iss in calls_provider.ingest_issues}
    assert reasons["c-badrating"] == "BAD_RATING"
    assert reasons["c-unknown"] == "UNKNOWN_TICKER"
    assert reasons["c-noentry"] == "NO_ENTRY_PRICE"
    assert reasons["c-open"] == "HORIZON_BEYOND_DATA"
    assert reasons["c-future"] == "CALL_DATE_OUT_OF_RANGE"


def test_from_frame_constructor(tmp_path):
    import numpy as np

    days = pd.bdate_range("2021-01-04", periods=300)
    i = np.arange(300)
    frame = pd.DataFrame({"IDX": 100.0 + 0.1 * i, "AAA": 50.0 + 0.2 * i}, index=days)
    p = HistoricalPriceFileProvider.from_frame(frame, benchmark_symbol="IDX")
    assert p.benchmark_symbol == "IDX"
    assert p.tickers() == ["AAA"]
    assert len(p.trading_days()) == 300
