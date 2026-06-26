"""PROVE the forecast subsystem never sees its own future — prices OR news.

Forecasting makes look-ahead the cardinal risk (the "future" sits in the file). These tests would
FAIL if any price after a prediction's deadline, or any news after its as_of date, changed a result.
"""

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.forecast.backtest import resolve_outcome
from analyst_scorecard.forecast.features import build_features
from analyst_scorecard.forecast.lookback import LookbackWindow, lookback_window
from analyst_scorecard.forecast.news import NewsFileProvider, NewsWindow
from analyst_scorecard.forecast.prediction import Prediction
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider
from analyst_scorecard.schemas import Direction

BENCH = "BENCH"


def _prov(aaa_values, days=None):
    days = days if days is not None else pd.bdate_range("2020-01-01", periods=len(aaa_values))
    frame = pd.DataFrame({"AAA": list(map(float, aaa_values)), BENCH: [1000.0] * len(days)}, index=days)
    return HistoricalPriceFileProvider.from_frame(frame, BENCH), frame, days


# ---- outcome resolution -------------------------------------------------------------


def test_resolve_outcome_ignores_everything_after_the_deadline():
    rise = list(np.linspace(100, 130, 101))    # touches 125 before the deadline
    full = rise + [50.0] * 80                   # crashes AFTER the deadline
    prov, frame, days = _prov(full)
    pred = Prediction(prediction_id="t", ticker="AAA", as_of=days[0].date(),
                      target_price=125.0, deadline=days[100].date(), direction=Direction.UP)
    base = resolve_outcome(prov, pred)
    assert base.hit is True

    tampered = frame.copy(); tampered.loc[tampered.index > days[100]] *= 1000.0
    cut = HistoricalPriceFileProvider.from_frame(frame.loc[frame.index <= days[100]], BENCH)
    assert resolve_outcome(HistoricalPriceFileProvider.from_frame(tampered, BENCH), pred) == base
    assert resolve_outcome(cut, pred) == base


def test_target_touched_only_after_deadline_does_not_count():
    # Reaches 125 only at day 150 (after the day-100 deadline) -> must be a MISS.
    slow = list(np.linspace(100, 124, 151)) + [130.0] * 40
    prov, _, days = _prov(slow)
    pred = Prediction(prediction_id="t", ticker="AAA", as_of=days[0].date(),
                      target_price=125.0, deadline=days[100].date(), direction=Direction.UP)
    assert resolve_outcome(prov, pred).hit is False
    # Move the same crossing INSIDE the deadline and it becomes a hit — outcome is window-driven.
    fast = list(np.linspace(100, 130, 90)) + [130.0] * 101
    prov2, _, days2 = _prov(fast)
    pred2 = pred.model_copy(update={"deadline": days2[100].date()})
    assert resolve_outcome(prov2, pred2).hit is True


# ---- feature building (the prediction inputs) ---------------------------------------


def test_features_ignore_post_as_of_prices():
    rise = list(np.linspace(100, 130, 60))
    full = rise + [5_000.0] * 71               # absurd future spike (len 131 so day 120 exists)
    prov_full, frame, days = _prov(full)
    as_of = days[59].date()
    pred = Prediction(prediction_id="t", ticker="AAA", as_of=as_of,
                      target_price=140.0, deadline=days[120].date(), direction=Direction.UP)

    base = build_features(prov_full, pred)
    # Truncating everything after as_of leaves features unchanged: they only use data up to as_of
    # (the deadline merely sets n_days via a business-day count, which touches no future price).
    cut = HistoricalPriceFileProvider.from_frame(frame.loc[frame.index <= days[59]], BENCH)
    assert build_features(cut, pred).features == base.features
    assert build_features(cut, pred).gbm_p == base.gbm_p


def test_features_ignore_future_news():
    rise = list(np.linspace(100, 130, 120))
    prov, _, days = _prov(rise)
    as_of = days[60]
    past_and_future = pd.DataFrame({
        "date": [days[40].date().isoformat(), days[55].date().isoformat(),
                 days[80].date().isoformat(), days[110].date().isoformat()],  # last two are AFTER as_of
        "symbol": ["AAA"] * 4,
        "sentiment": [0.6, 0.5, -0.9, 0.9],
    })
    only_past = past_and_future.iloc[:2]

    pred = Prediction(prediction_id="t", ticker="AAA", as_of=as_of.date(),
                      target_price=145.0, deadline=days[119].date(), direction=Direction.UP)
    f_future = build_features(prov, pred, NewsFileProvider.from_frame(past_and_future))
    f_past = build_features(prov, pred, NewsFileProvider.from_frame(only_past))
    assert f_future.features == f_past.features  # future articles never entered the window


# ---- the structural guards themselves ------------------------------------------------


def test_lookback_window_rejects_future_data():
    days = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(np.linspace(100, 110, 10), index=days)
    with pytest.raises(ValueError, match="LOOK-AHEAD"):
        LookbackWindow("AAA", days[5], s)  # series runs past the as_of day


def test_news_window_rejects_future_event():
    e = pd.DataFrame({"date": pd.to_datetime(["2020-01-10", "2020-02-01"]), "sentiment": [0.5, -0.3]})
    with pytest.raises(ValueError, match="LOOK-AHEAD"):
        NewsWindow("AAA", pd.Timestamp("2020-01-15"), e)  # 02-01 is after as_of
