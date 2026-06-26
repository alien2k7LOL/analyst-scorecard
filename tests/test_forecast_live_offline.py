"""Live forecaster — offline wiring + self-calibration, with an injected (non-network) fetcher.

The yfinance fetch can't run in the sandbox, but everything around it — self-calibration on the
ticker's own history, feature building, and the fallback when history is too short — runs through
the real code path on an injected synthetic price frame.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.forecast.live import grade_forecast_live
from analyst_scorecard.providers.live_web_price_provider import LiveGradeError, PriceFetcher
from analyst_scorecard.schemas import Direction

BENCH = "^GSPC"
AS_OF = date(2024, 1, 2)


class FrameFetcher(PriceFetcher):
    def __init__(self, frame):
        self.frame = frame

    def fetch(self, symbols, start, end):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        cols = [c for c in symbols if c in self.frame.columns]
        return self.frame.loc[(self.frame.index >= s) & (self.frame.index <= e), cols].copy()


def _frame(n=1600, seed=1, as_of=AS_OF):
    days = pd.bdate_range(end=pd.Timestamp(as_of), periods=n)
    rng = np.random.default_rng(seed)

    def gbm(mu, sig, s0):
        steps = (mu - 0.5 * sig ** 2) / 252 + sig * np.sqrt(1 / 252) * rng.standard_normal(n - 1)
        return s0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))

    return pd.DataFrame({"AAA": gbm(0.10, 0.30, 100.0), BENCH: gbm(0.07, 0.15, 4000.0)}, index=days)


@pytest.fixture(scope="module")
def frame():
    return _frame()


def test_grade_self_calibrates_on_ticker_history(frame):
    g = grade_forecast_live(
        ticker="AAA", target_price=120.0, deadline=AS_OF + timedelta(days=180),
        direction=Direction.UP, as_of=AS_OF, benchmark_symbol=BENCH, fetcher=FrameFetcher(frame),
    )
    assert 0.0 <= g.probability <= 1.0
    assert 0.0 <= g.raw_probability <= 1.0
    assert g.calibrated is True            # ~6y of history is plenty to self-calibrate
    assert g.self_cal_metrics and g.reliability
    assert g.n_days > 0 and g.s0 > 0
    assert g.history_end <= AS_OF          # never fetched past as_of


def test_direction_matters(frame):
    up = grade_forecast_live(ticker="AAA", target_price=130.0, deadline=AS_OF + timedelta(days=120),
                             direction=Direction.UP, as_of=AS_OF, fetcher=FrameFetcher(frame))
    down = grade_forecast_live(ticker="AAA", target_price=70.0, deadline=AS_OF + timedelta(days=120),
                               direction=Direction.DOWN, as_of=AS_OF, fetcher=FrameFetcher(frame))
    assert up.probability != down.probability  # different targets/directions -> different odds


def test_short_history_falls_back_to_raw():
    small = _frame(n=150, seed=3)
    g = grade_forecast_live(ticker="AAA", target_price=120.0, deadline=AS_OF + timedelta(days=120),
                            direction=Direction.UP, as_of=AS_OF, fetcher=FrameFetcher(small))
    assert g.calibrated is False
    assert g.probability == g.raw_probability   # no self-calibration -> raw GBM probability


def test_deadline_in_the_past_is_rejected(frame):
    with pytest.raises(LiveGradeError, match="future"):
        grade_forecast_live(ticker="AAA", target_price=120.0, deadline=AS_OF - timedelta(days=10),
                            direction=Direction.UP, as_of=AS_OF, fetcher=FrameFetcher(frame))


def test_unknown_ticker_is_friendly(frame):
    with pytest.raises(LiveGradeError, match="No live price data for ticker"):
        grade_forecast_live(ticker="ZZZZ", target_price=10.0, deadline=AS_OF + timedelta(days=90),
                            direction=Direction.UP, as_of=AS_OF, fetcher=FrameFetcher(frame))
