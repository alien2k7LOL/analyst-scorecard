"""Live forecaster — offline wiring + self-calibration, with an injected (non-network) fetcher.

The yfinance fetch can't run in the sandbox, but everything around it — self-calibration on the
ticker's own history, feature building, and the fallback when history is too short — runs through
the real code path on an injected synthetic price frame.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.forecast.interval import BarInterval
from analyst_scorecard.forecast.live import grade_forecast_live
from analyst_scorecard.forecast.prediction import PredictionKind
from analyst_scorecard.forecast.synth_intraday import generate_intraday_frame
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


class IntradayFrameFetcher(PriceFetcher):
    """Offline stand-in for the yfinance 30-min fetch: slices a synthetic intraday frame."""

    def __init__(self, frame):
        self.frame = frame

    def fetch(self, symbols, start, end, interval="1d"):
        e = pd.Timestamp(end) + pd.Timedelta(days=1)
        cols = [c for c in symbols if c in self.frame.columns]
        return self.frame.loc[self.frame.index < e, cols].copy()


def test_intraday_30min_grade_self_calibrates_on_intraday_bars():
    frame = generate_intraday_frame(seed=7, n_days=200)
    last = frame.index[-1]
    s0 = float(frame["VELO"].iloc[-1])
    deadline = (last.normalize() + pd.offsets.BDay(2) + pd.Timedelta(hours=12)).to_pydatetime()
    g = grade_forecast_live(
        ticker="VELO", target_price=round(s0 * 1.008, 2), deadline=deadline, direction=Direction.UP,
        kind=PredictionKind.TERMINAL, band_pct=0.006, interval=BarInterval.MIN30,
        as_of=last.date(), benchmark_symbol="IDX", fetcher=IntradayFrameFetcher(frame),
    )
    assert g.interval == BarInterval.MIN30
    assert g.calibrated is True               # ~200 days of 30-min bars is plenty to self-calibrate
    assert 0.0 <= g.probability <= 1.0
    assert g.n_days > 0                        # horizon counted in 30-min bars
    assert ":" in g.deadline_label            # intraday label carries the time of day


def test_intraday_far_short_horizon_target_is_low_probability():
    # The AAPL "$280 -> $285 by 3pm" case: a ~1.8% move in ~1-2 30-min bars is a ~4-5 sigma event.
    # Even with a recent uptrend in the data, intraday is treated as driftless, so the probability
    # must be SMALL — not inflated by extrapolating momentum.
    from analyst_scorecard.forecast.synth_intraday import market_hours_index
    idx = market_hours_index(80)
    n = len(idx)
    rng = np.random.default_rng(11)
    mu = np.where(np.arange(n - 1) > n - 220, 0.0009, 0.0)        # planted recent uptrend
    px = 232.0 * np.exp(np.concatenate([[0.0], np.cumsum(mu + 0.004 * rng.standard_normal(n - 1))]))
    bench = 5000.0 * np.exp(np.cumsum(np.concatenate([[0.0], 0.002 * rng.standard_normal(n - 1)])))
    frame = pd.DataFrame({"AAPL": px, "^GSPC": bench}, index=idx)
    last = idx[-1]
    s0 = float(frame["AAPL"].iloc[-1])
    deadline = (last.normalize() + pd.offsets.BDay(1) + pd.Timedelta(hours=10)).to_pydatetime()  # ~2 bars
    g = grade_forecast_live(
        ticker="AAPL", target_price=round(s0 * 1.018, 2), deadline=deadline, direction=Direction.UP,
        kind=PredictionKind.TERMINAL, interval=BarInterval.MIN30, as_of=last.date(),
        benchmark_symbol="^GSPC", fetcher=IntradayFrameFetcher(frame),
    )
    assert g.drift_bar == 0.0           # intraday is driftless (no momentum extrapolation)
    assert g.raw_probability < 0.05     # far target over ~2 bars -> tiny
    assert g.probability < 0.10         # and the calibrated number stays realistic


def test_intraday_deadline_before_last_bar_is_rejected():
    frame = generate_intraday_frame(seed=7, n_days=200)
    last = frame.index[-1]
    with pytest.raises(LiveGradeError, match="after the latest bar"):
        grade_forecast_live(
            ticker="VELO", target_price=100.0, deadline=(last - pd.Timedelta(hours=1)).to_pydatetime(),
            direction=Direction.UP, kind=PredictionKind.TERMINAL, band_pct=0.006,
            interval=BarInterval.MIN30, as_of=last.date(), benchmark_symbol="IDX",
            fetcher=IntradayFrameFetcher(frame),
        )


def test_terminal_mode_self_calibrates_and_is_band_aware(frame):
    # Terminal grading: lands within ±band of the target ON the deadline. Self-calibration must train
    # on terminal outcomes (the same question), and a wider band must never be less likely.
    common = dict(ticker="AAA", target_price=120.0, deadline=AS_OF + timedelta(days=180),
                  direction=Direction.UP, as_of=AS_OF, kind=PredictionKind.TERMINAL,
                  fetcher=FrameFetcher(frame))
    narrow = grade_forecast_live(**{**common, "band_pct": 0.02})
    wide = grade_forecast_live(**{**common, "band_pct": 0.10})
    assert narrow.kind == PredictionKind.TERMINAL and narrow.band_pct == 0.02
    assert narrow.calibrated is True and 0.0 <= narrow.probability <= 1.0
    assert wide.raw_probability >= narrow.raw_probability   # wider band can only help (raw model)


def test_terminal_is_not_more_likely_than_touch_live(frame):
    common = dict(ticker="AAA", target_price=125.0, deadline=AS_OF + timedelta(days=150),
                  direction=Direction.UP, as_of=AS_OF, fetcher=FrameFetcher(frame))
    touch = grade_forecast_live(**common, kind=PredictionKind.TOUCH)
    terminal = grade_forecast_live(**common, kind=PredictionKind.TERMINAL, band_pct=0.03)
    # Ending in a tight band around the level is rarer than ever touching it (raw, model-level fact).
    assert terminal.raw_probability < touch.raw_probability


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
