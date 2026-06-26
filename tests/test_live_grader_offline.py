"""Live Grader — offline wiring + look-ahead proof (no network, no yfinance).

The live web fetch itself can't run in this sandbox (no network), but the part that MATTERS
for correctness — provider -> resolve_call_with_provider -> score_call, and the look-ahead
bound — runs entirely on an INJECTED price frame through the exact same code path the real
yfinance fetcher feeds. If the live path could leak the future or mis-wire the engine, these
fail.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.providers.live_web_price_provider import (
    LiveGradeError,
    LiveWebPriceProvider,
    PriceFetcher,
    grade_live_prediction,
)
from analyst_scorecard.schemas import Rating

BENCH = "^GSPC"


class FrameFetcher(PriceFetcher):
    """A deterministic stand-in for yfinance: returns a slice of a fixed in-memory frame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch(self, symbols, start, end):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        cols = [c for c in symbols if c in self.frame.columns]
        return self.frame.loc[(self.frame.index >= s) & (self.frame.index <= e), cols].copy()


@pytest.fixture(scope="module")
def days():
    return pd.bdate_range("2025-01-02", periods=300)


@pytest.fixture(scope="module")
def frame(days):
    # AAA rises 100 -> 130 over the first half, then falls to 60. Benchmark rises steadily.
    aaa = np.concatenate([np.linspace(100, 130, 150), np.linspace(130, 60, 150)])
    bench = np.linspace(4000, 4300, 300)
    return pd.DataFrame({"AAA": aaa, BENCH: bench}, index=days)


def _grade(frame, **kw):
    return grade_live_prediction(ticker="AAA", benchmark_symbol=BENCH, fetcher=FrameFetcher(frame), **kw)


# --------------------------------------------------------------------------------------
# (1) End-to-end wiring: it grades through the real engine and returns a funnel result.
# --------------------------------------------------------------------------------------


def test_open_ended_call_grades_provisionally(frame, days):
    res = _grade(
        frame,
        rating=Rating.BUY,
        target_price=130.0,
        call_date=days[0].date(),
        asof=days[120].date(),     # no horizon, no deadline -> "so far"
    )
    assert res.provisional is True
    assert res.graded_through == days[120].date()
    assert res.call_score is not None
    # AAA rose over [0,120], so a Buy clears the direction gate and beat-market exists.
    assert res.call_score.direction_pass is True
    assert res.beat_market is not None


# --------------------------------------------------------------------------------------
# (2) THE look-ahead proof: post-grading-date prices cannot change a live grade.
# --------------------------------------------------------------------------------------


def test_live_grade_ignores_everything_after_the_grading_date(frame, days):
    base = _grade(frame, rating=Rating.BUY, target_price=130.0,
                  call_date=days[0].date(), asof=days[120].date())

    # (a) Truncate the underlying frame at the grading date -> identical score.
    trunc = frame.loc[frame.index <= days[120]]
    cut = _grade(trunc, rating=Rating.BUY, target_price=130.0,
                 call_date=days[0].date(), asof=days[120].date())

    # (b) Multiply ALL post-grading-date prices by 1000 -> still identical (they're out of window).
    tampered = frame.copy()
    tampered.loc[tampered.index > days[120]] *= 1000.0
    tamper = _grade(tampered, rating=Rating.BUY, target_price=130.0,
                    call_date=days[0].date(), asof=days[120].date())

    assert base.call_score == cut.call_score == tamper.call_score  # byte-identical: no leak


# --------------------------------------------------------------------------------------
# (3) FINAL vs PROVISIONAL: an already-elapsed deadline grades FINAL, at the deadline.
# --------------------------------------------------------------------------------------


def test_elapsed_deadline_is_final_and_pinned_to_the_deadline(frame, days):
    res = _grade(
        frame,
        rating=Rating.BUY,
        target_price=125.0,
        call_date=days[0].date(),
        resolution_date=days[100].date(),
        asof=days[200].date(),          # today is well past the deadline
    )
    assert res.provisional is False
    assert res.graded_through == days[100].date()

    # Look-ahead again: moving 'today' later must not move a FINAL grade (deadline is fixed).
    later = _grade(frame, rating=Rating.BUY, target_price=125.0,
                   call_date=days[0].date(), resolution_date=days[100].date(),
                   asof=days[260].date())
    assert later.call_score == res.call_score


def test_future_deadline_is_provisional(frame, days):
    res = _grade(
        frame,
        rating=Rating.BUY,
        target_price=130.0,
        call_date=days[0].date(),
        horizon_trading_days=999,       # horizon runs past the available data
        asof=days[120].date(),
    )
    assert res.provisional is True
    assert res.graded_through == days[120].date()


# --------------------------------------------------------------------------------------
# (4) The funnel still GATES on direction through the live path (a Buy that fell FAILS).
# --------------------------------------------------------------------------------------


def test_buy_that_fell_fails_direction_through_live_path(frame, days):
    res = _grade(
        frame,
        rating=Rating.BUY,
        target_price=140.0,
        call_date=days[0].date(),
        resolution_date=days[285].date(),   # AAA is deep in its decline by here
        asof=days[295].date(),
    )
    assert res.call_score.resolution.actual_price < res.call_score.resolution.call_price
    assert res.call_score.direction_pass is False    # a Buy that ended down does NOT pass


# --------------------------------------------------------------------------------------
# (5) Graceful, user-facing errors (these become st.error in the UI).
# --------------------------------------------------------------------------------------


def test_call_date_in_the_future_is_rejected(frame, days):
    with pytest.raises(LiveGradeError, match="must be in the past"):
        _grade(frame, rating=Rating.BUY, target_price=130.0,
               call_date=days[120].date(), asof=days[120].date())


def test_unknown_ticker_is_a_friendly_error(frame, days):
    with pytest.raises(LiveGradeError, match="No live price data found for ticker"):
        LiveWebPriceProvider(["NOPE"], BENCH, days[0].date(), days[120].date(), fetcher=FrameFetcher(frame))


def test_missing_benchmark_is_a_friendly_error(frame, days):
    with pytest.raises(LiveGradeError, match="benchmark"):
        LiveWebPriceProvider(["AAA"], "^MISSING", days[0].date(), days[120].date(), fetcher=FrameFetcher(frame))
