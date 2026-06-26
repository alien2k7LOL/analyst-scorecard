"""Report assembly — reuses the live grader + forecast engine, fully offline via injected fetchers."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.intel.extract import ExtractedRecommendation
from analyst_scorecard.intel.report import build_report
from analyst_scorecard.providers.live_web_price_provider import LiveGradeError, PriceFetcher
from analyst_scorecard.schemas import Direction

BENCH = "^GSPC"
ASOF = date(2025, 6, 2)


class FrameFetcher(PriceFetcher):
    def __init__(self, frame):
        self.frame = frame

    def fetch(self, symbols, start, end, interval="1d"):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        cols = [c for c in symbols if c in self.frame.columns]
        return self.frame.loc[(self.frame.index >= s) & (self.frame.index <= e), cols].copy()


@pytest.fixture(scope="module")
def fetcher():
    days = pd.bdate_range(end=pd.Timestamp(ASOF), periods=900)
    rng = np.random.default_rng(3)

    def gbm(mu, sig, s0):
        steps = (mu - 0.5 * sig ** 2) / 252 + sig * np.sqrt(1 / 252) * rng.standard_normal(len(days) - 1)
        return s0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))

    frame = pd.DataFrame({"AAPL": gbm(0.18, 0.28, 150.0), BENCH: gbm(0.08, 0.15, 4000.0)}, index=days)
    return FrameFetcher(frame)


def _rec(**kw):
    base = dict(ticker="AAPL", rating="Buy", target_price=260.0, analyst="Dan Ives", firm="Wedbush",
                publication_date=date(2025, 1, 2))
    base.update(kw)
    return ExtractedRecommendation(**base)


def _fake_history():
    cs = SimpleNamespace(ticker="AAPL", beat=0.06, rating=SimpleNamespace(value="Buy"),
                         implied_direction=Direction.UP,
                         resolution=SimpleNamespace(call_date=date(2024, 1, 2), resolution_date=date(2024, 7, 2),
                                                    stock_return=0.12, benchmark_return=0.05))
    analyst = SimpleNamespace(analyst_name="Dan Ives", direction_hit_rate=0.71, beat_market=0.063,
                              n_directional=18, call_scores=[cs])
    return SimpleNamespace(leaderboard=SimpleNamespace(rows=[analyst]))


def test_report_has_live_scorecard_and_forward(fetcher):
    rep = build_report(_rec(), asof=ASOF, benchmark_symbol=BENCH, price_fetcher=fetcher)
    assert rep.live is not None and rep.live_error is None
    assert rep.live.current_price > 0 and rep.live.days_since > 0
    # distance to target is relative to the live price
    assert rep.live.distance_to_target == pytest.approx((260.0 - rep.live.current_price) / rep.live.current_price)
    assert rep.live.provisional is True            # open-ended grade is a mark-to-market read
    assert rep.forward is not None and 0.0 <= rep.forward.probability <= 1.0
    assert rep.forward.direction == Direction.UP
    assert "Buy" in rep.summary


def test_historical_context_matches_by_analyst_name(fetcher):
    rep = build_report(_rec(), asof=ASOF, benchmark_symbol=BENCH, price_fetcher=fetcher,
                       historical_result=_fake_history())
    assert rep.historical is not None and rep.historical.found is True
    assert rep.historical.win_rate == 0.71 and rep.historical.n_recs == 18
    assert rep.historical.on_this_stock_n == 1
    assert "Dan Ives" in rep.summary and "71%" in rep.summary


def test_unknown_analyst_reports_no_track_record(fetcher):
    rep = build_report(_rec(analyst="Nobody McUnknown"), asof=ASOF, benchmark_symbol=BENCH,
                       price_fetcher=fetcher, historical_result=_fake_history())
    assert rep.historical is not None and rep.historical.found is False


def test_similar_calls_surface_same_ticker(fetcher):
    rep = build_report(_rec(), asof=ASOF, benchmark_symbol=BENCH, price_fetcher=fetcher,
                       historical_result=_fake_history())
    assert any(s.ticker == "AAPL" for s in rep.similar)


def test_hold_rating_skips_forward(fetcher):
    rep = build_report(_rec(rating="Hold"), asof=ASOF, benchmark_symbol=BENCH, price_fetcher=fetcher)
    assert rep.forward is None and "neutral" in (rep.forward_error or "").lower()


def test_not_gradeable_raises():
    with pytest.raises(LiveGradeError, match="ticker"):
        build_report(ExtractedRecommendation(rating="Buy", target_price=100.0), asof=ASOF)


def test_missing_pub_date_is_assumed(fetcher):
    rep = build_report(_rec(publication_date=None), asof=ASOF, benchmark_symbol=BENCH, price_fetcher=fetcher)
    assert rep.call_date_assumed is True and rep.call_date < ASOF
