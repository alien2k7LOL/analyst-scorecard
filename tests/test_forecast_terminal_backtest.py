"""Terminal-value resolution: correct, path-independent, and look-ahead-safe end to end.

A terminal prediction is graded on the price ON the deadline — not whether the path ever touched the
target. These tests pin that distinction down, prove the deadline close is all that matters (prices
after it can be tampered to nonsense without changing the outcome), and run a terminal-only backtest
to confirm the harder target stays calibrated.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.forecast.backtest import (
    ForecastGenConfig,
    resolve_outcome,
    run_forecast_backtest,
)
from analyst_scorecard.forecast.prediction import Prediction, PredictionKind
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider
from analyst_scorecard.schemas import Direction

BENCH = "BENCH"
SAMPLE = "data/sample_historical"


def _prov(values):
    days = pd.bdate_range("2020-01-01", periods=len(values))
    frame = pd.DataFrame({"AAA": list(map(float, values)), BENCH: [1000.0] * len(days)}, index=days)
    return HistoricalPriceFileProvider.from_frame(frame, BENCH), frame, days


def _terminal(days, target, deadline_i, band=0.03, direction=Direction.UP):
    return Prediction(
        prediction_id="t", ticker="AAA", as_of=days[0].date(), target_price=float(target),
        deadline=days[deadline_i].date(), direction=direction,
        kind=PredictionKind.TERMINAL, band_pct=band,
    )


# ---- terminal resolution is about the DEADLINE CLOSE, not the path ----------------------


def test_terminal_band_resolves_on_the_deadline_close():
    vals = list(np.linspace(100, 120, 101)) + [50.0] * 60   # close on day 100 is exactly 120
    prov, _, days = _prov(vals)
    out = resolve_outcome(prov, _terminal(days, target=120, deadline_i=100, band=0.03))
    assert out.hit is True
    assert out.terminal_price == pytest.approx(120.0)
    assert out.terminal_date == days[100].date()
    # An end far from the target misses, even though the path obviously passed through lower prices.
    assert resolve_outcome(prov, _terminal(days, target=135, deadline_i=100, band=0.03)).hit is False


def test_terminal_misses_what_touch_would_hit():
    # Path reaches 120 early, then drifts up to ~132 and ENDS there.
    vals = list(np.linspace(100, 125, 60)) + list(np.linspace(125, 132, 42))[1:]  # day 100 close ~132
    prov, _, days = _prov(vals)
    term = resolve_outcome(prov, _terminal(days, target=120, deadline_i=100, band=0.02))
    touch = resolve_outcome(prov, Prediction(
        prediction_id="t", ticker="AAA", as_of=days[0].date(), target_price=120.0,
        deadline=days[100].date(), direction=Direction.UP))  # default kind = TOUCH
    assert touch.hit is True and term.hit is False


def test_terminal_at_or_through_uses_no_band():
    vals = list(np.linspace(100, 122, 101)) + [100.0] * 40   # day 100 close = 122
    prov, _, days = _prov(vals)
    thru_hit = Prediction(prediction_id="t", ticker="AAA", as_of=days[0].date(), target_price=120.0,
                          deadline=days[100].date(), direction=Direction.UP, kind=PredictionKind.TERMINAL)
    thru_miss = thru_hit.model_copy(update={"target_price": 125.0})
    assert resolve_outcome(prov, thru_hit).hit is True       # 122 >= 120
    assert resolve_outcome(prov, thru_miss).hit is False      # 122 <  125


def test_terminal_resolution_ignores_everything_after_the_deadline():
    vals = list(np.linspace(100, 120, 101)) + [50.0] * 60
    prov, frame, days = _prov(vals)
    pred = _terminal(days, target=120, deadline_i=100, band=0.03)
    base = resolve_outcome(prov, pred)

    tampered = frame.copy()
    tampered.loc[tampered.index > days[100]] *= 1000.0       # absurd post-deadline prices
    cut = HistoricalPriceFileProvider.from_frame(frame.loc[frame.index <= days[100]], BENCH)
    assert resolve_outcome(HistoricalPriceFileProvider.from_frame(tampered, BENCH), pred) == base
    assert resolve_outcome(cut, pred) == base


# ---- end to end on the sample: a harder target that stays honest ------------------------

_GEN = ForecastGenConfig(stride_days=21, horizons=(63, 126), up_offsets=(0.06, 0.12),
                         down_offsets=(0.06, 0.12))


@pytest.fixture(scope="module")
def terminal_result():
    gen = replace(_GEN, kinds=(PredictionKind.TERMINAL,), terminal_bands=(0.05,))
    return run_forecast_backtest(SAMPLE, with_news=True, gen=gen)


def test_terminal_backtest_trains_and_calibrates(terminal_result):
    r = terminal_result
    assert r.n_predictions > 500
    assert r.n_train > 100 and r.n_test > 100
    assert 0.0 < r.test_base_rate < 1.0
    assert r.selected_name in r.metrics                       # a real feature set was chosen
    # The deployed (selected) model is at least as calibrated as raw climatology on the test span.
    assert r.selected_metrics["brier"] <= r.metrics["base_rate"]["brier"] * 1.02


def test_terminal_reliability_tracks_reality(terminal_result):
    big = [b for b in terminal_result.reliability if b.n >= 40]
    assert len(big) >= 3
    for b in big:
        assert abs(b.mean_pred - b.mean_actual) < 0.15, (b.lo, b.mean_pred, b.mean_actual)


def test_terminal_is_harder_than_touch_end_to_end():
    # Landing WITHIN a band on the deadline is rarer than ever touching the level — base rates show it.
    touch = run_forecast_backtest(SAMPLE, with_news=False, gen=replace(_GEN, kinds=(PredictionKind.TOUCH,)))
    term = run_forecast_backtest(SAMPLE, with_news=False,
                                 gen=replace(_GEN, kinds=(PredictionKind.TERMINAL,), terminal_bands=(0.04,)))
    assert term.test_base_rate < touch.test_base_rate
