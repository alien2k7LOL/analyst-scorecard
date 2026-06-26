"""Forecast calibration backtest — recalibration refines accuracy, news earns its keep, reproducibly.

Runs the real backtest over the sample data once (module-scoped) and asserts the properties that
make the probability trustworthy: recalibration improves calibration, the point-in-time news adds
held-out value, the reliability curve tracks reality, and the whole thing is reproducible.
"""

import pandas as pd
import pytest

from analyst_scorecard.forecast.backtest import (
    ForecastGenConfig,
    run_forecast_backtest,
)
from analyst_scorecard.forecast.news import NewsFileProvider

GEN = ForecastGenConfig(stride_days=21, horizons=(63, 126), up_offsets=(0.08, 0.15), down_offsets=(0.08, 0.15))


@pytest.fixture(scope="module")
def result():
    return run_forecast_backtest("data/sample_historical", with_news=True, gen=GEN)


def test_backtest_produces_a_train_test_split(result):
    assert result.n_predictions > 1000
    assert result.n_train > 100 and result.n_test > 100
    assert 0.0 < result.test_base_rate < 1.0


def test_recalibration_improves_on_the_raw_model(result):
    raw, recal = result.metrics["raw"], result.metrics["recalibrated"]
    assert recal["brier"] < raw["brier"]          # better Brier
    assert recal["log_loss"] < raw["log_loss"]    # better log-loss
    assert recal["ece"] < raw["ece"]              # and much better calibrated


def test_point_in_time_news_adds_held_out_value(result):
    # On this sample the news carries a (planted, noisy) signal — the backtest must DISCOVER it.
    assert result.news_helps is True
    assert result.metrics["+news"]["log_loss"] < result.metrics["recalibrated"]["log_loss"]
    assert result.metrics["full"]["log_loss"] < result.metrics["+momentum"]["log_loss"]


def test_full_model_beats_climatology(result):
    assert result.metrics["full"]["brier"] < result.metrics["base_rate"]["brier"]


def test_reliability_curve_tracks_reality(result):
    # In well-populated bins, predicted probability should be close to the actual touch frequency.
    big = [b for b in result.reliability if b.n >= 50]
    assert len(big) >= 4
    for b in big:
        assert abs(b.mean_pred - b.mean_actual) < 0.12, (b.lo, b.mean_pred, b.mean_actual)


def test_backtest_is_reproducible():
    r1 = run_forecast_backtest("data/sample_historical", with_news=True, gen=GEN)
    r2 = run_forecast_backtest("data/sample_historical", with_news=True, gen=GEN)
    assert r1.metrics == r2.metrics
    assert (r1.n_predictions, r1.n_train, r1.n_test) == (r2.n_predictions, r2.n_train, r2.n_test)


def test_deployed_model_predicts_in_range(result):
    from analyst_scorecard.forecast.backtest import generate_predictions, resolve_outcome
    from analyst_scorecard.forecast.features import build_features
    from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider

    price = HistoricalPriceFileProvider("data/sample_historical")
    news = NewsFileProvider("data/sample_historical")
    a_pred = generate_predictions(price, GEN)[0]
    row = build_features(price, a_pred, news)
    p = result.predict_one(row)
    assert 0.0 <= p <= 1.0


# ---- news provider unit behavior ----------------------------------------------------


def test_news_provider_filters_window_to_as_of():
    frame = pd.DataFrame({
        "date": ["2020-01-10", "2020-01-20", "2020-02-15"],
        "symbol": ["AAA", "AAA", "AAA"],
        "sentiment": [0.5, 0.4, 0.9],
    })
    w = NewsFileProvider.from_frame(frame).window("AAA", "2020-01-31", lookback_days=120)
    assert len(w.events) == 2                                  # the 02-15 article is excluded
    assert (w.events["date"] <= pd.Timestamp("2020-01-31")).all()
    assert w.features()["news_sentiment_30"] == pytest.approx(0.45)  # mean(0.5, 0.4)
