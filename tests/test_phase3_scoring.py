"""Phase 3 — scoring funnel mechanics (unit level)."""

from datetime import date

import pytest

from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.scoring import bucket_direction, compute_accuracy, score_call
from analyst_scorecard.aggregation import aggregate_analyst
from analyst_scorecard.schemas import Call, Direction, Rating, Resolution


# --------------------------------------------------------------------------------------
# Test builders
# --------------------------------------------------------------------------------------


def make_call(rating: Rating, target_price: float = 110.0) -> Call:
    return Call(
        call_id="c1", analyst_id="a1", analyst_name="A", firm="F", ticker="ASTR",
        rating=rating, target_price=target_price, call_date=date(2021, 2, 1),
        horizon_days=252, resolution_date=date(2022, 2, 1), initial_price=100.0,
    )


def make_res(stock_return: float, bench_return: float = 0.05, *, call_price: float = 100.0,
             target_price: float = 110.0, sigma_h: float = 0.25) -> Resolution:
    actual = call_price * (1 + stock_return)
    bench_call = 100.0
    return Resolution(
        call_id="c1", call_date=date(2021, 2, 1), resolution_date=date(2022, 2, 1),
        call_price=call_price, target_price=target_price, actual_price=actual,
        benchmark_call_price=bench_call, benchmark_actual_price=bench_call * (1 + bench_return),
        stock_return=stock_return, benchmark_return=bench_return,
        realized_horizon_vol=sigma_h, n_observations=253,
    )


# --------------------------------------------------------------------------------------
# Stage 1 — direction gate
# --------------------------------------------------------------------------------------


def test_bucket_direction_uses_the_band():
    band = DEFAULT_CONFIG.direction_flat_band
    assert bucket_direction(band + 1e-6) == Direction.UP
    assert bucket_direction(-band - 1e-6) == Direction.DOWN
    assert bucket_direction(0.0) == Direction.FLAT
    assert bucket_direction(band) == Direction.FLAT  # boundary is flat (not strictly past)
    assert bucket_direction(-band) == Direction.FLAT


def test_buy_that_rose_passes_buy_that_fell_fails():
    buy = make_call(Rating.BUY)
    assert score_call(buy, make_res(stock_return=+0.20)).direction_pass is True
    assert score_call(buy, make_res(stock_return=-0.20)).direction_pass is False


def test_flipping_outcome_flips_direction_for_buy_and_sell():
    buy = make_call(Rating.BUY)
    sell = make_call(Rating.SELL)
    assert score_call(buy, make_res(+0.15)).direction_pass is True
    assert score_call(buy, make_res(-0.15)).direction_pass is False
    assert score_call(sell, make_res(-0.15)).direction_pass is True
    assert score_call(sell, make_res(+0.15)).direction_pass is False


def test_hold_passes_only_when_flat():
    hold = make_call(Rating.HOLD, target_price=100.0)
    assert score_call(hold, make_res(0.0)).direction_pass is True
    assert score_call(hold, make_res(+0.20)).direction_pass is False


# --------------------------------------------------------------------------------------
# Stage 2 — accuracy (vol-normalized; only for passers)
# --------------------------------------------------------------------------------------


def test_exact_target_hit_scores_max_accuracy():
    # actual == target -> accuracy exactly 1.0 (the maximum possible)
    res = make_res(stock_return=0.10, target_price=110.0)  # actual = 110 == target
    assert res.actual_price == pytest.approx(res.target_price)
    assert compute_accuracy(res) == pytest.approx(1.0)


def test_bigger_miss_scores_lower_accuracy():
    near = make_res(stock_return=0.10, target_price=111.0)   # miss 1
    far = make_res(stock_return=0.10, target_price=140.0)    # miss 30
    assert compute_accuracy(near) > compute_accuracy(far)
    assert compute_accuracy(far) < 1.0


def test_same_miss_higher_vol_scores_higher_accuracy():
    """A tight call on a VOLATILE stock counts more than the same miss on a calm stock."""
    calm = make_res(stock_return=0.10, target_price=120.0, sigma_h=0.10)
    wild = make_res(stock_return=0.10, target_price=120.0, sigma_h=0.50)
    # identical absolute miss; the volatile stock's hit is more impressive -> higher accuracy
    assert compute_accuracy(wild) > compute_accuracy(calm)


def test_accuracy_is_none_when_direction_fails():
    buy = make_call(Rating.BUY)
    cs = score_call(buy, make_res(stock_return=-0.20))  # Buy that fell -> gate fail
    assert cs.direction_pass is False
    assert cs.accuracy is None


# --------------------------------------------------------------------------------------
# Stage 3 — beat-the-market
# --------------------------------------------------------------------------------------


def test_long_beat_is_stock_minus_benchmark():
    buy = make_call(Rating.BUY)
    cs = score_call(buy, make_res(stock_return=0.20, bench_return=0.05))
    assert cs.position == +1
    assert cs.call_return == pytest.approx(0.20)
    assert cs.beat == pytest.approx(0.20 - 0.05)


def test_short_beat_uses_negative_position():
    sell = make_call(Rating.SELL)
    cs = score_call(sell, make_res(stock_return=-0.20, bench_return=0.05))
    assert cs.position == -1
    assert cs.call_return == pytest.approx(0.20)   # short of a -20% move = +20%
    assert cs.beat == pytest.approx(0.20 - 0.05)


def test_wrong_direction_buy_still_loses_money_in_beat():
    """A Buy that fell FAILS direction but its loss STILL counts in beat-the-market."""
    buy = make_call(Rating.BUY)
    cs = score_call(buy, make_res(stock_return=-0.10, bench_return=0.05))
    assert cs.direction_pass is False
    assert cs.beat == pytest.approx(-0.10 - 0.05)   # money metric includes the loser

def test_hold_is_excluded_from_beat_book():
    hold = make_call(Rating.HOLD, target_price=100.0)
    cs = score_call(hold, make_res(stock_return=0.0))
    assert cs.position == 0
    assert cs.call_return is None
    assert cs.beat is None


def test_raising_benchmark_lowers_beat_by_same_delta():
    buy = make_call(Rating.BUY)
    base = score_call(buy, make_res(stock_return=0.20, bench_return=0.05)).beat
    higher = score_call(buy, make_res(stock_return=0.20, bench_return=0.05 + 0.07)).beat
    assert higher == pytest.approx(base - 0.07)


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def test_aggregate_analyst_basic_stats():
    buy = make_call(Rating.BUY)
    sell = make_call(Rating.SELL)
    hold = make_call(Rating.HOLD, target_price=100.0)
    scores = [
        score_call(buy, make_res(0.20, 0.05)),   # pass, directional
        score_call(buy, make_res(-0.10, 0.05)),  # fail, directional (loser)
        score_call(hold, make_res(0.0, 0.05)),   # pass, NOT directional
    ]
    agg = aggregate_analyst(scores, analyst_name="A", firm="F")
    assert agg.n_calls == 3
    assert agg.n_directional == 2          # hold excluded
    assert agg.n_direction_pass == 2       # the winning buy + the flat hold
    assert agg.direction_hit_rate == pytest.approx(2 / 3)
    # beat over the 2 directional calls only
    assert agg.beat_market == pytest.approx(((0.20 - 0.05) + (-0.10 - 0.05)) / 2)
    # accuracy only over the 2 PASSING calls (buy winner + hold)
    assert agg.mean_accuracy is not None


def test_aggregate_with_no_directional_calls_has_none_beat():
    hold = make_call(Rating.HOLD, target_price=100.0)
    scores = [score_call(hold, make_res(0.0))]
    agg = aggregate_analyst(scores, analyst_name="A", firm="F")
    assert agg.n_directional == 0
    assert agg.beat_market is None
