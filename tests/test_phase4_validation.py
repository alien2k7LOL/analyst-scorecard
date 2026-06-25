"""Phase 4 — validation: the engine must recover the KNOWN ground truth, fairly.

These are the tests a sophisticated finance user would write to try to catch the scoring
engine being unfair, look-ahead-biased, or inconsistent. They run the real synthetic dataset
through the full funnel and assert the planted skill profiles come back out.

Headline property under test: the beat-the-market metric STRIPS OUT free market gains, so an
analyst who merely rode a rising market (high direction hit-rate) does NOT look good on the
headline, while a genuine stock-picker does.
"""

from collections import defaultdict

import pytest

from analyst_scorecard.aggregation import aggregate_all, aggregate_analyst, build_leaderboard, score_calls
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.providers.call_provider import FixtureCallProvider
from analyst_scorecard.providers.price_provider import SyntheticPriceDataProvider
from analyst_scorecard.scoring import score_call


@pytest.fixture(scope="module")
def calls():
    return FixtureCallProvider().get_calls()


@pytest.fixture(scope="module")
def provider():
    return SyntheticPriceDataProvider(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def scores_by_id(calls, provider):
    return {s.analyst_id: s for s in aggregate_all(calls, provider, DEFAULT_CONFIG)}


# --------------------------------------------------------------------------------------
# The required ground-truth recoveries
# --------------------------------------------------------------------------------------


def test_buy_only_rider_high_direction_but_beat_market_at_or_below_zero(scores_by_id):
    """The headline correctly strips out free market gains."""
    rider = scores_by_id["momentum"]
    assert rider.direction_hit_rate >= 0.70, "rider rode a rising market -> high direction hit-rate"
    assert rider.beat_market is not None
    assert rider.beat_market <= 0.0, "rider added no value vs the index -> beat-market <= 0"


def test_skilled_picker_has_positive_beat_market(scores_by_id):
    skilled = scores_by_id["vega"]
    assert skilled.beat_market is not None
    assert skilled.beat_market > 0.0


def test_contrarian_good_direction_but_poor_accuracy(scores_by_id):
    contrarian = scores_by_id["ursa"]
    skilled = scores_by_id["vega"]
    assert contrarian.direction_hit_rate >= 0.70, "contrarian is right on direction"
    assert contrarian.mean_accuracy is not None
    assert contrarian.mean_accuracy < 0.65, "but bad on magnitude -> poor accuracy"
    # and clearly worse on magnitude than a genuinely accurate analyst
    assert contrarian.mean_accuracy < skilled.mean_accuracy - 0.2


def test_overconfident_wrong_is_bad_on_everything(scores_by_id):
    hubris = scores_by_id["hubris"]
    assert hubris.direction_hit_rate <= 0.40
    assert hubris.beat_market is not None and hubris.beat_market < 0.0


def test_short_seller_can_beat_the_market(scores_by_id):
    """beat-the-market must work for SHORTS too, not just longs."""
    shorts = scores_by_id["shortalpha"]
    assert shorts.beat_market is not None and shorts.beat_market > 0.0


# --------------------------------------------------------------------------------------
# The crucial separation: direction hit-rate must NOT be mistaken for skill
# --------------------------------------------------------------------------------------


def test_headline_separates_rider_from_skilled_despite_similar_direction(scores_by_id):
    rider = scores_by_id["momentum"]
    skilled = scores_by_id["vega"]
    # both look good on direction...
    assert rider.direction_hit_rate >= 0.70
    assert skilled.direction_hit_rate >= 0.70
    # ...but the headline ranks the genuine picker far above the rider
    assert skilled.beat_market > rider.beat_market
    assert rider.beat_market <= 0.0 < skilled.beat_market


def test_skilled_picker_tops_the_leaderboard(calls, provider):
    lb = build_leaderboard(calls, provider, DEFAULT_CONFIG)
    # ranked by beat-the-market; the genuine skilled picker should be at/near the very top
    top_ids = [row.analyst_id for row in lb.rows[:2]]
    assert "vega" in top_ids
    # the rider must rank in the bottom half
    ranked_ids = [row.analyst_id for row in lb.rows]
    assert ranked_ids.index("momentum") >= len(ranked_ids) // 2


# --------------------------------------------------------------------------------------
# Monotonicity / sanity
# --------------------------------------------------------------------------------------


def test_exact_target_hit_is_best_possible_accuracy(calls, provider):
    """No real call can beat a bullseye; verify the max is actually achievable & unbeaten."""
    from tests.test_phase3_scoring import make_call, make_res
    from analyst_scorecard.scoring import compute_accuracy
    from analyst_scorecard.schemas import Rating

    bullseye = make_res(stock_return=0.10, target_price=110.0)  # actual == target
    assert compute_accuracy(bullseye) == pytest.approx(1.0)
    # every scored fixture call's accuracy is <= the bullseye maximum
    for cs in score_calls(calls, provider, DEFAULT_CONFIG):
        if cs.accuracy is not None:
            assert cs.accuracy <= 1.0 + 1e-12


def test_flipping_a_call_outcome_flips_its_direction_result(provider):
    from tests.test_phase3_scoring import make_call, make_res
    from analyst_scorecard.schemas import Rating

    buy = make_call(Rating.BUY)
    up = score_call(buy, make_res(stock_return=+0.25))
    down = score_call(buy, make_res(stock_return=-0.25))
    assert up.direction_pass != down.direction_pass


def test_raising_benchmark_lowers_every_analyst_beat_identically(calls, provider):
    """Add a constant delta to EVERY call's benchmark return; every analyst's beat-market
    must drop by exactly that delta (a uniform, fair shift — no analyst is advantaged)."""
    cfg = DEFAULT_CONFIG
    delta = 0.05
    base = {s.analyst_id: s.beat_market for s in aggregate_all(calls, provider, cfg)}

    call_by_id = {c.call_id: c for c in calls}
    bumped = []
    for cs in score_calls(calls, provider, cfg):
        res = cs.resolution
        new_bench_return = res.benchmark_return + delta
        new_res = res.model_copy(
            update={
                "benchmark_return": new_bench_return,
                "benchmark_actual_price": res.benchmark_call_price * (1 + new_bench_return),
            }
        )
        bumped.append(score_call(call_by_id[cs.call_id], new_res, cfg))

    by_analyst = defaultdict(list)
    for cs in bumped:
        by_analyst[cs.analyst_id].append(cs)

    for aid, css in by_analyst.items():
        agg = aggregate_analyst(css, analyst_name="x", firm="y")
        if base[aid] is None:
            assert agg.beat_market is None
        else:
            assert agg.beat_market == pytest.approx(base[aid] - delta)


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_identical_seed_identical_scores(calls):
    p1 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    p2 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    lb1 = build_leaderboard(calls, p1, DEFAULT_CONFIG)
    lb2 = build_leaderboard(calls, p2, DEFAULT_CONFIG)
    assert lb1 == lb2  # frozen pydantic models compare by value, field-for-field


def test_scores_change_under_a_different_seed(calls):
    p_other = SyntheticPriceDataProvider(DEFAULT_CONFIG.with_overrides(seed=DEFAULT_CONFIG.seed + 7))
    lb_other = {s.analyst_id: s.beat_market for s in aggregate_all(calls, p_other, DEFAULT_CONFIG)}
    lb_base = {s.analyst_id: s.beat_market for s in aggregate_all(calls, SyntheticPriceDataProvider(DEFAULT_CONFIG))}
    # at least one analyst's headline differs under a different price world
    assert any(lb_other[a] != lb_base[a] for a in lb_base)
