"""Back-test Phase E — PROVE no future data leaks into any historical score.

Historical data makes look-ahead the #1 risk (the "future" is sitting in the file). These tests
would FAIL if any price after a call's resolution date influenced its score.
"""

import numpy as np
import pandas as pd
import pytest

from analyst_scorecard.backtest import HistoricalBacktest, run_backtest, SAMPLE_DATA_DIR
from analyst_scorecard.config import DEFAULT_CONFIG as CFG
from analyst_scorecard.providers.historical_call_provider import HistoricalCallFileProvider
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider
from analyst_scorecard.providers.price_provider import _ts
from analyst_scorecard.resolution import resolve_call_with_provider
from analyst_scorecard.schemas import Call, Rating
from analyst_scorecard.scoring import score_call


def _make(aaa_full, w: int, idx: float = 1000.0):
    """Build a controlled provider + a Buy on AAA with resolution at window index ``w``."""
    days = pd.bdate_range("2019-01-02", periods=len(aaa_full))
    frame = pd.DataFrame({"IDX": [idx] * len(days), "AAA": list(map(float, aaa_full))}, index=days)
    prov = HistoricalPriceFileProvider.from_frame(frame, "IDX")
    call = Call(
        call_id="t", analyst_id="a", analyst_name="A", firm="F", ticker="AAA",
        rating=Rating.BUY, target_price=120.0, call_date=days[0].date(),
        horizon_days=w, resolution_date=days[w].date(), initial_price=float(aaa_full[0]),
    )
    return prov, call, frame, days


def _score(prov, call):
    return score_call(call, resolve_call_with_provider(call, prov), CFG)


# --------------------------------------------------------------------------------------
# (1) The score is identical with, without, or with TAMPERED post-horizon prices
# --------------------------------------------------------------------------------------


def test_post_horizon_prices_are_provably_ignored():
    window = list(np.linspace(100, 130, 31))  # rises over the horizon (a Buy that passes)
    full = window + [50.0] * 20               # then crashes AFTER the horizon
    prov_full, call, frame, days = _make(full, w=30)
    cs_full = _score(prov_full, call)

    # (a) truncate everything after the resolution date
    prov_trunc = HistoricalPriceFileProvider.from_frame(frame.loc[: days[30]], "IDX")
    cs_trunc = _score(prov_trunc, call)

    # (b) multiply every post-horizon price by 1000 (stock AND benchmark)
    tampered = frame.copy()
    tampered.loc[tampered.index > days[30]] *= 1000.0
    prov_tamper = HistoricalPriceFileProvider.from_frame(tampered, "IDX")
    cs_tamper = _score(prov_tamper, call)

    assert cs_full == cs_trunc == cs_tamper      # byte-identical: the future never entered
    assert cs_full.direction_pass is True


# --------------------------------------------------------------------------------------
# (2) A call whose verdict WOULD flip if post-horizon data leaked
# --------------------------------------------------------------------------------------


def test_verdict_matches_no_leak_result_even_when_future_would_flip_it():
    # AAA falls to 90 by the horizon (a Buy that FAILS), then moons to 200 afterwards.
    declining = list(np.linspace(100, 90, 31))
    full = declining + [200.0] * 20
    prov, call, frame, days = _make(full, w=30)
    cs = _score(prov, call)

    # The correct, no-leak verdict: the Buy FAILED — the post-horizon moon is ignored.
    assert cs.direction_pass is False
    assert cs.resolution.actual_price == pytest.approx(90.0)

    # Truncating the future changes nothing.
    prov_trunc = HistoricalPriceFileProvider.from_frame(frame.loc[: days[30]], "IDX")
    assert _score(prov_trunc, call) == cs

    # Proof the stakes are real: move that same 200 INSIDE the horizon and the Buy PASSES — the
    # verdict is driven purely by in-window data, exactly as it should be.
    moon_in_window = declining[:-1] + [200.0] + [200.0] * 20
    prov2, call2, _, _ = _make(moon_in_window, w=30)
    assert _score(prov2, call2).direction_pass is True


# --------------------------------------------------------------------------------------
# (3) Reproducibility — same inputs, identical historical leaderboard
# --------------------------------------------------------------------------------------


def test_backtest_is_reproducible():
    r1 = run_backtest(SAMPLE_DATA_DIR, CFG)
    r2 = run_backtest(SAMPLE_DATA_DIR, CFG)
    assert r1.leaderboard == r2.leaderboard
    assert (r1.n_resolved, r1.n_skipped, r1.n_ingest_dropped) == (r2.n_resolved, r2.n_skipped, r2.n_ingest_dropped)


# --------------------------------------------------------------------------------------
# (4) Same fairness guarantees as the synthetic suite, now on historical-style data
# --------------------------------------------------------------------------------------


def test_perma_bull_high_direction_but_beat_at_or_below_zero():
    by_id = run_backtest(SAMPLE_DATA_DIR, CFG).analyst_scores
    perma = by_id["calloway"]
    skilled = by_id["petrova"]
    assert perma.direction_hit_rate >= 0.70           # rode the bull market
    assert perma.beat_market is not None and perma.beat_market <= 0.0   # added no value vs index
    assert skilled.beat_market > 0.0                  # genuine skill
    # the headline separates them even though both look good on direction
    assert perma.beat_market < skilled.beat_market


# --------------------------------------------------------------------------------------
# (5) End-to-end: removing ALL data after the last resolution date changes no score
# --------------------------------------------------------------------------------------


def test_truncating_the_global_future_does_not_change_the_leaderboard():
    full = run_backtest(SAMPLE_DATA_DIR, CFG)
    prov_full = HistoricalPriceFileProvider(SAMPLE_DATA_DIR)
    max_res = max(cs.resolution.resolution_date for cs in full.resolved_scores)

    truncated_frame = prov_full._frame.loc[prov_full._frame.index <= _ts(max_res)]
    prov_trunc = HistoricalPriceFileProvider.from_frame(truncated_frame, prov_full.benchmark_symbol, manifest=prov_full.manifest)
    calls_trunc = HistoricalCallFileProvider(SAMPLE_DATA_DIR, prov_trunc)
    trunc = HistoricalBacktest(prov_trunc, calls_trunc, CFG).run()

    # Every resolved call resolved on/before max_res, so deleting later data is a no-op on scores.
    assert trunc.leaderboard == full.leaderboard
