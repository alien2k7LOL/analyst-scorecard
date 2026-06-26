"""Back-test Phase D — the runner produces a historical leaderboard via the UNCHANGED engine."""

from pathlib import Path

import pytest

from analyst_scorecard import backtest_cli
from analyst_scorecard.backtest import SAMPLE_DATA_DIR, load_backtest, run_backtest
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider
from analyst_scorecard.resolution import resolve_call_with_provider
from analyst_scorecard.scoring import score_call


@pytest.fixture(scope="module")
def result():
    return run_backtest(SAMPLE_DATA_DIR, DEFAULT_CONFIG)


def test_runner_produces_complete_leaderboard(result):
    assert len(result.leaderboard.rows) >= 4
    assert result.n_resolved == 60
    assert result.n_skipped == 1
    assert result.n_ingest_dropped == 2
    assert result.span_start.year == 2017


def test_leaderboard_is_ranked_by_beat_market(result):
    beats = [r.beat_market for r in result.leaderboard.rows if r.beat_market is not None]
    assert beats == sorted(beats, reverse=True)


def test_delisting_is_skipped_with_reason(result):
    halt = [s for s in result.skipped if s.call.ticker == "HALT"]
    assert len(halt) == 1
    assert halt[0].reason == "DELISTED_OR_HALTED"
    # and it never entered any analyst's score
    assert all(cs.ticker != "HALT" for cs in result.resolved_scores)


def test_runner_uses_the_unchanged_engine_path(result):
    """Each resolved CallScore equals what the engine produces directly — no custom scoring."""
    prov = HistoricalPriceFileProvider(SAMPLE_DATA_DIR)
    calls = load_backtest(SAMPLE_DATA_DIR, DEFAULT_CONFIG).call_provider.get_calls()
    call_by_id = {c.call_id: c for c in calls}
    for cs in result.resolved_scores[:10]:
        call = call_by_id[cs.call_id]
        direct = score_call(call, resolve_call_with_provider(call, prov), DEFAULT_CONFIG)
        assert cs == direct


def test_perma_bull_and_skilled_appear_on_the_board(result):
    by_id = result.analyst_scores
    assert by_id["calloway"].direction_hit_rate >= 0.7   # rode the bull market
    assert by_id["calloway"].beat_market <= 0.0          # but added no value vs the index
    assert by_id["petrova"].beat_market > 0.0            # genuine skill


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_backtest_cli_main_smoke(capsys):
    rc = backtest_cli.main(["--show-skips"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "HISTORICAL LEADERBOARD" in out
    assert "DELISTED_OR_HALTED" in out
    assert "Reed Calloway" in out  # the perma-bull is on the board


def test_render_report_contains_span_and_counts(result):
    text = backtest_cli.render_report(result, show_skips=False)
    assert "Price span" in text
    assert "resolved & scored" in text
    assert "SAMPLE data" in text
