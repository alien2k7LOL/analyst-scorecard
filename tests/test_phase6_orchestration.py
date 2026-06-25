"""Phase 6 — end-to-end orchestration + the time-loop agent."""

import io

import pytest

from analyst_scorecard.aggregation import build_leaderboard
from analyst_scorecard.cli import main, run_simulation
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.orchestrator import TimeLoopOrchestrator
from analyst_scorecard.providers.call_provider import FixtureCallProvider
from analyst_scorecard.providers.price_provider import SyntheticPriceDataProvider


@pytest.fixture(scope="module")
def calls():
    return FixtureCallProvider().get_calls()


@pytest.fixture(scope="module")
def provider():
    return SyntheticPriceDataProvider(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def result(calls, provider):
    return TimeLoopOrchestrator(calls, provider, DEFAULT_CONFIG).run()


# --------------------------------------------------------------------------------------
# The loop runs untouched over the whole dataset
# --------------------------------------------------------------------------------------


def test_loop_resolves_every_call_exactly_once(result, calls):
    assert len(result.events) == len(calls)
    assert {e.call.call_id for e in result.events} == {c.call_id for c in calls}


def test_synthetic_time_advances_monotonically(result):
    clocks = [e.clock for e in result.events]
    assert clocks == sorted(clocks), "the clock must move forward, never backward"


def test_each_call_resolved_exactly_at_its_deadline(result):
    """The synthetic 'now' at resolution equals the record-time deadline — a call is never
    graded before time reaches its deadline (no look-ahead in the simulation)."""
    for e in result.events:
        assert e.clock == e.call.resolution_date


# --------------------------------------------------------------------------------------
# The time-loop produces the SAME correct result as the batch path
# --------------------------------------------------------------------------------------


def test_timeloop_leaderboard_matches_batch(result, calls, provider):
    batch = build_leaderboard(calls, provider, DEFAULT_CONFIG)
    assert result.leaderboard == batch


def test_running_snapshot_converges_to_final(result):
    """The last snapshot the loop emitted for each analyst equals their final score."""
    last_snapshot = {}
    for e in result.events:
        last_snapshot[e.call.analyst_id] = e.analyst_snapshot
    for aid, final in result.final_scores.items():
        assert last_snapshot[aid].beat_market == final.beat_market
        assert last_snapshot[aid].direction_hit_rate == final.direction_hit_rate


def test_verdict_line_has_expected_shape(result):
    sample = result.events[0].verdict_line
    assert "came due" in sample
    assert ("HIT" in sample) or ("MISSED" in sample)
    assert "Beat-market record now" in sample


def test_loop_is_reproducible(calls):
    p1 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    p2 = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    r1 = TimeLoopOrchestrator(calls, p1, DEFAULT_CONFIG).run()
    r2 = TimeLoopOrchestrator(calls, p2, DEFAULT_CONFIG).run()
    assert [e.verdict_line for e in r1.events] == [e.verdict_line for e in r2.events]
    assert r1.leaderboard == r2.leaderboard


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_run_simulation_prints_leaderboard():
    buf = io.StringIO()
    result = run_simulation(DEFAULT_CONFIG, quiet=False, max_events=5, stream=buf)
    out = buf.getvalue()
    assert "LEADERBOARD" in out
    assert "Beat-the-Market" in out
    # the genuine skilled picker should appear at the top of the printed board
    assert "Vega Capital" in out
    assert len(result.events) == 108


def test_cli_main_smoke(capsys):
    rc = main(["--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
