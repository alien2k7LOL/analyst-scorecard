"""Back-test Phase C — the shipped local sample loads cleanly and covers the edge cases."""

import importlib.util
from pathlib import Path

import pytest

from analyst_scorecard.providers.historical_call_provider import HistoricalCallFileProvider
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_historical"


@pytest.fixture(scope="module")
def prices():
    return HistoricalPriceFileProvider(SAMPLE_DIR)


@pytest.fixture(scope="module")
def call_provider(prices):
    cp = HistoricalCallFileProvider(SAMPLE_DIR, prices)
    cp.get_calls()  # populate ingest_issues
    return cp


def test_sample_files_exist():
    for name in ("prices.csv", "calls.csv", "manifest.json", "README.md"):
        assert (SAMPLE_DIR / name).exists(), name


def test_prices_load_with_ragged_delisted_coverage(prices):
    assert prices.benchmark_symbol == "SPX"
    assert len(prices.tickers()) == 17
    assert "HALT" in prices.tickers()
    spx = prices.price_series("SPX")
    halt = prices.price_series("HALT")
    # HALT delisted -> fewer observations and an earlier last date than the benchmark
    assert len(halt) < len(spx)
    assert halt.index.max() < spx.index.max()
    # multi-year span
    assert (spx.index.max() - spx.index.min()).days > 5 * 365
    assert prices.manifest.get("is_sample") is True


def test_calls_load_and_open_calls_are_dropped(call_provider):
    calls = call_provider.get_calls()
    assert len(calls) >= 50
    # the two recent (still-open) calls drop at ingest with the documented reason
    reasons = [i["reason"] for i in call_provider.ingest_issues]
    assert reasons.count("HORIZON_BEYOND_DATA") == 2


def test_delisting_call_present_but_unresolvable(prices, call_provider):
    calls = call_provider.get_calls()
    halt_calls = [c for c in calls if c.ticker == "HALT"]
    assert len(halt_calls) == 1, "the delisting edge case should ingest as a valid call"
    halt = halt_calls[0]
    # it has a valid entry price (ingestable) but no price at its resolution date (unresolvable)
    assert prices.has_data("HALT", halt.call_date) is True
    assert prices.has_data("HALT", halt.resolution_date) is False


def test_revision_pair_present(call_provider):
    calls = call_provider.get_calls()
    orca = sorted([c for c in calls if c.ticker == "ORCA" and c.analyst_id == "brandt"],
                  key=lambda c: c.call_date)
    assert len(orca) == 2, "a revised target = two dated rows on the same name"
    assert orca[0].call_date < orca[1].call_date
    # the revision falls within the original call's horizon -> a genuine mid-horizon revision
    assert orca[1].call_date < orca[0].resolution_date


def test_generator_is_reproducible():
    """The sample generator produces identical data on every run (seeded, hashlib-based)."""
    spec = importlib.util.spec_from_file_location("_gen_sample", SAMPLE_DIR / "_generate_sample.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    f1, f2 = gen.build_prices(), gen.build_prices()
    assert f1.equals(f2)
    assert gen.build_calls(f1) == gen.build_calls(f2)
