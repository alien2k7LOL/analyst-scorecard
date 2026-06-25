"""Phase 5 — analyst-call extraction agent (offline harness + LLM guardrails)."""

import os

import pytest

from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.extraction import (
    ExtractedCall,
    HeuristicCallExtractor,
    LLMCallExtractor,
    evaluate_extractor,
    finalize_extracted,
    load_research_notes,
)
from analyst_scorecard.providers.price_provider import SyntheticPriceDataProvider
from analyst_scorecard.schemas import Call, Rating


@pytest.fixture(scope="module")
def notes():
    return load_research_notes()


@pytest.fixture(scope="module")
def provider():
    return SyntheticPriceDataProvider(DEFAULT_CONFIG)


# --------------------------------------------------------------------------------------
# Fixtures load & validate
# --------------------------------------------------------------------------------------


def test_research_notes_load_and_validate(notes):
    assert len(notes) >= 5
    ratings = {n.expected.rating for n in notes}
    assert ratings == set(Rating)  # all five ratings exercised


# --------------------------------------------------------------------------------------
# Offline heuristic extractor matches ground truth (the Phase 5 done-criterion)
# --------------------------------------------------------------------------------------


def test_offline_extractor_matches_ground_truth(notes):
    report = evaluate_extractor(HeuristicCallExtractor(), notes)
    assert report.exact_match_rate == 1.0, report.summary()
    assert all(acc == 1.0 for acc in report.per_field_accuracy.values()), report.mismatches


def test_extractor_handles_each_rating_and_horizon(notes):
    extractor = HeuristicCallExtractor()
    for note in notes:
        got = extractor.extract(note.text)
        assert got.rating == note.expected.rating
        assert got.horizon_days == note.expected.horizon_days
        assert got.ticker == note.expected.ticker


# --------------------------------------------------------------------------------------
# Finalizing an extracted call into a full, look-ahead-safe Call
# --------------------------------------------------------------------------------------


def test_finalize_extracted_produces_valid_calls(notes, provider):
    extractor = HeuristicCallExtractor()
    for note in notes:
        extracted = extractor.extract(note.text)
        call = finalize_extracted(extracted, provider)
        assert isinstance(call, Call)
        # resolution date fixed at record time = call + horizon trading days, within data range
        assert call.resolution_date > call.call_date
        assert call.initial_price > 0
        # the deadline equals the provider's trading-day offset (no look-ahead shortcut)
        expected_res = provider.trading_day_offset(call.call_date, call.horizon_days).date()
        assert call.resolution_date == expected_res


def test_finalized_calls_flow_through_the_engine(notes, provider):
    """Extracted calls are first-class: they resolve and score like fixture calls."""
    from analyst_scorecard.aggregation import score_calls

    calls = [finalize_extracted(HeuristicCallExtractor().extract(n.text), provider) for n in notes]
    scores = score_calls(calls, provider, DEFAULT_CONFIG)
    assert len(scores) == len(calls)
    for cs in scores:
        assert cs.resolution.actual_price > 0


# --------------------------------------------------------------------------------------
# LLM extractor guardrails (offline) + real run (only if a key is present)
# --------------------------------------------------------------------------------------


def test_llm_extractor_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMCallExtractor()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY set; offline run validated via HeuristicCallExtractor",
)
def test_llm_extractor_matches_ground_truth_when_key_present(notes):
    report = evaluate_extractor(LLMCallExtractor(), notes)
    # Allow one slip vs. the deterministic harness; ratings/tickers/targets should be solid.
    assert report.per_field_accuracy["rating"] == 1.0, report.mismatches
    assert report.per_field_accuracy["ticker"] == 1.0, report.mismatches
    assert report.per_field_accuracy["target_price"] >= 0.8, report.mismatches
    assert report.exact_match_rate >= 0.8, report.summary()
