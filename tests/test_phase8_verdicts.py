"""Phase 8 — plain-English verdicts (offline templated fallback + LLM guardrails)."""

import os

import pytest

from analyst_scorecard.aggregation import aggregate_all
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.providers.call_provider import FixtureCallProvider
from analyst_scorecard.providers.price_provider import SyntheticPriceDataProvider
from analyst_scorecard.verdicts import (
    LLMVerdictGenerator,
    TemplatedVerdictGenerator,
    default_verdict_generator,
)


@pytest.fixture(scope="module")
def scores_by_id():
    provider = SyntheticPriceDataProvider(DEFAULT_CONFIG)
    calls = FixtureCallProvider().get_calls()
    return {s.analyst_id: s for s in aggregate_all(calls, provider, DEFAULT_CONFIG)}


def test_templated_verdict_is_readable_for_every_analyst(scores_by_id):
    gen = TemplatedVerdictGenerator()
    for score in scores_by_id.values():
        v = gen.verdict(score)
        assert isinstance(v, str) and len(v) > 10
        assert v.endswith(".")


def test_verdict_reflects_the_headline_not_just_direction(scores_by_id):
    gen = TemplatedVerdictGenerator()
    # Rider: high direction but lost to the index -> verdict must steer to the index, not praise.
    rider = gen.verdict(scores_by_id["momentum"]).lower()
    assert "direction" in rider
    assert "index" in rider
    assert "better just holding the index" in rider

    # Skilled picker: beat the index -> verdict should say so.
    skilled = gen.verdict(scores_by_id["vega"]).lower()
    assert "beat the index" in skilled

    # Contrarian: good direction but poor accuracy -> verdict flags wild targets.
    contrarian = gen.verdict(scores_by_id["ursa"]).lower()
    assert "wildly off" in contrarian


def test_direction_phrase_tracks_hit_rate(scores_by_id):
    gen = TemplatedVerdictGenerator()
    good = gen.verdict(scores_by_id["vega"]).lower()       # ~85% dir
    bad = gen.verdict(scores_by_id["hubris"]).lower()      # ~10% dir
    assert "right on direction" in good
    assert "usually wrong on direction" in bad


def test_default_generator_is_templated_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(default_verdict_generator(), TemplatedVerdictGenerator)


def test_llm_verdict_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMVerdictGenerator()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; offline verdicts validated via TemplatedVerdictGenerator",
)
def test_llm_verdict_when_key_present(scores_by_id):
    gen = LLMVerdictGenerator()
    v = gen.verdict(scores_by_id["momentum"])
    assert isinstance(v, str) and len(v) > 10
