"""The 'proof' layer: the maths breakdown, the headline sentiment read, and the proof chart."""

from datetime import date

import pytest

from analyst_scorecard.forecast.explanation import (
    HeadlineFetcher,
    build_math,
    classify_support,
    formula_text,
    news_lean,
    recent_headlines,
    score_sentiment,
)
from analyst_scorecard.forecast.explanation import NewsHeadline
from analyst_scorecard.forecast.interval import BarInterval
from analyst_scorecard.forecast.live import ForecastGrade
from analyst_scorecard.forecast.prediction import PredictionKind
from analyst_scorecard.schemas import Direction
from analyst_scorecard.viz import plot_forecast_proof


def _grade(**kw):
    base = dict(
        ticker="AAA", direction=Direction.UP, target_price=110.0, as_of=date(2024, 1, 2),
        deadline=date(2024, 4, 2), n_days=60, s0=100.0, raw_probability=0.30, probability=0.28,
        calibrated=True, kind=PredictionKind.TERMINAL, band_pct=0.03, interval=BarInterval.DAILY,
        deadline_label="2024-04-02", drift_bar=0.0003, vol_bar=0.02,
    )
    base.update(kw)
    return ForecastGrade(**base)


def test_build_math_exposes_the_ingredients():
    factors = build_math(_grade())
    labels = [f.label for f in factors]
    assert any("Start price" in l for l in labels)
    assert any("Drift" in l for l in labels)
    assert any("Volatility" in l for l in labels)
    # _grade() is a ±band call → symmetric → UNSIGNED distance to the band centre
    assert any("Distance to band centre" in l for l in labels)
    dist = next(f for f in factors if "Distance" in f.label)
    assert dist.value.replace("+", "").startswith("0.6")     # ~0.62 σ, no sign for a band call
    assert "%" in next(f for f in factors if "Raw model" in f.label).value
    # a directional TOUCH call keeps the SIGNED distance to target
    tfac = build_math(_grade(kind=PredictionKind.TOUCH, band_pct=None))
    tdist = next(f for f in tfac if "Distance" in f.label)
    assert "Distance to target" in tdist.label and tdist.value.startswith("+0.6")


def test_formula_text_is_kind_aware():
    assert "Terminal (band)" in formula_text(_grade(kind=PredictionKind.TERMINAL, band_pct=0.03))
    assert "Touch" in formula_text(_grade(kind=PredictionKind.TOUCH, band_pct=None))


def test_sentiment_reads_direction():
    assert score_sentiment("Company beats earnings, shares surge to record") > 0
    assert score_sentiment("Stock plunges on downgrade and lawsuit") < 0
    assert score_sentiment("Board to meet on Thursday afternoon") == 0.0


def test_sentiment_handles_inflections_and_avoids_fall_season():
    # regression: 'downgraded' + 'concerns' must read negative; 'fall event' must NOT (season ≠ drop)
    assert score_sentiment("Apple downgraded at Morgan Stanley on demand concerns") < 0
    assert score_sentiment("Apple unveils new iPhone lineup at fall event") == 0.0
    assert score_sentiment("Stock tumbles as guidance disappoints") < 0
    # the end-to-end symptom: a bearish headline must SUPPORT a DOWN call
    assert classify_support(score_sentiment("Apple downgraded on weak demand"), Direction.DOWN) == "supports"


def test_sentiment_is_sign_aware_negation_and_relief():
    # NEGATION: a negator before a sentiment word must flip the sign (the old word-count got these wrong)
    assert score_sentiment("Apple is not a strong buy") < 0
    assert score_sentiment("No major concerns for Apple this quarter") > 0
    assert score_sentiment("Analyst fails to see upside, cuts target") < 0
    # RELIEF IDIOMS: a scary word inside a relief phrase is GOOD news, in either order
    assert score_sentiment("iPhone demand concerns ease as orders rebound") > 0
    assert score_sentiment("Apple eases fears with strong guidance") > 0
    assert score_sentiment("Apple posts narrower loss, shares climb") > 0
    # and none of this breaks the plain cases
    assert score_sentiment("Apple beats earnings, shares surge to record") > 0
    assert score_sentiment("Stock tumbles as guidance disappoints") < 0


class _StubHeadlines(HeadlineFetcher):
    def __init__(self, items):
        self.items = items

    def fetch(self, ticker, limit=6):
        return self.items[:limit]


class _BadHeadlines(HeadlineFetcher):
    def fetch(self, ticker, limit=6):
        raise RuntimeError("network down")


def test_recent_headlines_parses_and_scores():
    items = [
        {"title": "Company beats and surges", "publisher": "Wire", "link": "http://x/1"},
        {"content": {"title": "Shares plunge on downgrade", "provider": {"displayName": "Feed"}}},
    ]
    out = recent_headlines("AAA", fetcher=_StubHeadlines(items))
    assert len(out) == 2
    assert out[0].sentiment > 0 and out[1].sentiment < 0
    assert out[0].publisher == "Wire" and out[1].publisher == "Feed"


def test_recent_headlines_never_raises():
    assert recent_headlines("AAA", fetcher=_BadHeadlines()) == []


def test_news_classify_support_is_direction_relative():
    # bullish headline supports an UP call, contradicts a DOWN call (and vice-versa)
    assert classify_support(0.6, Direction.UP) == "supports"
    assert classify_support(0.6, Direction.DOWN) == "contradicts"
    assert classify_support(-0.6, Direction.UP) == "contradicts"
    assert classify_support(-0.6, Direction.DOWN) == "supports"
    assert classify_support(0.0, Direction.UP) == "neutral"


def test_news_lean_tallies_against_the_call():
    heads = [NewsHeadline("up beat surges", "", None, None, 0.7),
             NewsHeadline("down plunge", "", None, None, -0.7),
             NewsHeadline("meeting thursday", "", None, None, 0.0)]
    lean = news_lean(heads, Direction.UP)
    assert lean["supports"] == 1 and lean["contradicts"] == 1 and lean["neutral"] == 1
    assert lean["labels"] == ["supports", "contradicts", "neutral"]


def test_proof_chart_renders_for_both_kinds():
    for kind, band in [(PredictionKind.TERMINAL, 0.03), (PredictionKind.TOUCH, None)]:
        fig = plot_forecast_proof(_grade(kind=kind, band_pct=band), dark=True)
        assert fig is not None
        assert len(fig.axes) == 2


def test_proof_chart_headline_number_is_calibrated_not_raw():
    # the BOLD number printed on the figure must be the CALIBRATED probability (what the hero, headline,
    # copy-string and alt text all show) — never the raw one, or the picture contradicts the verdict.
    for kind in (PredictionKind.TOUCH, PredictionKind.TERMINAL):
        g = _grade(kind=kind, band_pct=None, raw_probability=0.61, probability=0.55, calibrated=True)
        fig = plot_forecast_proof(g, dark=True)
        bold = [t.get_text() for ax in fig.axes for t in ax.texts if t.get_fontweight() == "bold"]
        assert any("55%" in t for t in bold), (kind, bold)
        assert not any(t.strip().endswith("≈ 61%") for t in bold), (kind, bold)
