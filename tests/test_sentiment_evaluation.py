"""Offline audit of the news-sentiment scorers against the labelled gold set (lexicon column)."""

from analyst_scorecard.forecast.explanation import LexiconSentimentScorer
from evaluation.sentiment_eval import (
    LABELS,
    accuracy,
    compare,
    confusion_matrix,
    evaluate,
    per_category,
    sign,
    sign_error_rate,
)
from evaluation.sentiment_gold import GOLD


def test_gold_set_is_well_formed():
    assert len(GOLD) >= 40
    assert all(ex.label in ("bull", "bear", "neutral") for ex in GOLD)
    # a hard set, not a trivial one: several distinct difficulty categories present
    assert {"negation", "relief", "mixed", "subtle"}.issubset({ex.category for ex in GOLD})


def test_sign_has_a_neutral_deadzone():
    assert sign(0.9) == "bull"
    assert sign(-0.9) == "bear"
    assert sign(0.0) == "neutral"
    assert sign(0.1) == "neutral"        # inside the ±0.15 band


def test_evaluate_scores_every_example():
    rows = evaluate(LexiconSentimentScorer())
    assert len(rows) == len(GOLD)
    assert all(r.predicted in LABELS for r in rows)


def test_lexicon_baseline_is_locked_and_never_flips_sign():
    rows = evaluate(LexiconSentimentScorer())
    # Baseline LOCKED at ~74% on this specific gold set (37/50). A tight band rather than ">="
    # so an accidental change to the lexicon or the gold set — up OR down — trips the test.
    assert abs(accuracy(rows) - 0.74) <= 0.02
    # ...and critically, it never actively reverses a sign (bull<->bear) — its errors are all
    # "fell back to neutral", which is the safe failure mode for a trader-facing read.
    assert sign_error_rate(rows) == 0.0


def test_confusion_matrix_shape_and_total():
    rows = evaluate(LexiconSentimentScorer())
    cm = confusion_matrix(rows)
    assert list(cm.index) == LABELS and list(cm.columns) == LABELS
    assert int(cm.values.sum()) == len(GOLD)


def test_per_category_covers_all_categories():
    rows = evaluate(LexiconSentimentScorer())
    by_cat = per_category(rows)
    assert set(by_cat.index) == {ex.category for ex in GOLD}
    assert (by_cat["accuracy"] <= 1.0).all() and (by_cat["accuracy"] >= 0.0).all()


def test_compare_returns_lexicon_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = compare()
    assert "lexicon" in out and "llm" not in out      # offline → no LLM column, no network
    assert 0.0 <= out["lexicon"]["accuracy"] <= 1.0
