"""Pytest wrapper for the evaluation suite — asserts the audit builds and behaves, fast.

Keeps the slow forecast backtest out of pytest (the forecast eval runs as a standalone script); here
we exercise the extraction matrices/metrics/figures and the forecast confusion/metric helpers on
hand-made arrays. The grading engine is never touched.
"""

import numpy as np

from evaluation.extraction_eval import (
    DIRECTION_CLASSES,
    FIELDS,
    RATING_CLASSES,
    classification_report,
    confusion_matrix,
    error_report,
    failure_breakdown,
    field_accuracy,
    generate_figures,
    run_extractions,
)
from evaluation.forecast_eval import binary_metrics, hit_miss_confusion


def test_rating_confusion_matrix_is_square_and_diagonal_dominant():
    cm = confusion_matrix(run_extractions(), "rating")
    assert list(cm.index) == RATING_CLASSES and list(cm.columns) == RATING_CLASSES
    assert np.trace(cm.values) >= 0.9 * cm.values.sum()   # parser is right ≥90% of the time


def test_direction_confusion_matrix():
    cm = confusion_matrix(run_extractions(), "direction")
    assert list(cm.index) == DIRECTION_CLASSES
    assert np.trace(cm.values) >= 0.9 * cm.values.sum()


def test_classification_report_has_accuracy_macro_weighted():
    rep = classification_report(confusion_matrix(run_extractions(), "rating"))
    for row in ("accuracy", "macro avg", "weighted avg"):
        assert row in rep.index
    assert rep.loc["accuracy", "f1"] >= 0.9
    assert 0.0 <= rep.loc["macro avg", "precision"] <= 1.0


def test_field_accuracy_reports_all_six_fields():
    fa = field_accuracy(run_extractions())
    assert set(fa.index) == set(FIELDS)
    assert fa.loc["ticker", "accuracy"] >= 0.95
    assert fa.loc["firm", "accuracy"] >= 0.95


def test_failure_breakdown_percentages_sum_to_100():
    fb = failure_breakdown(run_extractions())
    assert not fb.empty                      # the adversarial cases guarantee some residual errors
    assert abs(fb["pct_of_errors"].sum() - 100.0) < 0.5
    assert set(fb.columns) == {"failure_type", "count", "pct_of_errors"}


def test_error_report_lists_only_real_failures():
    er = error_report(run_extractions())
    assert not er.empty
    assert (er["n_wrong"] >= 1).all()
    assert {"text", "expected", "predicted", "wrong_fields", "source"} <= set(er.columns)


def test_figures_are_written(tmp_path):
    paths = generate_figures(run_extractions(), figdir=tmp_path)
    assert "rating_confusion" in paths and "field_accuracy" in paths
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0


def test_forecast_hit_miss_confusion_and_metrics():
    y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    p = np.array([0.9, 0.4, 0.1, 0.6, 0.8, 0.2])
    cm = hit_miss_confusion(y, p, 0.5)
    assert cm.loc["actual: HIT", "pred: HIT"] == 2     # 0.9 and 0.8
    assert cm.loc["actual: MISS", "pred: HIT"] == 1    # 0.6 false positive
    m = binary_metrics(cm)
    assert m["n"] == 6 and 0.0 <= m["f1"] <= 1.0
    assert m["precision"] == 2 / 3                     # 2 TP / (2 TP + 1 FP)


def test_forecast_high_threshold_majority_class_trap():
    # 80% misses; a 0.5 threshold that predicts all-miss is 80% 'accurate' but useless (F1=0).
    y = np.array([0] * 8 + [1, 1], dtype=float)
    p = np.full(10, 0.2)                                # model never crosses 0.5
    m = binary_metrics(hit_miss_confusion(y, p, 0.5))
    assert m["accuracy"] == 0.8 and m["f1"] == 0.0
