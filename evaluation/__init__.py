"""Evaluation suite for the analyst-recommendation pipeline — auditable, separate from production.

Nothing here is imported by the app or the grading engine; it only *measures* the extractor and the
forecaster. Run standalone:

    python -m evaluation.extraction_eval     # rating confusion matrix, metrics, errors, figures
    python -m evaluation.forecast_eval       # thresholded hit/miss matrix + calibration framing

or via pytest: tests/test_extraction_evaluation.py.
"""
