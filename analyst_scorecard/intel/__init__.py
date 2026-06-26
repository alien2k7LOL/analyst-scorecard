"""Analyst-intelligence layer — turn a pasted recommendation into a graded research report.

This package is all FRONT-of-engine: it extracts structured fields from messy real-world text (or a
URL) and assembles a report by REUSING the existing, unchanged grading engine (resolution + scoring +
the forecast probability). It never rewrites any scoring logic.

Security note: pasted text and fetched pages are UNTRUSTED DATA. The extractor treats them strictly
as data to parse — it never executes instructions found inside them — and the UI shows the extracted
fields for human confirmation before anything is graded.
"""

from .extract import (
    ExtractedRecommendation,
    HeuristicExtractor,
    extract_from_url,
    extract_recommendation,
    normalize_rating,
)
from .report import AnalystReport, build_report

__all__ = [
    "ExtractedRecommendation",
    "HeuristicExtractor",
    "extract_recommendation",
    "extract_from_url",
    "normalize_rating",
    "AnalystReport",
    "build_report",
]
