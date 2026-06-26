"""Data-acquisition layer — discover, extract, normalize, and store analyst calls.

This package ONLY collects and structures public analyst recommendations; it never scores, ranks, or
backtests (that's the rest of ``analyst_scorecard``). Output is appended to ``data/analyst_calls.jsonl``
in the strict schema and feeds the downstream scoring engine.

Sources are deliberately ToS-compliant and key-free for the MVP:
  * ``yfinance`` upgrades/downgrades — firm-attributed rating changes (no scraping).
  * public RSS feeds — headline/summary text parsed by the existing intel extractor.

Every source is an injectable adapter, so the whole pipeline is testable offline with fixtures, and
the final output is deterministic + idempotent (re-running on the same input adds nothing new).
"""

from .schema import AnalystCall
from .pipeline import IngestResult, run_ingest

__all__ = ["AnalystCall", "IngestResult", "run_ingest"]
