"""Analyst Scorecard — honest, fair, reproducible grading of analyst price targets.

The package is offline-first: every external dependency (price data, analyst calls,
LLM verdicts) sits behind an interface with a deterministic offline implementation, so
the whole scoring engine builds and runs with no network and no API key.

The scoring spine is a three-stage funnel, not a blended average:
    1. direction  — pass/fail gate
    2. accuracy   — volatility-normalized closeness, only for direction-passers
    3. beat-market — headline: did following the call beat just holding the index?
"""

__version__ = "0.1.0"
