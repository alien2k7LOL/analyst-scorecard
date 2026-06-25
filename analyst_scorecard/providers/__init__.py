"""Pluggable providers for prices and analyst calls.

Each external dependency is behind an interface with a deterministic offline default:
    - PriceDataProvider  -> SyntheticPriceDataProvider (seeded GBM)
    - AnalystCallProvider -> FixtureCallProvider (JSON fixtures)
A real price provider / LLM extractor can be dropped in without touching the engine.
"""
