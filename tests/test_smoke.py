"""Phase 0 smoke test: the package imports and the suite runs."""

import analyst_scorecard


def test_package_imports():
    assert analyst_scorecard.__version__ == "0.1.0"


def test_offline_first_no_api_key_required(monkeypatch):
    """Importing the package must never require an API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib

    importlib.reload(analyst_scorecard)
    assert analyst_scorecard is not None
