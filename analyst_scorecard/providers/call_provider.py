"""Analyst calls: the ``AnalystCallProvider`` interface and a JSON-fixture default.

``FixtureCallProvider`` is the offline default — it loads pre-baked, validated ``Call``
records from JSON. The real ``LLMCallExtractor`` (Phase 5) implements the same interface,
reading messy research text into the same ``Call`` schema, so the engine never cares which
one produced the calls.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ..schemas import Call

# Repo root = .../analyst_scorecard/providers/call_provider.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = REPO_ROOT / "fixtures" / "calls.json"


class AnalystCallProvider(ABC):
    """Interface: yield validated ``Call`` records from some source."""

    @abstractmethod
    def get_calls(self) -> list[Call]:
        ...


class FixtureCallProvider(AnalystCallProvider):
    """Load analyst calls from a JSON file (a list of Call objects).

    Validation is strict (Pydantic, extra fields forbidden): a malformed call fails loudly
    rather than entering the scoring funnel. Calls are returned sorted by (call_date,
    call_id) for a stable, reproducible order.
    """

    def __init__(self, path: Path | str = DEFAULT_FIXTURE_PATH):
        self.path = Path(path)

    def get_calls(self) -> list[Call]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Call fixture not found at {self.path}. Generate it with "
                f"`python -m analyst_scorecard.synth` (or pass an explicit path)."
            )
        raw = json.loads(self.path.read_text())
        if not isinstance(raw, list):
            raise ValueError(f"Fixture {self.path} must be a JSON list of calls")
        calls = [Call.model_validate(item) for item in raw]
        return _sorted_calls(calls)

    @staticmethod
    def from_calls(calls: Iterable[Call]) -> "InMemoryCallProvider":
        return InMemoryCallProvider(list(calls))


class InMemoryCallProvider(AnalystCallProvider):
    """Trivial provider over an in-memory list — handy for tests and the extractor harness."""

    def __init__(self, calls: list[Call]):
        self._calls = _sorted_calls(calls)

    def get_calls(self) -> list[Call]:
        return list(self._calls)


def _sorted_calls(calls: list[Call]) -> list[Call]:
    return sorted(calls, key=lambda c: (c.call_date, c.call_id))
