"""The source-adapter contract: discover analyst calls from one channel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..schema import AnalystCall


class SourceAdapter(ABC):
    """A discovery source. ``discover`` returns the raw calls it finds (un-deduped, un-validated)."""

    name: str = "source"

    @abstractmethod
    def discover(self) -> list[AnalystCall]: ...


def _now_iso(now: str | None) -> str:
    return now if now is not None else datetime.now().isoformat(timespec="seconds")
