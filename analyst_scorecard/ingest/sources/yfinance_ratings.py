"""yfinance upgrades/downgrades — firm-attributed rating changes, no scraping, no API key.

``Ticker.upgrades_downgrades`` returns a frame indexed by GradeDate with columns Firm / ToGrade /
FromGrade / Action. We map each row to an ``AnalystCall`` (firm + rating + action + date). Price
targets aren't in this feed, so ``target_price`` / ``previous_target`` / ``analyst`` are left null —
never guessed. The fetcher is injectable so the source is fully testable offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..schema import AnalystCall
from .base import SourceAdapter, _now_iso


class RatingsFetcher(ABC):
    @abstractmethod
    def upgrades_downgrades(self, ticker: str):  # -> pandas DataFrame (GradeDate index)
        ...


class YFinanceRatingsFetcher(RatingsFetcher):
    """Live fetcher backed by yfinance (lazy import — only needed when actually ingesting)."""

    def upgrades_downgrades(self, ticker: str):
        import yfinance as yf  # lazy
        return yf.Ticker(ticker).upgrades_downgrades


class YFinanceRatingsSource(SourceAdapter):
    name = "yfinance_ratings"

    def __init__(self, tickers: Sequence[str], fetcher: Optional[RatingsFetcher] = None,
                 now: Optional[str] = None):
        self.tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        self.fetcher = fetcher or YFinanceRatingsFetcher()
        self.now = now

    def discover(self) -> list[AnalystCall]:
        out: list[AnalystCall] = []
        stamp = _now_iso(self.now)
        for tk in self.tickers:
            try:
                df = self.fetcher.upgrades_downgrades(tk)
            except Exception:  # one bad ticker must never sink the run
                continue
            if df is None or len(df) == 0:
                continue
            for idx, row in df.iterrows():
                firm = row.get("Firm") if hasattr(row, "get") else row["Firm"]
                to_grade = row.get("ToGrade") if hasattr(row, "get") else row["ToGrade"]
                action = row.get("Action") if hasattr(row, "get") else row["Action"]
                graded = row.get("GradeDate", idx) if hasattr(row, "get") else idx
                if not firm:
                    continue
                out.append(AnalystCall(
                    ticker=tk, firm=str(firm), rating=(str(to_grade) if to_grade else None),
                    action=(str(action) if action else None), published_at=graded,
                    source_url=f"https://finance.yahoo.com/quote/{tk}", extracted_at=stamp,
                ))
        return out
