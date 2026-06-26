"""Source adapters — each discovers analyst calls from one channel and yields ``AnalystCall``s."""

from .base import SourceAdapter
from .rss import RssSource
from .yfinance_ratings import YFinanceRatingsSource

__all__ = ["SourceAdapter", "RssSource", "YFinanceRatingsSource"]
