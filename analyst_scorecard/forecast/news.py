"""Point-in-time news — the look-ahead guard for the NEWS side.

The cardinal risk with news in a backtest is leaking the future: using an article (or a revised
version of it) that wasn't actually visible on the as_of date. ``NewsWindow`` makes that
structurally impossible — it holds only events dated on/before ``as_of`` and RAISES if handed a
later one, exactly like ``PriceWindow``/``LookbackWindow`` do for prices. News features are derived
only from inside the window, so a probability can never see news from its own future.

`news.csv` schema (long format):  ``date,symbol,sentiment[,headline]``  with sentiment in [-1, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..providers.price_provider import DateLike, _ts

NEWS_FEATURE_NAMES = ["news_sentiment_30", "news_volume_30", "news_decay"]
_DECAY_HALFLIFE_DAYS = 15.0


@dataclass(frozen=True)
class NewsWindow:
    """Immutable news for one symbol, ending at ``as_of`` (nothing later is reachable)."""

    symbol: str
    as_of: pd.Timestamp
    events: pd.DataFrame  # columns ['date','sentiment'], sorted ascending, every date <= as_of

    def __post_init__(self) -> None:
        e = self.events
        if len(e):
            if e["date"].max() > self.as_of:
                raise ValueError(
                    f"LOOK-AHEAD BLOCKED: news window for {self.symbol!r} contains an event dated "
                    f"{e['date'].max().date()}, after as_of {self.as_of.date()} — future news leaked."
                )

    def _recent(self, days: int) -> pd.DataFrame:
        lo = self.as_of - pd.Timedelta(days=days)
        e = self.events
        return e[(e["date"] > lo) & (e["date"] <= self.as_of)]

    def features(self) -> dict:
        """Compact, look-ahead-safe news features as of the window's date (0 when there's no news)."""
        f: dict[str, float] = {}
        recent = self._recent(30)
        f["news_sentiment_30"] = float(recent["sentiment"].mean()) if len(recent) else 0.0
        f["news_volume_30"] = float(len(recent))
        if len(self.events):
            age = (self.as_of - self.events["date"]).dt.days.to_numpy(dtype=float)
            weight = np.exp(-age / _DECAY_HALFLIFE_DAYS)
            f["news_decay"] = float(np.sum(self.events["sentiment"].to_numpy(dtype=float) * weight))
        else:
            f["news_decay"] = 0.0
        return f


class NewsProvider(ABC):
    @abstractmethod
    def window(self, symbol: str, as_of: DateLike, lookback_days: int = 120) -> NewsWindow: ...


class NoNewsProvider(NewsProvider):
    """A provider with no news — every window is empty (all news features become 0)."""

    def window(self, symbol: str, as_of: DateLike, lookback_days: int = 120) -> NewsWindow:
        empty = pd.DataFrame({"date": pd.to_datetime([]), "sentiment": []})
        return NewsWindow(symbol=symbol, as_of=_ts(as_of), events=empty)


class NewsFileProvider(NewsProvider):
    """Reads a long-format ``news.csv`` and serves look-ahead-safe windows per symbol."""

    def __init__(self, data_dir: Optional[Path | str] = None, filename: str = "news.csv",
                 frame: Optional[pd.DataFrame] = None):
        if frame is None:
            path = Path(data_dir) / filename
            if not path.exists():
                raise FileNotFoundError(f"news file not found: {path}")
            frame = pd.read_csv(path)
        frame = frame.rename(columns={c: c.strip().lower() for c in frame.columns})
        missing = {"date", "symbol", "sentiment"} - set(frame.columns)
        if missing:
            raise ValueError(f"news data must have columns date,symbol,sentiment (missing {sorted(missing)})")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.strip()
        frame["sentiment"] = frame["sentiment"].astype(float).clip(-1.0, 1.0)
        frame = frame.sort_values("date")
        self._by_symbol: dict[str, pd.DataFrame] = {
            sym: g[["date", "sentiment"]].reset_index(drop=True) for sym, g in frame.groupby("symbol")
        }

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "NewsFileProvider":
        return cls(frame=frame)

    def window(self, symbol: str, as_of: DateLike, lookback_days: int = 120) -> NewsWindow:
        asof = _ts(as_of)
        events = self._by_symbol.get(symbol)
        if events is None or len(events) == 0:
            events = pd.DataFrame({"date": pd.to_datetime([]), "sentiment": []})
        else:
            lo = asof - pd.Timedelta(days=lookback_days)
            events = events[(events["date"] > lo) & (events["date"] <= asof)].reset_index(drop=True)
        return NewsWindow(symbol=symbol, as_of=asof, events=events)
