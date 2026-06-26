"""Real historical prices from local files — a ``PriceDataProvider`` for the back-test.

Reads a long-format `prices.csv` (`date,symbol,close`) plus a `manifest.json` naming the
benchmark symbol. Implements ONLY the four abstract methods of ``PriceDataProvider``; the
look-ahead-safe `window_for_call`, `price_window_series`, and `trading_day_offset` are
inherited unchanged, so historical resolution runs through the exact same engine path.

Conventions (documented, uniform — see BACKTEST_PLAN.md):
- `close` is the split- and dividend-ADJUSTED close; returns are computed on it directly.
- The TRADING CALENDAR is the benchmark's available dates (a broad index trades every market day).
- Coverage may be ragged: a delisted ticker simply stops having rows; interior gaps are allowed.
  A window whose endpoints aren't both present (e.g. the ticker delisted before the horizon) is
  rejected by ``PriceWindow`` — the back-test runner classifies that as a skip, never a leak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .price_provider import DateLike, PriceDataProvider, _ts


class HistoricalPriceFileProvider(PriceDataProvider):
    def __init__(
        self,
        data_dir: Path | str,
        benchmark_symbol: Optional[str] = None,
        prices_filename: str = "prices.csv",
        manifest_filename: str = "manifest.json",
    ):
        self.data_dir = Path(data_dir)
        self._manifest = self._read_manifest(self.data_dir / manifest_filename)
        bench = benchmark_symbol or self._manifest.get("benchmark_symbol")
        if not bench:
            raise ValueError(
                "benchmark_symbol must be given explicitly or set in manifest.json "
                f"(looked in {self.data_dir / manifest_filename})"
            )
        self._benchmark = bench
        frame = self._read_prices(self.data_dir / prices_filename)
        self._init_from_frame(frame)

    # -- alternate constructor (tests / in-memory) -------------------------------------
    @classmethod
    def from_frame(cls, frame: pd.DataFrame, benchmark_symbol: str, manifest: Optional[dict] = None) -> "HistoricalPriceFileProvider":
        """Build directly from a wide price frame (index=dates, columns=symbols, NaN=missing)."""
        obj = cls.__new__(cls)
        obj.data_dir = None
        obj._manifest = manifest or {}
        obj._benchmark = benchmark_symbol
        obj._init_from_frame(frame)
        return obj

    # -- loading -----------------------------------------------------------------------
    @staticmethod
    def _read_manifest(path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _read_prices(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"prices file not found: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        missing = {"date", "symbol", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} must have columns date,symbol,close (missing {sorted(missing)})")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = df["symbol"].astype(str).str.strip()
        # long -> wide; 'last' resolves accidental duplicate (date,symbol) rows deterministically
        return df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")

    def _init_from_frame(self, frame: pd.DataFrame) -> None:
        frame = frame.sort_index()
        if self._benchmark not in frame.columns:
            raise ValueError(f"benchmark {self._benchmark!r} not found in price data columns {list(frame.columns)}")
        self._series: dict[str, pd.Series] = {}
        for sym in frame.columns:
            s = frame[sym].dropna()
            if len(s) == 0:
                continue
            if (s <= 0).any():
                bad = s[s <= 0].index[:3]
                raise ValueError(f"symbol {sym!r} has non-positive close prices at {list(bad)}")
            self._series[sym] = s.astype(float)
        if self._benchmark not in self._series or len(self._series[self._benchmark]) < 2:
            raise ValueError(f"benchmark {self._benchmark!r} needs >= 2 price observations")
        self._trading_days = pd.DatetimeIndex(self._series[self._benchmark].index)
        self._frame = frame

    # -- PriceDataProvider interface ---------------------------------------------------
    @property
    def benchmark_symbol(self) -> str:
        return self._benchmark

    def tickers(self) -> list[str]:
        return [s for s in sorted(self._series) if s != self._benchmark]

    def trading_days(self) -> pd.DatetimeIndex:
        return self._trading_days

    def price_series(self, symbol: str) -> pd.Series:
        if symbol not in self._series:
            raise KeyError(f"Unknown symbol: {symbol!r}")
        return self._series[symbol]

    # -- helpers used by the call provider & back-test runner --------------------------
    @property
    def manifest(self) -> dict:
        return self._manifest

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._series

    def has_data(self, symbol: str, on: DateLike) -> bool:
        """True if ``symbol`` has a real price on ``on`` (used to classify resolvability)."""
        if symbol not in self._series:
            return False
        return _ts(on) in self._series[symbol].index

    def next_trading_day_on_or_after(self, on: DateLike) -> Optional[pd.Timestamp]:
        """Snap a raw date forward to the next benchmark trading day (None if past the data)."""
        on = _ts(on)
        days = self._trading_days
        pos = days.searchsorted(on, side="left")
        if pos >= len(days):
            return None
        return days[pos]
