"""Real historical analyst calls from local files — an ``AnalystCallProvider``.

Reads `calls.csv` (or `.json`) into the EXISTING ``Call`` schema. Because a `Call` carries a
record-time `resolution_date` and `initial_price`, this provider needs the price provider to
fix the deadline (call_date + horizon trading days on the benchmark calendar) and read the
call-date entry price — exactly the look-ahead-neutral, record-time logic the engine expects.

Documented, uniform ingest policies (see BACKTEST_PLAN.md). Raw rows that cannot become a valid,
closeable call are DROPPED and logged in ``ingest_issues`` (never silently); the rest become
strict, validated ``Call`` objects:
- unrecognized rating            -> BAD_RATING
- non-positive / missing target   -> BAD_TARGET
- ticker not in the price data    -> UNKNOWN_TICKER
- call date after the last session -> CALL_DATE_OUT_OF_RANGE
- no ticker price on the call date -> NO_ENTRY_PRICE
- deadline beyond the data (call still open) -> HORIZON_BEYOND_DATA
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR
from ..schemas import Call, Rating
from .call_provider import AnalystCallProvider
from .historical_price_provider import HistoricalPriceFileProvider

# Common analyst-rating vocabularies -> our five canonical ratings (lowercased keys).
RATING_SYNONYMS: dict[str, Rating] = {
    "buy": Rating.BUY, "strong buy": Rating.BUY, "conviction buy": Rating.BUY, "top pick": Rating.BUY,
    "overweight": Rating.OVERWEIGHT, "outperform": Rating.OVERWEIGHT, "accumulate": Rating.OVERWEIGHT,
    "add": Rating.OVERWEIGHT, "positive": Rating.OVERWEIGHT, "moderate buy": Rating.OVERWEIGHT,
    "hold": Rating.HOLD, "neutral": Rating.HOLD, "market perform": Rating.HOLD, "sector perform": Rating.HOLD,
    "equal weight": Rating.HOLD, "equalweight": Rating.HOLD, "in-line": Rating.HOLD, "in line": Rating.HOLD,
    "underweight": Rating.UNDERWEIGHT, "underperform": Rating.UNDERWEIGHT, "reduce": Rating.UNDERWEIGHT,
    "negative": Rating.UNDERWEIGHT, "moderate sell": Rating.UNDERWEIGHT,
    "sell": Rating.SELL, "strong sell": Rating.SELL,
}


def normalize_rating(raw: str) -> Optional[Rating]:
    return RATING_SYNONYMS.get(str(raw).strip().lower())


def horizon_months_to_trading_days(months: float) -> int:
    """12 months -> 252, 6 -> 126, 3 -> 63 (rounded). At least 1 trading day."""
    return max(1, round(months / 12.0 * TRADING_DAYS_PER_YEAR))


class HistoricalCallFileProvider(AnalystCallProvider):
    def __init__(
        self,
        data_dir: Path | str,
        price_provider: HistoricalPriceFileProvider,
        calls_filename: Optional[str] = None,
        default_horizon_months: Optional[float] = None,
    ):
        self.data_dir = Path(data_dir)
        self.prices = price_provider
        self.calls_path = self._resolve_calls_path(calls_filename)
        self.default_horizon_months = (
            default_horizon_months
            if default_horizon_months is not None
            else float(price_provider.manifest.get("default_horizon_months", 12))
        )
        self.ingest_issues: list[dict] = []  # populated on get_calls()

    def _resolve_calls_path(self, calls_filename: Optional[str]) -> Path:
        if calls_filename:
            return self.data_dir / calls_filename
        for name in ("calls.csv", "calls.json"):
            if (self.data_dir / name).exists():
                return self.data_dir / name
        return self.data_dir / "calls.csv"

    # -- loading -----------------------------------------------------------------------
    def _read_rows(self) -> list[dict]:
        path = self.calls_path
        if not path.exists():
            raise FileNotFoundError(f"calls file not found: {path}")
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text())
            return [{str(k).strip().lower(): v for k, v in r.items()} for r in rows]
        df = pd.read_csv(path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        return df.to_dict(orient="records")

    def get_calls(self) -> list[Call]:
        self.ingest_issues = []
        calls: list[Call] = []
        for i, row in enumerate(self._read_rows()):
            call, issue = self._build_call(row, i)
            if call is not None:
                calls.append(call)
            elif issue is not None:
                self.ingest_issues.append(issue)
        return sorted(calls, key=lambda c: (c.call_date, c.call_id))

    def _build_call(self, row: dict, idx: int) -> tuple[Optional[Call], Optional[dict]]:
        cid = str(row.get("call_id") or f"row-{idx}")
        ticker = str(row.get("ticker", "")).strip()

        def issue(reason: str, detail: str = "") -> tuple[None, dict]:
            return None, {"call_id": cid, "ticker": ticker, "reason": reason, "detail": detail}

        rating = normalize_rating(row.get("rating", ""))
        if rating is None:
            return issue("BAD_RATING", str(row.get("rating")))

        try:
            target = float(row.get("target_price"))
        except (TypeError, ValueError):
            return issue("BAD_TARGET", str(row.get("target_price")))
        if not (target > 0 and math.isfinite(target)):
            return issue("BAD_TARGET", str(row.get("target_price")))

        if not self.prices.has_symbol(ticker):
            return issue("UNKNOWN_TICKER", ticker)

        raw_date = pd.to_datetime(row.get("call_date")).normalize()
        snapped = self.prices.next_trading_day_on_or_after(raw_date)
        if snapped is None:
            return issue("CALL_DATE_OUT_OF_RANGE", str(raw_date.date()))

        months = row.get("horizon_months")
        try:
            months = float(months) if months not in (None, "") and not (isinstance(months, float) and math.isnan(months)) else self.default_horizon_months
        except (TypeError, ValueError):
            months = self.default_horizon_months
        horizon_days = horizon_months_to_trading_days(months)

        try:
            resolution_ts = self.prices.trading_day_offset(snapped, horizon_days)
        except IndexError:
            return issue("HORIZON_BEYOND_DATA", f"{snapped.date()} + {horizon_days}td")

        if not self.prices.has_data(ticker, snapped):
            return issue("NO_ENTRY_PRICE", f"{ticker} @ {snapped.date()}")
        entry_price = self.prices.price_on(ticker, snapped)

        try:
            call = Call(
                call_id=cid,
                analyst_id=str(row.get("analyst_id") or cid),
                analyst_name=str(row.get("analyst_name") or row.get("analyst_id") or "Unknown"),
                firm=str(row.get("firm") or "Unknown"),
                ticker=ticker,
                rating=rating,
                target_price=round(target, 4),
                call_date=snapped.date(),
                horizon_days=horizon_days,
                resolution_date=resolution_ts.date(),
                initial_price=float(entry_price),
            )
        except Exception as e:  # pragma: no cover - schema guards
            return issue("SCHEMA_ERROR", str(e))
        return call, None
