"""Live web price data — grade a prediction against TODAY's market, via the unchanged engine.

This module adds a *live* price source on top of the existing ``PriceDataProvider`` contract,
so a brand-new prediction can be scored through the SAME look-ahead-safe funnel the back-test
uses (``resolve_call_with_provider`` + ``score_call``). Nothing in the scoring engine changes.

WHAT IS DIFFERENT FROM THE BACK-TEST — read this before trusting a number
------------------------------------------------------------------------
* A live grade of an *open* call is **PROVISIONAL**. The call's horizon has not elapsed yet, so
  we grade it from the call date up to the most recent available trading day ("so far"). That is
  a mark-to-market interim read, NOT the final resolved score. ``LiveGradeResult.provisional``
  says which you got; a call whose original deadline has already passed is graded FINAL.
* A live grade is **NOT reproducible** — it depends on live prices and the date you ran it.
  This is why it is a separate feature, never folded into the reproducible scorecard.

WHAT IS STILL GUARANTEED
------------------------
* Look-ahead safety is intact. We build a ``PriceWindow`` ending at the grading date and never
  fetch or read a price after it; the window's own invariants reject any leak. A PROVISIONAL
  grade uses only data up to "now", which by definition cannot contain the future.
* The engine is reused verbatim — this file only sources prices and hands them to the provider
  contract. Look-ahead/scoring logic is untouched.

OFFLINE-FIRST
-------------
``yfinance`` is imported lazily and only inside ``YFinanceFetcher``. Importing this module costs
no network and no optional dependency. The fetch step is injected (``PriceFetcher``) so the whole
provider/resolve/score path is testable offline with a synthetic frame (see the offline test).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Sequence

import pandas as pd

from ..aggregation import aggregate_analyst
from ..config import DEFAULT_CONFIG, ScorecardConfig
from ..resolution import resolve_call_with_provider
from ..schemas import AnalystScore, Call, CallScore, Rating, Resolution
from ..scoring import score_call
from ..verdicts import default_verdict_generator
from .historical_price_provider import HistoricalPriceFileProvider
from .price_provider import _ts

DEFAULT_BENCHMARK = "^GSPC"  # S&P 500 index on Yahoo Finance


class LiveGradeError(Exception):
    """A user-facing problem fetching or grading live data (missing dep, no data, too recent…)."""


# --------------------------------------------------------------------------------------
# Fetcher seam — the only place the network is touched. Injectable for offline testing.
# --------------------------------------------------------------------------------------


class PriceFetcher(ABC):
    """Fetch adjusted closes for ``symbols`` over [start, end] as a wide frame.

    Returns: DataFrame indexed by tz-naive Timestamp (midnight for daily; the exact bar time for
    intraday), one column per symbol, values = adjusted close. Missing observations are NaN.
    ``interval`` is "1d" (daily) or an intraday string like "30m"; daily callers may omit it and
    daily implementations may ignore it.
    """

    @abstractmethod
    def fetch(self, symbols: Sequence[str], start: date, end: date, interval: str = "1d") -> pd.DataFrame: ...


class YFinanceFetcher(PriceFetcher):
    """Live fetcher backed by yfinance (lazy import — only needed when actually grading live)."""

    def fetch(self, symbols: Sequence[str], start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf  # lazy: keeps the core offline-installable
        except ImportError as e:  # pragma: no cover - environment dependent
            raise LiveGradeError(
                "Live grading needs the optional 'yfinance' package, which isn't installed.\n"
                "Install it into the project venv:  .venv/bin/pip install -r requirements-live.txt"
            ) from e

        symbols = list(dict.fromkeys(symbols))  # de-dup, keep order
        intraday = interval != "1d"
        try:
            # end is exclusive in yfinance -> pad by a day so 'end' itself is included.
            raw = yf.download(
                symbols,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval=interval,     # "1d" daily, "30m"/"15m"/… intraday (≤ ~60 days of history)
                auto_adjust=True,      # 'Close' is split/dividend-adjusted
                progress=False,
                threads=False,
            )
        except Exception as e:  # pragma: no cover - network dependent
            raise LiveGradeError(f"Could not fetch live prices (network error): {e}") from e

        if raw is None or len(raw) == 0:
            raise LiveGradeError(
                f"No live {interval} price data for {symbols} between {start} and {end}. "
                "Intraday history only goes back ~60 days; check the symbols and your connection."
            )

        # Normalize yfinance's shape (MultiIndex columns for many symbols, flat for one).
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
        else:
            close = raw[["Close"]].copy()
            close.columns = [symbols[0]]

        idx = pd.DatetimeIndex(close.index).tz_localize(None)
        close.index = idx if intraday else idx.normalize()   # keep the bar time for intraday
        close = close[~close.index.duplicated(keep="last")].sort_index()
        if not intraday:
            close = close.loc[close.index <= _ts(end)]        # belt-and-suspenders: never past 'end'
        return close.dropna(axis=1, how="all")


# --------------------------------------------------------------------------------------
# The live provider — reuses the proven frame-based provider, only the SOURCE differs.
# --------------------------------------------------------------------------------------


class LiveWebPriceProvider(HistoricalPriceFileProvider):
    """A ``PriceDataProvider`` whose prices come from the live web instead of local files.

    It fetches [start, end] adjusted daily closes for the requested tickers + benchmark, then
    reuses ``HistoricalPriceFileProvider``'s frame handling and the inherited, look-ahead-safe
    ``window_for_call``. The engine cannot tell (or care) that the prices arrived over the wire.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        benchmark_symbol: str,
        start: date,
        end: date,
        fetcher: Optional[PriceFetcher] = None,
    ):
        if end <= start:
            raise LiveGradeError(f"end date {end} must be after start date {start}.")
        self._fetcher = fetcher or YFinanceFetcher()
        symbols = list(dict.fromkeys([*tickers, benchmark_symbol]))
        frame = self._fetcher.fetch(symbols, start, end)

        if benchmark_symbol not in frame.columns:
            raise LiveGradeError(
                f"No live data for the benchmark {benchmark_symbol!r}. "
                "Pick a valid index symbol (e.g. ^GSPC for the S&P 500)."
            )
        for t in tickers:
            if t not in frame.columns:
                raise LiveGradeError(
                    f"No live price data found for ticker {t!r}. "
                    "Check the symbol (US tickers like AAPL, MSFT work best)."
                )

        # Reuse the file provider's validated frame init (positivity checks, trading calendar…).
        self.data_dir = None
        self._manifest = {"benchmark_symbol": benchmark_symbol, "source": "live-web", "is_sample": False}
        self._benchmark = benchmark_symbol
        self._init_from_frame(frame)

    def common_trading_days(self, ticker: str) -> pd.DatetimeIndex:
        """Dates present for BOTH ``ticker`` and the benchmark (valid window endpoints)."""
        return self.price_series(ticker).index.intersection(self.trading_days())


# --------------------------------------------------------------------------------------
# Result + the one-call front door used by the Streamlit "Live Grader" tab.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveGradeResult:
    call: Call
    resolution: Resolution
    call_score: CallScore
    analyst_score: AnalystScore
    verdict: str
    provisional: bool          # True = horizon not yet elapsed (interim, mark-to-market)
    graded_through: date       # the trading day the grade was computed at
    original_deadline: Optional[date]
    benchmark_symbol: str

    @property
    def beat_market(self) -> Optional[float]:
        return self.analyst_score.beat_market

    @property
    def status(self) -> str:
        return "PROVISIONAL (horizon still open)" if self.provisional else "FINAL (horizon elapsed)"


def _snap_forward(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(ts, side="left")
    return index[pos] if pos < len(index) else None


def _snap_back(index: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(ts, side="right") - 1
    return index[pos] if pos >= 0 else None


def grade_live_prediction(
    *,
    ticker: str,
    rating: Rating | str,
    target_price: float,
    call_date: date,
    horizon_trading_days: Optional[int] = None,
    resolution_date: Optional[date] = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
    analyst_name: str = "Your prediction",
    firm: str = "Live",
    asof: Optional[date] = None,
    fetcher: Optional[PriceFetcher] = None,
    config: ScorecardConfig = DEFAULT_CONFIG,
) -> LiveGradeResult:
    """Fetch live prices and grade ONE prediction through the unchanged engine.

    The window is [call_date → grading_date], where grading_date is the most recent available
    trading day on/before the call's deadline (if the deadline has passed → FINAL) or on/before
    today (if it hasn't → PROVISIONAL). Only in-window prices are ever read, so the grade is
    look-ahead-safe by construction even though it uses live data.
    """
    rating = Rating(rating) if not isinstance(rating, Rating) else rating
    asof = asof or date.today()
    ticker = ticker.strip().upper()
    if not ticker:
        raise LiveGradeError("Please enter a ticker symbol.")
    if call_date >= asof:
        raise LiveGradeError(
            f"The call date {call_date} must be in the past — you can't grade a call before it has had "
            "any time to play out."
        )
    if target_price <= 0:
        raise LiveGradeError("Target price must be greater than 0.")

    # Fetch a little past the call date through today; the window enforces the real bound.
    provider = LiveWebPriceProvider([ticker], benchmark_symbol, call_date, asof, fetcher=fetcher)
    days = provider.common_trading_days(ticker)
    if len(days) < 2:
        raise LiveGradeError(
            f"Not enough live trading data for {ticker} since {call_date} to grade a call "
            "(need at least two trading days). Try an earlier call date."
        )

    call_ts = _snap_forward(days, _ts(call_date))
    if call_ts is None or call_ts >= days[-1]:
        raise LiveGradeError(
            f"The call date {call_date} is too recent — there isn't a later trading day with data yet."
        )
    call_pos = days.get_loc(call_ts)

    # Decide the grading date + whether the horizon has actually elapsed.
    last_ts = days[-1]
    original_deadline: Optional[date] = None
    if horizon_trading_days is not None:
        if horizon_trading_days <= 0:
            raise LiveGradeError("Horizon (trading days) must be a positive number.")
        deadline_pos = call_pos + horizon_trading_days
        if deadline_pos < len(days):
            grading_ts = days[deadline_pos]          # deadline already within available data
            provisional = False
        else:
            grading_ts = last_ts                     # deadline is in the future -> interim grade
            provisional = True
        original_deadline = grading_ts.date() if not provisional else None
    elif resolution_date is not None:
        if resolution_date <= call_date:
            raise LiveGradeError(f"Resolution date {resolution_date} must be after the call date {call_date}.")
        original_deadline = resolution_date
        if _ts(resolution_date) <= last_ts:
            grading_ts = _snap_back(days, _ts(resolution_date))   # deadline passed -> FINAL
            provisional = False
        else:
            grading_ts = last_ts                                  # deadline future -> PROVISIONAL
            provisional = True
    else:
        grading_ts = last_ts                          # open-ended -> always "so far"
        provisional = True

    if grading_ts <= call_ts:
        raise LiveGradeError(
            "The grading date isn't after the call date yet — the horizon is too short to score. "
            "Use an earlier call date or a longer horizon."
        )

    horizon_days = days.get_loc(grading_ts) - call_pos
    initial_price = provider.price_on(ticker, call_ts)

    call = Call(
        call_id=f"live-{ticker}-{call_ts.date()}",
        analyst_id="live",
        analyst_name=analyst_name,
        firm=firm,
        ticker=ticker,
        rating=rating,
        target_price=float(target_price),
        call_date=call_ts.date(),
        horizon_days=int(horizon_days),
        resolution_date=grading_ts.date(),
        initial_price=float(initial_price),
    )

    # --- the UNCHANGED engine path (identical to the back-test) ---
    resolution = resolve_call_with_provider(call, provider)
    call_score = score_call(call, resolution, config)
    analyst_score = aggregate_analyst([call_score], analyst_name=analyst_name, firm=firm)
    verdict = default_verdict_generator().verdict(analyst_score)

    return LiveGradeResult(
        call=call,
        resolution=resolution,
        call_score=call_score,
        analyst_score=analyst_score,
        verdict=verdict,
        provisional=provisional,
        graded_through=grading_ts.date(),
        original_deadline=original_deadline,
        benchmark_symbol=benchmark_symbol,
    )
