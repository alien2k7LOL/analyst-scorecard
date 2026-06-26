"""Grade a user's FORWARD prediction with a calibrated probability, using real history.

The flow, all on the look-ahead-safe spine:
  1. Fetch ~``years_history`` of real daily prices for the ticker + benchmark (yfinance).
  2. Self-calibrate ON THAT TICKER'S OWN PAST — run the forecast calibration backtest over its
     history so the probability is corrected by how this stock has actually behaved ("use older
     data to refine accuracy"). If there isn't enough history, fall back to the raw GBM model and
     say so.
  3. Build the prediction's features from data up to as_of (today) and emit the calibrated touch
     probability, alongside the self-calibration's held-out reliability as the credibility context.

News is intentionally NOT used here: trustworthy point-in-time historical news isn't freely
available, and using today's news to score a backtest would leak the future. The news contribution
is demonstrated on the offline sample (see forecast/backtest.py); the live path is price-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from ..providers.historical_price_provider import HistoricalPriceFileProvider
from ..providers.live_web_price_provider import LiveGradeError, PriceFetcher, YFinanceFetcher
from ..schemas import Direction
from .backtest import ForecastBacktest, ForecastGenConfig
from .features import build_features
from .interval import BarInterval
from .prediction import Prediction, PredictionKind

DEFAULT_BENCHMARK = "^GSPC"
INTRADAY_LOOKBACK_DAYS = 58       # yfinance serves ~60 days of 30-min history
INTRADAY_LOOKBACK_BARS = 260      # ~20 trading days of 30-min bars for the feature lookback


@dataclass(frozen=True)
class ForecastGrade:
    ticker: str
    direction: Direction
    target_price: float
    as_of: date
    deadline: date
    n_days: int                     # horizon in bars (trading days for daily, 30-min bars for intraday)
    s0: float
    raw_probability: float          # closed-form GBM probability for this kind (uncalibrated)
    probability: float              # calibrated if possible, else == raw_probability
    calibrated: bool
    kind: PredictionKind = PredictionKind.TOUCH
    band_pct: Optional[float] = None
    interval: BarInterval = BarInterval.DAILY
    deadline_label: str = ""        # human, time-aware deadline string (intraday shows HH:MM)
    drift_bar: float = 0.0          # per-bar drift the model used (for the proof breakdown)
    vol_bar: float = 0.0            # per-bar volatility the model used
    self_cal_metrics: dict = field(default_factory=dict)  # held-out test metrics of the self-cal model
    reliability: list = field(default_factory=list)
    benchmark_symbol: str = DEFAULT_BENCHMARK
    history_start: Optional[date] = None
    history_end: Optional[date] = None


def _finalize_grade(
    *,
    provider: HistoricalPriceFileProvider,
    ticker: str,
    pred: Prediction,
    interval: BarInterval,
    benchmark_symbol: str,
    gen_fn: Callable[[int], ForecastGenConfig],
    lookback_days: int,
    l2: float,
    deadline_label: str,
) -> ForecastGrade:
    """Shared tail: build features, self-calibrate on the ticker's own past, assemble the grade."""
    row = build_features(provider, pred, None, lookback_days)  # price-only, look-ahead-safe
    raw_p = row.gbm_p
    probability, calibrated, metrics, reliability = raw_p, False, {}, []
    try:
        bt = ForecastBacktest(provider, news_provider=None, gen=gen_fn(int(row.n_days)),
                              tickers=[ticker], l2=l2).run()
        probability, calibrated, metrics, reliability = bt.predict_one(row), True, bt.metrics, bt.reliability
    except (ValueError, KeyError):
        pass  # not enough resolvable history -> uncalibrated raw probability
    idx = provider.price_series(ticker).index
    return ForecastGrade(
        ticker=ticker, direction=pred.direction, target_price=pred.target_price,
        as_of=row.as_of, deadline=pred.deadline.date(), n_days=row.n_days, s0=row.s0,
        raw_probability=raw_p, probability=float(probability), calibrated=calibrated,
        kind=pred.kind, band_pct=pred.band_pct, interval=interval, deadline_label=deadline_label,
        drift_bar=row.mu, vol_bar=row.sigma,
        self_cal_metrics=metrics, reliability=reliability, benchmark_symbol=benchmark_symbol,
        history_start=idx.min().date(), history_end=idx.max().date(),
    )


def grade_forecast_live(
    *,
    ticker: str,
    target_price: float,
    deadline,
    direction: Direction | str,
    kind: PredictionKind | str = PredictionKind.TOUCH,
    band_pct: Optional[float] = None,
    interval: BarInterval | str = BarInterval.DAILY,
    as_of: Optional[date] = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
    years_history: int = 6,
    fetcher: Optional[PriceFetcher] = None,
    l2: float = 1.0,
) -> ForecastGrade:
    fetcher = fetcher or YFinanceFetcher()
    direction = direction if isinstance(direction, Direction) else Direction(direction)
    kind = kind if isinstance(kind, PredictionKind) else PredictionKind(kind)
    interval = interval if isinstance(interval, BarInterval) else BarInterval(interval)
    if direction == Direction.FLAT:
        raise LiveGradeError("Pick a direction: UP (rises to target) or DOWN (falls to target).")
    ticker = ticker.strip().upper()
    if not ticker:
        raise LiveGradeError("Please enter a ticker symbol.")
    if target_price <= 0:
        raise LiveGradeError("Target price must be greater than 0.")
    # Band only applies to terminal predictions; an empty band means "at or through the target".
    band_pct = (float(band_pct) if band_pct else None) if kind == PredictionKind.TERMINAL else None

    if interval == BarInterval.MIN30:
        return _grade_intraday(
            ticker=ticker, target_price=float(target_price), deadline=deadline, direction=direction,
            kind=kind, band_pct=band_pct, end_day=as_of, benchmark_symbol=benchmark_symbol,
            fetcher=fetcher, l2=l2,
        )

    # ---- daily ----
    as_of = as_of or date.today()
    deadline = pd.Timestamp(deadline)
    if deadline.date() <= as_of:
        raise LiveGradeError(f"The deadline {deadline.date()} must be in the future (after {as_of}).")
    start = as_of - timedelta(days=int(365.25 * years_history))
    frame = fetcher.fetch([ticker, benchmark_symbol], start, as_of)
    if benchmark_symbol not in frame.columns:
        raise LiveGradeError(f"No live data for the benchmark {benchmark_symbol!r} (e.g. ^GSPC).")
    if ticker not in frame.columns:
        raise LiveGradeError(f"No live price data for ticker {ticker!r}. Check the symbol (e.g. AAPL).")

    provider = HistoricalPriceFileProvider.from_frame(frame, benchmark_symbol)
    pred = Prediction(
        prediction_id=f"live-{ticker}-{as_of}", ticker=ticker, as_of=as_of,
        target_price=float(target_price), deadline=deadline.to_pydatetime(),
        direction=direction, kind=kind, band_pct=band_pct, interval=BarInterval.DAILY,
    )

    def gen_fn(n_days: int) -> ForecastGenConfig:
        return ForecastGenConfig(
            stride_days=15, horizons=tuple(sorted({63, 126, max(5, n_days)})),
            up_offsets=(0.05, 0.10, 0.20), down_offsets=(0.05, 0.10, 0.20),
            kinds=(kind,), terminal_bands=(band_pct,),
        )

    return _finalize_grade(
        provider=provider, ticker=ticker, pred=pred, interval=BarInterval.DAILY,
        benchmark_symbol=benchmark_symbol, gen_fn=gen_fn, lookback_days=252, l2=l2,
        deadline_label=str(deadline.date()),
    )


def _grade_intraday(
    *,
    ticker: str,
    target_price: float,
    deadline,
    direction: Direction,
    kind: PredictionKind,
    band_pct: Optional[float],
    end_day: Optional[date],
    benchmark_symbol: str,
    fetcher: PriceFetcher,
    l2: float,
) -> ForecastGrade:
    """30-min intraday grade: fetch recent 30-min bars and self-calibrate on them (thin history)."""
    deadline = pd.Timestamp(deadline)
    end_day = end_day or date.today()
    start = end_day - timedelta(days=INTRADAY_LOOKBACK_DAYS)
    frame = fetcher.fetch([ticker, benchmark_symbol], start, end_day, interval="30m")
    if benchmark_symbol not in frame.columns:
        raise LiveGradeError(f"No live 30-min data for the benchmark {benchmark_symbol!r} (e.g. ^GSPC).")
    if ticker not in frame.columns:
        raise LiveGradeError(
            f"No live 30-min data for ticker {ticker!r} (intraday history is only ~60 days)."
        )

    provider = HistoricalPriceFileProvider.from_frame(frame, benchmark_symbol)
    as_of_ts = provider.price_series(ticker).index[-1]   # grade as of the most recent 30-min bar
    if deadline <= as_of_ts:
        raise LiveGradeError(
            f"The 30-min deadline {deadline} must be after the latest bar ({as_of_ts})."
        )

    pred = Prediction(
        prediction_id=f"live-{ticker}-{as_of_ts:%Y%m%d-%H%M}", ticker=ticker, as_of=as_of_ts.to_pydatetime(),
        target_price=float(target_price), deadline=deadline.to_pydatetime(),
        direction=direction, kind=kind, band_pct=band_pct, interval=BarInterval.MIN30,
    )

    def gen_fn(n_bars: int) -> ForecastGenConfig:
        return ForecastGenConfig(
            stride_days=13, horizons=tuple(sorted({13, 26, max(3, n_bars)})),
            up_offsets=(0.004, 0.008, 0.015), down_offsets=(0.004, 0.008, 0.015),
            kinds=(kind,), terminal_bands=(band_pct,),
            interval=BarInterval.MIN30, lookback_days=INTRADAY_LOOKBACK_BARS, min_history=130,
        )

    return _finalize_grade(
        provider=provider, ticker=ticker, pred=pred, interval=BarInterval.MIN30,
        benchmark_symbol=benchmark_symbol, gen_fn=gen_fn, lookback_days=INTRADAY_LOOKBACK_BARS, l2=l2,
        deadline_label=deadline.strftime("%Y-%m-%d %H:%M"),
    )
