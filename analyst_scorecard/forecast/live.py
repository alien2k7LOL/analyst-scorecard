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
from datetime import date, timedelta
from typing import Optional

from ..providers.historical_price_provider import HistoricalPriceFileProvider
from ..providers.live_web_price_provider import LiveGradeError, PriceFetcher, YFinanceFetcher
from ..schemas import Direction
from .backtest import ForecastBacktest, ForecastGenConfig
from .features import build_features
from .prediction import Prediction, PredictionKind

DEFAULT_BENCHMARK = "^GSPC"


@dataclass(frozen=True)
class ForecastGrade:
    ticker: str
    direction: Direction
    target_price: float
    as_of: date
    deadline: date
    n_days: int
    s0: float
    raw_probability: float          # closed-form GBM probability for this kind (uncalibrated)
    probability: float              # calibrated if possible, else == raw_probability
    calibrated: bool
    kind: PredictionKind = PredictionKind.TOUCH
    band_pct: Optional[float] = None
    self_cal_metrics: dict = field(default_factory=dict)  # held-out test metrics of the self-cal model
    reliability: list = field(default_factory=list)
    benchmark_symbol: str = DEFAULT_BENCHMARK
    history_start: Optional[date] = None
    history_end: Optional[date] = None


def grade_forecast_live(
    *,
    ticker: str,
    target_price: float,
    deadline: date,
    direction: Direction | str,
    kind: PredictionKind | str = PredictionKind.TOUCH,
    band_pct: Optional[float] = None,
    as_of: Optional[date] = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
    years_history: int = 6,
    fetcher: Optional[PriceFetcher] = None,
    l2: float = 1.0,
) -> ForecastGrade:
    fetcher = fetcher or YFinanceFetcher()
    as_of = as_of or date.today()
    direction = direction if isinstance(direction, Direction) else Direction(direction)
    kind = kind if isinstance(kind, PredictionKind) else PredictionKind(kind)
    if direction == Direction.FLAT:
        raise LiveGradeError("Pick a direction: UP (rises to target) or DOWN (falls to target).")
    ticker = ticker.strip().upper()
    if not ticker:
        raise LiveGradeError("Please enter a ticker symbol.")
    if target_price <= 0:
        raise LiveGradeError("Target price must be greater than 0.")
    if deadline <= as_of:
        raise LiveGradeError(f"The deadline {deadline} must be in the future (after {as_of}).")
    # Band only applies to terminal predictions; an empty band means "at or through the target".
    band_pct = (float(band_pct) if band_pct else None) if kind == PredictionKind.TERMINAL else None

    start = as_of - timedelta(days=int(365.25 * years_history))
    frame = fetcher.fetch([ticker, benchmark_symbol], start, as_of)
    if benchmark_symbol not in frame.columns:
        raise LiveGradeError(f"No live data for the benchmark {benchmark_symbol!r} (e.g. ^GSPC).")
    if ticker not in frame.columns:
        raise LiveGradeError(f"No live price data for ticker {ticker!r}. Check the symbol (e.g. AAPL).")

    provider = HistoricalPriceFileProvider.from_frame(frame, benchmark_symbol)
    pred = Prediction(
        prediction_id=f"live-{ticker}-{as_of}",
        ticker=ticker,
        as_of=as_of,
        target_price=float(target_price),
        deadline=deadline,
        direction=direction,
        kind=kind,
        band_pct=band_pct,
    )
    row = build_features(provider, pred)  # price-only; lookback ends at as_of (look-ahead-safe)
    raw_p = row.gbm_p

    # Self-calibrate on this ticker's own history, matching the SAME kind/band the user predicted (so
    # the calibrator is trained on the question being asked) and including the actual horizon. Price-
    # only; falls back to the raw model if there isn't enough resolvable history.
    probability, calibrated, metrics, reliability = raw_p, False, {}, []
    try:
        horizons = tuple(sorted({63, 126, max(5, int(row.n_days))}))
        gen = ForecastGenConfig(
            stride_days=15, horizons=horizons,
            up_offsets=(0.05, 0.10, 0.20), down_offsets=(0.05, 0.10, 0.20),
            kinds=(kind,), terminal_bands=(band_pct,),
        )
        bt = ForecastBacktest(provider, news_provider=None, gen=gen, tickers=[ticker], l2=l2).run()
        probability, calibrated, metrics, reliability = bt.predict_one(row), True, bt.metrics, bt.reliability
    except (ValueError, KeyError):
        pass  # not enough resolvable history -> uncalibrated raw probability

    idx = provider.price_series(ticker).index
    return ForecastGrade(
        ticker=ticker, direction=direction, target_price=float(target_price),
        as_of=row.as_of, deadline=deadline, n_days=row.n_days, s0=row.s0,
        raw_probability=raw_p, probability=float(probability), calibrated=calibrated,
        kind=kind, band_pct=band_pct,
        self_cal_metrics=metrics, reliability=reliability, benchmark_symbol=benchmark_symbol,
        history_start=idx.min().date(), history_end=idx.max().date(),
    )
