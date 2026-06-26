"""Assemble an analyst-intelligence report by REUSING the existing engine end to end.

Given an ``ExtractedRecommendation`` this produces a research report with:
  * Live scorecard      — grade_live_prediction on REAL prices (return, alpha, distance, days, status)
  * Forward outlook     — grade_forecast_live: calibrated P(target reached within ~12 months)
  * Historical context  — the analyst's track record from the loaded back-test, matched by name
  * Similar past calls   — resolved calls on the same ticker / rating, and how they fared
  * Plain-English summary — a DETERMINISTIC template filled from the computed numbers (no hallucination)

Nothing here re-implements scoring; it calls the same look-ahead-safe functions the rest of the app
uses. The live/forward steps need network; they're wrapped so a failure degrades the report instead
of crashing it, and every price fetcher is injectable so the whole thing is testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from ..providers.live_web_price_provider import (
    DEFAULT_BENCHMARK,
    LiveGradeError,
    PriceFetcher,
    YFinanceFetcher,
    grade_live_prediction,
)
from ..schemas import RATING_TO_DIRECTION, Direction, Rating
from .extract import ExtractedRecommendation

# Forecast import is module-local in build_report to avoid a heavy import at module load.


@dataclass
class LiveScorecard:
    current_price: float
    call_price: float
    return_since_pub: float
    benchmark_return: float
    alpha: Optional[float]
    distance_to_target: float
    days_since: int
    status: str                 # "PROVISIONAL …" / "FINAL …"
    provisional: bool
    graded_through: date
    direction_pass: bool
    verdict: str


@dataclass
class HistoricalContext:
    found: bool
    analyst: Optional[str] = None
    win_rate: Optional[float] = None
    avg_alpha: Optional[float] = None
    n_recs: int = 0
    on_this_stock_n: int = 0
    on_this_stock_avg_alpha: Optional[float] = None


@dataclass
class ForwardOutlook:
    probability: float
    calibrated: bool
    deadline: date
    horizon_days: int
    direction: Direction


@dataclass
class SimilarCall:
    ticker: str
    rating: str
    call_date: date
    resolution_date: date
    stock_return: float
    benchmark_return: float
    beat: Optional[float]


@dataclass
class ReportChart:
    dates: list                 # list[date], call_date … graded_through
    ticker: list                # ticker close path
    benchmark: list             # benchmark close path (aligned to ticker dates)
    call_price: float
    target: float


@dataclass
class AnalystReport:
    rec: ExtractedRecommendation
    call_date: date
    call_date_assumed: bool
    benchmark_symbol: str
    live: Optional[LiveScorecard] = None
    live_error: Optional[str] = None
    historical: Optional[HistoricalContext] = None
    forward: Optional[ForwardOutlook] = None
    forward_error: Optional[str] = None
    similar: list[SimilarCall] = field(default_factory=list)
    chart: Optional[ReportChart] = None
    summary: str = ""


def _historical_context(rec: ExtractedRecommendation, historical_result) -> Optional[HistoricalContext]:
    if historical_result is None or not rec.analyst:
        return None
    name = rec.analyst.strip().lower()
    match = None
    for score in getattr(historical_result, "leaderboard").rows:
        if score.analyst_name.strip().lower() == name:
            match = score
            break
    if match is None:
        return HistoricalContext(found=False, analyst=rec.analyst)

    on_stock = [cs for cs in match.call_scores if rec.ticker and cs.ticker.upper() == rec.ticker.upper()
                and cs.beat is not None]
    on_stock_alpha = (sum(cs.beat for cs in on_stock) / len(on_stock)) if on_stock else None
    return HistoricalContext(
        found=True, analyst=match.analyst_name, win_rate=match.direction_hit_rate,
        avg_alpha=match.beat_market, n_recs=match.n_directional,
        on_this_stock_n=len(on_stock), on_this_stock_avg_alpha=on_stock_alpha,
    )


def _similar_calls(rec: ExtractedRecommendation, historical_result, limit: int = 6) -> list[SimilarCall]:
    if historical_result is None:
        return []
    rating = rec.rating
    direction = RATING_TO_DIRECTION.get(Rating(rating)) if rating in {r.value for r in Rating} else None
    same_ticker, same_dir = [], []
    for score in getattr(historical_result, "leaderboard").rows:
        for cs in score.call_scores:
            r = cs.resolution
            sc = SimilarCall(ticker=cs.ticker, rating=cs.rating.value, call_date=r.call_date,
                             resolution_date=r.resolution_date, stock_return=r.stock_return,
                             benchmark_return=r.benchmark_return, beat=cs.beat)
            if rec.ticker and cs.ticker.upper() == rec.ticker.upper():
                same_ticker.append(sc)
            elif direction is not None and cs.implied_direction == direction:
                same_dir.append(sc)
    return (same_ticker + same_dir)[:limit]


def _forward_outlook(rec, call_date, asof, benchmark, price_fetcher) -> tuple[Optional[ForwardOutlook], Optional[str]]:
    direction = RATING_TO_DIRECTION.get(Rating(rec.rating)) if rec.rating in {r.value for r in Rating} else None
    if direction in (None, Direction.FLAT):
        return None, "Hold/neutral rating — no directional target to forecast."
    from ..forecast.interval import BarInterval
    from ..forecast.live import grade_forecast_live
    from ..forecast.prediction import PredictionKind

    deadline = asof + timedelta(days=365)   # classic ~12-month target horizon, from today
    try:
        g = grade_forecast_live(
            ticker=rec.ticker, target_price=float(rec.target_price), deadline=deadline,
            direction=direction, kind=PredictionKind.TOUCH, interval=BarInterval.DAILY,
            as_of=asof, benchmark_symbol=benchmark, fetcher=price_fetcher,
        )
    except LiveGradeError as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001 - never crash the report
        return None, f"forecast unavailable: {e}"
    return ForwardOutlook(probability=g.probability, calibrated=g.calibrated, deadline=deadline,
                          horizon_days=g.n_days, direction=direction), None


def _summary(rec, live, historical, forward) -> str:
    parts: list[str] = []
    rating = rec.rating or "recommendation"
    if live is not None:
        moved = "appreciated" if live.return_since_pub >= 0 else "declined"
        rel = "outperformed" if (live.alpha or 0) >= 0 else "underperformed"
        s = (f"This {rating} call has {moved} {abs(live.return_since_pub)*100:.1f}% since "
             f"{'publication' if not rec.publication_date else rec.publication_date}")
        if live.alpha is not None:
            s += f" and has {rel} the benchmark by {abs(live.alpha)*100:.1f} points"
        parts.append(s + ".")
        is_rating = rec.rating in {r.value for r in Rating}
        up = (RATING_TO_DIRECTION.get(Rating(rec.rating)) == Direction.UP) if is_rating else None
        target_price = float(rec.target_price)
        if abs(live.distance_to_target) < 0.005:
            parts.append("The price is essentially at the target now.")
        elif up is None:
            parts.append("This is a neutral (Hold) call, so there's no directional target to clear.")
        else:
            hit = (up and live.current_price >= target_price) or ((not up) and live.current_price <= target_price)
            side = "above" if live.distance_to_target >= 0 else "below"
            if hit:
                parts.append(f"The price has already reached the ${target_price:,.0f} target.")
            else:
                parts.append(f"The target is {abs(live.distance_to_target)*100:.0f}% {side} today's price — "
                             "not yet reached, so the call still has room to play out.")
    if historical is not None and historical.found:
        wr = f"{historical.win_rate*100:.0f}%" if historical.win_rate is not None else "n/a"
        aa = f"{historical.avg_alpha*100:+.1f}%" if historical.avg_alpha is not None else "n/a"
        parts.append(f"{historical.analyst} has a {wr} direction hit-rate and {aa} average alpha across "
                     f"{historical.n_recs} graded calls on file.")
    if forward is not None:
        parts.append(f"The calibrated model puts ~{forward.probability*100:.0f}% odds on reaching the "
                     f"target within ~12 months{' (self-calibrated)' if forward.calibrated else ''}.")
    if not parts:
        parts.append("Not enough live data to assess this recommendation yet.")
    return " ".join(parts)


def build_report(
    rec: ExtractedRecommendation,
    *,
    asof: Optional[date] = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
    price_fetcher: Optional[PriceFetcher] = None,
    historical_result=None,
    default_lookback_days: int = 120,
    want_forecast: bool = True,
) -> AnalystReport:
    """Assemble the full report. Requires at least ticker+rating+target on ``rec``."""
    if not rec.is_gradeable:
        raise LiveGradeError("Need at least a ticker, a rating, and a target price to build a report "
                             f"(missing: {', '.join(rec.missing_fields())}).")
    asof = asof or date.today()
    call_date = rec.publication_date
    assumed = call_date is None
    if assumed:
        call_date = asof - timedelta(days=default_lookback_days)
    if call_date >= asof:
        call_date = asof - timedelta(days=default_lookback_days)
        assumed = True

    report = AnalystReport(rec=rec, call_date=call_date, call_date_assumed=assumed,
                           benchmark_symbol=benchmark_symbol)

    # --- live scorecard (real prices, open-ended "so far") ---
    try:
        res = grade_live_prediction(
            ticker=rec.ticker, rating=Rating(rec.rating), target_price=float(rec.target_price),
            call_date=call_date, benchmark_symbol=benchmark_symbol, asof=asof,
            analyst_name=rec.analyst or "Pasted recommendation", firm=rec.firm or "—",
            fetcher=price_fetcher,
        )
        r = res.resolution
        report.live = LiveScorecard(
            current_price=r.actual_price, call_price=r.call_price, return_since_pub=r.stock_return,
            benchmark_return=r.benchmark_return, alpha=res.call_score.beat,
            distance_to_target=(float(rec.target_price) - r.actual_price) / r.actual_price,
            days_since=(res.graded_through - call_date).days, status=res.status,
            provisional=res.provisional, graded_through=res.graded_through,
            direction_pass=res.call_score.direction_pass, verdict=res.verdict,
        )
    except LiveGradeError as e:
        report.live_error = str(e)
    except Exception as e:  # noqa: BLE001
        report.live_error = f"Live grade unavailable: {e}"

    # --- price path for the chart (one light fetch; reuses the same injectable fetcher) ---
    try:
        frame = (price_fetcher or YFinanceFetcher()).fetch([rec.ticker, benchmark_symbol], call_date, asof)
        tk = frame[rec.ticker].dropna()
        bm = frame[benchmark_symbol].reindex(tk.index).ffill()
        if len(tk) >= 2:
            report.chart = ReportChart(
                dates=[d.date() if hasattr(d, "date") else d for d in tk.index],
                ticker=[float(x) for x in tk.values], benchmark=[float(x) for x in bm.values],
                call_price=float(tk.iloc[0]), target=float(rec.target_price),
            )
    except Exception:  # noqa: BLE001 - the chart is a nice-to-have, never fatal
        pass

    # --- historical context + similar calls (offline; from the loaded back-test) ---
    report.historical = _historical_context(rec, historical_result)
    report.similar = _similar_calls(rec, historical_result)

    # --- forward outlook (reuses the calibrated forecast engine) ---
    if want_forecast:
        report.forward, report.forward_error = _forward_outlook(
            rec, call_date, asof, benchmark_symbol, price_fetcher)

    report.summary = _summary(rec, report.live, report.historical, report.forward)
    return report
