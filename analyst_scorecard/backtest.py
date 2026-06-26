"""Historical back-test runner — feeds real past calls through the UNCHANGED engine.

`HistoricalBacktest.run()` walks forward through historical time (calls ordered by the date
their outcome becomes known) and, for each call, resolves it AT ITS ORIGINAL HORIZON using the
existing look-ahead-safe `resolve_call_with_provider` and `score_call`, then aggregates with the
existing `aggregate_analyst` / `Leaderboard.from_scores`. The only thing this module adds is
skip-handling: a call with no look-ahead-safe outcome (e.g. its ticker delisted before the
horizon) is classified and logged, never silently scored.

Look-ahead safety: resolution goes only through `resolve_call_with_provider`, which hands the
resolver a `PriceWindow` ending exactly at the resolution date — the future is unreachable. The
runner pre-classifies resolvability from data presence (not exception text) and otherwise calls
the unchanged engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from .aggregation import aggregate_analyst
from .config import DEFAULT_CONFIG, ScorecardConfig
from .providers.historical_call_provider import HistoricalCallFileProvider
from .providers.historical_price_provider import HistoricalPriceFileProvider
from .providers.price_provider import PriceDataProvider
from .resolution import resolve_call_with_provider
from .schemas import AnalystScore, Call, CallScore, Leaderboard
from .scoring import score_call

SAMPLE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_historical"


@dataclass(frozen=True)
class SkippedCall:
    call: Call
    reason: str
    detail: str = ""


@dataclass
class BacktestResult:
    leaderboard: Leaderboard
    analyst_scores: dict[str, AnalystScore]
    resolved_scores: list[CallScore]
    skipped: list[SkippedCall]
    ingest_issues: list[dict]
    span_start: Optional[date]
    span_end: Optional[date]
    n_ingested: int
    n_resolved: int
    n_skipped: int
    n_ingest_dropped: int
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    ingest_reason_counts: dict[str, int] = field(default_factory=dict)
    label: str = ""
    is_sample: bool = False


def _has(provider: PriceDataProvider, symbol: str, on) -> bool:
    """Data-presence check that works for any provider (prefers a native has_data)."""
    has = getattr(provider, "has_data", None)
    if has is not None:
        return has(symbol, on)
    try:
        provider.price_on(symbol, on)
        return True
    except Exception:
        return False


class HistoricalBacktest:
    def __init__(
        self,
        price_provider: PriceDataProvider,
        call_provider,
        config: ScorecardConfig = DEFAULT_CONFIG,
    ):
        self.price_provider = price_provider
        self.call_provider = call_provider
        self.config = config

    def _unresolvable_reason(self, call: Call) -> Optional[tuple[str, str]]:
        bench = self.price_provider.benchmark_symbol
        # Benchmark endpoints (defensive — they should always exist by construction).
        if not _has(self.price_provider, bench, call.call_date):
            return ("NO_BENCHMARK_AT_CALL", f"{bench} @ {call.call_date}")
        if not _has(self.price_provider, bench, call.resolution_date):
            return ("NO_BENCHMARK_AT_RESOLUTION", f"{bench} @ {call.resolution_date}")
        # Ticker endpoints.
        if not _has(self.price_provider, call.ticker, call.call_date):
            return ("NO_ENTRY_PRICE", f"{call.ticker} @ {call.call_date}")
        if not _has(self.price_provider, call.ticker, call.resolution_date):
            # The headline real-world case: the stock delisted/halted before its horizon.
            return ("DELISTED_OR_HALTED", f"{call.ticker} has no price @ {call.resolution_date}")
        return None

    def run(self) -> BacktestResult:
        calls = self.call_provider.get_calls()
        ingest_issues = list(getattr(self.call_provider, "ingest_issues", []))

        resolved_scores: list[CallScore] = []
        skipped: list[SkippedCall] = []

        # Walk forward: process calls in the order their outcomes become known.
        for call in sorted(calls, key=lambda c: (c.resolution_date, c.call_id)):
            reason = self._unresolvable_reason(call)
            if reason is None:
                try:
                    resolution = resolve_call_with_provider(call, self.price_provider)  # THE ENGINE
                    resolved_scores.append(score_call(call, resolution, self.config))   # THE ENGINE
                    continue
                except Exception as e:  # defense in depth; never crash the whole back-test
                    reason = ("RESOLVER_ERROR", str(e))
            skipped.append(SkippedCall(call=call, reason=reason[0], detail=reason[1]))

        # Aggregate over RESOLVED calls only (reusing the existing engine aggregation).
        meta: dict[str, tuple[str, str]] = {}
        for c in calls:
            meta.setdefault(c.analyst_id, (c.analyst_name, c.firm))
        by_analyst: dict[str, list[CallScore]] = {}
        for cs in resolved_scores:
            by_analyst.setdefault(cs.analyst_id, []).append(cs)
        analyst_scores = {
            aid: aggregate_analyst(css, analyst_name=meta[aid][0], firm=meta[aid][1])
            for aid, css in by_analyst.items()
        }
        leaderboard = Leaderboard.from_scores(list(analyst_scores.values()))

        days = self.price_provider.trading_days()
        skip_counts: dict[str, int] = {}
        for s in skipped:
            skip_counts[s.reason] = skip_counts.get(s.reason, 0) + 1
        ingest_counts: dict[str, int] = {}
        for iss in ingest_issues:
            ingest_counts[iss["reason"]] = ingest_counts.get(iss["reason"], 0) + 1

        manifest = getattr(self.price_provider, "manifest", {}) or {}
        return BacktestResult(
            leaderboard=leaderboard,
            analyst_scores=analyst_scores,
            resolved_scores=resolved_scores,
            skipped=skipped,
            ingest_issues=ingest_issues,
            span_start=days[0].date() if len(days) else None,
            span_end=days[-1].date() if len(days) else None,
            n_ingested=len(calls),
            n_resolved=len(resolved_scores),
            n_skipped=len(skipped),
            n_ingest_dropped=len(ingest_issues),
            skip_reason_counts=skip_counts,
            ingest_reason_counts=ingest_counts,
            label=str(manifest.get("label", "")),
            is_sample=bool(manifest.get("is_sample", False)),
        )


def load_backtest(data_dir: Path | str = SAMPLE_DATA_DIR, config: ScorecardConfig = DEFAULT_CONFIG) -> HistoricalBacktest:
    """Wire the historical price + call providers for a data folder and return a runner."""
    prices = HistoricalPriceFileProvider(data_dir)
    calls = HistoricalCallFileProvider(data_dir, prices)
    return HistoricalBacktest(prices, calls, config)


def run_backtest(data_dir: Path | str = SAMPLE_DATA_DIR, config: ScorecardConfig = DEFAULT_CONFIG) -> BacktestResult:
    return load_backtest(data_dir, config).run()
