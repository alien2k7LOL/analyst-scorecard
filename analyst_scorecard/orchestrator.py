"""End-to-end orchestration: the autonomous time-loop agent.

The loop advances a SYNTHETIC CLOCK through trading time. A call's resolution rule and
deadline are fixed when the call is recorded (in the `Call` itself); the loop does nothing
with a call until its deadline arrives. When the clock reaches a call's `resolution_date`,
the call is resolved look-ahead-safely (at that instant "now" == resolution_date, so only
data up to the deadline is ever used), the analyst's running scores update, and a one-line
verdict is drafted. No human is in the loop.

Because resolution is event-driven on the record-time deadline, the simulation is itself a
proof of no look-ahead: a call can only be graded once time has actually reached its deadline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Optional

from .aggregation import aggregate_analyst
from .config import DEFAULT_CONFIG, ScorecardConfig
from .providers.price_provider import PriceDataProvider
from .resolution import resolve_call_with_provider
from .schemas import AnalystScore, Call, CallScore, Leaderboard
from .scoring import score_call


@dataclass(frozen=True)
class ResolutionEvent:
    """One call coming due in synthetic time, with the running standings right after."""

    clock: date                  # synthetic "now" == the call's resolution date
    call: Call
    call_score: CallScore
    analyst_snapshot: AnalystScore   # the analyst's standings AFTER this resolution
    verdict_line: str


@dataclass
class SimulationResult:
    events: list[ResolutionEvent] = field(default_factory=list)
    leaderboard: Optional[Leaderboard] = None
    final_scores: dict[str, AnalystScore] = field(default_factory=dict)
    clock_start: Optional[date] = None
    clock_end: Optional[date] = None


def _verdict_line(clock: date, call: Call, cs: CallScore, snap: AnalystScore) -> str:
    """A short, human verdict for a call coming due (e.g. the spec's example line)."""
    outcome = "HIT" if cs.direction_pass else "MISSED"
    bm = snap.beat_market
    bm_str = f"{bm * 100:+.1f}%" if bm is not None else "n/a"
    acc = f", accuracy {cs.accuracy:.2f}" if cs.accuracy is not None else ""
    return (
        f"[{clock}] {call.analyst_name} — {call.ticker} {call.rating.value} "
        f"(target ${call.target_price:.2f}) came due: {outcome} on direction{acc}. "
        f"Beat-market record now {bm_str}."
    )


class TimeLoopOrchestrator:
    """Drives the synthetic clock and resolves calls as their deadlines arrive."""

    def __init__(self, calls: Iterable[Call], provider: PriceDataProvider, config: ScorecardConfig = DEFAULT_CONFIG):
        self.calls = list(calls)
        self.provider = provider
        self.config = config
        # name/firm lookup so running snapshots carry identity
        self._meta: dict[str, tuple[str, str]] = {}
        for c in self.calls:
            self._meta.setdefault(c.analyst_id, (c.analyst_name, c.firm))

    def run(self, on_event: Optional[Callable[[ResolutionEvent], None]] = None) -> SimulationResult:
        # Process calls in the order their deadlines arrive (advance synthetic time forward).
        ordered = sorted(self.calls, key=lambda c: (c.resolution_date, c.call_id))
        accumulated: dict[str, list[CallScore]] = defaultdict(list)
        result = SimulationResult()

        for call in ordered:
            clock = call.resolution_date  # advance the synthetic clock to this deadline

            # Look-ahead-safe: the resolver only sees [call_date, resolution_date] == up to now.
            resolution = resolve_call_with_provider(call, self.provider)
            cs = score_call(call, resolution, self.config)

            accumulated[call.analyst_id].append(cs)
            name, firm = self._meta[call.analyst_id]
            snapshot = aggregate_analyst(accumulated[call.analyst_id], analyst_name=name, firm=firm)

            event = ResolutionEvent(
                clock=clock,
                call=call,
                call_score=cs,
                analyst_snapshot=snapshot,
                verdict_line=_verdict_line(clock, call, cs, snapshot),
            )
            result.events.append(event)
            if on_event is not None:
                on_event(event)

        # Final standings from the fully accumulated scores.
        final_scores = {
            aid: aggregate_analyst(css, analyst_name=self._meta[aid][0], firm=self._meta[aid][1])
            for aid, css in accumulated.items()
        }
        result.final_scores = final_scores
        result.leaderboard = Leaderboard.from_scores(list(final_scores.values()))
        if result.events:
            result.clock_start = result.events[0].clock
            result.clock_end = result.events[-1].clock
        return result
