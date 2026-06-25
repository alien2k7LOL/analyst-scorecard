"""CLI entry point: run the whole synthetic simulation end to end, print the leaderboard.

    python -m analyst_scorecard.cli                 # full run, streams verdicts + leaderboard
    python -m analyst_scorecard.cli --quiet         # leaderboard only
    python -m analyst_scorecard.cli --max-events 20 # cap the streamed verdict lines
    python -m analyst_scorecard.cli --seed 123      # a different reproducible price world
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .config import DEFAULT_CONFIG, ScorecardConfig
from .orchestrator import SimulationResult, TimeLoopOrchestrator
from .providers.call_provider import FixtureCallProvider
from .providers.price_provider import SyntheticPriceDataProvider
from .schemas import Leaderboard


def _fmt_pct(x: Optional[float], width: int = 8) -> str:
    return (f"{x * 100:+.1f}%" if x is not None else "—").rjust(width)


def _fmt_acc(x: Optional[float]) -> str:
    return (f"{x:.3f}" if x is not None else "—").rjust(8)


def render_leaderboard(lb: Leaderboard, verdicts: Optional[dict[str, str]] = None) -> str:
    lines = []
    header = f"{'#':>2}  {'Analyst':<22}{'Firm':<28}{'Beat-Mkt':>9}{'Dir Hit':>9}{'Accuracy':>9}{'Calls':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, s in enumerate(lb.rows, start=1):
        lines.append(
            f"{i:>2}  {s.analyst_name:<22}{s.firm:<28}"
            f"{_fmt_pct(s.beat_market, 9)}{s.direction_hit_rate * 100:>8.0f}%"
            f"{_fmt_acc(s.mean_accuracy)}{s.n_calls:>6}"
        )
        if verdicts and s.analyst_id in verdicts:
            lines.append(f"     ↳ {verdicts[s.analyst_id]}")
    return "\n".join(lines)


def run_simulation(config: ScorecardConfig = DEFAULT_CONFIG, *, quiet: bool = False,
                   max_events: Optional[int] = None, stream=None) -> SimulationResult:
    # Resolve the stream at call time (NOT as a default) so it honors a redirected sys.stdout.
    if stream is None:
        stream = sys.stdout
    provider = SyntheticPriceDataProvider(config)
    calls = FixtureCallProvider().get_calls()

    printed = {"n": 0}

    def on_event(ev):
        if quiet:
            return
        if max_events is not None and printed["n"] >= max_events:
            return
        print("  " + ev.verdict_line, file=stream)
        printed["n"] += 1

    if not quiet:
        print(f"Analyst Scorecard — synthetic simulation (seed={config.seed})", file=stream)
        print(f"Resolving {len(calls)} calls as their deadlines arrive in synthetic time...\n", file=stream)

    orchestrator = TimeLoopOrchestrator(calls, provider, config)
    result = orchestrator.run(on_event=on_event)

    if not quiet and max_events is not None and len(result.events) > max_events:
        print(f"  ... ({len(result.events) - max_events} more resolutions)", file=stream)

    print("", file=stream)
    if result.clock_start and result.clock_end:
        print(f"Synthetic time advanced {result.clock_start} → {result.clock_end}.", file=stream)
    print(f"\n=== LEADERBOARD (ranked by Beat-the-Market — the headline) ===\n", file=stream)
    print(render_leaderboard(result.leaderboard), file=stream)
    print("", file=stream)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Analyst Scorecard synthetic simulation.")
    p.add_argument("--seed", type=int, default=None, help="override the price-world seed")
    p.add_argument("--quiet", action="store_true", help="leaderboard only (no streamed verdicts)")
    p.add_argument("--max-events", type=int, default=25, help="cap streamed verdict lines (default 25)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG if args.seed is None else DEFAULT_CONFIG.with_overrides(seed=args.seed)
    run_simulation(config, quiet=args.quiet, max_events=args.max_events)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
