"""CLI: run the historical back-test on local files and print the historical leaderboard.

    python -m analyst_scorecard.backtest_cli                      # the shipped sample
    python -m analyst_scorecard.backtest_cli --data-dir /path     # your own real files
    python -m analyst_scorecard.backtest_cli --show-skips         # list skipped/dropped calls
    python -m analyst_scorecard.backtest_cli --quiet              # leaderboard only
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .backtest import SAMPLE_DATA_DIR, BacktestResult, run_backtest
from .cli import render_leaderboard
from .config import DEFAULT_CONFIG
from .verdicts import default_verdict_generator


def render_report(result: BacktestResult, *, show_skips: bool = False, with_verdicts: bool = True) -> str:
    lines: list[str] = []
    tag = "SAMPLE data (synthetic, fictional)" if result.is_sample else "user-supplied data"
    lines.append(f"Analyst Scorecard — HISTORICAL back-test [{tag}]")
    if result.label:
        lines.append(f"  {result.label}")
    if result.span_start and result.span_end:
        lines.append(f"  Price span: {result.span_start} → {result.span_end}")
    lines.append(
        f"  Calls: {result.n_ingested} ingested, {result.n_resolved} resolved & scored, "
        f"{result.n_skipped} skipped, {result.n_ingest_dropped} dropped at ingest."
    )
    if result.skip_reason_counts:
        lines.append("  Skipped at resolution: " + ", ".join(
            f"{k}×{v}" for k, v in sorted(result.skip_reason_counts.items())))
    if result.ingest_reason_counts:
        lines.append("  Dropped at ingest:    " + ", ".join(
            f"{k}×{v}" for k, v in sorted(result.ingest_reason_counts.items())))

    verdicts = None
    if with_verdicts:
        gen = default_verdict_generator()
        verdicts = {s.analyst_id: gen.verdict(s) for s in result.leaderboard.rows}

    lines.append("")
    lines.append("=== HISTORICAL LEADERBOARD (ranked by Beat-the-Market — the headline) ===")
    lines.append("")
    lines.append(render_leaderboard(result.leaderboard, verdicts=verdicts))

    if show_skips and (result.skipped or result.ingest_issues):
        lines.append("")
        lines.append("--- Skipped at resolution (no look-ahead-safe outcome) ---")
        for s in result.skipped:
            lines.append(f"  {s.call.call_id:<16} {s.call.ticker:<6} {s.reason}: {s.detail}")
        lines.append("--- Dropped at ingest (not a closeable call) ---")
        for iss in result.ingest_issues:
            lines.append(f"  {iss['call_id']:<16} {iss.get('ticker',''):<6} {iss['reason']}: {iss.get('detail','')}")

    return "\n".join(lines)


def run(data_dir, *, show_skips: bool = False, quiet: bool = False, stream=None) -> BacktestResult:
    if stream is None:
        stream = sys.stdout
    result = run_backtest(data_dir, DEFAULT_CONFIG)
    print(render_report(result, show_skips=show_skips, with_verdicts=not quiet), file=stream)
    print("", file=stream)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Analyst Scorecard historical back-test.")
    p.add_argument("--data-dir", default=str(SAMPLE_DATA_DIR), help="folder with prices.csv/calls.csv/manifest.json")
    p.add_argument("--show-skips", action="store_true", help="list skipped and ingest-dropped calls")
    p.add_argument("--quiet", action="store_true", help="leaderboard only (no verdicts)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.data_dir, show_skips=args.show_skips, quiet=args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
