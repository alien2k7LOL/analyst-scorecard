"""CLI: run the forecast calibration backtest and print the calibration report.

    python -m analyst_scorecard.forecast.cli                     # the shipped sample (with news)
    python -m analyst_scorecard.forecast.cli --no-news           # ablate news to see its effect
    python -m analyst_scorecard.forecast.cli --data-dir /path --stride 15
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .backtest import ForecastGenConfig, run_forecast_backtest


def render_report(result) -> str:
    rows = []
    rows.append("Analyst Scorecard — FORECAST calibration backtest")
    rows.append(
        f"  {result.n_predictions} predictions  |  train {result.n_train} / test {result.n_test}  "
        f"|  split {result.split_date}  |  test touch-rate {result.test_base_rate:.3f}"
    )
    rows.append(f"  News in model: {'yes (point-in-time)' if result.has_news else 'no'}")
    rows.append("")
    rows.append(f"  {'model':<14}{'Brier':>9}{'LogLoss':>10}{'ECE':>8}{'AUC':>8}{'BSS':>8}")
    rows.append(f"  {'':<14}{'(low)':>9}{'(low)':>10}{'(low)':>8}{'(high)':>8}{'(high)':>8}")
    rows.append("  " + "-" * 57)
    order = ["base_rate", "raw", "recalibrated", "+momentum", "+news", "full"]
    for name in order:
        if name not in result.metrics:
            continue
        m = result.metrics[name]
        auc = m.get("auc", float("nan"))
        bss = m.get("bss", float("nan"))
        rows.append(f"  {name:<14}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['ece']:>8.4f}{auc:>8.3f}{bss:>+8.3f}")
    rows.append("")
    full = result.metrics.get("full", {})
    rows.append("  How to read this (the honest answer to \"how accurate is it?\"):")
    rows.append("  - This is a PROBABILITY forecaster, so \"% accuracy\" is the wrong yardstick. A")
    rows.append("    calibrated model that says 60% SHOULD be wrong 40% of the time — that's correct.")
    rows.append("  - Judge it by CALIBRATION (ECE: does 70% mean 70%?) and DISCRIMINATION (AUC: does")
    rows.append("    it rank likely touches above unlikely ones? 0.5 = coin flip, 1.0 = perfect).")
    rows.append(f"  - Best model here: ECE {full.get('ece', float('nan')):.3f} (well-calibrated), "
                f"AUC {full.get('auc', float('nan')):.3f}, Brier-skill {full.get('bss', float('nan')):+.3f} vs base rate.")
    rows.append("")
    if result.has_news:
        verdict = "ADDS held-out value" if result.news_helps else "does NOT beat price alone here"
        rows.append(f"  News verdict: point-in-time news {verdict}.")
    rows.append("")
    rows.append("  Reliability (full model, held-out test) — predicted vs actual touch rate:")
    for b in result.reliability:
        bar = "#" * int(round(b.mean_actual * 20))
        rows.append(f"    [{b.lo:.1f},{b.hi:.1f})  n={b.n:<5} pred={b.mean_pred:.3f}  actual={b.mean_actual:.3f}  {bar}")
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the forecast calibration backtest.")
    p.add_argument("--data-dir", default="data/sample_historical", help="folder with prices.csv [+ news.csv]")
    p.add_argument("--stride", type=int, default=21, help="trading days between as_of dates (smaller = denser)")
    p.add_argument("--no-news", action="store_true", help="ablate news (price-only model)")
    args = p.parse_args(argv)

    gen = ForecastGenConfig(stride_days=args.stride)
    result = run_forecast_backtest(args.data_dir, with_news=not args.no_news, gen=gen)
    print(render_report(result), file=sys.stdout)
    print("", file=sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
