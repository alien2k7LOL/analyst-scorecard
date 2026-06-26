"""CLI: run the forecast calibration backtest and print the calibration report.

This is the "be a past user" harness: it manufactures a large grid of BLIND predictions across
history (each made using only data up to its as_of date), grades them against what actually
happened, and reports how accurate — really, how *calibrated* and *discriminating* — the model is.

    python -m analyst_scorecard.forecast.cli                          # touch, with news (sample)
    python -m analyst_scorecard.forecast.cli --kind terminal --band 0.04
    python -m analyst_scorecard.forecast.cli --kind terminal --no-news        # ablate news
    python -m analyst_scorecard.forecast.cli --kind terminal --stride 12 \
        --horizons 42,63,126,189 --offsets 0.06,0.12,0.18                     # train harder
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .backtest import ForecastGenConfig, run_forecast_backtest
from .prediction import PredictionKind

_ORDER = ["base_rate", "raw", "recalibrated", "+momentum", "+regime", "+news", "full", "full+"]


def render_report(result, *, kind: PredictionKind = PredictionKind.TOUCH, band: Optional[float] = None) -> str:
    noun = "terminal-hit" if kind == PredictionKind.TERMINAL else "touch"
    band_txt = f" (±{int(band * 100)}% band)" if (kind == PredictionKind.TERMINAL and band) else ""
    rows = []
    rows.append(f"Analyst Scorecard — FORECAST calibration backtest  [{kind.value}{band_txt}]")
    rows.append(
        f"  {result.n_predictions} predictions  |  train {result.n_train} / test {result.n_test}  "
        f"|  split {result.split_date}  |  test {noun}-rate {result.test_base_rate:.3f}"
    )
    rows.append(f"  News in model: {'yes (point-in-time)' if result.has_news else 'no'}   "
                f"|  deployed (validation-selected) model: {result.selected_name}")
    rows.append("")
    rows.append(f"  {'model':<14}{'Brier':>9}{'LogLoss':>10}{'ECE':>8}{'AUC':>8}{'BSS':>8}")
    rows.append(f"  {'':<14}{'(low)':>9}{'(low)':>10}{'(low)':>8}{'(high)':>8}{'(high)':>8}")
    rows.append("  " + "-" * 57)
    for name in _ORDER:
        if name not in result.metrics:
            continue
        m = result.metrics[name]
        marker = "  <- deployed" if name == result.selected_name else ""
        rows.append(f"  {name:<14}{m['brier']:>9.4f}{m['log_loss']:>10.4f}{m['ece']:>8.4f}"
                    f"{m.get('auc', float('nan')):>8.3f}{m.get('bss', float('nan')):>+8.3f}{marker}")
    rows.append("")
    sel = result.selected_metrics or result.metrics.get(result.selected_name, {})
    rows.append("  How to read this (the honest answer to \"how accurate is it?\"):")
    rows.append("  - This is a PROBABILITY forecaster, so \"% accuracy\" is the wrong yardstick. A")
    rows.append("    calibrated model that says 60% SHOULD be wrong 40% of the time — that's correct.")
    rows.append("  - Judge it by CALIBRATION (ECE: does 70% mean 70%?) and DISCRIMINATION (AUC: does")
    rows.append("    it rank likely outcomes above unlikely ones? 0.5 = coin flip, 1.0 = perfect).")
    rows.append(f"  - Deployed model ({result.selected_name}): ECE {sel.get('ece', float('nan')):.3f}, "
                f"AUC {sel.get('auc', float('nan')):.3f}, Brier-skill {sel.get('bss', float('nan')):+.3f} vs base rate.")
    rows.append("")
    if result.has_news:
        verdict = "ADDS held-out value" if result.news_helps else "does NOT beat price alone here"
        rows.append(f"  News verdict: point-in-time news {verdict}.")
        rows.append("")
    rows.append("  Reliability (deployed model, held-out test) — predicted vs actual hit rate:")
    for b in result.reliability:
        bar = "#" * int(round(b.mean_actual * 20))
        rows.append(f"    [{b.lo:.1f},{b.hi:.1f})  n={b.n:<5} pred={b.mean_pred:.3f}  actual={b.mean_actual:.3f}  {bar}")
    return "\n".join(rows)


def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the forecast calibration backtest.")
    p.add_argument("--data-dir", default="data/sample_historical", help="folder with prices.csv [+ news.csv]")
    p.add_argument("--kind", choices=["touch", "terminal"], default="touch")
    p.add_argument("--band", type=float, default=0.04, help="terminal tolerance band (e.g. 0.04 = ±4%)")
    p.add_argument("--stride", type=int, default=21, help="trading days between as_of dates (smaller = denser)")
    p.add_argument("--horizons", type=str, default="63,126", help="comma list of trading-day horizons")
    p.add_argument("--offsets", type=str, default="0.08,0.15", help="comma list of target offsets from spot")
    p.add_argument("--no-news", action="store_true", help="ablate news (price-only model)")
    args = p.parse_args(argv)

    kind = PredictionKind(args.kind)
    offs = _floats(args.offsets)
    gen = ForecastGenConfig(
        stride_days=args.stride, horizons=_ints(args.horizons),
        up_offsets=offs, down_offsets=offs,
        kinds=(kind,), terminal_bands=(args.band,),
    )
    result = run_forecast_backtest(args.data_dir, with_news=not args.no_news, gen=gen)
    band = args.band if kind == PredictionKind.TERMINAL else None
    print(render_report(result, kind=kind, band=band), file=sys.stdout)
    print("", file=sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
