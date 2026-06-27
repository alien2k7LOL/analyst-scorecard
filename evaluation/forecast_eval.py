"""Audit the forecaster — but with the RIGHT tool for a probability model.

A confusion matrix needs a hard class; the forecaster outputs a probability, so the honest primary
metrics are CALIBRATION (does 70% mean 70%? → ECE) and DISCRIMINATION (AUC). We report those, and
ALSO provide the thing the user asked for — a hit/miss confusion matrix — by THRESHOLDING the
probability (predict "hit" when p ≥ threshold). That throws away the probability information, so it's
shown as a complementary view, not the headline.

Note on 'direction': unlike the extractor, the forecaster does NOT predict a direction — the user
supplies UP/DOWN as an input. So there is no direction confusion matrix to build here; the analogous
2-way question is hit vs miss, which is what the matrix below shows.

    python -m evaluation.forecast_eval
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyst_scorecard.forecast.backtest import (
    ForecastBacktest,
    ForecastGenConfig,
    generate_predictions,
    resolve_outcome,
)
from analyst_scorecard.forecast.calibration import LogisticCalibrator, metrics
from analyst_scorecard.forecast.features import build_features
from analyst_scorecard.forecast.news import NewsFileProvider
from analyst_scorecard.forecast.prediction import PredictionKind
from analyst_scorecard.providers.historical_price_provider import HistoricalPriceFileProvider

from .extraction_eval import FIG_DIR, save_confusion_heatmap

SAMPLE = "data/sample_historical"
DEFAULT_GEN = ForecastGenConfig(stride_days=21, horizons=(63, 126), up_offsets=(0.06, 0.12),
                                down_offsets=(0.06, 0.12), kinds=(PredictionKind.TERMINAL,),
                                terminal_bands=(0.05,))


def collect_holdout(data_dir: str = SAMPLE, gen: ForecastGenConfig = DEFAULT_GEN, *, with_news: bool = True):
    """Held-out (y, p): fit the deployed feature set on the train span, predict on the test span."""
    price = HistoricalPriceFileProvider(data_dir)
    news = NewsFileProvider(data_dir) if with_news and (Path(data_dir) / "news.csv").exists() else None
    bt = ForecastBacktest(price, news, gen=gen)
    result = bt.run()  # gives us the honestly-selected deployed feature set + headline metrics

    rows, ys = [], []
    for pred in generate_predictions(price, gen):
        try:
            row = build_features(price, pred, news, gen.lookback_days)
            outcome = resolve_outcome(price, pred)
        except (KeyError, ValueError):
            continue
        if outcome is None:
            continue
        rows.append(row)
        ys.append(1.0 if outcome.hit else 0.0)

    as_ofs = sorted({r.as_of for r in rows})
    split = as_ofs[min(int(len(as_ofs) * 0.6), len(as_ofs) - 1)]
    train = [(r, y) for r, y in zip(rows, ys) if r.deadline <= split]
    test = [(r, y) for r, y in zip(rows, ys) if r.as_of > split]
    cal = LogisticCalibrator(result.deployed_feature_set).fit([r for r, _ in train], [y for _, y in train])
    y = np.array([yy for _, yy in test], dtype=float)
    p = cal.predict([r for r, _ in test])
    return y, p, result


def hit_miss_confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> pd.DataFrame:
    pred_hit = p >= threshold
    act_hit = y >= 0.5
    tp = int(np.sum(pred_hit & act_hit))
    fp = int(np.sum(pred_hit & ~act_hit))
    fn = int(np.sum(~pred_hit & act_hit))
    tn = int(np.sum(~pred_hit & ~act_hit))
    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      index=["actual: MISS", "actual: HIT"], columns=["pred: MISS", "pred: HIT"])
    cm.index.name = f"threshold = {threshold:.2f}"
    return cm


def binary_metrics(cm: pd.DataFrame) -> dict:
    tn, fp = int(cm.iloc[0, 0]), int(cm.iloc[0, 1])
    fn, tp = int(cm.iloc[1, 0]), int(cm.iloc[1, 1])
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": (tp + tn) / total if total else 0.0, "precision": precision,
            "recall": recall, "f1": f1, "n": total}


def main() -> int:
    y, p, result = collect_holdout()
    base = float(y.mean())
    out = []
    out.append(f"# Forecast evaluation — held-out test span (n={len(y)}), deployed = {result.selected_name}\n")
    out.append("> **Read this first — what the numbers are measured on.** This eval runs on the bundled "
               "`data/sample_historical` set, whose prices are **synthetic geometric Brownian motion** — "
               "the *same* process the closed-form model assumes. So these are an **in-distribution upper "
               "bound** (no fat tails, vol clustering, or regime shifts, which is exactly what degrades "
               "calibration on real markets). The sample's news sentiment is also **engineered to lean "
               "toward each stock's next move**, so the 'news helps' result and the `full+` model "
               "selection are a *plumbing demonstration*, not evidence news helps on real data. Treat the "
               "price-only metrics as the conservative read, and re-run on a real `prices.csv` before "
               "quoting these to anyone.\n")
    out.append("## Primary metrics for a PROBABILITY model (the right tool)\n```")
    m = result.selected_metrics
    out.append(f"  calibration ECE : {m['ece']:.3f}   (does 70% mean 70%? lower is better)")
    out.append(f"  discrimination AUC: {m['auc']:.3f}  (ranks hits above misses; 0.5 = coin flip)")
    out.append(f"  Brier-skill      : {m['bss']:+.3f}  (vs always predicting the base rate)")
    out.append(f"  base hit-rate    : {base:.3f}\n```\n")

    out.append("## Thresholded hit/miss confusion matrix (complementary view)\n")
    figs = {}
    for thr in (0.50, round(base, 2)):
        cm = hit_miss_confusion(y, p, thr)
        bm = binary_metrics(cm)
        out.append(f"### threshold = {thr:.2f}\n```\n" + cm.to_string() + "\n```")
        out.append(f"accuracy {bm['accuracy']:.3f} · precision {bm['precision']:.3f} · "
                   f"recall {bm['recall']:.3f} · F1 {bm['f1']:.3f}\n")
        figs[thr] = save_confusion_heatmap(cm, FIG_DIR / f"forecast_hitmiss_thr{int(thr*100)}.png",
                                           f"Forecast hit/miss (threshold {thr:.2f})")
    out.append("Why two metrics: at a HIGH threshold the model rarely predicts 'hit', so accuracy can "
               "look high simply by predicting the majority class. That's why **calibration + AUC are "
               "the headline** for a probability forecaster, and the confusion matrix is the secondary, "
               "threshold-dependent view.\n")
    out.append("## Figures\n" + "\n".join(f"- threshold {k:.2f}: {v}" for k, v in figs.items()))
    report = "\n".join(out)
    (Path(__file__).resolve().parent / "FORECAST_REPORT.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
