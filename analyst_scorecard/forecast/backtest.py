"""The calibration backtest — 'use older data to refine the prediction and grading accuracy'.

It manufactures a large, honest evaluation set from history: for many (ticker, as_of, horizon,
target) combinations it computes the model's touch probability using ONLY data up to as_of, then
resolves what actually happened over (as_of, deadline]. It splits by time (train fully resolved
BEFORE any test prediction is made — no leak), fits the recalibration layer on train, and reports
calibration (Brier / log-loss / ECE) on the held-out test set for:

    raw            — the closed-form GBM touch probability, uncalibrated
    recalibrated   — GBM logit passed through a fitted logistic (Platt) layer
    +momentum      — recalibrated + price trend / distance features
    +news          — recalibrated + point-in-time news features
    full           — everything

The gap between ``raw`` and ``recalibrated`` shows how mis-calibrated the raw model is; the gap
between ``recalibrated`` and ``+news`` shows whether news actually earns its place on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..providers.historical_price_provider import HistoricalPriceFileProvider
from ..providers.price_provider import PriceDataProvider, _ts
from ..schemas import Direction
from .calibration import LogisticCalibrator, metrics, reliability_bins
from .features import ALL_FEATURE_NAMES, PRICE_FEATURE_NAMES, FeatureRow, build_features
from .news import NEWS_FEATURE_NAMES, NewsFileProvider, NewsProvider, NoNewsProvider
from .prediction import Prediction, PredictionOutcome

FEATURE_SETS: dict[str, Optional[list[str]]] = {
    "raw": None,  # special-cased: use the GBM probability directly
    "recalibrated": ["gbm_logit"],
    "+momentum": PRICE_FEATURE_NAMES,
    "+news": ["gbm_logit"] + NEWS_FEATURE_NAMES,
    "full": ALL_FEATURE_NAMES,
}


# --------------------------------------------------------------------------------------
# Resolution — look-ahead-safe touch over (as_of, deadline]
# --------------------------------------------------------------------------------------


def resolve_outcome(price_provider: PriceDataProvider, pred: Prediction) -> Optional[PredictionOutcome]:
    """Did the price touch the target over (as_of, deadline]? Uses ONLY prices in that window."""
    series = price_provider.price_series(pred.ticker)
    asof, deadline = _ts(pred.as_of), _ts(pred.deadline)
    past = series.loc[:asof]
    if len(past) < 1:
        return None
    asof_actual, asof_price = past.index[-1], float(past.iloc[-1])
    forward = series.loc[asof_actual:deadline]
    after = forward.iloc[1:]  # strictly AFTER the as_of day — a touch must occur post-prediction
    if len(after) < 1:
        return None

    vals = after.to_numpy(dtype=float)
    if pred.direction == Direction.UP:
        extreme = float(vals.max())
        hit_mask = vals >= pred.target_price
    else:
        extreme = float(vals.min())
        hit_mask = vals <= pred.target_price
    hit = bool(hit_mask.any())
    hit_date = after.index[np.argmax(hit_mask)].date() if hit else None

    return PredictionOutcome(
        prediction_id=pred.prediction_id,
        hit=hit,
        hit_date=hit_date,
        as_of_price=asof_price,
        extreme_price=extreme,
        n_observations=int(len(forward)),
    )


# --------------------------------------------------------------------------------------
# Generation — a grid of predictions across history
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastGenConfig:
    stride_days: int = 21               # space as_of dates ~monthly
    horizons: tuple[int, ...] = (63, 126)   # trading-day horizons (~3mo, ~6mo)
    up_offsets: tuple[float, ...] = (0.08, 0.15)
    down_offsets: tuple[float, ...] = (0.08, 0.15)
    lookback_days: int = 252
    min_history: int = 126              # require this many observations before as_of


def generate_predictions(
    price_provider: PriceDataProvider, gen: ForecastGenConfig, tickers: Optional[Sequence[str]] = None
) -> list[Prediction]:
    preds: list[Prediction] = []
    tickers = list(tickers) if tickers is not None else price_provider.tickers()
    horizon_max = max(gen.horizons)
    for tk in tickers:
        idx = price_provider.price_series(tk).index
        vals = price_provider.price_series(tk).to_numpy(dtype=float)
        last_start = len(idx) - horizon_max - 1
        for i in range(gen.min_history, last_start, gen.stride_days):
            as_of, s0 = idx[i], float(vals[i])
            for h in gen.horizons:
                j = i + h
                if j >= len(idx):
                    continue
                deadline = idx[j]
                for off in gen.up_offsets:
                    preds.append(Prediction(
                        prediction_id=f"{tk}|{as_of.date()}|{h}|U{int(off*100)}",
                        ticker=tk, as_of=as_of.date(), target_price=round(s0 * (1 + off), 4),
                        deadline=deadline.date(), direction=Direction.UP, made_by="backtest"))
                for off in gen.down_offsets:
                    preds.append(Prediction(
                        prediction_id=f"{tk}|{as_of.date()}|{h}|D{int(off*100)}",
                        ticker=tk, as_of=as_of.date(), target_price=round(s0 * (1 - off), 4),
                        deadline=deadline.date(), direction=Direction.DOWN, made_by="backtest"))
    return preds


# --------------------------------------------------------------------------------------
# The backtest
# --------------------------------------------------------------------------------------


@dataclass
class ForecastBacktestResult:
    n_predictions: int
    n_train: int
    n_test: int
    split_date: Optional[date]
    test_base_rate: float
    metrics: dict[str, dict]            # feature-set name -> {brier, log_loss, ece, n}
    reliability: list                   # reliability bins for the 'full' model on test
    news_helps: bool
    has_news: bool
    deployed_feature_set: list[str]
    _deployed: Optional[LogisticCalibrator] = field(default=None, repr=False)

    def predict_one(self, row: FeatureRow) -> float:
        """Calibrated touch probability for a single new prediction (uses the deployed model)."""
        if self._deployed is None:
            return float(row.gbm_p)
        return float(self._deployed.predict([row])[0])


class ForecastBacktest:
    def __init__(
        self,
        price_provider: PriceDataProvider,
        news_provider: Optional[NewsProvider] = None,
        gen: ForecastGenConfig = ForecastGenConfig(),
        split_frac: float = 0.6,
        l2: float = 1.0,
        tickers: Optional[Sequence[str]] = None,
    ):
        self.price = price_provider
        self.news = news_provider or NoNewsProvider()
        self.has_news = not isinstance(self.news, NoNewsProvider)
        self.gen = gen
        self.split_frac = split_frac
        self.l2 = l2
        self.tickers = tickers

    def _collect(self) -> tuple[list[FeatureRow], list[float]]:
        rows: list[FeatureRow] = []
        ys: list[float] = []
        for p in generate_predictions(self.price, self.gen, self.tickers):
            try:
                row = build_features(self.price, p, self.news, self.gen.lookback_days)
                outcome = resolve_outcome(self.price, p)
            except (KeyError, ValueError):
                continue
            if outcome is None:
                continue
            rows.append(row)
            ys.append(1.0 if outcome.hit else 0.0)
        return rows, ys

    def run(self) -> ForecastBacktestResult:
        rows, ys = self._collect()
        if len(rows) < 50:
            raise ValueError(f"too few resolvable predictions ({len(rows)}) — widen the generation grid")

        as_ofs = sorted({r.as_of for r in rows})
        split_date = as_ofs[min(int(len(as_ofs) * self.split_frac), len(as_ofs) - 1)]
        train = [(r, y) for r, y in zip(rows, ys) if r.deadline <= split_date]
        test = [(r, y) for r, y in zip(rows, ys) if r.as_of > split_date]
        if len(train) < 30 or len(test) < 30:
            raise ValueError(f"train/test too small (train={len(train)}, test={len(test)})")

        train_rows, ytr = [r for r, _ in train], [y for _, y in train]
        test_rows, yte = [r for r, _ in test], [y for _, y in test]
        base_rate = float(np.mean(yte))

        results: dict[str, dict] = {"base_rate": metrics(yte, [base_rate] * len(yte))}
        results["raw"] = metrics(yte, [r.gbm_p for r in test_rows])
        reliability: list = []
        for name, fs in FEATURE_SETS.items():
            if fs is None:
                continue
            cal = LogisticCalibrator(fs, l2=self.l2).fit(train_rows, ytr)
            pte = cal.predict(test_rows)
            results[name] = metrics(yte, pte)
            if name == "full":
                reliability = reliability_bins(yte, pte)

        news_helps = self.has_news and (results["+news"]["log_loss"] < results["recalibrated"]["log_loss"])

        # Deploy: refit the best honest feature set on ALL history for live use.
        deployed_fs = ALL_FEATURE_NAMES if self.has_news else PRICE_FEATURE_NAMES
        deployed = LogisticCalibrator(deployed_fs, l2=self.l2).fit(rows, ys)

        return ForecastBacktestResult(
            n_predictions=len(rows), n_train=len(train_rows), n_test=len(test_rows),
            split_date=split_date, test_base_rate=base_rate, metrics=results,
            reliability=reliability, news_helps=news_helps, has_news=self.has_news,
            deployed_feature_set=deployed_fs, _deployed=deployed,
        )


def run_forecast_backtest(
    data_dir: Path | str, *, with_news: bool = True, gen: ForecastGenConfig = ForecastGenConfig(), **kw
) -> ForecastBacktestResult:
    """Build providers from a back-test data folder (prices.csv [+ news.csv]) and run."""
    price = HistoricalPriceFileProvider(data_dir)
    news: Optional[NewsProvider] = None
    if with_news and (Path(data_dir) / "news.csv").exists():
        news = NewsFileProvider(data_dir)
    return ForecastBacktest(price, news, gen=gen, **kw).run()
