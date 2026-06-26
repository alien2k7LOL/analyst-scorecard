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
from ..providers.price_provider import PriceDataProvider
from ..schemas import Direction
from .calibration import LogisticCalibrator, log_loss, metrics, reliability_bins
from .interval import BarInterval, to_ts
from .features import (
    ALL_FEATURE_NAMES,
    ALL_PLUS_FEATURE_NAMES,
    EXT_PRICE_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    FeatureRow,
    build_features,
)
from .news import NEWS_FEATURE_NAMES, NewsFileProvider, NewsProvider, NoNewsProvider
from .prediction import Prediction, PredictionKind, PredictionOutcome

# Candidate recalibration feature sets, smallest to richest. ``raw`` is special-cased (the GBM
# probability passed straight through); every other set is fit as a logistic layer and the deployed
# model is chosen by held-out VALIDATION performance — never by the test span.
FEATURE_SETS: dict[str, Optional[list[str]]] = {
    "raw": None,
    "recalibrated": ["gbm_logit"],
    "+momentum": PRICE_FEATURE_NAMES,
    "+regime": PRICE_FEATURE_NAMES + EXT_PRICE_FEATURE_NAMES,
    "+news": ["gbm_logit"] + NEWS_FEATURE_NAMES,
    "full": ALL_FEATURE_NAMES,
    "full+": ALL_PLUS_FEATURE_NAMES,
}


# --------------------------------------------------------------------------------------
# Resolution — look-ahead-safe touch over (as_of, deadline]
# --------------------------------------------------------------------------------------


def _terminal_hit(terminal_price: float, pred: Prediction) -> bool:
    """Did the price END the window at/near the target, per the prediction's terminal rule?"""
    if pred.band_pct is not None:
        lo, hi = pred.target_price * (1 - pred.band_pct), pred.target_price * (1 + pred.band_pct)
        return bool(lo <= terminal_price <= hi)
    if pred.direction == Direction.UP:
        return bool(terminal_price >= pred.target_price)
    return bool(terminal_price <= pred.target_price)


def resolve_outcome(price_provider: PriceDataProvider, pred: Prediction) -> Optional[PredictionOutcome]:
    """Ground-truth a prediction using ONLY prices in (as_of, deadline].

    TOUCH    — hit if the path reaches the target at any point in the window.
    TERMINAL — hit if the close ON the deadline trading day satisfies the band / at-or-through rule.
               This looks only AT the deadline price (never past it), so it stays look-ahead-safe.
    """
    series = price_provider.price_series(pred.ticker)
    asof, deadline = to_ts(pred.as_of, pred.interval), to_ts(pred.deadline, pred.interval)
    past = series.loc[:asof]
    if len(past) < 1:
        return None
    asof_actual, asof_price = past.index[-1], float(past.iloc[-1])
    forward = series.loc[asof_actual:deadline]
    after = forward.iloc[1:]  # strictly AFTER the as_of day — resolution must occur post-prediction
    if len(after) < 1:
        return None

    vals = after.to_numpy(dtype=float)
    extreme = float(vals.max()) if pred.direction == Direction.UP else float(vals.min())
    # The deadline-day close (last trading day on/before the deadline) — what resolves a terminal call.
    terminal_price = float(after.iloc[-1])
    terminal_date = after.index[-1].date()

    if pred.kind == PredictionKind.TERMINAL:
        hit = _terminal_hit(terminal_price, pred)
        hit_date = terminal_date if hit else None
    else:
        hit_mask = vals >= pred.target_price if pred.direction == Direction.UP else vals <= pred.target_price
        hit = bool(hit_mask.any())
        hit_date = after.index[int(np.argmax(hit_mask))].date() if hit else None

    return PredictionOutcome(
        prediction_id=pred.prediction_id,
        hit=hit,
        hit_date=hit_date,
        as_of_price=asof_price,
        extreme_price=extreme,
        n_observations=int(len(forward)),
        terminal_price=terminal_price,
        terminal_date=terminal_date,
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
    # What kind(s) of prediction to manufacture. Default is touch-only (back-compat). The big
    # training run also generates TERMINAL predictions; ``terminal_bands`` lists the ± tolerance
    # bands to use (None = an "at or through the target on the deadline" terminal call).
    kinds: tuple[PredictionKind, ...] = (PredictionKind.TOUCH,)
    terminal_bands: tuple[Optional[float], ...] = (0.03,)
    interval: BarInterval = BarInterval.DAILY  # DAILY bars, or 30-min intraday bars


def _band_tag(kind: PredictionKind, band: Optional[float]) -> str:
    if kind == PredictionKind.TOUCH:
        return "TCH"
    return f"TRMb{int(band * 100)}" if band is not None else "TRMthru"


def generate_predictions(
    price_provider: PriceDataProvider, gen: ForecastGenConfig, tickers: Optional[Sequence[str]] = None
) -> list[Prediction]:
    preds: list[Prediction] = []
    tickers = list(tickers) if tickers is not None else price_provider.tickers()
    horizon_max = max(gen.horizons)
    # For each kind, the (band,) variants to emit. Touch ignores bands (one variant, band=None).
    kind_variants = [
        (k, b)
        for k in gen.kinds
        for b in (gen.terminal_bands if k == PredictionKind.TERMINAL else (None,))
    ]
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
                stamp = as_of.strftime("%Y%m%d-%H%M")  # unique per bar (handles intraday same-day)
                for kind, band in kind_variants:
                    tag = _band_tag(kind, band)
                    for off in gen.up_offsets:
                        preds.append(Prediction(
                            prediction_id=f"{tk}|{stamp}|{h}|U{int(off*100)}|{tag}",
                            ticker=tk, as_of=as_of, target_price=round(s0 * (1 + off), 4),
                            deadline=deadline, direction=Direction.UP,
                            kind=kind, band_pct=band, interval=gen.interval, made_by="backtest"))
                    for off in gen.down_offsets:
                        preds.append(Prediction(
                            prediction_id=f"{tk}|{stamp}|{h}|D{int(off*100)}|{tag}",
                            ticker=tk, as_of=as_of, target_price=round(s0 * (1 - off), 4),
                            deadline=deadline, direction=Direction.DOWN,
                            kind=kind, band_pct=band, interval=gen.interval, made_by="backtest"))
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
    metrics: dict[str, dict]            # feature-set name -> {brier, log_loss, ece, auc, bss, n}
    reliability: list                   # reliability bins for the DEPLOYED (selected) model on test
    news_helps: bool
    has_news: bool
    deployed_feature_set: list[str]
    selected_name: str = "recalibrated"  # which candidate set won on the validation span
    selected_metrics: dict = field(default_factory=dict)  # that set's held-out TEST metrics
    _deployed: Optional[LogisticCalibrator] = field(default=None, repr=False)

    def predict_one(self, row: FeatureRow) -> float:
        """Calibrated probability for a single new prediction (uses the deployed model)."""
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

    def _candidate_sets(self) -> dict[str, list[str]]:
        """Feature sets to evaluate/select among — news-based ones only when news is available."""
        cands: dict[str, list[str]] = {
            "recalibrated": ["gbm_logit"],
            "+momentum": PRICE_FEATURE_NAMES,
            "+regime": PRICE_FEATURE_NAMES + EXT_PRICE_FEATURE_NAMES,
        }
        if self.has_news:
            cands["+news"] = ["gbm_logit"] + NEWS_FEATURE_NAMES
            cands["full"] = ALL_FEATURE_NAMES
            cands["full+"] = ALL_PLUS_FEATURE_NAMES
        return cands

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

    def _select_on_validation(
        self, train_rows: list[FeatureRow], ytr: list[float], candidates: dict[str, list[str]]
    ) -> str:
        """Pick the feature set with the best log-loss on a validation span carved from TRAIN by time.

        Selecting on a held-out validation span (not the test span) is what keeps 'pick the most
        accurate model' honest — the reported test metrics never influence which model we deploy.
        """
        default = "full+" if self.has_news else "+regime"
        as_ofs = sorted({r.as_of for r in train_rows})
        if len(as_ofs) < 6:
            return default
        cut = as_ofs[int(len(as_ofs) * 0.7)]
        fit = [(r, y) for r, y in zip(train_rows, ytr) if r.deadline <= cut]
        val = [(r, y) for r, y in zip(train_rows, ytr) if r.as_of > cut]
        if len(fit) < 20 or len(val) < 20:
            return default
        fr, fy = [r for r, _ in fit], [y for _, y in fit]
        vr, vy = [r for r, _ in val], [y for _, y in val]
        best, best_ll = default, float("inf")
        for name, fs in candidates.items():
            try:
                cal = LogisticCalibrator(fs, l2=self.l2).fit(fr, fy)
                ll = log_loss(vy, cal.predict(vr))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if ll < best_ll:
                best, best_ll = name, ll
        return best

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

        candidates = self._candidate_sets()
        results: dict[str, dict] = {"base_rate": metrics(yte, [base_rate] * len(yte))}
        results["raw"] = metrics(yte, [r.gbm_p for r in test_rows])
        test_preds: dict[str, np.ndarray] = {}
        for name, fs in candidates.items():
            cal = LogisticCalibrator(fs, l2=self.l2).fit(train_rows, ytr)
            pte = cal.predict(test_rows)
            test_preds[name] = pte
            results[name] = metrics(yte, pte)

        news_helps = self.has_news and (results["+news"]["log_loss"] < results["recalibrated"]["log_loss"])

        # Honest selection: choose the deployed feature set by VALIDATION log-loss (carved from train).
        selected = self._select_on_validation(train_rows, ytr, candidates)
        reliability = reliability_bins(yte, test_preds[selected])
        selected_fs = candidates[selected]

        # Deploy: refit the selected feature set on ALL history for live use.
        deployed = LogisticCalibrator(selected_fs, l2=self.l2).fit(rows, ys)

        return ForecastBacktestResult(
            n_predictions=len(rows), n_train=len(train_rows), n_test=len(test_rows),
            split_date=split_date, test_base_rate=base_rate, metrics=results,
            reliability=reliability, news_helps=news_helps, has_news=self.has_news,
            deployed_feature_set=selected_fs, selected_name=selected,
            selected_metrics=results[selected], _deployed=deployed,
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
