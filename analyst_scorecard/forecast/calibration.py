"""Calibration metrics + a logistic recalibration layer (this is how the backtest 'refines' accuracy).

A probability is only as good as its calibration: when the model says 0.7, the event should happen
~70% of the time. We measure that with the Brier score, log-loss, and Expected Calibration Error
(ECE), and we IMPROVE it by fitting a logistic layer on a TRAIN span and applying it to a held-out
TEST span. That same layer does double duty: with only ``gbm_logit`` as input it is pure Platt
recalibration; adding momentum / news features lets those signals adjust the probability — but only
to the extent they help on held-out data, which the metrics then judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def brier_score(y: Sequence[float], p: Sequence[float]) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y: Sequence[float], p: Sequence[float]) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y: Sequence[float], p: Sequence[float]) -> float:
    """Discrimination: P(model ranks a random touch above a random non-touch). 0.5 = coin flip.

    Calibration alone can be perfect while a model is useless (always predict the base rate is
    perfectly calibrated but AUC ~0.5). AUC answers the other half: does it actually separate
    likely from unlikely? Rank-based (Mann-Whitney U), tie-aware.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks_sorted = np.arange(1, len(p) + 1, dtype=float)
    i = 0
    while i < len(sp):  # average ranks within ties
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(p), float)
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier_skill_score(y: Sequence[float], p: Sequence[float]) -> float:
    """Brier improvement vs always predicting the base rate. >0 = beats climatology; 0 = no edge."""
    y = np.asarray(y, float)
    if len(y) == 0:
        return float("nan")
    base = float(y.mean())
    b0 = brier_score(y, np.full(len(y), base))
    return float(1.0 - brier_score(y, p) / b0) if b0 > 0 else 0.0


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    mean_actual: float


def reliability_bins(y: Sequence[float], p: Sequence[float], n_bins: int = 10) -> list[ReliabilityBin]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.any():
            out.append(ReliabilityBin(lo, hi, int(mask.sum()),
                                      float(p[mask].mean()), float(y[mask].mean())))
    return out


def expected_calibration_error(y: Sequence[float], p: Sequence[float], n_bins: int = 10) -> float:
    """Sample-weighted average gap between predicted confidence and actual frequency."""
    bins = reliability_bins(y, p, n_bins)
    n = len(np.asarray(y))
    if n == 0:
        return float("nan")
    return float(sum(b.n * abs(b.mean_pred - b.mean_actual) for b in bins) / n)


def metrics(y: Sequence[float], p: Sequence[float], n_bins: int = 10) -> dict:
    return {
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "auc": roc_auc(y, p),
        "bss": brier_skill_score(y, p),
        "n": int(len(np.asarray(y))),
    }


class LogisticCalibrator:
    """Ridge-regularized logistic regression fit by IRLS — the recalibration / blending layer.

    Features are standardized (so the L2 penalty is even-handed) and the intercept is unpenalized.
    With ``feature_names = ['gbm_logit']`` this is Platt scaling; adding features blends them in.
    """

    def __init__(self, feature_names: Sequence[str], l2: float = 1.0, max_iter: int = 100):
        self.feature_names = list(feature_names)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.coef_: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _raw_matrix(self, rows) -> np.ndarray:
        return np.array([[float(r.features[f]) for f in self.feature_names] for r in rows], dtype=float)

    def _design(self, rows) -> np.ndarray:
        x = self._raw_matrix(rows)
        xs = (x - self._mean) / self._std
        return np.hstack([np.ones((len(xs), 1)), xs])

    def fit(self, rows, y: Sequence[float]) -> "LogisticCalibrator":
        x = self._raw_matrix(rows)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        self._std[self._std < 1e-9] = 1.0
        xd = np.hstack([np.ones((len(x), 1)), (x - self._mean) / self._std])
        y = np.asarray(y, float)

        reg = self.l2 * np.ones(xd.shape[1])
        reg[0] = 0.0  # don't penalize the intercept
        w = np.zeros(xd.shape[1])
        for _ in range(self.max_iter):
            p = np.clip(1.0 / (1.0 + np.exp(-(xd @ w))), 1e-9, 1 - 1e-9)
            grad = xd.T @ (p - y) + reg * w
            hess = xd.T @ (xd * (p * (1 - p))[:, None]) + np.diag(reg)
            step = np.linalg.solve(hess, grad)
            w = w - step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.coef_ = w
        return self

    def predict(self, rows) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("calibrator is not fit")
        return 1.0 / (1.0 + np.exp(-(self._design(rows) @ self.coef_)))
