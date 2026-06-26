"""Assemble the look-ahead-safe feature row for one prediction.

Every feature here is computed from data on/before ``as_of`` only: price features come from the
``LookbackWindow`` (which ends at as_of) and news features from the ``NewsWindow`` (events <= as_of).
The headline price feature is ``gbm_logit`` — the logit of the closed-form touch probability — which
later acts as the base score the calibrator recalibrates and the other features adjust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..providers.price_provider import PriceDataProvider, _ts
from ..schemas import Direction
from .lookback import lookback_window
from .news import NEWS_FEATURE_NAMES, NewsProvider, NoNewsProvider
from .prediction import Prediction, PredictionKind
from .probability import MIN_SIGMA, terminal_probability, touch_probability

PRICE_FEATURE_NAMES = ["gbm_logit", "momentum_20", "dist_sigma"]
# Extra look-ahead-safe price/regime features, discovered to help the (harder) terminal model:
#   drift_sigma — per-horizon drift toward the target, in vol units (signal-to-noise of the trend)
#   vol_ratio   — recent (20d) vol vs the full lookback vol (is volatility expanding or calming?)
#   horizon_log — log trading-days to the deadline (lets the calibrator correct by how far out it is)
#   reversion   — how stretched the price is vs its own lookback mean, oriented toward the call
EXT_PRICE_FEATURE_NAMES = ["drift_sigma", "vol_ratio", "horizon_log", "reversion"]
ALL_FEATURE_NAMES = PRICE_FEATURE_NAMES + NEWS_FEATURE_NAMES          # unchanged (back-compat)
ALL_PLUS_FEATURE_NAMES = PRICE_FEATURE_NAMES + EXT_PRICE_FEATURE_NAMES + NEWS_FEATURE_NAMES


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class FeatureRow:
    prediction_id: str
    ticker: str
    as_of: date
    deadline: date
    direction: Direction
    s0: float
    target: float
    n_days: int
    gbm_p: float            # raw closed-form touch probability
    features: dict          # all numeric features, incl gbm_logit (keys in ALL_FEATURE_NAMES)


def trading_days_to(as_of: date, deadline: date) -> int:
    """Business-day count to the deadline — known in advance, so it peeks at no future prices."""
    return int(np.busday_count(as_of, deadline))


def build_features(
    price_provider: PriceDataProvider,
    prediction: Prediction,
    news_provider: NewsProvider | None = None,
    lookback_days: int = 252,
    news_lookback_days: int = 120,
) -> FeatureRow:
    news_provider = news_provider or NoNewsProvider()
    lw = lookback_window(price_provider, prediction.ticker, prediction.as_of, lookback_days)
    s0 = lw.last_price
    mu, sigma = lw.drift_vol()
    sigma_floor = max(sigma, MIN_SIGMA)
    mom = lw.momentum(20)

    n_days = max(trading_days_to(lw.as_of.date(), prediction.deadline), 1)
    # Base probability for THIS prediction's kind: touch (path) or terminal (endpoint at the deadline).
    if prediction.kind == PredictionKind.TERMINAL:
        gbm_p = terminal_probability(
            s0, prediction.target_price, n_days, mu, sigma, prediction.direction, prediction.band_pct
        )
    else:
        gbm_p = touch_probability(s0, prediction.target_price, n_days, mu, sigma, prediction.direction)

    # signed standardized distance to target (>0 = target is far in vol units)
    sqrt_t = sigma_floor * np.sqrt(n_days)
    if prediction.direction == Direction.UP:
        dist_sigma = float(np.log(prediction.target_price / s0) / sqrt_t)
    else:
        dist_sigma = float(np.log(s0 / prediction.target_price) / sqrt_t)

    # Orient direction-sensitive features toward the PREDICTED direction so their sign means the
    # same thing for UP and DOWN predictions ("supports the call" = positive). Without this, an
    # up-leaning signal helps UP predictions but hurts DOWN predictions and the effect cancels.
    dir_sign = 1.0 if prediction.direction == Direction.UP else -1.0

    # --- extended price/regime features (all from the lookback, all look-ahead-safe) ---
    drift_sigma = dir_sign * float(mu * np.sqrt(n_days) / sigma_floor)  # drift toward target, in vol units
    returns = lw.daily_log_returns()
    recent = returns[-20:] if len(returns) >= 20 else returns
    sigma_short = float(np.std(recent, ddof=1)) if len(recent) >= 2 else sigma_floor
    vol_ratio = float(np.log(max(sigma_short, MIN_SIGMA) / sigma_floor))
    horizon_log = float(np.log(n_days))
    prices = lw.prices.to_numpy(dtype=float)
    p_mean, p_std = float(prices.mean()), float(prices.std(ddof=1))
    reversion = -dir_sign * float((s0 - p_mean) / max(p_std, 1e-9))  # +: stretched in the call's favor

    news_raw = news_provider.window(prediction.ticker, lw.as_of, news_lookback_days).features()
    feats = {
        "gbm_logit": _logit(gbm_p),
        "momentum_20": dir_sign * mom,
        "dist_sigma": dist_sigma,  # already computed per-direction (>0 = target far away)
        "drift_sigma": drift_sigma,
        "vol_ratio": vol_ratio,
        "horizon_log": horizon_log,
        "reversion": reversion,
        "news_sentiment_30": dir_sign * news_raw.get("news_sentiment_30", 0.0),
        "news_decay": dir_sign * news_raw.get("news_decay", 0.0),
        "news_volume_30": news_raw.get("news_volume_30", 0.0),  # volume is unsigned (confidence, not direction)
    }
    for name in ALL_PLUS_FEATURE_NAMES:
        feats.setdefault(name, 0.0)

    return FeatureRow(
        prediction_id=prediction.prediction_id,
        ticker=prediction.ticker,
        as_of=lw.as_of.date(),
        deadline=prediction.deadline,
        direction=prediction.direction,
        s0=s0,
        target=prediction.target_price,
        n_days=n_days,
        gbm_p=gbm_p,
        features=feats,
    )
