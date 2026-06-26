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
from .prediction import Prediction
from .probability import touch_probability

PRICE_FEATURE_NAMES = ["gbm_logit", "momentum_20", "dist_sigma"]
ALL_FEATURE_NAMES = PRICE_FEATURE_NAMES + NEWS_FEATURE_NAMES


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
    mom = lw.momentum(20)

    n_days = max(trading_days_to(lw.as_of.date(), prediction.deadline), 1)
    gbm_p = touch_probability(s0, prediction.target_price, n_days, mu, sigma, prediction.direction)

    # signed standardized distance to target (>0 = target is far in vol units)
    sqrt_t = max(sigma, 1e-6) * np.sqrt(n_days)
    if prediction.direction == Direction.UP:
        dist_sigma = float(np.log(prediction.target_price / s0) / sqrt_t)
    else:
        dist_sigma = float(np.log(s0 / prediction.target_price) / sqrt_t)

    # Orient direction-sensitive features toward the PREDICTED direction so their sign means the
    # same thing for UP and DOWN predictions ("supports the call" = positive). Without this, an
    # up-leaning signal helps UP-touch but hurts DOWN-touch predictions and the effect cancels.
    dir_sign = 1.0 if prediction.direction == Direction.UP else -1.0
    news_raw = news_provider.window(prediction.ticker, lw.as_of, news_lookback_days).features()
    feats = {
        "gbm_logit": _logit(gbm_p),
        "momentum_20": dir_sign * mom,
        "dist_sigma": dist_sigma,  # already computed per-direction (>0 = target far away)
        "news_sentiment_30": dir_sign * news_raw.get("news_sentiment_30", 0.0),
        "news_decay": dir_sign * news_raw.get("news_decay", 0.0),
        "news_volume_30": news_raw.get("news_volume_30", 0.0),  # volume is unsigned (confidence, not direction)
    }
    for name in ALL_FEATURE_NAMES:
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
