"""Scoring engine — the three-stage funnel, implemented exactly as specified.

A call is graded as a FUNNEL, not a blended average:

  1. DIRECTION  — a pass/fail GATE. The realized move is bucketed (UP / DOWN / FLAT) using a
     single flat band applied identically to every rating. The call passes iff the realized
     direction equals the rating's implied direction. A Buy that fell FAILS, full stop.

  2. ACCURACY   — refines ONLY direction-passers. Volatility-normalized closeness of the
     actual price to the target:  exp( -(|P_actual - P_target| / P_call) / sigma_h ).
     1.0 is a bullseye; it decays as the miss grows relative to how much THAT stock moved over
     THIS horizon, so a tight call on a volatile stock counts more than on a calm one.
     accuracy is None for calls that failed the gate.

  3. BEAT-THE-MARKET — the HEADLINE. Return of following the call minus the benchmark return
     over the same window:  position * stock_return - benchmark_return  (position +1 long /
     -1 short / 0 Hold). Computed for ALL directional calls — winners and losers alike, because
     it is the money metric: it must include the Buys that went down. Hold calls are neutral
     (no position) and are excluded from the beat-the-market book.

Note the funnel structure: direction GATES accuracy; beat-the-market is its own headline over
every directional call (it is NOT gated by direction — a wrong-direction Buy still loses money,
and that loss must count).
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, ScorecardConfig
from .schemas import CallScore, Direction, Call, Resolution


def bucket_direction(stock_return: float, config: ScorecardConfig = DEFAULT_CONFIG) -> Direction:
    """Bucket a realized return into UP / DOWN / FLAT using the single shared flat band."""
    band = config.direction_flat_band
    if stock_return > band:
        return Direction.UP
    if stock_return < -band:
        return Direction.DOWN
    return Direction.FLAT


def compute_accuracy(resolution: Resolution, config: ScorecardConfig = DEFAULT_CONFIG) -> float:
    """Volatility-normalized closeness in (0, 1]. Exactly 1.0 when actual == target.

    error_frac      = |P_actual - P_target| / P_call          (miss as a fraction of price)
    sigma_h         = realized stock vol over the horizon (floored to avoid /0)
    normalized_err  = error_frac / (sigma_h * accuracy_scale) (miss in 'horizon sigmas')
    accuracy        = exp(-normalized_err)
    """
    error_frac = abs(resolution.actual_price - resolution.target_price) / resolution.call_price
    sigma_h = max(resolution.realized_horizon_vol, config.min_sigma_h)
    normalized_error = error_frac / (sigma_h * config.accuracy_scale)
    return float(np.exp(-normalized_error))


def score_call(call: Call, resolution: Resolution, config: ScorecardConfig = DEFAULT_CONFIG) -> CallScore:
    """Run one call through the full funnel, carrying the resolution for traceability."""
    realized_direction = bucket_direction(resolution.stock_return, config)
    implied_direction = call.implied_direction
    direction_pass = realized_direction == implied_direction

    # Stage 2 — accuracy only for direction-passers.
    accuracy = compute_accuracy(resolution, config) if direction_pass else None

    # Stage 3 — beat-the-market over ALL directional calls (Hold excluded).
    position = call.implied_position
    call_return: float | None = None
    beat: float | None = None
    if position != 0:
        call_return = position * resolution.stock_return
        beat = call_return - resolution.benchmark_return

    return CallScore(
        call_id=call.call_id,
        analyst_id=call.analyst_id,
        ticker=call.ticker,
        rating=call.rating,
        implied_direction=implied_direction,
        realized_direction=realized_direction,
        direction_pass=direction_pass,
        accuracy=accuracy,
        position=position,
        call_return=call_return,
        beat=beat,
        resolution=resolution,
    )
