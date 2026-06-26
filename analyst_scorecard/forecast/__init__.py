"""Forecast & calibration subsystem — grade FORWARD predictions with a calibrated probability.

The existing scorecard grades calls *after* their horizon (what actually happened). This package
grades a prediction *before* the fact: given (ticker, target, deadline, direction), it estimates
the probability the target is TOUCHED by the deadline, and — crucially — backtests that probability
on history to measure and refine its CALIBRATION (when it says 70%, does it happen ~70%?).

Same look-ahead-safe spine as the rest of the project: a probability at a date uses ONLY data up
to that date (price lookback + point-in-time news); the outcome uses ONLY what happened after.
"""

from .prediction import Prediction, PredictionKind, PredictionOutcome
from .probability import (
    GbmTerminalModel,
    GbmTouchModel,
    empirical_terminal_probability,
    empirical_touch_probability,
    terminal_probability,
    touch_probability,
)

__all__ = [
    "Prediction",
    "PredictionKind",
    "PredictionOutcome",
    "GbmTouchModel",
    "GbmTerminalModel",
    "touch_probability",
    "terminal_probability",
    "empirical_touch_probability",
    "empirical_terminal_probability",
]
