"""Typed schemas (Pydantic v2) for calls, resolutions, and scores.

These are the contracts the whole engine speaks. Keeping them strict (validated prices,
fixed enums, frozen rating->direction map) is part of the fairness guarantee: a malformed
or ambiguous call cannot silently enter the scoring funnel.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Direction(str, Enum):
    """Implied/realized direction of a move."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class Rating(str, Enum):
    """The five analyst ratings we accept. Mapped to a direction by RATING_TO_DIRECTION."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# Frozen, total mapping from rating to the analyst's IMPLIED direction. This is fixed up
# front and applied identically to every analyst — no rating is ever reinterpreted.
RATING_TO_DIRECTION: dict[Rating, Direction] = {
    Rating.BUY: Direction.UP,
    Rating.OVERWEIGHT: Direction.UP,
    Rating.HOLD: Direction.FLAT,
    Rating.UNDERWEIGHT: Direction.DOWN,
    Rating.SELL: Direction.DOWN,
}

# Position taken by "following the call" in the beat-the-market book.
# HOLD -> 0 (neutral, excluded from the book).
DIRECTION_TO_POSITION: dict[Direction, int] = {
    Direction.UP: +1,
    Direction.DOWN: -1,
    Direction.FLAT: 0,
}


class Call(BaseModel):
    """One analyst price-target call, with its resolution rule fixed at record time.

    The deadline (``resolution_date``) and horizon are recorded WHEN THE CALL IS MADE and
    are never re-chosen after the outcome is known — this is the structural defense against
    look-ahead bias at the call level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    analyst_id: str
    analyst_name: str
    firm: str
    ticker: str
    rating: Rating
    target_price: float = Field(gt=0, description="Analyst's price target (> 0).")
    call_date: date = Field(description="Trading day the call was recorded.")
    horizon_days: int = Field(gt=0, description="Horizon in TRADING days, fixed at record time.")
    resolution_date: date = Field(description="Deadline = call_date + horizon_days trading days; fixed at record time.")
    initial_price: float = Field(gt=0, description="Market close on the call date (snapshot the analyst acted on).")

    @field_validator("resolution_date")
    @classmethod
    def _resolution_after_call(cls, v: date, info) -> date:
        call_date = info.data.get("call_date")
        if call_date is not None and v <= call_date:
            raise ValueError(f"resolution_date {v} must be strictly after call_date {call_date}")
        return v

    @property
    def implied_direction(self) -> Direction:
        """The direction the rating implies (Buy/OW->UP, Sell/UW->DOWN, Hold->FLAT)."""
        return RATING_TO_DIRECTION[self.rating]

    @property
    def implied_position(self) -> int:
        """+1 long, -1 short, 0 neutral (Hold) — the position used for beat-the-market."""
        return DIRECTION_TO_POSITION[self.implied_direction]

    @property
    def is_directional(self) -> bool:
        """True for Buy/OW/Sell/UW (takes a position); False for Hold (neutral)."""
        return self.implied_position != 0


class Resolution(BaseModel):
    """Look-ahead-safe outcome of a call, plus every input the scoring funnel needs.

    Produced ONLY from data in the window [call_date, resolution_date]. Carries the exact
    prices used, so any score is traceable back to the numbers that produced it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    call_date: date
    resolution_date: date

    # Prices actually used (traceability) ----------------------------------------------
    call_price: float = Field(gt=0, description="Stock close on the call date.")
    target_price: float = Field(gt=0)
    actual_price: float = Field(gt=0, description="Stock close on the resolution date.")
    benchmark_call_price: float = Field(gt=0)
    benchmark_actual_price: float = Field(gt=0)

    # Derived quantities over the horizon ----------------------------------------------
    stock_return: float = Field(description="actual/call - 1 over the horizon.")
    benchmark_return: float = Field(description="benchmark actual/call - 1 over the horizon.")
    realized_horizon_vol: float = Field(ge=0, description="Realized daily vol * sqrt(horizon_days).")
    n_observations: int = Field(gt=1, description="Daily price points used (call..resolution inclusive).")

    @property
    def realized_direction(self) -> Direction:
        # Bucketing is delegated to the scorer (which owns the band); this is a convenience
        # only. Kept here as a property so it is always derived, never stored stale.
        raise NotImplementedError("Use scoring.bucket_direction(stock_return, config).")


class CallScore(BaseModel):
    """The graded outcome of a single call: the funnel result + full traceability.

    accuracy is None unless the call PASSED the direction gate (accuracy refines only
    direction-passers). beat is None for Hold calls (neutral; out of the beat-market book).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    analyst_id: str
    ticker: str
    rating: Rating
    implied_direction: Direction

    # Stage 1 — direction gate
    realized_direction: Direction
    direction_pass: bool

    # Stage 2 — accuracy (only meaningful when direction_pass is True)
    accuracy: Optional[float] = Field(default=None, ge=0, le=1)

    # Stage 3 — beat-the-market (None for Hold/neutral)
    position: int = Field(description="+1 long, -1 short, 0 neutral.")
    call_return: Optional[float] = Field(default=None, description="position * stock_return; None for Hold.")
    beat: Optional[float] = Field(default=None, description="call_return - benchmark_return; None for Hold.")

    # Carried-through resolution for traceability
    resolution: Resolution

    @model_validator(mode="after")
    def _accuracy_only_for_passers(self) -> "CallScore":
        if not self.direction_pass and self.accuracy is not None:
            raise ValueError("accuracy must be None for calls that failed the direction gate")
        return self


class AnalystScore(BaseModel):
    """Per-analyst aggregate. beat_market is the HEADLINE; the rest are supporting stats."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyst_id: str
    analyst_name: str
    firm: str

    n_calls: int
    n_directional: int = Field(description="Buy/Sell calls (Hold excluded).")
    n_direction_pass: int

    # Headline
    beat_market: Optional[float] = Field(
        default=None,
        description="Mean (call_return - benchmark_return) over ALL directional calls; None if none.",
    )
    # Supporting
    direction_hit_rate: float = Field(ge=0, le=1, description="n_direction_pass / n_calls.")
    mean_accuracy: Optional[float] = Field(
        default=None, ge=0, le=1, description="Mean accuracy over direction-PASSING calls; None if none."
    )

    call_scores: tuple[CallScore, ...] = Field(default_factory=tuple)


class Leaderboard(BaseModel):
    """Analysts ranked by the headline beat-the-market figure (desc); None sorts last."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: tuple[AnalystScore, ...]

    @staticmethod
    def from_scores(scores: list[AnalystScore]) -> "Leaderboard":
        def sort_key(s: AnalystScore):
            # None beat_market (no directional calls) sorts last; ties broken by direction.
            has_beat = s.beat_market is not None
            return (has_beat, s.beat_market if has_beat else 0.0, s.direction_hit_rate)

        ordered = tuple(sorted(scores, key=sort_key, reverse=True))
        return Leaderboard(rows=ordered)
