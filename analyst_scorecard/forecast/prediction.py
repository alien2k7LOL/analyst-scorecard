"""Schemas for a FORWARD prediction and its resolved outcome.

A ``Prediction`` is made AS OF a date and may be informed only by data up to that date. Its outcome
is whether the target price was TOUCHED (reached at least once) over (as_of, deadline]. Both are
frozen so a recorded prediction can't be silently re-chosen after the outcome is known.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schemas import Direction


class Prediction(BaseModel):
    """One forward-looking, target-touch prediction, fixed at record time (``as_of``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str
    ticker: str
    as_of: date = Field(description="When the prediction is made — only data <= as_of may inform it.")
    target_price: float = Field(gt=0, description="The price the prediction says will be touched.")
    deadline: date = Field(description="By when the target must be touched (strictly after as_of).")
    direction: Direction = Field(description="UP = rises to/through target; DOWN = falls to/through target.")
    made_by: str = "user"

    @field_validator("deadline")
    @classmethod
    def _deadline_after_as_of(cls, v: date, info) -> date:
        a = info.data.get("as_of")
        if a is not None and v <= a:
            raise ValueError(f"deadline {v} must be strictly after as_of {a}")
        return v

    @field_validator("direction")
    @classmethod
    def _must_be_directional(cls, v: Direction) -> Direction:
        if v == Direction.FLAT:
            raise ValueError("a touch prediction needs a direction (UP or DOWN), not FLAT")
        return v


class PredictionOutcome(BaseModel):
    """The look-ahead-safe ground truth for a prediction: was the target touched in the window?"""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    hit: bool
    hit_date: Optional[date] = None
    as_of_price: float = Field(gt=0, description="Close on the as_of trading day (the starting price).")
    extreme_price: float = Field(gt=0, description="Max close (UP) or min close (DOWN) over the window.")
    n_observations: int = Field(ge=2)
