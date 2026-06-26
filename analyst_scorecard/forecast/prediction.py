"""Schemas for a FORWARD prediction and its resolved outcome.

A ``Prediction`` is made AS OF a date and may be informed only by data up to that date. Its outcome
depends on the prediction's ``kind``:

  * TOUCH    — was the target price REACHED at least once over (as_of, deadline]? (a path question)
  * TERMINAL — where is the price ON the deadline itself? Either within a ±band of the target
               (``band_pct`` set: "it will be at ~that price") or at/through it in the predicted
               direction (``band_pct`` None: "it will close the period at/through that price").

Both models are frozen so a recorded prediction can't be silently re-chosen after the outcome is known.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schemas import Direction


class PredictionKind(str, Enum):
    """What the prediction claims about the price over the window."""

    TOUCH = "touch"        # reaches the target at any point before the deadline (path max/min)
    TERMINAL = "terminal"  # is at/near the target ON the deadline (endpoint value)


class Prediction(BaseModel):
    """One forward-looking price prediction, fixed at record time (``as_of``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str
    ticker: str
    as_of: date = Field(description="When the prediction is made — only data <= as_of may inform it.")
    target_price: float = Field(gt=0, description="The price the prediction is about.")
    deadline: date = Field(description="The resolution date (strictly after as_of).")
    direction: Direction = Field(description="UP = rises to/through target; DOWN = falls to/through target.")
    kind: PredictionKind = Field(
        default=PredictionKind.TOUCH,
        description="TOUCH = reached any time before the deadline; TERMINAL = at the price on the deadline.",
    )
    band_pct: Optional[float] = Field(
        default=None,
        gt=0,
        description="TERMINAL only: ± tolerance band around the target (e.g. 0.03 = within 3%). "
        "None means 'at or through' the target in the predicted direction.",
    )
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
            raise ValueError("a price prediction needs a direction (UP or DOWN), not FLAT")
        return v


class PredictionOutcome(BaseModel):
    """The look-ahead-safe ground truth for a prediction — computed only from data within the window."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    hit: bool
    hit_date: Optional[date] = None
    as_of_price: float = Field(gt=0, description="Close on the as_of trading day (the starting price).")
    extreme_price: float = Field(gt=0, description="Max close (UP) or min close (DOWN) over the window.")
    n_observations: int = Field(ge=2)
    terminal_price: Optional[float] = Field(
        default=None, description="TERMINAL only: the close on the deadline trading day (what resolved it)."
    )
    terminal_date: Optional[date] = Field(
        default=None, description="TERMINAL only: the actual trading day used as the deadline."
    )
