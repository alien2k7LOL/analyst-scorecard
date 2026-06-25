"""Per-analyst aggregation and leaderboard construction.

Turns per-call scores into the per-analyst record. The HEADLINE is ``beat_market``: the mean
excess return (vs the index) of following the analyst's directional calls. Direction hit-rate
and mean accuracy are the supporting stats.

Aggregation conventions (fixed, uniform):
  - direction_hit_rate = (# calls passing the gate) / (# ALL calls), Holds included.
  - mean_accuracy      = mean accuracy over DIRECTION-PASSING calls only (None if none).
  - beat_market        = mean beat over ALL DIRECTIONAL calls — winners AND losers (None if
                         the analyst made no directional calls). Per-call excess returns are
                         averaged (not compounded) so the figure is comparable across analysts
                         regardless of how many calls they made.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .config import DEFAULT_CONFIG, ScorecardConfig
from .providers.price_provider import PriceDataProvider
from .resolution import resolve_call_with_provider
from .schemas import AnalystScore, Call, CallScore, Leaderboard
from .scoring import score_call


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def score_calls(
    calls: Iterable[Call], provider: PriceDataProvider, config: ScorecardConfig = DEFAULT_CONFIG
) -> list[CallScore]:
    """Resolve (look-ahead-safe) and score every call."""
    out: list[CallScore] = []
    for call in calls:
        resolution = resolve_call_with_provider(call, provider)
        out.append(score_call(call, resolution, config))
    return out


def aggregate_analyst(call_scores: list[CallScore], *, analyst_name: str, firm: str) -> AnalystScore:
    """Aggregate one analyst's call scores. All scores must share the same analyst_id."""
    if not call_scores:
        raise ValueError("aggregate_analyst requires at least one call score")
    analyst_id = call_scores[0].analyst_id
    if any(cs.analyst_id != analyst_id for cs in call_scores):
        raise ValueError("all call scores must belong to the same analyst")

    n_calls = len(call_scores)
    n_directional = sum(1 for cs in call_scores if cs.position != 0)
    n_pass = sum(1 for cs in call_scores if cs.direction_pass)

    direction_hit_rate = n_pass / n_calls
    passing_accuracies = [cs.accuracy for cs in call_scores if cs.direction_pass and cs.accuracy is not None]
    directional_beats = [cs.beat for cs in call_scores if cs.position != 0 and cs.beat is not None]

    return AnalystScore(
        analyst_id=analyst_id,
        analyst_name=analyst_name,
        firm=firm,
        n_calls=n_calls,
        n_directional=n_directional,
        n_direction_pass=n_pass,
        beat_market=_mean_or_none(directional_beats),
        direction_hit_rate=direction_hit_rate,
        mean_accuracy=_mean_or_none(passing_accuracies),
        call_scores=tuple(sorted(call_scores, key=lambda cs: cs.call_id)),
    )


def aggregate_all(
    calls: Iterable[Call], provider: PriceDataProvider, config: ScorecardConfig = DEFAULT_CONFIG
) -> list[AnalystScore]:
    """Score every call and aggregate into one AnalystScore per analyst."""
    calls = list(calls)
    scores = score_calls(calls, provider, config)

    # name/firm lookup from the calls themselves (first occurrence wins)
    meta: dict[str, tuple[str, str]] = {}
    for c in calls:
        meta.setdefault(c.analyst_id, (c.analyst_name, c.firm))

    by_analyst: dict[str, list[CallScore]] = {}
    for cs in scores:
        by_analyst.setdefault(cs.analyst_id, []).append(cs)

    out: list[AnalystScore] = []
    for analyst_id, css in by_analyst.items():
        name, firm = meta[analyst_id]
        out.append(aggregate_analyst(css, analyst_name=name, firm=firm))
    return out


def build_leaderboard(
    calls: Iterable[Call], provider: PriceDataProvider, config: ScorecardConfig = DEFAULT_CONFIG
) -> Leaderboard:
    """End-to-end: calls -> resolved -> scored -> aggregated -> ranked by beat-the-market."""
    return Leaderboard.from_scores(aggregate_all(calls, provider, config))
