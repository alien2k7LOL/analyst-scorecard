"""Plain-English analyst verdicts — Anthropic API with a deterministic offline fallback.

Same interface pattern as the rest of the engine:
  - ``TemplatedVerdictGenerator`` — deterministic, offline, no key. Builds an honest one-liner
    straight from the stats (e.g. "Right on direction most of the time, but you'd have done
    better just holding the index.").
  - ``LLMVerdictGenerator`` — Anthropic-backed, for a more natural phrasing when a key is set.

``default_verdict_generator()`` returns the LLM one if ``ANTHROPIC_API_KEY`` is present, else
the templated one — so verdicts always render offline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .schemas import AnalystScore

DEFAULT_VERDICT_MODEL = os.environ.get("SCORECARD_VERDICT_MODEL", "claude-opus-4-8")


class VerdictGenerator(ABC):
    @abstractmethod
    def verdict(self, score: AnalystScore) -> str:
        ...


# --------------------------------------------------------------------------------------
# Deterministic offline verdicts
# --------------------------------------------------------------------------------------


def _direction_phrase(hit_rate: float) -> str:
    if hit_rate >= 0.70:
        return "Right on direction most of the time"
    if hit_rate >= 0.55:
        return "Right on direction more often than not"
    if hit_rate >= 0.45:
        return "About a coin-flip on direction"
    return "Usually wrong on direction"


def _beat_phrase(beat_market: float | None) -> str:
    if beat_market is None:
        return "with no directional calls to judge against the index"
    if beat_market > 0.02:
        return "and following the calls beat the index"
    if beat_market >= -0.02:
        return "but following the calls only matched the index"
    return "but you'd have done better just holding the index"


def _accuracy_clause(mean_accuracy: float | None) -> str:
    if mean_accuracy is None:
        return ""
    if mean_accuracy >= 0.80:
        return " The price targets tend to land close, too."
    if mean_accuracy < 0.50:
        return " The price targets, though, are wildly off."
    return ""


class TemplatedVerdictGenerator(VerdictGenerator):
    """Honest, deterministic one-liner built from the headline + supporting stats."""

    def verdict(self, score: AnalystScore) -> str:
        sentence = f"{_direction_phrase(score.direction_hit_rate)} {_beat_phrase(score.beat_market)}."
        return sentence + _accuracy_clause(score.mean_accuracy)


# --------------------------------------------------------------------------------------
# LLM verdicts
# --------------------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You write a single, plain-English verdict (one or two short sentences) on a Wall Street "
    "analyst, for a retail audience. The HEADLINE metric is beat-the-market: whether following "
    "the analyst's calls would have beaten simply buying the benchmark index. Be honest and "
    "specific: a high direction hit-rate does NOT mean the analyst is good if they didn't beat "
    "the index. Do not invent numbers beyond those given. No preamble; just the verdict."
)


def _stats_blurb(score: AnalystScore) -> str:
    bm = "n/a" if score.beat_market is None else f"{score.beat_market * 100:+.1f}%"
    acc = "n/a" if score.mean_accuracy is None else f"{score.mean_accuracy:.2f}"
    return (
        f"Analyst: {score.analyst_name} ({score.firm}). "
        f"Beat-the-market (headline): {bm}. "
        f"Direction hit-rate: {score.direction_hit_rate * 100:.0f}%. "
        f"Volatility-scaled accuracy on direction-passing calls: {acc}. "
        f"Calls: {score.n_calls} ({score.n_directional} directional)."
    )


class LLMVerdictGenerator(VerdictGenerator):
    """Anthropic-backed verdict writer. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_VERDICT_MODEL, max_tokens: int = 120):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLMVerdictGenerator needs ANTHROPIC_API_KEY. Use TemplatedVerdictGenerator offline "
                "(or default_verdict_generator(), which falls back automatically)."
            )
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def verdict(self, score: AnalystScore) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _stats_blurb(score)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or TemplatedVerdictGenerator().verdict(score)


def default_verdict_generator() -> VerdictGenerator:
    """LLM generator if a key is present; otherwise the deterministic templated one."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMVerdictGenerator()
        except Exception:
            return TemplatedVerdictGenerator()
    return TemplatedVerdictGenerator()
