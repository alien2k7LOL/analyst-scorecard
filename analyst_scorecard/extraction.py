"""Analyst-call extraction: messy research text -> validated Call records.

Same interface, two implementations:
  - ``HeuristicCallExtractor`` — deterministic, offline, no API key. The default so the
    whole pipeline (and the accuracy harness) runs with no network.
  - ``LLMCallExtractor`` — the real implementation using the Anthropic API with structured
    JSON output (``messages.parse`` against a Pydantic schema), key read from
    ``ANTHROPIC_API_KEY``. Used when a key is present.

The extractor produces an ``ExtractedCall`` (the fields legible from the TEXT). Turning that
into a full ``Call`` (with a record-time resolution date and call-date price) is done by
``finalize_extracted`` using the price provider — keeping the look-ahead-safe deadline logic
in one place.
"""

from __future__ import annotations

import calendar
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import json

from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_CONFIG, TRADING_DAYS_PER_YEAR, ScorecardConfig
from .providers.call_provider import REPO_ROOT
from .providers.price_provider import PriceDataProvider
from .schemas import Call, Rating

DEFAULT_RESEARCH_NOTES = REPO_ROOT / "fixtures" / "research_notes" / "notes.json"

# The skill guidance: default to claude-opus-4-8 unless the user names another model.
DEFAULT_EXTRACTION_MODEL = os.environ.get("SCORECARD_EXTRACTION_MODEL", "claude-opus-4-8")


# --------------------------------------------------------------------------------------
# The extraction schema (also the structured-output schema for the LLM)
# --------------------------------------------------------------------------------------


class ExtractedCall(BaseModel):
    """The call fields legible from a research note. Validated; strict enum on rating."""

    model_config = ConfigDict(extra="forbid")

    analyst_name: str = Field(description="The named analyst, e.g. 'Dana Reyes'.")
    firm: str = Field(description="The research firm, e.g. 'Goldsmith & Co.'.")
    ticker: str = Field(description="The stock ticker symbol, uppercase.")
    rating: Rating = Field(description="One of Buy/Overweight/Hold/Underweight/Sell.")
    target_price: float = Field(gt=0, description="The stated price target as a number.")
    call_date: date = Field(description="The date of the note (the call date).")
    horizon_days: int = Field(gt=0, description="Horizon in TRADING days (12mo≈252, 6mo≈126, 3mo≈63).")


class ResearchNote(BaseModel):
    """A fixture note: messy text plus its ground-truth extraction (for the harness)."""

    model_config = ConfigDict(extra="forbid")

    note_id: str
    text: str
    expected: ExtractedCall


# --------------------------------------------------------------------------------------
# Extractor interface
# --------------------------------------------------------------------------------------


class CallExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> ExtractedCall:
        ...


# --------------------------------------------------------------------------------------
# Deterministic offline extractor
# --------------------------------------------------------------------------------------

# Rating scan order: the multi-word ratings first so "Overweight"/"Underweight" win over a
# stray "weight", and "Hold (Neutral)" maps to Hold.
_RATING_PATTERNS: list[tuple[re.Pattern, Rating]] = [
    (re.compile(r"\boverweight\b", re.I), Rating.OVERWEIGHT),
    (re.compile(r"\bunderweight\b", re.I), Rating.UNDERWEIGHT),
    (re.compile(r"\b(?:hold|neutral|market\s*perform)\b", re.I), Rating.HOLD),
    (re.compile(r"\bbuy\b", re.I), Rating.BUY),
    (re.compile(r"\bsell\b", re.I), Rating.SELL),
]
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_DATE_RE = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})")
_TARGET_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_MONTH_HORIZON_RE = re.compile(r"(\d+)\s*-?\s*month", re.I)
_FIELD_RE = lambda label: re.compile(rf"^{label}:\s*(.+)$", re.I | re.M)
_FIRM_RE = _FIELD_RE("Firm")
_ANALYST_RE = _FIELD_RE("Analyst")


class HeuristicCallExtractor(CallExtractor):
    """Regex/rule-based extractor — deterministic, offline, no API key required.

    Tuned to the synthetic research-note style (a labelled header + a prose body). It is the
    offline stand-in that lets the accuracy harness run and pass with no network; the real
    ``LLMCallExtractor`` handles arbitrary messy prose.
    """

    def __init__(self, config: ScorecardConfig = DEFAULT_CONFIG):
        # Tickers we recognize (the synthetic universe). A real version would resolve names.
        self._tickers = set(t.symbol for t in config.universe)

    def extract(self, text: str) -> ExtractedCall:
        return ExtractedCall(
            analyst_name=self._field(_ANALYST_RE, text, "analyst"),
            firm=self._field(_FIRM_RE, text, "firm"),
            ticker=self._ticker(text),
            rating=self._rating(text),
            target_price=self._target(text),
            call_date=self._date(text),
            horizon_days=self._horizon(text),
        )

    # -- field parsers -----------------------------------------------------------------
    @staticmethod
    def _field(pattern: re.Pattern, text: str, label: str) -> str:
        m = pattern.search(text)
        if not m:
            raise ValueError(f"could not extract {label} from note")
        return m.group(1).strip()

    def _ticker(self, text: str) -> str:
        for token in re.findall(r"\b[A-Z]{2,5}\b", text):
            if token in self._tickers:
                return token
        raise ValueError("no known ticker found in note")

    @staticmethod
    def _rating(text: str) -> Rating:
        for pattern, rating in _RATING_PATTERNS:
            if pattern.search(text):
                return rating
        raise ValueError("no rating keyword found in note")

    @staticmethod
    def _target(text: str) -> float:
        m = _TARGET_RE.search(text)
        if not m:
            raise ValueError("no price target found in note")
        return float(m.group(1).replace(",", ""))

    @staticmethod
    def _date(text: str) -> date:
        m = _DATE_RE.search(text)
        if not m:
            raise ValueError("no date found in note")
        month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        if month_name not in _MONTHS:
            raise ValueError(f"unrecognized month {month_name!r}")
        return date(year, _MONTHS[month_name], day)

    @staticmethod
    def _horizon(text: str) -> int:
        m = _MONTH_HORIZON_RE.search(text)
        if not m:
            return DEFAULT_CONFIG.default_horizon_days
        months = int(m.group(1))
        # Convert months to trading days (12mo -> 252, 6mo -> 126, 3mo -> 63).
        return round(months / 12 * TRADING_DAYS_PER_YEAR)


# --------------------------------------------------------------------------------------
# Real LLM extractor (Anthropic, structured JSON output)
# --------------------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You extract a single analyst price-target call from a research note. "
    "Return the named analyst, the research firm, the stock ticker (uppercase), the rating "
    "(exactly one of Buy, Overweight, Hold, Underweight, Sell), the numeric price target, the "
    "note's date as the call date, and the horizon expressed in TRADING days "
    "(12 months ≈ 252, 6 months ≈ 126, 3 months ≈ 63; if unstated, use 252). "
    "Map synonyms: 'Neutral'/'Market Perform' -> Hold, 'Equal Weight' -> Hold. "
    "Report only what the note states."
)


class LLMCallExtractor(CallExtractor):
    """Anthropic-backed extractor using structured JSON output (Pydantic-validated).

    Requires ``ANTHROPIC_API_KEY`` in the environment. This is the online path; the offline
    default is ``HeuristicCallExtractor`` (and ``FixtureCallProvider`` for pre-baked calls).
    """

    def __init__(self, model: str = DEFAULT_EXTRACTION_MODEL, max_tokens: int = 1024):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "LLMCallExtractor needs ANTHROPIC_API_KEY. For offline use, use "
                "HeuristicCallExtractor (or FixtureCallProvider for pre-baked calls)."
            )
        import anthropic  # imported lazily so the package stays import-clean offline

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def extract(self, text: str) -> ExtractedCall:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=ExtractedCall,
        )
        parsed = response.parsed_output
        if parsed is None:  # refusal or schema miss
            raise RuntimeError(f"extraction returned no parsed output (stop={response.stop_reason})")
        return parsed


# --------------------------------------------------------------------------------------
# Finalize extracted -> full Call (record-time deadline + call-date price)
# --------------------------------------------------------------------------------------


def finalize_extracted(
    extracted: ExtractedCall,
    provider: PriceDataProvider,
    *,
    analyst_id: Optional[str] = None,
    config: ScorecardConfig = DEFAULT_CONFIG,
) -> Call:
    """Turn an ExtractedCall into a full, look-ahead-safe Call.

    The resolution date is fixed HERE, at record time, as call_date + horizon trading days,
    and the call-date price is read from the provider — so a freshly extracted call obeys the
    exact same fairness contract as the fixture calls.
    """
    call_ts = provider.trading_day_offset(extracted.call_date, 0)  # validates it's a trading day
    resolution_ts = provider.trading_day_offset(extracted.call_date, extracted.horizon_days)
    initial_price = provider.price_on(extracted.ticker, extracted.call_date)
    aid = analyst_id or _slug(extracted.analyst_name)
    return Call(
        call_id=f"{aid}-{extracted.ticker}-{extracted.call_date.isoformat()}",
        analyst_id=aid,
        analyst_name=extracted.analyst_name,
        firm=extracted.firm,
        ticker=extracted.ticker,
        rating=extracted.rating,
        target_price=round(extracted.target_price, 2),
        call_date=call_ts.date(),
        horizon_days=extracted.horizon_days,
        resolution_date=resolution_ts.date(),
        initial_price=float(initial_price),
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# --------------------------------------------------------------------------------------
# Accuracy harness
# --------------------------------------------------------------------------------------


def load_research_notes(path: Path | str = DEFAULT_RESEARCH_NOTES) -> list[ResearchNote]:
    raw = json.loads(Path(path).read_text())
    return [ResearchNote.model_validate(item) for item in raw]


_FIELDS = ("analyst_name", "firm", "ticker", "rating", "target_price", "call_date", "horizon_days")


@dataclass
class ExtractionReport:
    n_notes: int
    per_field_accuracy: dict[str, float]
    exact_match_rate: float
    mismatches: list[dict]

    def summary(self) -> str:
        lines = [f"notes={self.n_notes}  exact_match={self.exact_match_rate:.0%}"]
        for f in _FIELDS:
            lines.append(f"  {f:<14} {self.per_field_accuracy[f]:.0%}")
        return "\n".join(lines)


def _field_equal(field: str, a, b) -> bool:
    if field == "target_price":
        return abs(float(a) - float(b)) < 0.01
    return a == b


def evaluate_extractor(extractor: CallExtractor, notes: list[ResearchNote]) -> ExtractionReport:
    """Run an extractor over labelled notes and score it field-by-field against ground truth."""
    field_hits = {f: 0 for f in _FIELDS}
    exact = 0
    mismatches: list[dict] = []

    for note in notes:
        got = extractor.extract(note.text)
        expected = note.expected
        all_ok = True
        for f in _FIELDS:
            ok = _field_equal(f, getattr(got, f), getattr(expected, f))
            if ok:
                field_hits[f] += 1
            else:
                all_ok = False
                mismatches.append(
                    {"note_id": note.note_id, "field": f,
                     "expected": getattr(expected, f), "got": getattr(got, f)}
                )
        if all_ok:
            exact += 1

    n = len(notes)
    return ExtractionReport(
        n_notes=n,
        per_field_accuracy={f: (field_hits[f] / n if n else 0.0) for f in _FIELDS},
        exact_match_rate=(exact / n if n else 0.0),
        mismatches=mismatches,
    )
