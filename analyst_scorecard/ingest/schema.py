"""The strict output schema for one analyst call, plus normalization helpers.

One ``AnalystCall`` == one line of ``data/analyst_calls.jsonl``. Missing fields are ``None`` (never
guessed). The ``id`` is a deterministic hash of the call's identity (NOT of ``extracted_at``), so the
same call from the same source always hashes the same — that's what makes dedup and re-runs idempotent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Optional

from ..intel.extract import normalize_rating

# Canonical action vocabulary.
ACTIONS = ("upgrade", "downgrade", "reiteration", "initiation")

_ACTION_SYNONYMS = {
    "up": "upgrade", "upgrade": "upgrade", "upgrades": "upgrade", "raised": "upgrade",
    "down": "downgrade", "downgrade": "downgrade", "downgrades": "downgrade", "cut": "downgrade",
    "init": "initiation", "initiates": "initiation", "initiated": "initiation", "initiate": "initiation",
    "launch": "initiation", "starts": "initiation", "start": "initiation",
    "main": "reiteration", "maintains": "reiteration", "maintain": "reiteration",
    "reit": "reiteration", "reiterate": "reiteration", "reiterates": "reiteration",
    "reaffirm": "reiteration", "reinit": "reiteration", "reiterated": "reiteration",
}

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")   # AAPL, BRK.B; a real universe check is a drop-in


def normalize_action(raw: Optional[str]) -> Optional[str]:
    """Map any phrasing of an analyst action to upgrade/downgrade/reiteration/initiation, or None."""
    if not raw:
        return None
    t = str(raw).strip().lower()
    if t in ACTIONS:
        return t
    if t in _ACTION_SYNONYMS:
        return _ACTION_SYNONYMS[t]
    for word, action in _ACTION_SYNONYMS.items():           # substring fallback for free text
        if word in t:
            return action
    return None


def detect_action(text: str) -> Optional[str]:
    """Infer an action from free text ('upgrades', 'initiates coverage', 'reiterates'…)."""
    low = (text or "").lower()
    if re.search(r"\bupgrad", low):
        return "upgrade"
    if re.search(r"\bdowngrad", low):
        return "downgrade"
    if re.search(r"\b(initiat|launch(es|ed)?\s+coverage|starts?\s+coverage)", low):
        return "initiation"
    if re.search(r"\b(reiterat|maintain|reaffirm|keep)", low):
        return "reiteration"
    return None


def _iso_date(d) -> Optional[str]:
    if d is None or d == "":
        return None
    if isinstance(d, (date, datetime)):
        return d.date().isoformat() if isinstance(d, datetime) else d.isoformat()
    try:
        import pandas as pd
        ts = pd.to_datetime(d, errors="coerce")
        return None if ts is None or pd.isna(ts) else ts.date().isoformat()
    except Exception:
        return None


@dataclass
class AnalystCall:
    ticker: Optional[str] = None
    company: Optional[str] = None
    analyst: Optional[str] = None
    firm: Optional[str] = None
    rating: Optional[str] = None
    target_price: Optional[float] = None
    previous_target: Optional[float] = None
    action: Optional[str] = None
    source_url: Optional[str] = None
    published_at: Optional[str] = None       # YYYY-MM-DD
    extracted_at: Optional[str] = None       # ISO timestamp, set at ingestion (NOT part of the id)

    def normalized(self) -> "AnalystCall":
        """Return a copy with tickers uppercased, ratings/actions canonicalized, dates ISO-formatted."""
        ticker = self.ticker.strip().upper().lstrip("$") if self.ticker else None
        return AnalystCall(
            ticker=ticker,
            company=self.company or None,
            analyst=self.analyst or None,
            firm=self.firm or None,
            rating=normalize_rating(self.rating) if self.rating else None,
            target_price=(float(self.target_price) if self.target_price not in (None, "") else None),
            previous_target=(float(self.previous_target) if self.previous_target not in (None, "") else None),
            action=normalize_action(self.action),
            source_url=self.source_url or None,
            published_at=_iso_date(self.published_at),
            extracted_at=self.extracted_at,
        )

    @property
    def id(self) -> str:
        """Deterministic identity hash (excludes extracted_at), so re-runs are idempotent."""
        key = "|".join(str(x) for x in (
            self.ticker, self.firm, self.rating, self.target_price, self.action, self.published_at))
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def ticker_is_valid(self) -> bool:
        return bool(self.ticker and _TICKER_RE.match(self.ticker))

    def to_record(self) -> dict:
        """The JSONL record: id first, then the schema fields in order."""
        rec = {"id": self.id}
        rec.update(asdict(self))
        return rec
