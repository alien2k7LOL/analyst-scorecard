"""Validation + deduplication — the gate between discovery and storage.

Rules (per spec): require a valid uppercase ticker, a firm, and at least a rating OR a target. Never
let a half-formed call through, and never store the same call twice — when the same identity appears
more than once, keep the earliest publication.
"""

from __future__ import annotations

from .schema import ACTIONS, AnalystCall


def is_valid(call: AnalystCall) -> tuple[bool, str]:
    """Validate an already-``normalized()`` call. Returns (ok, reason-if-not)."""
    if not call.ticker_is_valid():
        return False, "missing/invalid ticker"
    if not call.firm:
        return False, "missing firm"
    if not (call.rating or call.target_price is not None):
        return False, "no rating and no target"
    if call.action is not None and call.action not in ACTIONS:
        return False, f"bad action {call.action!r}"
    return True, ""


def _earlier(a: str | None, b: str | None) -> bool:
    """True if publication date ``a`` is strictly earlier than ``b`` (None counts as 'latest')."""
    if a is None:
        return False
    if b is None:
        return True
    return a < b


def dedup(calls: list[AnalystCall]) -> list[AnalystCall]:
    """Collapse duplicate identities (same ``id``), keeping the earliest publication of each."""
    by_id: dict[str, AnalystCall] = {}
    for c in calls:
        existing = by_id.get(c.id)
        if existing is None or _earlier(c.published_at, existing.published_at):
            by_id[c.id] = c
    return list(by_id.values())
