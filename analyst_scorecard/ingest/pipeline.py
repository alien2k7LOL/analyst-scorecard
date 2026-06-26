"""The three-stage pipeline: Discovery → Extraction → Validation → JSONL.

    Discovery   — each source adapter finds raw analyst calls
    Extraction  — adapters already structure the fields (RSS reuses intel.extract)
    Validation  — normalize, drop the invalid, dedup, append only NEW ids to data/analyst_calls.jsonl

Idempotent + deterministic: the id excludes the ingestion timestamp, so re-running on the same
sources adds nothing, and new rows are written in a stable sorted order.

    python -m analyst_scorecard.ingest.pipeline --tickers AAPL,MSFT,NVDA
    python -m analyst_scorecard.ingest.pipeline --rss https://feed1,https://feed2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .schema import AnalystCall
from .sources.base import SourceAdapter
from .validate import dedup, is_valid

DEFAULT_OUT = "data/analyst_calls.jsonl"


@dataclass
class IngestResult:
    discovered: int
    valid: int
    invalid: int
    new: int
    duplicates_skipped: int
    written_to: str

    def summary(self) -> str:
        return (f"discovered {self.discovered} · valid {self.valid} · invalid {self.invalid} · "
                f"new {self.new} · already-stored {self.duplicates_skipped} → {self.written_to}")


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _append_jsonl(path: Path, calls: list[AnalystCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")


def run_ingest(sources: Sequence[SourceAdapter], out_path: str | Path = DEFAULT_OUT) -> IngestResult:
    out = Path(out_path)
    raw: list[AnalystCall] = []
    for s in sources:
        raw.extend(s.discover())

    normalized = [c.normalized() for c in raw]
    valid = [c for c in normalized if is_valid(c)[0]]
    deduped = dedup(valid)

    existing = _load_existing_ids(out)
    new_calls = [c for c in deduped if c.id not in existing]
    # Deterministic write order (stable across runs).
    new_calls.sort(key=lambda c: (c.published_at or "9999-99-99", c.ticker or "", c.firm or "", c.id))
    _append_jsonl(out, new_calls)

    return IngestResult(
        discovered=len(raw), valid=len(valid), invalid=len(normalized) - len(valid),
        new=len(new_calls), duplicates_skipped=len(deduped) - len(new_calls), written_to=str(out),
    )


def build_sources(tickers: Optional[Sequence[str]], rss_urls: Optional[Sequence[str]]) -> list[SourceAdapter]:
    sources: list[SourceAdapter] = []
    if tickers:
        from .sources.yfinance_ratings import YFinanceRatingsSource
        sources.append(YFinanceRatingsSource(tickers))
    if rss_urls:
        from .sources.rss import RssSource
        sources.append(RssSource(rss_urls))
    return sources


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest analyst calls into a JSONL dataset.")
    p.add_argument("--tickers", type=str, default="", help="comma-list for yfinance upgrades/downgrades")
    p.add_argument("--rss", type=str, default="", help="comma-list of RSS/Atom feed URLs")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    tickers = [t for t in args.tickers.split(",") if t.strip()]
    rss = [u for u in args.rss.split(",") if u.strip()]
    if not (tickers or rss):
        p.error("give --tickers and/or --rss")
    result = run_ingest(build_sources(tickers, rss), args.out)
    print(result.summary(), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
