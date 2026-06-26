"""Public RSS/Atom feeds — the ToS-compliant news channel, parsed by the existing intel extractor.

For each feed entry we run ``title + summary`` through ``intel.extract`` (the same heuristic+LLM
parser used elsewhere) and keep only items that look like a real analyst call (firm + ticker +
rating or target). The HTTP fetcher is injectable, so this is fully testable offline with fixtures.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..schema import AnalystCall, detect_action
from .base import SourceAdapter, _now_iso


class FeedFetcher(ABC):
    @abstractmethod
    def fetch(self, url: str) -> str:  # -> raw RSS/Atom XML text
        ...


class UrllibFeedFetcher(FeedFetcher):
    """Stdlib HTTP fetcher (no extra dependency). RSS is public and built for machine consumption."""

    def fetch(self, url: str) -> str:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AnalystScorecard ingest)"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - user-configured public feed
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read(3_000_000).decode(charset, errors="replace")


def _localname(tag: str) -> str:
    return tag.split("}")[-1].lower()


def parse_feed(xml_text: str) -> list[dict]:
    """Parse RSS 2.0 or Atom into a list of {title, summary, link, published}. Namespace-agnostic."""
    root = ET.fromstring(xml_text)
    entries: list[dict] = []
    for el in root.iter():
        if _localname(el.tag) not in ("item", "entry"):
            continue
        title = summary = link = published = None
        for child in el:
            ln = _localname(child.tag)
            if ln == "title":
                title = (child.text or "").strip()
            elif ln in ("description", "summary", "content"):
                summary = (child.text or "").strip()
            elif ln == "link":
                link = child.get("href") or (child.text or "").strip() or None
            elif ln in ("pubdate", "published", "updated", "date"):
                published = (child.text or "").strip() or None
        entries.append({"title": title or "", "summary": summary or "", "link": link, "published": published})
    return entries


class RssSource(SourceAdapter):
    name = "rss"

    def __init__(self, feed_urls: Sequence[str], fetcher: Optional[FeedFetcher] = None,
                 now: Optional[str] = None):
        self.feed_urls = list(feed_urls)
        self.fetcher = fetcher or UrllibFeedFetcher()
        self.now = now

    def discover(self) -> list[AnalystCall]:
        from ...intel.extract import extract_recommendation  # reuse the existing extractor
        stamp = _now_iso(self.now)
        out: list[AnalystCall] = []
        for url in self.feed_urls:
            try:
                xml = self.fetcher.fetch(url)
                entries = parse_feed(xml)
            except Exception:  # a broken feed must never sink the run
                continue
            for e in entries:
                text = f"{e['title']} {e['summary']}".strip()
                rec = extract_recommendation(text, use_llm=False)
                # Only emit something that looks like a real call (firm + ticker + a rating OR a target).
                if not (rec.ticker and rec.firm and (rec.rating or rec.target_price)):
                    continue
                out.append(AnalystCall(
                    ticker=rec.ticker, analyst=rec.analyst, firm=rec.firm, rating=rec.rating,
                    target_price=rec.target_price, action=detect_action(text), source_url=e["link"],
                    published_at=(e["published"] or rec.publication_date), extracted_at=stamp,
                ))
        return out
