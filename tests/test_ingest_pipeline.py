"""Ingestion pipeline — schema, source adapters (offline fixtures), validation, dedup, idempotent JSONL."""

import json

import pandas as pd
import pytest

from analyst_scorecard.ingest.pipeline import run_ingest
from analyst_scorecard.ingest.schema import AnalystCall, detect_action, normalize_action
from analyst_scorecard.ingest.sources.rss import FeedFetcher, RssSource, parse_feed
from analyst_scorecard.ingest.sources.yfinance_ratings import RatingsFetcher, YFinanceRatingsSource
from analyst_scorecard.ingest.validate import dedup, is_valid

NOW = "2026-06-26T12:00:00"


# ---- fixtures: offline fetchers ----------------------------------------------------------


class FakeRatings(RatingsFetcher):
    def __init__(self, by_ticker):
        self.by_ticker = by_ticker

    def upgrades_downgrades(self, ticker):
        return self.by_ticker.get(ticker)


class FakeFeed(FeedFetcher):
    def __init__(self, xml):
        self.xml = xml

    def fetch(self, url):
        return self.xml


def _ratings_df():
    df = pd.DataFrame(
        {"Firm": ["Morgan Stanley", "Citi"], "ToGrade": ["Overweight", "Neutral"],
         "FromGrade": ["Equal-Weight", "Buy"], "Action": ["up", "down"]},
        index=pd.to_datetime(["2025-01-10", "2025-02-05"]),
    )
    df.index.name = "GradeDate"
    return df


_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Wedbush raises Apple (AAPL) price target to $300, reiterates Buy</title>
    <description>Analyst Dan Ives note</description>
    <link>https://example.com/a</link>
    <pubDate>Mon, 13 Jan 2025 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Markets rally as Fed signals rate cuts</title>
    <description>macro commentary, no analyst action</description>
    <link>https://example.com/b</link>
    <pubDate>Mon, 13 Jan 2025 11:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


# ---- schema --------------------------------------------------------------------------------


def test_schema_normalizes_and_hashes_deterministically():
    c = AnalystCall(ticker="aapl", firm="Citi", rating="Outperform", action="up",
                    published_at="2025-01-10", extracted_at="t1").normalized()
    assert c.ticker == "AAPL" and c.rating == "Overweight" and c.action == "upgrade"
    # id excludes extracted_at -> re-extracting the same call is idempotent
    c2 = AnalystCall(ticker="AAPL", firm="Citi", rating="Overweight", action="upgrade",
                     published_at="2025-01-10", extracted_at="DIFFERENT").normalized()
    assert c.id == c2.id


def test_action_normalization_and_detection():
    assert normalize_action("Downgrades") == "downgrade"
    assert normalize_action("reit") == "reiteration"
    assert detect_action("Goldman initiates coverage on NVDA") == "initiation"
    assert detect_action("Citi downgrades Boeing") == "downgrade"


def test_validation_requires_ticker_firm_and_rating_or_target():
    ok = AnalystCall(ticker="AAPL", firm="Citi", rating="Buy").normalized()
    assert is_valid(ok)[0] is True
    assert is_valid(AnalystCall(ticker="AAPL", rating="Buy").normalized())[0] is False      # no firm
    assert is_valid(AnalystCall(ticker="AAPL", firm="Citi").normalized())[0] is False        # no rating/target
    assert is_valid(AnalystCall(firm="Citi", rating="Buy").normalized())[0] is False         # no ticker
    assert is_valid(AnalystCall(ticker="AAPL", firm="Citi", target_price=300.0).normalized())[0] is True


# ---- sources -------------------------------------------------------------------------------


def test_yfinance_source_maps_rows_to_calls():
    src = YFinanceRatingsSource(["AAPL"], fetcher=FakeRatings({"AAPL": _ratings_df()}), now=NOW)
    calls = [c.normalized() for c in src.discover()]
    assert len(calls) == 2
    ms = next(c for c in calls if c.firm == "Morgan Stanley")
    assert ms.ticker == "AAPL" and ms.rating == "Overweight" and ms.action == "upgrade"
    assert ms.published_at == "2025-01-10" and ms.target_price is None     # no targets in this feed
    citi = next(c for c in calls if c.firm == "Citi")
    assert citi.rating == "Hold" and citi.action == "downgrade"            # Neutral -> Hold


def test_rss_source_extracts_only_real_calls():
    src = RssSource(["http://feed"], fetcher=FakeFeed(_RSS), now=NOW)
    calls = [c.normalized() for c in src.discover()]
    assert len(calls) == 1                                                 # the macro item is ignored
    c = calls[0]
    assert c.ticker == "AAPL" and c.firm == "Wedbush" and c.rating == "Buy"
    assert c.target_price == 300.0 and c.action == "reiteration"
    assert c.source_url == "https://example.com/a"


def test_parse_feed_handles_atom():
    atom = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>x</title>'
            '<summary>y</summary><link href="http://z"/><updated>2025-01-01</updated></entry></feed>')
    entries = parse_feed(atom)
    assert entries[0]["title"] == "x" and entries[0]["link"] == "http://z"


# ---- dedup + end-to-end pipeline -----------------------------------------------------------


def test_dedup_collapses_same_event_from_multiple_sources():
    # Same event (same firm/rating/date) reported by two feeds -> one canonical row.
    a = AnalystCall(ticker="AAPL", firm="Citi", rating="Buy", published_at="2025-01-15",
                    source_url="cnbc", extracted_at="t1").normalized()
    b = AnalystCall(ticker="AAPL", firm="Citi", rating="Buy", published_at="2025-01-15",
                    source_url="yahoo", extracted_at="t2").normalized()
    assert a.id == b.id
    assert len(dedup([a, b])) == 1


def test_dedup_keeps_distinct_events_on_different_dates():
    # A re-rating on a different date is a distinct event, not a duplicate.
    a = AnalystCall(ticker="AAPL", firm="Citi", rating="Buy", published_at="2025-01-15").normalized()
    b = AnalystCall(ticker="AAPL", firm="Citi", rating="Buy", published_at="2025-02-01").normalized()
    assert a.id != b.id
    assert len(dedup([a, b])) == 2


def test_pipeline_writes_jsonl_and_is_idempotent(tmp_path):
    out = tmp_path / "calls.jsonl"
    sources = [
        YFinanceRatingsSource(["AAPL"], fetcher=FakeRatings({"AAPL": _ratings_df()}), now=NOW),
        RssSource(["http://feed"], fetcher=FakeFeed(_RSS), now=NOW),
    ]
    r1 = run_ingest(sources, out)
    assert r1.discovered == 3 and r1.valid == 3 and r1.new == 3
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert set(rec) == {"id", "ticker", "company", "analyst", "firm", "rating", "target_price",
                        "previous_target", "action", "source_url", "published_at", "extracted_at"}

    # Re-running on the same sources (even with a different timestamp) adds nothing.
    sources2 = [
        YFinanceRatingsSource(["AAPL"], fetcher=FakeRatings({"AAPL": _ratings_df()}), now="2099-01-01T00:00:00"),
        RssSource(["http://feed"], fetcher=FakeFeed(_RSS), now="2099-01-01T00:00:00"),
    ]
    r2 = run_ingest(sources2, out)
    assert r2.new == 0 and r2.duplicates_skipped == 3
    assert len(out.read_text().splitlines()) == 3        # file unchanged
