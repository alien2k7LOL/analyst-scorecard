"""Proof for a forecast — the WHY behind the number, in three forms.

A probability with no reasoning is a black box. For every live grade we expose:

  * MATHEMATICAL — the exact ingredients (start price, per-bar drift/vol, horizon, how many standard
    deviations away the target is) and how they produce the raw closed-form number, plus the
    history-based calibration shift. ``build_math`` returns these as labelled, plain-English factors.
  * GRAPHICAL — the price cone + terminal distribution (see ``viz.plot_forecast_proof``).
  * NEWS — recent headlines for the ticker with a light sentiment read, as supporting CONTEXT (it is
    not fed into the live probability — that would risk look-ahead in the self-calibration — but it
    is exactly what a human checks before trusting a forecast).

Everything network-touching (headlines) goes through an injectable seam so the whole module is
testable offline; failures degrade to "no headlines", never to a crash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .prediction import PredictionKind


# --------------------------------------------------------------------------------------
# Mathematical proof
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MathFactor:
    label: str
    value: str
    meaning: str


def build_math(grade) -> list[MathFactor]:
    """The labelled ingredients of the probability, straight from the GBM model the grade used."""
    s0, mu, sigma, n = grade.s0, grade.drift_bar, max(grade.vol_bar, 1e-9), int(grade.n_days)
    unit = grade.interval.label
    sd = sigma * np.sqrt(n)                       # terminal log-vol over the horizon
    median = s0 * float(np.exp(mu * n))           # middle of the cone at the deadline
    up = grade.direction.value == "up"
    log_dist = float(np.log(grade.target_price / s0)) if up else float(np.log(s0 / grade.target_price))
    dist_sigma = log_dist / sd

    factors = [
        MathFactor("Start price (S₀)", f"${s0:,.2f}", "where the stock sits right now"),
        MathFactor(f"Drift per {unit} (μ)", f"{mu*100:+.3f}%",
                   "the trend the model reads from recent history"),
        MathFactor(f"Volatility per {unit} (σ)", f"{sigma*100:.3f}%",
                   "how far it typically moves each bar"),
        MathFactor("Horizon (T)", f"{n} {grade.interval.horizon_word}",
                   "how many bars from now until the deadline"),
        MathFactor("Projected median price", f"${median:,.2f}",
                   "S₀·e^(μ·T) — the centre of the cone at the deadline"),
        MathFactor("Distance to target", f"{dist_sigma:+.2f} σ",
                   "how many standard deviations away the target is (smaller = easier)"),
        MathFactor("Raw model probability", f"{grade.raw_probability*100:.1f}%",
                   "straight from the closed-form formula below"),
        MathFactor("Calibrated probability", f"{grade.probability*100:.1f}%",
                   "after correcting by how this stock has actually behaved"),
    ]
    return factors


def formula_text(grade) -> str:
    """The closed-form expression behind the raw number, as a readable one-liner."""
    if grade.kind == PredictionKind.TERMINAL:
        if grade.band_pct:
            return ("Terminal (band):  P(K·(1−b) ≤ S_T ≤ K·(1+b))  =  Φ(z_hi) − Φ(z_lo),   "
                    "z = (ln(level/S₀) − μ·T) / (σ·√T)")
        return ("Terminal (at/through):  P(S_T ≥ K)  =  Φ( (μ·T − ln(K/S₀)) / (σ·√T) )   "
                "(UP; DOWN is the mirror)")
    return ("Touch (first passage):  P(max S ≥ K)  =  Φ((μT−b)/σ√T) + e^{2μb/σ²}·Φ((−μT−b)/σ√T),   "
            "b = ln(K/S₀)")


# --------------------------------------------------------------------------------------
# News proof (supporting context — injectable, offline-safe)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsHeadline:
    title: str
    publisher: str
    when: Optional[str]
    url: Optional[str]
    sentiment: float            # naive lexicon read in [-1, 1]

    @property
    def mood(self) -> str:
        return "🟢 positive" if self.sentiment > 0.15 else "🔴 negative" if self.sentiment < -0.15 else "⚪ neutral"


_POS = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "record",
        "growth", "profit", "gains", "gain", "bullish", "outperform", "upgrade", "raise", "raised",
        "strong", "tops", "wins", "approval", "expand", "expands", "rebound", "buy"}
_NEG = {"miss", "misses", "plunge", "plunges", "sink", "sinks", "fall", "falls", "drop", "drops",
        "loss", "losses", "weak", "bearish", "downgrade", "cut", "cuts", "lawsuit", "probe",
        "slump", "warns", "warning", "slashes", "recall", "halt", "fraud", "selloff", "sell"}


def score_sentiment(text: str) -> float:
    """Tiny finance lexicon read of a headline: (pos − neg) / (pos + neg), in [-1, 1]."""
    words = [w.strip(".,!?:;'\"()").lower() for w in (text or "").split()]
    pos = sum(w in _POS for w in words)
    neg = sum(w in _NEG for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def classify_support(sentiment: float, direction) -> str:
    """Does a headline's sentiment SUPPORT or CONTRADICT the call's direction? (the point of news here)

    For an UP call, bullish news supports and bearish contradicts; for a DOWN call it's mirrored.
    This is what makes news back up the GRADE, rather than being abstract point-in-time context.
    """
    is_up = str(getattr(direction, "value", direction)).lower() == "up"
    if sentiment > 0.15:
        return "supports" if is_up else "contradicts"
    if sentiment < -0.15:
        return "contradicts" if is_up else "supports"
    return "neutral"


def news_lean(headlines: list["NewsHeadline"], direction) -> dict:
    """Tally how recent headlines line up with the call's direction."""
    labels = [classify_support(h.sentiment, direction) for h in headlines]
    return {"supports": labels.count("supports"), "contradicts": labels.count("contradicts"),
            "neutral": labels.count("neutral"), "labels": labels}


class HeadlineFetcher(ABC):
    @abstractmethod
    def fetch(self, ticker: str, limit: int = 6) -> list[dict]: ...


class YFinanceHeadlineFetcher(HeadlineFetcher):
    """Recent headlines via yfinance (lazy import; tolerant of its shifting JSON shapes)."""

    def fetch(self, ticker: str, limit: int = 6) -> list[dict]:
        import yfinance as yf  # lazy
        items = getattr(yf.Ticker(ticker), "news", None) or []
        return list(items)[:limit]


def _field(item: dict, *names, default=None):
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    for n in names:
        if item.get(n):
            return item[n]
        if content.get(n):
            return content[n]
    return default


def recent_headlines(ticker: str, fetcher: Optional[HeadlineFetcher] = None,
                     limit: int = 6) -> list[NewsHeadline]:
    """Recent headlines + a light sentiment read. Never raises — returns [] if news is unavailable."""
    fetcher = fetcher or YFinanceHeadlineFetcher()
    try:
        raw = fetcher.fetch(ticker, limit)
    except Exception:
        return []
    out: list[NewsHeadline] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        title = _field(item, "title", default="") or ""
        if not title:
            continue
        pub = _field(item, "publisher", "provider", default="") or ""
        if isinstance(pub, dict):
            pub = pub.get("displayName") or pub.get("name") or ""
        url = _field(item, "link", "canonicalUrl", "clickThroughUrl")
        if isinstance(url, dict):
            url = url.get("url")
        when = _field(item, "providerPublishTime", "pubDate", "displayTime")
        out.append(NewsHeadline(title=str(title), publisher=str(pub), when=str(when) if when else None,
                                url=url if isinstance(url, str) else None, sentiment=score_sentiment(title)))
    return out
