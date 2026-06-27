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
    is_band = grade.kind == PredictionKind.TERMINAL and grade.band_pct
    if is_band:
        # A ±band 'be within' call is symmetric — show the unsigned distance to the band centre,
        # not a signed value derived from a direction the probability ignores.
        dist_sigma = abs(float(np.log(grade.target_price / s0))) / sd
        dist_label, dist_val = "Distance to band centre", f"{dist_sigma:.2f} σ"
    else:
        log_dist = float(np.log(grade.target_price / s0)) if up else float(np.log(s0 / grade.target_price))
        dist_sigma = log_dist / sd
        dist_label, dist_val = "Distance to target", f"{dist_sigma:+.2f} σ"

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
        MathFactor(dist_label, dist_val,
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


# Explicit word sets (inflections included) rather than stems — avoids false hits like "mission"→miss
# or "topic"→top. Deliberately NO bare "fall" (the season: "fall event" is not a price drop) or bare
# "sell" ("sells record units" isn't bearish); the directional inflections carry those cases.
_POS = {"beat", "beats", "beating", "surge", "surges", "surged", "soar", "soars", "soared", "jump",
        "jumps", "jumped", "rally", "rallies", "rallied", "record", "growth", "profit", "profits",
        "gains", "gain", "bullish", "outperform", "outperforms", "outperformed", "upgrade", "upgraded",
        "upgrades", "raise", "raised", "raises", "strong", "strength", "tops", "topped", "wins",
        "approval", "approved", "expand", "expands", "rebound", "rebounds", "rebounded", "optimistic",
        "tailwind", "tailwinds", "climbs", "climbed", "accelerate", "accelerates", "boost", "boosts",
        "boosted", "momentum", "buy"}
_NEG = {"miss", "misses", "missed", "plunge", "plunges", "plunged", "sink", "sinks", "sank", "falls",
        "fell", "falling", "drop", "drops", "dropped", "decline", "declines", "declined", "loss",
        "losses", "weak", "weakness", "weakens", "bearish", "downgrade", "downgraded", "downgrades",
        "cut", "cuts", "slash", "slashes", "slashed", "lawsuit", "probe", "slump", "slumps", "slumped",
        "tumble", "tumbles", "tumbled", "warn", "warns", "warning", "recall", "fraud", "selloff",
        "sell-off", "headwind", "headwinds", "concern", "concerns", "disappoint", "disappoints",
        "disappointing", "plummet", "plummets", "plummeted", "halt", "halts"}

# Hard negators: a sentiment word in the few tokens AFTER one of these gets its polarity FLIPPED, so
# "not a strong buy" reads negative and "no major concerns" reads positive. This was the single biggest
# source of wrong signs — a bag-of-words read can't see that "not" reverses everything after it.
_NEGATORS = {"not", "no", "never", "without", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
             "weren't", "werent", "won't", "wont", "cannot", "can't", "cant", "fails", "failed",
             "failing", "lacks", "lacking", "avoids", "avoided"}
# Finance idioms where a scary word is actually GOOD news (relief) — read as a phrase so the bare
# negative word inside ("concerns", "loss") doesn't drag the sign the wrong way.
_RELIEF = {"ease", "eases", "eased", "easing", "allay", "allays", "allayed", "soothe", "soothes",
           "calm", "calms", "calmed", "shrug", "shrugs", "shrugged"}
_WORRY = {"concern", "concerns", "fear", "fears", "worry", "worries", "jitters", "doubt", "doubts",
          "selloff", "sell-off", "slump"}
_SHRINK = {"narrower", "narrowing", "narrows", "smaller", "slimmer", "shrinking", "shrinks"}


def score_sentiment(text: str) -> float:
    """Finance headline sentiment in [-1, 1], sign-aware: handles negation and a few relief idioms.

    Net = (#positive − #negative) / (#positive + #negative) over the lexicon hits, but BEFORE counting:
      * a hard negator ("not", "no", "fails to"…) in the preceding 3 tokens flips a word's polarity, and
      * relief idioms ("concerns ease", "ease fears", "narrower loss") are read as positive as a whole,
    so the common cases that made the old word-count give the wrong sign now come out right.
    """
    toks = [w.strip(".,!?:;'\"()[]—").lower() for w in (text or "").split()]
    toks = [t for t in toks if t]
    pos = neg = 0
    i, n = 0, len(toks)
    while i < n:
        w = toks[i]
        nxt = toks[i + 1] if i + 1 < n else ""
        # relief idioms (either word order) and "narrower loss" → positive; consume both tokens
        if (w in _RELIEF and nxt in _WORRY) or (w in _WORRY and nxt in _RELIEF) \
                or (w in _SHRINK and nxt in {"loss", "losses", "deficit"}):
            pos += 1
            i += 2
            continue
        polarity = 1 if w in _POS else (-1 if w in _NEG else 0)
        if polarity:
            if any(toks[j] in _NEGATORS or toks[j].endswith("n't") for j in range(max(0, i - 3), i)):
                polarity = -polarity            # a negator just before this word reverses it
            pos, neg = (pos + 1, neg) if polarity > 0 else (pos, neg + 1)
        i += 1
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
