"""A hand-labelled gold set of finance headlines for auditing the news-sentiment scorers.

~50 examples chosen to be REALISTIC and DELIBERATELY HARD, so the evaluation surfaces the genuine
gap between the offline word-list and an LLM read rather than a flattering diagonal. The hard
categories are exactly the ones a bag-of-words gets wrong:

  * negation        — "not a strong buy" (bearish), "no major concerns" (bullish)
  * relief idioms   — "demand concerns ease" (bullish), "posts narrower loss" (bullish)
  * sarcasm / tone  — "another 'great' quarter for shareholders" (bearish)
  * analyst-speak   — "initiated at Overweight" (bullish), "reiterates Sell" (bearish)
  * mixed clauses   — a positive and a negative clause where the NET direction is what matters

Ground truth is the sign a human reads for what the headline implies about the STOCK:
``"bull"`` (+), ``"bear"`` (−), or ``"neutral"`` (0).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentExample:
    text: str
    label: str          # "bull" | "bear" | "neutral"
    category: str       # what makes this case interesting (for the per-category breakdown)


GOLD: list[SentimentExample] = [
    # --- clean bullish ---
    SentimentExample("Apple beats earnings, shares surge to record high", "bull", "clean"),
    SentimentExample("Nvidia jumps as data-center revenue soars past estimates", "bull", "clean"),
    SentimentExample("Tesla rallies after strong delivery numbers", "bull", "clean"),
    SentimentExample("Amazon profit tops forecasts on cloud growth", "bull", "clean"),
    SentimentExample("Microsoft climbs to fresh high on AI momentum", "bull", "clean"),

    # --- clean bearish ---
    SentimentExample("Boeing plunges on widening losses and new probe", "bear", "clean"),
    SentimentExample("Intel sinks as guidance disappoints Wall Street", "bear", "clean"),
    SentimentExample("Netflix tumbles after subscriber miss", "bear", "clean"),
    SentimentExample("Ford drops on weak demand and recall costs", "bear", "clean"),
    SentimentExample("Meta slumps as ad revenue declines", "bear", "clean"),

    # --- neutral / factual ---
    SentimentExample("Apple to report fiscal third-quarter results on Thursday", "neutral", "neutral"),
    SentimentExample("Tesla board to meet next week on annual agenda", "neutral", "neutral"),
    SentimentExample("Nvidia announces 10-for-1 stock split effective in June", "neutral", "neutral"),
    SentimentExample("Amazon names new head of devices division", "neutral", "neutral"),
    SentimentExample("Microsoft schedules its annual developer conference for May", "neutral", "neutral"),

    # --- negation (the word-list's biggest blind spot) ---
    SentimentExample("Analysts say Apple is not a strong buy at these levels", "bear", "negation"),
    SentimentExample("No major concerns for Nvidia this quarter, says Morgan Stanley", "bull", "negation"),
    SentimentExample("Tesla fails to impress with latest delivery figures", "bear", "negation"),
    SentimentExample("Intel cannot escape weak PC demand, analyst warns", "bear", "negation"),
    SentimentExample("Boeing avoids a downgrade as cash flow stabilizes", "bull", "negation"),
    SentimentExample("Ford guidance was not as weak as feared", "bull", "negation"),

    # --- relief idioms (scary word, good news) ---
    SentimentExample("iPhone demand concerns ease as orders rebound", "bull", "relief"),
    SentimentExample("Nvidia eases supply fears with new capacity deal", "bull", "relief"),
    SentimentExample("Boeing posts narrower loss, shares climb", "bull", "relief"),
    SentimentExample("Recession worries shrug off as retail sales hold up", "bull", "relief"),
    SentimentExample("Selloff fears calm after Fed signals a pause", "bull", "relief"),

    # --- analyst-speak (rating/target language) ---
    SentimentExample("Goldman initiates Apple at Overweight, $260 target", "bull", "analyst"),
    SentimentExample("Morgan Stanley reiterates Sell on Lucid, cuts target to $2", "bear", "analyst"),
    SentimentExample("Barclays upgrades Amazon to Buy from Hold", "bull", "analyst"),
    SentimentExample("Citi downgrades Exxon to Neutral, trims price target", "bear", "analyst"),
    SentimentExample("Wedbush raises Tesla price target to $400, keeps Outperform", "bull", "analyst"),
    SentimentExample("UBS lowers Intel to Sell, sees further downside", "bear", "analyst"),

    # --- sarcasm / tone (hard for any lexicon) ---
    SentimentExample("Another 'great' quarter for Boeing shareholders as losses mount", "bear", "sarcasm"),
    SentimentExample("Intel's 'turnaround' delivers yet another guidance cut", "bear", "sarcasm"),
    SentimentExample("So much for the AI boom: Nvidia warns on China sales", "bear", "sarcasm"),

    # --- mixed clauses (net direction matters) ---
    SentimentExample("Apple beats on revenue but warns of slowing iPhone sales", "bear", "mixed"),
    SentimentExample("Tesla misses on margins yet reaffirms aggressive growth outlook", "bull", "mixed"),
    SentimentExample("Netflix adds subscribers but profit falls on content spend", "bear", "mixed"),
    SentimentExample("Amazon revenue light, but cloud margins expand sharply", "bull", "mixed"),

    # --- subtle / single-signal ---
    SentimentExample("Apple supplier flags soft orders for the next quarter", "bear", "subtle"),
    SentimentExample("Nvidia lands a major hyperscaler contract", "bull", "subtle"),
    SentimentExample("Tesla recalls 1.2 million vehicles over software glitch", "bear", "subtle"),
    SentimentExample("Microsoft secures multi-year government cloud deal", "bull", "subtle"),
    SentimentExample("Boeing wins fresh orders at the Paris Air Show", "bull", "subtle"),
    SentimentExample("Ford halts F-150 production amid parts shortage", "bear", "subtle"),

    # --- more clean, to balance the set ---
    SentimentExample("Coinbase soars as crypto volumes rebound", "bull", "clean"),
    SentimentExample("Disney falls on streaming losses", "bear", "clean"),
    SentimentExample("Pfizer gains on upbeat drug-trial results", "bull", "clean"),
    SentimentExample("Starbucks warns on weak China traffic", "bear", "clean"),
    SentimentExample("Walmart raises full-year outlook", "bull", "clean"),
]
