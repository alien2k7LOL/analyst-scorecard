"""Audit the news-sentiment scorers against the hand-labelled gold set.

Turns "the LLM should be better" into a number: sign-accuracy (bull / bear / neutral) overall and
per hard category, for the offline lexicon and — when ``ANTHROPIC_API_KEY`` is set — the LLM scorer,
side by side. Run with ``python -m evaluation.sentiment_eval``.

Everything is offline-safe: with no key, only the lexicon column is produced (the LLM column is
reported as skipped), so the suite and CI never need the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from analyst_scorecard.forecast.explanation import (
    LexiconSentimentScorer,
    LLMSentimentScorer,
    SentimentScorer,
)
from evaluation.sentiment_gold import GOLD, SentimentExample

LABELS = ["bull", "bear", "neutral"]
BAND = 0.15            # same dead-zone classify_support uses, so eval matches what the app shows
FIG_DIR = Path(__file__).parent / "figures"


def sign(score: float, band: float = BAND) -> str:
    """Map a [-1, 1] sentiment score to a label, with a neutral dead-zone around 0."""
    return "bull" if score > band else "bear" if score < -band else "neutral"


@dataclass(frozen=True)
class Scored:
    example: SentimentExample
    score: float
    predicted: str

    @property
    def correct(self) -> bool:
        return self.predicted == self.example.label


def evaluate(scorer: SentimentScorer, gold: list[SentimentExample] = GOLD) -> list[Scored]:
    """Score the whole gold set in one batched call and label each by sign."""
    scores = scorer.score_many([ex.text for ex in gold])
    return [Scored(ex, s, sign(s)) for ex, s in zip(gold, scores)]


def accuracy(rows: list[Scored]) -> float:
    return sum(r.correct for r in rows) / len(rows) if rows else 0.0


def confusion_matrix(rows: list[Scored]) -> pd.DataFrame:
    """Rows = true label, cols = predicted label, over {bull, bear, neutral}."""
    cm = pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=int)
    for r in rows:
        cm.loc[r.example.label, r.predicted] += 1
    cm.index.name, cm.columns.name = "true", "predicted"
    return cm


def per_category(rows: list[Scored]) -> pd.DataFrame:
    """Accuracy broken out by the gold set's hard categories — where the lexicon/LLM gap shows."""
    cats = sorted({r.example.category for r in rows})
    data = []
    for c in cats:
        sub = [r for r in rows if r.example.category == c]
        data.append({"category": c, "n": len(sub), "accuracy": round(accuracy(sub), 3)})
    return pd.DataFrame(data).set_index("category")


def sign_error_rate(rows: list[Scored]) -> float:
    """Fraction where the scorer got the SIGN actively wrong (bull↔bear) — the costly errors.

    Neutral-vs-directional confusions are excluded; calling a bull 'bear' is what misleads a trader.
    """
    flips = sum(
        1 for r in rows
        if {r.example.label, r.predicted} == {"bull", "bear"}
    )
    return flips / len(rows) if rows else 0.0


def compare(gold: list[SentimentExample] = GOLD) -> dict:
    """Lexicon always; LLM only when a key is present. Returns a dict of named result tables."""
    out: dict = {}
    lex_rows = evaluate(LexiconSentimentScorer(), gold)
    out["lexicon"] = {
        "rows": lex_rows, "accuracy": accuracy(lex_rows),
        "sign_error_rate": sign_error_rate(lex_rows),
        "confusion": confusion_matrix(lex_rows), "by_category": per_category(lex_rows),
    }
    if os.environ.get("ANTHROPIC_API_KEY"):
        llm_rows = evaluate(LLMSentimentScorer(), gold)
        out["llm"] = {
            "rows": llm_rows, "accuracy": accuracy(llm_rows),
            "sign_error_rate": sign_error_rate(llm_rows),
            "confusion": confusion_matrix(llm_rows), "by_category": per_category(llm_rows),
        }
    return out


def _print_block(name: str, res: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"overall sign-accuracy : {res['accuracy']:.1%}  ({len(res['rows'])} headlines)")
    print(f"bull<->bear flip rate : {res['sign_error_rate']:.1%}  (the costly errors)")
    print("\nby category:")
    print(res["by_category"].to_string())
    print("\nconfusion (rows=true, cols=predicted):")
    print(res["confusion"].to_string())


REPORT_PATH = Path(__file__).parent / "SENTIMENT_REPORT.md"


def _md_block(name: str, res: dict) -> str:
    lines = [f"### {name}", "",
             f"- **Overall sign-accuracy:** {res['accuracy']:.1%} ({len(res['rows'])} headlines)",
             f"- **Bull↔bear flip rate:** {res['sign_error_rate']:.1%} (the costly errors)", "",
             "**By category:**", "", "| category | n | accuracy |", "|---|---|---|"]
    bc = res["by_category"]
    for cat, row in bc.iterrows():
        lines.append(f"| {cat} | {int(row['n'])} | {row['accuracy']:.0%} |")
    lines += ["", "**Confusion (rows = true, cols = predicted):**", "",
              "| true ╲ pred | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    cm = res["confusion"]
    for t in LABELS:
        lines.append(f"| {t} | " + " | ".join(str(int(cm.loc[t, p])) for p in LABELS) + " |")
    return "\n".join(lines)


def write_report(path: Path = REPORT_PATH, results: Optional[dict] = None) -> Path:
    """Render the eval to a Markdown report (regenerable artifact, like REPORT.md/FORECAST_REPORT.md)."""
    from analyst_scorecard.forecast.explanation import DEFAULT_SENTIMENT_MODEL

    results = results or compare()
    parts = ["# News-sentiment scorer evaluation", "",
             "Sign-accuracy of the news-sentiment scorers on the hand-labelled hard gold set "
             f"(`evaluation/sentiment_gold.py`, {len(GOLD)} headlines). Regenerate with "
             "`python -m evaluation.sentiment_eval --report`.", "",
             _md_block("Lexicon (offline word-list)", results["lexicon"])]
    if "llm" in results:
        parts += ["", _md_block(f"LLM ({DEFAULT_SENTIMENT_MODEL})", results["llm"])]
        d = results["llm"]["accuracy"] - results["lexicon"]["accuracy"]
        parts += ["", f"**LLM − lexicon accuracy delta: {d:+.1%}**"]
    else:
        parts += ["", "_LLM column skipped — set `ANTHROPIC_API_KEY` to score the LLM scorer too._"]
    path.write_text("\n".join(parts) + "\n")
    return path


def main() -> int:
    import sys
    from analyst_scorecard.forecast.explanation import DEFAULT_SENTIMENT_MODEL

    results = compare()
    _print_block("LEXICON (offline word-list)", results["lexicon"])
    if "llm" in results:
        _print_block(f"LLM ({DEFAULT_SENTIMENT_MODEL})", results["llm"])
        d = results["llm"]["accuracy"] - results["lexicon"]["accuracy"]
        print(f"\nLLM − lexicon accuracy delta: {d:+.1%}")
        print("Tip: set SCORECARD_SENTIMENT_MODEL=<id> (e.g. a Sonnet/Opus id) and re-run to compare.")
    else:
        print("\n(LLM column skipped — set ANTHROPIC_API_KEY to score the LLM scorer too.)")
    if "--report" in sys.argv:
        print(f"\nWrote {write_report(results=results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
