"""Generate the SAMPLE point-in-time news feed for the forecast calibration backtest.

    python data/sample_historical/_generate_news.py    # writes news.csv next to prices.csv

DELIBERATELY SYNTHETIC, CLEARLY LABELLED. Like the sample prices, the news is constructed with
known ground truth: for a subset of tickers each article's sentiment leans (noisily) toward that
stock's NEXT ~month move, so there is a real-but-weak signal for the backtest to find and the
recalibration layer to exploit. For the rest, sentiment is pure noise.

This forward-conditioning is DATASET CONSTRUCTION ONLY (exactly like the seeded prices). The engine
stays blind to the future: ``NewsWindow`` reads only events dated on/before the as_of date, so a
prediction can never see news from its own future. Each row is one dated, point-in-time article.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

OUT = __import__("pathlib").Path(__file__).resolve().parent
SEED = 24_681_012
FWD_WINDOW = 63          # the move (in trading days) the sentiment is conditioned on (~a quarter,
                         # aligned to the shorter prediction horizon so the signal is detectable)
SIGNAL_K = 2.2           # how strongly sentiment leans toward the forward move (signal tickers)
SIGNAL_NOISE = 0.30
NOISE_STD = 0.50         # pure-noise tickers
EVENT_GAP = 6            # ~ one article every 6 trading days, jittered

# Tickers whose news carries a (noisy) real signal vs. those that are pure noise.
SIGNAL_TICKERS = {"AVTX", "QBIT", "HLIX", "ORCA", "PYRA",   # outperformers
                  "KRNC", "LMNT",                            # decliners
                  "BRGE", "CALD", "DLTA", "EMBR", "FRTH"}    # laggers
BENCHMARK = "SPX"


def _rng_for(name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _headline(sym: str, sentiment: float) -> str:
    if sentiment > 0.25:
        return f"{sym}: upbeat coverage and positive analyst chatter"
    if sentiment < -0.25:
        return f"{sym}: cautious notes and downbeat sector commentary"
    return f"{sym}: mixed signals, no clear catalyst"


def build_news(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in prices.columns:
        if sym == BENCHMARK:
            continue
        s = prices[sym].dropna()
        if len(s) < FWD_WINDOW + 5:
            continue
        logp = np.log(s.to_numpy(dtype=float))
        idx = s.index
        rng = _rng_for(sym)
        is_signal = sym in SIGNAL_TICKERS

        i = int(rng.integers(5, EVENT_GAP + 5))
        while i < len(s) - 1:
            if is_signal and i + FWD_WINDOW < len(s):
                fwd = logp[i + FWD_WINDOW] - logp[i]                  # next-month move (ground truth)
                sentiment = SIGNAL_K * fwd + rng.normal(0.0, SIGNAL_NOISE)
            else:
                sentiment = rng.normal(0.0, NOISE_STD)
            sentiment = float(np.clip(sentiment, -1.0, 1.0))
            rows.append((idx[i].date().isoformat(), sym, round(sentiment, 4), _headline(sym, sentiment)))
            i += int(rng.integers(EVENT_GAP - 2, EVENT_GAP + 4))  # jittered gap to next article

    df = pd.DataFrame(rows, columns=["date", "symbol", "sentiment", "headline"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    prices = (
        pd.read_csv(OUT / "prices.csv", parse_dates=["date"])
        .pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )
    news = build_news(prices)
    news.to_csv(OUT / "news.csv", index=False)
    n_signal = news[news.symbol.isin(SIGNAL_TICKERS)].shape[0]
    print(f"Wrote {OUT / 'news.csv'}: {len(news)} articles across {news.symbol.nunique()} tickers "
          f"({n_signal} from signal-bearing names, {len(news) - n_signal} pure-noise).")
    print(f"Span {news.date.min()} → {news.date.max()}")


if __name__ == "__main__":
    main()
