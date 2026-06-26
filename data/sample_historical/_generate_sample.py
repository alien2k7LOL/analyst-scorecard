"""Generate the SAMPLE historical dataset for the Analyst Scorecard back-test.

    python data/sample_historical/_generate_sample.py

DELIBERATELY SYNTHETIC, CLEARLY LABELLED SAMPLE DATA. Prices and calls are generated (seeded,
reproducible) to look like real historical data and to exercise the back-test end to end offline.
Tickers, firms, and analyst names are FICTIONAL — nothing here reflects any real security or any
real analyst's performance. Replace prices.csv / calls.csv with your own real files (same schema)
to grade real analysts; see this folder's README.md.

The dataset is conditioned on the realized synthetic future (world-building) so the back-test has
known ground truth: a perma-bull who rode a bull market (high direction, beat <= 0), a genuinely
skilled picker (beat > 0), a delisted ticker (a call that must be skipped), and a revised target.
This conditioning is dataset construction only — the scoring engine remains blind to the future.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SEED = 7_654_321
BENCHMARK = "SPX"
DELIST_DATE = pd.Timestamp("2020-06-15")
HORIZON_TD = 252          # 12-month default horizon, in trading days
BAND = 0.02

# Annualized (drift, vol, start). The benchmark rises (a bull market).
BENCH_SPEC = (BENCHMARK, 0.09, 0.15, 2400.0)
TICKER_SPECS = {
    # Outperformers — for the skilled picker's longs (drift well above the index)
    "AVTX": (0.28, 0.45, 30.0), "QBIT": (0.24, 0.40, 55.0), "HLIX": (0.20, 0.34, 80.0),
    "ORCA": (0.18, 0.30, 120.0), "PYRA": (0.16, 0.28, 65.0),
    # Index-laggers — rise in the bull market but lag it (the perma-bull's basket)
    "BRGE": (0.05, 0.18, 90.0), "CALD": (0.04, 0.16, 140.0), "DLTA": (0.06, 0.20, 45.0),
    "EMBR": (0.05, 0.22, 38.0), "FRTH": (0.03, 0.15, 210.0),
    # Middling, ~market
    "GNRC": (0.09, 0.20, 70.0), "HVNS": (0.08, 0.18, 110.0), "JADE": (0.10, 0.24, 52.0),
    # Decliners — for the skilled picker's shorts
    "KRNC": (-0.11, 0.36, 60.0), "LMNT": (-0.07, 0.30, 48.0),
    # Choppy / trendless
    "MVRC": (0.00, 0.40, 22.0),
    # Delists mid-span (a blow-up) — exercises the skip policy
    "HALT": (-0.22, 0.55, 25.0),
}
OUTPERFORMERS = ["AVTX", "QBIT", "HLIX", "ORCA", "PYRA"]
LAGGERS = ["BRGE", "CALD", "DLTA", "EMBR", "FRTH"]
DECLINERS = ["KRNC", "LMNT"]


def _gbm(rng, n, drift, vol, start, dt=1 / 252):
    z = rng.standard_normal(n - 1)
    steps = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    return start * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def _rng_for(name: str) -> np.random.Generator:
    # hashlib (NOT builtin hash, which is PYTHONHASHSEED-salted) -> deterministic across processes.
    digest = hashlib.sha256(f"{SEED}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def build_prices() -> pd.DataFrame:
    days = pd.bdate_range(start="2017-01-03", periods=1800)  # ~7 trading years
    cols = {}
    sym, dr, vol, start = BENCH_SPEC
    cols[sym] = _gbm(_rng_for(sym), len(days), dr, vol, start)
    for sym, (dr, vol, start) in TICKER_SPECS.items():
        cols[sym] = _gbm(_rng_for(sym), len(days), dr, vol, start)
    frame = pd.DataFrame(cols, index=days)
    # HALT delists: drop all rows on/after the delist date (no data thereafter).
    frame.loc[frame.index >= DELIST_DATE, "HALT"] = np.nan
    return frame


def _outcome(frame, ticker, call_ts, res_ts):
    p0, p1 = frame.at[call_ts, ticker], frame.at[res_ts, ticker]
    b0, b1 = frame.at[call_ts, BENCHMARK], frame.at[res_ts, BENCHMARK]
    return p0, p1, p1 / p0 - 1.0, b1 / b0 - 1.0


def _long_beats(r, b, m):  return r > BAND and (r - b) > m
def _long_lags(r, b, m):   return r > BAND and (r - b) < -m
def _short_beats(r, b, m): return r < -BAND and (-r - b) > m


def _pick(frame, days, rng, tickers, predicate, n, margin):
    last_start = len(days) - HORIZON_TD - 1
    cands = [(t, days[i]) for t in tickers for i in range(last_start)]
    order = rng.permutation(len(cands))
    out = []
    used = set()
    for k in order:
        if len(out) >= n:
            break
        t, d = cands[int(k)]
        if (t, d) in used:
            continue
        res = days[days.get_indexer([d])[0] + HORIZON_TD]
        if pd.isna(frame.at[d, t]) or pd.isna(frame.at[res, t]):
            continue
        p0, p1, r, b = _outcome(frame, t, d, res)
        if predicate(r, b, margin):
            used.add((t, d))
            out.append((t, d, res, p0, p1))
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} windows for predicate")
    return out


def build_calls(frame: pd.DataFrame) -> list[dict]:
    days = pd.DatetimeIndex(frame[BENCHMARK].dropna().index)
    rows: list[dict] = []

    def add(analyst_id, name, firm, ticker, rating, target, call_d, months=12, note=""):
        rows.append({
            "call_id": f"{analyst_id}-{len(rows):03d}",
            "analyst_id": analyst_id, "analyst_name": name, "firm": firm,
            "ticker": ticker, "rating": rating, "target_price": round(float(target), 2),
            "call_date": pd.Timestamp(call_d).date().isoformat(),
            "horizon_months": months, "note": note,
        })

    # 1. Perma-bull: only Buys, on laggers that rose but LAGGED the index -> high dir, beat <= 0.
    rng = _rng_for("calloway")
    for t, d, res, p0, p1 in _pick(frame, days, rng, LAGGERS, _long_lags, 16, margin=0.03):
        add("calloway", "Reed Calloway", "Mornington Capital", t, "Buy", p0 * 1.15, d,
            note="perma-bull: rode the index up but lagged it")

    # 2. Skilled picker: longs that beat the index + shorts on decliners, tight targets -> beat > 0.
    rng = _rng_for("petrova")
    for t, d, res, p0, p1 in _pick(frame, days, rng, OUTPERFORMERS, _long_beats, 12, margin=0.05):
        add("petrova", "Ana Petrova", "Halberd Research", t, "Buy", p1 * (1 + rng.uniform(-0.03, 0.03)), d,
            note="skilled long")
    for t, d, res, p0, p1 in _pick(frame, days, rng, DECLINERS, _short_beats, 4, margin=0.04):
        add("petrova", "Ana Petrova", "Halberd Research", t, "Sell", p1 * (1 + rng.uniform(-0.03, 0.03)), d,
            note="skilled short")

    # 3. Contrarian: right on direction, WILD targets -> good dir, poor accuracy.
    rng = _rng_for("demir")
    for t, d, res, p0, p1 in _pick(frame, days, rng, DECLINERS, _short_beats, 6, margin=0.04):
        add("demir", "Yusuf Demir", "Beaumont & Pike", t, "Sell", p0 * (1 - rng.uniform(0.4, 0.6)), d,
            note="contrarian: right direction, wild target")
    for t, d, res, p0, p1 in _pick(frame, days, rng, OUTPERFORMERS, _long_beats, 4, margin=0.05):
        add("demir", "Yusuf Demir", "Beaumont & Pike", t, "Buy", p0 * (1 + rng.uniform(0.4, 0.6)), d,
            note="contrarian: right direction, wild target")

    # 4. Near-random: ratings/targets unrelated to the future.
    rng = _rng_for("lindqvist")
    all_live = [t for t in TICKER_SPECS if t != "HALT"]
    ratings = ["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    last_start = len(days) - HORIZON_TD - 1
    picks = rng.permutation(last_start)
    placed = 0
    for k in picks:
        if placed >= 12:
            break
        d = days[int(k)]
        t = all_live[int(rng.integers(len(all_live)))]
        res = days[int(k) + HORIZON_TD]
        if pd.isna(frame.at[d, t]) or pd.isna(frame.at[res, t]):
            continue
        p0 = frame.at[d, t]
        add("lindqvist", "Maya Lindqvist", "Cobblestone Securities", t,
            ratings[int(rng.integers(5))], p0 * (1 + rng.uniform(-0.3, 0.3)), d, note="near-random")
        placed += 1

    # 5. A decent generalist + a REVISED target (two dated rows on the same name, resolved
    #    independently — this realizes the close-old / open-new revision policy directly).
    rng = _rng_for("brandt")
    # exclude ORCA here so brandt's only ORCA calls are the revision pair below (clean demo)
    brandt_longs = [t for t in OUTPERFORMERS if t != "ORCA"]
    for t, d, res, p0, p1 in _pick(frame, days, rng, brandt_longs, _long_beats, 4, margin=0.04):
        add("brandt", "Owen Brandt", "Trell & Vance", t, "Buy", p1 * (1 + rng.uniform(-0.04, 0.04)), d,
            note="generalist long")
    rev_day = days[300]
    rev_p0 = frame.at[rev_day, "ORCA"]
    add("brandt", "Owen Brandt", "Trell & Vance", "ORCA", "Buy", rev_p0 * 1.25, rev_day,
        note="original target (revised later)")
    rev_day2 = days[360]  # within the original 252-day horizon -> a genuine mid-horizon revision
    rev2_p0 = frame.at[rev_day2, "ORCA"]
    add("brandt", "Owen Brandt", "Trell & Vance", "ORCA", "Buy", rev2_p0 * 1.10, rev_day2,
        note="REVISED target (closes the original at its own horizon; opens this new call)")

    # 6. A call on the ticker that DELISTS mid-horizon -> must be SKIPPED at resolution.
    halt_call_day = days[days.get_indexer([pd.Timestamp("2019-10-01")], method="bfill")[0]]
    halt_p0 = frame.at[halt_call_day, "HALT"]
    add("demir", "Yusuf Demir", "Beaumont & Pike", "HALT", "Buy", halt_p0 * 1.30, halt_call_day,
        note="ticker delists mid-horizon -> back-test must skip this call")

    # 7. Recent calls whose 12-month horizon runs past the data end -> still OPEN (an ingest
    #    drop with reason HORIZON_BEYOND_DATA; a back-test grades only closed calls).
    recent = days[len(days) - 120]
    add("petrova", "Ana Petrova", "Halberd Research", "QBIT", "Buy",
        frame.at[recent, "QBIT"] * 1.20, recent, note="recent call, still open (resolves after data ends)")
    recent2 = days[len(days) - 80]
    add("calloway", "Reed Calloway", "Mornington Capital", "BRGE", "Buy",
        frame.at[recent2, "BRGE"] * 1.15, recent2, note="recent call, still open")

    return rows


def main():
    frame = build_prices()

    # prices.csv (long format; ragged for HALT after delist)
    long = frame.reset_index().melt(id_vars="index", var_name="symbol", value_name="close")
    long = long.rename(columns={"index": "date"}).dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.date
    long["close"] = long["close"].round(4)
    long = long.sort_values(["symbol", "date"])
    long.to_csv(OUT / "prices.csv", index=False)

    calls = build_calls(frame)
    pd.DataFrame(calls).to_csv(OUT / "calls.csv", index=False)

    (OUT / "manifest.json").write_text(json.dumps({
        "benchmark_symbol": BENCHMARK,
        "default_horizon_months": 12,
        "label": "SAMPLE historical data (synthetic, fictional tickers/firms/analysts)",
        "is_sample": True,
    }, indent=2) + "\n")

    print(f"Wrote prices.csv ({len(long)} rows), calls.csv ({len(calls)} calls), manifest.json -> {OUT}")


if __name__ == "__main__":
    main()
