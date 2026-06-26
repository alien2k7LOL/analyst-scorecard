# BACKTEST_PLAN — Historical back-test (seed the scoreboard with real past calls)

Grade real analysts on real past price-target calls whose outcomes are already known, so the
scoreboard launches with a genuine track record. The build runs fully offline on a shipped local
sample; real data drops in by replacing the sample files.

## Cardinal rule
A historical call is resolved using **only** price data from its original call date through its
original horizon date — never anything later, even though "later" is now in the past and sitting
in the file. Every historical resolution goes through the **existing** look-ahead-safe resolver.

## Reuse, do NOT reimplement (this is the whole point)
The historical path feeds REAL data through the SAME validated engine. Concretely it reuses,
unchanged:
- `resolution.resolve_call_with_provider(call, provider)` — the only bridge to the resolver; it
  slices `[call_date, resolution_date]` via `provider.window_for_call` and builds a `PriceWindow`
  whose invariant (series must start exactly at call_date and end exactly at resolution_date)
  makes look-ahead structurally impossible.
- `scoring.score_call(call, resolution, config)` — the three-stage funnel (direction gate →
  vol-scaled accuracy → beat-the-market), with `DEFAULT_CONFIG` so historical and synthetic use
  IDENTICAL scoring rules (`direction_flat_band=0.02`, `accuracy_scale=1.0`, `min_sigma_h=1e-4`).
- `aggregation.aggregate_analyst(...)` and `schemas.Leaderboard.from_scores(...)` — per-analyst
  rollup and beat-market ranking.
- `schemas.Call` — the same Pydantic contract; historical adapters produce these exact objects.

The ONLY new code is: two file-based providers behind the existing interfaces, a thin back-test
runner that adds skip-handling around the unchanged resolver, a CLI, a sample dataset, and an app
tab. No scoring or resolution logic is rewritten.

## New modules
```
analyst_scorecard/providers/historical_price_provider.py   HistoricalPriceFileProvider(PriceDataProvider)
analyst_scorecard/providers/historical_call_provider.py    HistoricalCallFileProvider(AnalystCallProvider) + rating normalization
analyst_scorecard/backtest.py                              HistoricalBacktest runner + BacktestResult/report
analyst_scorecard/backtest_cli.py                          `python -m analyst_scorecard.backtest_cli`
data/sample_historical/                                    prices.csv, calls.csv, manifest.json, README.md, _generate_sample.py
tests/test_backtest_*.py                                   adapters / runner / look-ahead-safety
```

## File schemas (how a user supplies real data)
A back-test data folder contains three files:
- **`manifest.json`** — `{ "benchmark_symbol": "SPX", "default_horizon_months": 12, "label": "..." }`.
- **`prices.csv`** — long format `date,symbol,close`. `close` is the **split- and
  dividend-ADJUSTED** close (corporate actions already handled in the data; the engine computes
  simple returns on adjusted closes — no in-engine adjustment). Ragged coverage is allowed: a
  delisted ticker simply stops having rows; an interior missing day is allowed.
- **`calls.csv`** — `call_id,analyst_id,analyst_name,firm,ticker,rating,target_price,call_date,horizon_months`.
  `rating` accepts common vocabularies (Strong Buy→Buy, Outperform/Accumulate→Overweight,
  Neutral/Market Perform/Equal Weight→Hold, Underperform/Reduce→Underweight, Strong Sell→Sell).
  `horizon_months` optional → default 12. `call_date` is the ORIGINAL publication date.

## Documented policies (fixed, uniform, applied to every analyst)
- **Trading calendar** = the benchmark's available dates. (The S&P-style index trades every market
  day; it is the canonical calendar.)
- **Call-date snapping:** a raw call date is snapped to the next benchmark trading day on/after it
  (weekend/holiday publication → next session).
- **Horizon → deadline:** `resolution_date = snapped_call_date + horizon_days` benchmark trading
  days, where `horizon_days = round(horizon_months/12 × 252)`; default horizon = 12 months. Fixed
  at record time, never re-chosen.
- **Adjusted close / splits:** prices are pre-adjusted; returns are `P_res/P_call − 1`.
- **Revisions:** a revised target is a SEPARATE dated call row; each row is resolved independently
  on its own horizon. This realizes the "close the old call at its original horizon, open a new
  call from the revision date" policy directly in the data — no special engine handling.
- **Ingest drops (can't even build a valid Call), logged with a reason:** raw call date after the
  last trading day (`CALL_DATE_OUT_OF_RANGE`); deadline beyond the data, i.e. the call is still
  open (`HORIZON_BEYOND_DATA`); no ticker price on the call date (`NO_ENTRY_PRICE`); unrecognized
  rating (`BAD_RATING`).
- **Resolution skips (valid Call, but no look-ahead-safe outcome), logged with a reason:** the
  ticker has no price on the resolution date — delisted/halted mid-horizon (`DELISTED_OR_HALTED`);
  any other window/resolver error (`RESOLVER_ERROR`). **Skipped calls are excluded from scores and
  surfaced in the report** — a back-test grades only closed, resolvable calls. (A future policy
  could resolve a delisting at its last/terminal price; v1 skips, which is conservative and
  never invents a price.)

## Look-ahead-safety strategy (#1 risk on historical data)
1. Resolution goes through `resolve_call_with_provider` only; the resolver sees a `PriceWindow`,
   never the provider, and the window physically ends at `resolution_date`.
2. The runner pre-checks resolvability via data presence at the exact call/resolution dates (no
   reliance on exception text) and otherwise calls the unchanged resolver.
3. Phase E proves it behaviorally: (a) deleting or tampering with EVERY post-horizon price leaves
   the `CallScore` byte-identical; (b) a call whose stock is down at the horizon (a Buy that should
   FAIL) but which moons afterwards is scored FAIL — the post-horizon moon is provably ignored.

## Phase plan
- **A Read & plan** — this file + PROGRESS update. Reuse, not reimplement.
- **B Adapters** — `HistoricalPriceFileProvider`, `HistoricalCallFileProvider` (+ policies). Synthetic providers untouched and still passing.
- **C Sample dataset** — `data/sample_historical/` with edge cases (perma-bull, skilled picker, delisting, revision) + folder README.
- **D Back-test runner + CLI** — walk forward, resolve each call at its horizon via the engine, accumulate, print the historical leaderboard.
- **E Look-ahead-safety validation** — the leakage proofs + reproducibility + perma-bull/skilled recovery on historical-style data.
- **F App** — a historical tab: leaderboard, profile, call drill-down with the exact original-window prices; sample-vs-user-supplied labelled.
- **Final** — README + PROGRESS: how to run, schemas, policies, leakage guarantees, go-live next steps.

## Ground-truth properties the sample must reproduce (validated in Phase E)
- A perma-bull (only Buys, on index-lagging names in a rising market): HIGH direction hit-rate,
  beat-market ≤ 0.
- A genuinely skilled picker (longs that beat the index, shorts on a decliner): beat-market > 0.
- Same inputs ⇒ identical historical leaderboard.
- The delisted-ticker call is skipped with reason `DELISTED_OR_HALTED`, not silently scored.
