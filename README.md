# Analyst Scorecard

> Honest, fair, reproducible grading of Wall Street analyst price targets.

Track analyst price-target calls, wait for each target's deadline, compare the call to what
the stock actually did, and grade each analyst honestly over time. The number at the top of
every profile answers the only question that matters:

**Would you have done better just buying the index?**

The whole engine is **offline-first**: it builds and runs with **no network and no API key**.
The Anthropic API is used only for two optional things — extracting calls from messy research
text, and writing natural-language verdicts — and both sit behind interfaces with deterministic
offline fallbacks.

---

## Quick start

**Easiest (macOS):** double-click `launch_scorecard.command` in Finder — it creates the virtualenv,
installs dependencies on first run, starts the app, and opens it in your browser.

**Manual (any platform):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # run the full test suite (offline)
python -m analyst_scorecard.cli          # synthetic end-to-end simulation, print the leaderboard
python -m analyst_scorecard.backtest_cli # HISTORICAL back-test on the sample data (real-call grading)
streamlit run app.py                     # launch the demo app (Synthetic + Historical tabs)
python -m analyst_scorecard.synth        # (re)generate the synthetic call fixtures
python -m analyst_scorecard.viz          # save the leaderboard + profile charts as PNGs
```

Optional, for the LLM-backed paths: `export ANTHROPIC_API_KEY=...` — the call extractor and the
verdict writer then use `claude-opus-4-8`, and the news-sentiment read upgrades from the offline
word-list to a Claude scorer (`claude-haiku-4-5`) that catches negation, relief idioms, and sarcasm.
Without a key, every path falls back to deterministic offline code.

Example leaderboard (default seed):

```
 #  Analyst               Beat-Mkt  Dir Hit Accuracy  Calls
 1  Vega Capital            +40.5%      85%   0.947     13   skilled stock-picker
 2  Ursa Research           +33.3%      82%   0.480     11   contrarian: right on dir, wild targets
 3  ShortAlpha Capital      +22.8%      92%   0.949     12   shorts that beat the index
 ...
 7  MomentumOne              -4.8%      93%   0.665     14   buy-only rider: high dir, lost to index
 8  Hubris Partners          -4.8%      10%   0.069     10   overconfident & wrong
```

MomentumOne has a **93% direction hit-rate** yet a **negative beat-the-market** — the headline
correctly strips out the free market gains an indiscriminate perma-bull rode up.

---

## Architecture / module map

```
analyst_scorecard/
  config.py            Single source of truth: seed, calendar, universe (drift/vol per ticker),
                       benchmark, horizon, direction band, accuracy params. Everything tunable lives here.
  schemas.py           Pydantic contracts: Direction, Rating, RATING_TO_DIRECTION, Call, Resolution,
                       CallScore, AnalystScore, Leaderboard.
  providers/
    price_provider.py  PriceDataProvider (interface) + SyntheticPriceDataProvider (seeded GBM) +
                       PriceWindow (the bounded [call,resolution] slice — the look-ahead guard).
    call_provider.py   AnalystCallProvider (interface) + FixtureCallProvider (JSON) + InMemoryCallProvider.
  resolution.py        resolve_call(call, window): the look-ahead-safe core. Only ever sees the window.
  scoring.py           The three-stage funnel: direction gate -> vol-scaled accuracy -> beat-the-market.
  aggregation.py       Per-analyst aggregates + beat-market-ranked leaderboard.
  orchestrator.py      TimeLoopOrchestrator: autonomous synthetic-time loop; resolves at each deadline.
  extraction.py        ExtractedCall schema; CallExtractor; HeuristicCallExtractor (offline) +
                       LLMCallExtractor (Anthropic structured output); accuracy harness.
  verdicts.py          VerdictGenerator; TemplatedVerdictGenerator (offline) + LLMVerdictGenerator.
  viz.py               Headless matplotlib charts + dataframes + PNG export.
  cli.py               `python -m analyst_scorecard.cli` — runs the simulation, prints the leaderboard.
  synth.py             Generates the synthetic ground-truth dataset (8 analysts) -> fixtures/calls.json.
fixtures/
  calls.json           108 synthetic calls across 8 known-skill analysts.
  research_notes/notes.json   5 messy research notes + ground-truth extractions (extractor harness).
tests/                 pytest, one module per phase (79 tests).
app.py                 Streamlit demo: leaderboard, per-analyst profile vs. index, call drill-down.
outputs/               Saved PNG charts (git-ignored; regenerate deterministically).
PLAN.md / PROGRESS.md  The phase plan and the running build log (every scoring assumption recorded).
```

---

## The scoring model — a funnel, not a blended average

Each price-target call is graded in three stages. Let `P_call`/`P_actual` be the stock close on
the call/resolution date, `P_target` the analyst's target, and `B_call`/`B_actual` the benchmark
closes. Over the horizon: `stock_return = P_actual/P_call − 1`, `benchmark_return = B_actual/B_call − 1`.

**Implied direction** (fixed map, applied identically to all): Buy/Overweight → UP, Sell/Underweight
→ DOWN, Hold → FLAT.

1. **DIRECTION — a pass/fail gate.** The realized move is bucketed with one shared band
   `b = 0.02`: `stock_return > b` → UP, `< −b` → DOWN, else FLAT. The call **passes iff realized
   direction == implied direction**. A Buy that fell FAILS — nothing downstream rescues it.

2. **ACCURACY — refines only direction-passers.** Volatility-normalized closeness, in `(0, 1]`:

   ```
   error_frac     = |P_actual − P_target| / P_call
   sigma_h        = realized daily-log-return std (over the horizon) × √(horizon trading days)
   accuracy       = exp( − error_frac / (sigma_h × accuracy_scale) )      # accuracy_scale = 1.0
   ```

   `accuracy == 1.0` exactly when `P_actual == P_target`. It decays as the miss grows relative to
   how much *that* stock moved over *that* horizon, so a tight call on a volatile stock counts more
   than the same miss on a calm one. `accuracy` is `None` for calls that failed the gate.

3. **BEAT-THE-MARKET — the headline.** With `position` = +1 long (UP) / −1 short (DOWN) / 0 (Hold):

   ```
   call_return = position × stock_return
   beat        = call_return − benchmark_return
   ```

   Computed for **every directional call — winners and losers** (a wrong-direction Buy still loses
   money, and that loss must count). Hold = neutral, **excluded** from the beat book. v1 models a
   simple ±1 position with no borrow/financing cost; long and short are symmetric.

**Per-analyst aggregates** (averaged, not compounded, so they compare across call counts):
`direction_hit_rate` = passes / all calls (Holds included) · `mean_accuracy` = mean over
direction-passing calls · **`beat_market`** = mean `beat` over all directional calls (the headline).
The leaderboard ranks by `beat_market` descending.

> The funnel shape matters: **direction gates accuracy**, but **beat-the-market is its own headline**
> over every directional call — it is *not* gated by direction.

---

## Fairness rules (enforced in code and tests)

- **No look-ahead bias — structural, not by convention.** A call is resolved using only data in
  `[call_date, resolution_date]`. The resolver is handed a `PriceWindow` (a physical slice ending
  at the resolution date), never the full provider — so it *cannot* see the future even in
  principle. Tests prove it two ways: (a) multiplying every post-resolution price by 1000 leaves the
  resolution byte-for-byte identical; (b) a window built with future data, or asked for a future
  price, raises "LOOK-AHEAD BLOCKED".
- **The resolution rule is fixed at record time.** A call's deadline and horizon are recorded when
  the call is made and never re-chosen after the outcome is known. Scoring a call against any other
  resolution date is rejected as a look-ahead back door.
- **One rule for everyone.** Same horizon definition, same benchmark, same direction band, same
  accuracy formula — all in `config.py`, applied identically to every analyst.
- **Horizon:** each call carries its own horizon in trading days; the synthetic fixtures use a single
  1-year (252-trading-day) horizon. The deadline is `call_date + horizon` trading days.
- **Benchmark:** a single diversified index (`MKT`) with positive drift — a deliberately rising
  market, so a buy-only rider can look good on direction while adding no real value.
- **Revision policy (declared, uniform):** a revised target **closes the old call** at its original
  resolution date and **opens a new call** from the revision date; the old call is still scored on
  its own horizon. (No revisions appear in the v1 synthetic set; the policy is fixed for when they do.)
- **Partial hits:** there is no special "partially hit" bucket and a target need never be *touched*.
  A call is graded purely on (a) realized direction vs implied and (b) volatility-normalized
  closeness of `P_actual` to `P_target`. Uniform and unambiguous.
- **Full traceability:** every `CallScore` carries the exact prices (call, target, actual, benchmark
  start/end) used to produce it; the app's drill-down surfaces them.
- **Reproducibility:** identical seed ⇒ identical prices ⇒ identical scores (tested).

### How the synthetic ground truth is built (and why it's still fair)

`synth.py` constructs 8 analysts with *known* skill by peeking at the realized synthetic future to
place their calls (e.g. the skilled picker is given longs that genuinely beat the index). This is
**world-building for validation** — it is entirely separate from the scoring engine, which is
structurally blind to the future. Phase 4 then verifies the blind engine *recovers* the planted
skill. The eight profiles: skilled picker, buy-only rider, contrarian (right direction/wild targets),
near-random, overconfident-and-wrong, Hold specialist, short-seller, and a middling generalist.

---

## How to run

**Tests** — `pytest` (79 tests, ~2s, fully offline; 2 LLM tests auto-skip without a key).

**CLI simulation** — `python -m analyst_scorecard.cli` streams a verdict line for each call as its
deadline arrives in synthetic time, then prints the leaderboard with plain-English verdicts.
Flags: `--seed N` (different reproducible price world), `--quiet` (leaderboard only),
`--max-events N` (cap streamed lines).

**Streamlit app** — `streamlit run app.py`: the leaderboard, a per-analyst profile chart (points
above the y=x line beat the index; below lagged it), and a call-level drill-down showing the exact
resolving prices. The sidebar can save the charts as PNGs.

**With an API key** (`ANTHROPIC_API_KEY`): `LLMCallExtractor` reads messy research notes into clean
`Call` records via structured JSON output, and `LLMVerdictGenerator` writes the analyst verdicts —
both `claude-opus-4-8`, both with deterministic offline fallbacks.

---

## Current capabilities

- Seeded, reproducible synthetic price universe (10 tickers + benchmark) and 8 ground-truth analysts.
- Structurally look-ahead-safe resolution engine.
- The full three-stage scoring funnel with per-analyst aggregation and a beat-market leaderboard.
- A validation suite that proves the engine recovers known skill and obeys the fairness invariants.
- LLM call extraction (structured output) with an accuracy harness + a deterministic offline extractor.
- An autonomous synthetic-time loop that resolves calls at their deadlines and updates running scores.
- A Streamlit app + PNG export with full call-level traceability.
- Plain-English verdicts (LLM + templated fallback).

## Limitations (honest)

- **Synthetic data only.** Prices and calls are generated, not real. The data layer is behind an
  interface so a real provider can drop in.
- **Single benchmark** for everyone; no sector/style-adjusted or beta-adjusted benchmark yet.
- **Simplified horizon handling.** Fixtures use one 1-year horizon; the resolution date is assumed
  to be a trading day with data (a real provider needs a "next trading day on/after" rule).
- **Beat-the-market is a point estimate** (a mean excess return) and is outlier-sensitive at small
  call counts — the near-random analyst can post a positive beat by luck even at 24 calls (its 29%
  direction hit-rate is what exposes it). Real use needs call-count weighting / significance testing.
- **Beat-the-market ignores borrow/financing costs** and treats long and short symmetrically.
- **Revisions and multi-horizon calls** are specified but not exercised by the v1 fixtures.

---

# Historical back-test — seed the scoreboard with real past calls

The synthetic engine above proves the scoring is fair and correct. The **historical back-test**
grades real analysts on real *past* price-target calls whose outcomes are already known, so the
board can launch with a genuine track record instead of from zero. It feeds real data through the
**exact same** look-ahead-safe resolver and three-stage funnel — only the data source changes
(`HistoricalPriceFileProvider` / `HistoricalCallFileProvider` behind the same interfaces). The
build is fully offline on a shipped sample; real data drops in by replacing the sample files.

Plan and decisions: [`BACKTEST_PLAN.md`](BACKTEST_PLAN.md).

## Run it

```bash
python -m analyst_scorecard.backtest_cli                  # the shipped SAMPLE dataset
python -m analyst_scorecard.backtest_cli --show-skips     # also list skipped / dropped calls
python -m analyst_scorecard.backtest_cli --data-dir /path/to/your/folder   # your own real data
streamlit run app.py                                      # the "Historical back-test" tab
```

Sample output: 61 calls ingested, **60 resolved & scored**, **1 skipped** (`HALT` delisted
mid-horizon), 2 dropped at ingest (recent, still-open). The perma-bull (`Reed Calloway`) shows a
**100% direction hit-rate but −10.7% beat-market** — the headline correctly strips the market
gains he merely rode — while the genuine picker (`Ana Petrova`) is **+34.7%**.

## Supplying your own real data (replace the sample)

Point `--data-dir` (or the app's sidebar box) at a folder with three files. Full schema and a
worked example: [`data/sample_historical/README.md`](data/sample_historical/README.md).

- **`manifest.json`** — `{ "benchmark_symbol": "SPX", "default_horizon_months": 12 }`. The
  benchmark defines the "vs the index" comparison and the trading calendar.
- **`prices.csv`** — long format `date,symbol,close`, where `close` is the **split- and
  dividend-adjusted** close (returns are computed on it directly). Ragged coverage is allowed: a
  delisted ticker simply stops having rows.
- **`calls.csv`** (or `.json`) — `call_id,analyst_id,analyst_name,firm,ticker,rating,target_price,
  call_date,horizon_months`. `rating` accepts common vocabularies (Outperform→Overweight,
  Neutral/Equal Weight→Hold, Underperform→Underweight, …); `horizon_months` defaults to 12;
  `call_date` is the original publication date.

## Documented policies (fixed, uniform, applied to every analyst)

- **Trading calendar** = the benchmark's available dates.
- **Call-date snapping:** a raw call date is moved FORWARD to the next benchmark trading day
  (weekend/holiday publication → next session). Forward-only, so no future information is used.
- **Default horizon:** 12 months → `resolution_date = call_date + round(months/12 × 252)` trading
  days, fixed at record time and never re-chosen.
- **Revisions:** record a revised target as a **separate dated row**; each row is resolved
  independently on its own horizon (this realizes "close the old call at its original horizon, open
  a new call from the revision date" with no special engine handling).
- **Missing windows / delisted tickers (skip + log):** if a ticker has no price on its resolution
  date (delisted/halted mid-horizon), the call is **skipped** with reason `DELISTED_OR_HALTED` and
  surfaced in the report — never silently scored. Calls that can't even form a closeable record are
  dropped at ingest with a reason (`HORIZON_BEYOND_DATA` for still-open calls, `NO_ENTRY_PRICE`,
  `UNKNOWN_TICKER`, `BAD_RATING`, …). A back-test grades only closed, resolvable calls.

## Look-ahead-safety guarantee (the #1 risk on historical data)

A historical call is resolved using **only** prices from its original call date through its
original horizon date — even though "later" is now in the past and sitting in the file. This is
structural: resolution flows only through `resolve_call_with_provider`, which hands the resolver a
`PriceWindow` that physically ends at the resolution date and rejects any series extending past it.
Proven in [`tests/test_backtest_phaseE_lookahead.py`](tests/test_backtest_phaseE_lookahead.py):

- A score is **byte-identical** whether post-horizon prices are present, truncated, or multiplied
  by 1000.
- A Buy that is **down at its horizon (FAIL)** but moons to 200× afterwards is scored **FAIL** —
  moving that same high price *inside* the horizon flips it to PASS, proving the verdict uses only
  in-window data.
- Same inputs ⇒ identical historical leaderboard; deleting **all** data after the last resolution
  date changes nothing; and the perma-bull/skilled fairness guarantees hold on historical-style
  data.

## Limitations specific to the back-test

- The shipped dataset is a **clearly-labelled synthetic sample** (fictional tickers/firms/analysts);
  it exercises the pipeline offline. Real grading requires supplying real files.
- **Single benchmark; no corporate-action handling in-engine** (prices are assumed pre-adjusted).
- **Delisted-to-zero calls are skipped, not scored** — conservative (never invents a price), but it
  lets a pick that blew up and got delisted escape a deserved bad mark.

## Live Grader — grade a prediction against today's market

The **🛰️ Live Grader** tab (`app.py`) grades a brand-new prediction against *real* market data
pulled live via `yfinance`, through the **exact same** look-ahead-safe funnel. Two honest caveats
are built into the design and surfaced in the UI:

- **PROVISIONAL vs FINAL.** A prediction whose horizon hasn't elapsed yet can only be graded
  *so far* — a mark-to-market interim read, not the resolved score. The grader labels it
  `PROVISIONAL`; only a call whose deadline has already passed is graded `FINAL` (and pinned to
  the deadline, never to "today").
- **Not reproducible.** A live grade depends on live prices and the day you ran it, so it is a
  point-in-time snapshot — deliberately kept *separate* from the reproducible back-test, never
  folded into it.

Look-ahead safety still holds: the grade is computed from a `PriceWindow` that ends at the grading
date, and no price after it is ever fetched or read (proved in
[`tests/test_live_grader_offline.py`](tests/test_live_grader_offline.py), which injects a synthetic
frame so the provider→resolve→score path is verified with no network).

Why `yfinance` and not a search-engine scrape: the engine needs the **daily series** of both the
stock *and* the benchmark across the window (to get the entry price, returns, and realized vol). A
scraped *spot* price is one number that can't resolve a call at all — so the live source is
[`LiveWebPriceProvider`](analyst_scorecard/providers/live_web_price_provider.py), a
`PriceDataProvider` backed by `yfinance`'s adjusted daily history.

```bash
.venv/bin/pip install -r requirements-live.txt   # optional dep — core runs without it
.venv/bin/streamlit run app.py                   # → 🛰️ Live Grader tab
```

## Forecast & calibration — probability that a FUTURE prediction comes true

Where the scorecard grades calls *after* the fact, the **🔮 Forecast** tab (and the
`analyst_scorecard.forecast` package) grades them *before*: given (ticker, target, deadline,
direction) it estimates the probability the price **touches the target by the deadline**, and
backtests that probability on history to measure and refine its **calibration**.

How it works, on the same look-ahead-safe spine:

1. **Touch probability** — a closed-form GBM barrier-hitting model (`probability.py`), validated
   against Monte Carlo. The raw model assumes continuous monitoring, so it's *mis-calibrated* on
   daily closes — which the backtest then corrects.
2. **Look-ahead-safe inputs** — the price `LookbackWindow` ends exactly at `as_of` and the
   point-in-time `NewsWindow` holds only articles dated on/before `as_of`; both *raise* if handed
   future data (proved in [`tests/test_forecast_lookahead.py`](tests/test_forecast_lookahead.py)).
3. **Calibration backtest** (`backtest.py`) — manufactures thousands of (ticker, as_of, horizon,
   target) predictions from history, scores each with only pre-`as_of` data, resolves the actual
   touch, splits by time (train fully resolved *before* any test prediction), fits a logistic
   recalibration layer on train, and reports Brier / log-loss / ECE on held-out test for: `raw` →
   `recalibrated` → `+momentum` → `+news` → `full`. **News is credited only if it beats the
   price-only model on data it never saw.** On the sample, recalibration cuts ECE from ~0.10 to
   ~0.03 and point-in-time news lowers held-out log-loss further.
4. **Live grading** (`live.py`) — fetches real history via `yfinance` and **self-calibrates on the
   ticker's own past** before grading your prediction (price-only; see the news note below).

```bash
python -m analyst_scorecard.forecast.cli              # the sample calibration report (with news)
python -m analyst_scorecard.forecast.cli --no-news    # ablate news to see its contribution
```

Honest caveats, surfaced in the UI:

- **A calibrated estimate, not a crystal ball.** Markets are near-efficient; the value is a
  *well-calibrated* probability (validated on history), never a guarantee.
- **News is point-in-time, and price-only when live.** Trustworthy historical point-in-time news
  isn't freely available, so the *live* path is price-only; the *value* of news is demonstrated on
  the offline sample, where the feed is synthetic but correctly timestamped. The sample news is
  ground-truth construction (like the seeded prices) — the engine still only ever reads news dated
  ≤ `as_of`.
- **Direction-oriented features.** Sentiment/trend features are oriented to the predicted direction
  (a bullish signal helps an UP-touch and hurts a DOWN-touch); without that, the effect cancels.

## Prioritized next steps (going live on top of the historical base)

1. **Wire a real, continuously-updating price provider** (market-data API / yfinance) behind the
   same `PriceDataProvider`, with a holiday-aware "next trading day on/after" rule and live
   corporate-action adjustment.
2. **Source and curate real analyst calls** — run `LLMCallExtractor` over real research/disclosures
   into `calls.csv`, deduped and revision-aware, to replace the sample.
3. **Resolve delistings at the terminal/last price** as an explicit, documented policy option, so a
   blow-up counts against the analyst instead of being skipped.
4. **Statistical significance for beat-the-market** — confidence intervals / call-count weighting so
   a lucky small sample (e.g. a 6-call analyst) can't top the board.
5. **Continuous operation:** a scheduled job that ingests new calls daily, resolves matured ones at
   their horizons (the time-loop already models this), and appends to the running track record built
   from the historical back-test.
6. **Richer benchmarks** (sector/style/beta-adjusted) and short-book financing/borrow costs.

---

See [`PLAN.md`](PLAN.md) / [`BACKTEST_PLAN.md`](BACKTEST_PLAN.md) for the phase plans and
[`PROGRESS.md`](PROGRESS.md) for the running build log and every scoring assumption recorded as it
was made.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

> **Not financial advice.** This project grades and forecasts for research and educational purposes
> only. Probabilities are model estimates, not guarantees; nothing here is a recommendation to buy
> or sell any security.
