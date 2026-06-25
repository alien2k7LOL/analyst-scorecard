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

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # run the full test suite (offline)
python -m analyst_scorecard.cli          # run the end-to-end simulation, print the leaderboard
streamlit run app.py                     # launch the demo app
python -m analyst_scorecard.synth        # (re)generate the synthetic call fixtures
python -m analyst_scorecard.viz          # save the leaderboard + profile charts as PNGs
```

Optional, for the LLM-backed paths: `export ANTHROPIC_API_KEY=...` (the call extractor and the
verdict writer will then use `claude-opus-4-8`; without it they fall back to deterministic code).

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

## Prioritized next steps

1. **Wire a real price-data provider** (market-data API / yfinance) behind `PriceDataProvider`,
   including a holiday-aware "next trading day on/after" resolution rule.
2. **Source real analyst calls** — run `LLMCallExtractor` over real research notes / disclosures and
   build a curated, deduped call database.
3. **Statistical significance for beat-the-market** — confidence intervals / call-count weighting so a
   lucky small sample can't top the board.
4. **Handle target revisions and multi-horizon calls** end to end (the close-old / open-new policy is
   already defined and enforced by the resolver).
5. **Seed a historical back-test** so the scoreboard launches with real history instead of synthetic.
6. **Richer benchmarks** (sector/style/beta-adjusted) and short-book financing costs.

---

See [`PLAN.md`](PLAN.md) for the full phase plan and [`PROGRESS.md`](PROGRESS.md) for the running
build log and every scoring assumption recorded as it was made.
