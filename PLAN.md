# PLAN — Analyst Scorecard

Honest, fair, reproducible grading of Wall Street analyst price targets.

The product's spine is **fair, correct, reproducible scoring**. Every decision below is
made in service of: no look-ahead bias, one rule applied identically to all analysts,
full traceability from score back to the exact prices used.

## Architecture / module map

```
analyst_scorecard/
  config.py                 # single source of truth for ALL params (seed, horizon, benchmark,
                            #   vol window, direction band, drift/vol per ticker, ...)
  schemas.py                # Pydantic models: Direction, Call, Resolution, CallScore,
                            #   AnalystScore, Leaderboard
  providers/
    price_provider.py       # PriceDataProvider (interface) + SyntheticPriceDataProvider (GBM, seeded)
    call_provider.py        # AnalystCallProvider (interface) + FixtureCallProvider (JSON)
    llm_extractor.py        # LLMCallExtractor (Anthropic structured JSON) — optional, key from env
  resolution.py             # resolve_call(): look-ahead-safe core. Only ever sees [call_date, resolution_date]
  scoring.py                # three-stage funnel: direction gate -> vol-scaled accuracy -> beat-the-market
  aggregation.py            # per-analyst aggregation into AnalystScore + Leaderboard
  verdicts.py               # plain-English verdicts: Anthropic API + deterministic templated fallback
  orchestrator.py           # autonomous time-loop agent: advance synthetic time, resolve as deadlines arrive
  cli.py                    # `python -m analyst_scorecard.cli` runs the whole simulation, prints leaderboard
fixtures/
  calls.json                # synthetic analyst calls (>=8 analysts, known skill profiles)
  research_notes/*.json     # messy research-note text + ground-truth Call records (extractor harness)
tests/                      # pytest, one module per phase
app.py                      # Streamlit demo: leaderboard, profile chart, call drill-down
outputs/                    # saved PNG charts + generated artifacts
```

## Scoring model (the funnel — implemented exactly)

A price-target call is graded as a **funnel**, not a blended average:

1. **DIRECTION — pass/fail gate.** From the rating we derive an implied direction:
   Buy/Overweight => UP, Sell/Underweight => DOWN, Hold => FLAT. We check whether the
   stock actually moved that way over the call horizon. UP requires actual return >
   +flat_band; DOWN requires actual return < -flat_band; FLAT requires |actual return|
   <= flat_band. A Buy that fell FAILS — nothing downstream can rescue it.
2. **ACCURACY — refines only direction-passers.** Volatility-normalized closeness of the
   actual horizon price to the target price: `accuracy = exp(-|P_actual - P_target| /
   (sigma_h * P_call))`, where `sigma_h` is the stock's realized volatility scaled to the
   horizon. 1.0 = bullseye; decays as the miss grows relative to how volatile the stock was.
   Tight calls on calm stocks are NOT rewarded the same as tight calls on wild stocks.
3. **BEAT-THE-MARKET — the headline.** Return of *following the call* minus the benchmark
   return over the identical window. Long if implied UP, short if implied DOWN, flat (0) if
   FLAT. `beat = position_sign * stock_return - benchmark_return` for the directional book;
   aggregated per analyst as the mean across their resolved calls. This is the number shown
   first on every profile: did you beat just buying the index?

## Fairness invariants (enforced in code + tests)

- **No look-ahead.** `resolve_call` is *only ever handed the price slice [call_date,
  resolution_date]`. Leakage is structurally impossible, and a test deliberately tries to
  leak future data and must be caught.
- **Rule fixed at record time.** The resolution rule (horizon -> resolution date, benchmark,
  partial-hit handling, revision policy) is fixed when the call is recorded, never chosen
  after the outcome.
- **One rule for everyone.** Same horizon definition, same benchmark, same bands, same
  accuracy formula for every analyst.
- **Revision policy (explicit, uniform):** a revised target **closes the old call** at its
  original resolution date and **opens a new call** from the revision date. Old call is still
  scored on its own horizon. Documented in PROGRESS.md; applied identically to all.
- **Traceability.** Every `CallScore` carries the exact prices (call price, target, actual,
  benchmark start/end) used to produce it.

## Phase plan

- **P0 Scaffold** — structure, venv, requirements, PLAN/PROGRESS, pytest, git. Done: empty suite runs, package imports.
- **P1 Data layer** — PriceDataProvider + SyntheticPriceDataProvider (seeded GBM), AnalystCallProvider + FixtureCallProvider, Pydantic schemas. Done: deterministic prices under seed; benchmark exists; fixtures load+validate.
- **P2 Resolution engine** — look-ahead-safe `resolve_call`. Done: tests prove no post-resolution data used + a leakage attempt is caught.
- **P3 Scoring engine** — direction gate -> vol accuracy -> beat-the-market; per-analyst aggregation. Done: Phase 4 passes.
- **P4 Validation** — ground-truth analysts behave correctly (buy-only rider <=0 beat-market but high direction; skilled picker >0; contrarian good direction/poor accuracy; monotonicity; reproducibility). Done: all green, definitions recorded.
- **P5 Extraction agent** — Pydantic-validated LLMCallExtractor (Anthropic JSON) + accuracy harness vs ground-truth notes; FixtureCallProvider offline fallback. Done: synthetic notes extract correctly.
- **P6 Orchestration + time loop** — wire ingest -> fix rule/deadline -> resolve on horizon -> update scores -> verdict line; autonomous loop over synthetic time; CLI prints leaderboard. Done: loop runs untouched, correct leaderboard.
- **P7 Visualization** — Streamlit: leaderboard, per-analyst profile vs index line, call drill-down with exact prices; save PNGs. Done: app launches and renders all three.
- **P8 Verdicts** — plain-English analyst verdicts via Anthropic + deterministic fallback. Done: readable verdict offline and via API.
- **Final** — full README (architecture, scoring defs, fairness rules, limitations, next steps); PROGRESS.md reflects final state.

## Working rules
- Offline-first: entire engine builds and runs with NO network and NO API key.
- Tests every phase; full suite must be green before advancing; commit after each phase.
- All params live in `config.py`. Numeric work vectorized. Seeds + params reproducible.
