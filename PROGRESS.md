# PROGRESS — Analyst Scorecard

Running log. Updated after every phase: what was built, test results, and every
assumption/limitation introduced. Newest phase appended at the bottom.

> **Scope reminder:** the single most important property is FAIR, CORRECT, REPRODUCIBLE
> scoring. Every assumption that could affect a score is recorded here.

---

## Environment
- Python 3.11.9, venv at `.venv/`.
- Dependencies pinned in `requirements.txt` (numpy 2.2.1, scipy 1.15.1, pandas 2.2.3,
  pydantic 2.10.5, matplotlib 3.10.0, plotly 5.24.1, streamlit 1.41.1, pytest 8.3.4,
  anthropic 0.42.0). All import cleanly.
- Offline-first: nothing in the engine requires network or `ANTHROPIC_API_KEY`.

---

## Phase 0 — Scaffold  ✅
**Built:** repo structure (`analyst_scorecard/` package + `providers/` subpkg, `fixtures/`,
`tests/`, `outputs/`), virtual environment, pinned `requirements.txt`, `PLAN.md`, this file,
`README.md` skeleton, `.gitignore`, pytest config, package `__init__` with version, a smoke
test. Initialized git.

**Test results:** `pytest` → **2 passed** (`tests/test_smoke.py`): package imports at
version 0.1.0, and import succeeds with no `ANTHROPIC_API_KEY` set (offline-first guard).

**Assumptions / decisions:**
- Single config module (`config.py`) will hold every scoring parameter and the RNG seed so
  reproducibility and fairness are auditable in one place.
- All prices/dates are synthetic and deterministic; "today" in the simulation is driven by a
  synthetic clock, never the wall clock, so runs are reproducible regardless of real date.

**Limitations so far:** none beyond "synthetic data only," which is by design for v1.

---

## Phase 1 — Data layer (synthetic, reproducible)  ✅
**Built:**
- `config.py` — the single source of truth: `ScorecardConfig` (frozen dataclass) holding the
  seed, synthetic calendar, universe (10 `TickerSpec`s + benchmark `MKT`), horizon, direction
  band, accuracy params. `ticker_seed()` derives a per-symbol RNG seed from `SHA-256(seed:symbol)`.
- `schemas.py` — Pydantic v2 models: `Direction`, `Rating`, frozen `RATING_TO_DIRECTION` /
  `DIRECTION_TO_POSITION` maps, `Call`, plus `Resolution`/`CallScore`/`AnalystScore`/`Leaderboard`
  (defined now as the full contract; populated in later phases).
- `providers/price_provider.py` — `PriceDataProvider` ABC + `SyntheticPriceDataProvider` (seeded
  GBM, one independent RNG per symbol) + `PriceWindow` (the bounded [call,resolution] slice).
- `providers/call_provider.py` — `AnalystCallProvider` ABC + `FixtureCallProvider` (strict JSON
  load) + `InMemoryCallProvider`.
- `synth.py` — generator for 8 ground-truth analysts (98 calls) written to `fixtures/calls.json`.

**Test results:** `pytest` → **16 passed** (14 Phase-1 + 2 smoke). Confirms: prices deterministic
under a fixed seed; different seed ⇒ different prices; per-ticker seed is order-independent; first
day == start price; benchmark series exists and is separate; `trading_day_offset` counts trading
days and refuses to run past the data; `PriceWindow` is bounded exactly and rejects future data;
fixtures load+validate with all 8 analysts and all 5 rating types; bad calls are rejected.

**Assumptions / decisions (scoring-relevant):**
- **Calendar:** synthetic business days (`pd.bdate_range`, Mon–Fri, no holidays), 1008 days
  (~4y) from 2021-01-04. "Trading day" = a business day in this index.
- **Prices:** seeded GBM with annualized drift/vol per ticker; daily step `dt = 1/252`. First
  day pinned to `start_price`. Benchmark `MKT` has positive drift (0.08) — a deliberately RISING
  MARKET so the buy-only rider can look good on direction while adding no real value.
- **Per-symbol independent RNG** (seed from `SHA-256(seed:symbol)`): adding/reordering tickers
  never perturbs another ticker's path → robust reproducibility.
- **Call price source of truth:** the resolver uses provider closes (full precision). A call's
  `initial_price` is stored as the exact provider close on the call date (a faithful snapshot);
  `target_price` is rounded to 2 d.p. (the analyst's stated number).
- **WORLD-BUILDING vs SCORING:** `synth.py` peeks at the realized synthetic future to *construct*
  analysts with known skill (e.g. the skilled picker gets longs that genuinely beat the index).
  This is fixture construction only — it is NOT look-ahead in scoring. The scoring engine is only
  ever handed a bounded window and is structurally blind to the future. Phase 4 verifies the blind
  engine recovers the planted skill. Each call's resolution rule (horizon→deadline) is still fixed
  at record time.

**Limitations:** synthetic data only (by design); single benchmark; single 1-year horizon for all
fixtures (engine supports per-call horizons; multi-horizon fixtures are a documented next step).

---

## Phase 2 — Resolution engine (look-ahead-safe core)  ✅
**Built:** `resolution.py` — `resolve_call(call, window)` computes, from a bounded window only:
`call_price`, `actual_price`, `stock_return`, `benchmark_return`, `realized_horizon_vol`, and
`n_observations` → a frozen `Resolution`. `resolve_call_with_provider(call, provider)` is the only
bridge from a full provider to the resolver and slices strictly to the record-time resolution date.
The resolver also asserts the window matches the call (ticker + both dates) so a misaligned/leaky
window can never be silently scored.

**Test results:** `pytest` → **25 passed** (9 Phase-2 added). Proofs:
- **(a) behavioral:** multiplying EVERY post-resolution price by 1000 leaves the `Resolution`
  byte-for-byte identical; mutating pre-call prices likewise. The future cannot enter the math.
- **(b) structural:** a window extending past the resolution date is rejected ("LOOK-AHEAD
  BLOCKED"); asking a window for a future price raises; the resolver refuses a window resolved on
  a later date than the call's record-time deadline, and refuses a mismatched ticker.
- every one of the 98 fixture calls resolves cleanly.

**Assumptions / decisions (scoring-relevant):**
- **Returns:** simple returns `P_res/P_call − 1` for both stock and benchmark, over [call,res].
- **Realized horizon vol:** sample std (ddof=1) of daily LOG returns in the window × √(horizon
  steps). This is the per-stock, per-horizon yardstick the accuracy stage divides by.
- **Deadline is sacred:** scoring a call against any resolution date other than its record-time
  `resolution_date` is treated as a look-ahead back door and raised, not silently allowed.

**Limitations:** resolution assumes the resolution date is a trading day with data (true for all
synthetic calls). A real provider needs an explicit "next trading day on/after" rule for holidays
/ missing data — noted as a next step.

---

## Phase 3 — Scoring engine (direction gate → accuracy → beat-the-market)  ✅
**Built:** `scoring.py` (`bucket_direction`, `compute_accuracy`, `score_call`) and
`aggregation.py` (`score_calls`, `aggregate_analyst`, `aggregate_all`, `build_leaderboard`).

### EXACT METRIC DEFINITIONS (the audit trail — applied identically to every analyst)
Let `P_call`, `P_actual` = stock close on call/resolution date; `P_target` = the analyst's
target; `B_call`, `B_actual` = benchmark close on call/resolution date. Over the horizon:
`stock_return = P_actual/P_call − 1`, `benchmark_return = B_actual/B_call − 1`.

- **Implied direction** (frozen map): Buy/Overweight→UP, Sell/Underweight→DOWN, Hold→FLAT.
- **Stage 1 — DIRECTION GATE (pass/fail).** Bucket the realized move with one shared band
  `b = direction_flat_band = 0.02`: `stock_return > b`→UP, `< −b`→DOWN, else FLAT. The call
  PASSES iff realized direction == implied direction. (Boundary `±b` is FLAT.)
- **Stage 2 — ACCURACY ∈ (0,1], only for direction-passers.**
  `error_frac = |P_actual − P_target| / P_call`;
  `sigma_h = max(realized daily-log-return std (ddof=1) × √(horizon steps), min_sigma_h)`;
  `accuracy = exp( − error_frac / (sigma_h × accuracy_scale) )`, `accuracy_scale = 1.0`.
  Exactly `1.0` iff `P_actual == P_target`. accuracy is `None` for calls that FAILED the gate.
- **Stage 3 — BEAT-THE-MARKET (headline).** `position` = +1 long (UP) / −1 short (DOWN) /
  0 (Hold). For directional calls: `call_return = position × stock_return`;
  `beat = call_return − benchmark_return`. Computed for **all directional calls — winners and
  losers** (a wrong-direction Buy still loses money and that loss counts). Hold = neutral,
  EXCLUDED from the beat book. v1 ignores borrow/financing cost; long & short are symmetric.
- **Per-analyst aggregates:** `direction_hit_rate` = passes / ALL calls (Holds included);
  `mean_accuracy` = mean accuracy over direction-PASSING calls (None if none);
  **`beat_market`** = mean `beat` over ALL directional calls (None if none) — averaged, not
  compounded, so it is comparable across analysts with different call counts.
- **Leaderboard:** ranked by `beat_market` desc (None — no directional calls — sorts last).

**Funnel shape:** direction GATES accuracy; beat-the-market is its OWN headline over every
directional call (NOT gated by direction).

**Observed leaderboard (default seed, 108 calls):** Vega +40.5% (skilled, top) · Ursa +33.3% ·
ShortAlpha +22.8% · Coinflip +15.0% (dir 29%) · Meridian +13.0% · Tortoise +9.5% ·
MomentumOne −4.8% (dir 93%) · Hubris −4.8% (dir 10%).

**Test results:** `pytest` → **52 passed** (15 Phase-3 unit tests added).

## Phase 4 — Validation suite (most important)  ✅
**Built:** `tests/test_phase4_validation.py` (12 tests) running the real synthetic dataset
through the full funnel. All pass:
- **Buy-only rider** (MomentumOne): direction hit-rate 93% (HIGH) yet `beat_market` = −4.8%
  (≤ 0) — the headline strips out free market gains. ✓
- **Skilled picker** (Vega): `beat_market` = +40.5% (> 0), tops the leaderboard. ✓
- **Contrarian** (Ursa): direction 82% (good) but mean accuracy 0.48 (poor; ≪ Vega's 0.95). ✓
- **Overconfident-wrong** (Hubris): direction 10%, beat < 0. ✓  **Short-seller** (ShortAlpha):
  beat > 0 — beat-the-market works for shorts. ✓
- **Separation:** rider and skilled both have high direction, but the headline ranks skilled far
  above the rider, and the rider lands in the bottom half. ✓
- **Monotonicity:** a bullseye scores exactly 1.0 and no call exceeds it; flipping an outcome
  flips the direction result; adding Δ to every benchmark return lowers EVERY analyst's
  beat-market by exactly Δ (uniform, fair). ✓
- **Reproducibility:** identical seed ⇒ identical leaderboard (value-equal); a different seed
  changes the scores. ✓

**Assumptions / decisions (scoring-relevant):**
- **Revision policy (declared, uniform):** a revised target CLOSES the old call at its original
  resolution date and OPENS a new call from the revision date; the old call is still scored on
  its own horizon. (No revisions appear in the v1 synthetic set; the policy is fixed for when
  they do — enforced by the "deadline is sacred" resolver rule.)
- **Partial hits:** there is no special "partially hit" bucket. A target is never required to be
  *touched*; the call is graded purely on (a) realized direction vs implied and (b) how close
  `P_actual` landed to `P_target`, normalized by volatility. This is uniform and unambiguous.
- **HONEST LIMITATION — small-sample beat:** `beat_market` is a mean excess return and is
  outlier-sensitive when an analyst has few directional calls and the universe has fat-tailed
  winners. The near-random analyst (Coinflip) posts a positive beat by luck even at 24 calls; its
  29% direction hit-rate is what exposes it as not-skilled. Real use needs call-count weighting /
  significance testing (next step). The point estimate alone is not proof of skill.

**Limitations:** ground truth is constructed by conditioning fixtures on the realized synthetic
future (world-building, documented in `synth.py`); validation shows the engine RECOVERS planted
skill, not that real analysts behave this way.

---

## Phase 5 — Analyst-call extraction agent  ✅
**Built:** `extraction.py` —
- `ExtractedCall` (Pydantic) = the fields legible from a note; `CallExtractor` interface.
- `LLMCallExtractor` — real Anthropic implementation using **structured JSON output**
  (`client.messages.parse(..., output_format=ExtractedCall)`), model `claude-opus-4-8` (per
  the Anthropic SDK skill guidance; override via `SCORECARD_EXTRACTION_MODEL`), key from
  `ANTHROPIC_API_KEY`. Raises a clear error if no key (points to the offline path).
- `HeuristicCallExtractor` — deterministic, offline, no-key regex/rule extractor (the default
  for the harness so Phase 5 runs and passes with no network).
- `finalize_extracted` — turns an `ExtractedCall` into a full look-ahead-safe `Call`: fixes
  the resolution date at record time (call_date + horizon trading days) and reads the
  call-date price from the provider, so extracted calls obey the same fairness contract.
- `evaluate_extractor` accuracy harness + `load_research_notes`. 5 synthetic research notes
  (`fixtures/research_notes/notes.json`) with ground-truth extractions covering all 5 ratings
  and 3/6/12-month horizons.

**Test results:** `pytest` → **58 passed, 1 skipped** (6 Phase-5 added; the live-API test is
skipped without a key). Offline `HeuristicCallExtractor` scores **100% exact-match** on all 7
fields of all 5 notes; extracted calls finalize into valid Calls and flow through the engine.

**Assumptions / decisions:**
- The extractor reads only the legible call fields; `resolution_date`/`initial_price` are NOT
  taken from the text — they are computed at record time from the calendar + provider, keeping
  the look-ahead-safe deadline logic in one place.
- Horizon phrasing maps months→trading days (12→252, 6→126, 3→63); default 252 if unstated.
- The offline heuristic is tuned to the synthetic note style (labelled header + prose body)
  and recognizes the synthetic universe tickers; the LLM path handles arbitrary messy prose.

**Limitations:** offline extractor is rule-based and not meant for arbitrary real notes (that is
the LLM path's job); the live-API accuracy is asserted only when a key is present.

---

## Phase 6 — End-to-end orchestration + the time-loop agent  ✅
**Built:**
- `orchestrator.py` — `TimeLoopOrchestrator` advances a SYNTHETIC CLOCK through trading time;
  each call is resolved only when the clock reaches its record-time `resolution_date` (so at
  resolution "now" == deadline → look-ahead is impossible even in the simulation), the
  analyst's running score updates, and a one-line verdict is drafted
  (e.g. `"… came due: MISSED on direction. Beat-market record now -7.6%."`). No human in the loop.
  Returns `SimulationResult` (events, final per-analyst scores, leaderboard, clock range).
- `cli.py` — `python -m analyst_scorecard.cli` runs the whole simulation, streams verdict lines
  as deadlines arrive, then prints the beat-market-ranked leaderboard. Flags: `--seed`,
  `--quiet`, `--max-events`.

**Test results:** `pytest` → **67 passed, 1 skipped** (8 Phase-6 added). Confirms: the loop
resolves every one of the 108 calls exactly once; synthetic time advances monotonically; each
call is graded exactly at its deadline (`clock == resolution_date`); the **time-loop leaderboard
is byte-equal to the batch leaderboard** (the streaming path and the batch path agree); running
snapshots converge to the final scores; verdict lines have the expected shape; the loop is
reproducible; the CLI prints the leaderboard with the skilled picker on top.

**Assumptions / decisions:**
- Event-driven simulation: calls are processed in `(resolution_date, call_id)` order. The clock
  jumps to each deadline rather than ticking every calendar day — same outcome, and it keeps the
  "resolve only at the deadline" guarantee explicit.
- Running `beat_market` is recomputed from the analyst's accumulated scores after each
  resolution (so the streamed record is always the true running mean over resolved directional
  calls).

**Limitations:** the simulation replays a fixed, fully-known synthetic price history; a live
deployment would instead wake on real calendar dates as deadlines pass.

---

## Phase 8 — Plain-English verdicts  ✅  *(built before Phase 7 so the app can show them)*
**Built:** `verdicts.py` — `VerdictGenerator` interface, `TemplatedVerdictGenerator` (offline,
deterministic, honest one-liner from the stats), `LLMVerdictGenerator` (Anthropic
`messages.create`, key from env, raises if absent), and `default_verdict_generator()` which
returns the LLM one when a key is present and the templated one otherwise. Wired into the CLI
leaderboard output.

**Test results:** `pytest` → **72 passed, 2 skipped** (6 Phase-8 added; live-API verdict skipped
without a key). The templated verdict keys off the HEADLINE, not just direction: the rider gets
"…but you'd have done better just holding the index"; the skilled picker gets "…beat the index";
the contrarian gets "…targets are wildly off". (Coinflip's honest "usually wrong on direction
*and* beat the index" line is exactly the small-sample caveat made visible.)

**Assumptions:** verdict thresholds (direction ≥0.70 / ≥0.55 / ≥0.45; beat ±0.02; accuracy
≥0.80 / <0.50) are fixed in one place and applied uniformly.

---

## Phase 7 — Visualization (Streamlit app + PNGs)  ✅
**Built:**
- `viz.py` (headless-safe matplotlib, importable/testable): `build_dashboard`,
  `leaderboard_dataframe`, `call_detail_dataframe` (per-call EXACT resolving prices),
  `plot_leaderboard` (beat-market bar), `plot_analyst_profile` (the headline chart), and
  `save_dashboard_pngs` (writes `outputs/*.png`).
- `app.py` — thin Streamlit app: (1) leaderboard ranked by beat-market with direction & accuracy
  columns + bar chart; (2) per-analyst profile chart + plain-English verdict + metric tiles;
  (3) call-level drill-down table showing the original call and the exact resolving prices.
- The profile chart plots each directional call at (index return, return-from-following-the-call)
  with a y=x "matches the index" line: **a skilled analyst (Vega) sits visibly ABOVE the line; the
  rider (MomentumOne) sits BELOW it** — verified in the saved PNGs.

**Test results:** `pytest` → **79 passed, 2 skipped** (7 Phase-7 added). Confirms: leaderboard df
ranked with Vega on top; drill-down prices equal the provider's actual call/resolution-date prices
(traceability); charts render; PNGs save non-empty; `app.py` compiles; and the **Streamlit app
renders all three views with no exceptions** (via `streamlit.testing.v1.AppTest`). Also verified
the app boots headless (HTTP 200, `/_stcore/health` → ok).

**Assumptions:** the profile scatter axis auto-scales, so one large-outlier winner (e.g. a +275%
call) can compress the rest of the cluster — honest but a future polish item (clip/symlog).
PNGs are regenerated deterministically and are git-ignored.

