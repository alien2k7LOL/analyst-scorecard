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

