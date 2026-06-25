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

<!-- Phase 1+ entries appended below as work proceeds. -->
