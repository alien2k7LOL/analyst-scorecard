"""Single source of truth for every tunable parameter.

FAIRNESS PRINCIPLE: every rule that can affect a score lives here, in one place, so it
can be audited and is provably applied identically to every analyst. Nothing downstream
invents its own horizon, benchmark, band, or seed.

All parameters are attached to an immutable ``ScorecardConfig`` so tests can construct
variant configs (different benchmark, horizon, band, ...) without mutating global state.
``DEFAULT_CONFIG`` is the canonical instance used by the CLI, the app and the fixtures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import date

# --------------------------------------------------------------------------------------
# Trading-calendar conventions
# --------------------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252  # standard convention for annualizing drift / volatility


@dataclass(frozen=True)
class TickerSpec:
    """Geometric-Brownian-motion parameters for one synthetic instrument.

    drift / vol are ANNUALIZED (per year). The synthetic provider converts them to a
    per-trading-day step using ``TRADING_DAYS_PER_YEAR``. ``start_price`` is the price on
    the first simulated trading day.
    """

    symbol: str
    drift: float        # annualized expected log-return (mu)
    vol: float          # annualized volatility (sigma), > 0
    start_price: float   # price on the first trading day, > 0


# --------------------------------------------------------------------------------------
# The synthetic universe
# --------------------------------------------------------------------------------------
# A spread of "personalities": hot growth names, sleepy low-vol names, decliners, choppy
# trendless names, and a diversified benchmark index with positive drift (a RISING MARKET,
# which is what lets the buy-only rider look good on direction while adding no real value).

BENCHMARK_SPEC = TickerSpec(symbol="MKT", drift=0.08, vol=0.13, start_price=100.0)

UNIVERSE_SPECS: tuple[TickerSpec, ...] = (
    TickerSpec("ASTR", drift=0.22, vol=0.40, start_price=60.0),    # hot growth, wild
    TickerSpec("BRIX", drift=0.06, vol=0.16, start_price=120.0),   # stable, below-mkt drift
    TickerSpec("CMET", drift=-0.05, vol=0.38, start_price=45.0),   # volatile decliner
    TickerSpec("DRNE", drift=-0.12, vol=0.30, start_price=30.0),   # a dog
    TickerSpec("EVRG", drift=0.10, vol=0.22, start_price=85.0),    # solid grower
    TickerSpec("FALC", drift=0.00, vol=0.45, start_price=25.0),    # choppy, trendless
    TickerSpec("GLDN", drift=0.16, vol=0.28, start_price=150.0),   # strong grower
    TickerSpec("HELX", drift=0.03, vol=0.11, start_price=200.0),   # sleepy, low vol
    TickerSpec("IONX", drift=0.20, vol=0.42, start_price=40.0),    # hot, wild
    TickerSpec("JUNI", drift=0.09, vol=0.20, start_price=70.0),    # market-like
)


# --------------------------------------------------------------------------------------
# The master config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardConfig:
    """Every parameter that can change a score. Immutable; copy with ``with_overrides``."""

    # Reproducibility -------------------------------------------------------------------
    seed: int = 20240601

    # Synthetic calendar ----------------------------------------------------------------
    sim_start: date = date(2021, 1, 4)        # first simulated trading day (a Monday)
    n_trading_days: int = 1008                # ~4 trading years of data

    # Universe --------------------------------------------------------------------------
    benchmark: TickerSpec = BENCHMARK_SPEC
    universe: tuple[TickerSpec, ...] = UNIVERSE_SPECS

    # Horizon ---------------------------------------------------------------------------
    # Default resolution horizon in TRADING days. A call's deadline is fixed at record
    # time as (call_date + horizon trading days). Calls may carry their own horizon, but
    # this is the canonical default applied uniformly when one is not specified.
    default_horizon_days: int = 252           # ~1 year

    # Direction gate --------------------------------------------------------------------
    # The realized move is bucketed by this band, applied identically to all ratings:
    #   return >  +flat_band  -> realized UP
    #   return <  -flat_band  -> realized DOWN
    #   otherwise             -> realized FLAT
    # A call PASSES the gate iff realized direction == implied direction. So a Buy needs a
    # genuine up-move (beyond the band) to pass; a Hold needs the stock to stay in the band.
    direction_flat_band: float = 0.02         # 2%

    # Accuracy --------------------------------------------------------------------------
    # Volatility-normalized closeness for direction-passing calls:
    #   error_frac = |P_actual - P_target| / P_call
    #   sigma_h    = realized daily vol over the horizon * sqrt(horizon_days)
    #   accuracy   = exp(-error_frac / (sigma_h * accuracy_scale))
    # accuracy_scale > 1 makes the score more forgiving (more sigmas tolerated).
    # A bullseye (P_actual == P_target) always scores exactly 1.0.
    accuracy_scale: float = 1.0
    # Floor on sigma_h so a (near) zero-volatility window can't divide by ~0.
    min_sigma_h: float = 1e-4

    # Beat-the-market -------------------------------------------------------------------
    # Directional position taken by "following the call": +1 long (UP), -1 short (DOWN),
    # 0 (HOLD -> neutral, excluded from the beat-the-market book). beat = sign*stock_return
    # - benchmark_return. v1 ignores borrow cost / financing; symmetric for long & short.
    # (Hold exclusion documented in PROGRESS.md: a Hold is "no action", so it does not
    # contribute to the beat-vs-index book; it is still graded on direction & accuracy.)

    def with_overrides(self, **kwargs) -> "ScorecardConfig":
        """Return a copy with selected fields replaced (for tests / experiments)."""
        return replace(self, **kwargs)

    # Per-ticker reproducible seed ------------------------------------------------------
    def ticker_seed(self, symbol: str) -> int:
        """Deterministic, order-independent child seed for one symbol.

        Derived from (master seed, symbol) via SHA-256 so that adding/removing/reordering
        tickers never perturbs another ticker's path. Critical for reproducibility.
        """
        digest = hashlib.sha256(f"{self.seed}:{symbol}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def all_symbols(self) -> list[str]:
        """Universe symbols plus the benchmark (benchmark last)."""
        return [t.symbol for t in self.universe] + [self.benchmark.symbol]

    def spec_for(self, symbol: str) -> TickerSpec:
        for t in self.universe:
            if t.symbol == symbol:
                return t
        if symbol == self.benchmark.symbol:
            return self.benchmark
        raise KeyError(f"Unknown symbol: {symbol!r}")


DEFAULT_CONFIG = ScorecardConfig()
