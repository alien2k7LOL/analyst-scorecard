"""Synthetic ground-truth dataset: analysts with KNOWN skill profiles.

WHY THIS EXISTS
---------------
To validate that the scoring engine is fair and correct, we need analysts whose true skill
we already know, so we can check the engine recovers it. We manufacture them here.

WORLD-BUILDING vs SCORING (read this — it matters for fairness)
--------------------------------------------------------------
This module PEEKS AT THE REALIZED SYNTHETIC FUTURE to place calls. For example, the "skilled
picker" is given Buy calls on windows that genuinely went up and beat the index. That is how
we *construct a world* containing a known-skilled analyst.

This is NOT look-ahead bias in scoring. The scoring engine (resolution.py / scoring.py) never
sees the future — it is only ever handed a bounded [call_date, resolution_date] window. Here
we are the game master writing the world; there the engine plays blind. Phase 4 verifies the
blind engine recovers the skill we planted.

Every call still obeys the fairness contract: its resolution rule (horizon -> deadline) is
fixed at record time, before any scoring happens.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, ScorecardConfig
from .providers.call_provider import DEFAULT_FIXTURE_PATH
from .providers.price_provider import PriceDataProvider, SyntheticPriceDataProvider
from .schemas import Call, Rating


# --------------------------------------------------------------------------------------
# Realized outcome of a candidate window (the "answer key" the generator is allowed to see)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowOutcome:
    ticker: str
    call_date: pd.Timestamp
    resolution_date: pd.Timestamp
    call_price: float
    actual_price: float
    stock_return: float
    bench_return: float


def _outcome(provider: PriceDataProvider, ticker: str, call_date: pd.Timestamp, horizon: int) -> WindowOutcome:
    res = provider.trading_day_offset(call_date, horizon)
    p0 = provider.price_on(ticker, call_date)
    p1 = provider.price_on(ticker, res)
    b0 = provider.price_on(provider.benchmark_symbol, call_date)
    b1 = provider.price_on(provider.benchmark_symbol, res)
    return WindowOutcome(
        ticker=ticker,
        call_date=call_date,
        resolution_date=res,
        call_price=p0,
        actual_price=p1,
        stock_return=p1 / p0 - 1.0,
        bench_return=b1 / b0 - 1.0,
    )


# --------------------------------------------------------------------------------------
# Predicates over a realized window: (outcome, band, margin) -> bool
# --------------------------------------------------------------------------------------
# These select windows that instantiate a desired ground-truth pattern. "beats"/"lags" are
# defined relative to the BENCHMARK so they line up exactly with how the engine computes
# beat-the-market (sign*stock_return - benchmark_return).


def long_beats(o: WindowOutcome, band: float, m: float) -> bool:
    """Stock rose past the band AND a long beat the index by >= m. (Genuine bullish skill.)"""
    return o.stock_return > band and (o.stock_return - o.bench_return) > m


def short_beats(o: WindowOutcome, band: float, m: float) -> bool:
    """Stock fell past the band AND a short beat the index by >= m. (Genuine bearish skill.)"""
    return o.stock_return < -band and (-o.stock_return - o.bench_return) > m


def long_lags(o: WindowOutcome, band: float, m: float) -> bool:
    """Stock rose (Buy passes direction) but a long LAGGED the index by >= m. The rider's
    sweet spot: looks right on direction, but you'd have done better in the index."""
    return o.stock_return > band and (o.stock_return - o.bench_return) < -m


def flat_abs(o: WindowOutcome, band: float, m: float) -> bool:
    """Stock stayed within the flat band (a correct Hold)."""
    return abs(o.stock_return) <= band


def mild_down(o: WindowOutcome, band: float, m: float) -> bool:
    """Fell modestly (a bounded faller — for confidently-wrong Buys)."""
    return -0.20 <= o.stock_return <= -band


def mild_up(o: WindowOutcome, band: float, m: float) -> bool:
    """Rose modestly (a bounded riser — for confidently-wrong Sells)."""
    return band <= o.stock_return <= 0.20


def any_window(o: WindowOutcome, band: float, m: float) -> bool:
    return True


# --------------------------------------------------------------------------------------
# Rating and target choosers
# --------------------------------------------------------------------------------------

RatingFn = Callable[[np.random.Generator, int, WindowOutcome], Rating]
TargetFn = Callable[[np.random.Generator, WindowOutcome], float]


def const_rating(r: Rating) -> RatingFn:
    return lambda rng, ci, o: r


def alt_up(rng: np.random.Generator, ci: int, o: WindowOutcome) -> Rating:
    return Rating.BUY if ci % 2 == 0 else Rating.OVERWEIGHT


def alt_down(rng: np.random.Generator, ci: int, o: WindowOutcome) -> Rating:
    return Rating.SELL if ci % 2 == 0 else Rating.UNDERWEIGHT


def rand_rating(rng: np.random.Generator, ci: int, o: WindowOutcome) -> Rating:
    choices = list(Rating)
    return choices[int(rng.integers(len(choices)))]


def tight_target(spread: float = 0.03) -> TargetFn:
    """Target near the ACTUAL outcome (good magnitude skill) -> high accuracy."""
    return lambda rng, o: o.actual_price * (1.0 + rng.uniform(-spread, spread))


def lazy_up_target(pct: float = 0.15) -> TargetFn:
    """A reflexive '+15% from here' target regardless of the stock -> mediocre accuracy."""
    return lambda rng, o: o.call_price * (1.0 + pct)


def wild_up_target(rng_lo: float = 0.40, rng_hi: float = 0.60) -> TargetFn:
    """Predicts a huge up-move -> poor magnitude even when direction is right."""
    return lambda rng, o: o.call_price * (1.0 + rng.uniform(rng_lo, rng_hi))


def wild_down_target(rng_lo: float = 0.40, rng_hi: float = 0.60) -> TargetFn:
    return lambda rng, o: o.call_price * (1.0 - rng.uniform(rng_lo, rng_hi))


def flat_target(spread: float = 0.01) -> TargetFn:
    """Target ~ today's price (a Hold expecting no move)."""
    return lambda rng, o: o.call_price * (1.0 + rng.uniform(-spread, spread))


def rand_pm_target(spread: float = 0.30) -> TargetFn:
    return lambda rng, o: o.call_price * (1.0 + rng.uniform(-spread, spread))


# --------------------------------------------------------------------------------------
# Profile / recipe definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    n: int
    predicate: Callable[[WindowOutcome, float, float], bool]
    rating_fn: RatingFn
    target_fn: TargetFn
    margin: float = 0.04


@dataclass(frozen=True)
class Profile:
    analyst_id: str
    name: str
    firm: str
    ground_truth: str          # plain-English description of the planted skill
    recipes: tuple[Recipe, ...]


# The 8 ground-truth analysts. Counts/margins are tuned so the planted properties hold with
# margin under the default seed (verified in Phase 4). See GROUND_TRUTH below.
PROFILES: tuple[Profile, ...] = (
    Profile(
        analyst_id="vega",
        name="Vega Capital",
        firm="Vega Capital Research",
        ground_truth="Genuinely skilled stock-picker: longs that beat the index, shorts that "
        "beat the index, tight targets. Expect HIGH direction, POSITIVE beat-market, good accuracy.",
        recipes=(
            Recipe(7, long_beats, alt_up, tight_target(), margin=0.05),
            Recipe(4, short_beats, alt_down, tight_target(), margin=0.04),
            Recipe(2, mild_down, const_rating(Rating.BUY), tight_target()),  # honest mistakes (Buy that fell)
        ),
    ),
    Profile(
        analyst_id="momentum",
        name="MomentumOne",
        firm="MomentumOne Advisors",
        ground_truth="Perma-bull who only ever says Buy and rides a rising market: stocks go up "
        "(high direction) but LAG the index. Expect HIGH direction, beat-market AT OR BELOW ZERO.",
        recipes=(
            Recipe(13, long_lags, const_rating(Rating.BUY), lazy_up_target(), margin=0.03),
            Recipe(1, mild_down, const_rating(Rating.BUY), lazy_up_target()),  # one faller -> direction <100%
        ),
    ),
    Profile(
        analyst_id="ursa",
        name="Ursa Research",
        firm="Ursa Research Partners",
        ground_truth="Contrarian who is RIGHT ON DIRECTION (often correctly bearish) but wildly "
        "wrong on MAGNITUDE (predicts huge moves). Expect good direction, POOR accuracy.",
        recipes=(
            Recipe(5, short_beats, alt_down, wild_down_target(), margin=0.04),
            Recipe(4, long_beats, alt_up, wild_up_target(), margin=0.04),
            Recipe(2, mild_down, const_rating(Rating.BUY), wild_up_target()),  # mistakes
        ),
    ),
    Profile(
        analyst_id="coinflip",
        name="Coinflip Securities",
        firm="Coinflip Securities",
        ground_truth="Near-random: ratings unrelated to the future, random targets. Expect "
        "middling direction (~base rate) and beat-market near zero.",
        recipes=(
            Recipe(14, any_window, rand_rating, rand_pm_target()),
        ),
    ),
    Profile(
        analyst_id="hubris",
        name="Hubris Partners",
        firm="Hubris Partners",
        ground_truth="Overconfident and wrong: confident Buys on fallers and Sells on risers, "
        "with big targets. Expect LOW direction, NEGATIVE beat-market, poor accuracy.",
        recipes=(
            Recipe(6, mild_down, const_rating(Rating.BUY), wild_up_target()),
            Recipe(3, mild_up, const_rating(Rating.SELL), wild_down_target()),
            Recipe(1, long_beats, const_rating(Rating.BUY), wild_up_target()),  # broken-clock correct call
        ),
    ),
    Profile(
        analyst_id="tortoise",
        name="Tortoise Asset Mgmt",
        firm="Tortoise Asset Management",
        ground_truth="Hold specialist: mostly correct Holds on calm names that stay flat, plus a "
        "few directional calls. Expect decent direction; beat-market defined over its few longs.",
        recipes=(
            Recipe(9, flat_abs, const_rating(Rating.HOLD), flat_target()),
            Recipe(2, long_beats, const_rating(Rating.BUY), tight_target(), margin=0.04),
            Recipe(1, long_lags, const_rating(Rating.BUY), lazy_up_target(), margin=0.03),
        ),
    ),
    Profile(
        analyst_id="shortalpha",
        name="ShortAlpha Capital",
        firm="ShortAlpha Capital",
        ground_truth="Specialist short-seller whose Sells come true and beat the index on the "
        "short side. Demonstrates beat-market works for shorts. Expect POSITIVE beat-market.",
        recipes=(
            Recipe(11, short_beats, alt_down, tight_target(), margin=0.05),
            Recipe(1, mild_up, const_rating(Rating.SELL), tight_target()),  # one wrong short
        ),
    ),
    Profile(
        analyst_id="meridian",
        name="Meridian Equity",
        firm="Meridian Equity Research",
        ground_truth="Realistic middling analyst: a mix of skilled longs/shorts, some index-laggers, "
        "and Holds. Expect modest positive (near-zero) beat-market.",
        recipes=(
            Recipe(4, long_beats, alt_up, tight_target(), margin=0.04),
            Recipe(2, short_beats, alt_down, tight_target(), margin=0.04),
            Recipe(3, long_lags, const_rating(Rating.BUY), lazy_up_target(), margin=0.03),
            Recipe(2, flat_abs, const_rating(Rating.HOLD), flat_target()),
            Recipe(1, mild_down, const_rating(Rating.BUY), tight_target()),
        ),
    ),
)


# Machine-readable ground truth for Phase 4 assertions and the app.
GROUND_TRUTH: dict[str, str] = {p.analyst_id: p.ground_truth for p in PROFILES}


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def _profile_rng(cfg: ScorecardConfig, profile_id: str) -> np.random.Generator:
    """Independent, reproducible RNG per profile (distinct from the price RNGs)."""
    digest = hashlib.sha256(f"{cfg.seed}:calls:{profile_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _candidate_dates(provider: PriceDataProvider, horizon: int) -> list[pd.Timestamp]:
    days = provider.trading_days()
    last_start = len(days) - horizon - 1
    if last_start < 0:
        raise ValueError("Horizon longer than the available price history")
    return list(days[: last_start + 1])


def generate_calls_for_profile(
    provider: PriceDataProvider, profile: Profile, cfg: ScorecardConfig
) -> list[Call]:
    rng = _profile_rng(cfg, profile.analyst_id)
    band = cfg.direction_flat_band
    horizon = cfg.default_horizon_days

    dates = _candidate_dates(provider, horizon)
    tickers = provider.tickers()
    candidates = [(t, d) for t in tickers for d in dates]
    order = rng.permutation(len(candidates))
    shuffled = [candidates[k] for k in order]

    used: set[tuple[str, pd.Timestamp]] = set()
    calls: list[Call] = []
    ci = 0
    cursor = 0

    for recipe in profile.recipes:
        got = 0
        while got < recipe.n:
            if cursor >= len(shuffled):
                raise RuntimeError(
                    f"Profile {profile.analyst_id!r}: ran out of candidate windows for a "
                    f"recipe needing {recipe.n} (got {got}). Loosen the predicate/margin."
                )
            ticker, cdate = shuffled[cursor]
            cursor += 1
            key = (ticker, cdate)
            if key in used:
                continue
            o = _outcome(provider, ticker, cdate, horizon)
            if not recipe.predicate(o, band, recipe.margin):
                continue
            used.add(key)
            rating = recipe.rating_fn(rng, ci, o)
            target = round(float(recipe.target_fn(rng, o)), 2)
            calls.append(
                Call(
                    call_id=f"{profile.analyst_id}-{ci:03d}",
                    analyst_id=profile.analyst_id,
                    analyst_name=profile.name,
                    firm=profile.firm,
                    ticker=ticker,
                    rating=rating,
                    target_price=target,
                    call_date=cdate.date(),
                    horizon_days=horizon,
                    resolution_date=o.resolution_date.date(),
                    initial_price=float(o.call_price),
                )
            )
            got += 1
            ci += 1

    return calls


def generate_all_calls(
    provider: PriceDataProvider | None = None, cfg: ScorecardConfig = DEFAULT_CONFIG
) -> list[Call]:
    """Generate the full synthetic call set for all 8 ground-truth analysts."""
    if provider is None:
        provider = SyntheticPriceDataProvider(cfg)
    calls: list[Call] = []
    for profile in PROFILES:
        calls.extend(generate_calls_for_profile(provider, profile, cfg))
    return calls


def write_fixtures(
    path: Path | str = DEFAULT_FIXTURE_PATH, cfg: ScorecardConfig = DEFAULT_CONFIG
) -> Path:
    """Generate and write calls.json (stable order for clean diffs)."""
    calls = generate_all_calls(cfg=cfg)
    calls_sorted = sorted(calls, key=lambda c: (c.analyst_id, c.call_id))
    payload = [c.model_dump(mode="json") for c in calls_sorted]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


if __name__ == "__main__":  # pragma: no cover - thin CLI
    out = write_fixtures()
    n = len(json.loads(Path(out).read_text()))
    print(f"Wrote {n} synthetic calls for {len(PROFILES)} analysts -> {out}")
