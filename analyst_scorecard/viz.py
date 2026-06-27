"""Visualization helpers — importable, headless-safe, and PNG-saving.

All plotting/data logic lives here (matplotlib, Agg backend) so it is testable without a
Streamlit runtime. `app.py` is a thin Streamlit wrapper over these functions.

The headline chart is `plot_analyst_profile`: each directional call is a point at
(index return over the horizon, return from following the call). The diagonal y = x is
"you matched the index". Points ABOVE the line beat the index; points BELOW lagged it — so a
genuine stock-picker visibly sits above the line and a hype machine sits below it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aggregation import aggregate_all
from .config import DEFAULT_CONFIG, ScorecardConfig
from .providers.call_provider import FixtureCallProvider
from .providers.price_provider import PriceDataProvider, SyntheticPriceDataProvider
from .schemas import AnalystScore, Leaderboard

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

_BEAT_COLOR = "#1a9850"   # green — beat the index
_LAG_COLOR = "#d73027"    # red — lagged the index

# Dark-theme palette so charts embedded in the dark Streamlit app aren't bright white boxes.
_DARK_INK = "#d7dce3"
_DARK_AXIS = "#5b636e"


def _ref_color(dark: bool) -> str:
    """Reference-line / zero-line color that reads on both light and dark backgrounds."""
    return "#8b95a3" if dark else "#555555"


def _apply_theme(fig, axes, dark: bool):
    """Make a figure transparent with light ink so it sits cleanly on the dark app.

    Default (dark=False) is a no-op, so the light static website keeps its white charts.
    """
    if not dark:
        return fig
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(colors=_DARK_INK)
        for spine in ax.spines.values():
            spine.set_color(_DARK_AXIS)
        ax.xaxis.label.set_color(_DARK_INK)
        ax.yaxis.label.set_color(_DARK_INK)
        ax.title.set_color(_DARK_INK)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_alpha(0.0)
            for txt in leg.get_texts():
                txt.set_color(_DARK_INK)
    return fig


@dataclass
class DashboardData:
    provider: PriceDataProvider
    config: ScorecardConfig
    leaderboard: Leaderboard
    scores_by_id: dict[str, AnalystScore]


def build_dashboard(config: ScorecardConfig = DEFAULT_CONFIG) -> DashboardData:
    provider = SyntheticPriceDataProvider(config)
    calls = FixtureCallProvider().get_calls()
    scores = aggregate_all(calls, provider, config)
    leaderboard = Leaderboard.from_scores(scores)
    return DashboardData(
        provider=provider,
        config=config,
        leaderboard=leaderboard,
        scores_by_id={s.analyst_id: s for s in scores},
    )


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def leaderboard_dataframe(leaderboard: Leaderboard) -> pd.DataFrame:
    rows = []
    for rank, s in enumerate(leaderboard.rows, start=1):
        rows.append(
            {
                "Rank": rank,
                "Analyst": s.analyst_name,
                "Firm": s.firm,
                "Beat-Market": None if s.beat_market is None else round(s.beat_market, 4),
                "Direction Hit-Rate": round(s.direction_hit_rate, 4),
                "Accuracy": None if s.mean_accuracy is None else round(s.mean_accuracy, 4),
                "Calls": s.n_calls,
                "Directional": s.n_directional,
            }
        )
    return pd.DataFrame(rows)


def call_detail_dataframe(score: AnalystScore) -> pd.DataFrame:
    """One row per call with the EXACT prices that resolved it (traceability made visible)."""
    rows = []
    for cs in score.call_scores:
        r = cs.resolution
        rows.append(
            {
                "Call ID": cs.call_id,
                "Ticker": cs.ticker,
                "Rating": cs.rating.value,
                "Implied Dir": cs.implied_direction.value,
                "Call Date": r.call_date.isoformat(),
                "Resolution Date": r.resolution_date.isoformat(),
                "Call Price": round(r.call_price, 2),
                "Target": round(r.target_price, 2),
                "Actual Price": round(r.actual_price, 2),
                "Stock Return": round(r.stock_return, 4),
                "Benchmark Return": round(r.benchmark_return, 4),
                "Direction": "PASS" if cs.direction_pass else "FAIL",
                "Accuracy": None if cs.accuracy is None else round(cs.accuracy, 3),
                "Beat": None if cs.beat is None else round(cs.beat, 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------


def plot_leaderboard(leaderboard: Leaderboard, *, dark: bool = False):
    """Horizontal bar of beat-the-market by analyst (the headline ranking)."""
    rows = [r for r in leaderboard.rows if r.beat_market is not None]
    rows = sorted(rows, key=lambda r: r.beat_market)
    names = [r.analyst_name for r in rows]
    values = [r.beat_market * 100 for r in rows]
    colors = [_BEAT_COLOR if v > 0 else _LAG_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, values, color=colors)
    ax.axvline(0, color=_ref_color(dark), linewidth=1.2)
    ax.set_xlabel("Beat-the-Market (mean excess return vs index, %)")
    if not dark:  # in-app the Streamlit heading already titles this; keep a title for PNG/light export
        ax.set_title("Analyst Leaderboard — would you have beaten the index?")
    for y, v in enumerate(values):
        ax.text(v + (0.5 if v >= 0 else -0.5), y, f"{v:+.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9,
                color=_DARK_INK if dark else "#222222")
    fig.tight_layout()
    return _apply_theme(fig, [ax], dark)


def plot_analyst_profile(score: AnalystScore, *, dark: bool = False):
    """Scatter of (index return, return-from-following-the-call) per directional call.

    The y = x line is "you matched the index". Points above beat the index; below lagged it.
    """
    pts = [
        (cs.resolution.benchmark_return * 100, cs.call_return * 100, cs.ticker, cs.beat)
        for cs in score.call_scores
        if cs.position != 0 and cs.call_return is not None
    ]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if pts:
        lo = min(min(x for x, *_ in pts), min(y for _, y, *_ in pts))
        hi = max(max(x for x, *_ in pts), max(y for _, y, *_ in pts))
        pad = max(5.0, (hi - lo) * 0.1)
        lim = (lo - pad, hi + pad)
        ink = _DARK_INK if dark else "#222222"
        # y = x reference: "matched the index"
        ax.plot(lim, lim, color=_ref_color(dark), linestyle="--", linewidth=1)
        for x, y, ticker, beat in pts:
            beat_pos = beat is not None and beat > 0
            # Encode beat/lag by SHAPE as well as color so it survives colorblind/grayscale (WCAG 1.4.1).
            ax.scatter(x, y, color=_BEAT_COLOR if beat_pos else _LAG_COLOR,
                       marker="^" if beat_pos else "v", s=55, edgecolor="white", zorder=3)
            ax.annotate(ticker, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points", color=ink)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.fill_between(lim, lim, [lim[1], lim[1]], color=_BEAT_COLOR, alpha=0.10)  # above line
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color=_ref_color(dark), linestyle="--", label="matches the index"),
            Line2D([], [], marker="^", linestyle="", color=_BEAT_COLOR, label="beat the index ▲"),
            Line2D([], [], marker="v", linestyle="", color=_LAG_COLOR, label="lagged the index ▼"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no directional calls", ha="center", va="center", transform=ax.transAxes,
                color=_DARK_INK if dark else "#222222")

    bm = score.beat_market
    bm_str = "n/a" if bm is None else f"{bm * 100:+.1f}%"
    ax.set_xlabel("Index (benchmark) return over the horizon (%)")
    ax.set_ylabel("Return from following the call (%)")
    ax.set_title(f"{score.analyst_name} — above the line beats the index\nMean beat-market: {bm_str}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _apply_theme(fig, [ax], dark)


def plot_reliability(bins, *, dark: bool = False):
    """Calibration curve: predicted probability (x) vs actual frequency (y). On the line = honest."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=_ref_color(dark), linewidth=1, label="perfect calibration")
    if bins:
        xs = [b.mean_pred for b in bins]
        ys = [b.mean_actual for b in bins]
        sizes = [max(20, min(320, b.n)) for b in bins]
        ax.scatter(xs, ys, s=sizes, color="#4c9be8" if dark else "#2b6cb0", alpha=0.85,
                   edgecolor="white" if dark else "#1b4a73", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")      # kind-neutral: used for touch AND terminal forecasts
    ax.set_ylabel("Actual frequency")
    ax.set_title("Calibration — points on the dashed line are trustworthy")
    ax.legend(loc="lower right", fontsize=8)  # out of the diagonal's path through upper-left
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _apply_theme(fig, [ax], dark)


def plot_forecast_proof(grade, *, dark: bool = False):
    """Graphical proof for a live grade: the price cone + the terminal distribution.

    Left  — where the model thinks the price goes (median path with ±1σ/±2σ cone, target line).
    Right — the distribution of the price ON the deadline; the SHADED area is exactly the probability
            for a terminal call (for a touch call, touching ever is at least this shaded mass).
    """
    s0 = float(grade.s0)
    mu = float(grade.drift_bar)
    sigma = max(float(grade.vol_bar), 1e-9)
    n = max(int(grade.n_days), 1)
    K = float(grade.target_price)
    up = grade.direction.value == "up"
    terminal = grade.kind.value == "terminal"
    band = grade.band_pct
    hit_color = _BEAT_COLOR
    tgt_color = _LAG_COLOR if dark else "#c0392b"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # -- Panel A: the price cone --
    t = np.arange(0, n + 1)
    sd_t = sigma * np.sqrt(t)
    median = s0 * np.exp(mu * t)
    for k, alpha in [(2, 0.10), (1, 0.18)]:
        axL.fill_between(t, s0 * np.exp(mu * t - k * sd_t), s0 * np.exp(mu * t + k * sd_t),
                         color="#4c9be8", alpha=alpha, linewidth=0)
    axL.plot(t, median, color="#4c9be8", lw=1.6, label="median path")
    axL.axhline(K, color=tgt_color, ls="--", lw=1.4, label=f"target ${K:,.2f}")
    if terminal and band:
        axL.plot([n, n], [K * (1 - band), K * (1 + band)], color=hit_color, lw=5, solid_capstyle="butt",
                 label=f"±{band*100:g}% band")
    axL.set_xlabel(f"{grade.interval.horizon_word} from now")
    axL.set_ylabel("price ($)")
    axL.set_title("Where the model thinks it goes")
    axL.legend(loc="best", fontsize=8)
    axL.grid(True, alpha=0.2)

    # -- Panel B: the terminal distribution at the deadline --
    m = np.log(s0) + mu * n
    sd = sigma * np.sqrt(n)
    # Always keep the target K inside the window, so a far-out target shows as a tail at ~0% rather
    # than an empty (apparently-broken) chart.
    lo = max(0.01, min(s0 * np.exp(mu * n - 3.6 * sd), K * 0.97))
    hi = max(s0 * np.exp(mu * n + 3.6 * sd), K * 1.03)
    xs = np.linspace(lo, hi, 400)
    pdf = np.exp(-((np.log(xs) - m) ** 2) / (2 * sd ** 2)) / (xs * sd * np.sqrt(2 * np.pi))
    axR.plot(xs, pdf, color="#4c9be8", lw=1.6)
    if terminal and band:
        mask = (xs >= K * (1 - band)) & (xs <= K * (1 + band))
        lab = f"within ±{band*100:g}% of target"
    elif up:
        mask = xs >= K
        lab = "ends at/above target" if terminal else "ends past target (a floor)"
    else:
        mask = xs <= K
        lab = "ends at/below target" if terminal else "ends past target (a floor)"
    axR.fill_between(xs[mask], pdf[mask], color=hit_color, alpha=0.55, label=lab)
    axR.axvline(K, color=tgt_color, ls="--", lw=1.2)
    # The big number printed ON the figure must be the CALIBRATED probability — the one the hero metric,
    # the headline, the copy-string and the alt text all show — so the proof picture can never contradict
    # the verdict beside it. The shaded area is the model's RAW mass (for touch it also undersells the
    # answer, since touch ≥ terminal); when calibration moved the number we name that gap explicitly.
    raw_pct, cal_pct = grade.raw_probability * 100, grade.probability * 100
    moved = grade.calibrated and round(raw_pct) != round(cal_pct)
    if not terminal:
        axR.text(0.5, 0.93, f"calibrated touch by deadline ≈ {cal_pct:.0f}%",
                 transform=axR.transAxes, ha="center", fontsize=10, fontweight="bold",
                 color=_DARK_INK if dark else "#1a7a3f")
        if moved:
            axR.text(0.5, 0.855, f"(raw model {raw_pct:.0f}%; shaded = terminal mass, lower than touch)",
                     transform=axR.transAxes, ha="center", fontsize=8,
                     color=_DARK_INK if dark else "#555555")
    elif moved:
        axR.text(0.5, 0.93, f"calibrated ≈ {cal_pct:.0f}%",
                 transform=axR.transAxes, ha="center", fontsize=10, fontweight="bold",
                 color=_DARK_INK if dark else "#1a7a3f")
        axR.text(0.5, 0.855, f"(shaded = raw model {raw_pct:.0f}%)", transform=axR.transAxes,
                 ha="center", fontsize=8, color=_DARK_INK if dark else "#555555")
    axR.set_xlabel("price at the deadline ($)")
    axR.set_ylabel("probability density")
    if mask.sum() == 0 or round(grade.probability * 100) == 0:
        axR.set_title("Target sits far out in the tail → ≈0%")
    elif terminal:
        axR.set_title("Shaded = raw model mass" if moved else "Shaded area = the probability")
    else:
        axR.set_title("Shaded = 'ends past target'; touch is higher (above)")
    axR.legend(loc="best", fontsize=8)
    axR.grid(True, alpha=0.2)

    if not dark:  # in-app the hero Probability metric already shows this; keep it for PNG/light export
        fig.suptitle(f"P ≈ {grade.probability*100:.0f}%   ·   {grade.kind.value} · {grade.interval.label}",
                     fontsize=12, color="#222222")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return _apply_theme(fig, [axL, axR], dark)


def plot_recommendation_chart(report, *, dark: bool = False):
    """Price since the call vs the benchmark, indexed to 100, with the analyst's target line."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ch = getattr(report, "chart", None)
    if ch is None or len(ch.ticker) < 2:
        ax.text(0.5, 0.5, "price chart unavailable (no live data)", ha="center", va="center",
                transform=ax.transAxes, color=_DARK_INK if dark else "#555555")
        ax.set_xticks([]); ax.set_yticks([])
        return _apply_theme(fig, [ax], dark)

    t = np.asarray(ch.ticker, dtype=float)
    b = np.asarray(ch.benchmark, dtype=float)
    tk = t / t[0] * 100.0
    bm = b / b[0] * 100.0
    tgt = ch.target / ch.call_price * 100.0
    x = np.arange(len(tk))
    ax.plot(x, tk, color="#4c9be8", lw=1.9, label=str(report.rec.ticker))
    ax.plot(x, bm, color="#8b95a3", lw=1.4, label=str(report.benchmark_symbol))
    ax.axhline(tgt, color=_BEAT_COLOR, ls="--", lw=1.4, label=f"target ({tgt-100:+.0f}%)")
    ax.axhline(100, color=_ref_color(dark), lw=1.0, alpha=0.6)
    last = tk[-1]
    ax.scatter([x[-1]], [last], color="#4c9be8", s=40, zorder=3, edgecolor="white")
    ax.set_xlabel("trading days since the call")
    ax.set_ylabel("indexed to 100 at the call")
    if not dark:  # the in-app Streamlit heading already titles this; keep it for PNG/light export
        ax.set_title(f"{report.rec.ticker} since the {report.rec.rating} call vs {report.benchmark_symbol}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _apply_theme(fig, [ax], dark)


# --------------------------------------------------------------------------------------
# PNG export
# --------------------------------------------------------------------------------------


def save_dashboard_pngs(outdir: Path | str = OUTPUT_DIR, config: ScorecardConfig = DEFAULT_CONFIG) -> list[Path]:
    """Save the leaderboard chart plus a few analyst profiles as PNGs. Returns the paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = build_dashboard(config)
    saved: list[Path] = []

    fig = plot_leaderboard(data.leaderboard)
    p = outdir / "leaderboard_beat_market.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # Profile the top (skilled) and bottom (lagging) ranked analysts for contrast, plus the rider.
    highlight_ids = []
    rows = data.leaderboard.rows
    if rows:
        highlight_ids.append(rows[0].analyst_id)        # top
        highlight_ids.append(rows[-1].analyst_id)       # bottom
    if "momentum" in data.scores_by_id:
        highlight_ids.append("momentum")                # the rider, explicitly
    for aid in dict.fromkeys(highlight_ids):            # de-dup, keep order
        score = data.scores_by_id[aid]
        fig = plot_analyst_profile(score)
        p = outdir / f"profile_{aid}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    return saved


if __name__ == "__main__":  # pragma: no cover
    paths = save_dashboard_pngs()
    print("Saved:")
    for p in paths:
        print(f"  {p}")
